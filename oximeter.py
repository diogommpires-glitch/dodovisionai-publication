"""
DODO VISION AI — Oximeter  v8.5
Pulox PO-300 / Contec CMS50E-style pulse oximeter.

Connection priority:
  1. USB HID  (VID=0x28E9 PID=0x028A — confirmed on this machine)
  2. Bluetooth SPP serial fallback (COM4, 19200 8O1)

HID Protocol — 7–9 byte reports, no report-ID byte.
Two formats auto-detected:
  Format A (GigaDevice 0x28E9, typical):
    Byte 0: status (bit0=searching, bit1=finger_out, bit2=beep)
    Byte 1: SpO2 (0–100, 0x7F/0xFF = invalid)
    Byte 2: PR high bit (bit0)
    Byte 3: PR low byte
    Byte 4: pleth waveform (0–127)
  Format B (CMS50 serial-over-HID, 7 bytes):
    Byte 0: sync (bit7 must be 1)
    Byte 1: pleth
    Byte 2: status
    Byte 3: PR high bits
    Byte 4: PR low byte
    Byte 5: SpO2
  Unknown → raw dump logged, probed for valid HR/SpO2 pairs.

Serial Protocol (fallback): 19200 baud, 8O1, 5-byte CMS50 packets.
"""
import threading, time
import config as cfg

# ── Known HID VID/PID for Pulox PO-300 and compatible devices ──
_HID_DEVICES = [
    (0x28E9, 0x028A),   # GigaDevice — confirmed on this machine
    (0x28C2, 0x0001),   # Contec CMS50E
    (0x28C2, 0x0002),   # Contec CMS50D+
    (0x0483, 0x5750),   # STMicro variant
    (0x28C2, 0xFFFF),   # generic Contec catch-all
]

_oxi_data = {
    "heart_rate": None,
    "spo2": None,
    "signal_strength": None,
    "pleth_waveform": None,
    "connected": False,
    "port": None,          # "HID:VID:PID" or "COMx"
    "error": None,
    "last_update": 0,
    "mode": None,          # "hid" or "serial"
}
_oxi_thread = None
_oxi_stop = threading.Event()
_oxi_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────

def get_oximeter_data():
    if not _oxi_data["connected"]:
        return None
    if time.time() - _oxi_data["last_update"] > 5:
        return None
    # v9.1.1: include last_update so the UI staleness badge can compute
    # the actual age (affect_vector.oxi_age_s). Without this field the
    # caller in main.py would fall back to "now" and the UI never saw
    # real latency information.
    return {
        "heart_rate": _oxi_data["heart_rate"],
        "spo2": _oxi_data["spo2"],
        "signal_strength": _oxi_data["signal_strength"],
        "pleth_waveform": _oxi_data.get("pleth_waveform"),
        "last_update": _oxi_data.get("last_update"),
    }


# v8.9: Radar-based soft fusion motion gating (ROSA architecture)
# When radar detects macro motion, lower the PPG confidence weight to prevent
# motion-artifact-induced mathematical crashes in the affect pipeline.
_MOTION_VARIANCE_THRESH = 0.04  # 4% displacement variance threshold

def get_oximeter_data_fused(radar_data=None):
    """Return oximeter data with radar-informed confidence gating.

    If radar indicates macro motion above threshold, signal_strength is
    attenuated to prevent artifact-corrupted PPG from dominating fusion.
    """
    base = get_oximeter_data()
    if base is None:
        return None
    if radar_data and radar_data.get("distance") is not None:
        motion_var = radar_data.get("motion_variance", 0) or 0
        if motion_var > _MOTION_VARIANCE_THRESH:
            # Attenuate signal: scale down by motion severity
            attenuation = max(0.1, 1.0 - min(1.0, motion_var / 0.2))
            base["signal_strength"] = max(0, int((base.get("signal_strength") or 0) * attenuation))
            base["motion_gated"] = True
            base["motion_attenuation"] = round(attenuation, 2)
        else:
            base["motion_gated"] = False
    return base

def get_oximeter_status():
    return {
        "connected": _oxi_data["connected"],
        "port": _oxi_data["port"],
        "error": _oxi_data["error"],
        "mode": _oxi_data.get("mode"),
    }

# ──────────────────────────────────────────────────────────────
#  HID helpers
# ──────────────────────────────────────────────────────────────

def _import_hid():
    """Return the hid module or None. Handles both hidapi and fallback."""
    try:
        import hid as _h
        # Quick sanity check — enumerate() must not throw
        list(_h.enumerate())
        return _h
    except Exception:
        return None


def _parse_hid_report(data: bytes):
    """
    Try to extract (hr, spo2, pleth, signal) from a raw HID report.
    Returns (hr, spo2, pleth, signal) or None on failure.
    Attempts Format A first, then Format B, then brute-force scan.
    """
    n = len(data)
    if n < 5:
        return None

    # ── No-finger early exit: 0x7F / 0xFF in byte 1 = device reports "no finger" ──
    # These are well-known invalid-reading markers for GigaDevice and Contec
    # devices ONLY when the report is Format A (byte 0 is a status byte,
    # not a sync byte). v9.1.2 bug: Format B uses byte 0 with bit7 SET as
    # a sync byte, and byte 1 is pleth (which can legitimately be 0x7F).
    # The old unconditional early-return was throwing away every valid
    # Format B packet, which is what made some oximeters appear "dead"
    # even though HID enumeration succeeded and data was flowing.
    _is_format_b_sync = (data[0] & 0x80) != 0
    if (not _is_format_b_sync) and n >= 2 and data[1] in (0x7F, 0xFF):
        return None

    # ── Format A: GigaDevice (0x28E9) ──
    # Byte 0: status, Byte 1: SpO2, Byte 2: PR_hi, Byte 3: PR_lo, Byte 4: pleth
    if n >= 5:
        status = data[0]
        spo2_a = data[1]
        pr_hi  = data[2] & 0x01
        pr_lo  = data[3]
        pleth  = data[4] & 0x7F
        hr_a   = (pr_hi << 8) | pr_lo
        sig_a  = 3 - min(3, (status >> 4) & 0x03)  # crude signal proxy
        if 70 <= spo2_a <= 100 and 30 < hr_a < 250:
            return (hr_a, spo2_a, pleth, sig_a)

    # ── Format B: CMS50 serial-over-HID (7 bytes, sync byte) ──
    # Byte 0 must have bit7 set; bytes 1-6 have bit7 clear
    if n >= 6 and (data[0] & 0x80):
        b0, b1, b2, b3, b4, b5 = data[0], data[1], data[2], data[3], data[4], data[5]
        if not any(b & 0x80 for b in (b1, b2, b3, b4, b5)):
            sig_b    = (b0 >> 4) & 0x07
            pleth_b  = ((b0 & 0x0F) << 7) | (b1 & 0x7F)
            pr_hi_b  = b2 & 0x07
            pr_lo_b  = b3 & 0x7F
            hr_b     = (pr_hi_b << 7) | pr_lo_b
            spo2_b   = b5 & 0x7F
            if 50 < spo2_b <= 100 and 30 < hr_b < 250:
                return (hr_b, spo2_b, pleth_b & 0x7F, sig_b)

    # ── Brute-force: scan for plausible (HR, SpO2) pairs in any 2 adjacent bytes ──
    # SpO2 narrowed to 92-100: real clinical SpO2 below 92 is uncommon at rest
    # and should only be trusted when it comes from the known Format A/B layouts,
    # NOT from a byte-scan guess. Before this narrowing, a status byte of e.g.
    # 0x5f (95) or 0x48 (72) would get accepted as "SpO2" and the outlier filter
    # would eventually confirm the repeated wrong reading, showing red "!!" on a
    # healthy lung (observed on Pulox PO-300 VID=0x28E9 PID=0x028A).
    # HR also narrowed to 45-180 (normal adult resting range) — the old 40-200
    # window let random status bytes get published as HR on first reading.
    for i in range(n - 1):
        v1, v2 = data[i] & 0xFF, data[i + 1] & 0xFF
        # Candidate 1: byte i=SpO2, byte i+1=HR (common layout)
        if 92 <= v1 <= 100 and 45 <= v2 <= 180:
            return (v2, v1, 0, 1)
        # Candidate 2: byte i=HR, byte i+1=SpO2
        if 45 <= v1 <= 180 and 92 <= v2 <= 100:
            return (v1, v2, 0, 1)

    return None


def _hid_reader(hid_dev):
    """Read HID reports in a loop, parse, and update _oxi_data."""
    _first = True
    while not _oxi_stop.is_set():
        try:
            raw = hid_dev.read(64, timeout_ms=500)
            if not raw:
                continue
            data = bytes(raw)
            if _first:
                print(f"[OXI-HID] First report ({len(data)}B): {data[:16].hex()}")
                _first = False
            result = _parse_hid_report(data)
            if result:
                hr, spo2, pleth, sig = result
                # v9.5.x — SPO2 outlier filter. The brute-force byte-pair
                # fallback in _parse_hid_report occasionally accepts noise
                # bytes as SpO2, producing sudden drops to 70-80% which are
                # clinically implausible. Reject sudden drops >8 points
                # within 2 s unless backed by a second consecutive reading.
                now = time.time()
                prev_spo2 = _oxi_data.get("spo2")
                last_t = _oxi_data.get("last_update") or 0
                if (prev_spo2 is not None and spo2 is not None
                        and (now - last_t) < 2.0
                        and (prev_spo2 - spo2) > 8):
                    # Require confirmation: pending slot, don't publish yet.
                    pending = _oxi_data.get("_pending_drop")
                    if pending and pending[0] == spo2 and (now - pending[1]) < 1.0:
                        # Second matching reading within 1 s — accept.
                        _oxi_data["_pending_drop"] = None
                    else:
                        _oxi_data["_pending_drop"] = (spo2, now)
                        # Keep the previous value; don't publish the drop.
                        with _oxi_lock:
                            _oxi_data["heart_rate"]      = hr
                            _oxi_data["pleth_waveform"]  = pleth
                            _oxi_data["signal_strength"] = sig
                            _oxi_data["last_update"]     = now
                        continue
                else:
                    _oxi_data["_pending_drop"] = None
                with _oxi_lock:
                    _oxi_data["heart_rate"]     = hr
                    _oxi_data["spo2"]           = spo2
                    _oxi_data["pleth_waveform"] = pleth
                    _oxi_data["signal_strength"]= sig
                    _oxi_data["last_update"]    = now
        except Exception:
            if not _oxi_stop.is_set():
                time.sleep(0.1)
    hid_dev.close()


def _connect_hid():
    """
    Try to open any known HID oximeter.
    Returns True on success (reader thread started).
    """
    hid = _import_hid()
    if hid is None:
        print("[OXI] hidapi not available — install with: pip install hidapi")
        return False

    found_devs = list(hid.enumerate())
    if not found_devs:
        print("[OXI] No HID devices found on system")
    else:
        # v9.1.2: print the full HID inventory so users can see exactly
        # which VID/PID their oximeter advertises. If the device is not
        # in _HID_DEVICES and doesn't match the keyword probe, they now
        # have the numbers needed to extend _HID_DEVICES themselves.
        print(f"[OXI] Scanning {len(found_devs)} HID devices...")
        for _d in found_devs:
            _vid = _d.get("vendor_id", 0)
            _pid = _d.get("product_id", 0)
            _prod = _d.get("product_string") or "?"
            _mfr = _d.get("manufacturer_string") or "?"
            _usage = _d.get("usage_page", 0)
            print(f"[OXI]   VID={_vid:#06x} PID={_pid:#06x} usage={_usage:#06x} "
                  f"mfr='{_mfr}' prod='{_prod}'")
    for vid, pid in _HID_DEVICES:
        for dev_info in found_devs:
            if dev_info["vendor_id"] == vid and dev_info["product_id"] == pid:
                try:
                    d = hid.device()
                    d.open(vid, pid)
                    d.set_nonblocking(0)   # blocking reads with timeout
                    _oxi_stop.clear()
                    t = threading.Thread(target=_hid_reader, args=(d,), daemon=True)
                    t.start()
                    tag = f"HID:{vid:#06x}:{pid:#06x}"
                    with _oxi_lock:
                        _oxi_data["connected"] = True
                        _oxi_data["port"]      = tag
                        _oxi_data["error"]     = None
                        _oxi_data["mode"]      = "hid"
                    print(f"[OK] Oximeter HID {tag} ({dev_info.get('product_string','?')})")
                    return True
                except Exception as e:
                    _oxi_data["error"] = str(e)
    # Fallback: probe any HID device that looks medical (usage_page 0xFF00 = vendor-specific)
    # v9.1.2: widened keyword set — covers Wellue/Viatom/Berry/Choicemmed/
    # Shenzhen generics and Jumper-brand rebadges that don't mention
    # "Contec" or "Pulox" but all serve the same CMS50-style HID protocol.
    for dev_info in found_devs:
        vid, pid = dev_info["vendor_id"], dev_info["product_id"]
        if (vid, pid) in _HID_DEVICES:
            continue
        prod = (dev_info.get("product_string") or "").lower()
        mfr = (dev_info.get("manufacturer_string") or "").lower()
        combo = prod + " " + mfr
        _kw = ("pulse", "oxi", "spo2", "cms", "contec", "pulox", "medical",
               "wellue", "viatom", "berry", "choicemmed", "jumper", "bodimetrics",
               "innovo", "zacurate", "po-", "po3", "po30", "heart rate", "sp02")
        if any(kw in combo for kw in _kw):
            try:
                d = hid.device()
                d.open(vid, pid)
                d.set_nonblocking(0)
                _oxi_stop.clear()
                t = threading.Thread(target=_hid_reader, args=(d,), daemon=True)
                t.start()
                tag = f"HID:{vid:#06x}:{pid:#06x}"
                with _oxi_lock:
                    _oxi_data["connected"] = True
                    _oxi_data["port"]      = tag
                    _oxi_data["error"]     = None
                    _oxi_data["mode"]      = "hid"
                print(f"[OK] Oximeter HID (probe) {tag} ({prod})")
                return True
            except Exception:
                pass
    return False

# ──────────────────────────────────────────────────────────────
#  Serial helpers (fallback)
# ──────────────────────────────────────────────────────────────

def _find_serial_port():
    """Auto-detect oximeter serial COM port (BT SPP fallback).

    v9.7.5: skip ports that are known to be claimed by other sensors.
    Previously this happily grabbed the GSR Arduino on COM7 (which
    matches "USB Serial Device" → keyword hit), then opened it and
    read garbage Arduino bytes, saw bit7 set, declared "oximeter
    found", and held the port — blocking the actual GSR thread with
    a PermissionError. The oximeter then connected via HID anyway,
    leaving COM7 wedged for ~30s before the SERIAL session timed
    out and released. Now we explicitly skip GSR_PORT and
    RADAR_PORT (mirrors gsr._find_gsr_port's logic at gsr.py:86)
    so the auto-detect can never collide with another sensor's
    pinned port.
    """
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        keywords = ["Prolific", "PL2303", "CH340", "CP210", "USB Serial",
                    "Contec", "CMS", "Silicon", "FTDI", "USB-SERIAL", "Virtual COM"]
        skip_kw  = ["ESP32", "XIAO", "Seeed", "Arduino"]

        # Build the skip-port set from config so we never grab a port
        # already reserved for radar / GSR.
        skip_ports = set()
        for attr in ("RADAR_PORT", "RADAR_B_PORT", "GSR_PORT"):
            v = getattr(cfg, attr, None)
            if v and v != "auto":
                skip_ports.add(str(v).upper())

        def _is_local_bt(p):
            hwid = p.hwid or ""
            return "LOCALMFG" in hwid or "000000000000" in hwid

        for p in ports:
            if p.device.upper() in skip_ports:
                continue
            desc = (p.description or "") + " " + (p.manufacturer or "")
            if any(sk.lower() in desc.lower() for sk in skip_kw):
                continue
            if _is_local_bt(p):
                continue
            if any(kw.lower() in desc.lower() for kw in keywords):
                return p.device

        # Probe fallback
        import serial
        for p in ports:
            if p.device.upper() in skip_ports:
                continue
            desc = (p.description or "") + " " + (p.manufacturer or "")
            if any(sk.lower() in desc.lower() for sk in skip_kw):
                continue
            if _is_local_bt(p):
                continue
            try:
                ser = serial.Serial(p.device, cfg.OXIMETER_BAUD,
                                    bytesize=serial.EIGHTBITS,
                                    parity=serial.PARITY_ODD,
                                    stopbits=serial.STOPBITS_ONE,
                                    timeout=2)
                data = ser.read(20)
                ser.close()
                if data and any(b & 0x80 for b in data):
                    print(f"[OXI] Serial probe hit on {p.device}")
                    return p.device
            except Exception:
                continue
    except Exception:
        pass
    return None


def _serial_reader(ser):
    """Read CMS50 5-byte serial packets."""
    buf = bytearray()
    while not _oxi_stop.is_set():
        try:
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue
            buf.extend(chunk)
            while len(buf) >= 5:
                # Find sync byte (bit7 set)
                si = next((i for i in range(len(buf)) if buf[i] & 0x80), -1)
                if si < 0:
                    buf.clear(); break
                if si > 0:
                    buf = buf[si:]
                if len(buf) < 5:
                    break
                if any(buf[j] & 0x80 for j in range(1, 5)):
                    buf = buf[1:]; continue
                b0, b1, b2, b3, b4 = buf[0], buf[1], buf[2], buf[3], buf[4]
                buf = buf[5:]
                sig  = (b0 >> 4) & 0x07
                pleth= ((b0 & 0x0F) << 7) | (b1 & 0x7F)
                hr   = ((b2 & 0x07) << 7) | (b3 & 0x7F)
                spo2 = b4 & 0x7F
                if 30 < hr < 250 and 50 < spo2 <= 100:
                    with _oxi_lock:
                        _oxi_data["heart_rate"]      = hr
                        _oxi_data["spo2"]            = spo2
                        _oxi_data["signal_strength"] = sig
                        _oxi_data["pleth_waveform"]  = pleth
                        _oxi_data["last_update"]     = time.time()
        except Exception:
            time.sleep(0.1)


def _connect_serial(port):
    """Open serial port and start reader. Returns True on success."""
    # v9.7.1: cooperative port_lock advisory check before opening so a
    # stale orphan Python holding COM5 produces a structured error
    # instead of the generic WinError 5.
    try:
        import port_lock as _plock_oxi
    except Exception:
        _plock_oxi = None  # type: ignore[assignment]
    if _plock_oxi is not None and not _plock_oxi.acquire(port, owner="oximeter"):
        held = _plock_oxi.describe(port) or {}
        msg = (f"[OXI] {port} held by pid={held.get('pid')} "
               f"({held.get('owner')}) — refusing to open")
        _oxi_data["error"] = msg
        _oxi_data["connected"] = False
        print(msg)
        return False
    try:
        import serial
        ser = serial.Serial(port, cfg.OXIMETER_BAUD,
                            bytesize=serial.EIGHTBITS,
                            parity=serial.PARITY_ODD,
                            stopbits=serial.STOPBITS_ONE,
                            timeout=1)
        _oxi_stop.clear()
        t = threading.Thread(target=_serial_reader, args=(ser,), daemon=True)
        t.start()
        with _oxi_lock:
            _oxi_data["connected"] = True
            _oxi_data["port"]      = port
            _oxi_data["error"]     = None
            _oxi_data["mode"]      = "serial"
        print(f"[OK] Oximeter serial {port} (19200 8O1)")
        return True
    except Exception as e:
        _oxi_data["error"] = str(e)
        _oxi_data["connected"] = False
        if _plock_oxi is not None:
            _plock_oxi.release(port)
        return False

# ──────────────────────────────────────────────────────────────
#  Watchdog — retry HID then serial every 5→30 s
# ──────────────────────────────────────────────────────────────

def _watchdog():
    delay = 5
    while not _oxi_stop.is_set():
        if not _oxi_data["connected"]:
            # 1) Try HID first
            if _connect_hid():
                return
            # 2) Try explicit serial port
            port = cfg.OXIMETER_PORT
            if port != "auto":
                print(f"[OXI] HID not found, trying serial {port}...")
                if _connect_serial(port):
                    return
            else:
                p = _find_serial_port()
                if p:
                    print(f"[OXI] Auto-detected serial on {p}, connecting...")
                    if _connect_serial(p):
                        return
            print(f"[OXI] No oximeter found (HID or serial), retry in {delay}s...")
        time.sleep(delay)
        delay = min(delay * 2, 30)

# ──────────────────────────────────────────────────────────────
#  Public: start / stop
# ──────────────────────────────────────────────────────────────

def start():
    global _oxi_thread
    if not cfg.OXIMETER_ENABLED:
        print("[--] Oximeter disabled in config")
        return False

    # 1) Try HID immediately (no dependency on pyserial)
    if _connect_hid():
        return True

    # 2) Try serial immediately
    try:
        import serial  # noqa: F401
        port = cfg.OXIMETER_PORT
        resolved = port if port != "auto" else _find_serial_port()
        if resolved and _connect_serial(resolved):
            return True
    except ImportError:
        pass

    # 3) Launch background watchdog
    print("[--] Oximeter: no immediate connection, watchdog retrying in background...")
    _oxi_stop.clear()
    _oxi_thread = threading.Thread(target=_watchdog, daemon=True)
    _oxi_thread.start()
    return False


def stop():
    _oxi_stop.set()
    with _oxi_lock:
        _oxi_data["connected"] = False


# ══════════════════════════════════════════════════════════════════
#  v9.0 Step 5: Time-of-Flight (ToF) Seat Validation (CMU Paradigm)
#
#  The MAX30102 oximeter frequently overflows when noise-amplification
#  algorithms attempt to extract heart rates from ambient lighting when
#  the finger is poorly seated.  Pair the oximeter with a supplementary
#  ToF / absolute proximity sensor.  If the proximity data indicates the
#  finger is improperly seated (distance > TOF_SEAT_THRESHOLD_MM), the
#  pipeline immediately halts I2C polling, saving vital computational
#  resources and preventing infinite buffer loops.
# ══════════════════════════════════════════════════════════════════

_tof_data = {
    "distance_mm": None,       # ToF measured gap (mm)
    "finger_seated": False,    # True when finger is within threshold
    "halted": False,           # True when I2C polling is halted
    "last_update": 0.0,
    "halt_count": 0,           # times we halted due to unseated finger
    "resume_count": 0,         # times we resumed after re-seating
}
_tof_lock = threading.Lock()


def update_tof_proximity(distance_mm: float) -> None:
    """Called by the UART bridge or I2C reader when ToF data arrives.

    If the finger-to-sensor gap exceeds TOF_SEAT_THRESHOLD_MM and
    TOF_HALT_ON_UNSEATED is True, the oximeter I2C polling is halted
    to prevent FIFO overrun and computational waste.
    """
    threshold = getattr(cfg, "TOF_SEAT_THRESHOLD_MM", 5.0)
    halt_enabled = getattr(cfg, "TOF_HALT_ON_UNSEATED", True)
    tof_enabled = getattr(cfg, "TOF_SEAT_ENABLED", True)

    if not tof_enabled:
        return

    with _tof_lock:
        _tof_data["distance_mm"] = distance_mm
        _tof_data["last_update"] = time.time()

        is_seated = distance_mm <= threshold

        _tof_data["finger_seated"] = is_seated

        if not is_seated and halt_enabled:
            # Finger NOT properly seated → halt I2C polling
            if not _tof_data["halted"]:
                _tof_data["halted"] = True
                _tof_data["halt_count"] += 1
                print(f"[OXI-ToF] Finger unseated ({distance_mm:.1f}mm > "
                      f"{threshold}mm) — I2C polling HALTED")
        elif is_seated and _tof_data["halted"]:
            # Finger re-seated → resume I2C polling
            _tof_data["halted"] = False
            _tof_data["resume_count"] += 1
            print(f"[OXI-ToF] Finger seated ({distance_mm:.1f}mm) — "
                  f"I2C polling RESUMED")


def is_tof_halted() -> bool:
    """Return True if ToF validation has halted oximeter polling."""
    with _tof_lock:
        return _tof_data["halted"]


def get_tof_status() -> dict:
    """Return ToF seat validation status."""
    with _tof_lock:
        return dict(_tof_data)
