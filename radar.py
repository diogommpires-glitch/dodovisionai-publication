"""
DODO VISION AI — mmWave Radar
Seeed MR60BHA2 60GHz FMCW radar for contactless heart rate + breathing.

CONNECTION SETUP:
  The MR60BHA2 kit has the radar soldered to a PCB with XIAO ESP32C6.
  By default, it runs ESPHome firmware. For this integration you need to
  reflash the ESP32C6 with the Arduino sketch (see radar_sketch/ folder).
  The sketch reads the radar via UART and prints JSON over USB serial.

  1. Install Arduino IDE + Seeed ESP32C6 board support
  2. Install Seeed_Arduino_mmWave library from GitHub
  3. Flash radar_sketch.ino to the ESP32C6
  4. Plug USB into PC, note the COM port (e.g. COM5)
  5. Set RADAR_PORT in config.py (or leave "auto" to scan)

PROTOCOL:
  Arduino sketch outputs one JSON line per reading:
    {"hr":72.3,"br":16.1,"dist":0.0,"presence":true,
     "age_hr_ms":210,"age_br_ms":480,"age_dist_ms":0,"age_phase_ms":0}

  NOTE on `dist` with sketch v7.3+ :
    The SFINAE capability probes that tried to pull distance out of
    getHeartBreathPhases / getDistance / getTargetDistance were
    miscompiled by the Arduino IDE preprocessor (see radar_sketch.ino
    v7.3 header). They are now hard-stubbed to return false, which
    means `dist` is ALWAYS 0.0 and `age_dist_ms` is ALWAYS 0. The
    Python side treats `dist == 0` as "unknown" (None) and drives
    presence purely off recent HR/BR freshness — which is how the
    sketch itself behaves in its fallback branch. No Python contract
    change is required.

  If you connect the radar UART directly to a USB-serial adapter
  (bypassing ESP32C6), set RADAR_MODE="raw" in config.py to parse
  the native MR60BHA2 binary protocol instead.
"""
import threading, time, struct, json
import config as cfg
from hr_filter import HRFilter, HR_ADULT_RESTING, BR_ADULT_RESTING
from radar_calibration import monitor as _cal_monitor

# v9.7.6 — per-radar HR correction loaded from calibration files.
# Mirrors the rppg.py auto-load pattern. Files written by
# `calibration.recompute_radar_from_disk("a"|"b")`. When absent,
# defaults to identity (slope=1, intercept=0) → no correction.
_cal_a_slope, _cal_a_intercept, _cal_a_loaded = 1.0, 0.0, False
_cal_b_slope, _cal_b_intercept, _cal_b_loaded = 1.0, 0.0, False
import os as _os, json as _json

def _load_radar_correction(which: str):
    """Load radar A or B correction from calibration/radar_<which>_correction.json."""
    global _cal_a_slope, _cal_a_intercept, _cal_a_loaded
    global _cal_b_slope, _cal_b_intercept, _cal_b_loaded
    path = _os.path.join(cfg.CALIBRATION_DIR, f"radar_{which}_correction.json")
    if not _os.path.exists(path):
        return
    try:
        with open(path, "r") as f:
            d = _json.load(f)
        sl = float(d.get("slope", 1.0))
        ic = float(d.get("intercept", 0.0))
        if which == "a":
            _cal_a_slope, _cal_a_intercept, _cal_a_loaded = sl, ic, True
        else:
            _cal_b_slope, _cal_b_intercept, _cal_b_loaded = sl, ic, True
        kind = d.get("kind", "?")
        n    = d.get("n_pairs", "?")
        print(f"[radar] Calibration {which.upper()} loaded ({kind}, n={n}): "
              f"y={sl:.4f}*x+{ic:.2f}")
    except Exception as e:
        print(f"[radar] Calibration {which.upper()} load failed: {type(e).__name__}: {e}")

def _apply_cal_a(raw_hr):
    """Apply radar A correction. Identity when no calibration file exists."""
    if raw_hr is None or not _cal_a_loaded:
        return raw_hr
    return round(_cal_a_slope * float(raw_hr) + _cal_a_intercept, 1)

def _apply_cal_b(raw_hr):
    """Apply radar B correction. Identity when no calibration file exists."""
    if raw_hr is None or not _cal_b_loaded:
        return raw_hr
    return round(_cal_b_slope * float(raw_hr) + _cal_b_intercept, 1)

# Auto-load on import
_load_radar_correction("a")
_load_radar_correction("b")

# ── Per-stream physiological filters (v7.4) ──
# Reject out-of-band and physiologically-implausible spikes before they
# reach the fusion layer. Each stream owns its own stateful filter.
_hr_filter_a = HRFilter(HR_ADULT_RESTING)
_br_filter_a = HRFilter(BR_ADULT_RESTING)
_hr_filter_b = HRFilter(HR_ADULT_RESTING)
_br_filter_b = HRFilter(BR_ADULT_RESTING)

# ── Shared radar state (thread-safe via GIL for simple reads) ──
_radar_data = {
    "heart_rate": None,
    "breath_rate": None,
    "distance": None,
    "presence": None,
    "people_count": None,  # v7.7: people count from sketch
    "last_update": 0,
    "connected": False,
    "port": None,
    "error": None,
}
_radar_thread = None
_radar_stop = threading.Event()

def get_radar_data():
    """Get latest radar readings. Returns dict or None if stale/unavailable."""
    if not _radar_data["connected"]:
        return None
    # Consider data stale after 5 seconds
    if time.time() - _radar_data["last_update"] > 5:
        return None
    return {
        "heart_rate": _radar_data["heart_rate"],
        "breath_rate": _radar_data["breath_rate"],
        "distance": _radar_data["distance"],
        "presence": _radar_data["presence"],
        # v7.7: raw phase streams so UI can drive real-time breath/EKG.
        "heart_phase": _radar_data.get("heart_phase"),
        "breath_phase": _radar_data.get("breath_phase"),
        # v7.7: people count
        "people_count": _radar_data.get("people_count"),
    }

def summary():
    """Snapshot used by forensic_export.

    v9.7.6: heart_rate is now run through `_apply_cal_a()` so any saved
    Radar A correction (calibration/radar_a_correction.json) is applied
    transparently. The internal _radar_data["heart_rate"] is left
    UNCHANGED — corrections only affect the public-facing snapshot, so
    if you ever need to retrain you still have the raw values to fit on.
    """
    return {
        "connected": _radar_data["connected"],
        "port": _radar_data["port"],
        "heart_rate": _apply_cal_a(_radar_data["heart_rate"]),
        "breath_rate": _radar_data["breath_rate"],
        "distance": _radar_data["distance"],
        "presence": _radar_data["presence"],
        "last_update": _radar_data["last_update"],
        "age_s": round(time.time() - _radar_data["last_update"], 2) if _radar_data["last_update"] else None,
        "error": _radar_data["error"],
        "hr_sqi": round(_hr_filter_a.sqi(), 3),
        "br_sqi": round(_br_filter_a.sqi(), 3),
        "hr_filter_stats": _hr_filter_a.stats(),
        "br_filter_stats": _br_filter_a.stats(),
    }


def get_filter_stats():
    """Return accept/reject counters for both radars (v7.4)."""
    return {
        "radar_a": {"hr": _hr_filter_a.stats(), "br": _br_filter_a.stats()},
        "radar_b": {"hr": _hr_filter_b.stats(), "br": _br_filter_b.stats()},
    }


def get_radar_status():
    return {
        "connected": _radar_data["connected"],
        "port": _radar_data["port"],
        "error": _radar_data["error"],
        "last_update": _radar_data["last_update"],
    }

# ══════════════════════════════════════════════
#  Auto-detect COM port
# ══════════════════════════════════════════════
def _find_radar_port():
    """Scan COM ports for the ESP32C6 / CH340 / CP2102 chip."""
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        # Prioritize known ESP32/CH340/CP2102 chips
        keywords = ["CH340", "CP210", "ESP32", "USB Serial", "USB-SERIAL", "Seeed", "XIAO"]
        for port in ports:
            desc = (port.description or "") + " " + (port.manufacturer or "")
            for kw in keywords:
                if kw.lower() in desc.lower():
                    return port.device
        # Fallback: return first available port
        if ports:
            return ports[0].device
    except Exception:
        pass
    return None

# ══════════════════════════════════════════════
#  JSON mode: read {"hr":..,"br":..} lines from Arduino sketch
# ══════════════════════════════════════════════
def _reader_json(ser):
    """Read JSON lines from Arduino-flashed ESP32C6.

    Expected payload (v7.5+ sketch):
      {"hr":72.3,"br":16.1,"dist":1.23,"presence":true,"people":1,
       "age_hr_ms":400,"age_br_ms":900,"age_dist_ms":200}

    v7.5+ calls getDistance() directly — dist is a real reading (0.2-6.0 m
    when a target is in range).  The distance EMA filter below accepts
    values >= 0.2 m and <= 6.0 m.

    Also tolerates the old v7.0 payload (which is always all-zeros because
    of a compile bug) and the startup {"status":"..."} line.
    """
    buf = b""
    debug_left = 20 if getattr(cfg, "RADAR_DEBUG", False) else 0
    # v7.7: tightened staleness windows. The sketch now emits every 100 ms,
    # so a metric that hasn't been refreshed in 3 s is dead (was 6–8 s).
    stale_hr_ms = 3000
    stale_br_ms = 4000
    old_sketch_warned = False
    all_zero_streak = 0
    while not _radar_stop.is_set():
        try:
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                if debug_left > 0:
                    try:
                        print("[radar-A raw]", line.decode("utf-8", errors="replace")[:200])
                    except Exception:
                        pass
                    debug_left -= 1
                try:
                    d = json.loads(line.decode("utf-8", errors="ignore"))
                except (json.JSONDecodeError, ValueError):
                    continue
                # Startup / status lines from the sketch carry no metrics.
                if "status" in d and not any(k in d for k in ("hr", "br", "dist")):
                    print(f"[radar-A] {d}")
                    continue
                # Detect the old v7.0 sketch: it never emits age_hr_ms,
                # and every payload is all zeros forever.
                # v8.4: tightened guard — only consider a packet "all-zero
                # garbage" when ALL of hr, br, dist, people are zero/null.
                # Previous check included presence=False which incorrectly
                # flagged valid packets where dist=0 (no target) but HR/BR
                # were non-zero.  Now packets with any non-zero vital pass.
                has_heartbeat_fields = ("age_hr_ms" in d) or ("age_br_ms" in d)
                is_all_zero = (
                    (d.get("hr") in (0, 0.0, None)) and
                    (d.get("br") in (0, 0.0, None)) and
                    (d.get("dist") in (0, 0.0, None)) and
                    (d.get("people") in (0, None))
                )
                if is_all_zero:
                    all_zero_streak += 1
                else:
                    all_zero_streak = 0
                if (not has_heartbeat_fields) and all_zero_streak >= 10 and not old_sketch_warned:
                    print("[radar-A][WARNING] Bridge is emitting the v7.0 all-zero payload "
                          "with no 'age_hr_ms' heartbeat. The OLD sketch is still on the ESP32C6. "
                          "Reflash radar_sketch/radar_sketch.ino (v7.1) with the Seeed_Arduino_mmWave "
                          "library installed. Readings will be suppressed until that is done.")
                    old_sketch_warned = True
                # Honor the age_* heartbeat fields: if a metric hasn't
                # refreshed recently, treat it as None rather than a stale zero.
                age_hr   = d.get("age_hr_ms",   0)
                age_br   = d.get("age_br_ms",   0)
                age_dist = d.get("age_dist_ms", 0)
                stale_dist_ms = 4000   # v7.7: was 5000
                hr = d.get("hr"); br = d.get("br"); dist = d.get("dist")
                # v7.4 — physiological filter (bounds + jump-rate + IQR)
                # 2026-04-25: previously `elif hr == 0` triggered an
                # immediate reset whenever the MR60BHA2 reported a
                # transient zero (which it does between every valid
                # sample, ~50% duty cycle while a person is seated).
                # The reset wiped the filter buffer so the next valid
                # 76 BPM was always treated as "first accepted"; the UI
                # then flickered to "--" between samples and the rolling
                # median never accumulated >= min_warmup samples for
                # outlier gating to engage. Fixed by tolerating zeros
                # while staleness remains under threshold; only reset
                # when the radar has truly stopped reporting (age above
                # stale threshold).
                if hr is not None and float(hr) > 0 and age_hr < stale_hr_ms:
                    filtered = _hr_filter_a.push(float(hr))
                    if filtered is not None:
                        _radar_data["heart_rate"] = round(filtered, 1)
                    # else: keep the last accepted value; do NOT overwrite
                elif age_hr >= stale_hr_ms:
                    # Genuinely stale -- nuke and reset the filter buffer
                    _radar_data["heart_rate"] = None
                    _hr_filter_a.reset()
                # else: hr == 0 but data is fresh -- keep last good value,
                # do NOT reset (the next valid sample will refresh)

                if br is not None and float(br) > 0 and age_br < stale_br_ms:
                    filtered_br = _br_filter_a.push(float(br))
                    if filtered_br is not None:
                        _radar_data["breath_rate"] = round(filtered_br, 1)
                elif age_br >= stale_br_ms:
                    _radar_data["breath_rate"] = None
                    _br_filter_a.reset()
                # else: br == 0 but fresh -- keep last good value
                # v7.7: accept any physically plausible distance (>=0.2 m,
                # <=6 m). Previously any 0.0 reading reset to None on every
                # frame — which blanked the UI between 500-ms updates.
                if dist is not None and float(dist) >= 0.2 and float(dist) <= 6.0 and age_dist < stale_dist_ms:
                    _dist_ema = _radar_data.get("_dist_ema") or float(dist)
                    _dist_ema = 0.35 * float(dist) + 0.65 * _dist_ema
                    _radar_data["_dist_ema"] = _dist_ema
                    _radar_data["distance"] = round(_dist_ema, 2)
                elif age_dist >= stale_dist_ms:
                    _radar_data["distance"] = None
                    _radar_data["_dist_ema"] = None
                if "presence" in d:
                    _radar_data["presence"] = bool(d["presence"])
                # v7.7: people count
                if "people" in d:
                    _radar_data["people_count"] = int(d["people"])
                # v9.1.1: sketch-version fallback. v7.3 and earlier stubs
                # always emit people=0 / presence=false / dist=0 even when
                # HR is valid — which makes the UI look broken. When HR
                # has been fresh in the last 5 s we can infer presence=True
                # and people_count=1. This keeps the panel useful on older
                # firmware while leaving newer (v7.5+) sketches untouched
                # because their non-zero payload already overrides below.
                _hr_val = _radar_data.get("heart_rate")
                _hr_age = time.time() - _radar_data.get("last_update", 0)
                if _hr_val is not None and _hr_age < 5.0:
                    if not _radar_data.get("presence"):
                        _radar_data["presence"] = True
                    if (_radar_data.get("people_count") in (None, 0)):
                        _radar_data["people_count"] = 1
                # v7.4: feed the calibration monitor
                _cal_monitor.push(_radar_data.get("heart_rate"),
                                  _radar_data.get("breath_rate"))
                # Store phases if present (for rPPG cross-validation)
                if "heart_ph" in d:
                    _radar_data["heart_phase"] = float(d["heart_ph"])
                if "breath_ph" in d:
                    _radar_data["breath_phase"] = float(d["breath_ph"])
                _radar_data["last_update"] = time.time()
        except Exception as e:
            _radar_data["error"] = f"reader: {e}"
            time.sleep(0.1)

# ══════════════════════════════════════════════
#  Raw mode: parse native MR60BHA2 binary protocol
# ══════════════════════════════════════════════
# Frame format: [53 59] [ctrl_h ctrl_l] [cmd_h cmd_l] [len_h len_l] [data...] [sum_h sum_l] [54 43]
# Header: 0x53 0x59, Footer: 0x54 0x43
# Known message types for MR60BHA2 (from Seeed/ESPHome sources):
#   ctrl=0x80 cmd=0x01: breathing report (data = 4-byte float LE)
#   ctrl=0x80 cmd=0x02: heart rate report (data = 4-byte float LE)
#   ctrl=0x80 cmd=0x03: distance report (data = 4-byte float LE)
#   ctrl=0x80 cmd=0x05: presence (data = 1 byte: 0/1)
# Note: exact type codes may vary by firmware version. The parser also
# tries alternate known codes from MR60BHA1/BHA2 variants.

FRAME_HEADER = bytes([0x53, 0x59])
FRAME_FOOTER = bytes([0x54, 0x43])

# Multiple possible type codes (ctrl<<8|cmd) for each reading
HR_TYPES = {0x8002, 0x8085, 0x0514}
BR_TYPES = {0x8001, 0x8082, 0x0515}
DIST_TYPES = {0x8003, 0x8083, 0x0516}
PRES_TYPES = {0x8005, 0x8081, 0x0A14}

def _parse_frame(frame_bytes):
    """Parse a single MR60BHA2 frame. Returns (type_code, data_bytes) or None."""
    if len(frame_bytes) < 10:
        return None
    if frame_bytes[:2] != FRAME_HEADER or frame_bytes[-2:] != FRAME_FOOTER:
        return None
    ctrl_h, ctrl_l = frame_bytes[2], frame_bytes[3]
    cmd_h, cmd_l = frame_bytes[4], frame_bytes[5]
    data_len = (frame_bytes[6] << 8) | frame_bytes[7]
    if len(frame_bytes) < 10 + data_len:
        return None
    data = frame_bytes[8:8 + data_len]
    type_code = (ctrl_h << 8) | cmd_h  # simplified type identifier
    type_code2 = (frame_bytes[2] << 8) | frame_bytes[3]  # ctrl word
    type_code3 = (frame_bytes[4] << 8) | frame_bytes[5]  # cmd word
    full_type = (ctrl_h << 24) | (ctrl_l << 16) | (cmd_h << 8) | cmd_l
    return (type_code, type_code2, type_code3, full_type, data)

def _extract_float_le(data):
    """Extract little-endian float from 4 bytes."""
    if len(data) >= 4:
        try:
            val = struct.unpack("<f", bytes(data[:4]))[0]
            if -1000 < val < 1000:  # sanity check
                return round(val, 1)
        except Exception:
            pass
    return None

def _reader_raw(ser):
    """Read and parse native MR60BHA2 binary protocol frames."""
    buf = bytearray()
    while not _radar_stop.is_set():
        try:
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue
            buf.extend(chunk)
            # Find and process frames
            while True:
                # Find header
                idx = buf.find(FRAME_HEADER)
                if idx < 0:
                    buf.clear()
                    break
                if idx > 0:
                    buf = buf[idx:]  # discard junk before header
                # Need at least 10 bytes for minimum frame
                if len(buf) < 10:
                    break
                # Read data length
                data_len = (buf[6] << 8) | buf[7]
                frame_len = 8 + data_len + 2 + 2  # header(2)+ctrl(2)+cmd(2)+len(2)+data+crc(2)+footer(2)
                if len(buf) < frame_len:
                    break
                # Check footer
                if buf[frame_len - 2:frame_len] != FRAME_FOOTER:
                    buf = buf[2:]  # skip bad header, try again
                    continue
                frame = bytes(buf[:frame_len])
                buf = buf[frame_len:]
                parsed = _parse_frame(frame)
                if parsed is None:
                    continue
                tc, tc2, tc3, full_tc, data = parsed
                # Try all known type code positions
                all_codes = {tc, tc2, tc3, full_tc & 0xFFFF, (full_tc >> 16) & 0xFFFF}
                if all_codes & HR_TYPES:
                    val = _extract_float_le(data)
                    if val:
                        f = _hr_filter_a.push(float(val))
                        if f is not None:
                            _radar_data["heart_rate"] = round(f, 1)
                            _radar_data["last_update"] = time.time()
                elif all_codes & BR_TYPES:
                    val = _extract_float_le(data)
                    if val:
                        f = _br_filter_a.push(float(val))
                        if f is not None:
                            _radar_data["breath_rate"] = round(f, 1)
                            _radar_data["last_update"] = time.time()
                elif all_codes & DIST_TYPES:
                    val = _extract_float_le(data)
                    if val and 0 < val < 10:
                        _radar_data["distance"] = round(val, 2)
                        _radar_data["last_update"] = time.time()
                elif all_codes & PRES_TYPES:
                    if len(data) >= 1:
                        _radar_data["presence"] = data[0] > 0
                        _radar_data["last_update"] = time.time()
        except Exception:
            time.sleep(0.1)

# ══════════════════════════════════════════════
#  Start / Stop
# ══════════════════════════════════════════════
def start():
    """Start radar reader in background thread."""
    global _radar_thread
    if not cfg.RADAR_ENABLED:
        print("[--] Radar disabled in config")
        return False
    try:
        import serial
    except ImportError:
        _radar_data["error"] = "pyserial not installed (pip install pyserial)"
        print(f"[--] Radar: {_radar_data['error']}")
        return False
    port = cfg.RADAR_PORT
    if port == "auto":
        port = _find_radar_port()
    # v9.8.1 — if the configured port isn't present in the OS (cable
    # got replugged into a different USB port, Windows enumerated it
    # on a different COM #), fall back to auto-detect instead of
    # giving up. Previous behaviour: failed silently with "COM3 not
    # found"; then start_dual() would grab the only-physically-present
    # radar as Radar B, surprising the operator who only has ONE radar.
    if port:
        try:
            import serial.tools.list_ports as _ltp
            _present = {p.device for p in _ltp.comports()}
            if port not in _present:
                _alt = _find_radar_port()
                if _alt and _alt != port:
                    print(f"[--] Radar: configured port {port} not present, "
                          f"falling back to auto-detected {_alt}")
                    port = _alt
        except Exception:
            pass  # nolint:silent-except port presence check best-effort
    if not port:
        _radar_data["error"] = "No serial port found. Set RADAR_PORT in config.py"
        print(f"[--] Radar: {_radar_data['error']}")
        return False
    # v8.3: pre-flight port check — test if we can open the port before
    # committing to a reader thread. Gives a clear error message when
    # another process (orphan Python, Arduino IDE) holds the port.
    # v9.7.1: cooperative port_lock advisory check before the serial open
    # so we surface a structured "held by pid=X" error instead of
    # WinError 5.
    try:
        import port_lock as _plock
    except Exception as _plock_err:
        _plock = None  # type: ignore[assignment]
        print(f"[--] Radar: port_lock unavailable ({_plock_err})")
    if _plock is not None and not _plock.acquire(port, owner="radar_a"):
        held = _plock.describe(port) or {}
        msg = (f"[RADAR] {port} held by pid={held.get('pid')} "
               f"({held.get('owner')}) — refusing to start")
        _radar_data["error"] = msg
        _radar_data["connected"] = False
        print(msg)
        return False
    try:
        _test_ser = serial.Serial(port, cfg.RADAR_BAUD, timeout=0.1)
        _test_ser.close()
    except PermissionError:
        msg = f"[RADAR] {port} held by another process -- kill old instances first"
        _radar_data["error"] = msg
        _radar_data["connected"] = False
        if _plock is not None:
            _plock.release(port)
        print(msg)
        return False
    except Exception as e:
        _radar_data["error"] = str(e)
        _radar_data["connected"] = False
        if _plock is not None:
            _plock.release(port)
        print(f"[--] Radar: {e}")
        return False
    try:
        ser = serial.Serial(port, cfg.RADAR_BAUD, timeout=1)
        _radar_data["connected"] = True
        _radar_data["port"] = port
        _radar_data["error"] = None
        _radar_stop.clear()
        reader = _reader_json if cfg.RADAR_MODE == "json" else _reader_raw
        _radar_thread = threading.Thread(target=reader, args=(ser,), daemon=True)
        _radar_thread.start()
        print(f"[OK] Radar on {port} (mode={cfg.RADAR_MODE})")
        return True
    except Exception as e:
        _radar_data["error"] = str(e)
        _radar_data["connected"] = False
        if _plock is not None:
            _plock.release(port)
        print(f"[--] Radar: {e}")
        return False

def stop():
    """Stop radar reader thread."""
    _radar_stop.set()
    _radar_data["connected"] = False
    # v9.7.1: release the port_lock so the next launch finds a clean slate.
    try:
        import port_lock as _plock
        if _radar_data.get("port"):
            _plock.release(_radar_data["port"])
    except Exception as e:
        print(f"[radar] port_lock release on stop: {type(e).__name__}: {e}")

# ══════════════════════════════════════════════
#  DUAL RADAR (Radar B — second MR60BHA2 unit)
# ══════════════════════════════════════════════
_radar_b_data = {
    "heart_rate": None, "breath_rate": None,
    "distance": None, "presence": None,
    "people_count": None,  # v7.7: people count from radar B
    "last_update": 0, "connected": False, "port": None, "error": None,
}
_radar_b_thread = None
_radar_b_stop = threading.Event()

def get_radar_b_data():
    """Get latest readings from second radar. Returns dict or None."""
    if not _radar_b_data["connected"]:
        return None
    if time.time() - _radar_b_data["last_update"] > 5:
        return None
    return {
        "heart_rate": _radar_b_data["heart_rate"],
        "breath_rate": _radar_b_data["breath_rate"],
        "distance": _radar_b_data["distance"],
        "presence": _radar_b_data["presence"],
        "people_count": _radar_b_data.get("people_count"),
    }

def get_radar_b_status():
    return {
        "connected": _radar_b_data["connected"],
        "port": _radar_b_data["port"],
        "error": _radar_b_data["error"],
    }


def summary_b():
    """Snapshot of Radar B identical in shape to `summary()` for Radar A.
    v9.7.4: added so the PAD `radar_presence` branch can merge data
    from both radars instead of relying on Radar A alone — when Radar A
    is still warming up its chest-wall HR/BR lock, Radar B may already
    have vitals (and vice versa). The branch picks whichever has the
    freshest signal."""
    # v9.7.6: heart_rate gated through _apply_cal_b — same pattern as
    # summary() above. radar_b_correction.json is auto-loaded at boot.
    return {
        "connected": _radar_b_data["connected"],
        "port":      _radar_b_data["port"],
        "heart_rate":  _apply_cal_b(_radar_b_data["heart_rate"]),
        "breath_rate": _radar_b_data["breath_rate"],
        "distance":    _radar_b_data["distance"],
        "presence":    _radar_b_data["presence"],
        "last_update": _radar_b_data["last_update"],
        "age_s": (round(time.time() - _radar_b_data["last_update"], 2)
                  if _radar_b_data["last_update"] else None),
        "error": _radar_b_data["error"],
        "people_count": _radar_b_data.get("people_count"),
    }

def _reader_json_b(ser):
    """JSON reader for radar B — writes to _radar_b_data."""
    buf = b""
    debug_left_b = 20 if getattr(cfg, "RADAR_DEBUG", False) else 0
    while not _radar_b_stop.is_set():
        try:
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                if debug_left_b > 0:
                    try:
                        print("[radar-B raw]", line.decode("utf-8", errors="replace")[:200])
                    except Exception:
                        pass
                    debug_left_b -= 1
                try:
                    d = json.loads(line.decode("utf-8", errors="ignore"))
                    # 2026-04-25 (v9.7.1): mirror radar A's tolerant-zero
                    # pattern (radar.py:222-254). Previously every transient
                    # hr=0 / br=0 from the MR60BHA2 (~50% duty cycle while a
                    # person is seated) triggered an immediate filter reset,
                    # so the rolling median never accumulated min_warmup
                    # samples and outlier gating never engaged. Result: UI
                    # flickered between '--' and wild post-reset values.
                    # Fixed by tolerating zeros while staleness remains under
                    # threshold; only reset when the radar has truly stopped
                    # reporting (age >= stale threshold).
                    age_hr_b = d.get("age_hr_ms", 0)
                    age_br_b = d.get("age_br_ms", 0)
                    stale_hr_ms_b = 3000
                    stale_br_ms_b = 4000
                    hr_b = d.get("hr"); br_b = d.get("br")
                    if hr_b is not None and float(hr_b) > 0 and age_hr_b < stale_hr_ms_b:
                        filtered_hr_b = _hr_filter_b.push(float(hr_b))
                        if filtered_hr_b is not None:
                            _radar_b_data["heart_rate"] = round(filtered_hr_b, 1)
                        # else: keep the last accepted value
                    elif age_hr_b >= stale_hr_ms_b:
                        _radar_b_data["heart_rate"] = None
                        _hr_filter_b.reset()
                    # else: hr == 0 but fresh -- keep last good value

                    if br_b is not None and float(br_b) > 0 and age_br_b < stale_br_ms_b:
                        filtered_br_b = _br_filter_b.push(float(br_b))
                        if filtered_br_b is not None:
                            _radar_b_data["breath_rate"] = round(filtered_br_b, 1)
                    elif age_br_b >= stale_br_ms_b:
                        _radar_b_data["breath_rate"] = None
                        _br_filter_b.reset()
                    # else: br == 0 but fresh -- keep last good value
                    # v7.8: mirror radar A's v7.7 distance handling — apply
                    # the same 0.20–6.00 m plausibility gate, α=0.35 EMA, and
                    # staleness window (4 s) so dual radar doesn't degrade to
                    # raw 0.0 spikes when the sensor hasn't locked yet.
                    if "dist" in d:
                        age_dist_b = d.get("age_dist_ms", 0)
                        stale_dist_ms_b = 4000
                        dist_raw = d.get("dist")
                        if (dist_raw is not None
                                and float(dist_raw) >= 0.2
                                and float(dist_raw) <= 6.0
                                and age_dist_b < stale_dist_ms_b):
                            _dist_ema_b = _radar_b_data.get("_dist_ema") or float(dist_raw)
                            _dist_ema_b = 0.35 * float(dist_raw) + 0.65 * _dist_ema_b
                            _radar_b_data["_dist_ema"] = _dist_ema_b
                            _radar_b_data["distance"] = round(_dist_ema_b, 2)
                        elif age_dist_b >= stale_dist_ms_b:
                            _radar_b_data["distance"] = None
                            _radar_b_data["_dist_ema"] = None
                    if "presence" in d:
                        _radar_b_data["presence"] = bool(d["presence"])
                    # v7.7: people count
                    if "people" in d:
                        _radar_b_data["people_count"] = int(d["people"])
                    # v9.1.1: sketch-version fallback (see radar A comment)
                    _hr_val_b = _radar_b_data.get("heart_rate")
                    _hr_age_b = time.time() - _radar_b_data.get("last_update", 0)
                    if _hr_val_b is not None and _hr_age_b < 5.0:
                        if not _radar_b_data.get("presence"):
                            _radar_b_data["presence"] = True
                        if _radar_b_data.get("people_count") in (None, 0):
                            _radar_b_data["people_count"] = 1
                    _radar_b_data["last_update"] = time.time()
                except (json.JSONDecodeError, ValueError):
                    pass
        except Exception:
            time.sleep(0.1)

def _find_all_radar_ports():
    """Find all available radar COM ports (for dual setup)."""
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        keywords = [
            "CH340", "CP210", "ESP32", "USB Serial", "USB-SERIAL",
            "Seeed", "XIAO", "CH9102",
            # Windows Portuguese / Spanish / French localisation variants.
            # The generic Microsoft USB-serial driver labels the port as
            # "Dispositivo de Série USB" (PT), "Dispositivo serie USB" (ES),
            # "Périphérique série USB" (FR) — none of which match the
            # English keywords above.
            "Série USB", "serie USB", "série USB",
            "Série UART", "Dispositivo de S", "USB-serial",
        ]
        found = []
        for port in ports:
            desc = (port.description or "") + " " + (port.manufacturer or "")
            for kw in keywords:
                if kw.lower() in desc.lower():
                    found.append(port.device)
                    break
        return found
    except Exception:
        return []

def start_dual():
    """Start second radar on the next available port after radar A."""
    global _radar_b_thread
    try:
        import serial
    except ImportError:
        return False
    all_ports = _find_all_radar_ports()
    # Filter out port already used by radar A
    used = _radar_data.get("port")
    available = [p for p in all_ports if p != used]
    if not available:
        _radar_b_data["error"] = "No second radar port found"
        return False
    port = available[0]
    # v9.7.1: cooperative port_lock for radar B (separate owner string so
    # we can tell radar_a vs radar_b apart when diagnosing).
    try:
        import port_lock as _plock_b
    except Exception:
        _plock_b = None  # type: ignore[assignment]
    if _plock_b is not None and not _plock_b.acquire(port, owner="radar_b"):
        held = _plock_b.describe(port) or {}
        msg = (f"[RADAR-B] {port} held by pid={held.get('pid')} "
               f"({held.get('owner')}) — refusing to start")
        _radar_b_data["error"] = msg
        print(msg)
        return False
    try:
        ser = serial.Serial(port, cfg.RADAR_BAUD, timeout=1)
        _radar_b_data["connected"] = True
        _radar_b_data["port"] = port
        _radar_b_data["error"] = None
        _radar_b_stop.clear()
        _radar_b_thread = threading.Thread(target=_reader_json_b, args=(ser,), daemon=True)
        _radar_b_thread.start()
        print(f"[OK] Radar B on {port}")
        return True
    except Exception as e:
        _radar_b_data["error"] = str(e)
        if _plock_b is not None:
            _plock_b.release(port)
        return False

def stop_dual():
    _radar_b_stop.set()
    _radar_b_data["connected"] = False
    try:
        import port_lock as _plock_b
        if _radar_b_data.get("port"):
            _plock_b.release(_radar_b_data["port"])
    except Exception as e:
        print(f"[radar-B] port_lock release on stop: {type(e).__name__}: {e}")
