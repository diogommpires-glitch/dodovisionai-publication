"""
DODO VISION AI — GSR (Galvanic Skin Response) Sensor
Reads electrodermal activity (EDA/GSR) via Arduino serial.

RECOMMENDED SENSOR: Grove GSR Sensor v1.2 by Seeed Studio
  - Price: ~$8-13 USD
  - Connection: Arduino analog pin A0
  - Finger electrodes included
  - Compatible with your existing Seeed ecosystem (same as MR60BHA2 radar)

SETUP:
  1. Buy: "Grove GSR Sensor" from Seeed Studio, Amazon, or eBay
  2. Connect: GSR sensor to Arduino analog pin A0 (or Grove Base Shield A0)
  3. Flash: gsr_sketch.ino to Arduino (Nano, Uno, or XIAO ESP32C6)
  4. Calibrate: Without wearing sensor, adjust potentiometer until reading ~512
  5. Wear: Attach finger electrodes to index + middle finger
  6. Set: GSR_PORT in config.py (or leave "auto")

PROTOCOL: Arduino outputs one JSON line per reading at ~10 Hz:
  {"gsr": 523, "conductance": 4.2}
  gsr = raw analog value (0-1023)
  conductance = computed skin conductance in microSiemens

WHY GSR MATTERS FOR LIE DETECTION:
  Rashid et al. (2024) found pupil response is #1, EDA/GSR is #2 for
  predicting deception. The autonomic nervous system causes sweating
  under cognitive load/stress that is nearly impossible to suppress voluntarily.
  GSR response latency is 1-3 seconds after the triggering event.
"""
import threading, time, json
import numpy as np
from collections import deque
import config as cfg

# ── Shared GSR state ──
_gsr_data = {
    "raw": None,              # raw analog value 0-1023
    "conductance_us": None,   # skin conductance in microSiemens
    "scr_amplitude": None,    # skin conductance response (phasic component)
    "scl_level": None,        # skin conductance level (tonic component)
    "connected": False,
    "port": None,
    "error": None,
    "last_update": 0,
}

# ── History for phasic/tonic decomposition ──
_gsr_history = deque(maxlen=600)  # 60 seconds at 10Hz
_gsr_baseline = None
_gsr_thread = None
_gsr_stop = threading.Event()

def get_gsr_data():
    """Get latest GSR readings. Returns dict or None if stale."""
    if not _gsr_data["connected"]:
        return None
    if time.time() - _gsr_data["last_update"] > 3:
        return None
    return {
        "raw": _gsr_data["raw"],
        "conductance_us": _gsr_data["conductance_us"],
        "scr_amplitude": _gsr_data["scr_amplitude"],
        "scl_level": _gsr_data["scl_level"],
    }

def get_gsr_status():
    return {
        "connected": _gsr_data["connected"],
        "port": _gsr_data["port"],
        "error": _gsr_data["error"],
    }

def _find_gsr_port():
    """Scan COM ports for Arduino/CH340."""
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        keywords = ["CH340", "Arduino", "USB Serial", "USB-SERIAL", "CP210", "FTDI"]
        for port in ports:
            desc = (port.description or "") + " " + (port.manufacturer or "")
            for kw in keywords:
                if kw.lower() in desc.lower():
                    # Skip ports already used by radar or oximeter
                    if hasattr(cfg, 'RADAR_PORT') and port.device == cfg.RADAR_PORT:
                        continue
                    if hasattr(cfg, 'OXIMETER_PORT') and port.device == cfg.OXIMETER_PORT:
                        continue
                    return port.device
    except Exception:
        pass
    return None

def _decompose_phasic_tonic(conductance):
    """Simple phasic/tonic decomposition.
    Tonic (SCL): slow-moving baseline (rolling mean over 30s)
    Phasic (SCR): fast deviations from tonic (event responses, 1-3s latency)
    """
    _gsr_history.append(conductance)
    if len(_gsr_history) < 30:
        return conductance, 0  # not enough data yet

    # Tonic: rolling mean of last 300 samples (~30s at 10Hz)
    arr = np.array(list(_gsr_history))
    window = min(len(arr), 300)
    scl = float(np.mean(arr[-window:]))

    # Phasic: current value minus tonic
    scr = max(0, conductance - scl)

    return round(scl, 3), round(scr, 3)

def _reader(ser):
    """Read JSON lines from Arduino GSR sketch."""
    buf = b""
    while not _gsr_stop.is_set():
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
                try:
                    d = json.loads(line.decode("utf-8", errors="ignore"))
                    raw = d.get("gsr")
                    conductance = d.get("conductance")
                    if raw is not None:
                        _gsr_data["raw"] = int(raw)
                    if conductance is not None:
                        cond = float(conductance)
                        _gsr_data["conductance_us"] = round(cond, 3)
                        scl, scr = _decompose_phasic_tonic(cond)
                        _gsr_data["scl_level"] = scl
                        _gsr_data["scr_amplitude"] = scr
                    _gsr_data["last_update"] = time.time()
                except (json.JSONDecodeError, ValueError):
                    pass
        except Exception:
            time.sleep(0.1)

def start():
    """Start GSR reader in background thread."""
    global _gsr_thread
    if not getattr(cfg, 'GSR_ENABLED', False):
        print("[--] GSR disabled in config")
        return False
    try:
        import serial
    except ImportError:
        _gsr_data["error"] = "pyserial not installed"
        print(f"[--] GSR: {_gsr_data['error']}")
        return False
    port = getattr(cfg, 'GSR_PORT', 'auto')
    if port == "auto":
        port = _find_gsr_port()
    if not port:
        _gsr_data["error"] = "No GSR port found. Set GSR_PORT in config.py"
        print(f"[--] GSR: {_gsr_data['error']}")
        return False
    try:
        baud = getattr(cfg, 'GSR_BAUD', 9600)
        ser = serial.Serial(port, baud, timeout=1)
        _gsr_data["connected"] = True
        _gsr_data["port"] = port
        _gsr_data["error"] = None
        _gsr_stop.clear()
        _gsr_thread = threading.Thread(target=_reader, args=(ser,), daemon=True)
        _gsr_thread.start()
        print(f"[OK] GSR on {port} ({baud} baud)")
        return True
    except Exception as e:
        _gsr_data["error"] = str(e)
        _gsr_data["connected"] = False
        print(f"[--] GSR: {e}")
        return False

def stop():
    _gsr_stop.set()
    _gsr_data["connected"] = False
