"""
DODO VISION AI v6 — Truth Engine Edition
All v5 features + multi-modal lie detection (Whisper + vocal stress + biometric baseline comparison).
Run: python main.py
"""
import os
import json
import sys as _sys
from pathlib import Path

# ════════════════════════════════════════════════════════════════════
# v9.8.1 (2026-05-12) — CRITICAL: alias "main" to "__main__" in
# sys.modules BEFORE anything else runs.
#
# WHY this exists: this file is launched as ``python -u main.py``, so
# its module name is ``__main__`` and it lives at
# ``sys.modules["__main__"]``. Several other modules do
# ``from main import person_states`` (llm.py:575,
# pad/branches/rppg_branch.py:66, app/api/webrtc.py:41 -- all inside
# request/coroutine handlers that fire after boot). When THAT runs,
# ``sys.modules.get("main")`` is None, so Python imports main.py FROM
# DISK A SECOND TIME as a fresh module called "main".
#
# Consequence (caught via py-spy on the running server): DUPLICATE
# ``sensor_poll`` + ``heartbeat`` threads, a SECOND ``person_states``
# dict that never gets WS frames, and routes.py's ``_main_globals()``
# helper (which checks "main" first, then "__main__") returning the
# EMPTY re-imported dict. End-user symptom: PAD AGREEMENT / PAD DEBUG
# panels stuck on "pid=0 -- server has no tracked persons" even though
# the WS path is actively tracking a face; HTTP endpoints stall under
# the doubled thread load.
#
# The two-line alias below makes any subsequent ``import main`` or
# ``from main import X`` resolve to the SAME running __main__ module,
# so the second import is a no-op. Must run BEFORE any other module is
# imported (otherwise that module might race ahead and trigger the
# re-import itself).
# ════════════════════════════════════════════════════════════════════
if "main" not in _sys.modules:
    _sys.modules["main"] = _sys.modules["__main__"]

# ────────────────────────────────────────────────────────────────────
# v9.8.1 (2026-05-12) — Windows-venv interpreter pinning.
#
# CONTEXT: On Windows, ``.venv\Scripts\python.exe`` created by
# ``python -m venv .venv`` is actually the Python *Launcher* (py.exe)
# stub. When it boots, it spawns the REAL python at
# ``sys._base_executable`` (the system install) as a child, passing
# PYTHONHOME so the child reports ``sys.prefix == .venv``. We observed
# this on the operator's box as a two-process tree -- the launcher
# stub at ~4 MB and the real interpreter at ~5 GB. Both correct;
# both running the venv. Confirmed via /health/full →
# ``identity.executable = .venv\Scripts\python.exe``.
#
# That's normal behaviour, NOT a bug. What WAS a bug, and easy to
# trigger here, is any code that uses ``sys._base_executable`` (or
# resolves ``python`` against PATH) to spawn helpers -- because that
# spawns the system python WITHOUT PYTHONHOME, dropping out of the
# venv. The defences below close that hole and emit a one-line boot
# trace so future investigations don't have to repeat ours.
# ────────────────────────────────────────────────────────────────────
def _v981_pin_venv():
    try:
        os.makedirs("logs", exist_ok=True)
    except Exception:
        pass
    _boot_exe = _sys.executable
    _boot_base = getattr(_sys, "_base_executable", None) or getattr(_sys, "base_executable", None)
    _boot_in_venv = (_sys.prefix != _sys.base_prefix)
    try:
        with open(os.path.join("logs", "boot_trace.log"), "a", encoding="utf-8") as _f:
            import time as _t
            _f.write(
                f"[{_t.strftime('%Y-%m-%d %H:%M:%S')}] pid={os.getpid()} "
                f"exe={_boot_exe!r} base_exe={_boot_base!r} "
                f"in_venv={_boot_in_venv}\n"
            )
    except Exception:
        pass

    # 1. Pin sys._base_executable to the venv python so any third-party
    #    code that uses base_executable for subprocess spawning stays
    #    inside the venv.
    try:
        if _boot_in_venv and _boot_base and _boot_base != _boot_exe:
            _sys._base_executable = _boot_exe  # type: ignore[attr-defined]
            try:
                _sys.base_executable = _boot_exe  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass

    # 2. Pin multiprocessing's spawn executable. Critical on Windows
    #    where mp.Process under "spawn" mode re-launches python; without
    #    this any DataLoader worker / mp.Pool spawn risks the system
    #    interpreter and a duplicate PAD/TF/ONNX init.
    try:
        import multiprocessing as _mp
        _mp.set_executable(_sys.executable)
        try:
            from multiprocessing import spawn as _mps
            _mps._python_exe = _sys.executable
        except Exception:
            pass
    except Exception:
        pass

    # 3. Export PYTHONEXECUTABLE so any child that honours the env var
    #    (some embedding hosts, jupyter kernel managers) picks up the
    #    venv interpreter.
    try:
        os.environ["PYTHONEXECUTABLE"] = _sys.executable
    except Exception:
        pass


_v981_pin_venv()

# v9.5.x: enable Python's faulthandler BEFORE any heavy native import
# (ONNX Runtime, MediaPipe, OpenCV DirectML). If one of them segfaults
# mid-frame — as we saw happen around minute 1 of a session on Windows +
# DirectML + RTX 3060 — faulthandler dumps the C / Python stack to
# stderr AND to logs/crash.log so the next restart has breadcrumbs.
from app.core.hardware import setup_hardware
setup_hardware()

import config as cfg
import asyncio, time
_start_time = time.time()  # v8.3: uptime tracking for /health
import cv2, numpy as np
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    WebSocket,
    WebSocketDisconnect,
    Request,
    HTTPException,
)
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse

import detectors
from app.services.deepface_worker import worker_instance
import analysis
import rppg as rppg_mod
import radar
import oximeter
import gsr
import llm
import logger
import face_registry
import calibration
import reference_ranges
import truth_engine
import audio as audio_mod
import voice_monitor
import forensic
import dual_camera
import calibration_lap   # v8.7 (Step 8): 90-second emotional baseline lap
import forensic_log      # v8.7 (Step 9): SQLite append-only event log
from utils import sanitize
from ui import UI_HTML
import sensor_state  # v8.9 Step 1-2: isolated sensor state buffer
import uart_bridge   # v8.9 Steps 10-11: MCU UART bridge for oximeter I2C isolation

# v9.0 Step 2: start the 50 Hz Zero-Order Hold mediator for continuous
# sensor fusion — guarantees downstream data flow even when individual
# sensors miss their 20 ms compute window.
try:
    _zoh_mediator = sensor_state.get_mediator()
    print("[OK] ZOH mediator started (50 Hz rigid tick, age-decay interpolation)")
except Exception as _zoh_err:
    _zoh_mediator = None
    print(f"[--] ZOH mediator: {_zoh_err}")

# v9.0 Step 9: Lempel-Ziv Complexity photic habituation safety monitor
try:
    from eeg import get_photic_monitor
    _photic_monitor = get_photic_monitor()
    print("[OK] LZC photic habituation monitor loaded (WCAG 2.3.1)")
except ImportError:
    _photic_monitor = None

# v9.0 Step 3: WSL2 containerization auto-detect
try:
    from gpu_enforce import detect_wsl2, enforce_wsl2_gpu
    if detect_wsl2():
        _wsl2_result = enforce_wsl2_gpu(getattr(cfg, "WSL2_GPU_MEM_LIMIT_GB", 4))
        if _wsl2_result.get("container"):
            print(f"[OK] WSL2 GPU passthrough: {_wsl2_result.get('gpu_info', 'detected')}")
        else:
            print(f"[--] WSL2 GPU: {_wsl2_result.get('error', 'unknown')}")
except Exception:
    pass

# v8.9 Step 7: DODO causal structure learning engine
try:
    import dodo_causal
    _DODO_ENABLED = True
    print("[OK] DODO causal engine loaded (vectorized DAG inference)")
except ImportError:
    _DODO_ENABLED = False
    dodo_causal = None

# v8.8: Intrinsic Cognitive Security — adversarial stimuli monitor
try:
    import ics_monitor
    _ICS_ENABLED = True
    print("[OK] ICS monitor loaded (Intrinsic Cognitive Security)")
except ImportError:
    _ICS_ENABLED = False

# v8.8: mmWave 120 GHz fusion + TDMNN cognitive proxy scaffolds
try:
    from mmwave_fusion import MmWaveRadarFusion
    _mmwave_fusion = MmWaveRadarFusion()
    print("[OK] mmWave 120 GHz fusion scaffold loaded")
except ImportError:
    _mmwave_fusion = None
try:
    from tdmnn_proxy import TDMNNProxy
    _tdmnn = TDMNNProxy()
    print("[OK] TDMNN cognitive proxy scaffold loaded")
except ImportError:
    _tdmnn = None

# v8.7 (Step 9): boot-time event so a fresh DB always has at least one row.
try:
    forensic_log.log_event("session_start",
                           severity=forensic_log.SEVERITY_INFO,
                           payload={"app_version": getattr(cfg, "APP_VERSION", "?")})
except Exception:
    pass

# ── v8: Identity persistence & spatial radar assignment ──
try:
    import face_identity
except ImportError:
    face_identity = None
try:
    import radar_assign
except ImportError:
    radar_assign = None
try:
    import eeg
except ImportError:
    eeg = None

# ── v7: Per-person tracking & state (Steps 1, 4, 9) ──
try:
    from tracker import FaceTracker
    from person_state import PersonState
    from frame_quality import frame_quality as fq
    face_tracker = FaceTracker()
    _V7_ENABLED = True
    print("[OK] v7 modules loaded (tracker, person_state, frame_quality)")
except Exception as _v7_err:
    print(f"[WARN] v7 modules failed to load ({_v7_err}), running in v6 compatibility mode")
    _V7_ENABLED = False
    # Stubs so the code doesn't crash
    class _StubTracker:
        def update(self, *a, **kw): return []
        def reset(self): pass
    class PersonState:
        def __init__(self, *a, **kw):
            self.face_affect_snapshot = None

        def build_affect_dict(self): return {}
    class _StubFQ:
        def assess(self, img): return {"overall_ok": True, "warnings": [], "confidence_mul": 1.0}
    face_tracker = _StubTracker()
    fq = _StubFQ()

person_states = {}  # track_id -> PersonState
_frame_count = 0

# ── v8.9 Step 1-2: Background sensor poller → SensorStateBuffer ──
# Pushes all external sensor data into the decoupled state buffer at ~10 Hz
# so the frame processor never blocks on sensor I/O.
import threading as _threading_mod

_sensor_buf = sensor_state.get_buffer()
_sensor_poll_stop = _threading_mod.Event()

def _sensor_poller():
    """Background thread: poll all sensors and push into state buffer."""
    while not _sensor_poll_stop.is_set():
        try:
            rd = radar.get_radar_data()
            _sensor_buf.push("radar_a", rd, confidence=1.0 if rd else 0.0)
        except Exception:
            _sensor_buf.push("radar_a", None, confidence=0.0, error="poll_error")
        try:
            rb = radar.get_radar_b_data()
            _sensor_buf.push("radar_b", rb, confidence=1.0 if rb else 0.0)
        except Exception:
            _sensor_buf.push("radar_b", None, confidence=0.0, error="poll_error")
        try:
            # v8.9 Steps 10-11: prefer UART bridge if enabled (MCU-buffered I2C at 400kHz)
            ox = uart_bridge.get_bridge_data()
            if ox is None:
                ox = oximeter.get_oximeter_data()  # fallback to direct HID/serial
            # v9.1.2: confidence was pinned at signal_strength/3, which made
            # the sensor_state buffer reject perfectly valid readings whenever
            # the oximeter reported signal=0 for a beat (common during motion
            # or when the finger is on but not centered). Floor to 0.5 when
            # ANY heart_rate or spo2 value is present so the buffer keeps the
            # data and the UI keeps showing the number.
            if ox:
                _sig = (ox.get("signal_strength") or 0) / 3.0
                _has_reading = (ox.get("heart_rate") is not None) or (ox.get("spo2") is not None)
                _conf = max(_sig, 0.5) if _has_reading else _sig
                _sensor_buf.push("oximeter", ox, confidence=min(1.0, _conf))
            else:
                _sensor_buf.push("oximeter", None, confidence=0.0)
        except Exception:
            _sensor_buf.push("oximeter", None, confidence=0.0, error="poll_error")
        try:
            gs = gsr.get_gsr_data()
            _sensor_buf.push("gsr", gs, confidence=1.0 if gs else 0.0)
        except Exception:
            _sensor_buf.push("gsr", None, confidence=0.0, error="poll_error")
        _sensor_poll_stop.wait(0.1)  # ~10 Hz

_sensor_poll_thread = _threading_mod.Thread(target=_sensor_poller, daemon=True, name="sensor_poll")
_sensor_poll_thread.start()
print("[OK] Sensor state buffer started (10 Hz polling, Dempster-Shafer weights)")

# ── v8.9 Step 4: Centralized last-valid-state cache for WS broadcast ──
# Thread-safe container that always holds the most recent valid frame result.
# The WS broadcast loop reads from here instead of waiting on frame processing,
# guaranteeing the WS never references an incomplete or uninitialized payload.
class _FrameResultCache:
    """Thread-safe single-slot cache holding the latest valid frame result.

    The frame processor writes here after every successful _process_frame_*().
    The WS handler reads the latest result; if processing stalls the WS loop
    can re-broadcast the last known good state rather than crashing.
    """
    def __init__(self):
        self._lock = _threading_mod.Lock()
        self._result = None
        self._timestamp = 0.0
        self._frame_id = 0

    def put(self, result: dict) -> None:
        """Store a new frame result (called by frame processor)."""
        with self._lock:
            self._result = result
            self._timestamp = time.time()
            self._frame_id += 1

    def get(self) -> tuple:
        """Get (result, age_s, frame_id). Returns (None, inf, 0) if empty."""
        with self._lock:
            if self._result is None:
                return (None, float("inf"), 0)
            age = time.time() - self._timestamp
            return (dict(self._result), age, self._frame_id)

    def get_or_stale(self, max_age_s: float = 2.0) -> dict:
        """Get latest result, or a safe stale-indicator dict if too old."""
        result, age, fid = self.get()
        if result is None or age > max_age_s:
            return {"error": "stale_frame", "age_s": round(age, 2),
                    "frame_id": fid, "faces": [], "av": {}, "mode": "stale"}
        return result

_frame_cache = _FrameResultCache()

# ── Background DeepFace Cache (never blocks frames) ──
import threading
from collections import OrderedDict, deque

# v9.7.1 — DeepFace circuit breaker.
# When TF / CUDA / model weights crash, we'd previously keep hammering
# the failing path at min_interval cadence forever, filling logs with
# the same exception and burning CPU on a sure failure. This breaker
# tracks recent failures in a sliding window:
#
#   * CLOSED:   submissions allowed (steady state)
#   * OPEN:     submissions skipped; a single ``probe`` is retried on a
#               cooldown so we automatically resume if the underlying
#               failure clears (e.g., GPU memory frees up)
#
# Tuning is intentionally conservative — DeepFace failures should be
# rare; if they're not, the user wants to know about it via the
# ``[deepface][circuit] OPEN`` log line.
_DF_BREAKER_LOCK = threading.Lock()
_DF_BREAKER_FAILS: "deque[float]" = deque(maxlen=64)
_DF_BREAKER_WINDOW_S = 30.0
_DF_BREAKER_THRESHOLD = 6
_DF_BREAKER_COOLDOWN_S = 30.0
_DF_BREAKER_OPEN_UNTIL = 0.0
_DF_BREAKER_PROBE_PENDING = False


def _df_breaker_record_outcome(outcome: str) -> None:
    """Update the breaker on every DF run. Outcome ∈ {success, miss, error}."""
    global _DF_BREAKER_OPEN_UNTIL, _DF_BREAKER_PROBE_PENDING
    now = time.time()
    with _DF_BREAKER_LOCK:
        if outcome == "error":
            _DF_BREAKER_FAILS.append(now)
            # Trim ancient entries lazily
            while _DF_BREAKER_FAILS and (now - _DF_BREAKER_FAILS[0]) > _DF_BREAKER_WINDOW_S:
                _DF_BREAKER_FAILS.popleft()
            if len(_DF_BREAKER_FAILS) >= _DF_BREAKER_THRESHOLD and now >= _DF_BREAKER_OPEN_UNTIL:
                _DF_BREAKER_OPEN_UNTIL = now + _DF_BREAKER_COOLDOWN_S
                print(
                    f"[deepface][circuit] OPEN — {len(_DF_BREAKER_FAILS)} failures "
                    f"in {_DF_BREAKER_WINDOW_S:.0f} s; pausing submissions for "
                    f"{_DF_BREAKER_COOLDOWN_S:.0f} s"
                )
                if _log_df is not None:
                    try:
                        _log_df.warn("circuit breaker open",
                                     fails=len(_DF_BREAKER_FAILS),
                                     window_s=_DF_BREAKER_WINDOW_S,
                                     cooldown_s=_DF_BREAKER_COOLDOWN_S)
                    except Exception:
                        pass  # nolint:silent-except telemetry best-effort
        else:
            # Successful run resets the failure window so we don't carry
            # stale failures into the next episode.
            if outcome == "success":
                _DF_BREAKER_FAILS.clear()
                if _DF_BREAKER_PROBE_PENDING:
                    print("[deepface][circuit] probe SUCCESS — closing circuit")
                _DF_BREAKER_OPEN_UNTIL = 0.0
                _DF_BREAKER_PROBE_PENDING = False


def _df_breaker_should_skip() -> bool:
    """Return True iff the breaker is open and no probe is due yet.

    Always lets ONE probe through past the cooldown so the system can
    self-heal without manual restart.
    """
    global _DF_BREAKER_PROBE_PENDING
    now = time.time()
    with _DF_BREAKER_LOCK:
        if now >= _DF_BREAKER_OPEN_UNTIL:
            return False
        if _DF_BREAKER_PROBE_PENDING:
            return True
        # Let exactly one probe through.
        _DF_BREAKER_PROBE_PENDING = True
        print("[deepface][circuit] probe attempt during cooldown")
        return False


def df_breaker_state() -> dict:
    """Snapshot for /health/full and /metrics."""
    now = time.time()
    with _DF_BREAKER_LOCK:
        recent = sum(
            1 for t in _DF_BREAKER_FAILS if (now - t) < _DF_BREAKER_WINDOW_S
        )
        return {
            "open": now < _DF_BREAKER_OPEN_UNTIL,
            "open_seconds_remaining": max(0.0, _DF_BREAKER_OPEN_UNTIL - now),
            "failures_in_window": recent,
            "threshold": _DF_BREAKER_THRESHOLD,
            "window_s": _DF_BREAKER_WINDOW_S,
            "probe_pending": _DF_BREAKER_PROBE_PENDING,
        }


# v9.7.1: cap _df_cache so face_idx churn (new tracker IDs in long
# sessions) cannot leak unbounded RAM. Each entry is ~1 MB (full df
# dict + fused emotion + av snapshot); without a cap, a 24-h session
# with frequent tracker re-IDs grows to multi-GB. OrderedDict gives us
# O(1) LRU touch via move_to_end. Cap is intentionally larger than
# MAX_TRACKED_FACES so brief identity drift doesn't evict useful state.
_DF_CACHE_MAX_ENTRIES = max(8, int(getattr(cfg, "MAX_TRACKED_FACES", 5)) * 2)
_df_cache: "OrderedDict[int, dict]" = OrderedDict()
_df_lock = threading.Lock()
_df_pending = set()
# v8.4: bbox-based skip — if face hasn't moved >10px, reuse cached result
_df_bbox_cache = {}  # face_idx -> (x, y, w, h) rounded to 10px grid
# v9.2 (plan step 10): track when DeepFace demographics last ran per face
# so we can rate-limit the slow age/gender/race path separately from the
# fast emotion path. Emotion now runs every fast frame via HSEmotion;
# demographics only need to refresh when a new face appears or a person
# shifts dramatically.
_df_demographics_last_run = {}  # face_idx -> wall-clock seconds


def _df_cache_set(face_idx: int, entry: dict) -> None:
    """LRU-bounded write into _df_cache. Caller holds _df_lock."""
    if face_idx in _df_cache:
        _df_cache.move_to_end(face_idx)
    _df_cache[face_idx] = entry
    while len(_df_cache) > _DF_CACHE_MAX_ENTRIES:
        evicted_idx, _ = _df_cache.popitem(last=False)
        # Drop sibling state so we don't leak the bbox / last-run ledger
        # for an evicted face.
        _df_bbox_cache.pop(evicted_idx, None)
        _df_demographics_last_run.pop(evicted_idx, None)


def _df_cache_get(face_idx: int) -> "dict | None":
    """Read with LRU touch. Caller holds _df_lock."""
    entry = _df_cache.get(face_idx)
    if entry is not None:
        _df_cache.move_to_end(face_idx)
    return entry

def _bbox_grid(fl, h, w):
    """Compute face bbox snapped to 10px grid for cache key."""
    xs = [lm.x * w for lm in fl]; ys = [lm.y * h for lm in fl]
    bx = int(min(xs)) // 10 * 10; by = int(min(ys)) // 10 * 10
    bw = (int(max(xs)) - int(min(xs))) // 10 * 10
    bh = (int(max(ys)) - int(min(ys))) // 10 * 10
    return (bx, by, bw, bh)

def _bbox_moved(idx, new_bbox, thresh_px=None):
    """Return True if face moved more than `thresh_px` (default 10) from
    cached bbox. v9.2 step 11: accepts custom threshold so DeepFace
    demographics can use a wider band than rPPG ROI refresh."""
    old = _df_bbox_cache.get(idx)
    if old is None:
        return True
    thr = 10 if thresh_px is None else thresh_px
    return any(abs(a - b) > thr for a, b in zip(old, new_bbox))

def _should_rerun_df_demographics(idx, new_bbox):
    """Return True iff we should submit DeepFace for this face.

    Rules:
      1. Never run if bbox shifted less than DF_BBOX_STABLE_PX AND a fresh
         result already exists within DF_DEMOGRAPHICS_MIN_INTERVAL_S.
      2. Otherwise always rerun — new face, significant movement, or a
         stale cache triggers a refresh.
    """
    now = time.time()
    last = _df_demographics_last_run.get(idx, 0.0)
    min_interval = float(getattr(cfg, "DF_DEMOGRAPHICS_MIN_INTERVAL_S", 2.0))
    stable_px = int(getattr(cfg, "DF_BBOX_STABLE_PX", 20))
    bbox_stable = not _bbox_moved(idx, new_bbox, thresh_px=stable_px)
    cached_fresh = (now - last) < min_interval
    return not (bbox_stable and cached_fresh)

def _bg_deepface(face_idx, crop, bs_emotion):
    """Background DeepFace processing. Updates cache when done."""
    _df_t0 = time.time()
    _df_outcome = "miss"  # success | miss | error
    try:
        df = detectors.run_deepface(crop)
        if df:
            _df_outcome = "success"
            sa = analysis.smooth_age(face_idx, df["raw_age"])
            df["age"] = {"value": sa, "range": f"{max(0, sa - 4)}-{sa + 4}"}
            # Smooth race/ethnicity with voting across frames
            raw_race_scores = df.get("race", {}).get("scores", {})
            if raw_race_scores:
                smoothed_race = analysis.smooth_race(face_idx, raw_race_scores)
                df["race"] = smoothed_race
            fused = analysis.fuse_emotions(df["emotion"]["scores"], analysis.blendshape_emotion(bs_emotion) if bs_emotion else {})
            df["emotion"]["fused_scores"] = fused
            dom, _dc = analysis.stable_dominant_emotion(fused)
            df["emotion"]["dominant"] = dom
            recog = face_registry.recognize(crop)
            if recog["name"]:
                df["identity"] = recog["name"]
            v_s, a_s, w_s = 0, 0, 0
            for e, s in fused.items():
                ev, ea = cfg.EMOTION_VA.get(e, (0, 0))
                v_s += ev * s; a_s += ea * s; w_s += s
            fv = round(v_s / w_s, 3) if w_s > 0 else 0
            fa = round(a_s / w_s, 3) if w_s > 0 else 0
            av_data = {"valence": fv, "arousal": fa, "dominant": dom,
                "age": sa, "age_range": f"{max(0, sa - 4)}-{sa + 4}",
                "gender": df.get("gender", {}).get("dominant"),
                "race": df.get("race", {}).get("dominant"),
                "identity": df.get("identity")}
            with _df_lock:
                _df_cache_set(face_idx, {"df": df, "fused": fused, "av": av_data, "t": time.time()})
    except Exception as e:
        # 2026-04-25 (v9.7.1): surface failures so we can diagnose empty
        # demographic tiles. The bbox-cache + min-interval gating in
        # _should_rerun_df_demographics keeps log volume bounded.
        _df_outcome = "error"
        try:
            print(f"[deepface][face={face_idx}] background failed: {type(e).__name__}: {e}")
            if _log_df is not None:
                _log_df.exc("background failed", e, face_idx=int(face_idx))
        except Exception:
            pass  # nolint:silent-except logger-of-last-resort
    finally:
        _df_pending.discard(face_idx)
        # v9.7.1: feed the circuit breaker before the metric write so an
        # error episode trips the breaker even if metrics fail.
        try:
            _df_breaker_record_outcome(_df_outcome)
        except Exception:
            pass  # nolint:silent-except breaker telemetry best-effort
        # v9.7.1: counters for /metrics. Outcome breakdown lets us compute
        # DF success-rate over time and alert on a sudden 0% (typical
        # symptom of a TF/CUDA crash).
        if _REG is not None:
            try:
                _REG.counter("deepface_total",
                             "DeepFace background runs by outcome").inc(
                                 1.0, labels={"outcome": _df_outcome})
                _REG.histogram("deepface_latency_ms",
                               "DeepFace background latency").observe(
                                   (time.time() - _df_t0) * 1000.0)
            except Exception:
                pass  # nolint:silent-except metrics best-effort

def _expire_cache():
    """Remove stale entries from DeepFace cache."""
    now = time.time()
    with _df_lock:
        stale = [k for k, v in _df_cache.items() if now - v["t"] > cfg.DF_CACHE_MAX_AGE]
        for k in stale:
            _df_cache.pop(k, None)
            _df_bbox_cache.pop(k, None)
            _df_demographics_last_run.pop(k, None)


# ──────────────────────────────────────────────
# v8.7 (Step 7): face identity attachment + state rehydration
# ──────────────────────────────────────────────
_REID_FRAMES_PER_TRACK = max(1, int(getattr(cfg, "FACE_RECOG_EVERY_N", 4)))
_reid_thread_pool = None  # lazily initialized so we don't block import


def _crop_for_track(image, landmarks, h, w, pad_frac=0.2):
    """Tight face crop sized for Facenet512. Returns BGR or None."""
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    fw, fh = max(xs) - min(xs), max(ys) - min(ys)
    px, py = int(fw * pad_frac), int(fh * pad_frac)
    x1, y1 = max(0, int(min(xs)) - px), max(0, int(min(ys)) - py)
    x2, y2 = min(w, int(max(xs)) + px), min(h, int(max(ys)) + py)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop if crop is not None and crop.size > 0 else None


def _attach_identity(track, image, landmarks, h, w):
    """Run face_identity on this track's crop and attach the result to the
    Track object via tracker.identify_and_attach (which throttles per track).
    Safe no-op if face_identity isn't loaded."""
    if face_identity is None:
        return
    if track is None or track.identity and track.reid_cooldown > 0:
        return
    crop = _crop_for_track(image, landmarks, h, w)
    if crop is None:
        return
    try:
        face_tracker.identify_and_attach(
            track, crop, face_identity.identify, every_n=_REID_FRAMES_PER_TRACK,
        )
    except Exception:
        pass


def _on_track_pruned(track):
    """Tracker prune callback — park the PersonState by identity for re-entry."""
    try:
        ps = person_states.pop(track.track_id, None)
        if ps is None:
            return
        try:
            ps.summarize()
        except Exception:
            pass
        if face_identity is not None and getattr(ps, "identity", None):
            face_identity.cache_state(ps.identity, ps)
        # v9.5.x — persist baseline across sessions
        if _calib_store is not None and getattr(ps, "identity", None):
            try:
                baseline = _calib_store.extract_from_person_state(ps)
                if baseline:
                    _calib_store.save(ps.identity, baseline)
            except Exception:
                pass
        # v8.7 (Step 9): forensic log
        try:
            forensic_log.log_event("person_pruned",
                person_id=track.track_id,
                identity=getattr(ps, "identity", None),
                payload={"frames_seen": getattr(track, "frames_seen", 0),
                         "frames_missed": getattr(track, "frames_missed", 0)})
        except Exception:
            pass
    except Exception:
        pass


# Wire the prune callback once. The orchestrator's per-iteration loop
# already does its own caching at lines ~449/681, but the callback is the
# safety net for asynchronous prune events triggered by tracker.update().
try:
    face_tracker.prune_callback = _on_track_pruned
except Exception:
    pass

def _merge_fast_bio():
    """Override 30fps biometrics with 120fps data when dual camera is active.
    The 120fps camera produces more precise blink/pupil/micro data."""
    fb = dual_camera.get_fast_biometrics()
    if not fb.get("active"):
        return
    av = analysis.affect_vector
    # 120fps blink/PERCLOS is 4x more precise
    if fb.get("blink_rate") is not None:
        av["blink_rate"] = fb["blink_rate"]
    if fb.get("perclos") is not None:
        av["perclos"] = fb["perclos"]
    if fb.get("drowsiness"):
        av["drowsiness"] = fb["drowsiness"]
    # v9.8.1 (2026-05-12): override `blinked` from the high-fps stream
    # when active. The HD USB camera at 60+ fps catches blink onsets the
    # 30 fps browser pipeline misses, so this is the authoritative
    # source for the edge-triggered boolean that feeds
    # dynamics_branch.feed_blink() in PAD fusion.
    if "blinked" in fb:
        av["blinked"] = bool(fb.get("blinked"))
    # 120fps pupil dilation is smoother
    if fb.get("pupil_dilation") is not None:
        av["pupil_dilation"] = fb["pupil_dilation"]
    # 120fps micro-expressions catch 40ms flickers 30fps misses
    if fb.get("micro_detected"):
        av["micro_expression"] = {
            "detected": True, "muscle": fb["micro_muscle"],
            "intensity": fb["micro_intensity"], "source": "120fps"
        }
    # Deception-relevant fast metrics (for truth engine)
    av["duchenne_smile"] = fb.get("duchenne", False)
    av["fake_smile"] = fb.get("fake_smile", False)
    av["facial_asymmetry"] = fb.get("asymmetry", 0)
    av["fear_au_score"] = fb.get("fear_score", 0)
    av["deception_au_score"] = fb.get("deception_au_score", 0)


# Lazy import: pad loads ONNX branches; avoid failing boot if pad breaks.
_nose_ear_pixels_fn = None  # callable | False


def _nose_ear_landmark_pixels(fl, frame_w: int, frame_h: int):
    global _nose_ear_pixels_fn
    if _nose_ear_pixels_fn is False:
        return None
    if _nose_ear_pixels_fn is None:
        try:
            from pad.landmark_topology import nose_ear_frame_pixels as _fn
            _nose_ear_pixels_fn = _fn
        except Exception:
            _nose_ear_pixels_fn = False
            return None
    return _nose_ear_pixels_fn(fl, int(frame_w), int(frame_h))


def _face_affect_snapshot_from_dyn(fi: int, dyn_feed) -> dict:
    """JSON-safe coarse per-face metrics for WS / PAD audit (v9.8.3)."""
    import math as _m
    out = {"face_idx": int(fi)}
    if not dyn_feed or not isinstance(dyn_feed, dict):
        return out
    hp = dyn_feed.get("head_pose")
    if isinstance(hp, dict):
        for k in ("yaw", "pitch", "roll"):
            v = hp.get(k)
            if v is not None and isinstance(v, (int, float)):
                f = float(v)
                if _m.isfinite(f):
                    out[f"pose_{k}"] = round(f, 4)
    gz = dyn_feed.get("gaze")
    if isinstance(gz, dict):
        for k in ("horizontal", "vertical"):
            v = gz.get(k)
            if v is not None and isinstance(v, (int, float)):
                f = float(v)
                if _m.isfinite(f):
                    out[f"gaze_{k}"] = round(f, 4)
    out["blinked"] = bool(dyn_feed.get("blinked"))
    pd = dyn_feed.get("pupil_dilation")
    if pd is not None:
        try:
            f = float(pd)
            if _m.isfinite(f):
                out["pupil_dilation"] = round(f, 5)
        except (TypeError, ValueError):
            pass
    em = dyn_feed.get("ear_mean")
    if em is not None:
        try:
            f = float(em)
            if _m.isfinite(f):
                out["ear_mean"] = round(f, 5)
        except (TypeError, ValueError):
            pass
    nx = dyn_feed.get("nose_xy")
    if isinstance(nx, (list, tuple)) and len(nx) == 2:
        try:
            out["nose_xy"] = [round(float(nx[0]), 2), round(float(nx[1]), 2)]
        except (TypeError, ValueError):
            pass
    return out


def _pad_dynamics_feed_for_face(fi, faces, frame_shape, face_landmarks):
    """Build per-track inputs for PAD dynamics (v9.8.2).

    Pose/gaze/blink/pupil must match the face_idx being scored — not the
    global affect_vector (which only mirrors faces[0]). For fi==0,
    blinked/pupil_dilation still follow analysis.affect_vector after
    _merge_fast_bio() so the HD USB dual-camera path stays authoritative.
    Nose/ear pixel pairs (via `pad.landmark_topology.nose_ear_frame_pixels`)
    and per-frame EAR (`ear_mean` via analysis._ear_from_landmarks) feed
    dynamics_branch.motion_ratio_history and ear_history.
    """
    if not faces or fi < 0 or fi >= len(faces):
        return None
    face = faces[fi]
    if not isinstance(face, dict):
        return None
    h, w = int(frame_shape[0]), int(frame_shape[1])
    av = analysis.affect_vector
    pose = face.get("pose")
    gaze = face.get("gaze")
    blink = face.get("blink") or {}
    iris = face.get("iris") or {}
    pupil = iris.get("avg_dilation")
    blinked = bool(blink.get("blinked"))
    if fi == 0:
        blinked = bool(av.get("blinked"))
        if av.get("pupil_dilation") is not None:
            pupil = av.get("pupil_dilation")
    nose_xy = left_ear_xy = right_ear_xy = None
    ear_mean = None
    if face_landmarks is not None and fi < len(face_landmarks):
        try:
            fl = face_landmarks[fi]
            ear_mean = analysis._ear_from_landmarks(fl)
            trio = _nose_ear_landmark_pixels(fl, w, h)
            if trio is not None:
                nose_xy, left_ear_xy, right_ear_xy = trio
        except Exception:
            pass
    return {
        "head_pose": pose if isinstance(pose, dict) else None,
        "gaze": gaze if isinstance(gaze, dict) else None,
        "pupil_dilation": pupil,
        "blinked": blinked,
        "ear_mean": ear_mean,
        "nose_xy": nose_xy,
        "left_ear_xy": left_ear_xy,
        "right_ear_xy": right_ear_xy,
    }


# v9.5.x — lazy-import side modules so they don't crash the boot if any
# one of them has a problem on an older Python. Each module is independent.
try:
    import auth as _auth
except Exception as _e:
    _auth = None
    print(f"[--] auth module unavailable ({_e})")
try:
    import metrics as _metrics
    _REG = _metrics.REGISTRY
except Exception as _e:
    _metrics = None
    _REG = None
    print(f"[--] metrics module unavailable ({_e})")
try:
    import session_replay as _replay
except Exception as _e:
    _replay = None
    print(f"[--] session_replay unavailable ({_e})")
try:
    import calibration_store as _calib_store
except Exception as _e:
    _calib_store = None
    print(f"[--] calibration_store unavailable ({_e})")

# v9.5.x — dedicated thread pool for the per-frame pipeline. Keeps the
# asyncio event loop unblocked during MediaPipe / ONNX / DeepFace calls.
try:
    from pad.thread_pool import get_pool as _get_onnx_pool
    _onnx_pool = _get_onnx_pool()
    print(f"[OK] ONNX thread pool ready ({_onnx_pool._max_workers} workers)")
except Exception as _e:
    _onnx_pool = None
    print(f"[--] ONNX thread pool unavailable ({_e})")

import asyncio  # needed by the WS handler

# v9.5.x — upgraded crash diagnostics: heartbeat dumps + signal handlers
# + /ops/dump_stacks on-demand route. Installed after `app` is defined;
# see the call further down.
try:
    import crash_diag as _crash_diag
except Exception as _e:
    _crash_diag = None
    print(f"[--] crash_diag unavailable ({_e})")

# Active-WS counter with a module-level lock so the gauge stays honest
_ws_active_count = 0
_ws_active_lock = __import__("threading").Lock()

def _ws_active_delta(delta: int) -> None:
    global _ws_active_count
    with _ws_active_lock:
        _ws_active_count = max(0, _ws_active_count + delta)
        if _REG:
            _REG.gauge("ws_active_connections",
                       "WS active connections").set(_ws_active_count)

# v9.7.1: identity stamp computed at module import so /health/full and the
# WS envelope work even when uvicorn.run() bypasses the __main__ block
# (e.g., gunicorn workers, pytest fixtures).
try:
    import server_identity as _server_identity
    _server_identity.init()
except Exception as _id_err:
    print(f"[server_identity] init failed at import: {type(_id_err).__name__}: {_id_err}")
    _server_identity = None  # type: ignore[assignment]

# v9.7.5: stamp the LLM-logger session id so every QLoRA training
# pair logged this run shares an id, allowing forensic_export's
# per-session filter to actually match. Format: ISO-like compact
# UTC stamp + a short content hash for uniqueness across rapid
# restarts. Falls back to plain timestamp if server_identity isn't
# available.
try:
    import llm_logger as _llm_logger
    import time as _time_for_sid
    _sid_stamp = _time_for_sid.strftime("%Y%m%d_%H%M%S",
                                         _time_for_sid.gmtime())
    _sid_hash = ""
    if _server_identity is not None:
        try:
            _sid_hash = (_server_identity.summary() or {}).get("content_hash", "")[:8]
        except Exception:
            _sid_hash = ""
    _session_id = f"{_sid_stamp}{('_' + _sid_hash) if _sid_hash else ''}"
    _llm_logger.set_session_id(_session_id)
    print(f"[llm_logger] session id = {_session_id}")
except Exception as _sid_err:
    print(f"[llm_logger] session-id init failed: {type(_sid_err).__name__}: {_sid_err}")

# v9.7.1: structured-event log. Wraps print() in a typed ``tagged`` helper
# so subsystem messages can be grepped from logs/events_<date>.jsonl.
try:
    import event_log as _evl
    _log_main = _evl.tagged("main")
    _log_df = _evl.tagged("deepface")
except Exception as _evl_err:
    print(f"[event_log] init failed: {type(_evl_err).__name__}: {_evl_err}")
    _evl = None  # type: ignore[assignment]
    _log_main = _log_df = None  # type: ignore[assignment]

# v9.7.1: per-frame timing CSV writer. Loaded but no-op until record()
# is called from the WS frame loop.
try:
    import runtime_profiler as _runtime_profiler
except Exception as _rtp_err:
    print(f"[runtime_profiler] init failed: {type(_rtp_err).__name__}: {_rtp_err}")
    _runtime_profiler = None  # type: ignore[assignment]

# v9.7.3: silence the Windows-specific asyncio Proactor close-handshake
# spam. When a browser tab closes its WebSocket abruptly (refresh, OS
# sleep, dropped Wi-Fi), the server sees `WinError 10054 — An existing
# connection was forcibly closed by the remote host` thrown out of
# `_ProactorBasePipeTransport._call_connection_lost`. It's harmless —
# the connection is already gone, the OS is just complaining about the
# shutdown(SHUT_RDWR) on a half-dead socket — but it floods the log and
# obscures real errors. We drop it via a logging filter on the
# `asyncio` logger rather than monkey-patching the transport, so any
# *other* asyncio error path keeps logging normally.
try:
    import logging as _logging
    class _ProactorResetFilter(_logging.Filter):
        """Suppress benign Windows asyncio noise that drowns out real
        errors. Three patterns:
          1. WinError 10054 from `_ProactorBasePipeTransport
             ._call_connection_lost` — fires when a browser WS peer
             closes the socket abruptly. Harmless.
          2. KeyboardInterrupt from our `crash_diag._term_handler` ↔
             uvicorn's `capture_signals` re-raise — fires on every
             clean Ctrl-C shutdown of the server. Harmless, but uvicorn
             logs it at ERROR level which makes Ctrl-C look like a bug.
          3. starlette lifespan `CancelledError` — the cascading
             cleanup of (2). Same exit path, also harmless.
        Anything else asyncio logs (real bugs, scheduler issues, other
        connection errors) keeps logging normally."""
        def filter(self, record: _logging.LogRecord) -> bool:
            try:
                exc = record.exc_info[1] if record.exc_info else None
                # Pattern 1
                if isinstance(exc, ConnectionResetError):
                    if getattr(exc, "winerror", None) == 10054:
                        return False
                # Pattern 2 — clean Ctrl-C shutdown
                if isinstance(exc, KeyboardInterrupt):
                    return False
                # Pattern 3 — lifespan cancellation, only if from our
                # signal-handler path. Identify via the formatted
                # traceback containing _term_handler.
                import asyncio as _asy
                if isinstance(exc, _asy.CancelledError):
                    if record.exc_text and "_term_handler" in record.exc_text:
                        return False
                    if record.exc_text and "lifespan" in (record.exc_text or "") \
                       and "capture_signals" in (record.exc_text or ""):
                        return False
                msg = record.getMessage()
                if "WinError 10054" in msg and "_call_connection_lost" in msg:
                    return False
                # Catch the formatted shutdown traceback even when the
                # exc_info has been pre-rendered into the message body.
                if "_term_handler" in msg and "KeyboardInterrupt" in msg:
                    return False
            except Exception:
                pass
            return True
    # Apply to every logger uvicorn uses; the shutdown traceback comes
    # through the root or `uvicorn.error` logger depending on version.
    for _ln in ("asyncio", "uvicorn", "uvicorn.error", "uvicorn.asgi"):
        _logging.getLogger(_ln).addFilter(_ProactorResetFilter())
except Exception as _flt_err:
    print(f"[asyncio-filter] init failed (non-fatal): {type(_flt_err).__name__}: {_flt_err}")

app = FastAPI(
    title="Dodo Vision AI",
    description="Multimodal PAD + affect + HR pipeline. v9.8.x. "
                "Every data endpoint is protected when DODO_AUTH_TOKEN is set.",
    version="9.8.0",
    openapi_tags=[
        {"name": "health",  "description": "Liveness & readiness probes"},
        {"name": "pad",     "description": "Presentation Attack Detection"},
        {"name": "vitals",  "description": "Per-person affect/HR/radar"},
        {"name": "session", "description": "Session recording & replay"},
        {"name": "ops",     "description": "Operational (metrics, selftest)"},
        {"name": "calib",   "description": "Calibration persistence"},
    ],
)

# v9.8.5 — global HTTP Bearer when DODO_AUTH_TOKEN is set (matches WS gate).
from auth import browser_request_allowed
_HTTP_AUTH_PUBLIC_EXACT = frozenset({
    "/",
    "/health",
    "/ready",
    "/metrics",
    "/webcam",
    "/flags",
    "/version",
    "/favicon.ico",
})


@app.middleware("http")
async def _dodo_http_bearer_middleware(request: Request, call_next):
    if not browser_request_allowed(request):
        return JSONResponse(status_code=403, content={"detail": "untrusted host or browser origin"})
    if _auth is None or not _auth.enabled():
        return await call_next(request)
    if request.method == "OPTIONS":
        return await call_next(request)
    path = request.url.path
    if path in _HTTP_AUTH_PUBLIC_EXACT or path.startswith("/static/"):
        return await call_next(request)
    if path in ("/openapi.json", "/redoc", "/docs") or path.startswith("/docs/"):
        return await call_next(request)
    token = _auth.extract_from_http(request)
    if not _auth.check(token):
        if _REG:
            try:
                _REG.counter("http_auth_rejected_total",
                             "HTTP requests rejected (bearer)").inc()
            except Exception:
                pass
        return JSONResponse(
            status_code=401,
            content={
                "detail": "unauthorized",
                "hint": (
                    "Send Authorization: Bearer <token>, or open /webcam "
                    "from this server and enter your configured token."
                ),
            },
        )
    return await call_next(request)

import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"Global Exception caught: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": type(exc).__name__, "message": str(exc)}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "error": "HTTPException", "message": exc.detail}
    )

try:
    from fastapi.staticfiles import StaticFiles as _StaticFiles
    import os
    _static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "api", "static")
    os.makedirs(_static_dir, exist_ok=True)
    app.mount("/static", _StaticFiles(directory=_static_dir), name="static")
    print(f"[OK] /static mounted from {_static_dir}")
except Exception as _se:
    print(f"[--] /static mount failed: {_se}")

if globals().get("_crash_diag") is not None:
    try:
        _crash_diag.install(app)
    except Exception as _e:
        print(f"[--] crash_diag.install failed ({_e})")



# v9.1 (optimization step 10): gzip the big /detect/full JSON payload and
# any other response >1 KB. Minimum_size=1024 skips tiny status responses
# where compression would be pure overhead. The 5-face full payload compresses
# roughly 4× which matters on LAN→browser transport at 3–6 FPS.
#
# v9.2 (plan step 16): also add a whitelist middleware layer that skips
# gzip for binary msgpack. Msgpack content is already byte-compact;
# gzipping it again is wasted CPU for ~2% extra size reduction.
try:
    from fastapi.middleware.gzip import GZipMiddleware
    from starlette.types import ASGIApp, Receive, Scope, Send
    class _MsgpackBypassGzip:
        """Thin ASGI wrapper that marks msgpack responses as already-compressed
        so the GZipMiddleware below skips them. Works by setting the
        `Accept-Encoding` request header to identity inside the request
        scope when the route is known to emit msgpack.
        """
        def __init__(self, app: ASGIApp) -> None:
            self.app = app
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope.get("type") == "http":
                path = scope.get("path", "") or ""
                qs = scope.get("query_string", b"") or b""
                # Strip gzip negotiation for websocket-style msgpack endpoints.
                if b"format=msgpack" in qs or path.endswith(".msgpack"):
                    headers = [(k, v) for (k, v) in scope.get("headers", [])
                               if k.lower() != b"accept-encoding"]
                    headers.append((b"accept-encoding", b"identity"))
                    scope = dict(scope)
                    scope["headers"] = headers
            await self.app(scope, receive, send)
    # Order matters: bypass runs FIRST (outer), gzip runs SECOND (inner).
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(_MsgpackBypassGzip)
except Exception as _e:
    print(f"[WARN] gzip middleware unavailable: {_e}")

# v9.3 (plan step 7): Cross-Origin Isolation middleware for SharedArrayBuffer.
# COOP + COEP together opt the page into an isolated agent cluster so we can
# allocate SABs shared between the main thread and the WS worker without
# security errors. Gated behind FF_COOP_COEP because it also disables
# cross-origin iframe embedding, which may break screenshot tooling.
try:
    import feature_flags as _ff
    from starlette.middleware.base import BaseHTTPMiddleware as _BaseHTTPMW
    class _CrossOriginIsolationMW(_BaseHTTPMW):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            if _ff.get("COOP_COEP"):
                response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
                response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
                response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
            return response
    app.add_middleware(_CrossOriginIsolationMW)
except Exception as _ce:
    print(f"[WARN] cross-origin-isolation middleware not available: {_ce}")

# v8.4: affect delta tracking for /affect/delta endpoint
_prev_affect_snapshot = {}

# v8.4: auto-save timer
_last_auto_save = time.time()

# ── Adaptive brightness compensation ──
# v9.1 (optimization step 3): reuse CLAHE objects across frames instead of
# allocating a new one 20+ times a second. Three clip levels × one tile
# size = three cached objects, selected per brightness tier.
_CLAHE_TILE = (4, 4)
_CLAHE_CACHE = {
    1.5: cv2.createCLAHE(clipLimit=1.5, tileGridSize=_CLAHE_TILE),
    2.0: cv2.createCLAHE(clipLimit=2.0, tileGridSize=_CLAHE_TILE),
    2.5: cv2.createCLAHE(clipLimit=2.5, tileGridSize=_CLAHE_TILE),
}

# v9.2 (plan step 5): motion-gate at the frame level. If the incoming JPEG
# looks essentially identical to the previous frame (mean brightness
# change <0.5 and a small downsampled-diff test), reuse the last full
# result instead of re-running the whole pipeline. Typical real-world
# impact: a person sitting still saves 50-70% of server CPU.
_MOTION_GATE_ENABLED = True
_MOTION_GATE_DIFF_THRESH = 3.5          # mean pixel L-diff in [0,255]
_MOTION_GATE_BRIGHTNESS_THRESH = 1.5    # |dL| in [0,255]
_MOTION_GATE_MAX_REUSE_MS = 250         # always recompute at least 4×/s

_last_gate_thumb = None
_last_gate_brightness = None
_last_gate_result_at = 0.0

def _motion_gate_skip(img):
    """Return True if this frame is safe to treat as a repeat of the
    previous one and we should reuse the cached frame result."""
    global _last_gate_thumb, _last_gate_brightness
    if not _MOTION_GATE_ENABLED:
        return False
    now_ms = time.time() * 1000.0
    if (now_ms - _last_gate_result_at) > _MOTION_GATE_MAX_REUSE_MS:
        return False
    # Cheap 32×32 mean-brightness thumbnail + per-pixel diff
    small = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(small, (32, 24), interpolation=cv2.INTER_AREA)
    b = float(np.mean(small))
    if _last_gate_thumb is None:
        _last_gate_thumb = small
        _last_gate_brightness = b
        return False
    db = abs(b - _last_gate_brightness)
    diff = float(np.mean(cv2.absdiff(small, _last_gate_thumb)))
    # Always update baseline to the latest observation
    _last_gate_thumb = small
    _last_gate_brightness = b
    return (db < _MOTION_GATE_BRIGHTNESS_THRESH
            and diff < _MOTION_GATE_DIFF_THRESH)

def _motion_gate_record_now():
    global _last_gate_result_at
    _last_gate_result_at = time.time() * 1000.0

def _extract_face_crop(bgr_img, face_landmarks, pad=0.15):
    """Return a tight BGR crop around the face landmarks, or None if the
    computed bbox is degenerate. Used by HSEmotion on every fast frame."""
    h, w = bgr_img.shape[:2]
    xs = [lm.x * w for lm in face_landmarks]
    ys = [lm.y * h for lm in face_landmarks]
    x0 = max(0, int(min(xs) - (max(xs) - min(xs)) * pad))
    x1 = min(w, int(max(xs) + (max(xs) - min(xs)) * pad))
    y0 = max(0, int(min(ys) - (max(ys) - min(ys)) * pad))
    y1 = min(h, int(max(ys) + (max(ys) - min(ys)) * pad))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    return bgr_img[y0:y1, x0:x1]

# ═══════════════════════════════════════════════════════════════════
# v9.3.1 — Client-landmark adapter (plan step 4 server consumer)
# ═══════════════════════════════════════════════════════════════════
#
# When the client sends a mode=2 packet with its own MediaPipe landmarks,
# we skip the server's MediaPipe and feed those landmarks into all the
# downstream consumers (rPPG ROI, FACS, blendshape emotion, deception).
# Everything downstream already accepts a landmark list; the only thing
# we need is a lightweight shim that quacks like MediaPipe's result
# object (`face_landmarks` list of landmark objects with `.x/.y/.z`,
# plus an optional `face_blendshapes` list).

class _ClientLandmark:
    """MediaPipe-compatible landmark shim. Accepts x/y/z floats."""
    __slots__ = ("x", "y", "z")
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

class _ClientBlendshape:
    """MediaPipe-compatible blendshape shim."""
    __slots__ = ("category_name", "score")
    def __init__(self, name: str, score: float):
        self.category_name = name
        self.score = float(score)

class _ClientFaceLandmarkerResult:
    """MediaPipe-compatible result shim — exposes face_landmarks and
    face_blendshapes in the exact shape FaceLandmarker.detect_for_video()
    returns so the existing downstream loops work unchanged.
    """
    __slots__ = ("face_landmarks", "face_blendshapes",
                 "facial_transformation_matrixes")
    def __init__(self, faces, blends):
        self.face_landmarks = faces
        self.face_blendshapes = blends
        self.facial_transformation_matrixes = []

def _result_from_client_lmk(lmk_array, blendshapes=None):
    """Build a MediaPipe-shaped result from a (N, 3) numpy array of
    landmarks and an optional {name: score} blendshape dict."""
    if lmk_array is None or len(lmk_array) == 0:
        return _ClientFaceLandmarkerResult([], [])
    face = [_ClientLandmark(x, y, z) for (x, y, z) in lmk_array]
    faces = [face]
    blends = []
    if blendshapes:
        blends = [[_ClientBlendshape(name, score)
                   for name, score in blendshapes.items()]]
    else:
        blends = [[]]
    return _ClientFaceLandmarkerResult(faces, blends)

def _process_frame_fast_with_lmk(image, client_result):
    """v9.3 step 4 — fast-path frame processor that skips MediaPipe and
    uses the client-provided landmark result. All downstream stages
    (analysis.build_face_fast, rppg_mod.add_sample, HSEmotion crop,
    emotion fusion, affect_vector update) read from `client_result`
    exactly as they would from a server MediaPipe result.

    Implemented as a thin wrapper that monkey-patches detectors.detect_faces
    for this one call, so we don't have to duplicate the ~200-line body
    of _process_frame_fast. The patch is undone in a finally block."""
    _orig_detect = detectors.detect_faces
    detectors.detect_faces = lambda _rgb: client_result
    try:
        return _process_frame_fast(image)
    finally:
        detectors.detect_faces = _orig_detect

def _adaptive_brightness(img):
    """CLAHE on L channel to protect rPPG under harsh/dim lighting.
    Clamps overexposed frames to prevent signal clipping."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    mean_l = float(np.mean(l_ch))
    clip = 1.5 if (mean_l > 200 or mean_l < 40) else (2.0 if (mean_l > 170 or mean_l < 60) else 2.5)
    clahe = _CLAHE_CACHE[clip]
    l_ch = clahe.apply(l_ch)
    if mean_l > 200:
        l_ch = np.clip(l_ch * (160 / (mean_l + 1e-6)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

# ══════════════════════════════════════════════
#  Shared processing logic
# ══════════════════════════════════════════════
def _process_frame_fast(image):
    """Process frame with MediaPipe only (no DeepFace).
    Fuses live blendshape emotion with cached DeepFace CNN scores every frame
    for smooth, responsive emotion tracking.

    v9.2 plan step 5: motion-gated. If the frame is essentially identical
    to the previous frame, return the cached result instead of re-running
    the whole pipeline. Capped at 250 ms max reuse so vitals still update
    4×/s even when someone is motionless."""
    # Motion gate BEFORE any expensive work. _motion_gate_skip reads only
    # a 32×24 grayscale thumbnail so the cost is negligible.
    if _motion_gate_skip(image):
        _cached_fast, _age, _ = _frame_cache.get()
        if _cached_fast is not None and _age < 0.25:
            # Copy so downstream mutations don't pollute the cache
            _result = dict(_cached_fast)
            _result["_motion_gated"] = True
            return _result
    rppg_mod.mark_start()
    t0 = time.time()
    image = _adaptive_brightness(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detectors.detect_faces(rgb)
    faces = []
    # v9.0 Step 1 (Stanford Monadic Option Types): Enforce strict pre-binding
    # of ALL sensor variables to typed None defaults at the TOP of lexical
    # scope, BEFORE any await/I/O/branching.  This guarantees the Python symbol
    # table is fully populated so a 1-second sensor timeout broadcasts a typed
    # None payload instead of triggering a fatal UnboundLocalError crash.
    # ── Face analysis outputs ──
    pose_out: "dict | None" = None
    gaze_out: "dict | None" = None
    blink_out: "dict | None" = None
    micro_out: "dict | None" = None
    bs_emo: "dict | None" = None
    iris_out: "dict | None" = None
    fused_emo: "dict | None" = None
    # ── Physiological pipeline outputs ──
    hr: "float | None" = None
    wave: "list | None" = None
    q: "float | None" = None
    resp: "float | None" = None
    hrv: "float | None" = None
    stress: "float | None" = None
    # ── Sensor data (from decoupled state buffer — never blocks) ──
    radar_data: "dict | None" = None
    radar_b: "dict | None" = None
    oxi_data: "dict | None" = None
    oxi_raw: "dict | None" = None
    gsr_data: "dict | None" = None
    # ── Dempster-Shafer fusion intermediates ──
    _ds_best_val: "float | None" = None
    _ds_best_conf: float = 0.0
    _ds_src_idx: int = -1
    _fused_hr: "float | None" = None
    if result.face_landmarks:
        # v9.2 (plan step 6): per-face analysis in parallel. build_face_fast
        # does pose + gaze + blink + micro + blendshape emotion + iris per
        # face — each call is CPU-bound and completely independent from
        # the others. With 3+ people that's 3-5× real wall-clock speedup.
        # Single-face skips the pool to avoid executor overhead.
        _fl_list = list(enumerate(result.face_landmarks))
        _bs_list = []
        for fi, fl in _fl_list:
            bs_dict = {}
            if result.face_blendshapes and fi < len(result.face_blendshapes):
                bs_dict = {b.category_name: round(b.score, 3) for b in result.face_blendshapes[fi]}
            _bs_list.append(bs_dict)
        if len(_fl_list) == 1:
            # Serial fast path — skip ThreadPoolExecutor overhead for 1 face
            fi, fl = _fl_list[0]
            faces.append(analysis.build_face_fast(fi, fl, _bs_list[0]))
        else:
            # Parallel fan-out for multi-face scenes
            _futures = [
                detectors.pool.submit(analysis.build_face_fast, fi, fl, bs)
                for (fi, fl), bs in zip(_fl_list, _bs_list)
            ]
            faces.extend(_f.result() for _f in _futures)
        if faces:
            face = faces[0]
            pose_out = face["pose"]; gaze_out = face["gaze"]
            blink_out = face["blink"]; micro_out = face["micro"]
            bs_emo = face["bs_emotion"]; iris_out = face.get("iris")

        # v9.2 (plan step 9): run HSEmotion on every fast frame for ALL
        # detected faces (was: face 0 only — v9.7.2 fix). DirectML-
        # accelerated ~8 ms per face on the 3060; budget ~16 ms at 2
        # faces, ~24 ms at 3, so the cap below keeps us inside the 33 ms
        # fast-frame envelope at 30 fps. Combined with live blendshapes
        # this gives real-time emotion updates at full fast-frame cadence.
        hsemo_scores_list = [None] * len(faces)
        if getattr(cfg, "HSEMOTION_EVERY_FAST_FRAME", True) and detectors.HSEMOTION_ENABLED:
            _max_faces = min(len(faces), getattr(cfg, "HSEMOTION_MAX_FACES", 3))
            for _fi in range(_max_faces):
                try:
                    _crop = _extract_face_crop(image, result.face_landmarks[_fi])
                    if _crop is not None:
                        # v9.10: pass face index as pid so the per-face EMA
                        # smoothing in detectors.run_hsemotion keeps state
                        # separate per detected face. _fi is 0..max_faces-1
                        # which is stable across frames as long as MediaPipe
                        # tracker doesn't reorder.
                        _hsres = detectors.run_hsemotion(_crop, pid=_fi)
                        if _hsres:
                            hsemo_scores_list[_fi] = _hsres.get("scores")
                except Exception:
                    pass

        # Per-frame emotion blending (per face) — priority order:
        #   1. Live HSEmotion (this frame) + live blendshapes  → true real-time
        #   2. Cached DeepFace (stale up to DF_CACHE_MAX_AGE) + live blendshapes
        #   3. Live blendshapes only
        for _fi, _face in enumerate(faces):
            _bs_emo = _face.get("bs_emotion")
            _hsemo_scores = hsemo_scores_list[_fi] if _fi < len(hsemo_scores_list) else None
            if _bs_emo and _hsemo_scores:
                _face["fused_emotion"] = analysis.fuse_emotions(_hsemo_scores, _bs_emo)
            elif _bs_emo:
                with _df_lock:
                    _cached = _df_cache_get(_fi)
                if _cached and time.time() - _cached["t"] < cfg.DF_CACHE_MAX_AGE:
                    _cached_scores = (_cached["df"].get("emotion", {}).get("scores")
                                      or _cached["df"].get("emotion", {}).get("fused_scores"))
                    if _cached_scores:
                        _face["fused_emotion"] = analysis.fuse_emotions(_cached_scores, _bs_emo)
        # Surface the primary face's emotion as `fused_emo` for downstream
        # consumers further in this function that consume a single value
        # (rPPG-gated affect, person-state update, etc.). Falls back to
        # the primary face's blendshape emotion when neither HSEmotion
        # nor cached DeepFace produced a fused score this frame.
        if faces:
            fused_emo = (faces[0].get("fused_emotion")
                         or faces[0].get("bs_emotion"))
        else:
            fused_emo = bs_emo if bs_emo else None

        # v9.8.1 (reverted gate): rppg.add_sample is thread-safe by design
        # and the module's docstring explicitly says it's "called from
        # both HQ camera (main thread) and 120fps camera (background
        # thread)". Feeding samples from BOTH UGREEN (HQ) and HD USB
        # (high-fps) into the same buffer is the intended dual-source
        # design -- two independent measurements of the same 1-2 Hz pulse
        # under different lighting improve SNR via ensemble averaging.
        # The motion gate inside add_sample drops bad samples; _estimate_fps
        # computes effective rate from rolling timestamps so mixed rates
        # are handled correctly downstream by the bandpass + FFT pipeline.
        rppg_mod.add_sample(image, result.face_landmarks[0])
    hr, wave, q, resp = rppg_mod.estimate_hr()
    hrv, stress = rppg_mod.estimate_hrv()
    # v8.9: read sensors from decoupled state buffer (never blocks on I/O)
    radar_data = _sensor_buf.get("radar_a")
    radar_b = _sensor_buf.get("radar_b")
    oxi_raw = _sensor_buf.get("oximeter")
    # v9.1.1: direct-read fallback. sensor_state's 5 s staleness cuts off
    # data if the 10 Hz poller skips a cycle (GIL contention, HID read
    # timeout). Fall back to oximeter.get_oximeter_data() which has its
    # own fresh-check — this keeps the UI responsive even when the poller
    # hiccups and avoids the "oximeter not working" symptom.
    if oxi_raw is None:
        try:
            oxi_raw = oximeter.get_oximeter_data()
        except Exception:
            oxi_raw = None
    # Apply radar motion gating to oximeter inline (replaces get_oximeter_data_fused)
    oxi_data = None
    if oxi_raw:
        oxi_data = dict(oxi_raw)
        if radar_data and radar_data.get("distance") is not None:
            _mv = radar_data.get("motion_variance", 0) or 0
            if _mv > 0.04:
                _att = max(0.1, 1.0 - min(1.0, _mv / 0.2))
                oxi_data["signal_strength"] = max(0, int((oxi_data.get("signal_strength") or 0) * _att))
                oxi_data["motion_gated"] = True
            else:
                oxi_data["motion_gated"] = False
    analysis.update_affect_vector(None, hr, hrv, stress, q, resp, pose_out, gaze_out, blink_out, micro_out, fused_emo, iris_out, radar_data)
    # Merge radar B data if dual radar connected
    if radar_b:
        analysis.affect_vector["radar_b_hr"] = radar_b.get("heart_rate")
        analysis.affect_vector["radar_b_breath"] = radar_b.get("breath_rate")
        analysis.affect_vector["radar_b_distance"] = radar_b.get("distance")
        analysis.affect_vector["radar_b_people_count"] = radar_b.get("people_count")  # v7.7
    # v7.8: dual-radar people aggregate — sum of A+B presence-derived counts,
    # clamped to [0, 5] to match the PersonState face-slot ceiling.
    _pa = analysis.affect_vector.get("radar_people_count") or 0
    _pb = analysis.affect_vector.get("radar_b_people_count") or 0
    analysis.affect_vector["radar_people_total"] = int(min(5, _pa + _pb))

    # v7.9: camera-based fallback for Dist / People when neither radar is
    # connected (or both are stale). Uses MediaPipe face landmarks: face
    # count → people_total, widest normalized bbox → distance proxy
    # (empirical 0.15 / width_frac for a standard ~60° FOV webcam,
    # clamped to 0.15–6.0 m). Tagged with provenance so the UI shows
    # "(cam)" next to the value.
    _radar_live = (radar_data is not None) or (radar_b is not None)
    # v9.5.x: the radar mmWave can't measure distances <20 cm (too close to
    # the antenna), which produced dist=None for the operator who sits
    # roughly 15-18 cm from the unit. Treat "radar present but no valid
    # distance right now" the same as "no radar" so the camera-based
    # fallback kicks in and the UI isn't blank.
    _radar_dist_live = (
        (radar_data is not None and radar_data.get("distance") is not None) or
        (radar_b is not None and radar_b.get("distance") is not None)
    )
    _use_cam_distance = (not _radar_dist_live) and result.face_landmarks
    if _use_cam_distance:
        try:
            _faces_n = len(result.face_landmarks)
            _widest_norm = 0.0
            for _fl in result.face_landmarks:
                _xs = [lm.x for lm in _fl]
                _bw = max(_xs) - min(_xs)
                if _bw > _widest_norm:
                    _widest_norm = _bw
            if _widest_norm > 0.01:
                _cam_dist = 0.15 / _widest_norm
                _cam_dist = max(0.15, min(6.0, _cam_dist))
                analysis.affect_vector["radar_distance"] = round(_cam_dist, 2)
                analysis.affect_vector["radar_distance_source"] = "camera"
            # Only override people_total when neither radar reports presence.
            if not _radar_live:
                analysis.affect_vector["radar_people_total"] = int(min(5, _faces_n))
                analysis.affect_vector["radar_people_source"] = "camera"
        except Exception:
            pass
    else:
        analysis.affect_vector["radar_distance_source"] = (
            "radar" if analysis.affect_vector.get("radar_distance") is not None else None
        )
        analysis.affect_vector["radar_people_source"] = "radar" if _radar_live else None

    # v8: Cost-minimized radar-to-person assignment (replaces old positional hack)
    if radar_assign is not None and result.face_landmarks and (radar_data or radar_b):
        try:
            h, w = image.shape[:2]
            face_bboxes = []
            for fl in result.face_landmarks:
                xs = [lm.x * w for lm in fl]; ys = [lm.y * h for lm in fl]
                face_bboxes.append((max(0, int(min(xs))), max(0, int(min(ys))),
                                   min(w, int(max(xs))), min(h, int(max(ys)))))
            radar_readings = {
                "A": radar_data if radar_data else None,
                "B": radar_b if radar_b else None,
            }
            mapping = radar_assign.assign(radar_readings, face_bboxes, (w, h))
            # Store full mapping so the person_states loop below can stamp
            # per-person radar readings from the correct sensor.
            analysis.affect_vector["_radar_mapping"] = mapping
        except Exception:
            pass
    _merge_fast_bio()  # override with 120fps data if available
    # v9.1 (optimization step 19): single source of truth for the oximeter's
    # view in affect_vector. Hardware state (connected/mode) always flows
    # through even when no finger is seated — that's what drives the UI
    # badge between Searching… / Connected / Stale.
    try:
        _oxi_link = oximeter.get_oximeter_status() or {}
    except Exception:
        _oxi_link = {}
    analysis.affect_vector["oxi_connected"] = bool(_oxi_link.get("connected"))
    analysis.affect_vector["oxi_mode"] = _oxi_link.get("mode")
    if oxi_data:
        analysis.affect_vector["oxi_hr"] = oxi_data.get("heart_rate")
        analysis.affect_vector["oxi_spo2"] = oxi_data.get("spo2")
        analysis.affect_vector["oxi_signal"] = oxi_data.get("signal_strength")
        analysis.affect_vector["oxi_pleth"] = oxi_data.get("pleth_waveform")
        analysis.affect_vector["oxi_motion_gated"] = oxi_data.get("motion_gated", False)
        _oxi_ts = oxi_data.get("last_update") or oxi_data.get("timestamp")
        if _oxi_ts:
            analysis.affect_vector["oxi_last_update"] = float(_oxi_ts)
            analysis.affect_vector["oxi_age_s"] = round(max(0.0, time.time() - float(_oxi_ts)), 2)
        else:
            analysis.affect_vector["oxi_last_update"] = time.time()
            analysis.affect_vector["oxi_age_s"] = 0.0
        # v9.7.6: pass BOTH radar A and B HR so calibration can train
        # per-radar corrections. radar_data is radar A (legacy single-
        # radar field name), radar_b_data is radar B.
        _radar_b_data = _sensor_buf.get("radar_b")
        calibration.add_sample(
            hr,
            oxi_data.get("heart_rate"),
            radar_a_hr=(radar_data.get("heart_rate") if radar_data else None),
            radar_b_hr=(_radar_b_data.get("heart_rate") if _radar_b_data else None),
        )
    else:
        analysis.affect_vector["oxi_hr"] = None
        analysis.affect_vector["oxi_spo2"] = None
        analysis.affect_vector["oxi_signal"] = None
        analysis.affect_vector["oxi_pleth"] = None
        # Keep last known timestamp so the UI can show actual age instead
        # of jumping to "--" every time a finger momentarily lifts.
        _prev_ts = analysis.affect_vector.get("oxi_last_update")
        if _prev_ts:
            analysis.affect_vector["oxi_age_s"] = round(max(0.0, time.time() - float(_prev_ts)), 2)
    if hr and calibration._calibration["samples_collected"] > 0:
        analysis.affect_vector["calibrated_hr"] = calibration.correct_hr(hr)
    analysis.affect_vector["confirmed_bpm"] = rppg_mod.get_confirmed_bpm()
    analysis.affect_vector["rppg_quality"] = q
    # v8.9: Dempster-Shafer HR source selection via confidence weights
    _oxi_hr, _oxi_conf = _sensor_buf.get_with_confidence("oximeter")
    _rad_hr_val = radar_data.get("heart_rate") if radar_data else None
    _, _rad_conf = _sensor_buf.get_with_confidence("radar_a")
    _rppg_conf = min(1.0, (q or 0) / 100.0) if q else 0.0
    _ds_best_val, _ds_best_conf, _ds_src_idx = sensor_state.ds_select_best(
        (_oxi_hr.get("heart_rate") if _oxi_hr else None, _oxi_conf),
        (_rad_hr_val, _rad_conf),
        (hr, _rppg_conf),
    )
    _ds_labels = ["OXI", "RADAR", "rPPG"]
    analysis.affect_vector["hr_source"] = _ds_labels[_ds_src_idx] if _ds_best_val else None
    analysis.affect_vector["hr_source_confidence"] = round(_ds_best_conf, 3)
    # Fused HR from all sources (Dempster-Shafer weighted average)
    _fused_hr = sensor_state.ds_combine_hr(
        (_oxi_hr.get("heart_rate") if _oxi_hr else None, _oxi_conf),
        (_rad_hr_val, _rad_conf),
        (hr, _rppg_conf),
    )
    if _fused_hr is not None:
        analysis.affect_vector["fused_hr_bpm"] = _fused_hr
    logger.log_affect(analysis.affect_vector)
    logger.push_lsl(analysis.affect_vector, hr, stress, hrv)
    # v8.9 Step 7: feed DODO causal engine
    if _DODO_ENABLED:
        dodo_causal.observe(analysis.affect_vector)
    # GSR data (v8.9: read from state buffer, not direct I/O)
    gsr_data = _sensor_buf.get("gsr")
    if gsr_data:
        analysis.affect_vector["gsr_conductance"] = gsr_data.get("conductance_us")
        analysis.affect_vector["gsr_phasic"] = gsr_data.get("scr_amplitude")
        analysis.affect_vector["gsr_tonic"] = gsr_data.get("scl_level")
    # Forensic logging
    sess = forensic.get_session()
    if sess:
        sess.log_affect(analysis.affect_vector)
    elapsed = rppg_mod.get_elapsed()
    rppg_mod.fps_counter.append(time.time() - t0)
    avg_ms = round(np.mean(rppg_mod.fps_counter) * 1000, 0) if rppg_mod.fps_counter else 0
    refs = reference_ranges.get_reference(analysis.affect_vector.get("age"), analysis.affect_vector.get("race"))

    # ── v7: Per-person tracking (failsafe: if this crashes, v6 still works) ──
    persons_data = []
    quality_data = {"overall_ok": True, "warnings": [], "confidence_mul": 1.0}
    try:
        if cfg.FRAME_QUALITY_ENABLED:
            quality_data = fq.assess(image)
        h, w = image.shape[:2]
        if result.face_landmarks:
            bboxes = []
            for fl in result.face_landmarks:
                xs = [lm.x * w for lm in fl]; ys = [lm.y * h for lm in fl]
                bboxes.append((max(0, int(min(xs) - 10)), max(0, int(min(ys) - 10)),
                               min(w, int(max(xs) + 10)), min(h, int(max(ys) + 10))))
            tracks = face_tracker.update(bboxes)
            # v9.8.4: PAD multi_face_replay compares radar vs faces *in this frame*.
            _pad_n_faces_in_frame = len(tracks)
            active_ids = set()
            for track in tracks:
                tid = track.track_id; active_ids.add(tid)
                # v8.7 (Step 7): run face identity on this track's crop, throttled
                _attach_identity(track, image, result.face_landmarks[track.face_idx], h, w)
                if tid not in person_states:
                    # v8: try to rehydrate PersonState from face_identity cache
                    restored_state = None
                    if face_identity is not None and track.identity:
                        restored_state = face_identity.pop_cached_state(track.identity)
                    if restored_state:
                        ps = restored_state
                        ps.track_id = tid  # update track ID for new session
                    else:
                        ps = PersonState(tid, track.identity)
                    person_states[tid] = ps
                else:
                    ps = person_states[tid]
                _prev_identity = ps.identity
                ps.identity = track.identity or analysis.affect_vector.get("identity")
                # v9.5.x — on first identity attach, hydrate from disk
                if (_calib_store is not None and ps.identity
                        and ps.identity != _prev_identity
                        and not getattr(ps, "_baseline_loaded", False)):
                    try:
                        bl = _calib_store.load(ps.identity)
                        if bl:
                            n = _calib_store.apply_to_person_state(ps, bl)
                            ps._baseline_loaded = True
                            if n:
                                print(f"[calib] hydrated {n} baseline fields for '{ps.identity}'")
                    except Exception:
                        pass
                av = analysis.affect_vector
                # v8.2: per-person rPPG isolation — extract this face's own ROI
                # v8.8: EMA-smooth landmark coords before ROI extraction
                fi = track.face_idx
                if fi < len(result.face_landmarks):
                    _smoothed = ps.smooth_landmarks(result.face_landmarks[fi], rppg_mod.ROI_INDICES)
                    _per_rgb = rppg_mod.extract_rgb_for_person(image, result.face_landmarks[fi], smoothed_coords=_smoothed)
                    if _per_rgb:
                        ps.rppg_rgb.append(_per_rgb)
                        ps.rppg_times.append(time.time())
                    _phr, _phrv, _presp, _pstress, _pcbpm, _pqual = rppg_mod.estimate_hr_person(ps)
                    if _phr is not None:
                        # v8.7 (Step 3): route through Kalman setters for smoothing
                        ps.set_heart_rate(_phr, confidence=_pqual)
                        ps.set_hrv(_phrv)
                        ps.set_resp_rate(_presp)
                        ps.set_stress(_pstress)
                        ps.confirmed_bpm = _pcbpm
                    ps.rppg_quality = _pqual
                    ps.hr_confidence = _pqual
                else:
                    # Fallback to global rPPG if face index is invalid
                    ps.set_heart_rate(av.get("heart_rate_bpm"),
                                      confidence=av.get("signal_quality", {}).get("rppg", 0))
                    ps.hr_confidence = av.get("signal_quality", {}).get("rppg", 0)
                    ps.set_hrv(av.get("hrv_rmssd")); ps.set_stress(av.get("stress_index"))
                    ps.set_resp_rate(av.get("resp_rate"))
                    ps.confirmed_bpm = av.get("confirmed_bpm")
                ps.calibrated_hr = av.get("calibrated_hr")
                ps.oxi_hr = av.get("oxi_hr"); ps.oxi_spo2 = av.get("oxi_spo2")
                # v8.1: per-person radar from spatial assignment mapping
                _rmapping = av.get("_radar_mapping") or {}
                _my_radar = None
                for _rk, _rpi in _rmapping.items():
                    if _rpi == track.face_idx:
                        _my_radar = radar_data if _rk == "A" else radar_b
                        break
                if _my_radar:
                    ps.radar_hr = _my_radar.get("heart_rate")
                    ps.radar_breath = _my_radar.get("breath_rate")
                    ps.radar_distance = _my_radar.get("distance")
                    ps.radar_presence = _my_radar.get("presence")
                else:
                    ps.radar_hr = av.get("radar_hr"); ps.radar_breath = av.get("radar_breath")
                    ps.radar_distance = av.get("radar_distance")
                ps.gsr_conductance = av.get("gsr_conductance")
                ps.head_pose = av.get("head_pose"); ps.gaze = av.get("gaze")
                ps.blink_data = {"blink_rate_pm": av.get("blink_rate"), "perclos": av.get("perclos", 0), "drowsiness": av.get("drowsiness"), "blinked": bool(av.get("blinked"))}
                # v9.8.1 (2026-05-12) — feed the rolling history buffers
                # used by pad/branches/blink_gaze_branch.score(). Without
                # these, blink_gaze stays at rel=0.000 forever (its
                # getattr() returns None and the warmup gate short-
                # circuits at duration is None). The branch needs at
                # least 10 s of (ts, blinked_bool) and (ts, gaze_h, gaze_v)
                # rows to emit a non-zero reliability.
                try:
                    _t_now = time.time()
                    ps.blink_history.append((_t_now, bool(av.get("blinked"))))
                    _g = av.get("gaze") or {}
                    ps.gaze_history.append((_t_now, float(_g.get("horizontal") or 0.0), float(_g.get("vertical") or 0.0)))
                except Exception:
                    pass  # nolint:silent-except best-effort per-frame history
                ps.micro_data = av.get("micro_expression")
                ps.iris_data = {"avg_dilation": av.get("pupil_dilation")} if av.get("pupil_dilation") else None
                ps.duchenne_smile = av.get("duchenne_smile", False)
                ps.fake_smile = av.get("fake_smile", False)
                ps.facial_asymmetry = av.get("facial_asymmetry", 0)
                ps.fear_au_score = av.get("fear_au_score", 0)
                # v8.7 (Step 3): smooth valence/arousal through per-person Kalman
                ps.set_valence_arousal(av.get("valence", 0), av.get("arousal", 0))
                if fi < len(faces):
                    fe = faces[fi].get("fused_emotion") or faces[fi].get("bs_emotion")
                    if fe: ps.update_emotions(fe)
                # v8.7 (Step 5): cold/warm baseline takes over from the
                # one-shot 30s set_baseline() — but keep the legacy call as
                # a guarantee in case feed_baseline_sample ran into trouble.
                if not ps.is_baselined and ps.heart_rate and ps.heart_rate > 0:
                    if time.time() - ps.created_at > 30:
                        ps.set_baseline()
                # v9.4 PAD hook — no-op when FF_PAD_FUSION is off.
                try:
                    _pad_crop = None
                    if fi < len(result.face_landmarks):
                        _pad_crop = _extract_face_crop(image, result.face_landmarks[fi])
                    # v9.8.1 2026-05-13: compute the face bbox from
                    # landmarks so the multi-face-replay heuristic in
                    # _pad_run_once can size-disambiguate real face
                    # vs phone-held face.
                    _pad_face_bbox = None
                    try:
                        _fh, _fw = image.shape[:2]
                        _fl_xs = [_lm.x * _fw for _lm in result.face_landmarks[fi]]
                        _fl_ys = [_lm.y * _fh for _lm in result.face_landmarks[fi]]
                        _pad_face_bbox = (
                            max(0, int(min(_fl_xs))),
                            max(0, int(min(_fl_ys))),
                            min(_fw, int(max(_fl_xs))),
                            min(_fh, int(max(_fl_ys))),
                        )
                    except Exception:
                        pass
                    _dyn_feed = _pad_dynamics_feed_for_face(
                        fi, faces, image.shape, result.face_landmarks)
                    try:
                        ps.face_affect_snapshot = _face_affect_snapshot_from_dyn(
                            fi, _dyn_feed)
                    except Exception:
                        ps.face_affect_snapshot = None
                    _pad_fused = _pad_run_once(
                        pid=track.track_id,
                        face_crop_bgr=_pad_crop,
                        frame_bgr=image,
                        face_bbox=_pad_face_bbox,
                        hr_hint=ps.heart_rate,
                        dynamics_feed=_dyn_feed,
                        n_faces_in_frame=_pad_n_faces_in_frame,
                    )
                except Exception:
                    _pad_fused = None
                _pd = ps.build_affect_dict()
                if _pad_fused is not None:
                    _pd["pad"] = _pad_fused
                persons_data.append(_pd)
            # v8.1: cross-link diarizer speakers to nearest person
            try:
                from diarization import get_diarizer
                dz = get_diarizer()
                dz._update_speakers_energy_fallback()
                _active_spk = dz.active_speakers()
                analysis.affect_vector["diarize_speakers"] = len(_active_spk)
                for spk in _active_spk:
                    if person_states:
                        nearest_tid = min(person_states.keys())
                        dz.attach_to_person(spk["speaker_id"], str(nearest_tid))
            except Exception:
                pass
            # v8: cache PersonState on stale prune (15s absence)
            # v8.4: summarize() writes forensic JSON before pruning
            # v9.5.x: persist baseline to disk so next session starts warm
            for tid in [t for t in person_states if t not in active_ids]:
                ps_stale = person_states[tid]
                try:
                    ps_stale.summarize()
                except Exception:
                    pass
                if face_identity is not None and ps_stale.identity:
                    face_identity.cache_state(ps_stale.identity, ps_stale)
                if _calib_store is not None and getattr(ps_stale, "identity", None):
                    try:
                        baseline = _calib_store.extract_from_person_state(ps_stale)
                        if baseline:
                            _calib_store.save(ps_stale.identity, baseline)
                    except Exception:
                        pass
                del person_states[tid]
    except Exception as e:
        print(f"[v7 WARN] Fast-path tracking error (v6 still works): {e}")

    # v8.7 (Step 8): feed the active calibration lap. No-op if no lap is
    # running. 1Hz internal throttle keeps this from inflating CPU.
    try:
        calibration_lap.maybe_feed(persons_data)
    except Exception:
        pass

    # v8.7 (Step 8): expose lap status so the UI can render the live prompt.
    _lap_status = None
    try:
        _ls = calibration_lap.status()
        if _ls.get("active") or _ls.get("completed") or _ls.get("aborted"):
            _lap_status = _ls
    except Exception:
        _lap_status = None

    _fast_result = sanitize({
        "faces": faces, "av": analysis.affect_vector, "hr": hr, "wave": wave,
        "confirmed_bpm": rppg_mod.get_confirmed_bpm(), "elapsed": elapsed,
        "buf": len(rppg_mod.rppg_rgb), "samples": len(rppg_mod.hr_samples),
        "mode": "pos", "ai": list(llm.ai_messages)[-3:], "ms": avg_ms,
        "ref": refs, "voice": voice_monitor.get_voice_metrics(),
        "gsr": _sensor_buf.get("gsr"), "dual_cam": dual_camera.get_status(),
        "rppg_q": rppg_mod.get_signal_quality(),
        "persons": persons_data, "frame_quality": quality_data,
        "calibration_lap": _lap_status,
    })
    _frame_cache.put(_fast_result)  # v8.9 Step 4: cache for WS fallback
    _motion_gate_record_now()        # v9.2 step 5: arm the 250 ms reuse window
    return _fast_result

def _process_frame_full(image):
    """Process frame with MediaPipe + background DeepFace + v7 per-person tracking."""
    global _frame_count
    _frame_count += 1
    rppg_mod.mark_start()
    t0 = time.time()

    # Step 9: Frame quality assessment
    quality_data = fq.assess(image) if cfg.FRAME_QUALITY_ENABLED else {"overall_ok": True, "warnings": [], "confidence_mul": 1.0}

    image = _adaptive_brightness(image)
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detectors.detect_faces(rgb)
    faces = []
    # v9.0 Step 1 (Stanford Monadic Option Types): Enforce strict pre-binding
    # of ALL variables to typed None at the TOP of lexical scope. Guarantees
    # the symbol table is fully populated so a sensor timeout broadcasts a
    # typed None payload instead of crashing with UnboundLocalError.
    # ── Physiological pipeline outputs ──
    hr: "float | None" = None
    wave: "list | None" = None
    q: "float | None" = None
    resp: "float | None" = None
    hrv: "float | None" = None
    stress: "float | None" = None
    # ── Sensor data (decoupled state buffer) ──
    radar_data: "dict | None" = None
    radar_b: "dict | None" = None
    oxi_data: "dict | None" = None
    oxi_raw: "dict | None" = None
    gsr_data: "dict | None" = None
    # ── DeepFace intermediates ──
    first_df: "dict | None" = None
    first_fused: "dict | None" = None
    # ── Dempster-Shafer fusion intermediates ──
    _ds_best_val: "float | None" = None
    _ds_best_conf: float = 0.0
    _ds_src_idx: int = -1
    _fused_hr: "float | None" = None
    # ── Per-person tracking scratch ──
    persons_data: list = []
    _my_radar: "dict | None" = None
    if result.face_landmarks:
        for i, fl in enumerate(result.face_landmarks):
            if i not in _df_pending:
                # v9.2 (plan step 10-11): DeepFace demographics are split
                # from emotion. HSEmotion handles emotion every fast frame;
                # demographics only run when the bbox actually shifted AND
                # at least DF_DEMOGRAPHICS_MIN_INTERVAL_S has elapsed since
                # the last run. This drops the ~120 ms DeepFace wall cost
                # from 2 Hz to 0.5 Hz in the steady state without losing
                # responsiveness — age/gender/race barely change.
                _bb = _bbox_grid(fl, h, w)
                if not _should_rerun_df_demographics(i, _bb):
                    continue
                # v9.7.1: respect the DeepFace circuit breaker. When the
                # breaker is open, skip submission unless we're in the
                # probe window — that lets the system self-heal without
                # a full restart while preventing burst-spam during the
                # crash window.
                if _df_breaker_should_skip():
                    continue
                crop = detectors.get_face_crop(image, fl, h, w)
                if crop is not None:
                    bs_dict = {}
                    if result.face_blendshapes and i < len(result.face_blendshapes):
                        bs_dict = {b.category_name: round(b.score, 3) for b in result.face_blendshapes[i]}
                    _df_pending.add(i)
                    _df_bbox_cache[i] = _bb
                    _df_demographics_last_run[i] = time.time()
                    # v9.8.1 FIX: revert to thread-pool submit. The
                    # API_anti modular refactor swapped this from
                    # `detectors.pool.submit(_bg_deepface, ...)` to
                    # `worker_instance.submit(...)`, but no result-
                    # collector for `worker_instance.output_queue` was
                    # ever wired up. Result: every DeepFace job vanished
                    # into the subprocess output queue, _df_cache stayed
                    # empty forever, dfBlock.connected was permanently
                    # False, the AGE/GENDER/ETHNICITY tiles in AI HISTORY
                    # flipped to FAILED after the 8 s warmup, and the top
                    # status bar showed `DF: idle` from boot to shutdown.
                    # `_bg_deepface` (defined above) does the LRU-bounded
                    # write into _df_cache and feeds the circuit breaker;
                    # detectors.pool is the 24-worker ThreadPoolExecutor
                    # `_bg_deepface` was designed for. The multiprocess
                    # `worker_instance` is now dead code; left imported so
                    # we can revisit subprocess isolation later without
                    # another refactor pass.
                    detectors.pool.submit(_bg_deepface, i, crop, bs_dict)
        # v9.8.1 (reverted gate): full-frame path also feeds rPPG -- both
        # cameras' samples enter the same buffer per the module's dual-
        # source design (see fast-frame path comment above).
        rppg_mod.add_sample(image, result.face_landmarks[0])
        hr, wave, q, resp = rppg_mod.estimate_hr()
        hrv, stress = rppg_mod.estimate_hrv()
        # Build face data with per-frame blended emotions
        # v9.2 (plan step 6): run build_face_full in parallel per face.
        first_df = first_fused = None
        _fl_list_full = list(enumerate(result.face_landmarks))
        _bs_list_full = []
        for fi, fl in _fl_list_full:
            bs_dict = {}
            if result.face_blendshapes and fi < len(result.face_blendshapes):
                bs_dict = {b.category_name: round(b.score, 3) for b in result.face_blendshapes[fi]}
            _bs_list_full.append(bs_dict)
        if len(_fl_list_full) == 1:
            fi, fl = _fl_list_full[0]
            _built = [analysis.build_face_full(fi, fl, _bs_list_full[0])]
        else:
            _built = list(detectors.pool.map(
                lambda triplet: analysis.build_face_full(triplet[0], triplet[1], triplet[2]),
                [(fi, fl, bs) for (fi, fl), bs in zip(_fl_list_full, _bs_list_full)],
            ))
        # Merge DF cache + fused emotion sequentially (touches the _df_lock)
        for (fi, _fl), face in zip(_fl_list_full, _built):
            with _df_lock:
                cached = _df_cache_get(fi)
            if cached and time.time() - cached["t"] < cfg.DF_CACHE_MAX_AGE:
                face["df"] = cached["df"]
                face["av"] = cached["av"]
                face["identity"] = cached["df"].get("identity")
                cached_scores = cached["df"].get("emotion", {}).get("scores") or cached["df"].get("emotion", {}).get("fused_scores")
                if cached_scores and face.get("bs_emotion"):
                    live_fused = analysis.fuse_emotions(cached_scores, face["bs_emotion"])
                    face["fused_emotion"] = live_fused
                    if fi == 0:
                        first_df = cached["df"]
                        first_fused = live_fused
                else:
                    if fi == 0:
                        first_df = cached["df"]
                        first_fused = cached["fused"]
            faces.append(face)
        pose0 = gaze0 = blink0 = micro0 = iris0 = None
        if faces:
            pose0 = faces[0].get("pose"); gaze0 = faces[0].get("gaze")
            blink0 = faces[0].get("blink"); micro0 = faces[0].get("micro")
            iris0 = faces[0].get("iris")
        # v8.9: read sensors from decoupled state buffer (never blocks on I/O)
        radar_data = _sensor_buf.get("radar_a")
        radar_b = _sensor_buf.get("radar_b")
        analysis.update_affect_vector(
            first_df, hr, hrv, stress, q, resp, pose0, gaze0, blink0, micro0,
            first_fused or (faces[0].get("bs_emotion") if faces else None),
            iris0, radar_data,
        )
        _merge_fast_bio()  # override with 120fps data if available
        if first_df and first_df.get("identity"):
            analysis.affect_vector["identity"] = first_df["identity"]
        else:
            analysis.affect_vector["identity"] = None
    else:
        hr, wave, q, resp = None, [], 0, None
    # v8.9: read oximeter from state buffer + radar motion gating inline
    oxi_raw = _sensor_buf.get("oximeter")
    oxi_data = None
    if oxi_raw:
        oxi_data = dict(oxi_raw)
        if radar_data and radar_data.get("distance") is not None:
            _mv = radar_data.get("motion_variance", 0) or 0
            if _mv > 0.04:
                _att = max(0.1, 1.0 - min(1.0, _mv / 0.2))
                oxi_data["signal_strength"] = max(0, int((oxi_data.get("signal_strength") or 0) * _att))
                oxi_data["motion_gated"] = True
            else:
                oxi_data["motion_gated"] = False
    # v9.1 (optimization step 19): hardware link state always flows, even
    # when no finger is seated — UI uses this for Searching/Connected/Stale.
    try:
        _oxi_link = oximeter.get_oximeter_status() or {}
    except Exception:
        _oxi_link = {}
    analysis.affect_vector["oxi_connected"] = bool(_oxi_link.get("connected"))
    analysis.affect_vector["oxi_mode"] = _oxi_link.get("mode")
    if oxi_data:
        analysis.affect_vector["oxi_hr"] = oxi_data.get("heart_rate")
        analysis.affect_vector["oxi_spo2"] = oxi_data.get("spo2")
        analysis.affect_vector["oxi_signal"] = oxi_data.get("signal_strength")
        analysis.affect_vector["oxi_motion_gated"] = oxi_data.get("motion_gated", False)
        analysis.affect_vector["oxi_pleth"] = oxi_data.get("pleth_waveform")
        _oxi_ts = oxi_data.get("last_update") or oxi_data.get("timestamp")
        if _oxi_ts:
            analysis.affect_vector["oxi_last_update"] = float(_oxi_ts)
            analysis.affect_vector["oxi_age_s"] = round(max(0.0, time.time() - float(_oxi_ts)), 2)
        else:
            analysis.affect_vector["oxi_last_update"] = time.time()
            analysis.affect_vector["oxi_age_s"] = 0.0
        # v9.7.6: pass BOTH radar A and B HR so calibration can train
        # per-radar corrections. radar_data is radar A (legacy single-
        # radar field name), radar_b_data is radar B.
        _radar_b_data = _sensor_buf.get("radar_b")
        calibration.add_sample(
            hr,
            oxi_data.get("heart_rate"),
            radar_a_hr=(radar_data.get("heart_rate") if radar_data else None),
            radar_b_hr=(_radar_b_data.get("heart_rate") if _radar_b_data else None),
        )
    else:
        analysis.affect_vector["oxi_hr"] = None
        analysis.affect_vector["oxi_spo2"] = None
        analysis.affect_vector["oxi_signal"] = None
        analysis.affect_vector["oxi_pleth"] = None
        _prev_ts = analysis.affect_vector.get("oxi_last_update")
        if _prev_ts:
            analysis.affect_vector["oxi_age_s"] = round(max(0.0, time.time() - float(_prev_ts)), 2)
    if hr and calibration._calibration["samples_collected"] > 0:
        analysis.affect_vector["calibrated_hr"] = calibration.correct_hr(hr)
    analysis.affect_vector["confirmed_bpm"] = rppg_mod.get_confirmed_bpm()
    analysis.affect_vector["rppg_quality"] = q
    # v8.9: Dempster-Shafer HR source selection via confidence weights
    _oxi_hr, _oxi_conf = _sensor_buf.get_with_confidence("oximeter")
    _rad_hr_val = radar_data.get("heart_rate") if radar_data else None
    _, _rad_conf = _sensor_buf.get_with_confidence("radar_a")
    _rppg_conf = min(1.0, (q or 0) / 100.0) if q else 0.0
    _ds_best_val, _ds_best_conf, _ds_src_idx = sensor_state.ds_select_best(
        (_oxi_hr.get("heart_rate") if _oxi_hr else None, _oxi_conf),
        (_rad_hr_val, _rad_conf),
        (hr, _rppg_conf),
    )
    _ds_labels = ["OXI", "RADAR", "rPPG"]
    analysis.affect_vector["hr_source"] = _ds_labels[_ds_src_idx] if _ds_best_val else None
    analysis.affect_vector["hr_source_confidence"] = round(_ds_best_conf, 3)
    _fused_hr = sensor_state.ds_combine_hr(
        (_oxi_hr.get("heart_rate") if _oxi_hr else None, _oxi_conf),
        (_rad_hr_val, _rad_conf),
        (hr, _rppg_conf),
    )
    if _fused_hr is not None:
        analysis.affect_vector["fused_hr_bpm"] = _fused_hr
    logger.log_affect(analysis.affect_vector)
    logger.push_lsl(analysis.affect_vector, hr, stress, hrv)
    # v8.9 Step 7: feed DODO causal engine
    if _DODO_ENABLED:
        dodo_causal.observe(analysis.affect_vector)
    # GSR data (v8.9: read from state buffer, not direct I/O)
    gsr_data = _sensor_buf.get("gsr")
    if gsr_data:
        analysis.affect_vector["gsr_conductance"] = gsr_data.get("conductance_us")
        analysis.affect_vector["gsr_phasic"] = gsr_data.get("scr_amplitude")
        analysis.affect_vector["gsr_tonic"] = gsr_data.get("scl_level")
    # Forensic logging
    sess = forensic.get_session()
    if sess:
        sess.log_affect(analysis.affect_vector)
    # Expire stale cache entries
    _expire_cache()
    elapsed = rppg_mod.get_elapsed()
    rppg_mod.fps_counter.append(time.time() - t0)
    avg_ms = round(np.mean(rppg_mod.fps_counter) * 1000, 0) if rppg_mod.fps_counter else 0
    refs = reference_ranges.get_reference(analysis.affect_vector.get("age"), analysis.affect_vector.get("race"))

    # ── Step 1+4: Per-person tracking & state (failsafe) ──
    persons_data = []
    try:
        if result.face_landmarks:
            bboxes = []
            for fl in result.face_landmarks:
                xs = [lm.x * w for lm in fl]; ys = [lm.y * h for lm in fl]
                bboxes.append((max(0, int(min(xs) - 10)), max(0, int(min(ys) - 10)),
                               min(w, int(max(xs) + 10)), min(h, int(max(ys) + 10))))
            tracks = face_tracker.update(bboxes)
            # v9.8.4: PAD multi_face_replay compares radar vs faces *in this frame*.
            _pad_n_faces_in_frame = len(tracks)
            active_ids = set()
            for track in tracks:
                tid = track.track_id; active_ids.add(tid)
                # v8.7 (Step 7): run face identity on this track's crop, throttled
                _attach_identity(track, image, result.face_landmarks[track.face_idx], h, w)
                if tid not in person_states:
                    # v8: try to rehydrate PersonState from face_identity cache
                    restored_state = None
                    if face_identity is not None and track.identity:
                        restored_state = face_identity.pop_cached_state(track.identity)
                    if restored_state:
                        ps = restored_state
                        ps.track_id = tid  # update track ID for new session
                    else:
                        ps = PersonState(tid, track.identity)
                    person_states[tid] = ps
                else:
                    ps = person_states[tid]
                _prev_identity = ps.identity
                ps.identity = track.identity or analysis.affect_vector.get("identity")
                # v9.5.x — on first identity attach, hydrate from disk
                if (_calib_store is not None and ps.identity
                        and ps.identity != _prev_identity
                        and not getattr(ps, "_baseline_loaded", False)):
                    try:
                        bl = _calib_store.load(ps.identity)
                        if bl:
                            n = _calib_store.apply_to_person_state(ps, bl)
                            ps._baseline_loaded = True
                            if n:
                                print(f"[calib] hydrated {n} baseline fields for '{ps.identity}'")
                    except Exception:
                        pass
                av = analysis.affect_vector
                # v8.2: per-person rPPG isolation — extract this face's own ROI
                # v8.8: EMA-smooth landmark coords before ROI extraction
                fi = track.face_idx
                if fi < len(result.face_landmarks):
                    _smoothed = ps.smooth_landmarks(result.face_landmarks[fi], rppg_mod.ROI_INDICES)
                    _per_rgb = rppg_mod.extract_rgb_for_person(image, result.face_landmarks[fi], smoothed_coords=_smoothed)
                    if _per_rgb:
                        ps.rppg_rgb.append(_per_rgb)
                        ps.rppg_times.append(time.time())
                    _phr, _phrv, _presp, _pstress, _pcbpm, _pqual = rppg_mod.estimate_hr_person(ps)
                    if _phr is not None:
                        # v8.7 (Step 3): route through Kalman setters for smoothing
                        ps.set_heart_rate(_phr, confidence=_pqual)
                        ps.set_hrv(_phrv)
                        ps.set_resp_rate(_presp)
                        ps.set_stress(_pstress)
                        ps.confirmed_bpm = _pcbpm
                    ps.rppg_quality = _pqual
                    ps.hr_confidence = _pqual
                else:
                    ps.set_heart_rate(av.get("heart_rate_bpm"),
                                      confidence=av.get("signal_quality", {}).get("rppg", 0))
                    ps.hr_confidence = av.get("signal_quality", {}).get("rppg", 0)
                    ps.set_hrv(av.get("hrv_rmssd")); ps.set_stress(av.get("stress_index"))
                    ps.set_resp_rate(av.get("resp_rate"))
                    ps.confirmed_bpm = av.get("confirmed_bpm")
                ps.calibrated_hr = av.get("calibrated_hr")
                ps.oxi_hr = av.get("oxi_hr"); ps.oxi_spo2 = av.get("oxi_spo2")
                # v8.1: per-person radar from spatial assignment mapping
                _rmapping = av.get("_radar_mapping") or {}
                _my_radar = None
                for _rk, _rpi in _rmapping.items():
                    if _rpi == track.face_idx:
                        _my_radar = radar_data if _rk == "A" else radar_b
                        break
                if _my_radar:
                    ps.radar_hr = _my_radar.get("heart_rate")
                    ps.radar_breath = _my_radar.get("breath_rate")
                    ps.radar_distance = _my_radar.get("distance")
                    ps.radar_presence = _my_radar.get("presence")
                else:
                    ps.radar_hr = av.get("radar_hr"); ps.radar_breath = av.get("radar_breath")
                    ps.radar_distance = av.get("radar_distance")
                ps.gsr_conductance = av.get("gsr_conductance")
                ps.head_pose = av.get("head_pose"); ps.gaze = av.get("gaze")
                ps.blink_data = {"blink_rate_pm": av.get("blink_rate"), "perclos": av.get("perclos", 0), "drowsiness": av.get("drowsiness"), "blinked": bool(av.get("blinked"))}
                # v9.8.1 (2026-05-12) — feed the rolling history buffers
                # used by pad/branches/blink_gaze_branch.score(). Without
                # these, blink_gaze stays at rel=0.000 forever (its
                # getattr() returns None and the warmup gate short-
                # circuits at duration is None). The branch needs at
                # least 10 s of (ts, blinked_bool) and (ts, gaze_h, gaze_v)
                # rows to emit a non-zero reliability.
                try:
                    _t_now = time.time()
                    ps.blink_history.append((_t_now, bool(av.get("blinked"))))
                    _g = av.get("gaze") or {}
                    ps.gaze_history.append((_t_now, float(_g.get("horizontal") or 0.0), float(_g.get("vertical") or 0.0)))
                except Exception:
                    pass  # nolint:silent-except best-effort per-frame history
                ps.micro_data = av.get("micro_expression")
                ps.iris_data = {"avg_dilation": av.get("pupil_dilation")} if av.get("pupil_dilation") else None
                ps.duchenne_smile = av.get("duchenne_smile", False)
                ps.fake_smile = av.get("fake_smile", False)
                ps.facial_asymmetry = av.get("facial_asymmetry", 0)
                ps.fear_au_score = av.get("fear_au_score", 0)
                # v8.7 (Step 3): Kalman-smoothed valence/arousal
                ps.set_valence_arousal(av.get("valence", 0), av.get("arousal", 0))
                if fi < len(faces):
                    fe = faces[fi].get("fused_emotion") or faces[fi].get("bs_emotion")
                    if fe: ps.update_emotions(fe)
                    if faces[fi].get("df"):
                        df_data = faces[fi]["df"]
                        ps.update_demographics(df_data.get("raw_age"), df_data.get("gender",{}).get("scores"), df_data.get("race",{}).get("scores"))
                if not ps.is_baselined and ps.heart_rate and ps.heart_rate > 0:
                    if time.time() - ps.created_at > 30:
                        ps.set_baseline()
                # v9.4 PAD hook — no-op when FF_PAD_FUSION is off.
                try:
                    _pad_crop = None
                    if fi < len(result.face_landmarks):
                        _pad_crop = _extract_face_crop(image, result.face_landmarks[fi])
                    # v9.8.1 2026-05-13: compute the face bbox from
                    # landmarks so the multi-face-replay heuristic in
                    # _pad_run_once can size-disambiguate real face
                    # vs phone-held face.
                    _pad_face_bbox = None
                    try:
                        _fh, _fw = image.shape[:2]
                        _fl_xs = [_lm.x * _fw for _lm in result.face_landmarks[fi]]
                        _fl_ys = [_lm.y * _fh for _lm in result.face_landmarks[fi]]
                        _pad_face_bbox = (
                            max(0, int(min(_fl_xs))),
                            max(0, int(min(_fl_ys))),
                            min(_fw, int(max(_fl_xs))),
                            min(_fh, int(max(_fl_ys))),
                        )
                    except Exception:
                        pass
                    _dyn_feed = _pad_dynamics_feed_for_face(
                        fi, faces, image.shape, result.face_landmarks)
                    try:
                        ps.face_affect_snapshot = _face_affect_snapshot_from_dyn(
                            fi, _dyn_feed)
                    except Exception:
                        ps.face_affect_snapshot = None
                    _pad_fused = _pad_run_once(
                        pid=track.track_id,
                        face_crop_bgr=_pad_crop,
                        frame_bgr=image,
                        face_bbox=_pad_face_bbox,
                        hr_hint=ps.heart_rate,
                        dynamics_feed=_dyn_feed,
                        n_faces_in_frame=_pad_n_faces_in_frame,
                    )
                except Exception:
                    _pad_fused = None
                _pd = ps.build_affect_dict()
                if _pad_fused is not None:
                    _pd["pad"] = _pad_fused
                persons_data.append(_pd)
            # v8: cache PersonState on stale prune (15s absence)
            # v8.4: summarize() writes forensic JSON before pruning
            # v9.5.x: persist baseline to disk so next session starts warm
            for tid in [t for t in person_states if t not in active_ids]:
                ps_stale = person_states[tid]
                try:
                    ps_stale.summarize()
                except Exception:
                    pass
                if face_identity is not None and ps_stale.identity:
                    face_identity.cache_state(ps_stale.identity, ps_stale)
                if _calib_store is not None and getattr(ps_stale, "identity", None):
                    try:
                        baseline = _calib_store.extract_from_person_state(ps_stale)
                        if baseline:
                            _calib_store.save(ps_stale.identity, baseline)
                    except Exception:
                        pass
                del person_states[tid]
    except Exception as e:
        print(f"[v7 WARN] Full-path tracking error (v6 still works): {e}")

    # v8.4: auto-save forensic snapshot every 60s
    global _last_auto_save
    if time.time() - _last_auto_save >= 60 and person_states:
        _last_auto_save = time.time()
        try:
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            auto_path = os.path.join(cfg.SESSIONS_DIR, f"auto_{ts_str}.json")
            snapshot = {
                "timestamp": time.time(),
                "persons": persons_data,
                "affect_vector": dict(analysis.affect_vector),
            }
            import json as _json
            with open(auto_path, "w") as _af:
                _json.dump(sanitize(snapshot), _af, indent=2, default=str)
        except Exception:
            pass

    # v8.7 (Step 8): feed the active calibration lap (full pipeline path).
    try:
        calibration_lap.maybe_feed(persons_data)
    except Exception:
        pass
    _lap_status = None
    try:
        _ls = calibration_lap.status()
        if _ls.get("active") or _ls.get("completed") or _ls.get("aborted"):
            _lap_status = _ls
    except Exception:
        _lap_status = None

    # v9.0 Step 9: feed EEG data to the photic habituation LZC monitor
    _photic_status = None
    if _photic_monitor is not None:
        try:
            eeg_state = eeg.get_eeg_state() if eeg else None
            if eeg_state and eeg_state.get("connected"):
                # Feed raw EEG samples from the buffer (under lock
                # to prevent race with the EEG worker thread)
                with eeg._LOCK:
                    eeg_buf_data = [v for (_, v) in eeg._eeg_buffer] if eeg._eeg_buffer else []
                if eeg_buf_data:
                    _photic_status = _photic_monitor.feed(
                        np.array(eeg_buf_data[-500:], dtype=np.float32),
                        stimulation_active=False,  # UI sets this via WS
                    )
                    if _photic_status and _photic_status.get("suppress"):
                        analysis.affect_vector["photic_suppress"] = True
                        print("[SAFETY] LZC distress detected — visual stimulus suppressed")
        except Exception:
            pass

    # v9.0 Step 5: attach ToF seat validation status to affect vector
    try:
        tof_status = oximeter.get_tof_status()
        analysis.affect_vector["tof_seated"] = tof_status.get("finger_seated")
        analysis.affect_vector["tof_halted"] = tof_status.get("halted", False)
    except Exception:
        pass

    _full_result = sanitize({
        "faces": faces, "av": analysis.affect_vector, "hr": hr, "wave": wave,
        "confirmed_bpm": rppg_mod.get_confirmed_bpm(), "elapsed": elapsed,
        "buf": len(rppg_mod.rppg_rgb), "samples": len(rppg_mod.hr_samples),
        "mode": "pos", "ai": list(llm.ai_messages)[-3:], "ms": avg_ms,
        "ref": refs, "voice": voice_monitor.get_voice_metrics(),
        "gsr": _sensor_buf.get("gsr"), "dual_cam": dual_camera.get_status(),
        "rppg_q": rppg_mod.get_signal_quality(),
        "persons": persons_data, "frame_quality": quality_data,
        "calibration_lap": _lap_status,
        "photic_monitor": _photic_status,
    })
    _frame_cache.put(_full_result)  # v8.9 Step 4: cache for WS fallback
    return _full_result

# ── v8.8: msgpack binary serialization for WebSocket throughput ──
try:
    import msgpack as _msgpack
    _MSGPACK_OK = True
    print("[OK] msgpack available — binary WebSocket mode enabled")
except ImportError:
    _MSGPACK_OK = False
    print("[--] msgpack not installed — WebSocket stays JSON (pip install msgpack)")

def _ws_pack(data: dict) -> bytes:
    """Serialize response dict to msgpack bytes, with fallback for non-serializable types."""
    if not _MSGPACK_OK:
        return json.dumps(data, default=str).encode()
    return _msgpack.packb(data, use_bin_type=True, default=str)

# ══════════════════════════════════════════════
#  WebSocket Endpoint (primary, ~40% less latency)
#  v8.8: binary msgpack responses when client sends "Accept: msgpack" subprotocol
#  or sends binary frames. Falls back to JSON for text-mode clients.
# ══════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# v9.3 plan step 5 — Multiplexed WS protocol
# ═══════════════════════════════════════════════════════════════════
#
# Client → server binary header (little-endian):
#
#   byte  0 : mode       0=fast, 1=full, 2=client_landmarks_only
#   byte  1 : flags      bit0=has_jpeg, bit1=has_lmk, bit2=wants_correction
#   bytes 2-3 : u16 lmk_count (number of landmarks in the lmk array)
#   bytes 4-7 : u32 jpeg_len (big-endian for historic compatibility)
#   bytes 8-11: u32 pid (client's stable face id for lmk array)
#   payload   : [lmk Float32 lmk_count*3]  then  [jpeg_bytes jpeg_len]
#
# Server → client envelopes (each is a standalone message):
#
#   {"t":"frame", ...}    legacy full frame result (mode 0/1)
#   {"t":"corr",  "pid":…, "lmk_delta":…, "conf":…}
#   {"t":"hr",    "pid":…, "bpm":…, "sqi":…, "ts":…}
#   {"t":"spo2",  "pid":…, "spo2":…, "ts":…}
#   {"t":"decep", "pid":…, "score":…, "indicators":{…}, "ts":…}
#   {"t":"llm",   "pid":…, "summary":…, "tokens":…, "ts":…}
#   {"t":"vitals","radar":{…}, "ts":…}
#
# All envelopes carry ts_server_mono = time.monotonic() so the client can
# histogram per-source latency (step 10 HUD).
import struct as _struct
import feature_flags as _ff_mod  # server-side flag resolution

def _parse_v93_header(buf: bytes):
    """Decode the v9.3 client→server binary header. Returns dict or None
    if the buffer is in the legacy v9.2 format (byte 0 = mode, rest = JPEG).

    Detection heuristic: v9.3 always sets flags byte's bit 2 (HEADER_MAGIC)
    if flags byte has bit 3 set we treat it as v9.3, otherwise legacy.
    """
    if len(buf) < 12:
        return None
    magic = buf[1] & 0x08  # bit 3 reserved as v9.3 magic
    if not magic:
        return None
    try:
        mode = buf[0]
        flags = buf[1]
        lmk_count = _struct.unpack_from("<H", buf, 2)[0]
        jpeg_len = _struct.unpack_from(">I", buf, 4)[0]
        pid = _struct.unpack_from("<I", buf, 8)[0]
        offset = 12
        lmk = None
        if flags & 0x02:  # has_lmk
            need = lmk_count * 3 * 4  # float32
            if len(buf) < offset + need:
                return None
            lmk = np.frombuffer(buf, dtype=np.float32, count=lmk_count * 3, offset=offset)
            lmk = lmk.reshape(-1, 3)
            offset += need
        jpeg = None
        if flags & 0x01 and jpeg_len > 0:
            if len(buf) < offset + jpeg_len:
                return None
            jpeg = buf[offset:offset + jpeg_len]
            offset += jpeg_len
        # v9.4.1 step 3: optional trailing float32 padFast.moire_rgb.
        pad_fast_moire = None
        if flags & 0x10:
            if len(buf) >= offset + 4:
                pad_fast_moire = float(_struct.unpack_from("<f", buf, offset)[0])
                offset += 4
        return {
            "mode": mode, "flags": flags, "lmk_count": lmk_count,
            "pid": pid, "lmk": lmk, "jpeg": jpeg,
            "wants_correction": bool(flags & 0x04),
            "pad_fast_moire": pad_fast_moire,
        }
    except Exception:
        return None

# Per-source cadence tracking so we only emit envelopes when the underlying
# source has fresh data. Stores (last_sent_ts, last_value_fingerprint).
_ws_last_emit: dict = {}

def _envelope_should_send(source: str, fingerprint, min_interval_ms: int) -> bool:
    now = time.monotonic() * 1000.0
    last_t, last_fp = _ws_last_emit.get(source, (0.0, None))
    if (now - last_t) < min_interval_ms and last_fp == fingerprint:
        return False
    _ws_last_emit[source] = (now, fingerprint)
    return True

def _build_envelopes(result: dict):
    """Split a frame result into per-source envelopes. Drops an envelope
    entirely if its fingerprint hasn't changed OR the minimum cadence
    window hasn't elapsed — this is the 'natural cadence' contract the
    v9.3 plan requires for per-source same-moment delivery."""
    envs = []
    av = result.get("av") or {}
    ts = time.monotonic()
    # HR envelope — natural cadence ~1 Hz
    _bpm = av.get("heart_rate_bpm") or av.get("confirmed_bpm")
    if _bpm is not None and _envelope_should_send("hr", round(_bpm, 1), 800):
        envs.append({
            "t": "hr", "pid": 0, "bpm": round(float(_bpm), 1),
            "sqi": av.get("rppg_quality", 0),
            "source": av.get("hr_source"),
            "ts": ts,
        })
    # SpO2 envelope — oximeter cadence
    _sp = av.get("oxi_spo2")
    if _sp is not None and _envelope_should_send("spo2", _sp, 500):
        envs.append({"t": "spo2", "pid": 0, "spo2": int(_sp), "ts": ts})
    # Deception envelope — 4 Hz
    _persons = result.get("persons") or []
    if _persons:
        _p0 = _persons[0] or {}
        _dec = (_p0.get("deception") or {}).get("score")
        if _dec is not None and _envelope_should_send("decep", round(_dec, 3), 250):
            envs.append({
                "t": "decep", "pid": 0, "score": round(float(_dec), 3),
                "indicators": (_p0.get("deception") or {}).get("indicators") or {},
                "ts": ts,
            })
    # LLM envelope — event-driven
    _ai = result.get("ai") or []
    if _ai:
        _latest = _ai[-1] if _ai else None
        if _latest and _envelope_should_send("llm", _latest.get("t"), 0):
            envs.append({
                "t": "llm", "pid": 0,
                "summary": _latest.get("text", ""),
                "tokens": _latest.get("tokens", 0),
                "ts": ts,
            })
    # Vitals envelope — radar/dual-cam cadence
    _rad_fp = (av.get("radar_hr"), av.get("radar_breath"),
               av.get("radar_b_hr"), av.get("radar_b_breath"))
    if any(x is not None for x in _rad_fp) and _envelope_should_send("vitals", _rad_fp, 200):
        envs.append({
            "t": "vitals",
            "radar": {
                "a_hr": av.get("radar_hr"), "a_br": av.get("radar_breath"),
                "a_dist": av.get("radar_distance"),
                "b_hr": av.get("radar_b_hr"), "b_br": av.get("radar_b_breath"),
                "b_dist": av.get("radar_b_distance"),
                "people_total": av.get("radar_people_total"),
            },
            "gsr": result.get("gsr"),
            "voice": result.get("voice"),
            "ts": ts,
        })
    return envs

async def _send_envelope(websocket, env, use_binary: bool):
    if use_binary:
        await websocket.send_bytes(_ws_pack(env))
    else:
        await websocket.send_json(env)

app.state.ws_ctx = {
    "_auth": _auth, "cfg": cfg, "_send_envelope": _send_envelope,
    "_onnx_pool": _onnx_pool, "_process_frame_fast": _process_frame_fast,
    "_process_frame_full": _process_frame_full, 
    "_process_frame_fast_with_lmk": _process_frame_fast_with_lmk,
    "_frame_cache": _frame_cache, "_build_envelopes": _build_envelopes,
    "_REG": _REG, "_ws_active_delta": _ws_active_delta,
    "_start_time": _start_time, "_replay": _replay,
    "_ff_mod": globals().get("_ff_mod", None),
    "_MSGPACK_OK": globals().get("_MSGPACK_OK", False),
    "_pad_last_client_fast": globals().get("_pad_last_client_fast", {}),
    "_result_from_client_lmk": globals().get("_result_from_client_lmk", None),
    "_runtime_profiler": globals().get("_runtime_profiler", None),
    "_parse_v93_header": globals().get("_parse_v93_header", None)
}

from app.api.ws import router as ws_router
from app.api.routes import router as http_router, bind_main_globals as _bind_routes_globals
# v9.8.1 (2026-05-12) — wire the actual PAD orchestrator into main's globals.
# RCA: `_pad_run_once()` is CALLED from _process_frame_full (main.py:1702)
# and _process_frame_fast (main.py:2145), but it lives in app/api/routes.py
# after the v9.7.0 modular extraction. Those call sites are wrapped in
# `try/except Exception:` (so any error -> `_pad_fused = None`), which
# meant a silent NameError fired every frame for every face. Result: PAD
# fusion never ran from the per-frame path, _PAD_WARMUP_TELEMETRY stayed
# empty, and /pad/{pid} forever returned `{last: null}` ("warming"
# forever). Surface the symbol here so the call sites resolve correctly.
from app.api.routes import _pad_run_once as _pad_run_once
from app.api.webrtc import router as webrtc_router
app.include_router(ws_router)
app.include_router(http_router)
app.include_router(webrtc_router)

# v9.7.0-merged 2026-05-10: route handlers in app/api/routes.py were extracted
# from main.py during the API_anti refactor but reference dozens of main.py
# module-level names (person_states, _start_time, _pad_challenge_sessions,
# helper fns, etc.) that were never wired through. Inject them at startup so
# the handlers see what they expect. Existing names in routes.py are not
# overwritten -- targeted re-imports there stay authoritative.
@app.on_event("startup")
async def _route_globals_inject():
    import sys
    _bind_routes_globals(sys.modules[__name__])
    print("[OK] routes.py globals bound from main")

if __name__ == "__main__":
    # v9.8.1: removed `worker_instance.start()` -- the multiprocessing
    # DeepFace worker was orphan dead code (no result-collector for
    # `worker_instance.output_queue` was ever wired up; see fix at
    # main.py:1658 + DeepFace bug RCA in SESSION_HANDOFF_2026-05-12.md).
    # The subprocess was wasting ~150 MB and never doing useful work.
    # The active DeepFace path is `detectors.pool.submit(_bg_deepface, ...)`
    # in _process_frame_full which writes directly to _df_cache.
    import uvicorn
    # v9.7.1: server identity stamp so we can tell if the running process
    # matches the on-disk source. Banner prints PID + content hash; the
    # /health/full endpoint and WS envelope echo it for the UI.
    try:
        import server_identity
        server_identity.init()
    except Exception as _e:
        print(f"[server_identity] init failed: {type(_e).__name__}: {_e}")
        server_identity = None  # type: ignore[assignment]
    print()
    print("  ============================================")
    print(f"  DODO VISION AI v{cfg.APP_VERSION} — 1Hz LOCKED ANALYSIS")
    print("  ============================================")
    if server_identity is not None:
        try:
            print(server_identity.banner_line(cfg.APP_VERSION))
        except Exception:
            pass  # nolint:silent-except banner-line cosmetic
    print(f"  http://{cfg.HOST}:{cfg.PORT}/webcam")
    print(f"  Health:     http://{cfg.HOST}:{cfg.PORT}/health")
    print(f"  Ready:      http://{cfg.HOST}:{cfg.PORT}/ready")
    print(f"  GPU:        ONNX={detectors.ONNX_PROVIDERS}  TF mem-growth={'ON' if getattr(cfg,'TF_GPU_MEMORY_GROWTH',True) else 'OFF'}")
    # v8.9 Step 7: GPU validation with fail-fast — loud warning if no discrete GPU
    try:
        from gpu_enforce import validate_gpu
        _gv = validate_gpu(fail_fast=True)
        if _gv["discrete_gpu_confirmed"]:
            _gpu_extras = []
            if _gv.get("onnx_dml"):
                _dml_did = getattr(cfg, "DML_PROVIDER_OPTIONS", {}).get("device_id", 1)
                _gpu_extras.append(f"DirectML(device_id={_dml_did})")
            if _gv.get("onnx_cuda"):
                _gpu_extras.append("ONNX-CUDA")
            if _gv.get("torch_cuda"):
                _gpu_extras.append(f"PyTorch={_gv.get('torch_device_name','?')}")
            print(f"  GPU valid:  discrete GPU confirmed — active paths: {', '.join(_gpu_extras) or 'none'}")
        else:
            print("  GPU WARN:   NO discrete GPU detected — ML inference will be slow!")
            print(f"              ONNX CUDA={_gv['onnx_cuda']}, ONNX DirectML={_gv.get('onnx_dml')}, TF GPUs={_gv['tf_gpu_count']}, Torch={_gv.get('torch_cuda')}")
    except Exception:
        pass
    print("  Transport:  WebSocket (HTTP fallback)")
    print(f"  Threads:    {cfg.THREAD_POOL_WORKERS} workers (auto-sized to CPU)")
    print("  Cadence:    2Hz full / 30Hz fast (UI auto-starts camera)")
    print(f"  DeepFace:   every {cfg.DEEPFACE_EVERY_N} full frame(s), bg pool, bbox-cache")
    print(f"  Emotion:    {'HSEmotion ONNX' if detectors.HSEMOTION_ENABLED else 'DeepFace'}")
    print(f"  Cache TTL:  {cfg.DF_CACHE_MAX_AGE}s")
    print(f"  Motion:     artifact rejection ON (thresh={cfg.RPPG_MOTION_THRESH})")
    backend = cfg.LLM_BACKEND.upper()
    if cfg.TOGETHER_API_KEY:
        print("  LLM:        Together AI (llama-3.3-70b, FULL prompt)")
    elif cfg.GROQ_API_KEY:
        print("  LLM:        Groq (llama-3.3-70b, compressed prompt)")
    else:
        print("  LLM:        NO API KEY SET — add to .env file")
    print(f"  Context:    rolling window of {cfg.LLM_CONTEXT_WINDOW} analyses")
    print(f"  Faces:      {len(face_registry.list_faces())} registered")
    print(f"  Parquet:    {'ON -> ' + cfg.PARQUET_LOG_DIR if logger.PARQUET_ENABLED else 'OFF'}")
    radar.start()
    rs = radar.get_radar_status()
    if rs.get('connected'):
        print(f"  Radar A:    CONNECTED on {rs.get('port')} (mode={cfg.RADAR_MODE})")
    else:
        print(f"  Radar A:    not found — {rs.get('error') or '(optional)'}")
    radar.start_dual()
    rs_b = radar.get_radar_b_status()
    if rs_b.get('connected'):
        print(f"  Radar B:    CONNECTED on {rs_b.get('port')} — DUAL-RADAR FUSION ACTIVE")
    else:
        print(f"  Radar B:    not found — {rs_b.get('error') or '(plug a second radar + restart)'}")
    # v7.2: watchdog + forensic export routes
    try:
        import radar_watchdog; radar_watchdog.start()
    except Exception as _e:
        print(f"  Radar wd:   not started ({_e})")
    try:
        import forensic_export; forensic_export.attach(app)
        print("  Forensic:   /forensic/export mounted")
    except Exception as _e:
        print(f"  Forensic:   attach failed ({_e})")
    try:
        import perf
        perf.enable_log_mirror()
        perf.attach(app)
        print("  Perf:       /perf mounted, console → logs/dodo_console.log")
    except Exception as _e:
        print(f"  Perf:       attach failed ({_e})")
    oximeter.start()
    os_ = oximeter.get_oximeter_status()
    _oxi_mode = os_.get('mode','').upper() if os_.get('mode') else ''
    _oxi_tag  = f"CONNECTED [{_oxi_mode}] {os_.get('port')}" if os_.get('connected') else 'not found (watchdog retrying)'
    print(f"  Oximeter:   {_oxi_tag}")
    print("  WiFi CSI:   disabled (stub)")
    print("  Auto-save:  sessions/ every 60s")
    print(f"  Endpoints:  /ready, /affect/delta, /sessions/list (v{cfg.APP_VERSION})")
    gsr.start()
    gs_ = gsr.get_gsr_status()
    print(f"  GSR:        {'CONNECTED ' + str(gs_.get('port')) if gs_.get('connected') else 'not found (optional)'}")
    print("  Voice:      continuous monitor available (start via UI)")
    # Step 3: Wire voice -> deception feeding
    def _voice_to_person(voice_result):
        """Callback: feed vocal metrics to primary person state for deception scoring."""
        if person_states:
            primary = next(iter(person_states.values()))
            primary.vocal_f0 = voice_result.get("f0")
            primary.vocal_jitter = voice_result.get("jitter")
            f0_bl = voice_result.get("f0_baseline")
            if f0_bl and f0_bl > 0 and primary.vocal_f0:
                dev = abs(primary.vocal_f0 - f0_bl) / f0_bl
                primary.vocal_stress = min(1.0, dev * 2)
    voice_monitor.set_deception_callback(_voice_to_person)
    print("  Voice->Dec: continuous deception feeding ACTIVE")
    print("  Forensic:   session recording available")
    # Start dual camera (auto-detect 120fps camera)
    if getattr(cfg, 'DUAL_CAMERA_ENABLED', True):
        dual_camera.start(rppg_mod)
        # v9.8.1: arm the stale-frame watchdog so a browser getUserMedia()
        # collision (which steals the cam mid-session and leaves the
        # capture loop reading the same buffered frame forever) auto-
        # recovers within WATCHDOG_STALL_S seconds. POST /dual_camera/restart
        # also exposes manual recovery.
        try:
            dual_camera.start_watchdog()
        except Exception as _wd_err:
            print(f"  Dual Cam:   watchdog start failed ({_wd_err!r})")
        time.sleep(1)  # give it a moment to probe cameras
        ds = dual_camera.get_status()
        if ds.get("active"):
            print(f"  Dual Cam:   120fps ACTIVE ({ds.get('fps', '?')} fps)")
        else:
            print("  Dual Cam:   no high-fps camera found (single camera mode)")
    print()
    # v9.5.x — TLS support (DODO_TLS_CERT + DODO_TLS_KEY, or DODO_TLS_SELFSIGN=1).
    try:
        import tls as _tls
        _tls_kwargs = _tls.uvicorn_kwargs()
    except Exception:
        _tls_kwargs = {}
    if _tls_kwargs:
        print(f"  TLS:        ENABLED ({'self-signed' if not os.environ.get('DODO_TLS_CERT') else 'operator-provided'}) → https://")
    uvicorn.run(app, host=cfg.HOST, port=cfg.PORT,
                log_level="warning", access_log=False,
                **_tls_kwargs)

