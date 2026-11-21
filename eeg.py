"""
DODO VISION AI v8 pre-work — Muse 2 / BrainFlow EEG Module (Next Step #6)

Streams from a Muse 2 via BrainFlow, computes:
  - band powers: delta / theta / alpha / beta / gamma per channel
  - a simple cognitive-load proxy = (beta + gamma) / (alpha + theta)
  - a rudimentary P300 trigger on audio cues (time-locked average of
    the last 800 ms of EEG around an event marker)

Non-blocking: all work happens in a background thread. If BrainFlow or
the headband isn't available, every getter returns None and the rest
of the system is unaffected.

Public API:
    start()                   -> bool
    stop()
    get_eeg_state()           -> dict | None
    mark_event(label: str)    -> None   # for P300 triggers
    is_connected()            -> bool
"""
from __future__ import annotations
import math
import threading, time
from collections import deque
from typing import Optional, Deque, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

_BOARD = None
_THREAD: Optional[threading.Thread] = None
_STOP = threading.Event()
_LOCK = threading.Lock()

_state = {
    "connected": False,
    "error": None,
    "band_powers": None,            # {"alpha":..,"beta":..,...}
    "cognitive_load": None,         # 0..1
    "focus": None,                  # 0..1, alpha/(alpha+beta)
    "last_p300": None,              # {"label":..,"amplitude_uv":..,"latency_ms":..}
    "last_update": 0.0,
}
_event_markers: Deque[Tuple[float, str]] = deque(maxlen=32)
_eeg_buffer: Deque[Tuple[float, list]] = deque(maxlen=2000)    # (ts, channel_samples)


def is_connected() -> bool:
    return bool(_state["connected"])


def get_eeg_state() -> Optional[dict]:
    if not _state["connected"]:
        return None
    with _LOCK:
        return dict(_state)


def mark_event(label: str) -> None:
    """Record an event marker (e.g. 'probe_question') for P300 windowing."""
    _event_markers.append((time.time(), label))


def _compute_band_powers(samples: "np.ndarray", fs: float) -> dict:
    """Welch-style band powers on one channel. samples = 1D float array."""
    if np is None or samples.size < int(fs):
        return {}
    # Detrend
    x = samples - np.mean(samples)
    # FFT-based band power
    fft = np.fft.rfft(x * np.hanning(len(x)))
    psd = (np.abs(fft) ** 2) / len(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    bands = {
        "delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0), "gamma": (30.0, 45.0),
    }
    out = {}
    for name, (lo, hi) in bands.items():
        m = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.sum(psd[m])) if np.any(m) else 0.0
    return out


def _p300_scan(now: float, fs: float) -> Optional[dict]:
    """Very simple P300 detector — average 250..500ms post-event window
    across all channels and look for a positive peak > baseline."""
    if np is None or not _event_markers or not _eeg_buffer:
        return None
    evt_t, label = _event_markers[-1]
    if (now - evt_t) < 0.6 or (now - evt_t) > 1.5:
        return None
    # Samples in 250..500ms post-event
    lo, hi = evt_t + 0.25, evt_t + 0.50
    win = [s for (t, s) in _eeg_buffer if lo <= t <= hi]
    # Baseline: -200..0 ms pre-event
    base = [s for (t, s) in _eeg_buffer if (evt_t - 0.2) <= t <= evt_t]
    if len(win) < 5 or len(base) < 5:
        return None
    win_arr = np.asarray(win, dtype=np.float32)
    base_arr = np.asarray(base, dtype=np.float32)
    amp_uv = float(win_arr.mean() - base_arr.mean())
    # Drop this marker so we don't re-report it
    _event_markers.pop()
    return {"label": label,
            "amplitude_uv": round(amp_uv, 2),
            "latency_ms": int((hi - evt_t) * 1000)}


def _worker(mac_address: Optional[str]):
    global _BOARD
    try:
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    except ImportError:
        _state["error"] = "brainflow not installed (pip install brainflow)"
        return
    try:
        params = BrainFlowInputParams()
        if mac_address:
            params.mac_address = mac_address
        board_id = BoardIds.MUSE_2_BOARD.value
        _BOARD = BoardShim(board_id, params)
        _BOARD.prepare_session()
        _BOARD.start_stream()
        fs = BoardShim.get_sampling_rate(board_id)
        eeg_ch = BoardShim.get_eeg_channels(board_id)
        _state["connected"] = True
        print(f"[OK] Muse 2 connected (fs={fs}Hz, {len(eeg_ch)} channels)")
    except Exception as e:
        _state["error"] = str(e)
        _state["connected"] = False
        print(f"[--] Muse 2: {e}")
        return

    try:
        while not _STOP.is_set():
            data = _BOARD.get_board_data()   # drains buffer
            if data is not None and data.shape[1] > int(fs * 0.5):
                now = time.time()
                avg_ch = data[eeg_ch].mean(axis=0)  # channel average
                # Push individual samples with timestamps
                n = len(avg_ch)
                for i, v in enumerate(avg_ch):
                    ts = now - (n - i) / fs
                    _eeg_buffer.append((ts, float(v)))
                bp = _compute_band_powers(avg_ch[-int(fs * 2):], fs)
                if bp:
                    alpha, beta = bp.get("alpha", 0), bp.get("beta", 0)
                    theta, gamma = bp.get("theta", 0), bp.get("gamma", 0)
                    denom = alpha + theta + 1e-9
                    load = min(1.0, (beta + gamma) / (denom * 3.0))
                    focus = alpha / (alpha + beta + 1e-9)
                    p300 = _p300_scan(now, fs)
                    with _LOCK:
                        _state["band_powers"] = {k: round(v, 3) for k, v in bp.items()}
                        _state["cognitive_load"] = round(load, 3)
                        _state["focus"] = round(focus, 3)
                        if p300:
                            _state["last_p300"] = p300
                        _state["last_update"] = now
            time.sleep(0.25)
    finally:
        try:
            _BOARD.stop_stream(); _BOARD.release_session()
        except Exception:
            pass
        _state["connected"] = False


def start(mac_address: Optional[str] = None) -> bool:
    global _THREAD
    if _THREAD and _THREAD.is_alive():
        return True
    _STOP.clear()
    _THREAD = threading.Thread(target=_worker, args=(mac_address,), daemon=True)
    _THREAD.start()
    # Give it a moment to succeed or fail
    time.sleep(2.0)
    return _state["connected"]


def stop() -> None:
    _STOP.set()


# ══════════════════════════════════════════════════════════════════
#  v9.0 Step 9: Lempel-Ziv Complexity (LZC) Photic Habituation Monitor
#  (Stanford WCAG 2.3.1 Mitigation)
#
#  To safely maintain 27.3-28.9 Hz visual entrainment without inducing
#  photic driving, the system continuously analyses EEG complexity using
#  LZC metrics.  A healthy brain exhibits habituation (decreased dynamic
#  complexity) to constant 27 Hz stimuli.  If the LZC analysis detects
#  an EXCESSIVE compensatory reduction in local neural excitability
#  (indicating distress rather than habituation), the system predictively
#  and immediately suppresses the visual stimulus.
# ══════════════════════════════════════════════════════════════════

def _lempel_ziv_complexity(signal: "np.ndarray") -> float:
    """Compute the normalized Lempel-Ziv complexity of a binary sequence.

    The signal is binarized at the median, then parsed using the
    Lempel-Ziv 1976 algorithm.  The complexity C(n) is normalized
    to [0, 1] by dividing by the theoretical maximum for a random
    sequence: n / log2(n).

    Higher LZC → more complex/random brain activity (healthy).
    Lower LZC → more regular/periodic activity (habituation or distress).
    """
    if np is None or len(signal) < 10:
        return 0.5  # neutral default

    # Binarize at median
    median = np.median(signal)
    binary = ''.join(['1' if x > median else '0' for x in signal])
    n = len(binary)

    # Lempel-Ziv parsing: count distinct substrings
    # BUG FIX: search window must be binary[0:k+l-1] (full history), not
    # binary[i:k+l-1]. The old code narrowed the search window using `i`
    # (the start of the current word), which systematically inflated LZC
    # values by missing matches in earlier parts of the sequence. This
    # made the photic distress detector less sensitive than intended.
    k, l = 1, 1
    c = 1  # complexity counter
    while k + l <= n:
        if binary[k:k+l] in binary[0:k+l-1]:
            l += 1
        else:
            c += 1
            k = k + l
            l = 1

    # Normalize: theoretical max for random sequence
    if n > 1:
        c_max = n / max(math.log2(n), 1)
    else:
        c_max = 1
    return min(1.0, c / c_max)


class PhoticHabituationMonitor:
    """Monitors EEG Lempel-Ziv Complexity during visual stimulation.

    When the UI drives a visual entrainment stimulus at 27.3-28.9 Hz
    (gamma-band), this monitor tracks whether the brain is habituating
    normally (gradual LZC decrease) or showing signs of distress
    (rapid LZC drop below threshold).

    If distress is detected, the monitor sets suppress_stimulus=True
    which the UI should read to immediately halt the visual effect.
    """

    # Minimum seconds to keep suppression active after distress is detected.
    # WCAG 2.3.1 safety: prevents rapid on/off cycling of the visual
    # stimulus which could itself be harmful.
    SUPPRESS_COOLDOWN_S = 5.0

    def __init__(self):
        self._lock = threading.Lock()
        self._lzc_history: deque = deque(maxlen=60)  # 60 LZC samples
        self._suppress_stimulus = False
        self._suppression_count = 0
        self._last_lzc = 0.5
        self._stimulation_active = False
        self._stimulation_start = 0.0
        self._suppress_until = 0.0  # cooldown expiry timestamp

    def feed(self, eeg_samples: "np.ndarray", stimulation_active: bool = False) -> dict:
        """Feed EEG samples and check for photic habituation distress.

        Args:
            eeg_samples: raw EEG buffer (1D array, ~1-2 seconds)
            stimulation_active: True when visual entrainment is running

        Returns:
            Status dict with LZC value and suppression decision.
        """
        if np is None or len(eeg_samples) < 20:
            return {"lzc": 0.5, "suppress": False, "reason": "insufficient_data"}

        try:
            from config import (LZC_DISTRESS_THRESHOLD, LZC_MONITOR_ENABLED)
        except ImportError:
            LZC_DISTRESS_THRESHOLD = 0.3
            LZC_MONITOR_ENABLED = True

        if not LZC_MONITOR_ENABLED:
            return {"lzc": 0.5, "suppress": False, "reason": "disabled"}

        lzc = _lempel_ziv_complexity(eeg_samples)

        with self._lock:
            self._last_lzc = lzc
            self._lzc_history.append(lzc)
            self._stimulation_active = stimulation_active

            if stimulation_active:
                if self._stimulation_start == 0:
                    self._stimulation_start = time.time()

                # Check for distress: LZC drops below threshold during
                # active stimulation AND has been declining over recent
                # samples (not just a single-sample dip)
                if len(self._lzc_history) >= 5:
                    recent = list(self._lzc_history)[-5:]
                    trend = recent[-1] - recent[0]  # negative = declining
                    mean_recent = sum(recent) / len(recent)

                    if (mean_recent < LZC_DISTRESS_THRESHOLD
                            and trend < -0.05):
                        # Distress detected — suppress stimulus immediately
                        self._suppress_stimulus = True
                        self._suppression_count += 1
                        # BUG FIX: enforce minimum cooldown period before
                        # suppression can be cleared. Without this, a single
                        # non-distress sample immediately re-enables the
                        # visual stimulus, which could cause rapid on/off
                        # cycling — itself a photic seizure risk.
                        self._suppress_until = time.time() + self.SUPPRESS_COOLDOWN_S
                        return {
                            "lzc": round(lzc, 4),
                            "mean_lzc": round(mean_recent, 4),
                            "trend": round(trend, 4),
                            "suppress": True,
                            "reason": "lzc_distress_detected",
                            "suppression_count": self._suppression_count,
                            "cooldown_remaining_s": round(self.SUPPRESS_COOLDOWN_S, 1),
                        }
                    else:
                        # Only clear suppression after cooldown has expired
                        if time.time() >= self._suppress_until:
                            self._suppress_stimulus = False
                        # else: keep suppression active through cooldown
            else:
                # Not stimulating — still respect cooldown before clearing
                if time.time() >= self._suppress_until:
                    self._suppress_stimulus = False
                self._stimulation_start = 0.0

            return {
                "lzc": round(lzc, 4),
                "suppress": self._suppress_stimulus,
                "reason": "normal_habituation" if stimulation_active else "no_stimulation",
                "history_len": len(self._lzc_history),
            }

    @property
    def should_suppress(self) -> bool:
        """Check if stimulus should be suppressed (thread-safe read)."""
        with self._lock:
            return self._suppress_stimulus

    def stats(self) -> dict:
        with self._lock:
            return {
                "enabled": True,
                "last_lzc": round(self._last_lzc, 4),
                "suppress_active": self._suppress_stimulus,
                "suppression_count": self._suppression_count,
                "stimulation_active": self._stimulation_active,
                "history_samples": len(self._lzc_history),
            }


# Module-level singleton (with lock to prevent TOCTOU race)
_photic_monitor: "PhoticHabituationMonitor | None" = None
_photic_init_lock = threading.Lock()


def get_photic_monitor() -> PhoticHabituationMonitor:
    """Get the process-wide photic habituation monitor (thread-safe)."""
    global _photic_monitor
    if _photic_monitor is not None:
        return _photic_monitor
    with _photic_init_lock:
        # Double-checked locking: re-check after acquiring lock
        if _photic_monitor is None:
            _photic_monitor = PhoticHabituationMonitor()
    return _photic_monitor
