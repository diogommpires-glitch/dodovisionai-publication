"""
DODO VISION AI — Cognitive-state proxies (v7.6)

ALL metrics in this module are INDIRECT proxies of cortical / autonomic
state derived from webcam + rPPG signals we already compute. This is NOT
EEG. This is NOT neural recording. The module is inspired by the DARPA
Augmented Cognition program's sensor roster (pupil, HRV, eye tracking,
blink) and the Sano et al. 2018 Frontiers paper showing pupillary-rhythm
harmonic entrainment with EEG spectral indices.

Every public output field is prefixed "proxy_" so downstream code and UI
cannot mistake these for neural measurements.

Seven proxies computed here:
  1. proxy_pupil_spectrum_autonomic   (0.04-0.15 Hz band power)
  2. proxy_pupil_spectrum_cognitive   (0.15-0.40 Hz band power)
  3. proxy_pupil_spectral_entropy     (Shannon entropy of normalized PSD)
  4. proxy_hrv_lf_hf_ratio            (sympathovagal balance)
  5. proxy_microsaccade_rate_per_min  (Engbert-Kliegl workload marker)
  6. proxy_blink_spectrum_dopaminergic(blink-rate spectral centroid)
  7. proxy_p300_analog_latency_ms     (pupil+HR reaction to flagged events)
Plus: proxy_cognitive_load_0_1 composite.

References:
  - Sano Y. et al. 2018. "Infrared Camera-Based Non-contact Measurement
    of Brain Activity From Pupillary Rhythms." Frontiers in Physiology.
  - Joshi & Gold 2020. "Pupil size as a window on neural substrates of
    cognition." Trends in Cognitive Sciences.
  - Privitera C.M. et al. 2010. "Pupil dilation during visual target
    detection." Psychophysiology.
  - Thayer J.F., Lane R.D. 2000. "A model of neurovisceral integration"
    (HRV LF/HF ratio).
  - Engbert R., Kliegl R. 2003. "Microsaccades uncover the orientation
    of covert attention." Vision Research.
  - Jongkees & Colzato 2016. "Spontaneous eye blink rate as predictor
    of dopamine-related cognitive function." Neurosci Biobehav Rev.
"""
from __future__ import annotations
import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Ring buffer windows (seconds) ──
PUPIL_WINDOW_S = 30.0
HRV_WINDOW_S = 60.0
SACCADE_WINDOW_S = 60.0
BLINK_WINDOW_S = 120.0

# Band definitions follow the physiology literature.
BAND_AUTONOMIC = (0.04, 0.15)     # cardiac/respiratory cross-talk
BAND_COGNITIVE = (0.15, 0.40)     # Sano-band (cognitive load)
HRV_LF_BAND = (0.04, 0.15)
HRV_HF_BAND = (0.15, 0.40)

# Micro-saccade: Engbert-Kliegl threshold is ~6 * median velocity over
# the eye-tracking channel. We approximate with normalized-coord jump:
MICROSACCADE_JUMP_NORM = 0.002
MICROSACCADE_MIN_IOI_S = 0.10     # min inter-onset interval

# P300-analog search window (pupil dilation + HR acceleration after a
# flagged attention event). Privitera 2010 window is ~200-800 ms.
P300_WIN_LO_MS = 200
P300_WIN_HI_MS = 800


def _band_power(f: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (f >= lo) & (f < hi)
    if not np.any(mask):
        return 0.0
    # numpy 2.x renamed np.trapz → np.trapezoid. Use getattr so this works
    # on both numpy 1.x (no trapezoid) and numpy 2.x (trapz deprecated).
    _trap = getattr(np, "trapezoid", np.trapz)
    return float(_trap(psd[mask], f[mask]))


def _psd(sig: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    sig = np.asarray(sig, dtype=float)
    if len(sig) < 8 or fs <= 0:
        return np.array([]), np.array([])
    sig = sig - np.mean(sig)
    win = np.hanning(len(sig))
    spec = np.abs(np.fft.rfft(sig * win)) ** 2
    f = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    return f, spec


def _spectral_entropy(psd: np.ndarray) -> float:
    if psd.size == 0:
        return 0.0
    p = psd / (np.sum(psd) + 1e-12)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    h = -float(np.sum(p * np.log2(p)))
    return h / math.log2(p.size) if p.size > 1 else 0.0


class CognitiveStateEstimator:
    """Accumulate time-stamped signals and emit proxy cognitive metrics.

    Usage per frame:
        push_pupil(dilation, t)         # from iris dilation estimate
        push_gaze(h, v, t)              # from eye_gaze output
        push_blink(blinked_bool, t)     # from BlinkCounter
        push_rr_ms(rr_ms)               # from rppg HRV RR intervals
        push_event(t)                   # optional: attention event marker
        snapshot() -> dict              # current proxies
    """

    def __init__(self):
        self._pupil: deque = deque()          # (t, dilation)
        self._gaze: deque = deque()           # (t, h, v)
        self._blinks: deque = deque()         # timestamps of blink onsets
        self._rr_ms: deque = deque(maxlen=512)
        self._events: deque = deque(maxlen=128)
        self._last_saccade_t = 0.0
        self._microsaccade_count = 0

    # ── ingestion ────────────────────────────────────────────────
    def push_pupil(self, dilation: Optional[float], t: Optional[float] = None):
        if dilation is None or not np.isfinite(dilation):
            return
        if t is None:
            t = time.time()
        self._pupil.append((t, float(dilation)))
        self._evict(self._pupil, t - PUPIL_WINDOW_S)

    def push_gaze(self, h: Optional[float], v: Optional[float], t: Optional[float] = None):
        if h is None or v is None:
            return
        if t is None:
            t = time.time()
        # micro-saccade detection
        if self._gaze:
            _, ph, pv = self._gaze[-1]
            dh, dv = float(h) - ph, float(v) - pv
            mag = (dh * dh + dv * dv) ** 0.5
            if mag > MICROSACCADE_JUMP_NORM and (t - self._last_saccade_t) > MICROSACCADE_MIN_IOI_S:
                self._last_saccade_t = t
                self._microsaccade_count += 1
        self._gaze.append((t, float(h), float(v)))
        self._evict(self._gaze, t - SACCADE_WINDOW_S)

    def push_blink(self, blinked: bool, t: Optional[float] = None):
        if not blinked:
            return
        if t is None:
            t = time.time()
        self._blinks.append(t)
        while self._blinks and self._blinks[0] < t - BLINK_WINDOW_S:
            self._blinks.popleft()

    def push_rr_ms(self, rr_ms: Optional[float]):
        if rr_ms is None or not (300.0 <= rr_ms <= 2000.0):
            return
        self._rr_ms.append(float(rr_ms))

    def push_event(self, t: Optional[float] = None):
        """Mark an attention-capturing event (e.g. radar motion, voice onset)."""
        self._events.append(t if t is not None else time.time())

    @staticmethod
    def _evict(dq: deque, cutoff: float):
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    # ── derived proxies ──────────────────────────────────────────
    def _pupil_psd(self) -> Tuple[np.ndarray, np.ndarray, float]:
        if len(self._pupil) < 16:
            return np.array([]), np.array([]), 0.0
        # v9.8 perf: build a single 2-col array and split via .T instead of
        # two list-comprehension passes. One numpy allocation instead of
        # two Python lists + two np.array conversions; meaningful at high
        # pupil-buffer depth (n=512 typical).
        arr = np.asarray(self._pupil, dtype=np.float64)
        ts = arr[:, 0]
        ys = arr[:, 1]
        dt = np.diff(ts)
        if len(dt) == 0 or np.median(dt) <= 0:
            return np.array([]), np.array([]), 0.0
        fs = 1.0 / float(np.median(dt))
        # resample to uniform grid so FFT is meaningful
        t_uniform = np.arange(ts[0], ts[-1], 1.0 / fs)
        if len(t_uniform) < 16:
            return np.array([]), np.array([]), 0.0
        y_uniform = np.interp(t_uniform, ts, ys)
        f, psd = _psd(y_uniform, fs)
        return f, psd, fs

    def _p300_analog_latency(self) -> Optional[float]:
        """Median latency from event → pupil peak within 200–800 ms."""
        if not self._events or len(self._pupil) < 8:
            return None
        # v9.8 perf: single-pass numpy allocation (see _pupil_psd note).
        arr = np.asarray(self._pupil, dtype=np.float64)
        pup_ts = arr[:, 0]
        pup_y = arr[:, 1]
        # Baseline = median over preceding 500 ms
        latencies: List[float] = []
        for ev in list(self._events)[-20:]:
            pre_mask = (pup_ts >= ev - 0.5) & (pup_ts < ev)
            win_mask = (pup_ts >= ev + P300_WIN_LO_MS / 1000.0) & \
                       (pup_ts <= ev + P300_WIN_HI_MS / 1000.0)
            if not np.any(pre_mask) or not np.any(win_mask):
                continue
            baseline = float(np.median(pup_y[pre_mask]))
            win_y = pup_y[win_mask]
            win_t = pup_ts[win_mask]
            # only count if there's a positive deflection
            if win_y.max() - baseline < 1e-3:
                continue
            peak_idx = int(np.argmax(win_y))
            latencies.append((win_t[peak_idx] - ev) * 1000.0)
        if not latencies:
            return None
        return float(np.median(latencies))

    def _hrv_lf_hf(self) -> Optional[float]:
        rr = list(self._rr_ms)
        if len(rr) < 16:
            return None
        # RR time series as a pseudo-uniform signal at 4 Hz (Task Force 1996 convention)
        t_rr = np.cumsum(rr) / 1000.0
        if t_rr[-1] <= t_rr[0]:
            return None
        fs = 4.0
        t_u = np.arange(t_rr[0], t_rr[-1], 1.0 / fs)
        if len(t_u) < 16:
            return None
        y_u = np.interp(t_u, t_rr, rr)
        f, psd = _psd(y_u, fs)
        if f.size == 0:
            return None
        lf = _band_power(f, psd, *HRV_LF_BAND)
        hf = _band_power(f, psd, *HRV_HF_BAND)
        if hf < 1e-9:
            return None
        return round(lf / hf, 3)

    def _blink_spectral_centroid(self) -> Optional[float]:
        if len(self._blinks) < 4:
            return None
        t0 = self._blinks[0]
        times = np.array([t - t0 for t in self._blinks])
        duration = times[-1] - times[0]
        if duration < 10.0:
            return None
        # Build blink-density signal at 4 Hz by histogram
        fs = 4.0
        nb = int(duration * fs) + 1
        sig = np.zeros(nb)
        for t in times:
            idx = int(t * fs)
            if 0 <= idx < nb:
                sig[idx] += 1
        f, psd = _psd(sig, fs)
        if f.size == 0 or psd.sum() < 1e-9:
            return None
        centroid = float(np.sum(f * psd) / np.sum(psd))
        return round(centroid, 4)

    def _blink_rate_per_min(self) -> float:
        if not self._blinks:
            return 0.0
        t_now = self._blinks[-1]
        recent = [t for t in self._blinks if t >= t_now - 60.0]
        return float(len(recent))

    def _microsaccade_rate_per_min(self) -> float:
        if not self._gaze:
            return 0.0
        t_end = self._gaze[-1][0]
        t_start = self._gaze[0][0]
        span = max(1e-3, t_end - t_start)
        return round(self._microsaccade_count * 60.0 / span, 2)

    # ── public snapshot ──────────────────────────────────────────
    def snapshot(self) -> Dict[str, Optional[float]]:
        f, psd, fs = self._pupil_psd()
        if f.size:
            auto_p = _band_power(f, psd, *BAND_AUTONOMIC)
            cog_p = _band_power(f, psd, *BAND_COGNITIVE)
            entropy = _spectral_entropy(psd)
        else:
            auto_p = cog_p = entropy = None

        lf_hf = self._hrv_lf_hf()
        micro = self._microsaccade_rate_per_min()
        blink_rpm = self._blink_rate_per_min()
        blink_c = self._blink_spectral_centroid()
        p300 = self._p300_analog_latency()

        # Composite cognitive load — a transparent weighted sum of
        # *normalized* proxies. Each term is clipped to [0,1]. This is
        # not a neural measurement; it is a load *index*.
        terms = []
        if cog_p is not None and auto_p is not None and (auto_p + cog_p) > 1e-9:
            # higher cognitive-band fraction → higher load
            terms.append(min(1.0, cog_p / (auto_p + cog_p + 1e-9)))
        if lf_hf is not None:
            # LF/HF typically 0.5–5; map to 0–1 via sigmoid around 2.0
            terms.append(1.0 / (1.0 + math.exp(-(lf_hf - 2.0))))
        if micro is not None:
            # 30–120 microsaccades/min is the published awake-attentive band
            terms.append(max(0.0, min(1.0, (micro - 20.0) / 100.0)))
        if blink_rpm > 0:
            # U-shape: very low or very high rates both indicate off-task;
            # 15–20 bpm is the resting-attentive norm. Distance-from-norm.
            terms.append(max(0.0, min(1.0, abs(blink_rpm - 17.5) / 25.0)))
        composite = round(float(np.mean(terms)), 3) if terms else None

        return {
            "proxy_pupil_spectrum_autonomic": round(auto_p, 6) if auto_p is not None else None,
            "proxy_pupil_spectrum_cognitive": round(cog_p, 6) if cog_p is not None else None,
            "proxy_pupil_spectral_entropy": round(entropy, 4) if entropy is not None else None,
            "proxy_pupil_sample_rate_hz": round(fs, 2) if fs else None,
            "proxy_hrv_lf_hf_ratio": lf_hf,
            "proxy_microsaccade_rate_per_min": micro,
            "proxy_blink_rate_per_min": blink_rpm,
            "proxy_blink_spectrum_centroid_hz": blink_c,
            "proxy_p300_analog_latency_ms": round(p300, 1) if p300 is not None else None,
            "proxy_cognitive_load_0_1": composite,
            "disclaimer": (
                "Indirect autonomic/visual proxies of cognitive state. "
                "NOT EEG or neural recording."
            ),
        }

    def reset(self):
        self._pupil.clear()
        self._gaze.clear()
        self._blinks.clear()
        self._rr_ms.clear()
        self._events.clear()
        self._last_saccade_t = 0.0
        self._microsaccade_count = 0


# Module-level singleton for the /cognitive endpoint.
_default = CognitiveStateEstimator()


def estimator() -> CognitiveStateEstimator:
    return _default
