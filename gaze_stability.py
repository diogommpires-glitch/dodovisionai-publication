"""
DODO VISION AI — Gaze / Eye / Head stability (v7.5)

References:
  * Soukupová & Čech 2016 — Eye Aspect Ratio (EAR) for blink detection.
  * Dementhon & Davis 1995 — POSIT for head pose (conceptual).
  * MediaPipe Iris 2020 — sub-pixel iris landmarks 468 / 473.

This module adds:
  - GazeStability: EMA of horizontal/vertical gaze + deadband to stop
    jitter in the UI. Calls the gaze "steady" when the EMA magnitude
    stays below STEADY_BAND for >=N frames.
  - BlinkCounter: proper EAR-based blink with refractory period,
    Double-counts prevented via a 300 ms lockout.
  - PERCLOS: percentage of time eyes are >=80% closed over last 60 s.
    Standard drowsiness metric (FHWA 1998 trucker study).
  - AttentionScore: 0–1 combining steady gaze + open eyes + small head
    yaw/pitch — matches DMS (Driver Monitoring System) SAE J3016 Level 2+.
"""
from __future__ import annotations
import time
from collections import deque

# ── EAR geometry ──
# Eye Aspect Ratio = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
EAR_BLINK_THRESH = 0.21      # closed eye
EAR_OPEN_THRESH = 0.26       # re-open hysteresis
BLINK_MIN_FRAMES = 2
BLINK_REFRACTORY_S = 0.30
PERCLOS_WINDOW_S = 60.0
STEADY_BAND = 0.035          # normalized gaze vector magnitude
STEADY_STREAK = 6            # frames to call it steady
_EMA_ALPHA = 0.35
# v9.1 (optimization step 13): dead-zone on raw gaze before smoothing.
# Sub-pixel jitter in iris landmarks produces gaze values in ±0.01 even
# when the subject is locked onto the camera. Zeroing anything inside
# the dead-zone stops that noise from entering the EMA at all — the
# steady detector converges faster and the cognitive proxies see a
# cleaner microsaccade signal.
_DEAD_ZONE = 0.02


def ear_from_points(p1, p2, p3, p4, p5, p6):
    """All points as (x,y) tuples in normalized coords."""
    def d(a, b): return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5
    num = d(p2, p6) + d(p3, p5)
    den = 2.0 * d(p1, p4) + 1e-9
    return num / den


class BlinkCounter:
    def __init__(self):
        self._eye_closed = False
        self._closed_frames = 0
        self._last_blink_t = 0.0
        self.blinks = 0
        self._closed_history = deque()  # (t, bool) for PERCLOS

    def push(self, ear_left: float, ear_right: float, now: float = None):
        """Returns {'closed': bool, 'blinked_this_frame': bool, 'perclos': float}."""
        if now is None:
            now = time.time()
        ear = (ear_left + ear_right) / 2.0

        closed_now = ear < EAR_BLINK_THRESH
        blinked = False

        if closed_now:
            self._closed_frames += 1
            if not self._eye_closed and self._closed_frames >= BLINK_MIN_FRAMES:
                self._eye_closed = True
        else:
            if ear > EAR_OPEN_THRESH and self._eye_closed:
                # re-open: was this a full blink?
                if (now - self._last_blink_t) >= BLINK_REFRACTORY_S:
                    self.blinks += 1
                    blinked = True
                    self._last_blink_t = now
                self._eye_closed = False
                self._closed_frames = 0

        # PERCLOS maintenance
        self._closed_history.append((now, closed_now))
        cutoff = now - PERCLOS_WINDOW_S
        while self._closed_history and self._closed_history[0][0] < cutoff:
            self._closed_history.popleft()
        n = len(self._closed_history) or 1
        perclos = sum(1 for _, c in self._closed_history if c) / n

        return {
            "closed": self._eye_closed,
            "blinked_this_frame": blinked,
            "ear": round(ear, 3),
            "perclos": round(perclos, 3),
            "total_blinks": self.blinks,
        }

    def reset(self):
        self._eye_closed = False
        self._closed_frames = 0
        self._last_blink_t = 0.0
        self.blinks = 0
        self._closed_history.clear()


class GazeStability:
    def __init__(self):
        self._hx = None
        self._hy = None
        self._streak = 0
        self.steady = False

    def push(self, gaze_h: float, gaze_v: float):
        """Return (steady_bool, smoothed_h, smoothed_v)."""
        # v9.1: apply dead-zone before smoothing so sub-pixel iris noise
        # isn't pumped into the EMA and later re-read as micro-drift.
        if abs(gaze_h) < _DEAD_ZONE:
            gaze_h = 0.0
        if abs(gaze_v) < _DEAD_ZONE:
            gaze_v = 0.0
        self._hx = gaze_h if self._hx is None else _EMA_ALPHA * gaze_h + (1 - _EMA_ALPHA) * self._hx
        self._hy = gaze_v if self._hy is None else _EMA_ALPHA * gaze_v + (1 - _EMA_ALPHA) * self._hy
        mag = (self._hx ** 2 + self._hy ** 2) ** 0.5
        if mag < STEADY_BAND:
            self._streak += 1
            if self._streak >= STEADY_STREAK:
                self.steady = True
        else:
            self._streak = 0
            self.steady = False
        return self.steady, round(self._hx, 4), round(self._hy, 4)

    def reset(self):
        self._hx = None
        self._hy = None
        self._streak = 0
        self.steady = False


def attention_score(steady_gaze: bool, perclos: float,
                    head_yaw: float, head_pitch: float) -> float:
    """Composite attention 0-1. Matches common DMS formulations."""
    s_gaze = 1.0 if steady_gaze else 0.3
    s_eye = 1.0 - min(1.0, perclos * 2.5)          # PERCLOS 0 → 1.0, 0.4+ → 0
    s_head = 1.0 - min(1.0, (abs(head_yaw) + abs(head_pitch)) / 45.0)
    return round(0.4 * s_gaze + 0.35 * s_eye + 0.25 * s_head, 3)
