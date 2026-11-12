"""
DODO VISION AI — Landmark smoothing (v9.1, optimization step 11)

Implements the One-Euro filter from Casiez, Roussel & Vogel, CHI 2012
("1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in
Interactive Systems"). It's a two-stage low-pass filter whose cutoff
frequency rises with the signal's own derivative — fast motions pass
through with minimal lag, but a still face is aggressively smoothed.

Why this over a plain EMA:
  - EMA with α=0.35 introduces visible lag during head turns.
  - EMA with α=0.75 lets per-frame landmark jitter through.
  - One-Euro adapts per landmark, per axis, per frame: high cutoff when
    the point is actually moving, low cutoff when it's stationary.

Used by analysis.build_face_{fast,full} in v9.1 to denoise the 478
MediaPipe face landmarks (3D) before any downstream consumer sees them.

Config (all module-level, tweak in place — no runtime knobs):
  MIN_CUTOFF = 5.0 Hz   — cutoff when motion is near zero
  BETA       = 0.2      — how fast cutoff rises with |velocity|
  D_CUTOFF   = 1.0 Hz   — derivative filter cutoff

v9.1.5: the user wants "every frame perfectly follow the face". Raised
MIN_CUTOFF to 5 Hz (was 2.5) and BETA to 0.2 (was 0.08) so the filter
is effectively transparent at any real head motion — only sub-pixel
noise at complete rest gets suppressed. On a 30 fps stream this means
<0.5 frame of lag during real motion, which is imperceptible.
"""
from __future__ import annotations
import math
import time
from typing import Optional

MIN_CUTOFF = 5.0
BETA = 0.2
D_CUTOFF = 1.0


def _alpha(cutoff_hz: float, dt_s: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / max(dt_s, 1e-6))


class _Scalar1EuroFilter:
    """Single-axis One-Euro filter. State is last value + last derivative."""
    __slots__ = ("min_cutoff", "beta", "d_cutoff",
                 "_x_prev", "_dx_prev", "_t_prev")

    def __init__(self, min_cutoff: float = MIN_CUTOFF,
                 beta: float = BETA, d_cutoff: float = D_CUTOFF):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    def filter(self, x: float, t: Optional[float] = None) -> float:
        if t is None:
            t = time.monotonic()
        if self._t_prev is None or self._x_prev is None:
            self._x_prev = float(x)
            self._dx_prev = 0.0
            self._t_prev = t
            return float(x)
        dt = max(t - self._t_prev, 1e-6)
        # Derivative, then low-pass the derivative.
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        # Adaptive cutoff on the value itself.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class LandmarkSmoother:
    """Per-face smoother for a flat list of MediaPipe landmark objects with
    `.x`, `.y`, `.z` attributes (or any indexable object). Stores three
    scalar filters per landmark index — 478 landmarks × 3 axes = 1434
    tiny filter structs per face. Memory: ~100 KB per face, cheap.

    Usage:
        sm = LandmarkSmoother()
        smoothed = sm.filter(face_landmarks, t=None)  # returns same shape

    Returns a new list of `_SmoothedPoint` objects so the downstream
    `.x`/`.y`/`.z` access pattern in analysis.py keeps working.
    """

    def __init__(self, min_cutoff: float = MIN_CUTOFF,
                 beta: float = BETA, d_cutoff: float = D_CUTOFF):
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x: list = []
        self._y: list = []
        self._z: list = []

    def _ensure_size(self, n: int) -> None:
        while len(self._x) < n:
            self._x.append(_Scalar1EuroFilter(self._min_cutoff, self._beta, self._d_cutoff))
            self._y.append(_Scalar1EuroFilter(self._min_cutoff, self._beta, self._d_cutoff))
            self._z.append(_Scalar1EuroFilter(self._min_cutoff, self._beta, self._d_cutoff))

    def filter(self, landmarks, t: Optional[float] = None):
        if t is None:
            t = time.monotonic()
        self._ensure_size(len(landmarks))
        out = []
        for i, lm in enumerate(landmarks):
            sx = self._x[i].filter(float(lm.x), t)
            sy = self._y[i].filter(float(lm.y), t)
            try:
                sz = self._z[i].filter(float(lm.z), t)
            except AttributeError:
                sz = 0.0
            out.append(_SmoothedPoint(sx, sy, sz))
        return out

    def reset(self) -> None:
        for f in self._x: f.reset()
        for f in self._y: f.reset()
        for f in self._z: f.reset()


class _SmoothedPoint:
    """Tiny record that quacks like a MediaPipe landmark (`.x`/`.y`/`.z`)."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


# ── Per-face registry so the same smoother persists across frames ──
_per_face: dict = {}


def smooth_landmarks(face_idx: int, landmarks, t: Optional[float] = None):
    """Convenience wrapper: smooth the landmarks for `face_idx`,
    creating a smoother on first sight. Call `reset_face(face_idx)` when
    the tracker re-assigns that slot to a different person."""
    sm = _per_face.get(face_idx)
    if sm is None:
        sm = LandmarkSmoother()
        _per_face[face_idx] = sm
    return sm.filter(landmarks, t)


def reset_face(face_idx: int) -> None:
    sm = _per_face.pop(face_idx, None)
    if sm is not None:
        sm.reset()


def reset_all() -> None:
    for sm in _per_face.values():
        sm.reset()
    _per_face.clear()
