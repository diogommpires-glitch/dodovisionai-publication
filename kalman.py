"""
DODO VISION AI v8.8 — Kalman Filter

Drop-in replacement for the EMA class in analysis.py. The constant-velocity
1-D Kalman filter tracks the latent value AND its rate of change, which
gives us:
  * Faster response to genuine trends than a fixed-alpha EMA
  * Heavier suppression of single-frame noise spikes
  * Predict-on-missing: returns last estimate when the measurement is None,
    instead of either freezing the EMA or blanking the UI
  * Adaptive gain via the measurement-noise term R, which we can widen when
    rPPG / DeepFace report low confidence

Also exposes a `KalmanDict` helper for per-class emotion smoothing — the
PersonState._emo_ema dict pattern is replaced one-for-one.

API mirrors the old EMA class:
    f = Kalman1D(q=..., r=...)
    f.update(measurement)            # may be None
    f.get()                          # current estimate
    f.reset()
    f.update(m, r_override=R)        # bump measurement noise this step

Pure NumPy (already imported elsewhere). No new dependencies.
"""
from __future__ import annotations
from typing import Dict, Optional


class Kalman1D:
    """Constant-velocity 1-D Kalman filter (state = [value, rate]).

    Parameters
    ----------
    q : float
        Process noise — how much we expect the true value to drift between
        samples. Larger Q -> filter responds faster (and lets in more noise).
    r : float
        Measurement noise — how noisy each observation is. Larger R ->
        filter trusts its own prediction more (smoother but laggier).
    initial : float | None
        Optional seed value. If None, the first non-None measurement seeds.
    """

    __slots__ = ("q", "r", "x", "v", "p", "_seeded")

    def __init__(self, q: float = 1e-3, r: float = 4.0,
                 initial: Optional[float] = None):
        self.q = float(q)
        self.r = float(r)
        # State: x = value, v = rate
        self.x: Optional[float] = float(initial) if initial is not None else None
        self.v: float = 0.0
        # 2x2 covariance, flattened as (P00, P01, P10, P11)
        self.p = [1.0, 0.0, 0.0, 1.0]
        self._seeded = initial is not None

    # ── EMA-compatible surface ──
    def update(self, z: Optional[float], r_override: Optional[float] = None) -> Optional[float]:
        if z is None:
            return self.predict()
        if not self._seeded:
            self.x = float(z)
            self.v = 0.0
            self._seeded = True
            return self.x
        # ── Predict ──
        # x' = x + v ;  v' = v
        x_pred = self.x + self.v
        v_pred = self.v
        # P' = F P F^T + Q  with F = [[1,1],[0,1]] and Q = q * [[1,1],[1,1]]
        p00, p01, p10, p11 = self.p
        # F P
        a00 = p00 + p10; a01 = p01 + p11
        a10 = p10;       a11 = p11
        # F P F^T
        np00 = a00 + a01; np01 = a01
        np10 = a10 + a11; np11 = a11
        # Add Q
        q = self.q
        p00, p01, p10, p11 = np00 + q, np01 + q, np10 + q, np11 + q
        # ── Update ──
        # H = [1, 0] -> innovation = z - x_pred ; S = P00 + R
        r_eff = float(r_override) if r_override is not None else self.r
        s = p00 + r_eff
        if s <= 0:  # numerical guard
            self.x, self.v = x_pred, v_pred
            self.p = [p00, p01, p10, p11]
            return self.x
        k0 = p00 / s
        k1 = p10 / s
        y = float(z) - x_pred
        self.x = x_pred + k0 * y
        self.v = v_pred + k1 * y
        # P := (I - K H) P
        self.p = [
            (1 - k0) * p00,    (1 - k0) * p01,
            -k1 * p00 + p10,   -k1 * p01 + p11,
        ]
        return self.x

    def predict(self) -> Optional[float]:
        """Step the state forward without a measurement."""
        if self.x is None:
            return None
        self.x = self.x + self.v
        # Inflate covariance by Q so trust decays during dropouts
        p00, p01, p10, p11 = self.p
        a00 = p00 + p10; a01 = p01 + p11
        a10 = p10;       a11 = p11
        np00 = a00 + a01; np01 = a01
        np10 = a10 + a11; np11 = a11
        q = self.q
        self.p = [np00 + q, np01 + q, np10 + q, np11 + q]
        return self.x

    def get(self) -> Optional[float]:
        return self.x

    def reset(self) -> None:
        self.x = None
        self.v = 0.0
        self.p = [1.0, 0.0, 0.0, 1.0]
        self._seeded = False


class KalmanDict:
    """Maintains one Kalman1D per key — drop-in replacement for the
    `_emo_ema: Dict[str, float]` pattern in PersonState and analysis."""

    __slots__ = ("_filters", "q", "r")

    def __init__(self, q: float = 5e-3, r: float = 0.05):
        self._filters: Dict[str, Kalman1D] = {}
        self.q = q
        self.r = r

    def update(self, scores: Dict[str, float], r_override: Optional[float] = None) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in scores.items():
            f = self._filters.get(k)
            if f is None:
                f = Kalman1D(q=self.q, r=self.r)
                self._filters[k] = f
            est = f.update(v, r_override=r_override)
            out[k] = est if est is not None else v
        # Step keys that didn't get an observation this round
        for k, f in self._filters.items():
            if k not in scores:
                f.predict()
        return out

    def get(self) -> Dict[str, float]:
        return {k: (f.get() or 0.0) for k, f in self._filters.items()}

    def reset(self) -> None:
        for f in self._filters.values():
            f.reset()


# ── Recommended tunings (used by analysis.py / person_state.py) ──
# v9.2 (plan step 27): lower Q values on the display smoothers so
# numerical values track real physiological trends while rejecting
# measurement noise more aggressively. Combined with client-side
# SmoothedNumber lerp (ui.py), this gives "buttery" number motion
# instead of the per-frame jitter the user reported.
# HR (60-180 BPM): ~0.5 BPM/sec drift tolerance, ±4 BPM noise per sample
KALMAN_HR_Q       = 0.25   # was 0.5 — tighter smoothing, still tracks trends
KALMAN_HR_R       = 16.0
# Valence / Arousal in [-1, 1]: smooth; ~0.05 / sample noise
KALMAN_AFFECT_Q   = 5e-4
KALMAN_AFFECT_R   = 0.04
# Stress in [0, 1]: already smooth — kept
KALMAN_STRESS_Q   = 1e-4
KALMAN_STRESS_R   = 0.04
# HRV RMSSD (ms): naturally swings, halve Q so display tracks slower
KALMAN_HRV_Q      = 0.02   # was 0.05 — less chase, steadier reading
KALMAN_HRV_R      = 9.0
# Respiratory rate (breaths/min): slow drift, moderate noise
KALMAN_RESP_Q     = 0.03   # was 0.05 — display-grade smoothing
KALMAN_RESP_R     = 4.0
# Per-emotion class probabilities in [0,1]: react fast, suppress flicker
# v9.1.7: near-fully transparent. The downstream EmotionSmoother does the
# actual stabilisation now (α=0.75, 1-frame streak, confidence gates).
# Q / R = 1.0 / 0.05 → steady-state Kalman gain ≈ 0.95, effectively raw
# pass-through. Kept non-zero so reset()/update() semantics still work
# identically to other Kalman smoothers in the pipeline.
KALMAN_EMO_Q      = 1.0
KALMAN_EMO_R      = 0.05
