"""
DODO VISION AI v7.4 — MMD-based temporal stability gate

Reference
---------
TDMNN (Multimedia Tools & Applications, 2023, 11571_2023) introduces a
temporal-domain Maximum Mean Discrepancy term to enforce that EEG emotion
representations are stable across consecutive windows. We adapt the idea
as a lightweight "temporal stability" check over any feature vector
stream: if the MMD between the "recent" window and the "now" window
exceeds a threshold, flag the segment as unstable — useful for
down-weighting deception_gate conclusions during regime changes
(subject stood up, coughed, laughed).

Uses a Gaussian RBF kernel. Runs in pure Python + numpy — no torch.
"""
from __future__ import annotations
from collections import deque
from typing import Deque, List, Optional, Sequence


def _rbf_kernel(x, y, sigma: float):
    import numpy as np
    X = np.asarray(x, dtype=float); Y = np.asarray(y, dtype=float)
    xx = np.sum(X * X, axis=1)[:, None]
    yy = np.sum(Y * Y, axis=1)[None, :]
    d2 = xx + yy - 2.0 * X @ Y.T
    return np.exp(-d2 / (2.0 * sigma * sigma))


def mmd2(a: Sequence[Sequence[float]],
         b: Sequence[Sequence[float]],
         sigma: float = 1.0) -> Optional[float]:
    """Biased MMD² between two small sample sets (rows = samples)."""
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        return None
    A = list(a); B = list(b)
    if not A or not B:
        return None
    Kaa = _rbf_kernel(A, A, sigma).mean()
    Kbb = _rbf_kernel(B, B, sigma).mean()
    Kab = _rbf_kernel(A, B, sigma).mean()
    return float(Kaa + Kbb - 2.0 * Kab)


class TemporalStabilityGate:
    """Rolling two-window MMD gate for feature vector streams.

    Push feature vectors one at a time; query `unstable()` for a bool.
    """

    def __init__(self, window: int = 8, sigma: float = 1.0,
                 mmd_threshold: float = 0.05) -> None:
        self.window = window
        self.sigma = sigma
        self.th = mmd_threshold
        self._recent: Deque[List[float]] = deque(maxlen=window)
        self._now:    Deque[List[float]] = deque(maxlen=window)
        self._last_mmd: Optional[float] = None

    def push(self, feat: Sequence[float]) -> None:
        fv = list(float(x) for x in feat)
        # Slide: oldest of _now becomes newest of _recent
        if len(self._now) == self.window:
            self._recent.append(self._now.popleft())
        self._now.append(fv)

    def score(self) -> Optional[float]:
        if len(self._recent) < 2 or len(self._now) < 2:
            return None
        self._last_mmd = mmd2(list(self._recent), list(self._now), self.sigma)
        return self._last_mmd

    def unstable(self) -> bool:
        s = self.score()
        return (s is not None) and (s > self.th)
