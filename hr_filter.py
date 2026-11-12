"""
DODO VISION AI v7.4 — Physiological HR/BR stream filter

Purpose
-------
The Seeed MR60BHA2 occasionally emits spurious HR spikes (e.g. 160+ BPM
for a resting subject) when motion couples into the phase channel or
when the radar briefly locks onto a harmonic. Raw rPPG can do the same
under sudden illumination changes.

This module provides a reusable per-stream filter that:

  1. Clamps to a physiologically plausible adult-resting window
     (configurable; defaults 40–180 BPM for HR, 6–32 br/min for BR).
  2. Rejects samples that violate a maximum rate-of-change constraint
     (default ±6 BPM/s — beat-to-beat RR variability can't legitimately
     jump faster than this for a stationary subject).
  3. Maintains a rolling median + IQR buffer and rejects anything outside
     median ± IQR_GATE * IQR (default 3.0 * IQR, i.e. ~Tukey outer fence).
  4. Exposes a simple signal-quality index (SQI) in [0,1] derived from the
     inverse of short-window standard deviation — plugged into hr_fusion
     so that a chattery stream loses weight automatically.

Research anchors
----------------
 * 120GHz mmWave vitals dataset (s41597-026-07016-6) — reference
   ECG/PPG-derived HR distributions for adult subjects at rest sit
   overwhelmingly inside 45–110 BPM; readings past 180 BPM for a seated
   subject are essentially always sensor artifacts.
 * CamWave-Emo (3767743) — uses radar micro-Doppler features fused with
   camera. Their preprocessing stage applies the same rolling-median /
   rate-limit idea to reject single-sample outliers before feature
   extraction.

The filter is stateful. Use one instance per independent stream
(radar A HR, radar B HR, rPPG HR, radar A BR, ...).
"""
from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple


@dataclass
class FilterParams:
    lo: float = 40.0          # hard lower bound (BPM for HR, br/min for BR)
    hi: float = 180.0         # hard upper bound
    max_delta_per_s: float = 6.0   # max physiological rate of change
    window: int = 12          # rolling samples for median/IQR
    iqr_gate: float = 3.0     # reject if |x - median| > iqr_gate * IQR
    min_warmup: int = 4       # need this many accepted samples before outlier-gating kicks in


@dataclass
class _State:
    buf: Deque[Tuple[float, float]] = field(default_factory=deque)  # (timestamp, value)
    last_accepted: Optional[Tuple[float, float]] = None   # (ts, value)
    rejected_bounds: int = 0
    rejected_jump:   int = 0
    rejected_outlier: int = 0
    accepted:        int = 0


class HRFilter:
    """Physiologically-informed single-stream filter.

    Example
    -------
        f = HRFilter(FilterParams(lo=40, hi=180))
        out = f.push(raw_bpm)           # None if rejected
        sqi = f.sqi()                   # 0..1 signal quality index
        stats = f.stats()               # dict of counters
    """

    def __init__(self, params: Optional[FilterParams] = None) -> None:
        self.p = params or FilterParams()
        self.s = _State(buf=deque(maxlen=self.p.window))

    # ── core ──
    def push(self, value: Optional[float], now: Optional[float] = None) -> Optional[float]:
        """Push a raw sample. Returns the accepted value or None if rejected."""
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        t = float(now) if now is not None else time.time()

        # 1) Hard bounds
        if not (self.p.lo <= v <= self.p.hi):
            self.s.rejected_bounds += 1
            return None

        # 2) Rate-of-change gate
        if self.s.last_accepted is not None:
            t_prev, v_prev = self.s.last_accepted
            dt = max(1e-3, t - t_prev)
            if dt < 10.0:  # only apply within a reasonable continuous window
                max_jump = self.p.max_delta_per_s * dt
                if abs(v - v_prev) > max_jump:
                    self.s.rejected_jump += 1
                    return None

        # 3) Rolling median / IQR outer-fence gate
        if len(self.s.buf) >= self.p.min_warmup:
            vals = sorted(x for _, x in self.s.buf)
            n = len(vals)
            med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
            q1 = vals[n // 4]
            q3 = vals[(3 * n) // 4]
            iqr = max(1e-3, q3 - q1)
            if abs(v - med) > self.p.iqr_gate * iqr + 2.0:  # +2 BPM slack
                self.s.rejected_outlier += 1
                return None

        # Accepted
        self.s.buf.append((t, v))
        self.s.last_accepted = (t, v)
        self.s.accepted += 1
        return v

    # ── telemetry ──
    def sqi(self) -> float:
        """Signal-quality index in [0,1]. Higher = steadier stream."""
        if len(self.s.buf) < 3:
            return 0.0
        vals = [v for _, v in self.s.buf]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        # 1 BPM std -> ~0.9, 10 BPM std -> ~0.1
        sqi = 1.0 / (1.0 + (std / 3.0) ** 2)
        return max(0.0, min(1.0, sqi))

    def stats(self) -> dict:
        return {
            "accepted": self.s.accepted,
            "rejected_bounds":  self.s.rejected_bounds,
            "rejected_jump":    self.s.rejected_jump,
            "rejected_outlier": self.s.rejected_outlier,
            "buffer":           len(self.s.buf),
            "sqi":              round(self.sqi(), 3),
        }

    def last(self) -> Optional[float]:
        return self.s.last_accepted[1] if self.s.last_accepted else None

    def reset(self) -> None:
        self.s = _State(buf=deque(maxlen=self.p.window))


# Default presets (research-anchored)
HR_ADULT_RESTING = FilterParams(lo=40.0, hi=180.0, max_delta_per_s=6.0,
                                window=12, iqr_gate=3.0, min_warmup=4)
BR_ADULT_RESTING = FilterParams(lo=6.0,  hi=32.0,  max_delta_per_s=1.0,
                                window=12, iqr_gate=3.0, min_warmup=4)
