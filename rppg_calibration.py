"""
DODO VISION AI v8 — rPPG (Remote Photoplethysmography) Calibration Monitor

Reference
---------
Quer et al. (JAMA Cardiology 2020) establish population reference ranges for
heart rate in seated adults:
    HR:  mean = 72 BPM,  sd = 12 BPM,  95% CI ≈ 49–95

We use this to compute a z-score per live rPPG sample and expose a
calibration_score() that hovers near 1.0 when the stream distribution
matches population norms, and degrades toward 0 under drift (optical
misalignment, motion artifact, skin tone).
"""
from __future__ import annotations
from collections import deque
from typing import Deque, Dict, Optional

HR_REF_MEAN, HR_REF_SD = 72.0, 12.0


class RppgCalibrationMonitor:
    def __init__(self, window: int = 120) -> None:
        self.confirmed_bpm: Deque[float] = deque(maxlen=window)

    def push(self, bpm: Optional[float]) -> None:
        if bpm is not None:
            try:
                self.confirmed_bpm.append(float(bpm))
            except (TypeError, ValueError):
                pass

    def snapshot(self) -> Dict[str, Optional[float]]:
        def _stats(buf: Deque[float]):
            if len(buf) < 5:
                return None, None
            m = sum(buf) / len(buf)
            v = sum((x - m) ** 2 for x in buf) / len(buf)
            return m, v ** 0.5

        mean_bpm, sd_bpm = _stats(self.confirmed_bpm)

        def _z(m, ref_m, ref_sd):
            if m is None:
                return None
            return round((m - ref_m) / ref_sd, 2)

        # Population-match score: 1.0 at |z|=0, 0.0 at |z|=3
        def _score(m, ref_m, ref_sd):
            if m is None:
                return None
            z = abs(m - ref_m) / ref_sd
            return round(max(0.0, 1.0 - z / 3.0), 2)

        return {
            "n": len(self.confirmed_bpm),
            "mean_bpm": round(mean_bpm, 1) if mean_bpm is not None else None,
            "sd_bpm": round(sd_bpm, 2) if sd_bpm is not None else None,
            "z": _z(mean_bpm, HR_REF_MEAN, HR_REF_SD),
            "match_score": _score(mean_bpm, HR_REF_MEAN, HR_REF_SD),
        }


# Shared singleton for convenience
monitor = RppgCalibrationMonitor()
