"""
DODO VISION AI v7.4 — Micro-Doppler feature helper (CamWave-Emo style)

Reference
---------
CamWave-Emo (ACM MM 2024, 3767743) fuses mmWave micro-Doppler spectrograms
with camera features for emotion recognition. Our Seeed MR60BHA2 does NOT
expose raw I/Q — it only publishes processed HR/BR and (optionally) heart
and breath phase samples. We therefore cannot compute a true STFT on
range-Doppler returns, but we CAN approximate a phase-derived
"micro-motion spectrogram" from the heart_phase / breath_phase stream
when the sketch publishes it.

This module:
 1. Accepts a running buffer of phase samples (radians).
 2. Computes a short-window FFT spectrogram.
 3. Extracts three scalar features per window:
      * spectral_centroid_hz
      * spectral_entropy
      * total_power
That 3-scalar feature vector is what deception_gate / truth_engine can
ingest, keeping the pipeline compatible even when only HR/BR are
available (all three values return None cleanly).

If v7.3+ sketch is in use `heart_phase` / `breath_phase` are pinned to 0,
so this module will return all-None. Re-enabling phase output in the
sketch (by fixing the upstream library capability probe) unlocks it.
"""
from __future__ import annotations
import math
from collections import deque
from typing import Deque, Dict, Optional


class MicroDopplerExtractor:
    def __init__(self, fs: float = 20.0, window: int = 64) -> None:
        self.fs = float(fs)
        self.window = int(window)
        self._buf: Deque[float] = deque(maxlen=window)

    def push(self, phase: Optional[float]) -> None:
        if phase is None:
            return
        try:
            self._buf.append(float(phase))
        except (TypeError, ValueError):
            return

    def features(self) -> Dict[str, Optional[float]]:
        if len(self._buf) < self.window:
            return {"spectral_centroid_hz": None,
                    "spectral_entropy":    None,
                    "total_power":         None}
        try:
            import numpy as np
        except ImportError:
            return {"spectral_centroid_hz": None,
                    "spectral_entropy":    None,
                    "total_power":         None}
        x = np.asarray(self._buf, dtype=float)
        x = x - x.mean()
        w = np.hanning(x.size)
        X = np.fft.rfft(x * w)
        psd = np.abs(X) ** 2
        freqs = np.fft.rfftfreq(x.size, 1.0 / self.fs)
        total = float(psd.sum())
        if total <= 0:
            return {"spectral_centroid_hz": None,
                    "spectral_entropy":    None,
                    "total_power":         0.0}
        p = psd / total
        centroid = float((freqs * p).sum())
        # Spectral entropy normalized to [0,1]
        ent = -float((p * np.log(p + 1e-12)).sum())
        ent_norm = ent / max(1e-12, math.log(p.size))
        return {
            "spectral_centroid_hz": round(centroid, 3),
            "spectral_entropy":    round(ent_norm, 3),
            "total_power":         round(total, 6),
        }
