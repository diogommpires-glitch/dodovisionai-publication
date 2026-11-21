"""
DODO VISION AI v7.4 — TPRO-NET style EEG Differential Entropy features

Reference
---------
TPRO-NET (Scientific Reports, 2024, s41598-024-62990-4) — EEG emotion
recognition using differential entropy (DE) per band (θ/α/β/γ) as input
to a transformer. We don't ship the transformer here (keeps CPU cost
bounded), but we do expose the DE feature extractor so the rest of the
pipeline (deception_gate, forensic bundle, truth engine prompts) can
ingest a consistent per-band arousal vector.

DE(x) for a Gaussian process is 0.5 * log(2 * pi * e * var(x)).

Bands (Hz):
    delta : 1–4    (not used in TPRO-NET; included for completeness)
    theta : 4–8
    alpha : 8–13
    beta  : 13–30
    gamma : 30–45
"""
from __future__ import annotations
import math
from typing import Dict, Sequence, Optional

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _fft_power(x: Sequence[float], fs: float) -> tuple:
    try:
        import numpy as np
    except ImportError:
        return None, None
    arr = np.asarray(x, dtype=float)
    if arr.size < 16:
        return None, None
    arr = arr - arr.mean()
    # Hann window to cut leakage
    w = np.hanning(arr.size)
    X = np.fft.rfft(arr * w)
    psd = (np.abs(X) ** 2) / (np.sum(w ** 2) * fs)
    freqs = np.fft.rfftfreq(arr.size, 1.0 / fs)
    return freqs, psd


def band_power(x: Sequence[float], fs: float, lo: float, hi: float) -> Optional[float]:
    freqs, psd = _fft_power(x, fs)
    if freqs is None:
        return None
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return None
    return float(psd[mask].mean())


def differential_entropy(x: Sequence[float], fs: float,
                         band: tuple = (4.0, 45.0)) -> Optional[float]:
    """DE in a single band. Assumes x is already bandpassed OR uses PSD."""
    p = band_power(x, fs, *band)
    if p is None or p <= 0:
        return None
    # DE for zero-mean Gaussian with variance ≈ band power * bandwidth
    var = p * (band[1] - band[0])
    if var <= 1e-12:
        return None
    return 0.5 * math.log(2.0 * math.pi * math.e * var)


def de_vector(x: Sequence[float], fs: float) -> Dict[str, Optional[float]]:
    """Return a {band_name: DE} vector for all 5 bands."""
    return {name: differential_entropy(x, fs, (lo, hi))
            for name, (lo, hi) in BANDS.items()}


def arousal_index(de: Dict[str, Optional[float]]) -> Optional[float]:
    """(beta + gamma) / (theta + alpha) — classic arousal proxy from TPRO-NET.

    High-arousal states (stress, cognitive load) shift power upward.
    Returns None if any relevant band is missing.
    """
    num = _safe_sum(de.get("beta"), de.get("gamma"))
    den = _safe_sum(de.get("theta"), de.get("alpha"))
    if num is None or den is None or den == 0:
        return None
    return float(num / den)


def _safe_sum(*vals) -> Optional[float]:
    vs = [v for v in vals if v is not None]
    return sum(vs) if vs else None
