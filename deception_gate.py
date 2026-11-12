"""
DODO VISION AI v7.1 — Deception Baseline Maturity Gate (Next Step #4)

Wraps a raw continuous-deception score with a confidence envelope so the
UI doesn't display a spurious 0.8 deception level six seconds into
tracking a new subject. Driven by two things:

  1. Baseline maturity  - how much of the 120-sample per-indicator
     window is filled. Below 50% we clamp the reported score and
     surface `confidence="low"`.
  2. Signal quality     - rPPG FFT SNR and face landmark quality are
     multiplied in as a second gate so we don't score deception off
     a half-visible face.

Public API:
    gate(raw_score, person_state, signal_quality) -> dict
       {"score": float, "confidence": "low"|"medium"|"high",
        "maturity": 0..1, "reason": str}
"""
from __future__ import annotations
from typing import Any, Optional

MIN_SAMPLES_FOR_SCORE = 60            # 3 s at 20fps — below this, suppress
MATURE_SAMPLES = 120                  # window fill at which we trust fully
SIGNAL_FLOOR = 0.25                   # below this, suppress regardless


def _estimate_maturity(person_state: Any) -> float:
    """Return window fill fraction in [0,1]."""
    dr = getattr(person_state, "deception_running", None)
    if dr is None:
        return 0.0
    # Supports either a dict-of-deques or a single deque named 'samples'
    try:
        if isinstance(dr, dict):
            lengths = [len(v) for v in dr.values() if hasattr(v, "__len__")]
            if not lengths:
                return 0.0
            return min(1.0, sum(lengths) / (len(lengths) * MATURE_SAMPLES))
        return min(1.0, len(dr) / MATURE_SAMPLES)
    except Exception:
        return 0.0


def gate(raw_score: Optional[float], person_state: Any, signal_quality: Optional[dict] = None) -> dict:
    if raw_score is None:
        return {"score": None, "confidence": "none",
                "maturity": 0.0, "reason": "no_raw_score"}
    maturity = _estimate_maturity(person_state)
    rppg_q = (signal_quality or {}).get("rppg", 0.0)
    face_q = (signal_quality or {}).get("face", 0.0)
    quality = min(1.0, max(0.0, 0.5 * rppg_q + 0.5 * face_q))

    if maturity < (MIN_SAMPLES_FOR_SCORE / MATURE_SAMPLES):
        return {"score": None, "confidence": "low",
                "maturity": maturity, "reason": "baseline_warming_up"}
    if quality < SIGNAL_FLOOR:
        return {"score": None, "confidence": "low",
                "maturity": maturity, "reason": "signal_poor"}

    # Attenuate the raw score by both maturity and signal quality.
    adjusted = float(raw_score) * maturity * (0.5 + 0.5 * quality)
    adjusted = max(0.0, min(1.0, adjusted))
    if maturity >= 0.9 and quality >= 0.7:
        conf = "high"
    elif maturity >= 0.5 and quality >= 0.4:
        conf = "medium"
    else:
        conf = "low"
    return {"score": round(adjusted, 3), "confidence": conf,
            "maturity": round(maturity, 2), "reason": "ok"}
