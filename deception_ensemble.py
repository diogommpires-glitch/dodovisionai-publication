"""
DODO VISION AI — Calibrated multimodal deception ensemble (v7.5)

Replaces the single-score gate with a proper late-fusion ensemble:

  Modalities    weight   signal_quality source
  ---------     ------   -----------------------
  micro-expr    0.18     face_quality * face_present
  gaze flight   0.15     gaze_conf
  facial asym   0.10     face_quality
  vocal stress  0.22     vad.speech_ratio
  HR reactivity 0.15     fusion_confidence
  skin GSR      0.10     gsr_present * baseline_ready
  HRV / RMSSD   0.10     hrv_valid

Each modality contributes w_i * q_i * score_i / sum(w_i * q_i).
Output is then mapped through a Platt sigmoid for calibration:
    p_deceptive = 1 / (1 + exp(-(A*raw + B)))
A / B are placeholders tuned from forensic_export ground-truth logs;
configurable at runtime by writing models/deception_calibration.json.

The v7.4 deception_gate's maturity & stability attenuation is
preserved by wrapping this function with the existing gate.
"""
from __future__ import annotations
import os
import json
import math
from dataclasses import dataclass, field
from typing import Dict

# Default Platt params (A negative => high raw raises prob)
_DEFAULT_A = 3.5
_DEFAULT_B = -1.5

# Calibration file written by: calibration/write_platt_calibration.py
_CALIB_PATH = os.path.join(os.path.dirname(__file__),
                           "models", "deception_calibration.json")

WEIGHTS: Dict[str, float] = {
    "micro_expr":    0.18,
    "gaze_flight":   0.15,
    "facial_asym":   0.10,
    "vocal_stress":  0.22,
    "hr_reactivity": 0.15,
    "gsr":           0.10,
    "hrv":           0.10,
}


@dataclass
class EnsembleResult:
    probability: float
    raw_score: float
    contributions: Dict[str, float] = field(default_factory=dict)
    total_quality: float = 0.0
    reason: str = ""


def _load_calibration():
    try:
        with open(_CALIB_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        return float(d.get("A", _DEFAULT_A)), float(d.get("B", _DEFAULT_B))
    except Exception:
        return _DEFAULT_A, _DEFAULT_B


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def score(modalities: Dict[str, Dict[str, float]]) -> EnsembleResult:
    """
    modalities = {
      'micro_expr':    {'score': 0-1, 'quality': 0-1},
      'gaze_flight':   {'score': 0-1, 'quality': 0-1},
      ...
    }
    Missing modalities are treated as quality=0 (no contribution).
    """
    contribs = {}
    raw_num = 0.0
    qual_sum = 0.0
    for name, w in WEIGHTS.items():
        m = modalities.get(name) or {}
        s = float(max(0.0, min(1.0, m.get("score", 0.0))))
        q = float(max(0.0, min(1.0, m.get("quality", 0.0))))
        contribs[name] = round(w * q * s, 4)
        raw_num += w * q * s
        qual_sum += w * q

    if qual_sum < 1e-3:
        return EnsembleResult(probability=0.0, raw_score=0.0,
                              contributions=contribs, total_quality=0.0,
                              reason="insufficient_signal_quality")

    raw = raw_num / qual_sum  # quality-normalized score in 0-1
    A, B = _load_calibration()
    prob = _sigmoid(A * raw + B)

    reason = "ok"
    if qual_sum < 0.2:
        reason = "low_quality_ensemble"

    return EnsembleResult(
        probability=round(prob, 3),
        raw_score=round(raw, 3),
        contributions=contribs,
        total_quality=round(qual_sum, 3),
        reason=reason,
    )
