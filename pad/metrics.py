"""
Unified APCER / BPCER / ACER metrics  (v9.8.1 2026-05-12, Audit C3).

Single source of truth for biometric-presentation-attack reporting.

Pre-2026-05-12, four files in this repo each defined their own APCER
helper with subtly different conventions:

    pad/report.py             BPCER = (p > t),   APCER = (p <= t)
    pad/bootstrap_ci.py       preds = (p > t),   APCER = 1 - preds[attack].mean()
    pad/apcer_by_class.py     APCER = (p < t),   BPCER = (p >= t)
    pad/eval/ablation.py      APCER = (p < t),   BPCER = (p >= t)

The boundary inequality differs (`<` vs `<=`, `>` vs `>=`) which shifts
operating points by one sample at the threshold. ISO/IEC 30107-3:2023
§7 specifies the convention:

    A sample is classified as ATTACK iff p_attack  >=  threshold.

So:
    APCER = #(attack classified as live) / #(attack)
          = mean(p_attack[attack] <  threshold)
    BPCER = #(live   classified as attack) / #(live)
          = mean(p_attack[live]   >= threshold)
    ACER  = (APCER + BPCER) / 2

This module provides one helper per call shape. Refactor existing
files to import from here so the four implementations can never
drift again.
"""
from __future__ import annotations
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Re-export attack-class taxonomy for callers (avoid a duplicate constant).
from pad.types import ATTACK_CLASSES


# ─── Canonical primitives ──────────────────────────────────────────


def apcer_bpcer_acer(
    p_attack: Sequence[float] | np.ndarray,
    label: Sequence[str] | np.ndarray,
    threshold: float,
) -> Tuple[float, float, float]:
    """Compute (APCER, BPCER, ACER) at a given threshold.

    Parameters
    ----------
    p_attack: per-sample p_attack score, float in [0, 1].
    label:    matching per-sample label, either "live"/"spoof" strings
              or 0/1 ints (0=live, 1=spoof).
    threshold: decision threshold; a sample classified ATTACK iff
              p_attack >= threshold (ISO/IEC 30107-3:2023 §7).

    Returns
    -------
    (apcer, bpcer, acer) — each NaN-safe (NaN when the relevant class is empty).
    """
    p = np.asarray(p_attack, dtype=np.float64)
    lab = _coerce_labels(label)
    attack = (lab == 1)
    live = (lab == 0)
    if attack.any():
        # APCER = attack classified as live = p_attack < threshold
        apcer = float((p[attack] < threshold).mean())
    else:
        apcer = float("nan")
    if live.any():
        # BPCER = live classified as attack = p_attack >= threshold
        bpcer = float((p[live] >= threshold).mean())
    else:
        bpcer = float("nan")
    if math.isnan(apcer) and math.isnan(bpcer):
        acer = float("nan")
    elif math.isnan(apcer):
        acer = bpcer / 2.0
    elif math.isnan(bpcer):
        acer = apcer / 2.0
    else:
        acer = (apcer + bpcer) / 2.0
    return apcer, bpcer, acer


def apcer_per_class(
    records: Iterable[Mapping],
    threshold: float,
    attack_classes: Sequence[str] = ATTACK_CLASSES,
) -> Tuple[Dict[str, float], float]:
    """Compute per-attack-class APCER and the single BPCER at threshold.

    Each record must have keys ``label`` (one of "live"|"spoof") and
    ``p_attack`` (float in [0,1]); spoof records additionally carry
    ``attack_class`` (one of ATTACK_CLASSES).

    APCER per class uses the ISO strict convention (`p_attack < threshold`).
    """
    recs = list(records)
    live = [r for r in recs if r.get("label") == "live"]
    spoof = [r for r in recs if r.get("label") == "spoof"]

    if live:
        n = len(live)
        # BPCER = live with p_attack >= threshold
        bpcer = sum(1 for r in live if float(r.get("p_attack", 0.0)) >= threshold) / n
    else:
        bpcer = float("nan")

    apcer: Dict[str, float] = {}
    for cls in attack_classes:
        sub = [r for r in spoof if r.get("attack_class") == cls]
        if not sub:
            apcer[cls] = float("nan")
            continue
        # APCER = attack classified as live = p_attack < threshold
        apcer[cls] = sum(1 for r in sub if float(r.get("p_attack", 0.0)) < threshold) / len(sub)
    return apcer, bpcer


def best_threshold_min_acer(
    p_attack: Sequence[float] | np.ndarray,
    label: Sequence[str] | np.ndarray,
    n_steps: int = 199,
) -> Tuple[float, float, float, float]:
    """Sweep thresholds and return (threshold, apcer, bpcer, acer) at
    the minimum-ACER operating point.

    Uses a 1% quantile-bracketed linear grid + the unique score values
    so the boundary case is always tested. Same algorithm as the
    bootstrap and apcer_by_class scripts use; centralised here.
    """
    scores = np.asarray(p_attack, dtype=np.float64)
    if scores.size == 0:
        return 0.5, float("nan"), float("nan"), float("nan")
    lab = _coerce_labels(label)
    lo = float(np.quantile(scores, 0.01))
    hi = float(np.quantile(scores, 0.99))
    lo = max(1e-5, lo); hi = min(1 - 1e-5, hi)
    grid = np.linspace(lo, hi, n_steps)
    # Augment with unique score values for boundary precision.
    grid = np.unique(np.concatenate([grid, np.unique(scores)]))
    best_t, best_a, best_b, best_acer = 0.5, float("nan"), float("nan"), float("inf")
    for t in grid:
        a, b, ac = apcer_bpcer_acer(scores, lab, float(t))
        if math.isnan(ac):
            continue
        if ac < best_acer:
            best_t, best_a, best_b, best_acer = float(t), a, b, ac
    return best_t, best_a, best_b, best_acer


def operating_point_at_bpcer(
    records: Iterable[Mapping],
    target_bpcer: float,
) -> Dict[str, float]:
    """ISO-style operating-point selection: find the largest threshold
    that keeps BPCER <= target_bpcer; report APCER per class and ACER
    at that threshold.

    Replaces the bespoke implementation in pad/report.py.
    """
    recs = list(records)
    if not recs:
        return {"threshold": None, "bpcer": None, "apcer": {},
                "apcer_mean": None, "acer": None}
    ts = sorted({round(float(r.get("p_attack", 0.0)), 4) for r in recs})
    best: Optional[Dict[str, float]] = None
    for t in ts:
        apcer, bpcer = apcer_per_class(recs, t)
        if bpcer <= target_bpcer:
            vals = [v for v in apcer.values() if not math.isnan(v)]
            apcer_mean = float(np.mean(vals)) if vals else float("nan")
            acer = (apcer_mean + bpcer) / 2.0 if not math.isnan(apcer_mean) else float("nan")
            candidate = {
                "threshold": float(t),
                "bpcer": float(bpcer),
                "apcer": apcer,
                "apcer_mean": apcer_mean,
                "acer": acer,
            }
            if best is None or candidate["acer"] < best["acer"]:
                best = candidate
    return best or {"threshold": None, "bpcer": None, "apcer": {},
                    "apcer_mean": None, "acer": None}


def det_curve(
    records: Iterable[Mapping],
    n_steps: int = 51,
) -> Dict[str, list]:
    """Detection-error trade-off curve (BPCER vs mean-APCER) across
    threshold grid."""
    recs = list(records)
    out: Dict[str, list] = {"threshold": [], "bpcer": [], "apcer_mean": [], "acer": []}
    for t in np.linspace(0.0, 1.0, n_steps):
        apcer, bpcer = apcer_per_class(recs, float(t))
        vals = [v for v in apcer.values() if not math.isnan(v)]
        mean_apcer = float(np.mean(vals)) if vals else float("nan")
        acer = (mean_apcer + bpcer) / 2.0 if not math.isnan(mean_apcer) else float("nan")
        out["threshold"].append(float(t))
        out["bpcer"].append(float(bpcer))
        out["apcer_mean"].append(mean_apcer)
        out["acer"].append(acer)
    return out


# ─── helpers ────────────────────────────────────────────────────────


def _coerce_labels(label: Sequence | np.ndarray) -> np.ndarray:
    """Accept either "live"/"spoof" strings or 0/1 ints; return uint8
    array with 0=live, 1=spoof. Unknown values raise so callers can
    catch corrupt data instead of silently mis-classifying."""
    arr = np.asarray(label)
    if arr.dtype.kind in ("U", "S", "O"):
        out = np.empty(arr.shape, dtype=np.uint8)
        for i, v in enumerate(arr.flat):
            s = str(v).lower()
            if s == "live":
                out.flat[i] = 0
            elif s == "spoof":
                out.flat[i] = 1
            else:
                raise ValueError(f"unrecognised label string: {v!r}")
        return out
    # Numeric path: must be 0 or 1.
    return np.asarray(label, dtype=np.uint8)
