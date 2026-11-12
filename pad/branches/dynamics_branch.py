"""
DODO VISION AI v9.4 PAD Step 5 — 3D + blink + gaze + pupil dynamics branch.

Detects flat attacks (print / cutout / screen) via 3D evidence that
photos and screens cannot produce:

  1. Eye-blink rate + timing dynamics (live norm: 8–21 blinks/min,
     refractory ≥ 300 ms, bilateral EAR drop). Paper photos without
     eye-holes = zero blinks; cutouts with eye-holes show stilted
     timing. Refs: Pan et al. ICCV 2007 eyeblink-based anti-spoofing;
     Soukupová & Čech — EAR landmark blink detection.
  2. 3D consistency via nose-tip vs ear motion ratio under small head
     rotations. A 3D head's nose-tip moves more than its ears because
     the nose protrudes; a 2D photo/screen shows uniform translation.
  3. Binocular gaze vergence — live users converge near 60 cm when
     looking at a screen; flat attacks show parallel (infinite-focus)
     rays.
  4. Pupil-size modulation under scene-luminance change. A replay
     attack keeps the replayed pupil size regardless of the operator's
     room lighting; live pupils respond in ~300 ms.

These four cues are fused by a tiny Bayesian network using `pgmpy`
with evidence-propagation; when `pgmpy` unavailable we fall back to a
log-odds sum (same shape as `pad.fusion`).

Refs:
  - Pan, Sun, Wu, Lao, ICCV 2007 — Eyeblink-based Anti-Spoofing.
  - Soukupová, Čech — Real-Time Eye Blink Detection (landmark EAR).
  - IDIAP REPLAY-ATTACK (baseline dataset + protocol).
"""
from __future__ import annotations
import time
import math
from typing import Optional, List, Dict

from pad.types import BranchScore

# ── Per-person rolling state ──
_STATE: Dict[int, Dict] = {}


def reset_all() -> None:
    _STATE.clear()


def _get(pid: int) -> Dict:
    st = _STATE.get(pid)
    if st is None:
        st = {
            "first_seen_t": None,
            "blinks": [],          # list of blink timestamps
            "ear_history": [],     # list of (t, ear_mean)
            "yaw_history": [],     # list of (t, yaw_deg)
            "pitch_history": [],   # API_unq: pose-variance 3D fallback
            "roll_history": [],    # API_unq: pose-variance 3D fallback
            "motion_ratio_history": [],  # nose-displacement / ear-displacement
            "gaze_history": [],    # list of (t, gaze_h, gaze_v)
            "pupil_history": [],   # list of (t, dilation)
            "luminance_history": [],  # list of (t, mean_L)
        }
        _STATE[pid] = st
    return st


def feed_blink(pid: int, ts: float) -> None:
    """Record one blink event timestamp."""
    st = _get(pid)
    st["blinks"].append(ts)
    # Prune to last 3 minutes
    cutoff = ts - 180.0
    st["blinks"] = [b for b in st["blinks"] if b >= cutoff]


def feed_frame(pid: int, ts: float,
               ear_mean: Optional[float] = None,
               yaw_deg: Optional[float] = None,
               pitch_deg: Optional[float] = None,
               roll_deg: Optional[float] = None,
               nose_xy: Optional[tuple] = None,
               left_ear_xy: Optional[tuple] = None,
               right_ear_xy: Optional[tuple] = None,
               gaze_h: Optional[float] = None,
               gaze_v: Optional[float] = None,
               pupil_dilation: Optional[float] = None,
               scene_luminance: Optional[float] = None) -> None:
    st = _get(pid)
    if st["first_seen_t"] is None:
        st["first_seen_t"] = ts
    if ear_mean is not None:
        st["ear_history"].append((ts, ear_mean))
    if yaw_deg is not None:
        st["yaw_history"].append((ts, yaw_deg))
    # API_unq 2026-05-14: pitch + roll history for 3D-consistency fallback
    # when nose/ear landmark coords are not fed (legacy affect_vector-only path).
    if pitch_deg is not None:
        st.setdefault("pitch_history", []).append((ts, pitch_deg))
    if roll_deg is not None:
        st.setdefault("roll_history", []).append((ts, roll_deg))
    # 3D consistency: track nose-tip motion vs ear-tip motion across frames.
    if nose_xy and left_ear_xy and right_ear_xy:
        prev = st.get("prev_landmarks")
        if prev is not None:
            pn, pl, pr = prev
            d_nose = math.hypot(nose_xy[0] - pn[0], nose_xy[1] - pn[1])
            d_ear_l = math.hypot(left_ear_xy[0] - pl[0], left_ear_xy[1] - pl[1])
            d_ear_r = math.hypot(right_ear_xy[0] - pr[0], right_ear_xy[1] - pr[1])
            d_ear = 0.5 * (d_ear_l + d_ear_r)
            if d_ear > 1e-5 and d_nose > 1e-5:
                ratio = d_nose / d_ear
                st["motion_ratio_history"].append((ts, ratio))
        st["prev_landmarks"] = (nose_xy, left_ear_xy, right_ear_xy)
    if gaze_h is not None and gaze_v is not None:
        st["gaze_history"].append((ts, gaze_h, gaze_v))
    if pupil_dilation is not None:
        st["pupil_history"].append((ts, pupil_dilation))
    if scene_luminance is not None:
        st["luminance_history"].append((ts, scene_luminance))
    # Prune everything to last 60 s
    cutoff = ts - 60.0
    for k in ("ear_history", "yaw_history", "pitch_history", "roll_history",
              "motion_ratio_history",
              "gaze_history", "pupil_history", "luminance_history"):
        if k not in st:
            continue
        st[k] = [entry for entry in st[k] if entry[0] >= cutoff]


def _blink_rate_per_min(st: Dict, now: float) -> Optional[float]:
    if not st["blinks"]:
        return None
    window = 60.0
    cutoff = now - window
    recent = [b for b in st["blinks"] if b >= cutoff]
    return len(recent) * (60.0 / window)


def _three_d_consistency(st: Dict) -> Optional[float]:
    """Return a score in [0, 1] where 1.0 means strongly 3D-consistent.
    We only emit when we have enough motion to be informative because a
    stationary head can't disprove a 2D photo either way.

    Primary path (motion_ratio_history): nose-tip motion vs ear-tip motion
    ratio. Requires nose/ear landmark coords fed via feed_frame(nose_xy=...,
    left_ear_xy=..., right_ear_xy=...).

     API_unq 2026-05-14 fallback (pose-variance): when the nose/ear motion-ratio
    path is not yet informative but yaw/pitch/roll history
    IS available, derive a 3D-consistency score from cross-axis pose
    variance. A real 3D head wobbles in pitch AND roll while yawing
    (involuntary micro-movements, breathing-driven head sway, gravity
    response) -- a 2D photo shows yaw "variation" only when the photo
    itself shakes, in which case all three axes are perfectly correlated
    (rigid translation). Decorrelation across pose axes = 3D evidence.
    """
    # Primary: nose/ear motion-ratio path (preferred when fed)
    if len(st.get("motion_ratio_history", [])) >= 10 and len(st["yaw_history"]) >= 10:
        yaws = [y for (_, y) in st["yaw_history"]]
        if max(yaws) - min(yaws) >= 3.0:
            ratios = [r for (_, r) in st["motion_ratio_history"]]
            import statistics
            mu = statistics.fmean(ratios)
            # Map mu near 1.0 → 0.0 (flat); mu > 1.3 → 1.0 (3D)
            return max(0.0, min(1.0, (mu - 1.0) / 0.5))

    # API_unq 2026-05-14 fallback: pose-variance proxy
    yaws = [y for (_, y) in st.get("yaw_history", [])]
    pitches = [p for (_, p) in st.get("pitch_history", [])]
    rolls = [r for (_, r) in st.get("roll_history", [])]
    if len(yaws) < 20 or len(pitches) < 20 or len(rolls) < 20:
        return None
    yaw_range = max(yaws) - min(yaws)
    pitch_range = max(pitches) - min(pitches)
    roll_range = max(rolls) - min(rolls)
    # Need at least SOME yaw motion to evaluate
    if yaw_range < 1.5:
        return None
    # Real 3D head: pitch_range + roll_range typically >= 0.30 * yaw_range
    # (natural micro-movements couple across axes via neck mechanics).
    # 2D photo on a stand: pitch + roll near zero regardless of yaw motion.
    # Synthetic / replay attack: often all three correlate near-perfectly.
    cross_axis = (pitch_range + roll_range) / max(yaw_range, 1e-3)
    # cross_axis < 0.10 → very flat (likely photo)       → score ~0.05
    # cross_axis 0.10-0.30 → mild 3D                     → score ~0.4-0.6
    # cross_axis >= 0.30   → strong 3D wobble            → score ~0.85
    score = min(1.0, max(0.0, (cross_axis - 0.05) / 0.40))
    return score


def _vergence_score(st: Dict) -> Optional[float]:
    """Live users look at a nearby screen; their gaze vectors converge
    rather than run parallel. Without true stereo we approximate via
    the variance-of-difference between left/right gaze — this is
    noisy. Reliability returned alongside.

    v9.8.1 (2026-05-12, 2nd pass): operator's actual gaze tuple read
    as h=0.01, v=-1.08. Horizontal is correctly ~0 (looking at
    camera), but vertical is OUTSIDE the documented [-1, 1] range:
    the iris-landmark vertical position has blown up because:
      (a) Operator wears glasses; MediaPipe's iris landmarks (468 +
          473) get pulled toward the lens-frame edge rather than the
          actual iris under it.
      (b) Camera sits ABOVE the laptop screen; with the operator's
          natural head position looking AT the screen the iris is
          near the bottom of the eye in the frame, which the
          normalised eye_gaze() helper saturates past ±1.

    The vertical component of the magnitude is therefore garbage on
    this hardware configuration. Use horizontal-only: a live operator
    keeps |h| near 0 while a replay/print attack (where the recorded
    subject was looking somewhere else) will have non-zero |h|.

    Keep the magnitude-style API in case a future iris estimator
    fixes the vertical bias -- comment out only the v contribution.
    """
    if len(st["gaze_history"]) < 20:
        return None
    # v9.8.1-2nd-pass: |h| only. See docstring for the v=-1.08 bug
    # that pushed magnitude past every reasonable threshold on the
    # operator's glasses-on-laptop setup.
    # API_unq 2026-05-14 (glasses-aware): use trimmed-median + variance
    # gate. The mean-of-|h| was getting spiked by occasional iris-on-
    # lens-frame outliers (glasses make MediaPipe's iris landmarks
    # snap to the spectacle frame for 1-2 frames at a time). Trimmed
    # median rejects those. Also gate on plausible variance: if
    # variance is near zero AND mean is high, that's a STATIC off-
    # center reading (real subject calmly looking off-screen, NOT a
    # replay).
    mags = [abs(h) for (_, h, v) in list(st["gaze_history"])[-60:]]
    if len(mags) < 10:
        return None
    mags_sorted = sorted(mags)
    # Trim top + bottom 15% (drops iris-on-frame outliers)
    n = len(mags_sorted)
    lo = int(0.15 * n)
    hi = n - lo
    trimmed = mags_sorted[lo:hi] if hi > lo else mags_sorted
    if not trimmed:
        return None
    # Use median of trimmed (more robust than mean)
    median_h = trimmed[len(trimmed) // 2]
    # Variance gate: a real attentive subject shows |h| in ~0.02-0.15
    # with non-trivial micro-saccade variance. A static photo / replay
    # static scene shows near-zero variance (frozen). A scanning replay
    # shows large variance (>0.20).
    mu_t = sum(trimmed) / len(trimmed)
    var_t = sum((x - mu_t) ** 2 for x in trimmed) / max(len(trimmed) - 1, 1)
    sd_t = math.sqrt(var_t)

    # Decision logic (glasses-aware):
    #   median <= 0.20 AND sd > 0.01 → centered + natural saccade → 0.9 LIVE
    #   median <= 0.20 AND sd <= 0.01 → suspiciously frozen → 0.3 mild-spoof
    #   median > 0.30 AND sd > 0.10 → scanning replay → 0.15 SPOOF
    #   median > 0.40 AND sd <= 0.05 → static off-center (real) → 0.6 mild-LIVE
    #   otherwise → 0.5 inconclusive
    if median_h <= 0.20:
        return 0.9 if sd_t > 0.01 else 0.3
    if median_h > 0.30 and sd_t > 0.10:
        return 0.15  # scanning replay (covers genuine and high-amplitude scan)
    if median_h > 0.40 and sd_t <= 0.05:
        return 0.6   # static off-center (real subject; lens-frame bias)
    return 0.5


def _pupil_luminance_coupling(st: Dict) -> Optional[float]:
    """Live pupils dilate when luminance drops (ms–s lag). Replays keep
    the replayed pupil regardless. We compute Pearson correlation
    between luminance drops and pupil-size rises over the last 15 s.
    """
    if len(st["pupil_history"]) < 20 or len(st["luminance_history"]) < 20:
        return None
    # Interpolate luminance to pupil timestamps
    pupil = st["pupil_history"][-30:]
    lum = st["luminance_history"][-30:]
    if not pupil or not lum:
        return None
    lum_ts = [t for (t, _) in lum]
    lum_v = [v for (_, v) in lum]
    # Build interpolation (linear)
    def interp(t):
        if t <= lum_ts[0]:
            return lum_v[0]
        if t >= lum_ts[-1]:
            return lum_v[-1]
        for i in range(1, len(lum_ts)):
            if t <= lum_ts[i]:
                t0, t1 = lum_ts[i - 1], lum_ts[i]
                v0, v1 = lum_v[i - 1], lum_v[i]
                u = (t - t0) / max(t1 - t0, 1e-6)
                return v0 + u * (v1 - v0)
        return lum_v[-1]
    paired = [(interp(t), p) for (t, p) in pupil]
    n = len(paired)
    if n < 5:
        return None
    sx = sum(a for a, _ in paired)
    sy = sum(b for _, b in paired)
    mx = sx / n
    my = sy / n
    num = sum((a - mx) * (b - my) for a, b in paired)
    dx = math.sqrt(sum((a - mx) ** 2 for a, _ in paired) + 1e-9)
    dy = math.sqrt(sum((b - my) ** 2 for _, b in paired) + 1e-9)
    r = num / (dx * dy) if (dx > 0 and dy > 0) else 0.0
    # Live expect negative correlation (luminance up → pupil down).
    # Map r ≈ -0.4 → 0.9 (very live); r ≈ 0 → 0.5; r ≈ +0.4 → 0.1.
    return max(0.0, min(1.0, 0.5 - r))


def _log_odds_fuse(features: Dict[str, float]) -> float:
    """Fallback fusion when pgmpy isn't available. Emit p_attack."""
    logit = 0.0
    weights = {"blink": 1.4, "three_d": 1.6, "vergence": 0.7, "pupil": 0.9}
    for name, w in weights.items():
        v = features.get(name)
        if v is None:
            continue
        v = min(max(v, 1e-3), 1 - 1e-3)
        logit -= w * math.log(v / (1 - v))
    p_attack = 1.0 / (1.0 + math.exp(-logit))
    return p_attack


# ── v9.4.1: Real pgmpy Bayesian network ────────────────────────────
# DAG:    blink  ──┐
#              three_d  ──┼──► liveness  (binary: live / attack)
#              vergence  ──┤
#              pupil_luminance ──┘
#
# Each evidence node is discretized to {low, mid, high} after a two-
# threshold cut (0.33 / 0.66). CPDs are parameterized from priors: live
# data is expected to produce "high" on every evidence node with prob.
# ~0.75; attacks flip those probabilities. Inference is via
# VariableElimination.P(liveness = attack | evidence).
#
# A 60 s warm-up grace window is inherited from the outer scorer:
# this function is only called once at least ONE evidence node is
# present. When fewer than two are present the BN is over-determined by
# priors alone, so we degrade to log-odds to preserve calibration.
_BN_MODEL = None
_BN_INFER = None
_BN_IMPORT_FAILED = False


def _discretize(v):
    if v is None:
        return None
    if v < 0.34:
        return 0        # low (attack-like)
    if v < 0.67:
        return 1        # mid
    return 2            # high (live-like)


# API_unq 2026-05-13 (P1.6): EM-fit CPDs scaffold.
#
# References:
#   Koller, D. & Friedman, N. (2009). Probabilistic Graphical Models.
#     MIT Press, ch. 17 (parameter learning from data).
#   pgmpy docs: BayesianEstimator / MaximumLikelihoodEstimator / EM.
#
# Hand-tuned CPDs are a reviewer red flag in 2026 unless explicitly
# justified as an interpretability prior with sensitivity analysis.
# This module provides:
#   - `fit_dynamics_bn_from_records(records)`: MLE/EM fit from a
#     labelled records jsonl (path + label='live'|'spoof' + dynamics
#     feature columns). Returns a fitted BayesianNetwork.
#   - `save_fitted_cpds(bn, path)` / `_load_fitted_cpds(path)`:
#     persistence for the queue's u3 phase.
#   - `_lazy_bn()` dispatcher: prefers fitted CPDs from
#     `pad/models/dynamics_bn_cpds.json` when present, falls back to
#     the legacy hand-tuned values (kept verbatim, with explicit
#     docstring tagging as the v9.8.1 hand-tuned interpretability prior).
import os as _os
_DYNAMICS_CPDS_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "models", "dynamics_bn_cpds.json"
)


def fit_dynamics_bn_from_records(records_jsonl_path: str,
                                 estimator: str = "mle",
                                 alpha: float = 1.0) -> Optional[Dict]:
    """Fit BN CPDs from a labelled records jsonl. Returns a dict of
    {variable: 2D-array values} suitable for save/load.

    records_jsonl_path: jsonl with rows containing
        {label: 'live'|'spoof', features: {blink_rate_pm, three_d_score,
         vergence_score, pupil_lum_coupling}}
    estimator: 'mle' (MaximumLikelihoodEstimator) or 'bayesian'
        (BayesianEstimator with Dirichlet pseudo-counts).
    alpha: Dirichlet pseudo-count when estimator='bayesian'.

    Returns None when pgmpy or pandas is unavailable, or when the
    records file is empty. Failure-soft so callers can fall back to
    the hand-tuned CPDs.
    """
    try:
        import pandas as pd
        import json as _json
        from pgmpy.estimators import (
            MaximumLikelihoodEstimator, BayesianEstimator,
        )
        try:
            from pgmpy.models import DiscreteBayesianNetwork as _BN
        except Exception:
            from pgmpy.models import BayesianNetwork as _BN
    except Exception as e:
        print(f"[dynamics BN-fit] missing deps: {e}")
        return None

    rows: List[Dict] = []
    try:
        with open(records_jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except Exception:
                    continue
                label = str(rec.get("label", "")).lower()
                if label not in ("live", "spoof"):
                    continue
                feats = rec.get("features") or {}
                blink_raw = feats.get("blink_rate_pm")
                td_raw = feats.get("three_d_score")
                vg_raw = feats.get("vergence_score")
                pu_raw = feats.get("pupil_lum_coupling")
                if None in (blink_raw, td_raw, vg_raw, pu_raw):
                    continue
                rows.append({
                    "liveness": 1 if label == "live" else 0,
                    "blink": _discretize(blink_raw),
                    "three_d": _discretize(td_raw),
                    "vergence": _discretize(vg_raw),
                    "pupil": _discretize(pu_raw),
                })
    except FileNotFoundError:
        print(f"[dynamics BN-fit] file not found: {records_jsonl_path}")
        return None
    if not rows:
        print(f"[dynamics BN-fit] no usable rows in {records_jsonl_path}")
        return None
    df = pd.DataFrame(rows)
    bn = _BN([
        ("liveness", "blink"),
        ("liveness", "three_d"),
        ("liveness", "vergence"),
        ("liveness", "pupil"),
    ])
    if estimator == "bayesian":
        # v9.8.1-unq-patch (2026-05-14): pgmpy 1.1.2 dropped the
        # `prior_type` / `pseudo_counts` kwargs from `BayesianNetwork.fit()`.
        # Those parameters now live on `BayesianEstimator.estimate_cpd()`
        # per node. We fit each CPD by hand with Dirichlet-alpha smoothing,
        # then attach them to the network.
        est = BayesianEstimator(bn, df)
        cpds_to_add = []
        for node in bn.nodes():
            cpd = est.estimate_cpd(
                node,
                prior_type="dirichlet",
                pseudo_counts=float(alpha),
            )
            cpds_to_add.append(cpd)
        bn.add_cpds(*cpds_to_add)
    else:
        bn.fit(df, estimator=MaximumLikelihoodEstimator)
    cpds_out: Dict[str, list] = {}
    for cpd in bn.get_cpds():
        # cpd.values shape e.g. (3, 2) for [low/mid/high] x [attack/live]
        cpds_out[cpd.variable] = cpd.values.tolist()
    cpds_out["__meta__"] = {  # type: ignore[assignment]
        "n_rows": len(df),
        "n_live": int(df["liveness"].sum()),
        "n_spoof": int((1 - df["liveness"]).sum()),
        "estimator": estimator,
        "alpha": float(alpha),
    }
    return cpds_out


def save_fitted_cpds(cpds: Dict, path: str = _DYNAMICS_CPDS_PATH) -> None:
    """Persist fitted CPDs as JSON for _lazy_bn() to pick up next boot."""
    import json as _json
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        _json.dump(cpds, fh, indent=2)
    invalidate_bn_cache()


def invalidate_bn_cache() -> None:
    """Drop cached pgmpy model so the next score() reloads fitted CPDs."""
    global _BN_MODEL, _BN_INFER
    _BN_MODEL = None
    _BN_INFER = None


def _load_fitted_cpds(path: str = _DYNAMICS_CPDS_PATH) -> Optional[Dict]:
    if not _os.path.exists(path):
        return None
    try:
        import json as _json
        with open(path, "r", encoding="utf-8") as fh:
            return _json.load(fh)
    except Exception:
        return None


def _lazy_bn():
    """Build (once) the pgmpy BayesianNetwork + VariableElimination
    inference engine. Returns (model, infer) or (None, None) when
    pgmpy is unavailable.

    API_unq 2026-05-13 (P1.6): prefer EM-fit CPDs from
    `pad/models/dynamics_bn_cpds.json` when present. Falls back to the
    legacy hand-tuned values below (kept verbatim, explicitly tagged
    as the v9.8.1 interpretability prior so a reviewer auditing the
    pkl chain knows which mode was active).
    """
    global _BN_MODEL, _BN_INFER, _BN_IMPORT_FAILED
    if _BN_MODEL is not None:
        return _BN_MODEL, _BN_INFER
    if _BN_IMPORT_FAILED:
        return None, None
    try:
        # pgmpy ≥ 0.1.26 moved BayesianNetwork to pgmpy.models
        try:
            from pgmpy.models import DiscreteBayesianNetwork as _BayesianNetwork
        except Exception:
            from pgmpy.models import BayesianNetwork as _BayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.inference import VariableElimination
    except Exception:
        _BN_IMPORT_FAILED = True
        return None, None

    model = _BayesianNetwork([
        ("liveness", "blink"),
        ("liveness", "three_d"),
        ("liveness", "vergence"),
        ("liveness", "pupil"),
    ])
    # Prior: 0.5/0.5 (uninformative)
    cpd_live = TabularCPD(variable="liveness", variable_card=2,
                          values=[[0.5], [0.5]])

    # API_unq P1.6: try to load fitted CPDs first
    fitted = _load_fitted_cpds()
    if fitted is not None:
        try:
            def _mk_cpd(var: str, default_vals):
                vals = fitted.get(var, default_vals)
                # Sanity: must be 3x2 for the evidence vars
                import numpy as _np
                arr = _np.asarray(vals, dtype=float)
                if arr.shape == (3, 2):
                    return TabularCPD(variable=var, variable_card=3,
                                      values=arr.tolist(),
                                      evidence=["liveness"], evidence_card=[2])
                return None
            cpd_blink = _mk_cpd("blink", [[0.60, 0.10], [0.25, 0.20], [0.15, 0.70]])
            cpd_three_d = _mk_cpd("three_d", [[0.70, 0.05], [0.20, 0.15], [0.10, 0.80]])
            cpd_verg = _mk_cpd("vergence", [[0.40, 0.20], [0.35, 0.30], [0.25, 0.50]])
            cpd_pupil = _mk_cpd("pupil", [[0.45, 0.15], [0.30, 0.25], [0.25, 0.60]])
            if all(c is not None for c in (cpd_blink, cpd_three_d, cpd_verg, cpd_pupil)):
                model.add_cpds(cpd_live, cpd_blink, cpd_three_d, cpd_verg, cpd_pupil)
                if model.check_model():
                    _BN_MODEL = model
                    _BN_INFER = VariableElimination(model)
                    meta = fitted.get("__meta__", {})
                    print(f"[dynamics BN] loaded EM-fit CPDs ({meta.get('estimator','?')}, "
                          f"n={meta.get('n_rows','?')})")
                    return _BN_MODEL, _BN_INFER
        except Exception as e:
            print(f"[dynamics BN] EM-fit load failed ({e}); falling back to hand-tuned")

    # Legacy v9.8.1 hand-tuned interpretability prior. Each evidence
    # CPD is shape (3, 2): rows = {low, mid, high}; columns = parent
    # live state {0=attack, 1=live}. Strongest bias on blink & three_d
    # (anatomy cues), softer on vergence/pupil (noisy/laggy).
    cpd_blink = TabularCPD(
        variable="blink", variable_card=3,
        values=[[0.60, 0.10],   # low
                [0.25, 0.20],   # mid
                [0.15, 0.70]],  # high
        evidence=["liveness"], evidence_card=[2])
    cpd_three_d = TabularCPD(
        variable="three_d", variable_card=3,
        values=[[0.70, 0.05],
                [0.20, 0.15],
                [0.10, 0.80]],
        evidence=["liveness"], evidence_card=[2])
    cpd_verg = TabularCPD(
        variable="vergence", variable_card=3,
        values=[[0.40, 0.20],
                [0.35, 0.30],
                [0.25, 0.50]],
        evidence=["liveness"], evidence_card=[2])
    cpd_pupil = TabularCPD(
        variable="pupil", variable_card=3,
        values=[[0.45, 0.15],
                [0.30, 0.25],
                [0.25, 0.60]],
        evidence=["liveness"], evidence_card=[2])
    model.add_cpds(cpd_live, cpd_blink, cpd_three_d, cpd_verg, cpd_pupil)
    assert model.check_model()
    _BN_MODEL = model
    _BN_INFER = VariableElimination(model)
    return _BN_MODEL, _BN_INFER


def _pgmpy_fuse(features: Dict[str, float]):
    """Run the Bayesian network inference. Returns (p_attack, method)
    where method is 'bayes' on success, else 'log_odds' (fallback).
    """
    model, infer = _lazy_bn()
    if model is None or infer is None:
        return _log_odds_fuse(features), "log_odds"
    evidence = {}
    for k in ("blink", "three_d", "vergence", "pupil"):
        d = _discretize(features.get(k))
        if d is not None:
            evidence[k] = d
    if len(evidence) < 2:
        return _log_odds_fuse(features), "log_odds"
    try:
        q = infer.query(variables=["liveness"], evidence=evidence,
                        show_progress=False)
        # liveness: index 0 = attack, 1 = live
        probs = q.values
        p_attack = float(probs[0])
        return max(0.0, min(1.0, p_attack)), "bayes"
    except Exception:
        return _log_odds_fuse(features), "log_odds"


def score(pid: int) -> BranchScore:
    t0 = time.monotonic()
    st = _get(pid)
    now = time.time()
    first_seen = st.get("first_seen_t") or now
    session_age = now - first_seen

    features: Dict[str, float] = {}
    # Blink rate — first 45 s we don't penalise absent blinks.
    # v9.8.1 2026-05-12: bumped from 15 s. Operator's
    # blink-history is fed via routes.py:2278 (edge-triggered on
    # av["blinked"] True transition). Glasses + EAR landmark
    # noise on his hardware delays the first detected blink to ~30
    # s. Penalising before 45 s was firing a false PRINT label
    # every fresh session even though the operator was clearly
    # blinking 29 bpm on the client-side counter. Pushing the gate
    # to 45 s gives the server-side blink detector enough warmup
    # to record at least 2-3 edges before the branch starts
    # commenting on rate.
    if session_age >= 45.0:
        br = _blink_rate_per_min(st, now)
        if br is None:
            features["blink"] = 0.1    # no blinks after 45 s = suspicious
        elif br < 2.0:
            features["blink"] = 0.15
        elif 4.0 <= br <= 30.0:
            features["blink"] = 0.9    # normal
        elif br > 40.0:
            features["blink"] = 0.3    # unnatural burst
        else:
            features["blink"] = 0.6

    td = _three_d_consistency(st)
    if td is not None:
        features["three_d"] = td

    vg = _vergence_score(st)
    if vg is not None:
        features["vergence"] = vg

    pu = _pupil_luminance_coupling(st)
    if pu is not None:
        features["pupil"] = pu

    # Fusion: pgmpy Bayesian network if available; else log-odds
    if features:
        p_attack, fusion_method = _pgmpy_fuse(features)
    else:
        # v9.8.1 2026-05-12 (9th-pass deep audit): when NO features are
        # observable (cold session, no landmarks, no radar, no rppg)
        # the branch previously returned p=0.5 with reliability=0.2.
        # That cleared `min_reliability_emit=0.15` so dynamics was
        # voting an actual half-strength UNCERTAIN into fusion every
        # frame at the very moment when no evidence at all exists. The
        # log-odds math under prior=0.30 turned that into a small but
        # consistent pull toward SPOOF (logit(0.5)=0 - logit(0.30)=
        # -0.847 means each weight-0.2 contribution adds +0.17 toward
        # the SPOOF tail). True abstain (reliability=0) drops the
        # branch from `usable` entirely.
        p_attack, fusion_method = 0.5, "none"
        return BranchScore(
            name="dynamics",
            p_attack=0.5,
            reliability=0.0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            meta={
                "session_age_s": round(session_age, 2),
                "n_features": 0.0,
                "fusion_method": "abstained_no_features",
            },
            dominant_class=None,
        )
    # Reliability scales with how many features we could actually compute.
    reliability = min(1.0, 0.2 + 0.2 * len(features))
    # If the session is extremely young (< 5 s) we can't trust most cues.
    if session_age < 5.0:
        reliability *= 0.3

    # API_unq 2026-05-14 (DYNAMICS FALSE-POSITIVE HOTFIX):
    # Operator repeatedly reports dynamics flagging SPOOF on a real face.
    # Root cause: a single noisy feature (e.g. three_d on a glasses-wearer
    # whose head barely moves, or vergence on a wobble) drives p_attack
    # > 0.5 via the BN under a generic prior, while reliability =
    # 0.2 + 0.2*1 = 0.4 -- well above the fusion's
    # `min_reliability_emit=0.15`. So dynamics votes a confident SPOOF
    # at the very moment its evidence is least.
    #
    # Defenses (additive, each independently turns off the false alarm):
    #   (1) require >= 2 observable features before non-trivial p_attack
    #   (2) damp p_attack toward 0.5 during cold session (< 30 s)
    #   (3) when ALL features look spoof-leaning (no >=0.6 live signal),
    #       cap p_attack at 0.65 and halve reliability
    n_feats = len(features)
    n_live_signals = sum(1 for v in features.values() if v >= 0.6)

    if n_feats < 2:
        # single-feature evidence is too noisy to be trusted -- abstain.
        # The branch still APPEARS in the panel (for telemetry) but
        # fusion ignores it (reliability < min_reliability_emit).
        return BranchScore(
            name="dynamics",
            p_attack=round(p_attack, 4),
            reliability=0.0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            meta={
                "session_age_s": round(session_age, 2),
                "n_features": float(n_feats),
                "n_live_signals": float(n_live_signals),
                "fusion_method": fusion_method,
                "abstained_reason": "single_feature_too_noisy",
                **features,
            },
            dominant_class=None,
        )

    if session_age < 30.0:
        # Cold session damping: linearly ramp p_attack from 0.5 (at 5 s)
        # toward the BN output (at 30 s).
        damp = max(0.0, min(1.0, (session_age - 5.0) / 25.0))
        p_attack = 0.5 + damp * (p_attack - 0.5)
        reliability *= max(0.3, damp)
        fusion_method = fusion_method + "+cold_damp"

    if n_live_signals == 0 and p_attack > 0.65:
        # All features look spoof-leaning but no positive live signal.
        # This is the most common false-positive shape on glasses-wearers:
        # three_d low, vergence borderline, no blinks yet. Cap + halve.
        p_attack = 0.65
        reliability *= 0.5
        fusion_method = fusion_method + "+no_live_cap"

    # API_unq 2026-05-14 (HIGH-FIDELITY-REPLAY ASYMMETRIC CLAMP):
    # Operator screenshot (phone-displaying-face replay attack) showed
    # dynamics voting p=0.000 rel=1.00 LIVE. Root cause: dynamics features
    # (pose-variance, vergence, pupil-lum, blink) ALL pass through a
    # video replay unchanged -- the recorded subject IS doing all those
    # things. Dynamics CANNOT distinguish "real face" from "video
    # recording of real face" -- that requires rPPG (chromatic skin
    # pulse) or screen_flash (active ambient-response challenge), NOT
    # behavioural dynamics.
    #
    # Asymmetric clamp:
    #   * dynamics REMAINS informative for SPOOF (multiple features
    #     anti-live = strong evidence of static photo / cutout / mask)
    #   * dynamics CANNOT be confident-LIVE (can be fooled by replay)
    #
    # Cap LIVE-side confidence at p_attack=0.30 and drop reliability
    # to ~0.3 so dynamics nudges (rather than dominates) the LIVE side
    # of fusion. Other branches (radar physical, rppg pulse, screen_flash
    # active) carry the LIVE evidence.
    if p_attack < 0.30:
        p_attack = 0.30
        reliability = min(reliability, 0.30)
        fusion_method = fusion_method + "+asym_live_cap"

    dominant = None
    # v9.8.1 2026-05-12 (12th pass per operator: dynamics PRINT label
    # leaking to attack_class on live subject ~30s in):
    #
    # The 9th-pass fix gated `three_d < 0.25` PRINT on session_age >= 30
    # which was the immediate trigger we observed in live telemetry --
    # operator's glasses defeated the depth-from-shading heuristic so
    # `three_d` consistently came out below 0.25 once the EMA stabilised
    # at t=30s. Tightening:
    #   (a) Bump the three_d threshold from 0.25 -> 0.15 -- we need
    #       MUCH stronger evidence of geometric flatness before
    #       declaring PRINT.
    #   (b) Bump the session_age gate from 30s -> 60s so the EMA has
    #       MUCH more time to either stabilise on a non-flat reading
    #       or pick up genuine geometric variation.
    #   (c) Require a co-occurring suspicious blink signal -- a print
    #       attack is BOTH geometrically flat AND non-blinking; either
    #       feature alone is too noisy on real subjects (glasses,
    #       low-light shadows, etc).
    # Net effect: dynamics PRINT only fires on the (rare) sustained
    # 60+ s window where BOTH heuristics agree, instead of the previous
    # behaviour where a single bad heuristic at 30s triggered the alarm.
    if (features.get("three_d", 1.0) < 0.15
            and features.get("blink", 1.0) < 0.4
            and session_age >= 60.0):
        dominant = "PRINT"
    # v9.8.1 2026-05-12: PRINT-on-no-blinks needs > 60 s session age
    # (was 20 s). Glasses-induced EAR noise + slow first-blink
    # detection consistently caused dynamics to flag PRINT on real
    # subjects ~25-40 s into a session. The fused verdict survives
    # this because radar + patch_depth + deepfake outweigh dynamics,
    # but the "X dynamics SPOOF" row in the PAD AGREEMENT panel is
    # confusing UX. Pushing the dominant_class gate out lets the
    # branch settle into 0.6-0.9 blink scores naturally.
    elif features.get("blink", 1.0) < 0.2 and session_age > 60.0:
        dominant = "PRINT"

    return BranchScore(
        name="dynamics",
        p_attack=round(p_attack, 4),
        reliability=round(reliability, 3),
        latency_ms=(time.monotonic() - t0) * 1000.0,
        meta={
            "session_age_s": round(session_age, 2),
            "n_features": float(len(features)),
            "fusion_method": fusion_method,
            **features,
        },
        dominant_class=dominant,
    )
