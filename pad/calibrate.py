"""pad/calibrate.py
==================

Per-branch probability calibration for the PAD stack.

Raw branch outputs are not uniformly well-calibrated — a rPPG-branch p_attack
of 0.8 does not necessarily mean "80 % of frames with this score are spoofs."
This module fits two canonical calibrators on labeled validation records and
writes them to `pad/models/calibrators.pkl`, keyed by branch name:

    Platt scaling       — logistic regression on the raw p_attack.
                          Parametric, robust under small samples.
    Isotonic regression — non-parametric monotone fit.
                          Better when the miscalibration is non-sigmoidal.

Post-fusion miscalibration is addressed by the learned meta-fuser
(`pad.meta_fuse`), which replaces `pad.fusion.fuse` rather than calibrating
its output. There is intentionally no post-fusion calibrator here.

At runtime, `load_calibrators()` returns a dict the branch modules can use:

    from pad.calibrate import load_calibrators
    cals = load_calibrators()
    if cals:
        raw_p = my_branch_score()
        calibrated_p = cals["rppg"].transform(raw_p)

If `pad/models/calibrators.pkl` is missing (first run, or training hasn't
produced val predictions yet), this module is a no-op and fusion continues
with raw branch outputs. That matches the v9.4 policy: graceful degrade on
missing artefacts.

API_up v9.8 footgun (2026-05-07): when a branch's raw p_attack distribution
is concentrated in a very narrow band (sub-0.01 in [0,1]), both Platt and
isotonic fits can collapse into near-constant functions that map every
input to the marginal class probability. ECE on the training distribution
looks great (constant=base-rate gives perfect bin alignment) but the
calibrator no longer discriminates -- live and spoof get the same output.
`pad.refit_single_branch` now probes the response curve at five anchor
points and rejects candidates with spread < 0.05 to catch this. See
`tests/test_calibrator_collapse_guard.py` for the regression coverage.

CLI:
    # Fit from a JSONL/parquet of records (same schema as pad.report)
    python -m pad.calibrate --from val_records.jsonl

    # Also update thresholds.py to the optimised BPCER=0.01 operating point
    python -m pad.calibrate --from val_records.jsonl --write-thresholds

Input record schema (matches pad.report._load_records):
    {"label": "live" | "spoof",
     "attack_class": one of ATTACK_CLASSES | None,
     "p_attack": float,
     "branches": [{"name": str, "p_attack": float, "reliability": float}, ...]}
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import MODEL_DIR


# ----------------------------------------------------------------------
# calibrators
# ----------------------------------------------------------------------
@dataclass
class PlattCalibrator:
    """Logistic regression on a single score dimension. Fast to fit, 2
    parameters; what sklearn's CalibratedClassifierCV uses under the hood.
    """
    a: float
    b: float
    n: int = 0

    def transform(self, p: float | np.ndarray) -> float | np.ndarray:
        x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
        z = self.a * x + self.b
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def fit(cls, raw: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression
        if len(raw) < 4:
            return cls(a=1.0, b=0.0, n=len(raw))
        x = np.log(np.clip(raw, 1e-6, 1 - 1e-6) /
                   np.clip(1 - raw, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
        lr.fit(x, labels.astype(int))
        return cls(a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0]), n=len(raw))


@dataclass
class TemperatureCalibrator:
    """Single-parameter temperature scaling (Guo et al. 2017, On Calibration
    of Modern Neural Networks). Calibrates a binary score by scaling its
    logit by 1/T, where T is fit to minimize NLL on the training split:

        p_cal = sigmoid(logit(p_raw) / T)

    Why temperature scaling matters in v9.9
    ---------------------------------------
    Platt collapses to a near-constant on narrow input ranges (the v9.8
    incident). Isotonic can have flat plateaus that match the marginal
    class probability. Temperature scaling has a single parameter, is
    *always* monotonic, and cannot collapse to a constant unless T -> inf
    (in which case sigmoid(logit/inf) = sigmoid(0) = 0.5 -- still flat
    but at least with the correct posterior assumption).

    Stored as one float so the pickle is trivially version-stable.
    """
    T: float = 1.0
    n: int = 0

    def transform(self, p: float | np.ndarray) -> float | np.ndarray:
        eps = 1e-6
        p_clip = np.clip(p, eps, 1.0 - eps)
        logit = np.log(p_clip / (1.0 - p_clip))
        z = logit / float(max(self.T, eps))
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def fit(cls, raw: np.ndarray, labels: np.ndarray) -> "TemperatureCalibrator":
        """Fit T by 1-D search over log-T to minimize binary NLL on the
        training data. Bounds [-3, 3] cover the practically useful
        range (T ~ 0.05 .. 20). 200 grid points + a 5-step refinement
        is enough for ECE-stable fits in our experiments."""
        if len(raw) < 4:
            return cls(T=1.0, n=len(raw))
        eps = 1e-6
        p_clip = np.clip(raw.astype(np.float64), eps, 1.0 - eps)
        logit = np.log(p_clip / (1.0 - p_clip))
        y = labels.astype(np.float64)

        def _nll(T_val: float) -> float:
            z = logit / max(float(T_val), eps)
            # numerically-stable BCE via log-sum-exp
            ls = np.logaddexp(0.0, -z)            # log(1 + e^-z)  = -log(sigmoid)
            ls_neg = np.logaddexp(0.0, z)         # log(1 + e^z)   = -log(1-sigmoid)
            return float((y * ls + (1.0 - y) * ls_neg).mean())

        # coarse grid in log-space, then 5-step golden-section refinement
        log_grid = np.linspace(-3.0, 3.0, 200)
        nlls = [(_nll(np.exp(lt)), lt) for lt in log_grid]
        nlls.sort()
        best_lt = nlls[0][1]
        # refine
        lo, hi = best_lt - 0.05, best_lt + 0.05
        for _ in range(5):
            mid_lo = lo + (hi - lo) / 3.0
            mid_hi = hi - (hi - lo) / 3.0
            if _nll(np.exp(mid_lo)) < _nll(np.exp(mid_hi)):
                hi = mid_hi
            else:
                lo = mid_lo
        best_T = float(np.exp((lo + hi) / 2.0))
        return cls(T=best_T, n=len(raw))


@dataclass
class IsotonicCalibrator:
    """Piecewise-constant monotone regression. Stored as the fitted
    breakpoints + values so the pickle survives sklearn version churn.
    """
    x_thresholds: np.ndarray
    y_values: np.ndarray
    n: int = 0

    def transform(self, p: float | np.ndarray) -> float | np.ndarray:
        # np.interp matches sklearn's IsotonicRegression.predict when
        # out_of_bounds='clip' (default for the monotone regression).
        return np.interp(p, self.x_thresholds, self.y_values,
                         left=self.y_values[0],
                         right=self.y_values[-1])

    @classmethod
    def fit(cls, raw: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression
        if len(raw) < 4:
            return cls(x_thresholds=np.array([0.0, 1.0]),
                       y_values=np.array([0.0, 1.0]),
                       n=len(raw))
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw, labels.astype(float))
        xs = np.asarray(iso.X_thresholds_, dtype=np.float32)
        ys = np.asarray(iso.y_thresholds_, dtype=np.float32)
        return cls(x_thresholds=xs, y_values=ys, n=len(raw))


@dataclass
class CalibratorSet:
    """Per-branch calibrators. The choice between Platt and isotonic is
    made per-branch by whichever yields lower ECE on a held-out 20 % split.

    Deprecated `fused` field
    -------------------------
    The `fused` slot used to hold a post-fusion calibrator; the matching
    `transform_fused` method was removed in v9.9 because the meta-fuser
    (`pad.meta_fuse`) replaced the fused-calibrator code path entirely.
    The field is retained ONLY for backward-compat with pickles written
    before the removal -- if you load one of those, `fused` will be a
    sklearn calibrator object that nothing reads at runtime. New fits
    write `fused=None` and the metadata pkl never grows a `__fused__`
    key. Slated for removal in v10.0 after the next pkl rewrite.

    Deprecated `__fused__` metadata key
    -----------------------------------
    Older pickles also carry a `metadata['__fused__']` entry from when
    the fused calibrator was fit. New code doesn't write it. Reading
    it is safe (it just shows up in joblib.load()'s metadata dict).
    """
    per_branch: Dict[str, object]
    fused: Optional[object]   # DEPRECATED: see class docstring; v10.0 removal
    metadata: Dict[str, Dict[str, float]]
    version: int = 1

    def transform_branch(self, name: str, p: float) -> float:
        cal = self.per_branch.get(name)
        return float(cal.transform(p)) if cal is not None else float(p)


# ----------------------------------------------------------------------
# API_unq (P1.1): ReliabilityCalibrator
# ----------------------------------------------------------------------
@dataclass
class ReliabilityCalibrator:
    """Map a raw per-branch quality scalar to P(branch decision correct).

    Per the Phase-2 lit review (compass_artifact 2026-05-12), the
    heterogeneous-reliability problem (Audit C6) is the single largest
    methodological gap in DodoVisionAI fusion: every branch reports a
    "reliability" in [0,1] but the scales mean different things
    (blur quality vs feature count vs time-since-update vs signal SNR).
    The log-opinion-pool weights them as if comparable, which they
    aren't.

    Target: across branches, `calibrated_reliability=0.6` should mean
    the same posterior precision (Brier ~0.24, ECE ~0.05). Method:
    per-branch isotonic regression of raw quality to {0,1} branch-
    correct (i.e. whether the branch's verdict agreed with the ground
    truth label on a dev-set sample).

    References:
      Sensoy, M., Kaplan, L. & Kandemir, M. (2018). Evidential Deep
        Learning to Quantify Classification Uncertainty. NeurIPS 31.
        [Dirichlet vacuity u = K/S is the well-typed target reliability.]
      Lin, X. et al. (2024). Suppress and Rebalance: Towards
        Generalized Multi-Modal Face Anti-Spoofing. CVPR 2024, 211-221.
        [MMDG U-Adapter uses MC-Dropout variance as the suppression mask.]
      Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017).
        Simple and Scalable Predictive Uncertainty Estimation using
        Deep Ensembles. NeurIPS.

    Usage:
        rc = ReliabilityCalibrator(branch="moire")
        rc.fit(raw_quality_dev, branch_correct_dev)
        live_reliability = rc.transform(raw_quality_live)
    """
    branch: str
    # Isotonic-regression knots (filled by .fit())
    x_thresholds: np.ndarray | None = None
    y_values: np.ndarray | None = None
    # Fallback when not yet fit: pass-through with a damping multiplier
    fallback_damp: float = 1.0
    # Diagnostics
    n_fit: int = 0
    brier_pre: float = float("nan")
    brier_post: float = float("nan")
    ece_pre: float = float("nan")
    ece_post: float = float("nan")
    version: int = 1

    def fit(self, raw_quality: np.ndarray, branch_correct: np.ndarray) -> None:
        """Fit isotonic regression on (raw_quality, branch_correct ∈ {0,1}).

        Stores knots; reports pre/post Brier + ECE so callers can gate
        deployment (refuse to deploy if ECE_post > 0.07 per the Phase-2
        acceptance test).
        """
        try:
            from sklearn.isotonic import IsotonicRegression
        except Exception as e:
            raise RuntimeError(f"sklearn required for ReliabilityCalibrator: {e}")
        raw = np.asarray(raw_quality, dtype=np.float64).clip(0.0, 1.0)
        y = np.asarray(branch_correct, dtype=np.float64).clip(0.0, 1.0)
        if raw.size != y.size or raw.size < 8:
            raise ValueError(
                f"ReliabilityCalibrator.fit needs ≥8 paired samples; "
                f"got raw.size={raw.size} y.size={y.size}"
            )
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw, y)
        self.x_thresholds = np.asarray(iso.X_thresholds_, dtype=np.float32)
        self.y_values = np.asarray(iso.y_thresholds_, dtype=np.float32)
        self.n_fit = int(raw.size)
        # Brier + ECE pre/post
        self.brier_pre = float(np.mean((raw - y) ** 2))
        post = iso.predict(raw)
        self.brier_post = float(np.mean((post - y) ** 2))
        self.ece_pre = _ece(raw, y)
        self.ece_post = _ece(post, y)

    def transform(self, raw_quality: float) -> float:
        """Map a single raw quality to a calibrated reliability."""
        x = float(np.clip(raw_quality, 0.0, 1.0))
        if self.x_thresholds is None or self.y_values is None:
            return x * self.fallback_damp
        # Linear-interpolate between knots
        return float(np.interp(x, self.x_thresholds, self.y_values))

    def is_fit(self) -> bool:
        return self.x_thresholds is not None and self.y_values is not None


# Module-level registry: branch_name -> ReliabilityCalibrator.
# Populated by fit-time code in pad.refit_single_branch / queue scripts.
# Branches call `transform_branch_reliability(name, raw)` at score time;
# missing entries return raw passthrough (no behavioural change for
# branches that haven't been calibrated yet — backward compatible).
_RELIABILITY_CALIBRATORS: Dict[str, ReliabilityCalibrator] = {}
_RC_PATH = MODEL_DIR / "reliability_calibrators.pkl"


def register_reliability_calibrator(rc: ReliabilityCalibrator) -> None:
    if not isinstance(rc, ReliabilityCalibrator):
        raise TypeError(f"expected ReliabilityCalibrator, got {type(rc)}")
    _RELIABILITY_CALIBRATORS[rc.branch] = rc


def _autoload_reliability_calibrators() -> int:
    """API_unq P1.1 auto-load: if pad/models/reliability_calibrators.pkl
    exists on disk (produced by scripts/_unq_fit_reliability_calibrators.py),
    load it into the module-level registry so the runtime branches see
    calibrated reliabilities without an explicit register call.

    Returns the number of calibrators loaded. Safe to call multiple times
    (idempotent re-register).
    """
    if not _RC_PATH.exists():
        return 0
    try:
        import joblib as _joblib
        bundle = _joblib.load(_RC_PATH)
        if isinstance(bundle, dict):
            n = 0
            for name, rc in bundle.items():
                if isinstance(rc, ReliabilityCalibrator):
                    _RELIABILITY_CALIBRATORS[name] = rc
                    n += 1
            # API_unq 2026-05-15 e1 fix: emit to stderr, not stdout.
            # Module-import diagnostics polluting stdout broke
            # `diagnose_env.py --json` consumers (test_diagnose_env 6 errors).
            if n > 0:
                import sys as _sys
                print(f"[reliability-cal] auto-loaded {n} calibrators "
                      f"from {_RC_PATH.name}", file=_sys.stderr)
            return n
    except Exception as e:
        import sys as _sys
        print(f"[reliability-cal] auto-load failed: {e}", file=_sys.stderr)
    return 0


# Fire the auto-load at module import so production code paths don't
# need an explicit hook.
_autoload_reliability_calibrators()


def transform_branch_reliability(branch_name: str, raw_quality: float) -> float:
    """Map raw branch quality through the registered ReliabilityCalibrator
    if one exists; otherwise pass through unchanged.

    v9.8.1-unq-patch (2026-05-14): two extra guards added after observing
    that the v3 fit produced near-degenerate isotonic curves for moire and
    audio (single knot, constant output ~0.5 across the entire raw scale).
    The previous implementation inflated `raw=0.0` to ~0.5, which leaked
    silent branches into the fuser (regression: test_case_12).

      1. Structural-zero passthrough: `raw <= 0` means "this branch has no
         signal this frame" (NOT a quality below a calibrator-known threshold).
         Always return 0 -- the fuser will then drop the branch via the
         `min_reliability_emit` filter, restoring the prior-only fallback.

      2. Degenerate-calibrator skip: if the fitted isotonic curve has a
         single knot OR a y-range < 0.05 (effectively a constant function),
         skip the calibrator and pass the raw value through unchanged. The
         calibrator carries no useful re-ranking; better to trust the raw
         signal than to flatten everything to ~0.5.
    """
    raw = float(raw_quality)
    # Guard 1 — structural zero
    if raw <= 0.0:
        return 0.0
    rc = _RELIABILITY_CALIBRATORS.get(branch_name)
    if rc is None:
        return raw
    # Guard 2 — degenerate calibrator
    try:
        if rc.y_values is not None:
            yvs = rc.y_values
            if len(yvs) <= 1 or (float(yvs.max()) - float(yvs.min())) < 0.05:
                return raw
    except Exception:
        pass
    return rc.transform(raw)


def reliability_calibrator_status() -> Dict[str, Dict[str, float]]:
    """Diagnostic snapshot of registered ReliabilityCalibrators."""
    out: Dict[str, Dict[str, float]] = {}
    for name, rc in _RELIABILITY_CALIBRATORS.items():
        out[name] = {
            "fit": float(rc.is_fit()),
            "n_fit": float(rc.n_fit),
            "brier_pre": rc.brier_pre,
            "brier_post": rc.brier_post,
            "ece_pre": rc.ece_pre,
            "ece_post": rc.ece_post,
        }
    return out


# ----------------------------------------------------------------------
# fit
# ----------------------------------------------------------------------
def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    if len(probs) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    N = len(probs)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if not mask.any():
            continue
        acc = float(labels[mask].mean())
        conf = float(probs[mask].mean())
        ece += (mask.sum() / N) * abs(acc - conf)
    return float(ece)


def _pick_best(raw: np.ndarray, labels: np.ndarray) -> Tuple[object, Dict[str, float]]:
    """Fit both calibrators; return whichever yields lower ECE on an
    80/20 internal split. Falls back to identity if we have too little
    data to split at all.
    """
    if len(raw) < 20:
        return (PlattCalibrator(a=1.0, b=0.0, n=len(raw)),
                {"reason": "too_few_samples",
                 "n": float(len(raw)), "ece_before": float("nan"),
                 "ece_after": float("nan")})
    rng = np.random.default_rng(0)
    idx = np.arange(len(raw))
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    tr, val = idx[:cut], idx[cut:]

    ece_before = _ece(raw[val], labels[val])

    platt = PlattCalibrator.fit(raw[tr], labels[tr])
    iso = IsotonicCalibrator.fit(raw[tr], labels[tr])
    # v9.9: temperature scaling as a third candidate. Single-parameter,
    # always monotonic, less prone to collapse than Platt on narrow
    # input ranges -- fills the gap between Platt's parametric form
    # and isotonic's piecewise-constant flexibility.
    temp = TemperatureCalibrator.fit(raw[tr], labels[tr])

    ece_platt = _ece(np.asarray(platt.transform(raw[val])), labels[val])
    ece_iso = _ece(np.asarray(iso.transform(raw[val])), labels[val])
    ece_temp = _ece(np.asarray(temp.transform(raw[val])), labels[val])

    # API_up v9.8/v9.9: collapse-aware three-way selection. A calibrator
    # that maps every input to the marginal class probability has
    # artificially low ECE because constant=base-rate gives perfect bin
    # alignment, but it has zero discrimination. Detect this BEFORE
    # choosing (probing each candidate on raw training data), and
    # disqualify any collapsed candidates. Among healthy candidates,
    # lower validation ECE wins. If all three collapse (genuinely
    # uninformative branch), fall through to the no-signal guardrail.
    def _train_spread(cal):
        out = np.asarray(cal.transform(raw), dtype=np.float32)
        return float(out.max() - out.min())
    candidates = [
        ("isotonic", iso, ece_iso, _train_spread(iso)),
        ("platt", platt, ece_platt, _train_spread(platt)),
        ("temperature", temp, ece_temp, _train_spread(temp)),
    ]
    healthy = [c for c in candidates if c[3] >= 0.05]
    if healthy:
        # Pick lowest-ECE healthy candidate
        choice_name, chosen, ece_chosen, _ = min(healthy, key=lambda c: c[2])
    else:
        # All collapsed: pick the lowest-ECE one anyway; the no-signal
        # guardrail below will replace it with identity if needed
        choice_name, chosen, ece_chosen, _ = min(candidates, key=lambda c: c[2])
    # Record other-ECE values for the metadata audit trail
    other_eces: Dict[str, float] = {f"ece_{n}": float(e)
                                    for n, _, e, _ in candidates
                                    if n != choice_name}

    # No-signal guardrail: fall back to identity if the calibrator either
    # (a) collapses to ~constant output over the ACTUAL training distribution
    #     (indicating the branch had no discriminative signal), AND
    # (b) doesn't materially improve ECE.
    # Probe on the training raw scores themselves -- this avoids the
    # false-positive from calibrators fit to a narrow raw range
    # (e.g. audio branch where raw p_attack spans only [0.001, 0.01]).
    probe_out_on_train = np.asarray(chosen.transform(raw), dtype=np.float32)
    train_range = float(probe_out_on_train.max() - probe_out_on_train.min())
    # Also compute the synthetic [0.01, 0.99] probe range for reporting
    probe = np.linspace(0.01, 0.99, 50, dtype=np.float32)
    probe_out = np.asarray(chosen.transform(probe), dtype=np.float32)
    probe_range = float(probe_out.max() - probe_out.min())
    # ECE improvement ratio (relative)
    ece_rel = (ece_before - ece_chosen) / max(ece_before, 1e-6)
    # Pass if: (strong output spread on training data) OR (meaningful ECE improvement)
    strong_spread = train_range >= 0.10
    strong_ece = ece_rel >= 0.25  # >= 25% relative ECE reduction
    if not strong_spread and not strong_ece:
        return (PlattCalibrator(a=1.0, b=0.0, n=len(raw)),
                {"choice": "identity_nosignal",
                 "n": float(len(raw)),
                 "ece_before": float(ece_before),
                 "ece_after": float(ece_before),
                 "fit_range": probe_range,
                 "train_output_range": train_range,
                 "ece_rel_improvement": ece_rel,
                 "reason": "no_signal_on_training_data"})

    meta_out = {"choice": choice_name, "n": float(len(raw)),
                "ece_before": float(ece_before),
                "ece_after": float(ece_chosen),
                "fit_range": probe_range,
                "train_output_range": train_range,
                "ece_rel_improvement": ece_rel}
    meta_out.update(other_eces)
    return chosen, meta_out


def fit_from_records(records: List[Dict]) -> CalibratorSet:
    """Fit per-branch calibrators from scored records.
    Records follow the pad.report schema."""
    labels = np.asarray(
        [1 if r.get("label") == "spoof" else 0 for r in records],
        dtype=np.int32)

    # Collect per-branch score arrays
    branch_raw: Dict[str, List[float]] = {}
    branch_lbl: Dict[str, List[int]] = {}
    for r, y in zip(records, labels):
        for b in (r.get("branches") or []):
            if isinstance(b, dict):
                name = b.get("name")
                p = b.get("p_attack")
                rel = b.get("reliability", 1.0)
                if name is None or p is None or rel is None or rel <= 0:
                    continue
                branch_raw.setdefault(name, []).append(float(p))
                branch_lbl.setdefault(name, []).append(int(y))

    per_branch: Dict[str, object] = {}
    metadata: Dict[str, Dict[str, float]] = {}
    for name, xs in branch_raw.items():
        cal, meta = _pick_best(np.asarray(xs, dtype=np.float32),
                               np.asarray(branch_lbl[name], dtype=np.int32))
        per_branch[name] = cal
        metadata[name] = meta

    return CalibratorSet(per_branch=per_branch, fused=None,
                         metadata=metadata, version=1)


# ----------------------------------------------------------------------
# persistence
# ----------------------------------------------------------------------
CALIBRATOR_PATH = MODEL_DIR / "calibrators.pkl"


def save_calibrators(cs: CalibratorSet,
                     path: Path | str = CALIBRATOR_PATH) -> None:
    """Persist a CalibratorSet to disk via joblib.

    We use joblib (not pickle) so the file format matches what
    `load_calibrators` reads with sklearn's wire format -- direct
    `pickle.load` against a joblib-written file fails with
    `_pickle.UnpicklingError: invalid load key` even though both
    formats end in .pkl. The path's parent is created if missing.
    Calling code (`pad.refit_single_branch`) backs up the current
    file before invoking this; this function does no backup itself.
    """
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cs, path)


_CALIBRATORS_CACHE: "tuple[bool, Optional[CalibratorSet], float]" = (False, None, 0.0)
#                          loaded? cached-set                     pkl mtime snapshot


def _calibrators_mtime(path: "Path | str" = None) -> float:
    """Current mtime of the calibrators.pkl on disk (0.0 if missing).
    Used by apply_calibration to detect hot-reload events without
    stat'ing more than once per call. Returns 0.0 on any IO error
    (file deleted mid-read, permission flap, network share blip);
    the caller treats 0.0 as "no change since last load" because the
    cache's cached_mtime is also 0.0 on first call."""
    from pathlib import Path as _Path
    p = _Path(path) if path else CALIBRATOR_PATH
    try:
        return float(p.stat().st_mtime)
    except Exception:
        return 0.0


# v9.10: per-branch training-set base rates. Used by the optional
# Saerens-et-al-2002 prior-shift correction below. These are the
# fraction of "spoof" labels in the records the calibrator was fit
# on. Refresh whenever a calibrator is refit on substantially
# different label balance.
_BRANCH_TRAIN_PRIOR_SPOOF: "Dict[str, float]" = {
    "moire":             0.74,   # merged_v6 + ffpp_v5  (8869 live + 32000 spoof)
    "deepfake":          0.74,   # merged_v6 + ffpp_v5  (8869 live + 32000 spoof)
    "patch_depth":       0.55,   # patchdepth_v7 + ffxray_v7 (~45k live + ~55k spoof)
    "deepfake_temporal": 0.50,   # celebdf + ffpp (roughly balanced)
    "audio":             0.50,   # asvspoof_v3 (roughly balanced)
    # v9.8.1 2026-05-12 (9th-pass deep audit): these branches don't
    # have a calibrator-style training corpus (they're rule-based or
    # heuristic-driven), but the prior-shift formula still needs SOME
    # value for π_train. 0.5 = "no presumed training imbalance" --
    # prior-shift becomes a pure deployment-prior pull when raw_p is
    # extreme but a no-op near 0.5. This lets DODO_DEPLOY_SPOOF_PRIOR
    # actually take effect for these branches (it was a silent no-op
    # before, because `_BRANCH_TRAIN_PRIOR_SPOOF.get("dynamics", 0.5)`
    # returned 0.5 which then equals pi_deploy -> no shift).
    "dynamics":          0.50,   # heuristic blink/three_d/vergence
    "blink_gaze":        0.50,   # rule-based bpm/gaze_std
    "rppg":              0.50,   # signal-processing only, no ML
    "radar_presence":    0.10,   # 60 GHz hardware: real subjects
                                 #   dominate training (and operation)
                                 #   so the natural base rate is ~10%
                                 #   spoof. Calibrated values stay near
                                 #   raw -- intentional.
    # API_unq additions (2026-05-13):
    "screen_flash":      0.50,   # active liveness: branch only fires on
                                 #   challenge; no inherent training prior.
                                 #   Symmetric so DODO_DEPLOY_SPOOF_PRIOR
                                 #   pulls toward deployment rate naturally.
    "foundation_pad":    0.50,   # placeholder. Replace with actual training
                                 #   prior (live/spoof balance of the
                                 #   records jsonl used by train_foundation_pad)
                                 #   when calibrator is fit post-training.
}


def _prior_shift(p: float, pi_train: float, pi_deploy: float) -> float:
    """Saerens et al. (2002) prior-shift correction: rescale a
    calibrated probability from one base-rate to another.

        p' = (p · π_d / π_t) / (p · π_d / π_t + (1-p) · (1-π_d)/(1-π_t))

    Use case: a calibrator was fit on records with `pi_train` spoof
    fraction (e.g. 0.74 for moire's training set), but the actual
    deployment scenario has a much lower base rate (e.g. 0.05 for a
    user normally sitting in front of their own webcam, with spoofs
    rare). Without the shift, the calibrator pegs a live face at
    p ≈ 0.74 because that's the training prior; with the shift, the
    same face drops to p ≈ 0.17 -- still calibrated, just to the
    correct deployment prior.

    The transformation is monotonic, so AUC is preserved; only the
    threshold-sensitive metrics (APCER/BPCER/ACER) shift.
    """
    eps = 1e-9
    pi_t = max(min(float(pi_train), 1 - eps), eps)
    pi_d = max(min(float(pi_deploy), 1 - eps), eps)
    p = max(min(float(p), 1 - eps), eps)
    a = p * (pi_d / pi_t)
    b = (1.0 - p) * ((1.0 - pi_d) / (1.0 - pi_t))
    return float(a / (a + b))


def apply_calibration(branch_name: str, raw_p: float) -> float:
    """Apply the per-branch calibrator to a raw p_attack score.

    Lazy-load + hot-reload: checks calibrators.pkl mtime on every
    call. If it changes on disk (e.g. after a retrain + `deploy_models.cmd`
    touch), the new CalibratorSet is swapped in without a server restart.

    Concurrency note: `_CALIBRATORS_CACHE` is a module-level tuple read
    and written without a lock. Under the CPython GIL, the read and the
    rebind are individually atomic, so a worker can never observe a
    half-built cache. Two threads racing during a hot-reload may both
    call `load_calibrators()` once -- the cost is one redundant pickle
    read, never incorrect output. We accept that over the per-call lock
    overhead because the steady-state path (cache hit) must stay tight.

    Returns raw_p unchanged when the calibrator set is absent or fails
    to load -- matches the graceful-degrade policy.

    v9.10 prior-shift: when env DODO_DEPLOY_SPOOF_PRIOR is set to a
    float in (0, 1), the calibrated probability gets prior-shifted from
    the per-branch training base rate (`_BRANCH_TRAIN_PRIOR_SPOOF`) to
    that deployment prior. The default (env unset, or 'auto', or '0')
    is no shift -- backward compatible.

    Recommended values (empirically tuned 2026-05-08 against the
    moire-bias incident):
        0.50  matches typical PAD-paper evaluation (no shift if your
              training data was 50/50 too; near-no-shift otherwise)
        0.30  (recommended default for dodo) -- a live face at raw=0.76
              shifts to ~0.32 (UNCERTAIN, leaning live), a real spoof
              at raw=0.95 still lands at ~0.74 (crosses spoof_high=0.75)
        0.20  more aggressive bias-correction; live face -> 0.22, real
              spoof -> 0.62. Better for "this is my own webcam, attacks
              are rare" scenarios but starts missing borderline spoofs.
        0.05  CAUTION: real spoofs at raw=0.95 drop to ~0.26 -- below
              spoof_high=0.75 -- so attacks get missed. Only use if
              your live-attack-test corpus shows attack rates below
              5%, AND you have an independent attack detector.

    The shift is applied AFTER the per-branch calibrator transform,
    so it doesn't interfere with the collapse-aware selector or the
    user-live Platt anchors. If you previously ran
    `calibrate_user_live.py`, the personal Platt shift already lands
    live at p~0.01 and prior-shift on top is redundant -- in that case
    leave DODO_DEPLOY_SPOOF_PRIOR unset.
    """
    global _CALIBRATORS_CACHE
    loaded, cs, cached_mtime = _CALIBRATORS_CACHE
    current_mtime = _calibrators_mtime()
    if (not loaded) or current_mtime != cached_mtime:
        cs = load_calibrators()
        _CALIBRATORS_CACHE = (True, cs, current_mtime)
    if cs is None:
        return float(raw_p)
    p = cs.transform_branch(branch_name, float(raw_p))

    # Optional prior-shift -- backward compatible (no shift if env unset)
    pi_d_str = os.environ.get("DODO_DEPLOY_SPOOF_PRIOR", "").strip().lower()
    if pi_d_str and pi_d_str not in ("auto", "0", "0.0", "off", "no", "false"):
        try:
            pi_d = float(pi_d_str)
            if 0.0 < pi_d < 1.0:
                pi_t = _BRANCH_TRAIN_PRIOR_SPOOF.get(branch_name, 0.5)
                p = _prior_shift(p, pi_t, pi_d)
        except ValueError:
            pass
    return p


def _reset_calibrators_cache() -> None:
    """Test hook: drop the cached CalibratorSet so the next apply_calibration
    call re-reads the pickle. Not used in production."""
    global _CALIBRATORS_CACHE
    _CALIBRATORS_CACHE = (False, None, 0.0)


def jsonl_sha256(paths: "List[Path | str]") -> str:
    """SHA-256 of the concatenated content of one or more JSONL files,
    computed deterministically (paths sorted by basename to be
    order-insensitive at the caller). Empty / missing paths are skipped
    silently. Returned as the first 16 hex chars of the full digest —
    enough for drift detection without bloating metadata.

    Stamped into `metadata[branch]["training_jsonls_sha256"]` by the
    CLI fit entry-points so `/models/status` can flag calibrators that
    were trained on jsonls which have since changed on disk.
    """
    import hashlib
    from pathlib import Path as _Path
    h = hashlib.sha256()
    norm = sorted(str(_Path(p).name) for p in paths)
    h.update(("||".join(norm)).encode("utf-8"))
    for name in norm:
        # Find the full path again (caller may have passed absolutes)
        p = next((_Path(q) for q in paths if _Path(q).name == name), None)
        if p is None or not p.is_file():
            continue
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def calibrator_pkl_sha256(path: Path | str = CALIBRATOR_PATH) -> str:
    """API_unq P2.3: SHA-256 of the on-disk calibrator pickle, for the
    forensic audit trail. Empty string when path doesn't exist or read
    fails. Read-once-per-mtime cached at the module level so per-frame
    audit emission is O(1).
    """
    import hashlib
    p = Path(path)
    if not p.exists():
        return ""
    try:
        # Cache by (path, mtime, size) to avoid hashing every frame
        key = (str(p), float(p.stat().st_mtime), int(p.stat().st_size))
        cache = getattr(calibrator_pkl_sha256, "_cache", {})
        if key in cache:
            return cache[key]
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            while True:
                chunk = fh.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        sha = h.hexdigest()
        # Bound the cache
        if len(cache) > 8:
            cache.clear()
        cache[key] = sha
        calibrator_pkl_sha256._cache = cache  # type: ignore[attr-defined]
        return sha
    except Exception:
        return ""


def calibrator_status() -> Dict[str, object]:
    """Snapshot of the currently-loaded calibrator set, for /models/status
    or Prometheus metrics endpoints. Cheap -- no disk I/O unless mtime
    changed since last apply_calibration().

    API_unq P2.3: adds `pkl_sha256` to the top-level snapshot so per-
    frame forensic records can stamp it.
    """
    global _CALIBRATORS_CACHE
    loaded, cs, cached_mtime = _CALIBRATORS_CACHE
    current_mtime = _calibrators_mtime()
    pkl_sha = calibrator_pkl_sha256()
    if cs is None:
        return {"loaded": bool(loaded), "path": str(CALIBRATOR_PATH),
                "mtime": cached_mtime, "pkl_sha256": pkl_sha,
                "branches": {}, "available": False}
    branches = {}
    for name, meta in (cs.metadata or {}).items():
        branches[name] = {
            "choice": meta.get("choice", "?"),
            "ece_before": meta.get("ece_before"),
            "ece_after":  meta.get("ece_after"),
            "n": int(meta.get("n", 0)),
            "n_fit": int(meta.get("n_fit", meta.get("n", 0))),
            "degenerate": meta.get("choice") == "identity_nosignal",
            "fit_source": meta.get("fit_source", "<unspecified>"),
            "training_jsonls_sha256": meta.get("training_jsonls_sha256", ""),
            "fit_range": meta.get("fit_range"),
        }
    return {"loaded": True, "path": str(CALIBRATOR_PATH),
            "mtime": cached_mtime, "fresh": current_mtime == cached_mtime,
            "pkl_sha256": pkl_sha,
            "branches": branches, "available": True}


def load_calibrators(path: Path | str = CALIBRATOR_PATH) -> Optional[CalibratorSet]:
    """Load the pickled calibrator set. Returns None if missing so
    runtime code can fall back to raw scores gracefully."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        # Pickle load can fail on version skew (sklearn major-bumps),
        # corrupt file, or partial write during a deploy. PAD policy is
        # graceful-degrade -- returning None lets the runtime branches
        # produce uncalibrated raw probabilities. Failure is observable
        # via the `cs is None` branch in apply_calibration.
        return None  # nolint:silent-except graceful-degrade-on-pkl-corruption


# ----------------------------------------------------------------------
# optional: rewrite thresholds.py to the optimised operating point
# ----------------------------------------------------------------------
def recommend_thresholds(records: List[Dict],
                         target_bpcer: float = 0.01) -> Dict[str, float]:
    """Given the calibrated fused probabilities, find the threshold that
    achieves target BPCER and the matching live_low threshold.
    """
    from pad.report import operating_point_at_bpcer
    op = operating_point_at_bpcer(records, target_bpcer)
    hi = op.get("threshold") or 0.75

    # Live-low = cross-over where BPCER drops to 0 (all live). Compute
    # by scanning from 0 upward for the first threshold producing
    # BPCER==0 (no live misclassified).
    live = [r for r in records if r.get("label") == "live"]
    if live:
        scan = sorted({round(r["p_attack"], 4) for r in records})
        live_low = 0.15
        for t in scan:
            bpcer = sum(1 for r in live if r["p_attack"] > t) / len(live)
            if bpcer == 0.0:
                live_low = float(t)
                break
    else:
        live_low = 0.15
    live_low = min(live_low, hi - 0.05)
    return {"spoof_high": float(hi), "live_low": float(live_low)}


def _write_thresholds(th: Dict[str, float]) -> None:
    """Edit pad/thresholds.py in place to replace the two dataclass
    default values. Preserves the rest of the file verbatim."""
    import re
    src = Path(__file__).parent / "thresholds.py"
    text = src.read_text(encoding="utf-8")
    for key in ("spoof_high", "live_low"):
        pat = re.compile(
            rf"({key}\s*:\s*float\s*=\s*)([0-9.]+)")
        text = pat.sub(rf"\g<1>{th[key]:.4f}", text, count=1)
    src.write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: fit per-branch calibrators from a JSONL of scored
    records and write them to `pad/models/calibrators.pkl`. Optionally
    update `pad/thresholds.py` to a BPCER target operating point.

    Refits ALL branches whose scores appear in the input. To refit only
    one branch (the typical case after a single-model swap), use
    `pad.refit_single_branch` instead -- that path is calibrator-collapse
    aware and won't deploy a degenerate fit.

    Returns 0 on success, 2 if input has no records.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="src", required=True,
                   help="path to scored-records .jsonl or .parquet")
    p.add_argument("--out", dest="out", type=Path, default=CALIBRATOR_PATH,
                   help=f"output path (default: {CALIBRATOR_PATH})")
    p.add_argument("--write-thresholds", action="store_true",
                   help="also update pad/thresholds.py to BPCER=0.01 operating point")
    p.add_argument("--target-bpcer", type=float, default=0.01,
                   help="target BPCER for operating-point selection (default: 0.01)")
    args = p.parse_args(argv)

    from pad.report import _load_records
    records = _load_records(args.src)
    if not records:
        print(f"[calibrate] no records in {args.src}", file=sys.stderr)
        return 2

    t0 = time.time()
    cs = fit_from_records(records)
    save_calibrators(cs, args.out)
    dt = time.time() - t0

    print(f"[calibrate] fit {len(cs.per_branch)} per-branch "
          f"calibrator(s) in {dt:.2f}s")
    for name, meta in sorted(cs.metadata.items()):
        before = meta.get("ece_before", float("nan"))
        after = meta.get("ece_after", float("nan"))
        choice = meta.get("choice", "?")
        n = int(meta.get("n", 0))
        print(f"  {name:>12s}  n={n:6d}  choice={choice:8s}  "
              f"ECE {before:.4f} -> {after:.4f}")
    print(f"[calibrate] saved -> {args.out}")

    if args.write_thresholds:
        th = recommend_thresholds(records, target_bpcer=args.target_bpcer)
        _write_thresholds(th)
        print(f"[calibrate] thresholds updated: spoof_high={th['spoof_high']:.4f}, "
              f"live_low={th['live_low']:.4f}")

    return 0


if __name__ == "__main__":
    # Re-enter through the package path so pickle sees dataclasses as
    # `pad.calibrate.CalibratorSet` instead of `__main__.CalibratorSet`.
    # Without this, any calibrators.pkl written via `python -m pad.calibrate`
    # fails to load via `from pad.calibrate import load_calibrators`.
    from pad.calibrate import main as _pkg_main
    sys.exit(_pkg_main())
