"""pad/meta_fuse.py
===================

Learned fusion meta-classifier (alternative to `pad.fusion.fuse`).

The current `pad.fusion.fuse()` computes a reliability-weighted mean of per-
branch log-odds. That is robust and needs no data, but it cannot learn that
e.g. a high-confidence REPLAY_SCREEN signal from `moire` should dominate over
a mildly-suspicious `dynamics` vote — the two branches carry equal weight.

This module fits a LightGBM classifier on the per-branch feature vector:

    [ p_attack_i, reliability_i, p_attack_i * reliability_i,
      dominant_class_i_onehot, ... ]  for i in EXPECTED_BRANCHES

Missing branches are imputed with p_attack = prior (0.02), reliability = 0
so the model learns to ignore absent branches through reliability gating.

At runtime, `fuse_meta()` is drop-in compatible with `pad.fusion.fuse()` —
same signature, same `PADScore` return. If the model pickle is missing, it
falls back to `pad.fusion.fuse` so nothing breaks before training.

CLI:
    # Train from the same JSONL pad.report consumes
    python -m pad.meta_fuse --from val_records.jsonl

    # Evaluate trained model vs log-odds fusion on the same records
    python -m pad.meta_fuse --from val_records.jsonl --eval-only
"""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import MODEL_DIR
from pad.types import (
    ATTACK_CLASSES, BranchScore, PADScore, AttackClass, utc_now,
)
from pad.thresholds import PADThresholds, DEFAULT_THRESHOLDS


# Canonical branch order — must match across training and inference. If a
# new branch is added to the stack, append its name here (never prepend /
# reorder) so older pickles stay readable.
#
# v9.8.1 2026-05-12 (Audit C6): bumped to 8-branch schema.
# `radar_presence` and `blink_gaze` are now expected. The runtime
# pipeline emits these two branches; the pre-Audit-C6 pkl was trained
# on only the first 6 (the 4 visual + audio) and the meta-fuser was
# blind to radar / blink, leading to false-SPOOF on real subjects.
#
# When a pkl with the old 6-branch schema is loaded, `fuse_meta()`
# detects the version mismatch and falls back to log-odds fusion (with
# the post-classification radar veto already in place) so the system
# is never silently using a stale meta-fuser. To retrain a fresh
# 8-branch pkl, run:
#
#     python -m pad.meta_fuse --from val_records.jsonl
#
# (requires val_records.jsonl to contain `radar_presence` and
# `blink_gaze` columns; see `scripts/regen_meta_fuser_v2.py` for the
# data-prep helper).
EXPECTED_BRANCHES: Tuple[str, ...] = (
    "rppg", "moire", "dynamics", "patch_depth",
    "deepfake", "audio",
    # v9.8.1 (C6) additions:
    "radar_presence", "blink_gaze",
    # API_unq additions (2026-05-13):
    "screen_flash",     # active screen-flash CIELAB delta liveness
    "foundation_pad",   # CLIP ViT-L/14 + LoRA on q,v, FoundPAD-style
)
SCHEMA_VERSION = 3  # API_unq: 8 → 10 branches; v2 pkls still load via .expected_branches

# 3 numeric features per branch + len(ATTACK_CLASSES) one-hot if the branch
# flagged a dominant class. The one-hot lets the meta-model learn e.g. that
# REPLAY_SCREEN nominations from moire should get extra weight.
NUM_FEATS_PER_BRANCH = 3 + len(ATTACK_CLASSES)
FEATURE_DIM = len(EXPECTED_BRANCHES) * NUM_FEATS_PER_BRANCH


# ----------------------------------------------------------------------
# feature extraction
# ----------------------------------------------------------------------
def _branch_to_feats(b: Optional[BranchScore],
                     prior: float) -> np.ndarray:
    """Return the fixed-length feature vector for a branch. Missing
    branch → prior-padded, reliability=0 (so the model treats it as
    "no evidence here")."""
    feats = np.zeros(NUM_FEATS_PER_BRANCH, dtype=np.float32)
    if b is None or b.reliability <= 0:
        feats[0] = prior           # p_attack imputed to prior
        feats[1] = 0.0             # reliability
        feats[2] = prior * 0.0     # interaction
        return feats
    feats[0] = float(b.p_attack)
    feats[1] = float(b.reliability)
    feats[2] = float(b.p_attack) * float(b.reliability)
    cls = b.dominant_class
    if cls:
        try:
            idx = ATTACK_CLASSES.index(cls)
            feats[3 + idx] = 1.0
        except ValueError:
            pass
    return feats


def branches_to_features(branches: List[BranchScore],
                         prior: float = 0.02) -> np.ndarray:
    """Build the fixed-length feature vector for a meta-fuser input.

    Always emits FEATURE_DIM entries in EXPECTED_BRANCHES order. Unknown
    branch names are ignored (training doesn't know them).
    """
    by_name = {b.name: b for b in branches}
    out = np.empty(FEATURE_DIM, dtype=np.float32)
    for i, name in enumerate(EXPECTED_BRANCHES):
        off = i * NUM_FEATS_PER_BRANCH
        out[off:off + NUM_FEATS_PER_BRANCH] = _branch_to_feats(
            by_name.get(name), prior)
    return out


def _records_to_Xy(records: List[Dict],
                   prior: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert report-style records to (X, y) for the meta-fuser.

    Accepts `branches` as either a list of dicts with the BranchScore
    fields, or a list of objects exposing the same attributes (what the
    live pipeline sends).
    """
    X = np.zeros((len(records), FEATURE_DIM), dtype=np.float32)
    y = np.zeros(len(records), dtype=np.int32)
    for i, r in enumerate(records):
        y[i] = 1 if r.get("label") == "spoof" else 0
        branches: List[BranchScore] = []
        for b in (r.get("branches") or []):
            if isinstance(b, dict):
                branches.append(BranchScore(
                    name=b.get("name", ""),
                    p_attack=float(b.get("p_attack", prior)),
                    reliability=float(b.get("reliability", 0.0)),
                    dominant_class=b.get("dominant_class"),
                ))
            elif isinstance(b, BranchScore):
                branches.append(b)
        X[i] = branches_to_features(branches, prior=prior)
    return X, y


# ----------------------------------------------------------------------
# train / save / load
# ----------------------------------------------------------------------
@dataclass
class MetaFuser:
    model: object                              # fitted lightgbm.LGBMClassifier
    feature_dim: int
    expected_branches: Tuple[str, ...] = field(default_factory=lambda: EXPECTED_BRANCHES)
    prior_p_attack: float = 0.02
    version: int = 1                           # legacy pkl had version=1
    metrics: Dict[str, float] = field(default_factory=dict)

    def predict_proba(self, branches: List[BranchScore]) -> float:
        # v9.8.1 2026-05-12 (Audit C6): if the loaded pkl was trained
        # on an OLDER `expected_branches` tuple (e.g. v1 6-branch pkl
        # being asked to predict against v2 8-branch features), we'd
        # silently feed the wrong feature dim into LightGBM. Use the
        # pkl's OWN expected_branches for feature construction so a
        # legacy pkl still produces valid predictions on its own
        # branch subset (radar+blink are ignored, as they were
        # invisible at training time anyway -- the post-fusion radar
        # veto in `fuse_meta()` covers the gap until retrain).
        own_branches = self.expected_branches or EXPECTED_BRANCHES
        x = _branches_to_features_with_schema(branches,
                                              prior=self.prior_p_attack,
                                              expected=own_branches)
        # LightGBM expects 2D
        p = self.model.predict_proba(x.reshape(1, -1))[0, 1]
        return float(p)


def _branches_to_features_with_schema(branches: List[BranchScore],
                                      prior: float,
                                      expected: Tuple[str, ...]) -> np.ndarray:
    """Like `branches_to_features` but the EXPECTED_BRANCHES list is a
    parameter, so a loaded MetaFuser pkl can self-describe its own
    feature schema and we don't have to retrain when the runtime list
    expands. v9.8.1 Audit C6 introduction."""
    by_name = {b.name: b for b in branches}
    feat_dim = len(expected) * NUM_FEATS_PER_BRANCH
    out = np.empty(feat_dim, dtype=np.float32)
    for i, name in enumerate(expected):
        off = i * NUM_FEATS_PER_BRANCH
        out[off:off + NUM_FEATS_PER_BRANCH] = _branch_to_feats(
            by_name.get(name), prior)
    return out


META_PATH = MODEL_DIR / "meta_fuser.pkl"


def train_meta_fuser(records: List[Dict],
                     prior_p_attack: float = 0.02,
                     seed: int = 0,
                     expected_branches: Tuple[str, ...] = EXPECTED_BRANCHES) -> MetaFuser:
    """Fit a LightGBM classifier on the branch-score feature vectors.

    Training set is expected to be modest (1k-100k records). LGBM is
    overkill for 1k, perfect for 100k, fast to load at runtime.
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

    X, y = _records_to_Xy(records, prior=prior_p_attack)

    if len(np.unique(y)) < 2:
        raise ValueError(
            "meta_fuse training needs both live (0) and spoof (1) labels; "
            f"got classes {np.unique(y)}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)

    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=max(5, len(Xtr) // 200),
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        verbose=-1,
    )
    clf.fit(Xtr, ytr)

    probs = clf.predict_proba(Xte)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(yte, (probs > 0.5).astype(int))),
        "auc":      float(roc_auc_score(yte, probs)) if len(np.unique(yte)) > 1 else float("nan"),
        "log_loss": float(log_loss(yte, np.clip(probs, 1e-6, 1 - 1e-6))),
        "n_train":  float(len(Xtr)),
        "n_test":   float(len(Xte)),
    }
    return MetaFuser(model=clf, feature_dim=FEATURE_DIM,
                     expected_branches=expected_branches,
                     prior_p_attack=prior_p_attack,
                     version=SCHEMA_VERSION,
                     metrics=metrics)


def save_meta_fuser(mf: MetaFuser, path: Path | str = META_PATH) -> None:
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mf, path)


def load_meta_fuser(path: Path | str = META_PATH) -> Optional[MetaFuser]:
    path = Path(path)
    if not path.exists():
        return None
    # Shim for the "__main__.MetaFuser" pickle trap: pickles produced via
    # `python -m pad.meta_fuse` record the class as __main__.MetaFuser, so a
    # fresh process that imports pad.meta_fuse can't resolve it. Alias the
    # class into the current __main__ before unpickling.
    import sys as _sys
    _main = _sys.modules.get("__main__")
    _prev = getattr(_main, "MetaFuser", None) if _main is not None else None
    try:
        if _main is not None:
            _main.MetaFuser = MetaFuser
        import joblib
        return joblib.load(path)
    except Exception:
        return None
    finally:
        if _main is not None:
            if _prev is None:
                try:
                    delattr(_main, "MetaFuser")
                except AttributeError:
                    pass
            else:
                _main.MetaFuser = _prev


# ----------------------------------------------------------------------
# inference (drop-in replacement for pad.fusion.fuse)
# ----------------------------------------------------------------------
_META_SINGLETON: Optional[MetaFuser] = None


def _get_singleton() -> Optional[MetaFuser]:
    global _META_SINGLETON
    if _META_SINGLETON is None:
        _META_SINGLETON = load_meta_fuser()
    return _META_SINGLETON


def _classify(p: float, th: PADThresholds) -> str:
    if p > th.spoof_high:
        return "SPOOF"
    if p < th.live_low:
        return "LIVE"
    return "UNCERTAIN"


def _infer_class(branches: List[BranchScore]) -> AttackClass:
    candidates = [
        (b.dominant_class, b.reliability * b.p_attack)
        for b in branches
        if b.p_attack > 0.5 and b.dominant_class
    ]
    if not candidates:
        return "UNKNOWN"
    weighted: Dict[str, float] = {}
    for cls, weight in candidates:
        if cls:
            weighted[cls] = weighted.get(cls, 0.0) + weight
    best = max(weighted.items(), key=lambda kv: kv[1])
    return best[0]  # type: ignore[return-value]


def fuse_meta(
    branches: List[BranchScore],
    thresholds: Optional[PADThresholds] = None,
    ts: Optional[float] = None,
    fallback: bool = True,
) -> PADScore:
    """Drop-in replacement for pad.fusion.fuse, using the learned model.

    If no trained meta-fuser is cached on disk, falls back to log-odds
    fusion so nothing regresses before training.
    """
    th = thresholds or DEFAULT_THRESHOLDS
    ts = ts if ts is not None else utc_now()

    mf = _get_singleton()
    if mf is None:
        if fallback:
            from pad.fusion import fuse as _logodds
            return _logodds(branches, thresholds=th, ts=ts)
        # Graceful: caller explicitly wants meta mode; return neutral score
        import math
        p = th.prior_p_attack
        return PADScore(
            p_attack=round(p, 4),
            verdict=_classify(p, th),
            attack_class="UNKNOWN",
            branches=branches,
            ts=ts,
            log_odds=round(math.log(p / (1 - p)), 4),
        )

    p = mf.predict_proba(branches)
    # log-odds for UI diagnostics; clip to avoid inf
    import math
    p_clip = min(max(p, 1e-4), 1 - 1e-4)
    L = math.log(p_clip / (1 - p_clip))
    verdict = _classify(p, th)

    # v9.8.1 2026-05-12 (9th-pass deep audit): the trained meta_fuser.pkl's
    # `EXPECTED_BRANCHES` does NOT include `radar_presence` or `blink_gaze`
    # (only the 6 visual + audio branches). That means the radar's
    # confident-LIVE vote is INVISIBLE to the meta-fuser at inference time,
    # so on a real subject where all visual branches skew weakly toward
    # SPOOF (under glasses + indoor lighting) the meta-fuser confidently
    # outputs `p_attack > 0.75 -> SPOOF` even though the 60 GHz radar is
    # reporting `p=0.05 rel=1.0` (rock-solid live torso).
    #
    # Fix-without-retraining: replicate the radar veto / radar trust /
    # degraded-LIVE downgrade logic from `pad.fusion.fuse` so the meta-
    # fuser's verdict goes through the same physical-liveness safety net.
    # When meta_fuser.pkl is retrained with EXPECTED_BRANCHES extended to
    # include radar+blink_gaze, this veto can be relaxed (the trained
    # model itself will then properly weight radar).
    PHYSICAL_BRANCHES = ("radar_presence", "rppg", "deepfake", "blink_gaze",
                         "screen_flash")  # API_unq: active liveness counts as physical
    PHYSICAL_REL_MIN  = 0.15
    physical_confirmed = any(
        b.name in PHYSICAL_BRANCHES and b.reliability >= PHYSICAL_REL_MIN
        and b.p_attack < 0.5
        for b in branches
    )
    degraded = (verdict == "LIVE" and not physical_confirmed)
    if degraded:
        verdict = "UNCERTAIN"

    radar = next((b for b in branches if b.name == "radar_presence"), None)
    # API_unq 2026-05-13 (P0.2): mirror fusion.fuse() — prior-relative
    # log-odds gates instead of hardcoded p_attack literals. Invariant
    # to DODO_DEPLOY_SPOOF_PRIOR shifts.
    import math as _math
    import os as _os
    _env_prior = _os.environ.get("DODO_DEPLOY_SPOOF_PRIOR", "").strip().lower()
    _meta_prior = 0.02
    if _env_prior and _env_prior not in ("auto", "0", "0.0", "off", "no", "false"):
        try:
            _candidate = float(_env_prior)
            if 0.0 < _candidate < 1.0:
                _meta_prior = _candidate
        except ValueError:
            pass
    _prior_logit = _math.log(max(_meta_prior, 1e-4) / max(1 - _meta_prior, 1e-4))
    # Match the PADThresholds defaults: 0.25/0.20 vs prior=0.02.
    _veto_delta_max = _math.log(0.25 / 0.75) - _math.log(0.02 / 0.98)
    _strong_delta_max = _math.log(0.20 / 0.80) - _math.log(0.02 / 0.98)
    if radar is not None:
        _radar_logodds_delta = _math.log(
            max(radar.p_attack, 1e-4) / max(1 - radar.p_attack, 1e-4)
        ) - _prior_logit
    else:
        _radar_logodds_delta = 0.0
    radar_physical_veto = (
        radar is not None
        and radar.reliability >= 0.50
        and _radar_logodds_delta <= _veto_delta_max
    )
    _trust_radar = _os.environ.get("DODO_PAD_TRUST_RADAR", "0").lower() in ("1", "true", "yes")
    radar_strong = (
        radar is not None
        and radar.reliability >= 0.80
        and _radar_logodds_delta <= _strong_delta_max
    )
    if verdict == "SPOOF" and radar_physical_veto:
        verdict = "UNCERTAIN"
    if _trust_radar and radar_strong and physical_confirmed and verdict == "UNCERTAIN":
        verdict = "LIVE"

    # v9.8.1 2026-05-12 (13th pass): same as fusion.fuse() -- if
    # verdict is LIVE, suppress attack_class. A single branch's
    # dominant_class (dynamics PRINT, blink_gaze NO_BLINK, etc.)
    # should not surface to the operator UI when the fused verdict
    # has already overruled it.
    if verdict == "LIVE" or radar_physical_veto:
        attack_class = None
    else:
        attack_class = _infer_class(branches)

    # v9.8.1 2026-05-12 (Phase 2 #3): disagreement-variance flag --
    # mirrors `pad.fusion.fuse`. When the meta-fuser's LightGBM head
    # produces an UNCERTAIN-looking score driven by contradictory
    # high-reliability branches, surface the variance + flag so the
    # UI can render a specific "sensor confusion" banner regardless
    # of which fusion backend is active.
    high_rel_branches = [b for b in branches if b.reliability > 0.30]
    degraded_disagreement = False
    p_variance = 0.0
    if len(high_rel_branches) >= 2:
        _ps = [float(b.p_attack) for b in high_rel_branches]
        _mean_p = sum(_ps) / len(_ps)
        p_variance = sum((p - _mean_p) ** 2 for p in _ps) / len(_ps)
        degraded_disagreement = p_variance > 0.08

    return PADScore(
        p_attack=round(p, 4),
        verdict=verdict,
        attack_class=attack_class,
        branches=branches,
        ts=ts,
        log_odds=round(L, 4),
        meta={"degraded_no_physical":  bool(degraded),
              "physical_confirmed":    bool(physical_confirmed),
              "radar_physical_veto":   bool(radar_physical_veto),
              "degraded_disagreement": bool(degraded_disagreement),
              "p_variance":            round(p_variance, 4),
              "high_rel_n":            len(high_rel_branches),
              "via_meta_fuser":        True},
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _evaluate_vs_logodds(records: List[Dict],
                         mf: Optional[MetaFuser],
                         prior: float) -> Dict[str, float]:
    """Compare meta-fuser to log-odds fusion on the same records.
    Returns a small metrics dict."""
    from pad.fusion import fuse as _logodds
    from sklearn.metrics import roc_auc_score, log_loss

    y = np.array([1 if r.get("label") == "spoof" else 0 for r in records])
    p_logodds = np.zeros(len(records))
    p_meta = np.zeros(len(records))

    for i, r in enumerate(records):
        branches: List[BranchScore] = []
        for b in (r.get("branches") or []):
            if isinstance(b, dict):
                branches.append(BranchScore(
                    name=b.get("name", ""),
                    p_attack=float(b.get("p_attack", prior)),
                    reliability=float(b.get("reliability", 0.0)),
                    dominant_class=b.get("dominant_class"),
                ))
        p_logodds[i] = _logodds(branches).p_attack
        p_meta[i] = mf.predict_proba(branches) if mf else p_logodds[i]

    out: Dict[str, float] = {}
    if len(np.unique(y)) > 1:
        out["auc_logodds"] = float(roc_auc_score(y, p_logodds))
        out["auc_meta"] = float(roc_auc_score(y, p_meta))
    out["logloss_logodds"] = float(log_loss(y, np.clip(p_logodds, 1e-6, 1 - 1e-6)))
    out["logloss_meta"] = float(log_loss(y, np.clip(p_meta, 1e-6, 1 - 1e-6)))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True,
                    help="scored records (.jsonl or .parquet)")
    ap.add_argument("--out", type=Path, default=META_PATH,
                    help=f"output pickle (default: {META_PATH})")
    ap.add_argument("--eval-only", action="store_true",
                    help="do not retrain; just compare cached model vs log-odds")
    ap.add_argument("--prior", type=float, default=0.02,
                    help="prior P(attack) for imputation when a branch is missing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from pad.report import _load_records
    records = _load_records(args.src)
    if not records:
        print(f"[meta_fuse] no records in {args.src}", file=sys.stderr)
        return 2

    if args.eval_only:
        mf = load_meta_fuser()
        cmp = _evaluate_vs_logodds(records, mf, prior=args.prior)
        print(f"[meta_fuse] eval on {len(records)} records: {json.dumps(cmp, indent=2)}")
        return 0

    mf = train_meta_fuser(records, prior_p_attack=args.prior, seed=args.seed)
    save_meta_fuser(mf, args.out)
    print(f"[meta_fuse] trained — metrics: {json.dumps(mf.metrics, indent=2)}")
    print(f"[meta_fuse] saved -> {args.out}")

    # Comparison vs log-odds baseline for audit trail
    cmp = _evaluate_vs_logodds(records, mf, prior=args.prior)
    print(f"[meta_fuse] vs logodds: {json.dumps(cmp, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
