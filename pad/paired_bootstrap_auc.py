"""paired_bootstrap_auc.py -- Paired-bootstrap AUC delta, meta-fuser vs log-odds.

For each input scoring jsonl, computes:
  - AUC of the LightGBM meta-fuser score (loaded from pad/models/meta_fuser.pkl)
  - AUC of the reliability-weighted log-odds baseline (matches pad/bootstrap_ci.py)
  - Paired bootstrap on (AUC_meta - AUC_logodds) at n=10,000 resamples,
    reporting 95 % CI and a one-sided p-value (P[delta <= 0]).

This validates the paper's claim that the meta-fuser dominates the log-odds
baseline per dataset. One sentence drops into Section 5.2 Headline numbers.

Usage:
    python -m pad.paired_bootstrap_auc \\
        --inputs reports/meta_train_records_celeba_v6.jsonl \\
                 reports/meta_train_records_replay_v5.jsonl \\
                 reports/deepfake_temporal_v2_records_celebdf.jsonl \\
                 reports/audio_asvspoof_records_v3.jsonl \\
        --n-boot 10000
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from pad.bootstrap_ci import _load, _fused_p
from pad.meta_fuse import _records_to_Xy, FEATURE_DIM, load_meta_fuser


META_PATH = Path("pad/models/meta_fuser.pkl")

# Auto-detect mapping from a dataset jsonl name to the specialised meta-fuser
# pickle bundled in pad/models/. Section 5.5 of the paper notes that
# per-dataset specialised meta-fusers improve AUC further; the apples-to-apples
# comparison vs the log-odds baseline therefore uses these per-dataset fusers.
_DATASET_TO_META = (
    ("celebdf",   "meta_fuser_celebdf.pkl"),
    ("celeba",    "meta_fuser_celeba.pkl"),
    ("replay",    "meta_fuser_replay.pkl"),
    ("asvspoof",  "meta_fuser_asvspoof.pkl"),
    ("audio",     "meta_fuser_asvspoof.pkl"),
)


def _auto_meta_path(input_path: Path, models_dir: Path) -> Path:
    """Pick the per-dataset specialised meta-fuser by simple filename match.
    Falls back to the global meta_fuser.pkl if no token matches."""
    s = input_path.name.lower()
    for token, pkl in _DATASET_TO_META:
        if token in s:
            cand = models_dir / pkl
            if cand.is_file():
                return cand
    return models_dir / "meta_fuser.pkl"


def _load_meta_fuser(path: Path):
    """Return an object with .predict_proba(X) -> (n,2) array.

    Uses pad.meta_fuse.load_meta_fuser() which handles the joblib format and
    the __main__.MetaFuser pickle-aliasing trap baked in there.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    mf = load_meta_fuser(path)
    if mf is None:
        raise RuntimeError(
            f"pad.meta_fuse.load_meta_fuser({path}) returned None; "
            f"the artefact may be corrupted or use an incompatible format")
    if hasattr(mf, "model") and hasattr(mf.model, "predict_proba"):
        return mf.model
    if hasattr(mf, "predict_proba"):
        return mf
    raise TypeError(
        f"meta_fuser at {path} has no predict_proba interface "
        f"(got {type(mf).__name__})")


def _score_dataset(records: List[dict], meta_clf) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y, p_meta, p_logodds) per record."""
    X, y = _records_to_Xy(records, prior=0.02)
    if X.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"feature-dim mismatch: built {X.shape[1]} but FEATURE_DIM={FEATURE_DIM}")
    p_meta = meta_clf.predict_proba(X)[:, 1]
    p_lo = np.array([_fused_p(r) for r in records], dtype=np.float64)
    return y.astype(np.int32), p_meta.astype(np.float64), p_lo.astype(np.float64)


def _paired_bootstrap(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                      n_boot: int = 10000, seed: int = 42):
    """Returns dict with point AUCs, delta, 95% CI on delta, one-sided p-value."""
    n = len(y)
    auc_a = float(roc_auc_score(y, p_a))
    auc_b = float(roc_auc_score(y, p_b))
    delta_pt = auc_a - auc_b
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    valid = 0
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        if yi.sum() == 0 or yi.sum() == n:
            continue  # degenerate resample
        try:
            a = roc_auc_score(yi, p_a[idx])
            b = roc_auc_score(yi, p_b[idx])
            deltas[valid] = a - b
            valid += 1
        except ValueError:
            continue
    deltas = deltas[:valid]
    if valid == 0:
        return None
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_value = float((deltas <= 0).mean())  # one-sided: H0 = delta <= 0
    return {
        "n": n,
        "auc_a": auc_a,        # meta-fuser
        "auc_b": auc_b,        # log-odds
        "delta_pt": delta_pt,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_value": p_value,
        "n_boot_valid": valid,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--meta-path", default=None, type=Path,
                    help="single meta-fuser to use for all inputs; if unset, "
                         "the script auto-picks the per-dataset specialised "
                         "meta-fuser by matching the input filename")
    ap.add_argument("--models-dir", default=Path("pad/models"), type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    rows = []
    for path_s in args.inputs:
        path = Path(path_s)
        records = _load(path)
        if not records:
            print(f"[paired-boot] {path.name}: SKIP empty", file=sys.stderr)
            continue
        meta_path = args.meta_path or _auto_meta_path(path, args.models_dir)
        try:
            meta_clf = _load_meta_fuser(meta_path)
        except (FileNotFoundError, TypeError, RuntimeError) as e:
            print(f"[paired-boot] {path.name}: SKIP meta-fuser load: {e}",
                  file=sys.stderr)
            continue
        print(f"[paired-boot] {path.name}: using meta-fuser {meta_path.name}")
        try:
            y, p_meta, p_lo = _score_dataset(records, meta_clf)
        except Exception as e:
            print(f"[paired-boot] {path.name}: ERROR {type(e).__name__}: {e}",
                  file=sys.stderr)
            continue
        if len(np.unique(y)) < 2:
            print(f"[paired-boot] {path.name}: SKIP single-class",
                  file=sys.stderr)
            continue
        res = _paired_bootstrap(y, p_meta, p_lo,
                                n_boot=args.n_boot, seed=args.seed)
        if res is None:
            continue
        res["meta_used"] = meta_path.name
        rows.append((path.stem, res))
        print(f"[paired-boot] {path.name}: n={res['n']} "
              f"AUC_meta={res['auc_a']:.4f}  AUC_logodds={res['auc_b']:.4f}  "
              f"delta={res['delta_pt']:+.4f}  "
              f"95%CI=({res['ci_lo']:+.4f}, {res['ci_hi']:+.4f})  "
              f"p={res['p_value']:.4f}")

    if not rows:
        return 1

    print()
    print("# Markdown table (drop into Section 5.2 or appendix):")
    print()
    print("| Dataset | n | AUC meta | AUC log-odds | $\\Delta$ | 95% CI on $\\Delta$ | $p$ |")
    print("|---|---|---|---|---|---|---|")
    for stem, r in rows:
        sig = f"$<10^{{-4}}$" if r["p_value"] < 1e-4 else f"{r['p_value']:.4f}"
        print(f"| `{stem}` | {r['n']} | {r['auc_a']:.4f} | {r['auc_b']:.4f} | "
              f"{r['delta_pt']:+.4f} | "
              f"({r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}) | {sig} |")

    print()
    print("# Paper-ready sentence (drop into Section 5.2):")
    n_sig = sum(1 for _, r in rows if r["p_value"] < 0.05)
    print(f"# A paired bootstrap (n={args.n_boot}) on the AUC delta between the")
    print(f"# meta-fuser and the reliability-weighted log-odds baseline rejects the")
    print(f"# null hypothesis (Delta <= 0) at p < 0.05 on {n_sig} of {len(rows)} headline")
    print(f"# benchmarks; per-dataset deltas and CIs are tabulated above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
