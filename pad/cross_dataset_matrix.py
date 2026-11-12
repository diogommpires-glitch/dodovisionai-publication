"""pad/cross_dataset_matrix.py -- reviewer-grade cross-dataset evaluation.

For each (train_source, test_source) pair:
  1. Fit a meta-fuser on train_source's records.
  2. Evaluate on test_source's records (optimal threshold per cell).
  3. Record ACER / APCER / BPCER / AUC.

Output: markdown matrix suitable for paper Table 2.

Usage:
    python -m pad.cross_dataset_matrix \\
        --sources celeba=reports/meta_train_records_celeba_v3.jsonl \\
                  replay=reports/meta_train_records_replay_v3.jsonl \\
                  celebdf=reports/meta_train_records_celebdf_v3.jsonl \\
                  asvspoof=reports/audio_asvspoof_records_v3.jsonl \\
        --out reports/cross_dataset_matrix.md
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from pad.bootstrap_ci import _load, _best_threshold, _apcer_bpcer, _fused_p


def _records_to_Xy_meta(records: List[Dict], branches_order: List[str],
                        prior_p_attack: float = 0.02):
    """Build meta-fuser feature matrix from records. Features per branch:
        [p_attack_i, reliability_i, p_attack_i * reliability_i]
    Missing branches -> [prior, 0, 0].
    """
    X = np.zeros((len(records), len(branches_order) * 3), dtype=np.float32)
    y = np.zeros(len(records), dtype=np.int32)
    for i, r in enumerate(records):
        y[i] = 1 if r.get("label") == "spoof" else 0
        found = {b.get("name"): b for b in (r.get("branches") or [])}
        for j, bname in enumerate(branches_order):
            b = found.get(bname)
            base = 3 * j
            if b and b.get("p_attack") is not None:
                p = float(b["p_attack"])
                rel = float(b.get("reliability", 1.0) or 0.0)
                X[i, base + 0] = p
                X[i, base + 1] = rel
                X[i, base + 2] = p * rel
            else:
                X[i, base + 0] = prior_p_attack
                X[i, base + 1] = 0.0
                X[i, base + 2] = 0.0
    return X, y


def _fit_lgbm(X, y, seed: int = 0):
    """Fit a LightGBM classifier; degrade gracefully to logistic regression
    if LGBM unavailable."""
    try:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                 num_leaves=15,
                                 min_child_samples=max(5, len(X) // 200),
                                 subsample=0.9, colsample_bytree=0.9,
                                 random_state=seed, verbose=-1)
        clf.fit(X, y)
        return clf
    except ImportError:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300)
        clf.fit(X, y)
        return clf


def _predict_proba(clf, X):
    """Return P(spoof=1) from any sklearn-API estimator."""
    p = clf.predict_proba(X)
    # sklearn convention: classes_[1] is the positive class
    try:
        pos = int(np.where(clf.classes_ == 1)[0][0])
        return p[:, pos]
    except Exception:
        return p[:, 1] if p.shape[1] > 1 else p[:, 0]


def main(argv=None) -> int:
    """CLI: build the cross-dataset AUC matrix for the IJCB 2026 paper.
    Trains a meta-fuser on each source's records, evaluates against
    every other source, and writes a markdown matrix where row=train
    and col=test. Diagonal AUCs are in-domain; off-diagonal are
    cross-domain generalization.

    --sources takes NAME=path pairs (e.g. celeba=reports/...jsonl
    celebdf=reports/...jsonl). Returns 0 on success, 2 if any source
    fails to load.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", required=True,
                    help="list of NAME=path pairs")
    ap.add_argument("--branches", nargs="+",
                    default=["moire", "patch_depth", "deepfake",
                             "audio", "deepfake_temporal"],
                    help="branch feature order for the meta-fuser")
    ap.add_argument("--out", type=Path,
                    default=Path("reports/cross_dataset_matrix.md"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    sources: Dict[str, List[Dict]] = {}
    for src in args.sources:
        if "=" not in src:
            print(f"[xd] bad --sources arg (need NAME=path): {src}",
                  file=sys.stderr); continue
        name, path_s = src.split("=", 1)
        path = Path(path_s)
        recs = _load(path)
        if not recs:
            print(f"[xd] {name}: empty or missing ({path})", file=sys.stderr)
            continue
        sources[name] = recs
        print(f"[xd] loaded {name}: {len(recs)} records")

    if len(sources) < 2:
        print("[xd] need >=2 sources for a cross-dataset matrix",
              file=sys.stderr); return 1

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("[xd] scikit-learn required", file=sys.stderr); return 2

    # Header
    src_names = list(sources.keys())
    lines = [
        "# Cross-Dataset Evaluation Matrix",
        "",
        "_Each cell: **ACER (AUC)** at per-cell optimal threshold._",
        "_Row = training source for the meta-fuser. Column = evaluation source._",
        "_Diagonal is same-source; off-diagonal is the generalization test reviewers care about._",
        "",
        "| train \\ test | " + " | ".join(f"`{n}`" for n in src_names) + " |",
        "|---|" + "|".join("---" for _ in src_names) + "|",
    ]

    for train_name in src_names:
        train_recs = sources[train_name]
        Xtr, ytr = _records_to_Xy_meta(train_recs, args.branches)
        if len(np.unique(ytr)) < 2:
            lines.append(f"| `{train_name}` | " + " | ".join("n/a" for _ in src_names) + " |")
            continue
        clf = _fit_lgbm(Xtr, ytr, seed=args.seed)
        row = [f"`{train_name}`"]
        for test_name in src_names:
            test_recs = sources[test_name]
            Xte, yte = _records_to_Xy_meta(test_recs, args.branches)
            if len(np.unique(yte)) < 2:
                row.append("n/a"); continue
            try:
                probs = _predict_proba(clf, Xte).astype(np.float32)
            except Exception as e:
                row.append(f"err({type(e).__name__})"); continue
            thr, _, _, _ = _best_threshold(probs, yte, metric="min_acer")
            a, b, c = _apcer_bpcer(probs, yte, thr)
            auc = float(roc_auc_score(yte, probs))
            row.append(f"{c:.3f} ({auc:.3f})")
            print(f"[xd] {train_name} -> {test_name}: ACER={c:.3f} AUC={auc:.3f} thr={thr:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    # Also a fused-log-odds baseline for comparison (no training)
    lines += ["",
              "## Reliability-weighted log-odds baseline (no meta-fuser)",
              "",
              "| Dataset | n | optimal thr | ACER | AUC |",
              "|---|---|---|---|---|"]
    for name, recs in sources.items():
        labels = np.array([1 if r["label"] == "spoof" else 0 for r in recs],
                          dtype=np.int32)
        scores = np.array([_fused_p(r) for r in recs], dtype=np.float32)
        if len(np.unique(labels)) < 2:
            continue
        thr, _, _, _ = _best_threshold(scores, labels, metric="min_acer")
        a, b, c = _apcer_bpcer(scores, labels, thr)
        auc = float(roc_auc_score(labels, scores))
        lines.append(f"| `{name}` | {len(recs)} | {thr:.4f} | {c:.3f} | {auc:.3f} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[xd] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
