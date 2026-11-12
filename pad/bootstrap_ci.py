"""bootstrap_ci.py -- Bootstrap confidence intervals on PAD metrics.

Reads scored records jsonls and computes APCER/BPCER/ACER point estimates
+ 95% bootstrap CIs for each dataset's records. Required for any reviewer-
grade PAD paper.

Usage:
    python -m pad.bootstrap_ci \\
        --inputs reports/meta_train_records_*_v3.jsonl \\
                 reports/audio_asvspoof_records_v3.jsonl \\
        --n-bootstrap 1000 \\
        --out reports/bootstrap_ci.md
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np


def _load(path: Path) -> List[Dict]:
    if not path.is_file():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _fused_p(record: Dict) -> float:
    """Reliability-weighted mean log-odds across the record's branches."""
    eps = 1e-6
    num, den = 0.0, 0.0
    for b in record.get("branches", []):
        p = b.get("p_attack")
        rel = b.get("reliability", 1.0)
        if p is None or rel is None or rel <= 0:
            continue
        w = min(max(float(rel), 0.0), 0.95)
        p = min(max(float(p), eps), 1.0 - eps)
        num += w * np.log(p / (1 - p))
        den += w
    if den <= 0:
        return 0.5
    return float(1.0 / (1.0 + np.exp(-num / den)))


def _apcer_bpcer(scores: np.ndarray, labels: np.ndarray, thresh: float = 0.5):
    """APCER = attack classified as live; BPCER = live classified as attack.

    v9.8.1 2026-05-12 (Audit C3): delegates to the unified
    `pad.metrics.apcer_bpcer_acer`. The previous local implementation
    used `preds = (scores > thresh)` (strict >); the canonical form
    uses `>=` per ISO/IEC 30107-3:2023 §7. The numeric difference is
    one-sample-at-boundary, but consistent reporting across the four
    files matters for paper-table reproducibility.
    """
    from pad.metrics import apcer_bpcer_acer as _canonical
    return _canonical(scores, labels, float(thresh))


def _best_threshold(scores: np.ndarray, labels: np.ndarray,
                    n_steps: int = 199, metric: str = "min_acer"):
    """Sweep thresholds and return the one optimising the given metric.
    metric='min_acer' -> minimise (APCER + BPCER)/2  (default)
    metric='eer'      -> minimise |APCER - BPCER|  (Equal Error Rate operating point)"""
    if scores.size == 0:
        return 0.5, float("nan"), float("nan"), float("nan")
    # Build a grid spanning the actual score distribution plus interior
    lo = float(np.quantile(scores, 0.01))
    hi = float(np.quantile(scores, 0.99))
    lo = max(1e-5, lo); hi = min(1 - 1e-5, hi)
    grid = np.linspace(lo, hi, n_steps)
    best = None
    for t in grid:
        a, b, c = _apcer_bpcer(scores, labels, float(t))
        if metric == "eer":
            obj = abs(a - b)
        else:
            obj = c
        if best is None or obj < best[0]:
            best = (obj, float(t), a, b, c)
    return best[1], best[2], best[3], best[4]   # threshold, apcer, bpcer, acer


def _per_branch_sweep(records: List[Dict], metric: str = "min_acer"):
    """Per-branch threshold sweep; returns {branch: (thresh, apcer, bpcer, acer, auc)}."""
    from sklearn.metrics import roc_auc_score
    labels = np.array([1 if r["label"] == "spoof" else 0 for r in records],
                      dtype=np.int32)
    out = {}
    branches_seen = set()
    for r in records:
        for b in r.get("branches", []):
            if b.get("name") and b.get("p_attack") is not None:
                branches_seen.add(b["name"])
    for bname in sorted(branches_seen):
        scores = []
        valid_labels = []
        for r, y in zip(records, labels):
            for b in r.get("branches", []):
                if b.get("name") == bname and b.get("p_attack") is not None:
                    scores.append(float(b["p_attack"]))
                    valid_labels.append(int(y))
                    break
        if len(scores) < 20:
            continue
        scores = np.asarray(scores, dtype=np.float32)
        vl = np.asarray(valid_labels, dtype=np.int32)
        if len(np.unique(vl)) < 2:
            continue
        t, a, b_, c = _best_threshold(scores, vl, metric=metric)
        auc = float(roc_auc_score(vl, scores))
        out[bname] = (t, a, b_, c, auc)
    return out


def bootstrap_ci(scores: np.ndarray, labels: np.ndarray,
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 0,
                 thresh: float = 0.5) -> dict:
    """Return dict with point estimate + (low, high) CIs for APCER/BPCER/ACER.

    For an empty input array, returns a dict with n=0 and NaN
    point/CI tuples rather than None — callers can format the row
    uniformly without a special-case branch. (Was: returned None on
    empty, which forced every caller to check before unpacking the
    dict — a footgun that bit cross_dataset_matrix.py.)
    """
    rng = np.random.default_rng(seed)
    n = len(scores)
    if n == 0:
        nan = float("nan")
        return {
            "n": 0,
            "apcer": (nan, nan, nan),
            "bpcer": (nan, nan, nan),
            "acer":  (nan, nan, nan),
        }
    apcer0, bpcer0, acer0 = _apcer_bpcer(scores, labels, thresh)
    apcer_b, bpcer_b, acer_b = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a, b, c = _apcer_bpcer(scores[idx], labels[idx], thresh)
        apcer_b.append(a); bpcer_b.append(b); acer_b.append(c)
    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100
    return {
        "n": n,
        "apcer":  (apcer0, float(np.percentile(apcer_b, lo)), float(np.percentile(apcer_b, hi))),
        "bpcer":  (bpcer0, float(np.percentile(bpcer_b, lo)), float(np.percentile(bpcer_b, hi))),
        "acer":   (acer0,  float(np.percentile(acer_b,  lo)), float(np.percentile(acer_b,  hi))),
    }


def main(argv=None) -> int:
    """CLI: bootstrap-resample APCER/BPCER/ACER on one or more JSONL
    record files and emit a markdown table of point estimates with
    (low, high) confidence intervals. Used to attach honest uncertainty
    bars to the cross-domain results table in the IJCB 2026 paper.

    Returns 0 on success, 1 if no records can be loaded.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="fixed threshold (ignored when --optimize is set)")
    ap.add_argument("--optimize", choices=["min_acer", "eer"], default=None,
                    help="per-dataset threshold optimization: min_acer | eer")
    ap.add_argument("--per-branch", action="store_true",
                    help="also report per-branch threshold sweep + AUC")
    ap.add_argument("--out", type=Path, default=Path("reports/bootstrap_ci.md"))
    args = ap.parse_args(argv)

    thresh_mode = (f"per-dataset optimal ({args.optimize})"
                   if args.optimize else f"fixed = {args.threshold}")
    lines = [
        "# PAD Bootstrap Confidence Intervals",
        "",
        f"_n_bootstrap = {args.n_bootstrap}, alpha = {args.alpha}, "
        f"decision threshold = {thresh_mode}_",
        "",
        "## Fused-fusion metrics",
        "",
        "| Dataset | n | threshold | APCER (95 % CI) | BPCER (95 % CI) | ACER (95 % CI) |",
        "|---|---|---|---|---|---|",
    ]
    per_branch_blocks = []
    for path_s in args.inputs:
        path = Path(path_s)
        records = _load(path)
        if not records:
            print(f"[boot-ci] {path.name}: skipped ({path} missing or empty)",
                  file=sys.stderr)
            continue
        labels = np.array([1 if r["label"] == "spoof" else 0 for r in records],
                          dtype=np.int32)
        scores = np.array([_fused_p(r) for r in records], dtype=np.float32)
        if args.optimize:
            thresh, _, _, _ = _best_threshold(scores, labels, metric=args.optimize)
        else:
            thresh = args.threshold
        ci = bootstrap_ci(scores, labels, args.n_bootstrap, args.alpha,
                          thresh=thresh)
        if ci is None:
            continue
        a, al, ah = ci["apcer"]; b, bl, bh = ci["bpcer"]; c, cl, ch = ci["acer"]
        lines.append(
            f"| `{path.stem}` | {ci['n']} | {thresh:.4f} | "
            f"{a:.3f} ({al:.3f}, {ah:.3f}) | "
            f"{b:.3f} ({bl:.3f}, {bh:.3f}) | "
            f"{c:.3f} ({cl:.3f}, {ch:.3f}) |")
        print(f"[boot-ci] {path.name}: n={ci['n']} thr={thresh:.4f} "
              f"ACER={c:.3f} ({cl:.3f}, {ch:.3f})")
        if args.per_branch:
            pb = _per_branch_sweep(records,
                                   metric=args.optimize or "min_acer")
            if pb:
                per_branch_blocks.append((path.stem, pb))

    if per_branch_blocks:
        lines += ["", "## Per-branch optimal-threshold sweep (AUC is threshold-invariant)"]
        for ds_name, pb in per_branch_blocks:
            lines += ["", f"### `{ds_name}`", "",
                      "| Branch | AUC | best thr | APCER | BPCER | ACER |",
                      "|---|---|---|---|---|---|"]
            for bname, (t, a, b_, c, auc) in pb.items():
                lines.append(
                    f"| `{bname}` | {auc:.4f} | {t:.4f} | "
                    f"{a:.3f} | {b_:.3f} | {c:.3f} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[boot-ci] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
