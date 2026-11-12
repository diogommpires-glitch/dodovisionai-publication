"""run_ablation_matrix.py -- Drop-one-branch ablation matrix.

For each (branch, dataset) pair, re-computes fused APCER/BPCER/ACER with
that branch's score masked out (reliability -> 0). Quantifies each branch's
individual contribution to fusion.

Usage:
    python -m pad.run_ablation_matrix \\
        --inputs reports/meta_train_records_*_v3.jsonl \\
                 reports/audio_asvspoof_records_v3.jsonl \\
        --out reports/ablation_matrix.md
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np

from pad.bootstrap_ci import _fused_p, _apcer_bpcer, _load


def score_with_mask(records: List[Dict], mask_branch: str = None):
    labels, scores = [], []
    for r in records:
        labels.append(1 if r["label"] == "spoof" else 0)
        if mask_branch is None:
            scores.append(_fused_p(r))
            continue
        r2 = copy.deepcopy(r)
        for b in r2.get("branches", []):
            if b.get("name") == mask_branch:
                b["reliability"] = 0.0
        scores.append(_fused_p(r2))
    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--branches", nargs="+",
                    default=["moire", "patch_depth", "deepfake", "audio"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("reports/ablation_matrix.md"))
    args = ap.parse_args(argv)

    lines = [
        "# Drop-One-Branch Ablation Matrix",
        "",
        "_Each cell: fused-fusion ACER with that branch masked out_",
        "_Delta vs. full: how much ACER degrades without the branch (bigger = more important)_",
        "",
        "| Dataset | n | full-fusion ACER | "
        + " | ".join(f"drop `{b}` ACER (Δ)" for b in args.branches) + " |",
        "|---|---|---|"
        + "|".join("---" for _ in args.branches) + "|",
    ]
    for path_s in args.inputs:
        path = Path(path_s)
        records = _load(path)
        if not records:
            continue
        # Full fusion
        sc, lb = score_with_mask(records, None)
        _, _, full_acer = _apcer_bpcer(sc, lb, args.threshold)
        cells = [f"`{path.stem}`", str(len(records)), f"{full_acer:.3f}"]
        for b in args.branches:
            sc_m, lb_m = score_with_mask(records, b)
            _, _, acer_m = _apcer_bpcer(sc_m, lb_m, args.threshold)
            delta = acer_m - full_acer
            cells.append(f"{acer_m:.3f} ({delta:+.3f})")
        lines.append("| " + " | ".join(cells) + " |")
        print(f"[abl] {path.name}: full ACER={full_acer:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[abl] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
