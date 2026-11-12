"""det_curves.py -- DET (Detection Error Tradeoff) curves for PAD.

Reads scored records jsonls, computes the reliability-weighted log-odds
fused score per record (matching pad/bootstrap_ci.py for paper-grade
consistency), sweeps thresholds, and plots APCER vs BPCER on
probit-scaled axes. The output figure is the operating-curve convention
of ISO/IEC 30107-3 Section 7.

Usage:
    python -m pad.det_curves \\
        --inputs reports/meta_train_records_celeba_v6.jsonl \\
                 reports/meta_train_records_replay_v5.jsonl \\
                 reports/deepfake_temporal_v2_records_celebdf.jsonl \\
                 reports/audio_asvspoof_records_v3.jsonl \\
        --labels "CelebA-Spoof v6" "Replay-Attack v5" \\
                 "Celeb-DF v2 (temporal)" "ASVspoof 2019 LA" \\
        --out figures/det_curves.pdf
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Use the same operating-point search as pad/bootstrap_ci.py so the
# figure's min-ACER stars match Table 4 in the paper exactly.
from pad.bootstrap_ci import _best_threshold as _paper_best_threshold


def _load(path: Path) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _fused_p(record: Dict) -> float:
    """Reliability-weighted mean log-odds across the record's branches.
    Identical to pad/bootstrap_ci.py::_fused_p so the curve point at the
    paper's tau* lands exactly on the bootstrap_ci.md ACER number."""
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


def _det_points(scores: np.ndarray, labels: np.ndarray, n_grid: int = 400):
    """APCER/BPCER pairs across the threshold sweep."""
    grid_extra = np.linspace(1e-5, 1 - 1e-5, n_grid)
    thrs = np.unique(np.concatenate([scores, grid_extra]))
    thrs = np.sort(thrs)
    attack_mask = (labels == 1)
    live_mask = (labels == 0)
    apcer_arr = np.empty_like(thrs)
    bpcer_arr = np.empty_like(thrs)
    for i, t in enumerate(thrs):
        preds_attack = (scores > t)
        if attack_mask.any():
            apcer_arr[i] = 1.0 - preds_attack[attack_mask].mean()
        else:
            apcer_arr[i] = np.nan
        if live_mask.any():
            bpcer_arr[i] = preds_attack[live_mask].mean()
        else:
            bpcer_arr[i] = np.nan
    return apcer_arr, bpcer_arr, thrs


def _min_acer_point_paper_grade(scores: np.ndarray, labels: np.ndarray):
    """Operating-point search aligned with pad/bootstrap_ci.py
    (1st-to-99th percentile of scores, 199-point linspace), so the
    star + legend ACER on each curve match Table 4 exactly."""
    t, a, b, c = _paper_best_threshold(scores, labels, metric="min_acer")
    return float(a), float(b), float(t), float(c)


def _probit(x):
    eps = 1e-4
    return norm.ppf(np.clip(x, eps, 1 - eps))


def main(argv=None) -> int:
    """CLI: render Detection-Error-Tradeoff (DET) curves for one or more
    JSONL records files into a single PDF. Each input gets a labeled
    line on the standard FAR/FRR log-probit axes. Used for the IJCB
    2026 paper Figure 3.

    Each `--inputs` path is paired by index with `--labels`; lengths
    must match. Returns 0 on success, 1 on length mismatch.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("figures/det_curves.pdf"))
    args = ap.parse_args(argv)
    if len(args.inputs) != len(args.labels):
        print("--inputs and --labels must have same length", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    tick_pcts = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
    tick_labels_pct = ["0.1", "0.5", "1", "2", "5", "10", "20", "40"]

    colors = ["#1b6ca8", "#c45a00", "#147a3a", "#9b2235", "#5b2a86"]
    summary = []

    for i, (path_s, label) in enumerate(zip(args.inputs, args.labels)):
        path = Path(path_s)
        recs = _load(path)
        if not recs:
            print(f"[det] {path.name}: skipped (missing or empty)",
                  file=sys.stderr)
            continue
        labs = np.array([1 if r.get("label") == "spoof" else 0 for r in recs],
                        dtype=np.int32)
        if len(np.unique(labs)) < 2:
            print(f"[det] {path.name}: skipped (no class diversity)",
                  file=sys.stderr)
            continue
        scores = np.array([_fused_p(r) for r in recs], dtype=np.float64)
        apcer, bpcer, thrs = _det_points(scores, labs)
        a_star, b_star, t_star, acer_star = _min_acer_point_paper_grade(
            scores, labs)
        c = colors[i % len(colors)]
        ax.plot(_probit(apcer), _probit(bpcer),
                color=c, lw=1.7,
                label=f"{label}  (ACER {acer_star:.3f})")
        ax.plot(_probit(a_star), _probit(b_star),
                marker="*", color=c, markersize=11,
                markeredgecolor="black", markeredgewidth=0.6, ls="")
        summary.append((label, len(recs), t_star, a_star, b_star, acer_star))

    tick_loc = [_probit(x) for x in tick_pcts]
    ax.set_xticks(tick_loc)
    ax.set_xticklabels(tick_labels_pct)
    ax.set_yticks(tick_loc)
    ax.set_yticklabels(tick_labels_pct)
    lo, hi = _probit(0.001), _probit(0.55)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("APCER (%)")
    ax.set_ylabel("BPCER (%)")
    ax.grid(True, ls="--", lw=0.4, alpha=0.6)
    ax.set_aspect("equal")

    # iso-EER diagonal
    diag = np.linspace(lo, hi, 100)
    ax.plot(diag, diag, color="grey", lw=0.7, ls=":", alpha=0.6)

    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.92,
              title=r"$\bigstar$ = min-ACER operating point",
              title_fontsize=7.5)
    ax.set_title("DET curves, fused pipeline (ISO/IEC 30107-3 §7)",
                 fontsize=10)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[det] wrote {args.out}")
    print("[det] summary (label, n, tau*, APCER, BPCER, ACER):")
    for row in summary:
        print(f"  {row[0]:<28}  n={row[1]:>5}  tau*={row[2]:.4f}  "
              f"APCER={row[3]:.3f}  BPCER={row[4]:.3f}  ACER={row[5]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
