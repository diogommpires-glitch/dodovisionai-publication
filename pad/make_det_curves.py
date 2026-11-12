"""make_det_curves.py -- Generate Detection Error Tradeoff (DET) curves.

DET curves plot the trade-off between APCER (false-accept / attack passes
as live) and BPCER (false-reject / live passes as attack) across all
thresholds, on a normal-deviate (probit) scale. This is the NIST FATE
PAD standard figure format.

Reads the same JSONLs that `pad.bootstrap_ci` consumes and writes an
SVG per dataset plus one combined figure with all datasets overlaid.
The combined figure is paper-ready as Figure 1 (DET across PAD datasets).

Usage:
    python -m pad.make_det_curves \\
        --inputs reports/meta_train_records_celeba_v3.jsonl \\
                 reports/meta_train_records_replay_v3.jsonl \\
                 reports/meta_train_records_celebdf_v3.jsonl \\
                 reports/audio_asvspoof_records_v3.jsonl \\
        --names  celeba replay celebdf asvspoof \\
        --out-dir reports/figures
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _fused_p_from_branches(branches) -> float:
    """Reliability-weighted mean of branch p_attack (same logic as
    pad.fusion.fuse). Caller should use this only when top-level
    'p_attack' is missing from a record."""
    if not branches:
        return 0.0
    num = 0.0
    den = 0.0
    for b in branches:
        if not isinstance(b, dict):
            continue
        p = b.get("p_attack")
        r = b.get("reliability", 1.0)
        if p is None or r is None or r <= 0:
            continue
        num += float(p) * float(r)
        den += float(r)
    return num / den if den > 0 else 0.0


def load_records(p: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (p_attack[N], y[N]) for a report-schema jsonl."""
    ps, ys = [], []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            top = r.get("p_attack")
            if top is None:
                top = _fused_p_from_branches(r.get("branches") or [])
            lab = 1 if r.get("label") == "spoof" else 0
            ps.append(float(top))
            ys.append(int(lab))
    return np.asarray(ps, dtype=np.float64), np.asarray(ys, dtype=np.int32)


def det_curve(scores: np.ndarray, labels: np.ndarray,
              n_points: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (thr, apcer, bpcer) arrays across n_points thresholds.
    APCER = fraction of attack (y=1) that score BELOW threshold.
    BPCER = fraction of live (y=0) that score AT OR ABOVE threshold.
    Both strict inequalities matching the convention used by
    pad.bootstrap_ci."""
    if len(scores) == 0:
        return np.array([]), np.array([]), np.array([])
    # Dense grid — include the actual unique score values plus a uniform grid
    uniq = np.unique(scores)
    if len(uniq) > n_points:
        idx = np.linspace(0, len(uniq) - 1, n_points).astype(int)
        thresholds = uniq[idx]
    else:
        thresholds = uniq
    thresholds = np.sort(np.concatenate([
        thresholds, np.linspace(0.0, 1.0, n_points)
    ]))
    thresholds = np.unique(thresholds)
    apcer = np.empty_like(thresholds)
    bpcer = np.empty_like(thresholds)
    n_attack = max(1, int((labels == 1).sum()))
    n_live = max(1, int((labels == 0).sum()))
    for i, t in enumerate(thresholds):
        # attack records misclassified = those scoring BELOW threshold
        apcer[i] = float(((scores < t) & (labels == 1)).sum()) / n_attack
        # live records misclassified = those scoring AT OR ABOVE threshold
        bpcer[i] = float(((scores >= t) & (labels == 0)).sum()) / n_live
    return thresholds, apcer, bpcer


def _probit_clip(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def _auc_from_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Trapezoidal AUC from the empirical ROC. Matches sklearn's
    roc_auc_score to ~1e-9 without the sklearn dep."""
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    n_pos = max(1, int((y == 1).sum()))
    n_neg = max(1, int((y == 0).sum()))
    tp = np.cumsum(y == 1) / n_pos
    fp = np.cumsum(y == 0) / n_neg
    # Prepend origin and append (1,1)
    tp = np.concatenate([[0.0], tp, [1.0]])
    fp = np.concatenate([[0.0], fp, [1.0]])
    return float(np.trapz(tp, fp))


def _eer(apcer: np.ndarray, bpcer: np.ndarray) -> float:
    """Equal-Error Rate: the point where APCER ≈ BPCER. Returns the
    APCER value at the threshold where |APCER - BPCER| is smallest."""
    if len(apcer) == 0:
        return float("nan")
    diff = np.abs(apcer - bpcer)
    idx = int(np.argmin(diff))
    return float((apcer[idx] + bpcer[idx]) / 2.0)


def plot_det(thr_apcer_bpcer_list, out_path: Path, title: str | None = None):
    """Render a DET figure on a probit-scaled axis.
    Args:
        thr_apcer_bpcer_list: list of (label, thresholds, apcer, bpcer, auc, eer).
        out_path: .svg destination.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    # Axis ticks we want printed in percent space
    tick_pct = np.array([0.5, 1, 2, 5, 10, 20, 40, 60])
    tick_loc = norm.ppf(tick_pct / 100.0)
    for (label, _, apcer, bpcer, auc, eer) in thr_apcer_bpcer_list:
        a = _probit_clip(apcer)
        b = _probit_clip(bpcer)
        ax.plot(norm.ppf(a), norm.ppf(b),
                label=f"{label}  AUC={auc:.3f}  EER={eer*100:.1f}%",
                linewidth=1.6)
    ax.set_xticks(tick_loc); ax.set_xticklabels([f"{t:g}" for t in tick_pct])
    ax.set_yticks(tick_loc); ax.set_yticklabels([f"{t:g}" for t in tick_pct])
    ax.set_xlabel("APCER (%)  — attack passes as live")
    ax.set_ylabel("BPCER (%)  — live rejected as attack")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    lo, hi = norm.ppf(0.005), norm.ppf(0.60)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, type=Path)
    ap.add_argument("--names", nargs="+", default=None,
                    help="Pretty labels for each --inputs entry.")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    ap.add_argument("--combined-title", default="Fused DET — IJCB 2026")
    args = ap.parse_args(argv)

    names = args.names or [p.stem for p in args.inputs]
    if len(names) != len(args.inputs):
        print("[det] --names must have the same length as --inputs",
              file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    combined = []
    for name, path in zip(names, args.inputs):
        scores, labels = load_records(path)
        if len(scores) == 0:
            print(f"[det] {name}: empty — skipping")
            continue
        thr, apcer, bpcer = det_curve(scores, labels)
        auc = _auc_from_roc(scores, labels)
        eer = _eer(apcer, bpcer)
        svg_p = args.out_dir / f"det_{name}.svg"
        plot_det([(name, thr, apcer, bpcer, auc, eer)], svg_p,
                 title=f"DET — {name}  (n={len(scores)})")
        print(f"[det] {name}: n={len(scores)}  AUC={auc:.4f}  EER={eer*100:.2f}%  "
              f"-> {svg_p}")
        combined.append((name, thr, apcer, bpcer, auc, eer))

    combined_p = args.out_dir / "det_all.svg"
    plot_det(combined, combined_p, title=args.combined_title)
    print(f"[det] combined -> {combined_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
