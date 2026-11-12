"""make_reliability_diagrams.py -- per-branch calibration curves.

A reliability diagram plots predicted probability (x-axis, binned) against
the empirical frequency of the positive class within each bin (y-axis).
A perfectly-calibrated classifier lies on the y=x diagonal; Expected
Calibration Error (ECE) is the weighted L1 distance from the diagonal
(Guo et al. 2017, "On Calibration of Modern Neural Networks").

This script walks each PAD branch, plots the raw (pre-calibration) and
calibrated (post-calibration) reliability curves on the same axes, and
emits one SVG per branch plus a combined grid figure. Intended as paper
Figure 2 (Calibration Before/After per Branch).

Usage:
    python -m pad.make_reliability_diagrams \\
        --inputs reports/meta_train_records_celeba_v3.jsonl \\
                 reports/meta_train_records_replay_v3.jsonl \\
                 reports/meta_train_records_celebdf_v3.jsonl \\
                 reports/audio_asvspoof_records_v3.jsonl \\
        --out-dir reports/figures
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _load_records(p: Path) -> List[Dict]:
    out = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def collect_per_branch(records: List[Dict]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """From records, extract {branch_name: (raw_p[], labels[])} .
    The labels are 1 for spoof, 0 for live."""
    raw: Dict[str, List[float]] = {}
    lab: Dict[str, List[int]] = {}
    for r in records:
        y = 1 if r.get("label") == "spoof" else 0
        for b in (r.get("branches") or []):
            if not isinstance(b, dict):
                continue
            name = b.get("name")
            p = b.get("p_attack")
            rel = b.get("reliability", 1.0)
            if name is None or p is None or rel is None or rel <= 0:
                continue
            raw.setdefault(name, []).append(float(p))
            lab.setdefault(name, []).append(y)
    return {n: (np.asarray(raw[n], dtype=np.float64),
                np.asarray(lab[n], dtype=np.int32))
            for n in raw}


def calibrate(branch: str, raw_p: np.ndarray) -> np.ndarray:
    from pad.calibrate import apply_calibration
    return np.asarray([apply_calibration(branch, float(p)) for p in raw_p],
                      dtype=np.float64)


def reliability_bins(p: np.ndarray, y: np.ndarray,
                     n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, emp_freq, bin_counts) for the classic equal-
    width reliability diagram over [0,1]. bin_centers is the mean
    predicted probability within each bin, emp_freq is the fraction of
    y=1 within that bin, bin_counts is the number of records per bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    centers = np.zeros(n_bins)
    emp = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        mask = (idx == i)
        n_i = int(mask.sum())
        counts[i] = n_i
        if n_i == 0:
            # Fall back to bin midpoint so the marker sits somewhere sensible
            centers[i] = (edges[i] + edges[i+1]) / 2.0
            emp[i] = np.nan
        else:
            centers[i] = float(p[mask].mean())
            emp[i] = float(y[mask].mean())
    return centers, emp, counts


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error per Guo et al. 2017."""
    centers, emp, counts = reliability_bins(p, y, n_bins)
    n_total = max(1, int(counts.sum()))
    ece_val = 0.0
    for c, e, n in zip(centers, emp, counts):
        if n == 0 or np.isnan(e):
            continue
        ece_val += (n / n_total) * abs(c - e)
    return float(ece_val)


def plot_branch(branch: str, raw: np.ndarray, cal: np.ndarray,
                y: np.ndarray, out_path: Path,
                n_bins: int = 10) -> Tuple[float, float, int]:
    import matplotlib.pyplot as plt
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ece_raw = ece(raw, y)
    ece_cal = ece(cal, y)

    centers_r, emp_r, counts_r = reliability_bins(raw, y, n_bins)
    centers_c, emp_c, counts_c = reliability_bins(cal, y, n_bins)

    fig, (ax, ax_h) = plt.subplots(2, 1, figsize=(4.6, 5.2),
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   sharex=True)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1, label="perfect")
    # Raw curve
    ok_r = ~np.isnan(emp_r)
    ax.plot(centers_r[ok_r], emp_r[ok_r], "o-",
            color="#d62728", linewidth=1.5, markersize=5,
            label=f"raw  ECE={ece_raw:.3f}")
    # Calibrated curve
    ok_c = ~np.isnan(emp_c)
    ax.plot(centers_c[ok_c], emp_c[ok_c], "s-",
            color="#2ca02c", linewidth=1.5, markersize=5,
            label=f"calibrated  ECE={ece_cal:.3f}")
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("empirical P(spoof)")
    ax.set_title(f"{branch}  (n={len(raw)})", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_aspect("equal", adjustable="box")

    # Histogram of raw predictions (grey, for count context)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    width = 1.0 / n_bins
    ax_h.bar(edges[:-1], counts_r, width=width, align="edge",
             color="#aaaaaa", edgecolor="white", linewidth=0.5,
             label="raw p_attack count")
    ax_h.set_xlabel("predicted P(spoof)")
    ax_h.set_ylabel("count")
    ax_h.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    import matplotlib.pyplot as _plt; _plt.close(fig)
    return ece_raw, ece_cal, int(len(raw))


def plot_grid(branch_data, out_path: Path) -> None:
    """A 2x3 grid that fits up to six branches in one figure."""
    import matplotlib.pyplot as plt
    n = len(branch_data)
    if n == 0:
        return
    ncols = 2
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows))
    axs = np.asarray(axs).reshape(-1)
    for ax, (branch, raw, cal, y) in zip(axs, branch_data):
        ece_raw = ece(raw, y); ece_cal = ece(cal, y)
        centers_r, emp_r, _ = reliability_bins(raw, y)
        centers_c, emp_c, _ = reliability_bins(cal, y)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
        ok_r = ~np.isnan(emp_r); ok_c = ~np.isnan(emp_c)
        ax.plot(centers_r[ok_r], emp_r[ok_r], "o-", color="#d62728",
                label=f"raw ECE={ece_raw:.3f}", linewidth=1.3, markersize=4)
        ax.plot(centers_c[ok_c], emp_c[ok_c], "s-", color="#2ca02c",
                label=f"cal ECE={ece_cal:.3f}", linewidth=1.3, markersize=4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{branch}  (n={len(raw)})", fontsize=10)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.95)
        ax.grid(True, ls=":", alpha=0.5)
        ax.set_aspect("equal", adjustable="box")
    for ax in axs[len(branch_data):]:
        ax.axis("off")
    fig.suptitle("Per-branch reliability diagrams — IJCB 2026", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    args = ap.parse_args(argv)

    all_records: List[Dict] = []
    for p in args.inputs:
        recs = _load_records(p)
        print(f"[rel] {p.name}: {len(recs)} records")
        all_records.extend(recs)

    # Bucket per branch across all datasets
    per_branch = collect_per_branch(all_records)
    if not per_branch:
        print("[rel] no branch records found; aborting", file=sys.stderr)
        return 1

    # Reset the calibrator cache so we read the current pkl
    from pad.calibrate import _reset_calibrators_cache
    _reset_calibrators_cache()

    branch_data = []
    grid_items = []
    for branch in sorted(per_branch.keys()):
        raw, y = per_branch[branch]
        if len(raw) == 0:
            continue
        cal = calibrate(branch, raw)
        svg_p = args.out_dir / f"reliability_{branch}.svg"
        ece_raw, ece_cal, n = plot_branch(branch, raw, cal, y, svg_p)
        print(f"[rel] {branch}: n={n}  ECE {ece_raw:.4f} -> {ece_cal:.4f}  "
              f"-> {svg_p}")
        branch_data.append((branch, ece_raw, ece_cal, n))
        grid_items.append((branch, raw, cal, y))

    grid_p = args.out_dir / "reliability_all.svg"
    plot_grid(grid_items, grid_p)
    print(f"[rel] grid -> {grid_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
