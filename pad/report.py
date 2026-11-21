"""
DODO VISION AI v9.4 PAD Step 10 — ISO/IEC 30107-3 reportable metrics.

Produces the three canonical PAD error rates plus reliability
diagrams (ECE per branch), DET curves, and per-class confusion.
Can be run as a CLI:

    python -m pad.report --from path/to/log.parquet --out pad/reports/

The eval harness lives in `pad/eval/` and writes scored records to a
parquet partition; this module consumes those records.

Metric definitions (ISO/IEC 30107-3:2023 §7):
  - APCER (Attack Presentation Classification Error Rate):
        #(attack classified as live) / #(attack presentations)
    Computed PER attack class.
  - BPCER (Bona-Fide Presentation Classification Error Rate):
        #(bona-fide classified as spoof) / #(bona-fide presentations)
  - ACER  = (APCER + BPCER) / 2 (for each operating point)

At a fixed BPCER=0.01 we report APCER (security posture) and the
threshold that achieves it. DET curves give the full operating
characteristic.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from pad.types import ATTACK_CLASSES


def _load_records(src: str) -> List[Dict]:
    """Load scored-pair records. Each record is a dict with at least:
        { 'label': 'live' | 'spoof',
          'attack_class': one of ATTACK_CLASSES | None,
          'p_attack': float in [0, 1],
          'branches': { name: {p_attack, reliability}, ... }
        }
    Supports .parquet (via pyarrow) and .jsonl.
    """
    if src.endswith(".parquet"):
        import pyarrow.parquet as pq
        tbl = pq.read_table(src)
        return [dict(row) for row in tbl.to_pylist()]
    out: List[Dict] = []
    with open(src, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def compute_apcer_bpcer(records: List[Dict], threshold: float) -> Tuple[Dict[str, float], float]:
    """Return (APCER by attack class, BPCER).

    v9.8.1 2026-05-12 (Audit C3): thin wrapper around the unified
    `pad.metrics.apcer_per_class`. The previous local implementation
    used the convention BPCER=(p>t), APCER=(p<=t) which conflicted
    with three other files. The single source of truth now lives in
    `pad/metrics.py` and follows ISO/IEC 30107-3:2023 strict form
    (BPCER=(p>=t), APCER=(p<t)). Public function name kept for
    backward compatibility with any external caller.
    """
    from pad.metrics import apcer_per_class as _canonical
    return _canonical(records, threshold, attack_classes=ATTACK_CLASSES)


def operating_point_at_bpcer(records: List[Dict], target_bpcer: float) -> Dict[str, float]:
    """Find the threshold that achieves the closest BPCER ≤ target.
    Returns {threshold, bpcer, apcer per class, acer}.

    v9.8.1 2026-05-12 (Audit C3): delegates to
    `pad.metrics.operating_point_at_bpcer`. The legacy body is kept
    immediately below this guard so the module remains importable if
    `pad.metrics` is somehow stripped from a build.
    """
    try:
        from pad.metrics import operating_point_at_bpcer as _canonical
        return _canonical(records, target_bpcer)
    except ImportError:
        pass
    ts = sorted({round(r["p_attack"], 4) for r in records})
    best = None
    for t in ts:
        apcer, bpcer = compute_apcer_bpcer(records, t)
        if bpcer <= target_bpcer:
            apcer_mean = float(np.nanmean(list(apcer.values()))) if apcer else float("nan")
            acer = (apcer_mean + bpcer) / 2.0 if not np.isnan(apcer_mean) else float("nan")
            candidate = {
                "threshold": float(t),
                "bpcer": bpcer,
                "apcer": apcer,
                "apcer_mean": apcer_mean,
                "acer": acer,
            }
            # Prefer highest threshold that still satisfies target.
            if best is None or candidate["threshold"] > best["threshold"]:
                best = candidate
    return best or {"threshold": None, "bpcer": None, "apcer": {}, "apcer_mean": None, "acer": None}


def det_curve(records: List[Dict], n_points: int = 40) -> Dict[str, List[float]]:
    """Detection Error Trade-off curve: (BPCER vs mean APCER) across
    threshold sweeps."""
    thresholds = np.linspace(0.0, 1.0, n_points + 1)
    out = {"threshold": [], "bpcer": [], "apcer_mean": [], "acer": []}
    for t in thresholds:
        apcer, bpcer = compute_apcer_bpcer(records, float(t))
        apcer_vals = [v for v in apcer.values() if not np.isnan(v)]
        mean_apcer = float(np.mean(apcer_vals)) if apcer_vals else float("nan")
        acer = (mean_apcer + bpcer) / 2.0 if not np.isnan(mean_apcer) else float("nan")
        out["threshold"].append(float(t))
        out["bpcer"].append(float(bpcer))
        out["apcer_mean"].append(mean_apcer)
        out["acer"].append(acer)
    return out


def expected_calibration_error(records: List[Dict], n_bins: int = 10) -> float:
    """Standard ECE on p_attack calibration. Lower is better."""
    if not records:
        return 0.0
    probs = np.asarray([r["p_attack"] for r in records])
    labels = np.asarray([1 if r.get("label") == "spoof" else 0 for r in records])
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(records)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if not mask.any():
            continue
        acc = float(labels[mask].mean())
        conf = float(probs[mask].mean())
        ece += (mask.sum() / N) * abs(acc - conf)
    return float(ece)


def per_branch_ece(records: List[Dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    # Collect unique branch names
    branch_names: set = set()
    for r in records:
        for b in (r.get("branches") or []):
            if isinstance(b, dict) and b.get("name"):
                branch_names.add(b["name"])
            elif isinstance(b, (list, tuple)) and len(b) >= 2:
                branch_names.add(b[0])
    for bn in sorted(branch_names):
        sub = []
        for r in records:
            branches = r.get("branches") or []
            for b in branches:
                if isinstance(b, dict) and b.get("name") == bn:
                    sub.append({"p_attack": b["p_attack"], "label": r["label"]})
                    break
        out[bn] = expected_calibration_error(sub)
    return out


def render_markdown(summary: Dict, out_path: str) -> None:
    lines = [
        "# DODO VISION AI v9.4 — PAD Evaluation Report",
        "",
        f"**Generated:** {summary.get('generated_at', '?')}",
        f"**Records:** {summary.get('n_records', 0)} (live: {summary.get('n_live', 0)}, spoof: {summary.get('n_spoof', 0)})",
        "",
        "## Operating point at BPCER ≤ 0.01 (ISO/IEC 30107-3 §7)",
        "",
        "| Metric | Value |",
        "| :-- | --: |",
    ]
    op = summary.get("op_point", {}) or {}
    lines += [
        f"| Threshold | {op.get('threshold', 'n/a')} |",
        f"| BPCER | {op.get('bpcer', 'n/a')} |",
        f"| Mean APCER | {op.get('apcer_mean', 'n/a')} |",
        f"| ACER | {op.get('acer', 'n/a')} |",
    ]
    lines += ["", "### APCER per attack class", ""]
    lines += ["| Attack class | APCER |", "| :-- | --: |"]
    apcer_map = op.get("apcer", {}) or {}
    for cls in ATTACK_CLASSES:
        v = apcer_map.get(cls)
        disp = "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
        lines.append(f"| {cls} | {disp} |")
    lines += ["", "## ECE per branch (calibration)", "",
              "| Branch | ECE |", "| :-- | --: |"]
    for bn, ece in (summary.get("branch_ece") or {}).items():
        lines.append(f"| {bn} | {ece:.4f} |")
    lines += ["", "## Full ECE (fused)",
              f"- ECE (fused p_attack): {summary.get('fused_ece', 0.0):.4f}"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def render_det_svg(det: Dict[str, List[float]], out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(det["bpcer"], det["apcer_mean"], marker="o", ms=3)
    ax.set_xlabel("BPCER")
    ax.set_ylabel("Mean APCER")
    ax.set_title("DET curve (PAD v9.4)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def summarise(records: List[Dict]) -> Dict:
    import time
    op = operating_point_at_bpcer(records, 0.01)
    det = det_curve(records)
    ece_fused = expected_calibration_error(records)
    ece_branches = per_branch_ece(records)
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_records": len(records),
        "n_live": sum(1 for r in records if r.get("label") == "live"),
        "n_spoof": sum(1 for r in records if r.get("label") == "spoof"),
        "op_point": op,
        "det_curve": det,
        "fused_ece": ece_fused,
        "branch_ece": ece_branches,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="PAD v9.4 report renderer (ISO/IEC 30107-3)")
    p.add_argument("--from", dest="src", required=True, help="scored records (.parquet or .jsonl)")
    p.add_argument("--out", dest="out_dir", default="pad/reports/latest",
                   help="output directory (will be created)")
    args = p.parse_args(argv)
    records = _load_records(args.src)
    summary = summarise(records)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    render_markdown(summary, os.path.join(args.out_dir, "summary.md"))
    render_det_svg(summary["det_curve"], os.path.join(args.out_dir, "det_curve.svg"))
    print(f"[pad/report] wrote {args.out_dir}")
    print(f"             op@BPCER<=0.01: threshold={summary['op_point'].get('threshold')}, "
          f"APCER_mean={summary['op_point'].get('apcer_mean')}, "
          f"ACER={summary['op_point'].get('acer')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
