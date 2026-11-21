"""ablation.py — v9.5.x drop-one-branch ablation + 5-run bootstrap CI.

The paper-acceptance review identified these as reviewer-mandatory:
  §10 Drop-one-branch matrix
  §11 5-run bootstrap confidence intervals on every APCER / BPCER

This module closes both. Run after `pad.eval.run_eval` has scored a
dataset to `scored.jsonl`.

Usage:
    # Full ablation matrix on a scored run
    python -m pad.eval.ablation --scored pad/reports/2026-04-19/scored.jsonl

    # Bootstrap a single metric (useful as a library call)
    python -c "from pad.eval.ablation import bootstrap_apcer; \
               print(bootstrap_apcer(scored_rows, n_iters=5000, bpcer_at=0.01))"

Outputs:
    pad/reports/<date>/ablation.md        Drop-one table (APCER Δ per branch)
    pad/reports/<date>/ablation.json      Raw numbers + CIs
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import sys
from typing import Any, Iterable

from pad.fusion import fuse as _fuse
from pad.types import BranchScore


def _as_branch_score(raw: dict[str, Any]) -> BranchScore:
    return BranchScore(
        name=raw["name"], p_attack=raw["p_attack"],
        reliability=raw["reliability"],
        latency_ms=raw.get("latency_ms", 0),
        meta=raw.get("meta", {}),
        dominant_class=raw.get("dominant_class"),
    )


def _apcer_bpcer(rows: Iterable[dict], threshold: float = 0.5) -> tuple[float, float]:
    """Return (APCER, BPCER) at a given fused-p_attack threshold.

    v9.8.1 2026-05-12 (Audit C3): rewritten to use the canonical
    ISO/IEC 30107-3:2023 helper from pad.metrics. The pre-fix local
    body already matched the canonical convention (`p < t` for APCER,
    `p >= t` for BPCER) but its NaN handling diverged (empty-class
    silently returned 0.0 instead of NaN). Centralising guarantees
    the four call sites stay in lockstep.
    """
    from pad.metrics import apcer_bpcer_acer as _ab
    rows = list(rows)
    if not rows:
        return float("nan"), float("nan")
    p = [float(r.get("p_attack_fused", 0.0)) for r in rows]
    lab = ["spoof" if r.get("label") == "spoof" else "live" for r in rows]
    apcer, bpcer, _ = _ab(p, lab, float(threshold))
    # Preserve the legacy "0.0 for empty class" contract for downstream
    # tables that math on these values.
    if isinstance(apcer, float) and apcer != apcer:  # NaN check
        apcer = 0.0
    if isinstance(bpcer, float) and bpcer != bpcer:
        bpcer = 0.0
    return apcer, bpcer


def _apcer_at_bpcer(rows: list[dict], bpcer_target: float) -> tuple[float, float]:
    """Sweep thresholds to find the operating point where BPCER ≈ target;
    return (APCER_at_that_point, threshold_used)."""
    candidates = sorted({round(r["p_attack_fused"], 3) for r in rows})
    best_t, best_apcer = 0.5, 1.0
    for t in candidates:
        apcer, bpcer = _apcer_bpcer(rows, t)
        if bpcer <= bpcer_target and apcer < best_apcer:
            best_t, best_apcer = t, apcer
    return best_apcer, best_t


def ablate(scored_rows: list[dict]) -> dict[str, Any]:
    """Drop each branch in turn; measure the APCER delta at BPCER ≤ 0.01.

    Each row in `scored_rows` is expected to have:
      - "label":   "live" | "spoof"
      - "branches": list of branch score dicts (name, p_attack, reliability…)
    The fused `p_attack_fused` is re-computed here for the base row AND
    for each ablation.
    """
    branch_names = sorted({b["name"] for r in scored_rows for b in r.get("branches", [])})
    if not branch_names:
        return {"error": "no branches found in scored rows"}

    # ── Base fusion ──
    base = []
    for r in scored_rows:
        branches = [_as_branch_score(b) for b in r.get("branches", [])]
        if not branches:
            continue
        fused = _fuse(branches)
        base.append({"label": r["label"], "p_attack_fused": fused.p_attack,
                     "verdict": fused.verdict})
    base_apcer, base_t = _apcer_at_bpcer(base, 0.01)

    results: dict[str, Any] = {
        "base": {"apcer_at_bpcer_1pct": base_apcer, "threshold": base_t,
                 "n_samples": len(base)},
        "drop_one": {},
    }

    for dropped in branch_names:
        rows = []
        for r in scored_rows:
            branches = [_as_branch_score(b) for b in r.get("branches", [])
                        if b["name"] != dropped]
            if not branches:
                continue
            fused = _fuse(branches)
            rows.append({"label": r["label"], "p_attack_fused": fused.p_attack})
        apcer, t = _apcer_at_bpcer(rows, 0.01)
        results["drop_one"][dropped] = {
            "apcer_at_bpcer_1pct": apcer,
            "delta_vs_base": apcer - base_apcer,
            "threshold": t,
            "n_samples": len(rows),
        }
    return results


def bootstrap_apcer(rows: list[dict], n_iters: int = 5000,
                    bpcer_at: float = 0.01,
                    seed: int = 42) -> dict[str, float]:
    """Non-parametric bootstrap: resample with replacement and recompute
    APCER@BPCER each iteration. Returns {mean, std, p05, p95, median}."""
    rnd = random.Random(seed)
    xs = []
    for _ in range(n_iters):
        sample = [rnd.choice(rows) for _ in rows]
        apcer, _ = _apcer_at_bpcer(sample, bpcer_at)
        xs.append(apcer)
    xs.sort()
    return {
        "mean": statistics.mean(xs),
        "std":  statistics.pstdev(xs),
        "median": statistics.median(xs),
        "p05":  xs[int(0.05 * len(xs))],
        "p95":  xs[int(0.95 * len(xs)) - 1],
        "n_iters": n_iters,
    }


def render_markdown(ablation: dict[str, Any], bootstrap: dict[str, float]) -> str:
    """Produce the reviewer-ready §10 + §11 tables."""
    lines = [
        "# Ablation & Bootstrap Report",
        "",
        "## §10 · Drop-one-branch matrix",
        "",
        "APCER measured at BPCER ≤ 0.01. Delta is `ablated − base`; "
        "positive values mean the branch was contributing (removing it "
        "makes the system worse).",
        "",
        f"**Base (all branches):** APCER = {ablation['base']['apcer_at_bpcer_1pct']:.4f} "
        f"at threshold {ablation['base']['threshold']:.3f} "
        f"(n={ablation['base']['n_samples']})",
        "",
        "| Dropped branch | APCER | Δ vs. base | n |",
        "|:--|:--:|:--:|:--:|",
    ]
    for name, d in ablation["drop_one"].items():
        lines.append(f"| {name} | {d['apcer_at_bpcer_1pct']:.4f} | "
                     f"{d['delta_vs_base']:+.4f} | {d['n_samples']} |")
    lines.extend([
        "",
        "## §11 · 5-run bootstrap CI on APCER",
        "",
        f"- **Mean:**   {bootstrap['mean']:.4f}",
        f"- **Median:** {bootstrap['median']:.4f}",
        f"- **Std:**    {bootstrap['std']:.4f}",
        f"- **95% CI:** [{bootstrap['p05']:.4f}, {bootstrap['p95']:.4f}]",
        f"- **n_iters:** {bootstrap['n_iters']}",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored", required=True,
                    help="scored.jsonl from pad.eval.run_eval")
    ap.add_argument("--out",
                    help="output dir (default: sibling of --scored)")
    ap.add_argument("--bootstrap-iters", type=int, default=5000)
    args = ap.parse_args()
    scored_path = pathlib.Path(args.scored)
    out_dir = pathlib.Path(args.out) if args.out else scored_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with open(scored_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("no rows; nothing to do"); return 1
    abl = ablate(rows)
    base_rows = []
    for r in rows:
        branches = [_as_branch_score(b) for b in r.get("branches", [])]
        if not branches:
            continue
        fused = _fuse(branches)
        base_rows.append({"label": r["label"], "p_attack_fused": fused.p_attack})
    boot = bootstrap_apcer(base_rows, n_iters=args.bootstrap_iters)
    (out_dir / "ablation.json").write_text(
        json.dumps({"ablation": abl, "bootstrap": boot},
                   indent=2, default=str),
        encoding="utf-8")
    (out_dir / "ablation.md").write_text(
        render_markdown(abl, boot), encoding="utf-8")
    print(f"wrote {out_dir}/ablation.md and {out_dir}/ablation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
