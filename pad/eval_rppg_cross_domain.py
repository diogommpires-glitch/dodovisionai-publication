"""pad/eval_rppg_cross_domain.py  --  API_unq P-eval-cross-domain.

PhysDG (CVPR 2026)-style domain-generalization evaluation for rPPG /
liveness scores. Given a meta_train_records jsonl with `source_tag`
fields (e.g. CelebA-Spoof / Replay-Attack / CelebDF / FF++), this
script:

  1) Partitions records by source_tag (the "domain")
  2) Computes per-domain rppg-branch metrics:
        - score distribution stats (mean/std/min/max)
        - APCER / BPCER / ACER at threshold 0.5
        - HR-band power fraction histogram (when available)
  3) Computes leave-one-source-out generalization deltas
  4) Writes reports/rppg_cross_domain_v9.8.1.md

This is a *liveness-score* cross-domain audit, not a HR-MAE eval (which
requires HR ground truth and is reserved for the FacePhys-style trainer
when that lands). Still surfaces the basic generalization problem
PhysDG benchmarks attack: how stable is the rppg branch's output across
camera / lighting / dataset domains?

Usage:
    python -m pad.eval_rppg_cross_domain
    python -m pad.eval_rppg_cross_domain --records reports/meta_train_records_merged_v6.jsonl
    python -m pad.eval_rppg_cross_domain --out reports/rppg_cross_domain_v9.8.1.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_RECORDS = REPO / "reports" / "meta_train_records_merged_v6.jsonl"
DEFAULT_OUT = REPO / "reports" / "rppg_cross_domain_v9.8.1.md"


def _source_from_path(path: str) -> str:
    """Heuristic source-tag inference from a JSONL record's path field.
    Handles the canonical dataset folder names in E:\\pad_datasets\\."""
    p = path.lower().replace("\\", "/")
    if "celeba" in p:
        return "CelebA-Spoof"
    if "replay" in p:
        return "Replay-Attack"
    if "celeb-df" in p or "celebdf" in p:
        return "CelebDF"
    if "ffpp" in p or "faceforensics" in p or "ff++" in p:
        return "FF++"
    if "asvspoof" in p or "wav" in p:
        return "ASVspoof"
    if "oulu" in p:
        return "OULU-NPU"
    if "siw" in p:
        return "SiW-Mv2"
    return "Unknown"


def _branch_score(rec: Dict, branch: str) -> float | None:
    """Pull the named branch's p_attack from a record. Handles both
    {branches: [{name, p_attack, ...}, ...]} and a flat {<branch>:p}
    layout.
    """
    branches = rec.get("branches")
    if isinstance(branches, list):
        for b in branches:
            if b.get("name") == branch and b.get("p_attack") is not None:
                return float(b["p_attack"])
    if isinstance(branches, dict):
        v = branches.get(branch)
        if isinstance(v, dict) and v.get("p_attack") is not None:
            return float(v["p_attack"])
    flat = rec.get(branch)
    if isinstance(flat, (int, float)):
        return float(flat)
    return None


def _apcer_bpcer_acer(scores: List[float], labels: List[int],
                     threshold: float = 0.5) -> Dict[str, float]:
    """ISO/IEC 30107-3 strict-form APCER/BPCER/ACER at threshold.
    labels: 0 = live, 1 = spoof. APCER = attack_as_live, BPCER = live_as_attack.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if s.size == 0 or y.size == 0:
        return {"apcer": float("nan"), "bpcer": float("nan"),
                "acer": float("nan"), "n": 0}
    is_attack = y == 1
    is_live = y == 0
    # Strict: p >= threshold -> classified attack
    flagged_attack = s >= threshold
    apcer = float(np.mean(~flagged_attack[is_attack])) if is_attack.any() else float("nan")
    bpcer = float(np.mean(flagged_attack[is_live])) if is_live.any() else float("nan")
    if not math.isnan(apcer) and not math.isnan(bpcer):
        acer = 0.5 * (apcer + bpcer)
    else:
        acer = float("nan")
    return {"apcer": apcer, "bpcer": bpcer, "acer": acer,
            "n": int(s.size), "n_attack": int(is_attack.sum()),
            "n_live": int(is_live.sum())}


def _auc(scores: List[float], labels: List[int]) -> float:
    """ROC AUC via the rank-sum formulation. Returns NaN when degenerate."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, s.size + 1)
    sum_pos_ranks = float(ranks[pos].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _basic_stats(arr: List[float]) -> Dict[str, float]:
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"),
                "median": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()),
            "median": float(np.median(a)), "n": int(a.size)}


def evaluate(records_path: Path, out_path: Path, branch: str = "rppg",
             threshold: float = 0.5) -> int:
    if not records_path.is_file():
        print(f"[rppg-eval] records file not found: {records_path}")
        return 2
    rows: List[Dict] = []
    with records_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            label_str = str(rec.get("label", "")).lower()
            if label_str not in ("live", "spoof"):
                continue
            label = 0 if label_str == "live" else 1
            p = _branch_score(rec, branch)
            if p is None:
                continue
            source = (str(rec.get("source_tag", ""))
                      or _source_from_path(str(rec.get("path", ""))))
            rows.append({
                "label": label,
                "p_attack": float(p),
                "source": source,
            })
    if not rows:
        print(f"[rppg-eval] no usable rows for branch={branch}")
        return 3
    print(f"[rppg-eval] loaded {len(rows)} rows for branch={branch}")

    # By-source breakdown
    by_source: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    # Per-domain metrics
    per_domain_metrics: Dict[str, Dict] = {}
    for src, src_rows in by_source.items():
        scores = [r["p_attack"] for r in src_rows]
        labels = [r["label"] for r in src_rows]
        m = _apcer_bpcer_acer(scores, labels, threshold=threshold)
        m["auc"] = _auc(scores, labels)
        m["score_stats"] = _basic_stats(scores)
        m["live_score_stats"] = _basic_stats(
            [r["p_attack"] for r in src_rows if r["label"] == 0])
        m["spoof_score_stats"] = _basic_stats(
            [r["p_attack"] for r in src_rows if r["label"] == 1])
        per_domain_metrics[src] = m

    # Leave-one-source-out generalization
    all_sources = sorted(by_source.keys())
    looso_deltas: Dict[str, Dict] = {}
    for src in all_sources:
        in_rows = [r for r in rows if r["source"] != src]
        out_rows = by_source[src]
        in_m = _apcer_bpcer_acer([r["p_attack"] for r in in_rows],
                                 [r["label"] for r in in_rows],
                                 threshold=threshold)
        out_m = _apcer_bpcer_acer([r["p_attack"] for r in out_rows],
                                  [r["label"] for r in out_rows],
                                  threshold=threshold)
        looso_deltas[src] = {
            "in_acer": in_m["acer"],
            "out_acer": out_m["acer"],
            "delta": (out_m["acer"] - in_m["acer"]
                      if not math.isnan(out_m["acer"]) and not math.isnan(in_m["acer"])
                      else float("nan")),
        }

    # Render markdown
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# rPPG branch cross-domain evaluation (PhysDG-style)")
    lines.append("")
    lines.append(f"- branch: `{branch}`")
    lines.append(f"- records: `{records_path.name}`")
    lines.append(f"- threshold: {threshold:.2f} (ISO/IEC 30107-3 strict form)")
    lines.append(f"- n rows: {len(rows)}")
    lines.append("")
    lines.append("## Per-domain ISO 30107-3 metrics")
    lines.append("")
    lines.append("| Domain | n | n_live | n_spoof | APCER | BPCER | ACER | AUC | "
                 "live-mean | spoof-mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for src in all_sources:
        m = per_domain_metrics[src]
        lines.append(
            f"| {src} | {m['n']} | {m['n_live']} | {m['n_attack']} | "
            f"{m['apcer']:.4f} | {m['bpcer']:.4f} | {m['acer']:.4f} | "
            f"{m['auc']:.4f} | "
            f"{m['live_score_stats']['mean']:.3f} | "
            f"{m['spoof_score_stats']['mean']:.3f} |"
        )
    lines.append("")
    lines.append("## Leave-one-source-out generalization")
    lines.append("")
    lines.append("ACER computed on the held-out source vs ACER on the remaining "
                 "sources. Positive delta = the held-out domain is harder.")
    lines.append("")
    lines.append("| Held-out | in-domain ACER | out-of-domain ACER | Δ |")
    lines.append("|---|---:|---:|---:|")
    for src in all_sources:
        d = looso_deltas[src]
        lines.append(
            f"| {src} | {d['in_acer']:.4f} | {d['out_acer']:.4f} | "
            f"{d['delta']:+.4f} |"
        )
    lines.append("")
    lines.append("## Score distributions")
    lines.append("")
    for src in all_sources:
        m = per_domain_metrics[src]
        lines.append(f"### {src}")
        ls = m["live_score_stats"]
        ss = m["spoof_score_stats"]
        lines.append(f"- LIVE   (n={ls['n']}):  mean={ls['mean']:.3f} std={ls['std']:.3f} "
                     f"min={ls['min']:.3f} med={ls['median']:.3f} max={ls['max']:.3f}")
        lines.append(f"- SPOOF  (n={ss['n']}):  mean={ss['mean']:.3f} std={ss['std']:.3f} "
                     f"min={ss['min']:.3f} med={ss['median']:.3f} max={ss['max']:.3f}")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**Method.** Per-domain metrics computed at strict ISO/IEC 30107-3:2023")
    lines.append("APCER (attack-as-live rate) and BPCER (live-as-attack rate) with")
    lines.append("threshold = `p_attack >= 0.5`. AUC via rank-sum formulation. The")
    lines.append("LOSO panel mimics the PhysDG (CVPR 2026) protocol: train (i.e. fit)")
    lines.append("on N-1 sources, eval on the held-out source. For the rppg branch")
    lines.append("specifically this audit is fit-free -- it pulls the runtime p_attack")
    lines.append("directly from the records jsonl -- so the LOSO deltas reflect domain")
    lines.append("invariance of the calibrator + branch fusion rather than a deep model.")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[rppg-eval] wrote {out_path}  ({len(rows)} rows, "
          f"{len(all_sources)} domains)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--branch", type=str, default="rppg")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    return evaluate(args.records, args.out, branch=args.branch,
                    threshold=args.threshold)


if __name__ == "__main__":
    sys.exit(main())
