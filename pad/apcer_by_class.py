"""apcer_by_class.py -- Per-PAI APCER breakdown (ISO/IEC 30107-3 §7.1).

ISO/IEC 30107-3 requires reporting APCER per Presentation Attack
Instrument (PAI) species. Our JSONLs only carry a generic
`attack_class` field (mostly "UNKNOWN"), but the source paths encode
the PAI sub-class. This module derives the PAI from `path`, then
reports APCER at the per-dataset optimal threshold.

Output: a markdown fragment + a CSV. Intended to slot into
`PAD_REPORT.md` §3 as "APCER per attack class".

Usage:
    python -m pad.apcer_by_class \\
        --inputs reports/meta_train_records_celeba_v3.jsonl \\
                 reports/meta_train_records_replay_v3.jsonl \\
                 reports/meta_train_records_celebdf_v3.jsonl \\
                 reports/audio_asvspoof_records_v3.jsonl \\
        --names celeba replay celebdf asvspoof \\
        --out reports/apcer_by_class.md
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np


def _fused_p_from_branches(branches) -> float:
    if not branches:
        return 0.0
    num = 0.0; den = 0.0
    for b in branches:
        if not isinstance(b, dict):
            continue
        p = b.get("p_attack"); r = b.get("reliability", 1.0)
        if p is None or r is None or r <= 0:
            continue
        num += float(p) * float(r)
        den += float(r)
    return num / den if den > 0 else 0.0


def _classify_celeba(path: str, label: str) -> str:
    if label == "live":
        return "LIVE"
    # CelebA-Spoof flat layout: no sub-class discoverable from filename.
    # Mark as a generic mixed bucket so the reviewer sees the ambiguity
    # rather than claiming a class we can't substantiate.
    return "PAI_MIXED_PRINT_REPLAY_MASK"


def _classify_replay(path: str, label: str) -> str:
    p = path.lower()
    if label == "live":
        return "LIVE"
    # Replay-Attack filename tokens come directly from session ids.
    if "printed_photo" in p:   return "PRINT"
    if "mobile_photo" in p:    return "PRINT_MOBILE"
    if "highdef_photo" in p:   return "PRINT_HD"
    if "mobile_video" in p:    return "REPLAY_MOBILE"
    if "highdef_video" in p:   return "REPLAY_HD"
    if "ipad_video" in p:      return "REPLAY_TABLET"
    if "ipad_photo" in p:      return "REPLAY_TABLET_PHOTO"
    return "REPLAY_UNKNOWN"


def _classify_celebdf(path: str, label: str) -> str:
    p = path.lower()
    if label == "live":
        return "LIVE"
    if "celeb-synthesis" in p:
        return "DEEPFAKE_FACESWAP"
    return "DEEPFAKE_UNKNOWN"


def _classify_asvspoof(path: str, label: str) -> str:
    if label == "live":
        return "LIVE"
    # Without the CM protocol (ASVspoof2019.LA.cm.train.trn.txt), we
    # cannot distinguish A01-A19 sub-species. Lump under one synthetic
    # class and note the coarse-grained limitation in PAD_REPORT §9.
    return "AUDIO_SYNTHETIC"


CLASSIFIERS: Dict[str, Callable[[str, str], str]] = {
    "celeba":   _classify_celeba,
    "replay":   _classify_replay,
    "celebdf":  _classify_celebdf,
    "asvspoof": _classify_asvspoof,
}


def load_records(p: Path):
    """Read a JSONL of pad.report-format records, skipping blank or
    malformed lines silently. Returns a list of dicts. Used by the
    apcer-by-class CLI to compute per-attack-class break-down tables."""
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


def optimal_threshold(scores: np.ndarray,
                      labels: np.ndarray) -> Tuple[float, float, float, float]:
    """Pick the fused-fusion threshold that minimises ACER.

    v9.8.1 2026-05-12 (Audit C3): the local apcer/bpcer math is
    rewritten to call `pad.metrics.apcer_bpcer_acer` so the threshold
    convention (ISO/IEC 30107-3:2023 §7 strict form) is identical to
    every other report in the repo. The candidate-threshold grid
    expansion (quantile bracket + unique scores) is preserved.
    """
    if len(scores) == 0:
        return 0.5, float("nan"), float("nan"), float("nan")
    from pad.metrics import apcer_bpcer_acer as _ab
    uniq = np.unique(scores)
    if len(uniq) > 400:
        idx = np.linspace(0, len(uniq) - 1, 400).astype(int)
        candidates = uniq[idx]
    else:
        candidates = uniq
    candidates = np.unique(np.concatenate(
        [candidates, np.linspace(0.0, 1.0, 200)]))
    best = (0.5, 1.0, 1.0, 1.0)
    for t in candidates:
        apcer, bpcer, acer = _ab(scores, labels, float(t))
        # NaN-safe replacement of legacy worst-case sentinel.
        a_val = 1.0 if np.isnan(apcer) else apcer
        b_val = 1.0 if np.isnan(bpcer) else bpcer
        ac_val = 1.0 if np.isnan(acer) else acer
        if ac_val < best[3]:
            best = (float(t), a_val, b_val, ac_val)
    return best


def compute_table(inputs: List[Path], names: List[str]):
    """Build the per-(input, class) APCER summary table. Returns a list of dict rows ready for `format_markdown`."""
    rows = []
    for name, path in zip(names, inputs):
        recs = load_records(path)
        if not recs:
            continue
        cls_fn = CLASSIFIERS.get(name, _classify_celeba)
        scores = np.asarray([
            r.get("p_attack") if r.get("p_attack") is not None
            else _fused_p_from_branches(r.get("branches") or [])
            for r in recs
        ], dtype=np.float64)
        labels = np.asarray([
            1 if r.get("label") == "spoof" else 0 for r in recs
        ], dtype=np.int32)
        pai = np.asarray([cls_fn(r.get("path", ""), r.get("label", "spoof"))
                          for r in recs])
        t_opt, _, _, acer = optimal_threshold(scores, labels)
        # Per-PAI breakdown — APCER only applies to spoof classes
        classes = sorted(set(pai[labels == 1]))
        live_mask = (labels == 0)
        n_live = int(live_mask.sum())
        bpcer_overall = float(((scores >= t_opt) & live_mask).sum()) / max(1, n_live)
        for c in classes:
            m = (pai == c) & (labels == 1)
            n = int(m.sum())
            if n == 0:
                continue
            apcer = float((scores[m] < t_opt).sum()) / n
            rows.append({
                "dataset": name,
                "pai": c,
                "n": n,
                "apcer": apcer,
                "bpcer_dataset": bpcer_overall,
                "threshold": t_opt,
                "acer_dataset": acer,
            })
    return rows


def format_markdown(rows: List[dict]) -> str:
    """Render an APCER summary table as a markdown string for paper Section 6.2."""
    out = ["# APCER per Attack Class (ISO/IEC 30107-3 §7.1)",
           "",
           "_Decision threshold = per-dataset min-ACER optimum (same as `bootstrap_ci.md`)._",
           "_BPCER is reported at dataset level since it is independent of PAI class._",
           "",
           "| Dataset | PAI class | n | APCER @ τ* | BPCER @ τ* (dataset) | τ* |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| `{r['dataset']}` | `{r['pai']}` | {r['n']} | "
            f"{r['apcer']:.3f} | {r['bpcer_dataset']:.3f} | {r['threshold']:.4f} |"
        )
    out.append("")
    out.append("## Interpretation")
    out.append("")
    out.append("ISO/IEC 30107-3 §7.1.b asks for per-PAI APCER so weak spots are visible.")
    out.append("Worst-PAI per dataset is the reviewer-meaningful single number; we report "
               "all PAIs we can derive from source paths without relying on the")
    out.append("(not-carried) per-record `attack_class` field. Classes tagged `*_UNKNOWN` "
               "or `_MIXED_*` are honest placeholders where the jsonl source cannot be "
               "disambiguated (e.g. CelebA-Spoof flat layout) and should be refined by")
    out.append("re-scoring with `--attack-class` propagation once the preprocessors are "
               "extended.")
    out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    """CLI: walk one or more JSONL records, compute APCER per attack class, write a paper-grade markdown breakdown."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, type=Path)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/apcer_by_class.md"))
    ap.add_argument("--csv", type=Path, default=Path("reports/apcer_by_class.csv"))
    args = ap.parse_args(argv)

    if len(args.inputs) != len(args.names):
        print("[apcer] --inputs and --names length mismatch", file=sys.stderr)
        return 2

    rows = compute_table(args.inputs, args.names)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(format_markdown(rows))
    print(f"[apcer] wrote {args.out}")

    with open(args.csv, "w", encoding="utf-8") as fh:
        fh.write("dataset,pai,n,apcer,bpcer_dataset,threshold,acer_dataset\n")
        for r in rows:
            fh.write(f"{r['dataset']},{r['pai']},{r['n']},{r['apcer']:.6f},"
                     f"{r['bpcer_dataset']:.6f},{r['threshold']:.6f},"
                     f"{r['acer_dataset']:.6f}\n")
    print(f"[apcer] wrote {args.csv}")

    print()
    for r in rows:
        print(f"  {r['dataset']:<10s} {r['pai']:<32s} "
              f"n={r['n']:<5d} APCER={r['apcer']:.3f}  "
              f"thr={r['threshold']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
