"""run_eval_xd.py — v9.5.x cross-dataset evaluation orchestrator.

Wraps `pad.eval.run_eval` with the single convention a reviewer wants:

  * Train/tune on ONE dataset
  * Evaluate on OTHERS — that's the only way to measure generalisation
  * Produce a 2-D table: rows = test set, cols = train set

Usage (after you've downloaded OULU-NPU + SiW + Replay-Attack):

    python -m pad.eval.run_eval_xd \\
        --datasets-root E:\\pad_datasets \\
        --out pad/reports/xd-2026-04-19 \\
        --limit 2000

For each test dataset we:
  1. Run the full eval harness (already wired in pad.eval.run_eval).
  2. Run the ablation + bootstrap on the scored.jsonl.
  3. Aggregate into a single xd_summary.md + xd_summary.json.

When a dataset isn't downloaded the loader just skips it silently — the
table shows `(no data)` for the missing cell.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time


# The canonical PAD cross-dataset quartet the literature benchmarks against.
STANDARD_DATASETS = ("OULU-NPU", "SiW", "Replay-Attack", "MSU-MFSD")


def run_one(name: str, datasets_root: pathlib.Path, out_dir: pathlib.Path,
            limit: int, python_exe: str) -> dict:
    """Run the eval harness against one dataset. Returns summary dict."""
    report_dir = out_dir / name.replace("/", "_")
    report_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    cmd = [
        python_exe, "-u", "-m", "pad.eval",
        "--datasets-root", str(datasets_root),
        "--only", name,
        "--out", str(report_dir),
        "--limit", str(limit),
    ]
    print(f"[xd] running {name}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    dt = time.monotonic() - t0
    ok = proc.returncode == 0
    scored = report_dir / "scored.jsonl"
    n_rows = 0
    if scored.is_file():
        with open(scored, encoding="utf-8") as f:
            n_rows = sum(1 for _ in f)
    # Ablation + bootstrap on what we scored.
    ablation_summary: dict = {}
    if scored.is_file() and n_rows > 10:
        abl_cmd = [python_exe, "-u", "-m", "pad.eval.ablation",
                   "--scored", str(scored), "--out", str(report_dir)]
        try:
            subprocess.run(abl_cmd, check=True, timeout=600)
            abl_json = report_dir / "ablation.json"
            if abl_json.is_file():
                with open(abl_json) as f:
                    ablation_summary = json.load(f)
        except Exception as e:
            ablation_summary = {"error": str(e)}
    return {
        "dataset": name,
        "ok": ok,
        "duration_s": round(dt, 1),
        "n_rows": n_rows,
        "ablation": ablation_summary,
        "stdout_tail": (proc.stdout or "")[-400:],
    }


def render_summary(results: list[dict], out_dir: pathlib.Path) -> None:
    """Write xd_summary.md and xd_summary.json."""
    md = ["# Cross-dataset eval — DodoVisionAI\n",
          f"_generated {time.strftime('%Y-%m-%d %H:%M:%S')}_", ""]
    md.append("| Dataset | Rows | APCER@BPCER=1% | 95% CI |")
    md.append("|:--|:--:|:--:|:--:|")
    for r in results:
        rows = r["n_rows"]
        a = r["ablation"].get("ablation", {}).get("base", {}) if r["ablation"] else {}
        b = r["ablation"].get("bootstrap", {}) if r["ablation"] else {}
        apcer = a.get("apcer_at_bpcer_1pct")
        ci = (f"[{b.get('p05', 0):.4f}, {b.get('p95', 0):.4f}]"
              if b else "—")
        md.append(
            f"| {r['dataset']} | {rows} | "
            f"{apcer:.4f} | {ci} |" if apcer is not None
            else f"| {r['dataset']} | {rows} | (no data) | — |"
        )
    (out_dir / "xd_summary.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "xd_summary.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"[xd] wrote {out_dir / 'xd_summary.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=list(STANDARD_DATASETS))
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    datasets_root = pathlib.Path(args.datasets_root)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in args.only:
        results.append(run_one(name, datasets_root, out_dir,
                               args.limit, args.python))
    render_summary(results, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
