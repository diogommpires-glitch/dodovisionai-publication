"""Write a markdown summary of the calibration sprint state.

Collects:
  - calibrators.pkl per-branch metadata (choice, ECE, fit range)
  - meta_fuser.pkl metrics (auc, log_loss, n_train)
  - jsonl record counts per dataset
  - cross-dataset eval summary links

Writes to reports/calib_sprint_summary.md.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "reports"


def _safe_count(path: Path) -> int:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except Exception:
        return 0


def main() -> int:
    lines = [
        "# Calibration Sprint Summary",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
    ]

    # Calibrators
    lines += ["## Calibrator outcomes", ""]
    try:
        from pad.calibrate import load_calibrators, _reset_calibrators_cache
        _reset_calibrators_cache()
        cs = load_calibrators()
    except Exception as e:
        cs = None
        lines.append(f"_calibrators load failed: {e}_")
    if cs is not None:
        lines.append("| Branch | Choice | ECE before | ECE after | Range | Fit source | Training SHA |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, meta in sorted(cs.metadata.items()):
            choice = meta.get("choice", "?")
            before = meta.get("ece_before", float("nan"))
            after = meta.get("ece_after", float("nan"))
            rng = meta.get("fit_range", "-")
            rng_str = f"{rng:.3f}" if isinstance(rng, float) else str(rng)
            src = meta.get("fit_source", "—")
            sha = meta.get("training_jsonls_sha256", "") or "—"
            lines.append(f"| {name} | {choice} | {before:.4f} | {after:.4f} | "
                         f"{rng_str} | {src} | `{sha}` |")
    else:
        lines.append("_no calibrators.pkl on disk_")
    lines.append("")

    # Temporal deepfake head (if ONNX is on disk)
    temporal_onnx = REPO / "pad" / "models" / "deepfake_temporal.onnx"
    temporal_jsonl = REPORTS / "deepfake_temporal_records_celebdf.jsonl"
    lines += ["## Temporal deepfake head", ""]
    if temporal_onnx.is_file():
        sz = temporal_onnx.stat().st_size
        lines.append(f"- ONNX on disk: `{temporal_onnx.name}` "
                     f"({sz/1024/1024:.1f} MB, mtime "
                     f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(temporal_onnx.stat().st_mtime))})")
        if temporal_jsonl.is_file():
            n = _safe_count(temporal_jsonl)
            lines.append(f"- Scored records on Celeb-DF: {n:,} "
                         f"(`{temporal_jsonl.name}`)")
        else:
            lines.append("- No Celeb-DF temporal score jsonl yet "
                         "(run `python -m pad.score_videos_temporal`).")
    else:
        lines.append("_no pad/models/deepfake_temporal.onnx on disk_")
    lines.append("")

    # Meta-fuser
    lines += ["## Meta-fuser metrics", ""]
    try:
        from pad.meta_fuse import load_meta_fuser
        mf = load_meta_fuser()
    except Exception as e:
        mf = None
        lines.append(f"_meta-fuser load failed: {e}_")
    if mf is not None and getattr(mf, "metrics", None):
        for k, v in mf.metrics.items():
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("_no meta_fuser.pkl or metrics missing_")
    lines.append("")

    # Jsonl record counts
    lines += ["## Scored record counts", ""]
    patterns = ("meta_train_records*.jsonl", "audio_*.jsonl")
    found = False
    for pat in patterns:
        for p in sorted(REPORTS.glob(pat)):
            lines.append(f"- `{p.name}`: {_safe_count(p):,} records")
            found = True
    if not found:
        lines.append("_no jsonls on disk_")
    lines.append("")

    # Cross-dataset eval reports
    lines += ["## Cross-dataset eval reports", ""]
    xd_root = REPORTS / "cross_dataset"
    if xd_root.is_dir():
        for sub in sorted(p for p in xd_root.iterdir() if p.is_dir()):
            summary = sub / "summary.md"
            mark = "✓" if summary.is_file() else "…"
            lines.append(f"- {mark} `cross_dataset/{sub.name}/` "
                         f"{'(summary.md present)' if summary.is_file() else '(no summary yet)'}")
    else:
        lines.append("_no cross-dataset eval directory_")
    lines.append("")

    out = REPORTS / "calib_sprint_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[summary] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
