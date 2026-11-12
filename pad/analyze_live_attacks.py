"""pad/analyze_live_attacks.py -- per-PAI APCER from self-captured
live-attack recordings.

Consumes the `.ws.jsonl` session logs produced by the DodoVisionAI
WebSocket replay recorder (`DODO_REPLAY_RECORD=1` flag at server
startup). Each clip's jsonl carries one record per processed frame
with the per-frame `p_attack`. Class labels come from the filename
prefix (see LIVE_ATTACK_PROTOCOL.md §3).

Usage
-----
    python -m pad.analyze_live_attacks \\
        --root live_attack/2026-04-NN_session_01 \\
        --threshold 0.5 \\
        --out summary.md

Filename convention (enforces ground truth):
    bona_fide_NN.ws.jsonl      class = BONA_FIDE
    print_matte_NN.ws.jsonl    class = PRINT_MATTE
    print_glossy_NN.ws.jsonl   class = PRINT_GLOSSY
    replay_mobile_NN.ws.jsonl  class = REPLAY_MOBILE
    replay_tablet_NN.ws.jsonl  class = REPLAY_TABLET
    mask_paper_NN.ws.jsonl     class = MASK_PAPER
    deepfake_stream_NN.ws.jsonl  class = DEEPFAKE_STREAM
    occlusion_control_NN.ws.jsonl  class = OCCLUSION_CONTROL (bona-fide)

Output
------
`summary.md` with:
  - Per-class table: n_clips, n_frames, mean_p_attack, median_p_attack,
    APCER (or BPCER for bona_fide / occlusion_control classes).
  - Aggregate BPCER (bona_fide + occlusion_control).
  - Aggregate APCER (all spoof classes combined).
  - Per-clip breakdown (for auditing outlier clips).

`summary.csv` with the same data for paper ingestion.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# Class-to-label map (1 = spoof / 0 = live for APCER/BPCER accounting)
CLASS_MAP: Dict[str, int] = {
    "bona_fide":         0,
    "occlusion_control": 0,
    "print_matte":       1,
    "print_glossy":      1,
    "replay_mobile":     1,
    "replay_tablet":     1,
    "mask_paper":        1,
    "deepfake_stream":   1,
}


def class_from_filename(fname: str) -> str:
    """e.g. 'print_matte_03.ws.jsonl' -> 'print_matte'."""
    stem = Path(fname).name
    # strip any trailing _NN before the .ws.jsonl suffix
    stem = re.sub(r"\.ws\.jsonl$", "", stem)
    stem = re.sub(r"_\d+$", "", stem)
    return stem


def load_clip(path: Path) -> Tuple[str, np.ndarray]:
    """Return (class_label, p_attack_per_frame_array)."""
    cls = class_from_filename(path.name)
    scores: List[float] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # WS replay schema varies; look for p_attack under a few common keys
            p = r.get("p_attack")
            if p is None:
                p = r.get("fused", {}).get("p_attack") if isinstance(r.get("fused"), dict) else None
            if p is None:
                p = r.get("pad", {}).get("p_attack") if isinstance(r.get("pad"), dict) else None
            if p is None:
                continue
            scores.append(float(p))
    return cls, np.asarray(scores, dtype=np.float64)


def per_class_stats(clips: Dict[str, List[np.ndarray]], threshold: float) -> List[Dict]:
    """For each class, compute n_clips, n_frames, mean, median, and
    APCER-or-BPCER depending on class label."""
    rows = []
    for cls in sorted(clips.keys()):
        cls_scores_per_clip = clips[cls]
        n_clips = len(cls_scores_per_clip)
        all_frames = np.concatenate([s for s in cls_scores_per_clip if len(s)])
        if len(all_frames) == 0:
            continue
        label = CLASS_MAP.get(cls, 1)  # assume spoof if unknown
        if label == 1:
            # APCER = fraction of ATTACK frames scored BELOW threshold
            error_rate = float((all_frames < threshold).sum()) / len(all_frames)
            metric = "APCER"
        else:
            # BPCER = fraction of LIVE frames scored AT OR ABOVE threshold
            error_rate = float((all_frames >= threshold).sum()) / len(all_frames)
            metric = "BPCER"
        rows.append({
            "class": cls,
            "label": "spoof" if label == 1 else "live",
            "n_clips": n_clips,
            "n_frames": int(len(all_frames)),
            "mean_p_attack": float(all_frames.mean()),
            "median_p_attack": float(np.median(all_frames)),
            "std_p_attack": float(all_frames.std()),
            "metric": metric,
            "error_rate": error_rate,
        })
    return rows


def aggregate_apcer_bpcer(clips: Dict[str, List[np.ndarray]], threshold: float) -> Tuple[float, float]:
    """Overall APCER + BPCER."""
    attack_frames = []
    live_frames = []
    for cls, cs in clips.items():
        label = CLASS_MAP.get(cls, 1)
        frames = np.concatenate([s for s in cs if len(s)])
        if label == 1:
            attack_frames.append(frames)
        else:
            live_frames.append(frames)
    attack_frames = np.concatenate(attack_frames) if attack_frames else np.array([])
    live_frames = np.concatenate(live_frames) if live_frames else np.array([])
    apcer = float((attack_frames < threshold).sum()) / max(1, len(attack_frames))
    bpcer = float((live_frames >= threshold).sum()) / max(1, len(live_frames))
    return apcer, bpcer


def per_clip_breakdown(clips: Dict[str, List[np.ndarray]], clip_paths: List[Path],
                       threshold: float) -> List[Dict]:
    """One row per recorded clip, for outlier auditing."""
    rows = []
    for path in sorted(clip_paths, key=lambda p: p.name):
        cls = class_from_filename(path.name)
        scores_list = [s for s in clips.get(cls, [])
                       if id(s) == id(_find_match(clips[cls], path))]
        # Simpler: re-read from file
        _, scores = load_clip(path)
        if len(scores) == 0:
            continue
        label = CLASS_MAP.get(cls, 1)
        bad_frames = int((scores < threshold).sum()) if label == 1 else int((scores >= threshold).sum())
        rows.append({
            "clip": path.name,
            "class": cls,
            "label": "spoof" if label == 1 else "live",
            "n_frames": len(scores),
            "mean_p": float(scores.mean()),
            "max_p": float(scores.max()),
            "min_p": float(scores.min()),
            "n_bad_frames": bad_frames,
            "frac_bad": bad_frames / len(scores),
        })
    return rows


def _find_match(scores_list, path):
    # placeholder to keep per_clip_breakdown tidy
    return scores_list[0] if scores_list else None


def emit_markdown(per_class: List[Dict], apcer_all: float, bpcer_all: float,
                  per_clip: List[Dict], threshold: float) -> str:
    """Render the live-attack-analysis report as a markdown string with
    three sections: per-PAI breakdown table, aggregate APCER/BPCER/ACER
    line, and per-clip detail table. The threshold value is included in
    the heading so the same JSONL evaluated at multiple operating points
    can be diffed without ambiguity."""
    out = [
        "# DodoVisionAI live-attack analysis",
        "",
        f"_Decision threshold = {threshold:.2f}_",
        f"_Total clips = {sum(r['n_clips'] for r in per_class)}_",
        f"_Total frames = {sum(r['n_frames'] for r in per_class)}_",
        "",
        "## Per-PAI / per-class breakdown",
        "",
        "| Class | label | clips | frames | mean p_attack | median | metric | rate |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in per_class:
        out.append(f"| `{r['class']}` | {r['label']} | {r['n_clips']} | {r['n_frames']} | "
                   f"{r['mean_p_attack']:.3f} | {r['median_p_attack']:.3f} | "
                   f"{r['metric']} | **{r['error_rate']:.3f}** |")
    out += [
        "",
        "## Aggregate",
        "",
        f"- **APCER** (all spoof classes pooled): **{apcer_all:.3f}**",
        f"- **BPCER** (all live classes pooled): **{bpcer_all:.3f}**",
        f"- **ACER** = (APCER + BPCER) / 2 = **{(apcer_all+bpcer_all)/2:.3f}**",
        "",
        "## Per-clip detail",
        "",
        "| Clip | class | frames | mean | max | bad frames | % bad |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in per_clip:
        out.append(f"| `{r['clip']}` | {r['class']} | {r['n_frames']} | "
                   f"{r['mean_p']:.3f} | {r['max_p']:.3f} | {r['n_bad_frames']} | "
                   f"{r['frac_bad']*100:.1f}% |")
    out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    """CLI: walk a directory of session-replay JSONL logs, compute APCER
    / BPCER / ACER per PAI class at the given threshold, and emit a
    markdown report. Used post-hoc on captured live-attack-test sessions
    to quantify per-PAI miss rates for the IJCB 2026 paper.
    Returns 0 on success, 1 if no clips found."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="dir containing *.ws.jsonl session logs")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="decision threshold for APCER / BPCER accounting (default 0.5)")
    ap.add_argument("--out", type=Path, default=None,
                    help="markdown output (default: <root>/summary.md)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="csv output (default: <root>/summary.csv)")
    args = ap.parse_args(argv)

    if not args.root.is_dir():
        print(f"[analyze-live] no such dir: {args.root}", file=sys.stderr)
        return 1

    clip_paths = sorted(args.root.glob("*.ws.jsonl"))
    if not clip_paths:
        print(f"[analyze-live] no *.ws.jsonl under {args.root}", file=sys.stderr)
        return 2

    clips: Dict[str, List[np.ndarray]] = {}
    for p in clip_paths:
        cls, scores = load_clip(p)
        if len(scores) == 0:
            print(f"[analyze-live] skip (no p_attack): {p.name}", file=sys.stderr)
            continue
        clips.setdefault(cls, []).append(scores)
        print(f"[analyze-live] {p.name}: {len(scores)} frames, "
              f"mean_p={scores.mean():.3f}")

    per_class = per_class_stats(clips, args.threshold)
    apcer_all, bpcer_all = aggregate_apcer_bpcer(clips, args.threshold)
    per_clip = per_clip_breakdown(clips, clip_paths, args.threshold)

    md = emit_markdown(per_class, apcer_all, bpcer_all, per_clip, args.threshold)
    out_md = args.out or (args.root / "summary.md")
    out_md.write_text(md, encoding="utf-8")
    print(f"[analyze-live] wrote {out_md}")

    out_csv = args.csv or (args.root / "summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "clip", "class", "label", "n_frames",
            "mean_p", "max_p", "min_p", "n_bad_frames", "frac_bad",
        ])
        w.writeheader()
        for r in per_clip:
            w.writerow(r)
    print(f"[analyze-live] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
