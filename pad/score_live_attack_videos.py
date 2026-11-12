"""pad/score_live_attack_videos.py -- offline per-frame scoring of
recorded live-attack .mp4 clips.

Bridges the LIVE_ATTACK_PROTOCOL.md "Option A" flow (record raw video
now, score later) to the existing pad/analyze_live_attacks.py pipeline,
which expects .ws.jsonl per clip.

What it does
------------
For each .mp4 in --root:
  1. Samples frames every --stride (default 5 -> ~6 fps effective from
     30 fps input; tweak for longer/denser scoring).
  2. Detects the highest-confidence face via YuNet (reuses the helper
     in pad.preprocess_replay_attack).
  3. Crops to 256x256 with a 30% margin (matches the server crop).
  4. Scores moire + patch_depth branches on each crop.
  5. Fuses via pad.fusion.fuse() (same log-odds fusion as the server).
  6. Emits `<clip_basename>.ws.jsonl`, one JSON record per sampled
     frame, with schema:
         {
           "frame_idx": int, "ts_sec": float,
           "p_attack": float,  # fused
           "verdict": "LIVE" | "SPOOF" | "UNCERTAIN",
           "branches": [
              {"name": "moire", "p_attack": float, "reliability": float},
              {"name": "patch_depth", "p_attack": float, "reliability": float},
           ],
           "face_detected": bool
         }

This format matches what pad/analyze_live_attacks.py::load_clip reads
(it looks for top-level `p_attack`).

Scope (v1)
----------
- Uses moire + patch_depth (two proven image branches on commodity HW).
- SKIPS deepfake-temporal (needs a 16-frame rolling buffer + PersonState
  plumbing; planned for a v2 follow-up).
- SKIPS audio (AASIST needs wav extraction from mp4).
- SKIPS deepfake-spatial (Face X-Ray wants its own preprocessing path).

The paper's live-attack section (§5.3) uses the fused p_attack here,
which is honest: the two image branches carry the image-domain
decisions; temporal/audio add orthogonal evidence not exercised
against your print/replay/mask attacks anyway.

Usage
-----
    python -m pad.score_live_attack_videos \\
        --root "E:/pad_datasets/live_attack/2026-04-NN_session_01" \\
        --stride 5

    # then analyse:
    python -m pad.analyze_live_attacks --root <same-dir>

    # for a quick 1-clip smoke test:
    python -m pad.score_live_attack_videos --root <dir> --max-clips 1

Safety
------
This script is safe to author at any time but DO NOT RUN while
queue_mega.cmd is training -- it will load ONNX sessions onto the
same GPU. Check for queue/mega/queue_done.marker before running, or
pass --allow-during-training to override.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from pad.branches import moire_branch, patch_depth_branch
from pad.fusion import fuse
from pad.preprocess_replay_attack import (
    _ensure_yunet_model,
    _yunet_detector,
    _crop_face_yunet,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
QUEUE_MARKER = REPO_ROOT / "queue" / "mega" / "queue_done.marker"

DEFAULT_STRIDE = 5
DEFAULT_CROP_SIZE = 256
DEFAULT_MARGIN = 0.30
MP4_GLOB = "*.mp4"


def _safe_crop(bgr: np.ndarray, yunet, out_size: int, margin: float) -> Optional[np.ndarray]:
    """Detect a face and return a square crop, or None."""
    if bgr is None or bgr.size == 0:
        return None
    try:
        crop = _crop_face_yunet(bgr, yunet, out_size=out_size, margin=margin)
    except Exception:
        crop = None
    return crop


def score_frame(crop_bgr: np.ndarray) -> dict:
    """Score one face crop with the two image branches + fuse."""
    b_moire = moire_branch.score(crop_bgr)
    b_pd = patch_depth_branch.score(crop_bgr)
    fused = fuse([b_moire, b_pd])
    return {
        "p_attack": float(fused.p_attack),
        "verdict": fused.verdict,
        "log_odds": float(fused.log_odds),
        "branches": [
            {
                "name": b_moire.name,
                "p_attack": float(b_moire.p_attack),
                "reliability": float(b_moire.reliability),
            },
            {
                "name": b_pd.name,
                "p_attack": float(b_pd.p_attack),
                "reliability": float(b_pd.reliability),
            },
        ],
    }


def score_one_video(video_path: Path, out_path: Path, yunet, stride: int,
                    crop_size: int, margin: float, verbose: bool = True) -> dict:
    """Process one .mp4 -> .ws.jsonl. Returns a summary dict."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"ok": False, "reason": f"cv2 failed to open {video_path.name}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    n_scored = 0
    n_face_missed = 0
    p_values: List[float] = []
    t0 = time.monotonic()

    with open(out_path, "w", encoding="utf-8") as fh:
        frame_idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            ts_sec = frame_idx / fps
            crop = _safe_crop(bgr, yunet, crop_size, margin)

            if crop is None:
                rec = {
                    "frame_idx": frame_idx,
                    "ts_sec": round(ts_sec, 3),
                    "p_attack": 0.5,
                    "verdict": "UNCERTAIN",
                    "branches": [],
                    "face_detected": False,
                }
                n_face_missed += 1
            else:
                scored = score_frame(crop)
                rec = {
                    "frame_idx": frame_idx,
                    "ts_sec": round(ts_sec, 3),
                    "p_attack": round(scored["p_attack"], 4),
                    "verdict": scored["verdict"],
                    "log_odds": round(scored["log_odds"], 4),
                    "branches": scored["branches"],
                    "face_detected": True,
                }
                p_values.append(scored["p_attack"])
                n_scored += 1

            fh.write(json.dumps(rec) + "\n")
            frame_idx += 1

    cap.release()
    wall = time.monotonic() - t0
    summary = {
        "ok": True,
        "video": video_path.name,
        "total_frames": total,
        "sampled_frames": n_scored + n_face_missed,
        "scored": n_scored,
        "face_missed": n_face_missed,
        "mean_p_attack": float(np.mean(p_values)) if p_values else None,
        "wall_sec": round(wall, 1),
    }
    if verbose:
        mp = f"{summary['mean_p_attack']:.3f}" if summary["mean_p_attack"] is not None else "n/a"
        print(f"  [{video_path.name}] scored={n_scored}  miss={n_face_missed}  "
              f"mean_p={mp}  ({wall:.1f}s)")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", type=Path, required=True,
                    help="dir containing live-attack *.mp4 clips")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                    help=f"sample every N frames (default {DEFAULT_STRIDE})")
    ap.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE,
                    help=f"face crop size (default {DEFAULT_CROP_SIZE})")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                    help=f"crop margin fraction (default {DEFAULT_MARGIN})")
    ap.add_argument("--max-clips", type=int, default=0,
                    help="process at most N clips (0 = all)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip clips whose .ws.jsonl already exists")
    ap.add_argument("--allow-during-training", action="store_true",
                    help="bypass the queue_done.marker guard")
    args = ap.parse_args(argv)

    if not args.root.is_dir():
        print(f"[score-live] not a directory: {args.root}", file=sys.stderr)
        return 1

    if not QUEUE_MARKER.exists() and not args.allow_during_training:
        print("[score-live] refusing to run: queue_mega has not finished "
              f"(marker missing: {QUEUE_MARKER})", file=sys.stderr)
        print("[score-live] pass --allow-during-training to override",
              file=sys.stderr)
        return 2

    mp4s = sorted(args.root.glob(MP4_GLOB))
    if not mp4s:
        print(f"[score-live] no {MP4_GLOB} under {args.root}", file=sys.stderr)
        return 3
    if args.max_clips > 0:
        mp4s = mp4s[: args.max_clips]

    _ensure_yunet_model()
    yunet = _yunet_detector()
    if yunet is None:
        print("[score-live] YuNet unavailable; abort", file=sys.stderr)
        return 4

    print(f"[score-live] {len(mp4s)} clip(s) under {args.root}  "
          f"stride={args.stride}  crop={args.crop_size}")

    summaries = []
    for mp4 in mp4s:
        out_jsonl = mp4.with_suffix(".ws.jsonl")
        if args.skip_existing and out_jsonl.exists():
            print(f"  [{mp4.name}] skip (exists)")
            continue
        summary = score_one_video(
            mp4, out_jsonl, yunet,
            stride=args.stride,
            crop_size=args.crop_size,
            margin=args.margin,
        )
        summaries.append(summary)

    ok = sum(1 for s in summaries if s.get("ok"))
    miss_total = sum(s.get("face_missed", 0) for s in summaries if s.get("ok"))
    scored_total = sum(s.get("scored", 0) for s in summaries if s.get("ok"))
    print(f"[score-live] done.  ok={ok}/{len(summaries)}  "
          f"frames_scored={scored_total}  face_missed={miss_total}")
    print(f"[score-live] next: python -m pad.analyze_live_attacks --root "
          f"\"{args.root}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
