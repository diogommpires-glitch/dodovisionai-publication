"""preprocess_celeb_df.py
=========================

Turn a Celeb-DF v2 MP4 tree into `live/` + `spoof/` face-cropped JPGs.

Input layout (from Celeb-DF v2 download):

    <root>/
        Celeb-real/id*.mp4       <- live
        YouTube-real/*.mp4       <- live (may be absent in some packs)
        Celeb-synthesis/id*.mp4  <- spoof (deepfakes)
        List_of_testing_videos.txt (optional — not used here)

Output layout:

    <out>/
        live/<video_id>_<frame_idx>.jpg
        spoof/<video_id>_<frame_idx>.jpg
        _manifest.jsonl

Usage:
    python -m pad.preprocess_celeb_df \\
        --root E:\\pad_datasets\\Celeb-DF-v2 \\
        --out  E:\\pad_datasets\\Celeb-DF-v2 \\
        --frames-per-video 5

Typical runtime: ~20-30 min for default subsample (5 frames × ~6000 videos
= ~30k crops) on a 3060 laptop.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

from pad.preprocess_ffpp import _haar_detector, _crop_face, _video_frames


CATEGORIES = {
    "Celeb-real":      "live",
    "YouTube-real":    "live",
    "Celeb-synthesis": "spoof",
}


def _iter_videos(root: Path):
    for sub, label in CATEGORIES.items():
        d = root / sub
        if d.is_dir():
            for mp4 in sorted(d.glob("*.mp4")):
                yield label, sub, mp4


def preprocess(root: Path, out: Path, frames_per_video: int,
               limit: Optional[int]) -> tuple[int, int]:
    detector = _haar_detector()
    if detector is None:
        print("[celebdf-prep] FATAL: haar cascade unavailable")
        return 0, 0
    (out / "live").mkdir(parents=True, exist_ok=True)
    (out / "spoof").mkdir(parents=True, exist_ok=True)
    manifest = out / "_manifest.jsonl"
    n_videos = 0
    n_crops = 0
    t0 = time.time()
    with open(manifest, "a", encoding="utf-8") as mh:
        for label, category, path in _iter_videos(root):
            if limit is not None and n_videos >= limit:
                break
            # Prefix with category so Celeb-real/id0_0001.mp4 and
            # YouTube-real/id0_0001.mp4 don't collide.
            video_id = f"{category}_{path.stem}"
            sub = "live" if label == "live" else "spoof"
            existing = list((out / sub).glob(f"{video_id}_*.jpg"))
            if len(existing) >= frames_per_video:
                n_videos += 1
                continue
            wrote = 0
            for idx, frame in _video_frames(str(path), frames_per_video):
                crop = _crop_face(frame, detector)
                if crop is None:
                    continue
                out_path = out / sub / f"{video_id}_{idx:04d}.jpg"
                if cv2.imwrite(str(out_path), crop,
                               [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    wrote += 1
            if wrote:
                mh.write(json.dumps({"video": video_id, "label": label,
                                     "category": category,
                                     "n_crops": wrote}) + "\n")
                n_crops += wrote
            n_videos += 1
            if n_videos % 200 == 0:
                rate = n_videos / max(time.time() - t0, 0.001)
                print(f"[celebdf-prep] {n_videos} videos, {n_crops} crops, "
                      f"{rate:.1f} vid/s")
    dt = time.time() - t0
    print(f"[celebdf-prep] done in {dt/60:.1f} min: "
          f"{n_videos} videos, {n_crops} crops")
    return n_videos, n_crops


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="Celeb-DF-v2 root (contains Celeb-real/, "
                         "Celeb-synthesis/, etc.)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output dir (will create live/ + spoof/ inside)")
    ap.add_argument("--frames-per-video", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N videos (for smoke tests)")
    args = ap.parse_args(argv)
    # At least one of the expected subdirs must exist
    has_any = any((args.root / s).is_dir() for s in CATEGORIES)
    if not has_any:
        print(f"[celebdf-prep] no Celeb-DF category dirs under {args.root}")
        return 1
    preprocess(args.root, args.out, args.frames_per_video, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
