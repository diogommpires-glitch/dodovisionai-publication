"""pad/preprocess_video_clips.py -- extract 16-frame face-crop clips for
temporal deepfake training.

Input layouts (pick via --layout):

  --layout ffpp
    <root>/original_sequences/youtube/<compression>/videos/*.mp4        -> LIVE
    <root>/manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,
                                  NeuralTextures}/<compression>/videos/*.mp4 -> SPOOF

  --layout celebdf
    <root>/Celeb-real/*.mp4            -> LIVE
    <root>/YouTube-real/*.mp4          -> LIVE (may be absent)
    <root>/Celeb-synthesis/*.mp4       -> SPOOF

Output layout:
    <out>/live/<video_id>_<clip_idx>.npy      shape [16, 3, 112, 112] uint8
    <out>/spoof/<video_id>_<clip_idx>.npy
    <out>/_manifest.jsonl

Each clip is a non-overlapping contiguous 16-frame chunk, center-cropped to
a square and resized to 112x112. Uses center-crop rather than face detection
because FF++ and Celeb-DF videos are already face-centered, and Haar per
frame is >10x slower (~50ms/frame vs ~2ms/frame).

Usage:
    python -m pad.preprocess_video_clips \\
        --root E:\\pad_datasets\\FaceForensics \\
        --layout ffpp --compression c40 \\
        --out E:\\pad_datasets\\video_clips_ffpp \\
        --clips-per-video 10 --limit 3000
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import cv2
import numpy as np


CLIP_LEN = 16
CROP_SIZE = 112
FFPP_MANIP = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
CELEBDF_CATEGORIES = {
    "Celeb-real":      "live",
    "YouTube-real":    "live",
    "Celeb-synthesis": "spoof",
}


def _iter_videos_ffpp(root: Path, compression: str) -> Iterable[Tuple[str, str, Path]]:
    """Yield (label, source, video_path) for FF++ tree."""
    orig = root / "original_sequences" / "youtube" / compression / "videos"
    if orig.is_dir():
        for mp4 in sorted(orig.glob("*.mp4")):
            yield "live", "ORIGINAL", mp4
    manip_root = root / "manipulated_sequences"
    if manip_root.is_dir():
        for sub in FFPP_MANIP:
            d = manip_root / sub / compression / "videos"
            if d.is_dir():
                for mp4 in sorted(d.glob("*.mp4")):
                    yield "spoof", sub, mp4


def _iter_videos_celebdf(root: Path) -> Iterable[Tuple[str, str, Path]]:
    for sub, label in CELEBDF_CATEGORIES.items():
        d = root / sub
        if d.is_dir():
            for mp4 in sorted(d.glob("*.mp4")):
                yield label, sub, mp4


def _center_crop_resize(frame: np.ndarray, out_size: int = CROP_SIZE) -> np.ndarray:
    h, w = frame.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2; x0 = (w - s) // 2
    square = frame[y0:y0 + s, x0:x0 + s]
    return cv2.resize(square, (out_size, out_size), interpolation=cv2.INTER_AREA)


def _extract_clips(video_path: Path, n_clips: int) -> List[np.ndarray]:
    """Return up to n_clips non-overlapping 16-frame clips, each
    [16, 3, 112, 112] uint8 (T, C, H, W)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < CLIP_LEN:
        cap.release()
        return []

    # Pick evenly-spaced start frames so clips are spread across video
    max_start = total - CLIP_LEN
    if n_clips >= (max_start // CLIP_LEN + 1):
        starts = list(range(0, max_start + 1, CLIP_LEN))[:n_clips]
    else:
        step = max(1, max_start // n_clips)
        starts = list(range(0, max_start + 1, step))[:n_clips]

    clips = []
    for s in starts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        frames = []
        for _ in range(CLIP_LEN):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            cc = _center_crop_resize(frame)
            rgb = cv2.cvtColor(cc, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        if len(frames) == CLIP_LEN:
            arr = np.stack(frames, axis=0)                     # [T, H, W, C]
            arr = np.transpose(arr, (0, 3, 1, 2)).astype(np.uint8)  # [T, C, H, W]
            clips.append(arr)
    cap.release()
    return clips


def preprocess(videos, out: Path, clips_per_video: int,
               limit: Optional[int]) -> Tuple[int, int]:
    (out / "live").mkdir(parents=True, exist_ok=True)
    (out / "spoof").mkdir(parents=True, exist_ok=True)
    manifest = out / "_manifest.jsonl"
    n_videos = 0
    n_clips_total = 0
    t0 = time.time()
    with open(manifest, "a", encoding="utf-8") as mh:
        for label, source, path in videos:
            if limit is not None and n_videos >= limit:
                break
            vid_id = f"{source}_{path.stem}"
            sub = "live" if label == "live" else "spoof"
            existing = list((out / sub).glob(f"{vid_id}_*.npy"))
            if len(existing) >= clips_per_video:
                n_videos += 1
                continue
            try:
                clips = _extract_clips(path, clips_per_video)
            except Exception as e:
                print(f"[vid-prep] skip {path.name}: {e}", file=sys.stderr)
                continue
            wrote = 0
            for idx, clip in enumerate(clips):
                np.save(out / sub / f"{vid_id}_{idx:03d}.npy", clip)
                wrote += 1
            if wrote:
                mh.write(json.dumps({"video": vid_id, "label": label,
                                     "source": source, "n_clips": wrote}) + "\n")
                n_clips_total += wrote
            n_videos += 1
            if n_videos % 100 == 0:
                rate = n_videos / max(time.time() - t0, 0.001)
                print(f"[vid-prep] {n_videos} videos, {n_clips_total} clips, "
                      f"{rate:.1f} vid/s", flush=True)
    dt = time.time() - t0
    print(f"[vid-prep] done in {dt/60:.1f} min: {n_videos} videos, "
          f"{n_clips_total} clips", flush=True)
    return n_videos, n_clips_total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--layout", choices=["ffpp", "celebdf"], required=True)
    ap.add_argument("--compression", default="c40",
                    help="FF++ compression (c23 HQ / c40 LQ); ignored for celebdf")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clips-per-video", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if args.layout == "ffpp":
        if not args.root.is_dir():
            print(f"[vid-prep] FF++ root missing: {args.root}", file=sys.stderr)
            return 1
        videos = list(_iter_videos_ffpp(args.root, args.compression))
    else:
        videos = list(_iter_videos_celebdf(args.root))
    if not videos:
        print(f"[vid-prep] no videos found under {args.root}", file=sys.stderr)
        return 1
    random.Random(args.seed).shuffle(videos)
    print(f"[vid-prep] {len(videos)} videos discovered ({args.layout})")
    preprocess(videos, args.out, args.clips_per_video, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
