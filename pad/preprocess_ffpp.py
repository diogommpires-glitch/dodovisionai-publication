"""preprocess_ffpp.py
====================

Turn a FaceForensics++ download tree into a live/spoof face-crop folder
suitable for `pad.train_patch_depth` (and therefore for re-exporting
as `pad/models/face_xray.onnx`).

Input layout (produced by the official FF++ download.py):

    <root>/
        original_sequences/youtube/<compression>/videos/*.mp4
        manipulated_sequences/Deepfakes/<compression>/videos/*.mp4
        manipulated_sequences/Face2Face/<compression>/videos/*.mp4
        manipulated_sequences/FaceSwap/<compression>/videos/*.mp4
        manipulated_sequences/NeuralTextures/<compression>/videos/*.mp4
        manipulated_sequences/FaceShifter/<compression>/videos/*.mp4      (optional)

Output layout (what the existing trainer expects):

    <out>/
        live/<video_id>_<frame_idx>.jpg
        spoof/<video_id>_<frame_idx>.jpg

Resilience:
 - Resume-safe: if an output file already exists, the video is skipped
   without re-decoding.
 - Face detection uses OpenCV Haar cascade first (cheap, built-in);
   if Haar returns no face, we take the frame's central square crop
   as a fallback (FF++ videos are aligned well enough that this
   rarely loses the face entirely).
 - Writes a manifest `<out>/_manifest.jsonl` (one line per video)
   so downstream steps can audit what was processed.
 - Clear progress output every N videos.

Usage:
    python -m pad.preprocess_ffpp \\
        --ffpp E:\\pad_datasets\\FaceForensics \\
        --out  E:\\pad_datasets\\face_xray \\
        --frames-per-video 30 \\
        --compression c40

    # smoke test (writes synthetic videos, then runs the pipeline):
    python -m pad.preprocess_ffpp --smoke --out /tmp/ffpp_smoke

Exit codes:
    0 success, 1 missing input tree, 2 face-detector unavailable,
    3 zero videos found (likely wrong compression arg).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


MANIP_DIRS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures",
              "FaceShifter")


def _haar_detector():
    p = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if not os.path.isfile(p):
        return None
    return cv2.CascadeClassifier(p)


def _crop_face(bgr: np.ndarray, detector, out_size: int = 256) -> Optional[np.ndarray]:
    """Detect the largest face; return a squared, padded crop at out_size×out_size.
    Falls back to a centre square crop if detection fails."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = []
    if detector is not None:
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                          minSize=(80, 80))
    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        cx, cy = x + fw // 2, y + fh // 2
        side = int(max(fw, fh) * 1.4)
    else:
        cx, cy = w // 2, h // 2
        side = min(w, h)
    half = side // 2
    x0 = max(0, cx - half); y0 = max(0, cy - half)
    x1 = min(w, cx + half); y1 = min(h, cy + half)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def _video_frames(path: str, n: int) -> Iterable[Tuple[int, np.ndarray]]:
    """Yield up to n evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release(); return
    step = max(1, total // max(1, n))
    picks = list(range(0, total, step))[:n]
    for idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            yield idx, frame
    cap.release()


def _iter_videos(ffpp_root: Path, compression: str):
    """Yield (label, manip_class, video_path) for every FF++ MP4 found."""
    orig = ffpp_root / "original_sequences" / "youtube" / compression / "videos"
    if orig.is_dir():
        for mp4 in sorted(orig.glob("*.mp4")):
            yield "live", "ORIGINAL", mp4
    manip_root = ffpp_root / "manipulated_sequences"
    if manip_root.is_dir():
        for sub in MANIP_DIRS:
            d = manip_root / sub / compression / "videos"
            if d.is_dir():
                for mp4 in sorted(d.glob("*.mp4")):
                    yield "spoof", sub.upper(), mp4


def preprocess(ffpp_root: Path, out_root: Path, frames_per_video: int,
               compression: str, limit: Optional[int] = None) -> Tuple[int, int]:
    """Run the whole tree. Returns (n_videos_processed, n_crops_written)."""
    detector = _haar_detector()
    if detector is None:
        print("[ffpp-prep] FATAL: haar cascade not found in opencv")
        return 0, 0
    (out_root / "live").mkdir(parents=True, exist_ok=True)
    (out_root / "spoof").mkdir(parents=True, exist_ok=True)
    manifest = out_root / "_manifest.jsonl"
    n_videos = 0
    n_crops = 0
    t0 = time.time()
    with open(manifest, "a", encoding="utf-8") as mh:
        for label, manip, path in _iter_videos(ffpp_root, compression):
            if limit is not None and n_videos >= limit:
                break
            video_id = path.stem
            sub = "live" if label == "live" else "spoof"
            # Resume: count existing crops for this video and skip if full.
            existing = list((out_root / sub).glob(f"{video_id}_*.jpg"))
            if len(existing) >= frames_per_video:
                n_videos += 1
                continue
            wrote = 0
            for frame_idx, frame in _video_frames(str(path), frames_per_video):
                crop = _crop_face(frame, detector, out_size=256)
                if crop is None:
                    continue
                jpg_path = out_root / sub / f"{video_id}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(jpg_path), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                wrote += 1
            n_videos += 1
            n_crops += wrote
            mh.write(json.dumps({
                "video_id": video_id, "label": label,
                "manipulation": manip, "src": str(path),
                "frames_written": wrote,
            }) + "\n")
            mh.flush()
            if n_videos % 25 == 0:
                dt = time.time() - t0
                rate = n_videos / max(dt, 1e-6)
                print(f"[ffpp-prep] {n_videos} videos ({n_crops} crops) "
                      f"in {dt/60:.1f} min — {rate:.2f} vid/s", flush=True)
    dt = time.time() - t0
    print(f"[ffpp-prep] DONE: {n_videos} videos, {n_crops} crops in "
          f"{dt/60:.1f} min", flush=True)
    return n_videos, n_crops


def _smoke(out_root: Path) -> int:
    """Generate two tiny synthetic videos and run the pipeline on them."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ffpp = Path(td) / "FaceForensics"
        for rel in [
            "original_sequences/youtube/c40/videos",
            "manipulated_sequences/Deepfakes/c40/videos",
        ]:
            (ffpp / rel).mkdir(parents=True)
        for label_dir, color in [
            ("original_sequences/youtube/c40/videos", (180, 180, 200)),
            ("manipulated_sequences/Deepfakes/c40/videos", (160, 140, 180)),
        ]:
            for j in range(2):
                vid = ffpp / label_dir / f"fake{j:03d}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vw = cv2.VideoWriter(str(vid), fourcc, 15.0, (320, 240))
                for k in range(10):
                    frame = np.full((240, 320, 3), color, dtype=np.uint8)
                    cv2.ellipse(frame, (160, 120), (70, 90), 0, 0, 360,
                                (210, 190, 180), -1)
                    cv2.circle(frame, (140, 105), 7, (30, 30, 30), -1)
                    cv2.circle(frame, (180, 105), 7, (30, 30, 30), -1)
                    vw.write(frame)
                vw.release()
        n_vid, n_crops = preprocess(ffpp, out_root, frames_per_video=4,
                                    compression="c40", limit=None)
        if n_vid == 0 or n_crops == 0:
            print("[smoke] FAIL — no crops produced")
            return 1
        print(f"[smoke] OK — {n_vid} videos, {n_crops} crops")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ffpp", help="FaceForensics++ download root")
    ap.add_argument("--out", required=True,
                    help="output root for live/ + spoof/ crops")
    ap.add_argument("--frames-per-video", type=int, default=30)
    ap.add_argument("--compression", default="c40",
                    choices=["c0", "c23", "c40"],
                    help="compression tier (must match what you downloaded)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap videos processed (smoke only)")
    ap.add_argument("--smoke", action="store_true",
                    help="run a synthetic-video smoke test")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        return _smoke(out_root)

    if not args.ffpp:
        print("[ffpp-prep] --ffpp required unless --smoke")
        return 1
    ffpp_root = Path(args.ffpp)
    if not ffpp_root.is_dir():
        print(f"[ffpp-prep] {ffpp_root} not found")
        return 1

    n_vid, n_crops = preprocess(ffpp_root, out_root, args.frames_per_video,
                                args.compression, args.limit)
    if n_vid == 0:
        print(f"[ffpp-prep] no videos found under {ffpp_root} with compression "
              f"{args.compression}. Check the download or --compression arg.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
