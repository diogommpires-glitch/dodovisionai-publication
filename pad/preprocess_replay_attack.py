"""preprocess_replay_attack.py
==============================

Turn the Idiap Replay-Attack PNG-frame tree into `live/` + `spoof/`
face-cropped JPGs suitable for `pad.train_meta_fusion`.

Input layout (from the Zenodo archive, already frame-extracted):

    <root>/processed/
        train/real/<session>/frame_XXXX.png
        train/attack/<session>/frame_XXXX.png
        devel/real/...
        devel/attack/...
        test/real/...
        test/attack/...

Session names look like `real_client001_laptop_SD_scene01` or
`attack_client001_android_SD_ipad_video_scene01`.

Output layout:

    <out>/
        live/<session>_<frame_idx>.jpg
        spoof/<session>_<frame_idx>.jpg
        _manifest.jsonl

Usage:
    python -m pad.preprocess_replay_attack \\
        --root E:\\pad_datasets\\Replay-Attack \\
        --out  E:\\pad_datasets\\Replay-Attack \\
        --frames-per-session 5

Face detector options (v9.6.0):
  --detector yunet       OpenCV Zoo YuNet CNN (default, ~4-5× higher
                         recall than Haar on Replay-Attack frames).
                         Auto-downloads the 340 KB ONNX weight on first
                         use.
  --detector haar        OpenCV Haar cascade (legacy behaviour).
  --detector hybrid      Try YuNet first, fall back to Haar's centre-crop
                         behaviour — emits *something* per frame.

The 2026-04-21 session reported 1,395 crops from ~1,300 sessions using
the Haar-only path. YuNet's published recall @ FAR 1e-3 on WIDER-FACE
Easy is 0.887 vs Haar's ~0.55 on small-face inputs, so expected yield
on this pipeline is ~6,000-6,500 crops.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Reuse the already-tested FF++ Haar helpers.
from pad.preprocess_ffpp import _haar_detector, _crop_face


# ----------------------------------------------------------------------
# YuNet auto-download
# ----------------------------------------------------------------------
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
_YUNET_LOCAL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"


def _ensure_yunet_model() -> Optional[Path]:
    """Return a path to the YuNet ONNX, downloading on first use.
    Returns None on network failure (caller should fall back to Haar)."""
    if _YUNET_LOCAL.is_file() and _YUNET_LOCAL.stat().st_size > 100_000:
        return _YUNET_LOCAL
    _YUNET_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    print(f"[replay-prep] downloading YuNet face detector -> {_YUNET_LOCAL} ...")
    try:
        req = urllib.request.Request(
            _YUNET_URL,
            headers={"User-Agent": "DodoVisionAI/9.6 preprocess_replay_attack"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        sha = hashlib.sha256(data).hexdigest()
        if sha != _YUNET_SHA256:
            warnings.warn(f"[replay-prep] YuNet sha mismatch: got {sha[:16]}... "
                          f"expected {_YUNET_SHA256[:16]}...; downloading anyway")
        _YUNET_LOCAL.write_bytes(data)
        print(f"[replay-prep] YuNet downloaded ({len(data)/1024:.1f} KB)")
        return _YUNET_LOCAL
    except Exception as e:
        warnings.warn(f"[replay-prep] YuNet download failed ({e}); "
                      f"falling back to Haar.")
        return None


def _yunet_detector(input_size: tuple[int, int] = (320, 240),
                    score_threshold: float = 0.6,
                    nms_threshold: float = 0.3,
                    top_k: int = 5):
    """Return an OpenCV YuNet face detector, or None if the model is
    missing or cv2 is too old for FaceDetectorYN_create."""
    if not hasattr(cv2, "FaceDetectorYN_create"):
        warnings.warn("[replay-prep] cv2.FaceDetectorYN_create unavailable "
                      "(opencv-python < 4.5.4?); falling back to Haar.")
        return None
    model_path = _ensure_yunet_model()
    if model_path is None:
        return None
    try:
        det = cv2.FaceDetectorYN_create(
            str(model_path), "", input_size,
            score_threshold, nms_threshold, top_k,
        )
        return det
    except Exception as e:
        warnings.warn(f"[replay-prep] YuNet construction failed: {e}")
        return None


def _crop_face_yunet(bgr: np.ndarray, yunet, out_size: int = 256,
                     margin: float = 0.3) -> Optional[np.ndarray]:
    """Detect the highest-confidence face with YuNet and return a square
    crop. Returns None when nothing is detected (strict)."""
    if yunet is None:
        return None
    h, w = bgr.shape[:2]
    if h < 16 or w < 16:
        return None
    yunet.setInputSize((w, h))
    retval, dets = yunet.detect(bgr)
    if dets is None or len(dets) == 0:
        return None
    # YuNet 2023mar dets rows (15 cols): [x, y, w, h,
    #   right_eye_x, right_eye_y, left_eye_x, left_eye_y,
    #   nose_tip_x,  nose_tip_y,
    #   right_mouth_corner_x, right_mouth_corner_y,
    #   left_mouth_corner_x,  left_mouth_corner_y, score].
    # We only need bbox + score — best[-1] picks the score regardless
    # of landmark column count (future-proof for newer YuNet revs).
    best = max(dets, key=lambda d: float(d[-1]))
    bx, by, bw, bh = int(best[0]), int(best[1]), int(best[2]), int(best[3])
    if bw <= 0 or bh <= 0:
        return None
    side = int(max(bw, bh) * (1.0 + margin))
    cx = bx + bw // 2
    cy = by + bh // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0 or min(crop.shape[:2]) < 16:
        return None
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def _crop_with_strategy(bgr: np.ndarray, strategy: str,
                        haar_det, yunet_det) -> Optional[np.ndarray]:
    """Dispatch a single-frame crop to the chosen detector(s)."""
    if strategy == "yunet":
        return _crop_face_yunet(bgr, yunet_det)
    if strategy == "haar":
        return _crop_face(bgr, haar_det)
    # hybrid: YuNet (strict — None on no-face), then Haar's centre-crop
    # fallback so we still emit *something* for the rare frame where both
    # detectors fail. Matches legacy behaviour for backwards compat.
    crop = _crop_face_yunet(bgr, yunet_det)
    if crop is not None:
        return crop
    return _crop_face(bgr, haar_det)


def _iter_sessions(root: Path):
    """Yield (label, split_name, session_dir) for every session under
    processed/."""
    proc = root / "processed"
    if not proc.is_dir():
        return
    for split in sorted(p for p in proc.iterdir() if p.is_dir()):
        for cls_dir in sorted(p for p in split.iterdir() if p.is_dir()):
            cls = cls_dir.name.lower()
            label = "live" if cls == "real" else "spoof"
            for session in sorted(p for p in cls_dir.iterdir() if p.is_dir()):
                yield label, split.name, session


def preprocess(root: Path, out: Path, frames_per_session: int,
               limit: Optional[int], detector_strategy: str,
               overwrite: bool = False) -> tuple[int, int]:
    haar_det = _haar_detector()
    yunet_det = _yunet_detector() if detector_strategy != "haar" else None
    if detector_strategy == "haar" and haar_det is None:
        print("[replay-prep] FATAL: haar cascade unavailable and "
              "--detector=haar explicitly requested")
        return 0, 0
    if detector_strategy == "yunet" and yunet_det is None:
        print("[replay-prep] FATAL: YuNet detector unavailable "
              "(download/construction failed). Rerun with "
              "--detector=haar or --detector=hybrid to continue "
              "with the legacy cascade.")
        return 0, 0
    if detector_strategy == "hybrid" and yunet_det is None:
        print("[replay-prep] WARN: YuNet unavailable, hybrid mode "
              "falling back to Haar + centre-crop only")
    (out / "live").mkdir(parents=True, exist_ok=True)
    (out / "spoof").mkdir(parents=True, exist_ok=True)
    manifest = out / "_manifest.jsonl"
    n_sessions = 0
    n_crops = 0
    n_miss = 0
    t0 = time.time()
    with open(manifest, "a", encoding="utf-8") as mh:
        for label, split, session in _iter_sessions(root):
            if limit is not None and n_sessions >= limit:
                break
            session_id = f"{split}_{session.name}"
            sub = "live" if label == "live" else "spoof"
            existing = list((out / sub).glob(f"{session_id}_*.jpg"))
            if existing and not overwrite and len(existing) >= frames_per_session:
                n_sessions += 1
                continue
            if overwrite:
                # Wipe old crops for this session so the new detector's
                # output isn't silently overshadowed by stale files.
                for old in existing:
                    try:
                        old.unlink()
                    except OSError:
                        pass
            frames = sorted(session.glob("frame_*.png"))
            if not frames:
                continue
            step = max(1, len(frames) // max(1, frames_per_session))
            picks = frames[::step][:frames_per_session]
            wrote = 0
            for idx, frame_path in enumerate(picks):
                bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                crop = _crop_with_strategy(bgr, detector_strategy,
                                           haar_det, yunet_det)
                if crop is None:
                    n_miss += 1
                    continue
                out_path = out / sub / f"{session_id}_{idx:02d}.jpg"
                if cv2.imwrite(str(out_path), crop,
                               [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    wrote += 1
            if wrote:
                mh.write(json.dumps({"session": session_id, "label": label,
                                     "n_crops": wrote,
                                     "detector": detector_strategy}) + "\n")
                n_crops += wrote
            n_sessions += 1
            if n_sessions % 100 == 0:
                rate = n_sessions / max(time.time() - t0, 0.001)
                print(f"[replay-prep] {n_sessions} sessions, {n_crops} crops, "
                      f"{n_miss} detection misses, "
                      f"{rate:.1f} sessions/s")
    dt = time.time() - t0
    print(f"[replay-prep] done in {dt/60:.1f} min: "
          f"{n_sessions} sessions, {n_crops} crops, {n_miss} misses "
          f"(detector={detector_strategy})")
    return n_sessions, n_crops


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="Replay-Attack root (contains processed/)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output dir (will create live/ + spoof/ inside)")
    ap.add_argument("--frames-per-session", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N sessions (for smoke tests)")
    ap.add_argument("--detector",
                    choices=("yunet", "haar", "hybrid"),
                    default="yunet",
                    help="face detector (default: yunet — CNN-based, "
                         "~4-5× higher recall than haar on Replay-Attack "
                         "frames; auto-downloads the 340 KB weight)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing session crops instead of skipping "
                         "them (use when switching detectors, e.g. haar -> "
                         "yunet, so the stale crops don't mask the new run)")
    args = ap.parse_args(argv)
    if not (args.root / "processed").is_dir():
        print(f"[replay-prep] no processed/ dir under {args.root}")
        return 1
    preprocess(args.root, args.out, args.frames_per_session, args.limit,
               args.detector, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
