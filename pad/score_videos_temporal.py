"""pad/score_videos_temporal.py -- Score videos through deepfake_temporal.onnx.

For each video in a dataset, extracts one 16-frame clip (center window),
runs it through the MC3-18 temporal deepfake detector, and writes a
records jsonl compatible with pad.calibrate_per_branch.

Usage:
    python -m pad.score_videos_temporal \\
        --root E:\\pad_datasets\\Celeb-DF-v2 \\
        --layout celebdf \\
        --limit 2000 \\
        --out reports\\deepfake_temporal_records_celebdf.jsonl
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np


ONNX = Path(__file__).resolve().parent / "models" / "deepfake_temporal.onnx"
CLIP_LEN = 16
IMG_SIZE = 112

FFPP_MANIP = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
CELEBDF_CATEGORIES = {
    "Celeb-real":      "live",
    "YouTube-real":    "live",
    "Celeb-synthesis": "spoof",
}

# Kinetics-400 mean/std (MC3-18 pretrained)
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32).reshape(3, 1, 1)
STD  = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32).reshape(3, 1, 1)


def _iter_videos_ffpp(root: Path, compression: str) -> Iterable[Tuple[str, str, Path]]:
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


def _extract_center_clip(video_path: Path) -> Optional[np.ndarray]:
    """Return a center 16-frame clip as [C=3, T=16, H=112, W=112] float32."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < CLIP_LEN:
        cap.release()
        return None
    start = max(0, (total - CLIP_LEN) // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames_chw = []
    for _ in range(CLIP_LEN):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        h, w = frame.shape[:2]
        s = min(h, w); y0 = (h - s) // 2; x0 = (w - s) // 2
        sq = frame[y0:y0 + s, x0:x0 + s]
        rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
        chw = (chw - MEAN) / STD
        frames_chw.append(chw)
    cap.release()
    if len(frames_chw) < CLIP_LEN:
        return None
    return np.stack(frames_chw, axis=1)  # [C, T, H, W]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--layout", choices=["ffpp", "celebdf"], required=True)
    ap.add_argument("--compression", default="c40")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if not ONNX.is_file():
        print(f"[vid-temp] missing {ONNX}", file=sys.stderr); return 2

    try:
        import onnxruntime as ort
    except ImportError:
        print("[vid-temp] onnxruntime required", file=sys.stderr); return 2

    if args.layout == "ffpp":
        videos = list(_iter_videos_ffpp(args.root, args.compression))
    else:
        videos = list(_iter_videos_celebdf(args.root))
    if not videos:
        print(f"[vid-temp] no videos under {args.root}", file=sys.stderr); return 1

    rng = random.Random(args.seed); rng.shuffle(videos)
    # Balanced 50/50 sample
    live = [v for v in videos if v[0] == "live"]
    spoof = [v for v in videos if v[0] == "spoof"]
    half = args.limit // 2
    picks = live[:half] + spoof[:args.limit - half]
    rng.shuffle(picks)
    print(f"[vid-temp] scoring {len(picks)} videos "
          f"(live {sum(1 for v in picks if v[0]=='live')}, "
          f"spoof {sum(1 for v in picks if v[0]=='spoof')})", flush=True)

    # Try CUDA first, fall back to CPU
    providers = ["CPUExecutionProvider"]
    try:
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "DmlExecutionProvider" in avail:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    sess = ort.InferenceSession(str(ONNX), providers=providers)
    inp = sess.get_inputs()[0].name

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_ok = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, (label, source, path) in enumerate(picks):
            clip = _extract_center_clip(path)
            if clip is None:
                continue
            x = clip[None, ...].astype(np.float32)    # [1, C, T, H, W]
            try:
                out = sess.run(None, {inp: x})
            except Exception as e:
                print(f"[vid-temp] onnx failed on {path.name}: {e}",
                      file=sys.stderr); continue
            logits = np.asarray(out[0]).reshape(-1)
            ex = np.exp(logits - logits.max()); probs = ex / ex.sum()
            p_attack = float(probs[1]) if probs.size >= 2 else 0.5
            # Apply calibration if available
            try:
                from pad.calibrate import apply_calibration
                p_attack = apply_calibration("deepfake_temporal", p_attack)
            except Exception:
                pass
            rec = {
                "path": str(path),
                "label": label,
                "attack_class": None if label == "live" else source,
                "branches": [{
                    "name": "deepfake_temporal",
                    "p_attack": p_attack,
                    "reliability": 1.0,
                    "dominant_class": None,
                }],
            }
            fh.write(json.dumps(rec) + "\n")
            n_ok += 1
            if n_ok % 50 == 0:
                rate = n_ok / max(time.time() - t0, 0.001)
                print(f"[vid-temp] {n_ok}/{len(picks)} "
                      f"{rate:.2f} vid/s", flush=True)
    dt = time.time() - t0
    print(f"[vid-temp] done in {dt/60:.1f} min: {n_ok} records -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
