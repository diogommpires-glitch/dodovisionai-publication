"""pad/score_ffpp_branches.py -- score FaceForensics++ live/spoof
clips through the spatial PAD branches (moire, patch_depth,
deepfake-spatial).

Backs queue_mega_v2 stages 60-62 and POST_QUEUE_PLAN.md §E.1
(adding the FF++ row to Table 2 cross-dataset matrix).

Input layout (already preprocessed by pad.preprocess_ffpp):
    E:/pad_datasets/video_clips_ffpp/
        live/<video_name>.npy        # (16, 112, 112, 3) uint8 clips
        spoof/<video_name>.npy
        _manifest.jsonl              # {video, label, source, n_clips}

Output (one jsonl per branch under reports/):
    reports/score_ffpp_moire_v5.jsonl
    reports/score_ffpp_patch_depth_v5.jsonl
    reports/score_ffpp_deepfake_v5.jsonl

Each record:
    {
        "path": "live/ORIGINAL_000.npy",
        "label": 0|1,                 # 0=live, 1=spoof
        "branch": "moire",
        "p_attack": float,
        "reliability": float,
        "frame_idx_used": int,        # which frame in the 16-clip we scored
        "source": "Deepfakes" | "FaceSwap" | "NeuralTextures" | "ORIGINAL"
    }

Caveats
-------
The video_clips_ffpp/ pre-processing was done at 112x112 to match the
MC3-18 temporal head's input contract. Spatial branches normally
expect 224x224 (patch_depth) or 256x256 (moire) face crops. We upsample
the middle frame to the branch's expected size; this loses some
high-frequency detail that moire would normally see, so the FF++
moire AUC reported here is a LOWER BOUND. Re-extract face crops from
the raw FaceForensics videos at 256x256 (via pad.preprocess_ffpp with
--out-size 256) for a tighter number; keep this for v5 and replace at
camera-ready time if reviewers ask.

Usage
-----
    python -m pad.score_ffpp_branches \\
        --root E:/pad_datasets/video_clips_ffpp \\
        --branches moire patch_depth deepfake \\
        --out-dir reports \\
        --tag v5

    # speed: --max-clips 100 for smoke
    # frame: --frame-idx 8 (default = middle frame)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
QUEUE_MARKER = REPO_ROOT / "queue" / "mega" / "queue_done.marker"


def _load_clip(npy_path: Path) -> Optional[np.ndarray]:
    """Load a (T, H, W, 3) uint8 clip from .npy. Return None on failure.

    v9.7.1: also accept (T, 3, H, W) channels-first layout (PyTorch
    convention). The pad.preprocess_ffpp pipeline saves channels-last
    by default but Some upstream variants (e.g., MC3-18 temporal-head
    feed) emit channels-first. Detect and transpose so the queue v2
    FF++ stages stop discarding 100% of inputs.
    """
    try:
        arr = np.load(npy_path)
    except Exception as e:
        print(f"[ffpp] {npy_path.name}: load failed ({e})", file=sys.stderr)
        return None
    # Accept (T, H, W, 3) and (T, 3, H, W); transpose the latter to
    # the channels-last contract this module documents.
    if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
        arr = arr.transpose(0, 2, 3, 1)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        print(f"[ffpp] {npy_path.name}: unexpected shape {arr.shape}", file=sys.stderr)
        return None
    return arr


def _frame_for_branch(clip: np.ndarray, frame_idx: int, branch: str) -> np.ndarray:
    """Pull one frame from the clip and resize to the branch's expected
    spatial resolution. Returns BGR uint8 (the branch contract).
    """
    if frame_idx < 0 or frame_idx >= clip.shape[0]:
        frame_idx = clip.shape[0] // 2
    rgb = clip[frame_idx]  # H, W, 3 uint8
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    target = 256 if branch == "moire" else 224
    if bgr.shape[0] != target or bgr.shape[1] != target:
        bgr = cv2.resize(bgr, (target, target), interpolation=cv2.INTER_CUBIC)
    return bgr


def _source_from_name(stem: str) -> str:
    """'NeuralTextures_590_034' -> 'NeuralTextures'. 'ORIGINAL_000' -> 'ORIGINAL'."""
    return stem.split("_", 1)[0] if "_" in stem else stem


def score_clip_one_branch(clip: np.ndarray, branch: str, frame_idx: int) -> Dict:
    """Run one branch on one frame; return dict with p_attack, reliability."""
    img = _frame_for_branch(clip, frame_idx, branch)
    if branch == "moire":
        from pad.branches import moire_branch
        s = moire_branch.score(img)
    elif branch == "patch_depth":
        from pad.branches import patch_depth_branch
        s = patch_depth_branch.score(img)
    elif branch == "deepfake":
        # deepfake-spatial = face_xray ONNX. The branch's score() is
        # stateful on `pid`; we sidestep it by calling the underlying
        # ONNX session directly via a thin helper.
        s = _score_face_xray_direct(img)
    else:
        raise ValueError(f"unknown branch: {branch}")
    return {
        "p_attack": float(getattr(s, "p_attack", 0.5)),
        "reliability": float(getattr(s, "reliability", 0.0)),
    }


def _score_face_xray_direct(bgr_224: np.ndarray) -> "object":
    """Call the face_xray ONNX directly so we don't need PersonState.
    Returns an object with .p_attack and .reliability attributes.
    """
    from pad.branches import deepfake_branch as _df
    sess = _df._load_xray_session() if hasattr(_df, "_load_xray_session") else None
    if sess is None:
        # Fallback: just load directly
        try:
            import onnxruntime as ort
            xray_path = REPO_ROOT / "pad" / "models" / "face_xray.onnx"
            if not xray_path.exists():
                class _Stub: p_attack = 0.5; reliability = 0.0
                return _Stub()
            sess = ort.InferenceSession(
                str(xray_path),
                providers=[p for p in ort.get_available_providers()
                           if p in ("CUDAExecutionProvider", "CPUExecutionProvider")]
            )
        except Exception:
            class _Stub: p_attack = 0.5; reliability = 0.0
            return _Stub()
    rgb = cv2.cvtColor(bgr_224, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = rgb.transpose(2, 0, 1)[None]  # 1,3,224,224
    try:
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: rgb})
        # Either a sigmoid 'p_attack' or 2-class logits — try both
        arr = out[0]
        if arr.ndim == 1 or arr.shape[-1] == 1:
            p = float(np.asarray(arr).flatten()[0])
            if p > 1.0:
                p = 1.0 / (1.0 + np.exp(-p))
        else:
            logits = np.asarray(arr).flatten()[:2]
            ex = np.exp(logits - logits.max())
            p = float(ex[1] / ex.sum())
        class _R: pass
        r = _R()
        r.p_attack = p
        r.reliability = 1.0
        return r
    except Exception as e:
        print(f"[ffpp] face_xray direct call failed: {e}", file=sys.stderr)
        class _Stub: p_attack = 0.5; reliability = 0.0
        return _Stub()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", type=Path, required=True,
                    help="dir containing live/*.npy + spoof/*.npy")
    ap.add_argument("--branches", nargs="+",
                    default=["moire", "patch_depth", "deepfake"],
                    choices=["moire", "patch_depth", "deepfake"])
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    ap.add_argument("--tag", default="v5",
                    help="tag inserted into output filenames")
    ap.add_argument("--frame-idx", type=int, default=8,
                    help="which frame of the 16-frame clip to score (default 8 = middle)")
    ap.add_argument("--max-clips", type=int, default=0,
                    help="process at most N clips per class (0 = all)")
    ap.add_argument("--allow-during-training", action="store_true",
                    help="bypass queue_done.marker guard")
    args = ap.parse_args(argv)

    if not QUEUE_MARKER.exists() and not args.allow_during_training:
        print("[ffpp] refusing to run: queue_mega has not finished",
              file=sys.stderr)
        print("       pass --allow-during-training to override",
              file=sys.stderr)
        return 2

    if not args.root.is_dir():
        print(f"[ffpp] not a directory: {args.root}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Gather clips by label
    files: List[tuple[Path, int]] = []  # (path, label)
    for sub, label in (("live", 0), ("spoof", 1)):
        d = args.root / sub
        if not d.is_dir():
            continue
        npys = sorted(d.glob("*.npy"))
        if args.max_clips > 0:
            npys = npys[: args.max_clips]
        files.extend([(p, label) for p in npys])

    if not files:
        print(f"[ffpp] no .npy clips under {args.root}/{{live,spoof}}",
              file=sys.stderr)
        return 3

    print(f"[ffpp] {len(files)} clips total  branches={args.branches}  "
          f"frame_idx={args.frame_idx}")

    # Pre-load each branch ONCE (lazy import inside score_clip_one_branch)
    # is fine — first call paid the cost.

    # Open output JSONLs (one per branch)
    out_handles: Dict[str, "object"] = {}
    out_paths: Dict[str, Path] = {}
    for br in args.branches:
        out_paths[br] = args.out_dir / f"score_ffpp_{br}_{args.tag}.jsonl"
        out_handles[br] = open(out_paths[br], "w", encoding="utf-8")

    counts: Dict[str, int] = {br: 0 for br in args.branches}
    t0 = time.monotonic()

    try:
        for npy, label in files:
            clip = _load_clip(npy)
            if clip is None:
                continue
            rel = npy.relative_to(args.root)
            source = _source_from_name(npy.stem)
            for br in args.branches:
                try:
                    r = score_clip_one_branch(clip, br, args.frame_idx)
                except Exception as e:
                    print(f"[ffpp] {npy.name}/{br}: {e}", file=sys.stderr)
                    continue
                rec = {
                    "path": str(rel).replace("\\", "/"),
                    "label": int(label),
                    "branch": br,
                    "p_attack": round(r["p_attack"], 4),
                    "reliability": round(r["reliability"], 4),
                    "frame_idx_used": int(args.frame_idx),
                    "source": source,
                }
                out_handles[br].write(json.dumps(rec) + "\n")
                counts[br] += 1
            if (counts.get(args.branches[0], 0) % 50) == 0:
                elapsed = time.monotonic() - t0
                rate = counts[args.branches[0]] / max(elapsed, 1e-3)
                print(f"  [{npy.name}] label={label} "
                      f"({counts[args.branches[0]]}/{len(files)}, "
                      f"{rate:.1f} clips/s)")
    finally:
        for fh in out_handles.values():
            fh.close()

    elapsed = time.monotonic() - t0
    print(f"[ffpp] done in {elapsed/60:.1f} min")
    for br in args.branches:
        print(f"  {br:14s} -> {out_paths[br]} ({counts[br]} records)")
    print(f"[ffpp] next: merge with deepfake_temporal_v2_records_ffpp.jsonl "
          f"via pad.merge_jsonls (queue_mega_v2 stage 63)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
