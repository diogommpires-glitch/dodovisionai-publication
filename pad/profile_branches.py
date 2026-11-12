"""pad/profile_branches.py -- measure per-branch inference latency.

Loads every PAD ONNX on disk, feeds 50 synthetic inputs of the right
shape, and reports mean / p95 / p99 ms per call. Required for paper
§5.7 "runtime" table and the reviewer-expected "runs at X FPS"
statement.

Runs entirely on synthetic data -- no datasets required. The numbers
are therefore "structural" (model-inference-only, excluding detector
preprocessing). For a fuller end-to-end FPS number, feed the results
into PAD_REPORT.md §7 with the noted caveats.

Usage
-----
    python -m pad.profile_branches
    python -m pad.profile_branches --provider CPUExecutionProvider
    python -m pad.profile_branches --out reports/branch_latency.md --trials 100

Output
------
`reports/branch_latency.md` + `reports/branch_latency.csv` with per-
branch mean / p50 / p95 / p99 ms, and a total-end-to-end fused-stack
ms/frame estimate.
"""
from __future__ import annotations
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# Map each ONNX file to its input shape + "role" (for display).
# Shape is NCHW for image models, (N, C, T, H, W) for video, (N, T) for audio.
ONNX_PROFILES: Dict[str, Dict] = {
    "face_xray.onnx": {
        "role": "deepfake_branch_spatial",
        "shape": (1, 3, 224, 224),
        "dtype": "float32",
    },
    "face_xray_v3.onnx": {
        "role": "deepfake_branch_spatial_v3",
        "shape": (1, 3, 224, 224),
        "dtype": "float32",
    },
    "deepfake_temporal.onnx": {
        "role": "deepfake_branch_temporal",
        "shape": (1, 3, 16, 112, 112),
        "dtype": "float32",
    },
    "patch_depth_fas.onnx": {
        "role": "patch_depth_branch",
        "shape": (1, 3, 256, 256),
        "dtype": "float32",
    },
    "patch_depth_fas_v4.onnx": {
        "role": "patch_depth_branch_v4",
        "shape": (1, 3, 256, 256),
        "dtype": "float32",
    },
    "audio_pad.onnx": {
        "role": "audio_branch",
        "shape": (1, 64600),
        "dtype": "float32",
    },
    "face_detection_yunet_2023mar.onnx": {
        "role": "face_detector_preproc",
        # YuNet is setInputSize-dynamic; skip the generic profile
        "skip": True,
    },
}


def profile_one(sess, shape: Tuple[int, ...], trials: int,
                warmup: int = 5) -> Dict[str, float]:
    name = sess.get_inputs()[0].name
    # Try the declared shape first; if the model has dynamic dims, use ours
    x = np.random.randn(*shape).astype(np.float32)
    # Warm-up
    for _ in range(warmup):
        _ = sess.run(None, {name: x})
    # Measure
    deltas_ms = []
    for _ in range(trials):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        deltas_ms.append((time.perf_counter() - t0) * 1000.0)
    deltas_ms.sort()
    return {
        "trials":    trials,
        "mean_ms":   statistics.mean(deltas_ms),
        "stdev_ms":  statistics.pstdev(deltas_ms) if trials > 1 else 0.0,
        "p50_ms":    statistics.median(deltas_ms),
        "p95_ms":    deltas_ms[int(0.95 * trials)],
        "p99_ms":    deltas_ms[int(0.99 * trials)],
        "min_ms":    deltas_ms[0],
        "max_ms":    deltas_ms[-1],
    }


def emit_markdown(results: List[Dict], provider: str) -> str:
    out = [
        "# DodoVisionAI branch inference latency",
        "",
        f"_Provider: `{provider}`. {len(results)} branches profiled,_",
        f"_{results[0]['trials'] if results else 0} trials per branch,_",
        "_synthetic inputs (no preprocessing overhead)._",
        "",
        "| Branch / role | file | shape | mean (ms) | p50 | p95 | p99 | stdev |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        out.append(f"| `{r['role']}` | `{r['file']}` | `{r['shape']}` | "
                   f"{r['mean_ms']:.2f} | {r['p50_ms']:.2f} | "
                   f"{r['p95_ms']:.2f} | {r['p99_ms']:.2f} | "
                   f"{r['stdev_ms']:.2f} |")

    # Fused-stack estimate: one image frame -> moire + patch_depth +
    # deepfake_spatial + audio (assume 1 per frame) + deepfake_temporal
    # amortised over 16 frames (1/16 × its time).
    image_branches = {"deepfake_branch_spatial", "patch_depth_branch",
                      "audio_branch"}
    per_frame_ms = sum(r["mean_ms"] for r in results
                       if r["role"] in image_branches)
    temporal_ms = next((r["mean_ms"] for r in results
                        if r["role"] == "deepfake_branch_temporal"), 0.0)
    stack_ms = per_frame_ms + (temporal_ms / 16.0)
    fps = 1000.0 / stack_ms if stack_ms > 0 else 0.0
    out += [
        "",
        "## Fused stack (paper §5.7)",
        "",
        f"- Per-frame image branches (spatial deepfake + patch_depth + audio): **{per_frame_ms:.2f} ms**",
        f"- Temporal deepfake amortised over 16-frame clip: **{temporal_ms/16.0:.2f} ms/frame**",
        f"- **Total fused stack: {stack_ms:.2f} ms/frame -> {fps:.1f} FPS**",
        "",
        "_Caveat:_ this excludes face-detection (YuNet preprocessing,",
        "~5 ms/frame) and meta-fuser (~0.1 ms). End-to-end will be",
        "slower by those constants.",
    ]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider",
                    default="CUDAExecutionProvider",
                    help="onnxruntime EP (CUDAExecutionProvider or CPUExecutionProvider)")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--out", type=Path,
                    default=Path("reports/branch_latency.md"))
    ap.add_argument("--csv", type=Path,
                    default=Path("reports/branch_latency.csv"))
    ap.add_argument("--models-dir", type=Path,
                    default=Path(__file__).resolve().parent / "models")
    args = ap.parse_args(argv)

    import onnxruntime as ort

    avail = ort.get_available_providers()
    provider = args.provider if args.provider in avail else "CPUExecutionProvider"
    if provider != args.provider:
        print(f"[profile] requested {args.provider} unavailable; "
              f"falling back to {provider}. Available: {avail}",
              file=sys.stderr)

    results = []
    for fname, profile in ONNX_PROFILES.items():
        if profile.get("skip"):
            continue
        path = args.models_dir / fname
        if not path.is_file():
            print(f"[profile] skip (missing): {fname}")
            continue
        try:
            sess = ort.InferenceSession(str(path), providers=[provider])
            metrics = profile_one(sess, profile["shape"], args.trials)
        except Exception as e:
            print(f"[profile] FAIL on {fname}: {e}", file=sys.stderr)
            continue
        results.append({
            "file":  fname,
            "role":  profile["role"],
            "shape": profile["shape"],
            **metrics,
        })
        print(f"[profile] {profile['role']:<32s} {metrics['mean_ms']:6.2f} ms "
              f"(p95={metrics['p95_ms']:6.2f})")

    if not results:
        print("[profile] no models profiled; aborting", file=sys.stderr)
        return 3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(emit_markdown(results, provider), encoding="utf-8")
    print(f"[profile] wrote {args.out}")

    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "role", "shape", "trials", "mean_ms", "p50_ms",
                    "p95_ms", "p99_ms", "stdev_ms", "min_ms", "max_ms"])
        for r in results:
            w.writerow([r["file"], r["role"], str(r["shape"]), r["trials"],
                        f"{r['mean_ms']:.3f}", f"{r['p50_ms']:.3f}",
                        f"{r['p95_ms']:.3f}", f"{r['p99_ms']:.3f}",
                        f"{r['stdev_ms']:.3f}", f"{r['min_ms']:.3f}",
                        f"{r['max_ms']:.3f}"])
    print(f"[profile] wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
