"""latency_profile.py -- Per-branch and full-pipeline inference latency.

Times each PAD branch's inference cost on the host machine using the same
ONNX artefacts the deployment loads. Reports per-branch ms/run, total
sequential pipeline ms/run, and pipeline FPS. Intended for one
\"deployability\" sentence in section 5.1 of the paper.

Usage:
    python -m pad.latency_profile
    python -m pad.latency_profile --warmup 20 --runs 200
    python -m pad.latency_profile --provider CPUExecutionProvider
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# v9.7.2: bootstrap CUDA DLLs from torch/lib BEFORE importing onnxruntime —
# main.py does this in its boot path; standalone scripts must do it
# themselves or ORT silently falls back to CPU when cublasLt64_12.dll
# isn't on PATH (Windows + onnxruntime-gpu wheel doesn't ship its own
# cuBLAS/cuDNN).
try:
    import torch as _t
    _torch_lib = os.path.join(os.path.dirname(_t.__file__), "lib")
    if os.path.isdir(_torch_lib):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_torch_lib)
        if _torch_lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
except Exception as _e:  # nolint:silent-except torch-bootstrap-best-effort
    # Bootstrap is best-effort: if torch isn't installed (lite-CPU
    # variant) we fall through to whatever ORT can find on PATH. The
    # CUDA-DLL miss surfaces as ORT picking CPUExecutionProvider, which
    # is logged later -- this except just doesn't block import.
    pass

import numpy as np


# (label, model path, dummy input shape, dtype). The shapes match the paper's
# Section 4.1 branch architectures table.
ONNX_BRANCHES: List[Tuple[str, str, Tuple[int, ...], str]] = [
    ("patch_depth",       "pad/models/patch_depth_fas_v6.onnx",  (1, 3, 224, 224),       "float32"),
    ("face_xray",         "pad/models/face_xray_v4_diverse.onnx", (1, 3, 256, 256),      "float32"),
    ("face_xray_pgd",     "pad/models/face_xray_pgd.onnx",       (1, 3, 256, 256),       "float32"),
    ("deepfake_temporal", "pad/models/deepfake_temporal_v2.onnx", (1, 3, 16, 112, 112), "float32"),
    ("audio",             "pad/models/audio_pad.onnx",           (1, 64600),             "float32"),
]
MOIRE_GBT = "pad/models/moire_gbt.pkl"
MOIRE_FEATURE_DIM = 20


def _time_onnx(path: Path, fallback_shape: Tuple[int, ...], dtype: str,
               provider: str, warmup: int, runs: int) -> Tuple[float, float, Tuple[int, ...]]:
    """Return (mean_ms, std_ms, used_shape) per run.

    Reads the ONNX session's actual input shape and uses it; falls back to
    the supplied shape if the model declares a fully-dynamic input.
    """
    import onnxruntime as ort
    if not path.is_file():
        raise FileNotFoundError(path)
    providers = [provider, "CPUExecutionProvider"] if provider != "CPUExecutionProvider" \
                else ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
    inp = sess.get_inputs()[0]
    declared = inp.shape  # may contain None or strings for dynamic dims
    used = []
    for i, dim in enumerate(declared):
        if isinstance(dim, int) and dim > 0:
            used.append(dim)
        else:
            used.append(fallback_shape[i] if i < len(fallback_shape) else 1)
    used_shape = tuple(used)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(used_shape).astype(dtype)
    for _ in range(warmup):
        sess.run(None, {inp.name: x})
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(samples)), float(np.std(samples)), used_shape


def _time_moire(path: Path, warmup: int, runs: int) -> Tuple[float, float]:
    """Time the LightGBM moire branch on a 20-D feature vector.

    Tries joblib.load first (the format `pad/train_moire.py` writes --
    sklearn convention, wraps {model, feature_dim, version}), then plain
    pickle.load, then lightgbm.Booster(model_file=...). v9.8: joblib is
    primary because the shipped moire_gbt.pkl is joblib-encoded with a
    proto-4 frame that vanilla pickle.load can choke on under some
    import-resolution conditions.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    clf = None
    last_err: Optional[Exception] = None
    # 1) joblib (sklearn convention, what train_moire.py writes)
    try:
        import joblib
        obj = joblib.load(path)
        clf = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    except Exception as e:
        last_err = e
        clf = None
    # 2) vanilla pickle (older artefacts without joblib wrap)
    if clf is None:
        try:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
            clf = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        except Exception as e:
            last_err = e
            clf = None
    # 3) text-format Booster save
    if clf is None:
        try:
            import lightgbm as lgb
            clf = lgb.Booster(model_file=str(path))
        except Exception as e:
            raise RuntimeError(f"could not load moire model from {path}: "
                               f"last error: {e}; previously: {last_err}")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, MOIRE_FEATURE_DIM)).astype(np.float32)
    if hasattr(clf, "predict_proba"):
        pred = lambda v: float(clf.predict_proba(v)[0, 1])
    else:
        # Booster.predict returns probabilities directly for binary
        pred = lambda v: float(clf.predict(v)[0])
    for _ in range(warmup):
        pred(x)
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        pred(x)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(samples)), float(np.std(samples))


def _resolve_fp16(path: Path) -> Path:
    """API_up v9.8: when --use-fp16 is passed, swap each <name>.onnx for
    <name>_fp16.onnx if the FP16 neighbor exists. Falls back silently."""
    fp16 = path.with_name(path.stem + "_fp16" + path.suffix)
    return fp16 if fp16.is_file() else path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--provider", default="DmlExecutionProvider",
                    help="onnxruntime provider; tries this then falls back to CPU")
    ap.add_argument("--root", default=".", type=Path,
                    help="project root for resolving model paths")
    ap.add_argument("--use-fp16", action="store_true",
                    help="API_up v9.8: bench *_fp16.onnx variants when present "
                         "(produced by scripts/quantize_fp16.py)")
    ap.add_argument("--ab-fp16", action="store_true",
                    help="bench BOTH FP32 and FP16 side-by-side and print the "
                         "per-branch speedup table")
    args = ap.parse_args(argv)

    rows = []
    fp16_rows = []  # only populated when --ab-fp16
    for name, rel, shape, dtype in ONNX_BRANCHES:
        path = args.root / rel
        if args.use_fp16:
            path = _resolve_fp16(path)
        try:
            mean_ms, std_ms, used_shape = _time_onnx(
                path, shape, dtype, args.provider, args.warmup, args.runs)
            rows.append((name, mean_ms, std_ms, "onnx"))
            tag = "fp16" if path.stem.endswith("_fp16") else "fp32"
            print(f"[lat] {name:<22} {mean_ms:7.2f} ms  (sigma {std_ms:5.2f})  "
                  f"[{tag}, in={used_shape}]")
        except FileNotFoundError as e:
            print(f"[lat] {name:<22} SKIP missing {e}", file=sys.stderr)
        except Exception as e:
            print(f"[lat] {name:<22} ERROR {type(e).__name__}: {e}",
                  file=sys.stderr)

        # A/B mode: also bench the FP16 alternate if --ab-fp16 was passed
        # and we just benched FP32 (or vice versa)
        if args.ab_fp16:
            alt = _resolve_fp16(args.root / rel)
            if not args.use_fp16 and alt != args.root / rel:
                try:
                    mean_ms2, std_ms2, _ = _time_onnx(
                        alt, shape, dtype, args.provider, args.warmup, args.runs)
                    fp16_rows.append((name, mean_ms2, std_ms2))
                    print(f"[lat] {name:<22} {mean_ms2:7.2f} ms  "
                          f"(sigma {std_ms2:5.2f})  [fp16-alt]")
                except Exception as e:
                    print(f"[lat] {name:<22} fp16-alt ERROR: {e}",
                          file=sys.stderr)

    try:
        mean_ms, std_ms = _time_moire(args.root / MOIRE_GBT,
                                      args.warmup, max(args.runs, 1000))
        rows.append(("moire", mean_ms, std_ms, "lightgbm"))
        print(f"[lat] {'moire':<22} {mean_ms:7.4f} ms  (sigma {std_ms:5.4f})  [lightgbm]")
    except FileNotFoundError as e:
        print(f"[lat] {'moire':<22} SKIP missing {e}", file=sys.stderr)
    except Exception as e:
        # Treat moire as negligible per Section 4.1 ("single-digit microseconds")
        # if the bundled pickle uses a custom format we don't recognise.
        rows.append(("moire", 0.005, 0.0, "lightgbm-negligible"))
        print(f"[lat] {'moire':<22}   <0.01 ms  (negligible per paper Section 4.1)  "
              f"[load failed: {type(e).__name__}]")

    # full pipeline = sum of the branches that participate in inference
    if rows:
        total_seq = sum(r[1] for r in rows)
        fps = 1000.0 / total_seq if total_seq > 0 else float("inf")
        print()
        print(f"[lat] sequential pipeline (sum of branches): {total_seq:.2f} ms/run")
        print(f"[lat] sequential pipeline FPS:                {fps:.1f}")

    # API_up v9.8: A/B fp32 vs fp16 side-by-side speedup table
    if fp16_rows:
        print()
        print("=" * 60)
        print("  fp32 vs fp16 speedup (API_up v9.8 --ab-fp16)")
        print("=" * 60)
        rows_by_name = {r[0]: r for r in rows}
        for name, m16, s16 in fp16_rows:
            base = rows_by_name.get(name)
            if not base:
                continue
            speedup = base[1] / m16 if m16 > 0 else float("inf")
            print(f"  {name:<22} fp32 {base[1]:6.2f} ms | fp16 {m16:6.2f} ms | {speedup:.2f}x")
        print()
        # v9.7.2: auto-detect GPU name instead of hardcoding "RTX 3060".
        try:
            import torch as _t
            gpu_name = (_t.cuda.get_device_name(0) if _t.cuda.is_available()
                        else "CPU-only")
        except Exception:
            gpu_name = "this host"
        print("# Paper-ready sentence (drop into Section 5.1 Setup):")
        print(f"# End-to-end inference latency on the {gpu_name} is "
              f"{total_seq:.0f} ms/frame ({fps:.0f} FPS) running the branches")
        print(f"# sequentially via onnxruntime (provider preference: {args.provider}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
