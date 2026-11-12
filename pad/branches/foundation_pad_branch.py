"""pad/branches/foundation_pad_branch.py  --  FoundPAD-style branch.

API_unq new branch. Wraps the FoundPAD model (`pad.train_foundation_pad`)
for runtime PAD scoring. The trainer exports `pad/models/foundation_pad_v1.onnx`
which this branch loads via the standard ORT session pattern. If the ONNX
isn't on disk yet (pre-train), this branch graceful-degrades to abstain.

Inputs:
  face_crop_bgr  - HxWx3 uint8 BGR
  pid            - optional, for telemetry only (model is stateless)

Output:
  BranchScore(name="foundation_pad", p_attack, reliability, latency_ms, ...)

Preprocessing matches CLIP ViT-L/14:
  - resize to 224x224 (bicubic)
  - CHW float32
  - mean=(0.48145466, 0.4578275, 0.40821073)
  - std =(0.26862954, 0.26130258, 0.27577711)
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

import numpy as np

from pad.types import BranchScore
from pad.calibrate import apply_calibration

BRANCH_NAME = "foundation_pad"

_MOD_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(os.path.dirname(_MOD_DIR), "models")
_MODEL_ONNX = os.path.join(_MODEL_DIR, "foundation_pad_v1.onnx")
_MODEL_ONNX_FP16 = os.path.join(_MODEL_DIR, "foundation_pad_v1.fp16.onnx")

# CLIP ViT-L/14 normalization (from openai/clip-vit-large-patch14 config).
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_INPUT_SIZE = 224

_SESSION = None
_SESSION_LOCK = threading.Lock()
_LOAD_ATTEMPTED = False
_MODEL_PATH = None  # set when load succeeds


def _maybe_fp16() -> bool:
    return os.environ.get("FF_USE_FP16", "0").lower() in ("1", "true", "yes")


def _load_session():
    global _SESSION, _LOAD_ATTEMPTED, _MODEL_PATH
    with _SESSION_LOCK:
        if _LOAD_ATTEMPTED:
            return _SESSION
        _LOAD_ATTEMPTED = True
        candidates = []
        if _maybe_fp16():
            candidates.append(_MODEL_ONNX_FP16)
        candidates.append(_MODEL_ONNX)
        try:
            import onnxruntime as ort
        except Exception as e:
            print(f"[PAD foundation] onnxruntime import failed: {e}")
            return None
        providers = ["CPUExecutionProvider"]
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            providers.insert(0, ("CUDAExecutionProvider", {"device_id": 0}))
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                _SESSION = ort.InferenceSession(path, providers=providers)
                _MODEL_PATH = path
                print(f"[PAD foundation] loaded {os.path.basename(path)} "
                      f"providers={[p[0] if isinstance(p, tuple) else p for p in providers]}")
                return _SESSION
            except Exception as e:
                print(f"[PAD foundation] load failed for {path}: {e}")
        print(f"[PAD foundation] no usable model in {_MODEL_DIR}; branch will abstain")
        return None


def _preprocess(face_crop_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 → CHW float32 normalized batch (1,3,224,224)."""
    import cv2
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        raise ValueError("empty crop")
    if face_crop_bgr.ndim == 2:
        face_crop_bgr = cv2.cvtColor(face_crop_bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (_INPUT_SIZE, _INPUT_SIZE):
        rgb = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(2, 0, 1)  # CHW
    return arr[None, ...].astype(np.float32)


def _abstain(reason: str, t0: float) -> BranchScore:
    return BranchScore(
        name=BRANCH_NAME,
        p_attack=0.5,
        reliability=0.0,
        latency_ms=float((time.monotonic() - t0) * 1000.0),
        meta={"reason": reason},
        dominant_class=None,
    )


def score(face_crop_bgr: np.ndarray, pid: Optional[int] = None) -> BranchScore:
    t0 = time.monotonic()
    sess = _load_session()
    if sess is None:
        return _abstain("no_model_loaded", t0)
    if face_crop_bgr is None or getattr(face_crop_bgr, "size", 0) == 0:
        return _abstain("empty_crop", t0)
    try:
        x = _preprocess(face_crop_bgr)
    except Exception as e:
        return _abstain(f"preprocess_failed:{e}", t0)
    try:
        out = sess.run(None, {"pixel_values": x})
        logits = np.asarray(out[0])
        # softmax binary
        ex = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = ex / ex.sum(axis=-1, keepdims=True)
        p_attack = float(probs[0, 1])
    except Exception as e:
        return _abstain(f"inference_failed:{e}", t0)

    p_calib = apply_calibration(BRANCH_NAME, p_attack)
    # FoundPAD is a strong global signal — full reliability when model loaded.
    # Conservative downgrade if the input was extreme (very small / partial face).
    h, w = face_crop_bgr.shape[:2]
    rel = 0.95 if min(h, w) >= 96 else 0.50

    dom = None
    if p_calib >= 0.85:
        # The branch doesn't differentiate attack subclass directly.
        # Leave dominant_class None so fusion's class-agreement gate doesn't
        # over-weight a single-source guess.
        dom = None

    return BranchScore(
        name=BRANCH_NAME,
        p_attack=float(p_calib),
        reliability=float(rel),
        latency_ms=float((time.monotonic() - t0) * 1000.0),
        meta={
            "p_raw": float(p_attack),
            "model_path": _MODEL_PATH,
            "crop_h": int(h),
            "crop_w": int(w),
        },
        dominant_class=dom,
    )


def is_available() -> bool:
    """Cheap probe so the WS loop can skip the branch when no model on disk."""
    return os.path.exists(_MODEL_ONNX) or os.path.exists(_MODEL_ONNX_FP16)
