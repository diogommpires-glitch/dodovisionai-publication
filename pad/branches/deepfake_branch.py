"""
DODO VISION AI v9.4 PAD Step 6 — Deepfake-stream detector.

Catches real-time face-swap / synthetic face streams injected via a
virtual camera — the threat rPPG can't handle because a sophisticated
pipeline re-injects a plausible pulse signal.

Four sub-signals fused:

  1. Spatial head — Face X-ray-style blending-boundary predictor.
     Refs: Li, Bao et al., CVPR 2020 "Face X-ray for More General
     Face Forgery Detection". Inference-only, ONNX at
     pad/models/face_xray.onnx.

  2. Temporal head — 16-frame video-swin or ViT-tiny clip classifier.
     Refs: Rössler et al., ICCV 2019 "FaceForensics++". Gated behind
     env flag `pad.deepfake.heavy=1` because the model is ~80 MB
     and inference is ~30 ms per clip.

  3. 3DMM identity coherence — variance of identity coefficients
     across frames. Real users show stable 3DMM fit; swaps show
     variance because the source→target mapping wobbles. Uses
     face_alignment's FAN (Face Alignment Network) for keypoints;
     swaps the identity basis with MediaPipe's transformation
     matrix as a lightweight stand-in when face_alignment's 3DMM
     isn't available on this host.

  4. Camera-fingerprint (PRNU) consistency — Photo Response
     Non-Uniformity estimated on the non-face background. A virtual
     camera injecting a pre-rendered stream breaks PRNU coherence
     across the frame because the background is from the user's
     real camera but the face region was rendered elsewhere.

Each sub-signal emits a p_attack in [0, 1]; fusion is a tiny MLP
(pad/models/deepfake_fusion.onnx) if available, else a weighted
log-odds sum.

References:
  - Li, Bao et al., Face X-ray, CVPR 2020.
  - Rössler et al., FaceForensics++, ICCV 2019.
  - Spatial-Temporal Deep Networks for Deepfake Detection (2025).
  - MIT Media Lab Detect Fakes (2020–2025).
"""
from __future__ import annotations
import os
import time
import threading
from typing import Optional, Dict, List, Deque
from collections import deque

import numpy as np

from pad.types import BranchScore
from pad.calibrate import apply_calibration

_MOD_DIR = os.path.dirname(os.path.abspath(__file__))
_XRAY_PATH = os.path.join(
    os.path.dirname(_MOD_DIR), "models", "face_xray.onnx")
_TEMPORAL_PATH = os.path.join(
    os.path.dirname(_MOD_DIR), "models", "deepfake_temporal.onnx")
_FUSION_PATH = os.path.join(
    os.path.dirname(_MOD_DIR), "models", "deepfake_fusion.onnx")

# API_up v9.8: FF_USE_FP16=1 routes to *_fp16.onnx variants when present.
# The script `scripts/quantize_fp16.py` produces them from the FP32 sources
# with onnxconverter_common; numerical parity verified within fp32 noise.
# Speedup on the 5090 Laptop: temporal head ~2.7x, face_xray ~1.0x (already
# at scheduling floor at 1.5 ms/inf), patch_depth ~1.2x. See
# pad/reports/fp16_quant/{bench,parity}.txt and API_UP_CHANGELOG.md.
_FF_USE_FP16 = os.environ.get("FF_USE_FP16", "0").lower() in ("1", "true", "yes")


from functools import lru_cache as _lru_cache

@_lru_cache(maxsize=8)
def _resolve_fp16(path: str) -> str:
    """If FF_USE_FP16 is set and a `<basename>_fp16.onnx` neighbor exists,
    return that path; otherwise return the original. Logs the swap once.

    v9.8: lru-cached -- the underlying os.path.isfile call is cheap but
    not free, and this runs in the loader path on every session boot
    plus inside the test suite many times in a row."""
    if not _FF_USE_FP16:
        return path
    head, tail = os.path.split(path)
    name, ext = os.path.splitext(tail)
    if name.endswith("_fp16"):
        return path
    fp16_path = os.path.join(head, f"{name}_fp16{ext}")
    if os.path.isfile(fp16_path):
        return fp16_path
    return path

_XRAY_SESSION = None
_XRAY_LOADED = False
_TEMPORAL_SESSION = None
_TEMPORAL_LOADED = False
_FUSION_SESSION = None
_FUSION_LOADED = False

_HEAVY_ENABLED = os.environ.get("PAD_DEEPFAKE_HEAVY", "").lower() in ("1", "true", "yes")
# v9.4.1+: operator override when the HF ONNX has its labels swapped
# (some "AI-detector" ONNX put FAKE at index 0, REAL at index 1). Set
# PAD_DEEPFAKE_FAKE_INDEX=0 if the branch is systematically reporting
# high p_attack on obviously live faces.
_FAKE_INDEX_OVERRIDE = None
try:
    _raw = os.environ.get("PAD_DEEPFAKE_FAKE_INDEX")
    if _raw is not None and _raw.strip():
        _FAKE_INDEX_OVERRIDE = int(_raw)
except Exception:
    pass

# v9.8.1 2026-05-12 (Audit L5: label-swap audit). Loud, single-line
# boot log so the operator can verify in a glance which index this
# branch is treating as "spoof". Silent label inversion (PAD_DEEPFAKE_
# FAKE_INDEX=0 when the model actually emits FAKE at index 1, or vice
# versa) is a footgun that flips every verdict the branch produces.
# We log here once at module import; the per-load probe in _load_xray
# adds a second confirmation line with the resolved preprocessing
# bundle's `fake_index`.
def _log_label_audit_banner():
    msg_override = ("OVERRIDE" if _FAKE_INDEX_OVERRIDE is not None else "default")
    eff = (str(_FAKE_INDEX_OVERRIDE) if _FAKE_INDEX_OVERRIDE is not None
           else "config-dependent (see _PREP_CANDIDATES)")
    print(f"[PAD deepfake] LABEL AUDIT: PAD_DEEPFAKE_FAKE_INDEX={msg_override} "
          f"effective_fake_index={eff} -- "
          f"a number 0|1; whichever index of the output softmax is "
          f"treated as the SPOOF probability. If verdicts look flipped "
          f"(real subject = high p_attack), invert this env.")

_log_label_audit_banner()

# Per-person rolling state: 3DMM identity coefficients + frame history
# for the temporal head + PRNU estimate + recent x-ray scores for a
# stability readout.
_STATE: Dict[int, Dict] = {}


def reset_all() -> None:
    _STATE.clear()


def _get(pid: int) -> Dict:
    st = _STATE.get(pid)
    if st is None:
        st = {
            "frame_buf": deque(maxlen=16),    # 16-frame window for temporal head
            "xray_history": deque(maxlen=30),
            "identity_coeffs": deque(maxlen=30),
            "prnu_residuals": deque(maxlen=20),
        }
        _STATE[pid] = st
    return st


def _load_onnx(path: str, label: str):
    if not os.path.exists(path):
        return None
    try:
        import onnxruntime as ort
        # v9.8.1 2026-05-12 (operator: ScatterND warnings still flooding
        # the boot log). The ScatterND warnings come from ORT's GLOBAL
        # "Default" logger (`[W:onnxruntime:Default, scatter_nd.h:51]`),
        # which is a separate sink from session-level loggers. The
        # session-options `log_severity_level=3` only catches
        # session-scope warnings (`[W:onnxruntime:, transformer_memcpy.cc]`,
        # empty logger name). The Default logger needs the module-level
        # API call. Defense-in-depth: call BOTH.
        # 0=verbose 1=info 2=warning 3=error 4=fatal.
        try:
            ort.set_default_logger_severity(3)
        except Exception:
            pass
        provs = [p for p in ort.get_available_providers()
                 if p in ("DmlExecutionProvider", "CUDAExecutionProvider")]
        if not provs:
            provs = ["CPUExecutionProvider"]
        else:
            provs = provs + ["CPUExecutionProvider"]
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess = ort.InferenceSession(path, sess_options=sess_opts, providers=provs)
        print(f"[PAD deepfake] {label} loaded: {sess.get_providers()}")
        return sess
    except Exception as e:
        print(f"[PAD deepfake] {label} load failed ({e}); skipping")
        return None


_XRAY_PREP = None     # dict with size / mean / std / fake_index / label


def _load_xray():
    global _XRAY_SESSION, _XRAY_LOADED, _XRAY_PREP
    if _XRAY_LOADED:
        return _XRAY_SESSION
    _XRAY_LOADED = True
    path = _resolve_fp16(_XRAY_PATH)
    if not os.path.exists(path):
        print(f"[PAD deepfake] face_xray.onnx not found — run pad/fetch_face_xray.py")
        return None
    if path != _XRAY_PATH:
        print(f"[PAD deepfake] FF_USE_FP16: face_xray loading {os.path.basename(path)} (FP16)")
    _XRAY_SESSION = _load_onnx(path, "face_xray")
    if _XRAY_SESSION is None:
        return None
    # v9.4.1+: auto-detect the correct preprocessing by probing the model
    # with dummy inputs. Matches three flavours:
    #   A) Classic Face X-ray U-Net: 256×256, /255.0, scalar-ish output
    #   B) HF ViT classifier (e.g. onnx-community deep-fake-detector-v2):
    #      224×224, ImageNet mean/std, [N, 2] logits → softmax, take index 1
    #   C) EfficientNet / XceptionNet deepfake classifiers: 299×299 or
    #      300×300, ImageNet norm, [N, 2] logits
    inp = _XRAY_SESSION.get_inputs()[0]
    out = _XRAY_SESSION.get_outputs()[0]
    n_classes = None
    if len(out.shape) == 2 and isinstance(out.shape[1], int):
        n_classes = int(out.shape[1])
    candidates = []
    if n_classes == 2:
        # HF-style binary classifier — try ViT, EfficientNet, Xception sizes
        candidates = [
            {"size": 224, "mean": (0.485, 0.456, 0.406),
             "std": (0.229, 0.224, 0.225), "fake_index": 1, "label": "HF-ViT-224"},
            {"size": 299, "mean": (0.485, 0.456, 0.406),
             "std": (0.229, 0.224, 0.225), "fake_index": 1, "label": "Xception-299"},
            {"size": 300, "mean": (0.485, 0.456, 0.406),
             "std": (0.229, 0.224, 0.225), "fake_index": 1, "label": "EffNet-300"},
        ]
    # Always include the classic Face X-ray layout as a last-resort fallback
    candidates.append({"size": 256, "mean": None, "std": None,
                       "fake_index": None, "label": "FaceXray-U-Net-256"})
    for cfg in candidates:
        try:
            dummy = np.zeros((1, 3, cfg["size"], cfg["size"]), dtype=np.float32)
            _XRAY_SESSION.run(None, {inp.name: dummy})
            _XRAY_PREP = cfg
            print(f"[PAD deepfake] face_xray preprocessing auto-detected: {cfg['label']}")
            return _XRAY_SESSION
        except Exception:
            continue
    print("[PAD deepfake] face_xray: no compatible preprocessing — branch disabled")
    _XRAY_SESSION = None
    _XRAY_PREP = None
    return None


def _load_temporal():
    """Load the MC3-18 temporal deepfake ONNX if present on disk.
    Auto-enables when pad/models/deepfake_temporal.onnx exists. Force-off with
    PAD_DEEPFAKE_HEAVY=0. With FF_USE_FP16=1, prefers the *_fp16.onnx variant
    (~2.7x speedup on Blackwell, fp32-noise parity)."""
    global _TEMPORAL_SESSION, _TEMPORAL_LOADED
    if _TEMPORAL_LOADED:
        return _TEMPORAL_SESSION
    _TEMPORAL_LOADED = True
    if os.environ.get("PAD_DEEPFAKE_HEAVY", "").lower() in ("0", "false", "no"):
        return None
    path = _resolve_fp16(_TEMPORAL_PATH)
    if not os.path.isfile(path):
        return None
    if path != _TEMPORAL_PATH:
        print(f"[PAD deepfake] FF_USE_FP16: temporal loading {os.path.basename(path)} (FP16, ~2.7x faster)")
    _TEMPORAL_SESSION = _load_onnx(path, "deepfake_temporal")
    return _TEMPORAL_SESSION


def _load_fusion():
    global _FUSION_SESSION, _FUSION_LOADED
    if _FUSION_LOADED:
        return _FUSION_SESSION
    _FUSION_LOADED = True
    _FUSION_SESSION = _load_onnx(_FUSION_PATH, "deepfake_fusion")
    return _FUSION_SESSION


def _prep_xray(face_crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    cfg = _XRAY_PREP or {"size": 256, "mean": None, "std": None}
    try:
        import cv2
        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (cfg["size"], cfg["size"]), interpolation=cv2.INTER_AREA)
    except Exception:
        return None
    x = rgb.astype(np.float32) / 255.0
    if cfg.get("mean") is not None and cfg.get("std") is not None:
        mean = np.asarray(cfg["mean"], dtype=np.float32)
        std = np.asarray(cfg["std"], dtype=np.float32)
        x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x


def _xray_score(face_crop_bgr: np.ndarray) -> Optional[float]:
    sess = _load_xray()
    if sess is None:
        return None
    x = _prep_xray(face_crop_bgr)
    if x is None:
        return None
    try:
        outs = sess.run(None, {sess.get_inputs()[0].name: x})
    except Exception:
        return None
    last = np.asarray(outs[-1])
    cfg = _XRAY_PREP or {}
    fake_index = cfg.get("fake_index")
    if _FAKE_INDEX_OVERRIDE is not None:
        fake_index = _FAKE_INDEX_OVERRIDE
    if fake_index is not None and last.ndim == 2 and last.shape[1] >= 2:
        # HF-style: logits over [real, fake]. Softmax then take fake prob.
        logits = last[0]
        ex = np.exp(logits - logits.max())
        probs = ex / ex.sum()
        return float(max(0.0, min(1.0, probs[fake_index])))
    # Classic Face X-ray U-Net: scalar or first element is already p_attack
    arr = last.reshape(-1)
    return float(max(0.0, min(1.0, arr[0])))


def _temporal_score(frame_buf: Deque[np.ndarray]) -> Optional[float]:
    """Score a 16-frame window through deepfake_temporal.onnx.
    Resolution AUTO-DETECTED from the loaded session's input shape so
    the same code path works for MC3-18 (112×112) and MViT v9p11
    (224×224). Output is [1, 2] logits (class 1 = spoof). Softmax →
    p_attack.

    v9.8.1 (2026-05-12) BUGFIX: previously hardcoded resize(112,112).
    After pack4 deployed the new MViT v9p11 model (138 MB,
    val_acc=0.9161) to deepfake_temporal.onnx, the ONNX session
    expected [1,3,16,224,224] but the resize still produced
    [1,3,16,112,112] → ORT shape-mismatch error → caught by the
    `except Exception: return None` below → temporal score silently
    absent from fusion meta. Operator was getting deepfake p_attack
    based on x-ray ALONE (~0.5 raw), missing the much stronger
    temporal liveness signal. Reading the input shape from the
    session itself makes the branch model-agnostic.
    """
    if os.environ.get("PAD_DEEPFAKE_HEAVY", "").lower() in ("0", "false", "no"):
        return None
    sess = _load_temporal()
    if sess is None or len(frame_buf) < 16:
        return None
    # Auto-detect required spatial size from the session input shape.
    # Expected: [batch, 3, 16, H, W]. Fall back to 112 if introspection
    # fails (matches the legacy MC3-18 path).
    target_hw = 112
    try:
        inp_shape = sess.get_inputs()[0].shape
        # ONNX dims may be ints or symbolic strings ('batch' etc.)
        # Spatial dims are positions [-2] and [-1].
        h_dim = inp_shape[-2] if isinstance(inp_shape[-2], int) else None
        w_dim = inp_shape[-1] if isinstance(inp_shape[-1], int) else None
        if h_dim and w_dim and h_dim == w_dim:
            target_hw = int(h_dim)
    except Exception:
        pass
    import cv2
    # Kinetics-400 mean/std (MC3-18 pretrained; MViT v9p11 reuses these
    # in our training pipeline so the same normalisation applies).
    MEAN = np.array([0.43216, 0.394666, 0.37645],
                    dtype=np.float32).reshape(3, 1, 1)
    STD = np.array([0.22803, 0.22145, 0.216989],
                   dtype=np.float32).reshape(3, 1, 1)
    clip_chw = []
    for fr in list(frame_buf):
        try:
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (target_hw, target_hw),
                             interpolation=cv2.INTER_AREA)
            chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
            chw = (chw - MEAN) / STD
            clip_chw.append(chw)
        except Exception:
            return None
    # Stack along time axis -> [C, T, H, W], add batch -> [1, C, T, H, W]
    x = np.stack(clip_chw, axis=1)[None, ...]
    try:
        outs = sess.run(None, {sess.get_inputs()[0].name: x})
    except Exception:
        return None
    logits = np.asarray(outs[0]).reshape(-1)
    if logits.size >= 2:
        ex = np.exp(logits - logits.max())
        probs = ex / ex.sum()
        # In our training: label 0 = live, label 1 = spoof
        return float(max(0.0, min(1.0, probs[1])))
    # Fallback: single-sigmoid model
    return float(max(0.0, min(1.0, logits[0])))


def _threedmm_score(identity_coeffs: Deque[np.ndarray]) -> Optional[float]:
    """High variance of 3DMM identity coefficients across frames
    suggests a wobbling source→target identity map — typical of
    real-time face swaps. We use the mean coefficient std over a
    30-frame window.
    """
    if len(identity_coeffs) < 8:
        return None
    arr = np.stack(list(identity_coeffs), axis=0)
    # Normalise per-coefficient scale first
    std = arr.std(axis=0)
    mu = np.abs(arr.mean(axis=0)) + 1e-6
    rel_std = std / mu
    mean_rel_std = float(np.mean(rel_std))
    # Live faces: rel_std ~0.02–0.08. Swaps: often > 0.15.
    return max(0.0, min(1.0, (mean_rel_std - 0.05) / 0.15))


# API_unq 2026-05-13 (P1.7): proper Lukas-Fridrich-Goljan MLE PRNU.
#
# References:
#   Lukáš, J., Fridrich, J. & Goljan, M. (2006). Digital Camera
#     Identification from Sensor Pattern Noise. IEEE TIFS 1(2),
#     205-214. [Canonical PRNU paper: W = I - F(I) where F is a
#     denoising filter; K_hat = mean(W_i) over many images of the
#     same camera; test query via NCC(W_query, K_hat).]
#   Chen, M., Fridrich, J., Goljan, M. & Lukáš, J. (2008).
#     Determining Image Origin and Integrity Using Sensor Noise.
#     IEEE TIFS 3(1). [Formalises PRNU as forgery localisation.]
#   Koopman, M., Macarulla Rodriguez, A. & Geradts, Z. (2018).
#     Detection of Deepfake Video Manipulation. IMVIP 2018.
#     [Face-vs-background consistency test for deepfakes.]
#
# Module-level per-device fingerprint cache. K_hat is enrolled by
# `prnu_enroll(frame_bgr)` over a quiet 50+ frame calibration phase
# (operator-driven: the first 50 healthy frames from each camera).
# Without K_hat, the branch falls back to the LIGHTWEIGHT variance-
# ratio placeholder (kept as `_prnu_score_legacy`) so the existing
# fusion math remains stable.

_PRNU_K_HAT: Optional[np.ndarray] = None
_PRNU_ENROLL_BUFFER: list = []
_PRNU_MIN_ENROLL = 50
_PRNU_WINDOW = 5  # Wiener-filter window
_PRNU_LOCK = threading.Lock() if "threading" in globals() else None
_PRNU_K_HAT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "prnu_k_hat.npy"
)


def _autoload_prnu_k_hat() -> bool:
    """API_unq P1.7 auto-load: if pad/models/prnu_k_hat.npy exists (from
    scripts/_unq_prnu_synth_enroll.py or a real-enrollment run), set the
    module-level K_hat so the MLE PRNU branch fires the face-vs-bg NCC
    check at runtime instead of falling back to variance-ratio.

    Returns True on successful load.
    """
    global _PRNU_K_HAT
    if not os.path.exists(_PRNU_K_HAT_PATH):
        return False
    try:
        k_hat = np.load(_PRNU_K_HAT_PATH).astype(np.float32)
        _PRNU_K_HAT = k_hat
        print(f"[PAD deepfake] PRNU K_hat auto-loaded from "
              f"{os.path.basename(_PRNU_K_HAT_PATH)}  shape={k_hat.shape}")
        return True
    except Exception as e:
        print(f"[PAD deepfake] PRNU K_hat auto-load failed: {e}")
        return False


# Fire auto-load at module import.
_autoload_prnu_k_hat()


def _wiener_denoise(gray: np.ndarray, window: int = _PRNU_WINDOW) -> np.ndarray:
    """Local-mean / local-variance Wiener filter approximation, Lukas
    Fridrich Goljan 2006 baseline. scipy.signal.wiener mirrors this
    when available; otherwise we fall back to a Gaussian-blur proxy
    (good enough for the residual extraction).
    """
    try:
        from scipy.signal import wiener as _scipy_wiener
        return _scipy_wiener(gray.astype(np.float32), mysize=window)
    except Exception:
        import cv2
        return cv2.GaussianBlur(gray.astype(np.float32),
                                (window, window), 1.0)


def _prnu_residual(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Extract sensor noise residual W = I - F(I) per LFG 2006."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    try:
        import cv2
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    except Exception:
        return None
    if gray.shape[0] < 32 or gray.shape[1] < 32:
        return None
    denoised = _wiener_denoise(gray)
    return gray - denoised


def prnu_enroll(frame_bgr: np.ndarray) -> Dict[str, float]:
    """Add a frame to the enrollment buffer; once >= _PRNU_MIN_ENROLL
    healthy frames have been accumulated, freeze K_hat = mean(W_i).

    Operator usage: call this during a quiet calibration phase (e.g.
    the 90 s `calibration_lap` neutral stage). Returns a small status
    dict so the UI can display progress.
    """
    w = _prnu_residual(frame_bgr)
    if w is None:
        return {"ok": 0.0, "n": float(len(_PRNU_ENROLL_BUFFER))}
    if _PRNU_LOCK is not None:
        with _PRNU_LOCK:
            return _prnu_enroll_locked(w)
    return _prnu_enroll_locked(w)


def _prnu_enroll_locked(w: np.ndarray) -> Dict[str, float]:
    global _PRNU_K_HAT
    _PRNU_ENROLL_BUFFER.append(w)
    # Cap buffer size to keep memory bounded
    if len(_PRNU_ENROLL_BUFFER) > _PRNU_MIN_ENROLL * 4:
        _PRNU_ENROLL_BUFFER.pop(0)
    if len(_PRNU_ENROLL_BUFFER) >= _PRNU_MIN_ENROLL and _PRNU_K_HAT is None:
        try:
            # Normalize all residuals to same shape; use the most-common.
            shapes = [r.shape for r in _PRNU_ENROLL_BUFFER]
            from collections import Counter as _C
            target = _C(shapes).most_common(1)[0][0]
            stack = np.stack([r for r in _PRNU_ENROLL_BUFFER if r.shape == target])
            _PRNU_K_HAT = stack.mean(axis=0).astype(np.float32)
        except Exception:
            _PRNU_K_HAT = None
    return {
        "ok": 1.0,
        "n": float(len(_PRNU_ENROLL_BUFFER)),
        "enrolled": float(_PRNU_K_HAT is not None),
    }


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation in [-1, 1]."""
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = float(np.sqrt((a_c * a_c).sum() * (b_c * b_c).sum()))
    if denom < 1e-9:
        return 0.0
    return float((a_c * b_c).sum() / denom)


def _prnu_score_mle(frame_bgr: np.ndarray, face_bbox: Optional[tuple]) -> Optional[float]:
    """API_unq P1.7: face-vs-background NCC consistency test.

    Per Koopman et al. 2018, a real face shares the camera PRNU
    fingerprint with the surrounding background; a deepfake breaks
    that consistency. We compute:
      - face_ncc : NCC(W_face,    K_hat[face_bbox])
      - bg_ncc   : NCC(W_bg,      K_hat[bg_mask])
    and flag inconsistency:
      - face_ncc << bg_ncc  → deepfake (face replaced, bg authentic)
      - both << threshold   → virtual camera (no shared fingerprint)
      - both ≈ enroll_mean  → live (consistent fingerprint)

    Returns p_attack in [0, 1], or None if enrollment not complete.
    """
    if _PRNU_K_HAT is None:
        return None
    if face_bbox is None:
        return None
    w = _prnu_residual(frame_bgr)
    if w is None or w.shape != _PRNU_K_HAT.shape:
        return None
    x0, y0, x1, y1 = face_bbox
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(w.shape[1], int(x1)); y1 = min(w.shape[0], int(y1))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    # Face NCC
    face_w = w[y0:y1, x0:x1]
    face_k = _PRNU_K_HAT[y0:y1, x0:x1]
    face_ncc = _ncc(face_w, face_k)
    # Background NCC: mask out the face region
    bg_mask = np.ones(w.shape, dtype=bool)
    bg_mask[y0:y1, x0:x1] = False
    if bg_mask.sum() < 4096:
        return None
    bg_w = w[bg_mask]
    bg_k = _PRNU_K_HAT[bg_mask]
    bg_ncc = _ncc(bg_w, bg_k)
    # Decision logic (Koopman 2018 style thresholds; the actual deploy
    # thresholds should be re-fit from an enrollment-derived
    # distribution on the operator's device).
    delta = bg_ncc - face_ncc
    if face_ncc < 0.02 and bg_ncc < 0.02:
        # Both fingerprints absent → virtual camera / injection
        return 0.85
    if face_ncc < 0.02 and bg_ncc >= 0.05:
        # Background authentic, face missing → classic deepfake
        return 0.90
    if delta > 0.05:
        # Significant face-vs-background gap → likely deepfake
        return 0.50 + min(0.40, delta * 4.0)
    # Consistent fingerprint → live
    return max(0.05, 0.20 - face_ncc * 0.5)


def _prnu_score_legacy(frame_bgr: np.ndarray, face_bbox: Optional[tuple]) -> Optional[float]:
    """LIGHTWEIGHT pre-API_unq placeholder; kept as fallback when
    K_hat enrollment is incomplete. Variance-ratio between face
    residual and background residual.
    """
    if frame_bgr is None or frame_bgr.size == 0 or face_bbox is None:
        return None
    try:
        import cv2
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
        residual = gray.astype(np.float32) - blur.astype(np.float32)
    except Exception:
        return None
    x0, y0, x1, y1 = face_bbox
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(residual.shape[1], x1); y1 = min(residual.shape[0], y1)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    face_res = residual[y0:y1, x0:x1]
    bg_mask = np.ones(residual.shape, dtype=bool)
    bg_mask[y0:y1, x0:x1] = False
    if bg_mask.sum() < 1000:
        return None
    bg_res = residual[bg_mask]
    face_var = float(np.var(face_res))
    bg_var = float(np.var(bg_res))
    if bg_var < 1e-6:
        return None
    ratio = face_var / bg_var
    if ratio < 0.3 or ratio > 3.0:
        return 0.85
    if 0.5 < ratio < 2.0:
        return 0.15
    return 0.5


def _prnu_score(frame_bgr: np.ndarray, face_bbox: Optional[tuple]) -> Optional[float]:
    """Dispatcher: prefer MLE PRNU once K_hat is enrolled; fall back
    to the legacy variance-ratio placeholder otherwise.

    Enrollment is silent / opt-in: call `prnu_enroll(frame)` from the
    calibration_lap neutral phase or from a dedicated calibration
    endpoint. Without enrollment, behaviour is identical to pre-API_unq.
    """
    if _PRNU_K_HAT is not None:
        out = _prnu_score_mle(frame_bgr, face_bbox)
        if out is not None:
            return out
    return _prnu_score_legacy(frame_bgr, face_bbox)


def prnu_status() -> Dict[str, float]:
    """Diagnostic snapshot for /pad/0 telemetry + operator UI."""
    return {
        "enrolled": float(_PRNU_K_HAT is not None),
        "n_buffer": float(len(_PRNU_ENROLL_BUFFER)),
        "min_enroll": float(_PRNU_MIN_ENROLL),
    }


def _fuse(feats: Dict[str, float]) -> float:
    """Weighted log-odds fusion; sub-signals weighted per the
    FaceForensics++ and MIT Detect Fakes literature on multi-modal
    contributions (spatial dominant, temporal next, 3DMM + PRNU tie).
    """
    import math
    weights = {"xray": 1.6, "temporal": 1.2, "threedmm": 0.8, "prnu": 0.6}
    logit = 0.0
    any_observed = False
    for name, w in weights.items():
        v = feats.get(name)
        if v is None:
            continue
        any_observed = True
        v = min(max(v, 1e-3), 1 - 1e-3)
        logit += w * math.log(v / (1 - v))
    if not any_observed:
        return 0.5
    return 1.0 / (1.0 + math.exp(-logit))


def feed_frame(pid: int, frame_bgr: np.ndarray,
               identity_coeffs: Optional[np.ndarray] = None,
               face_bbox: Optional[tuple] = None) -> None:
    """Optional feed of raw-frame state for the temporal + 3DMM sub-signals.
    The `score()` call then reads from this state."""
    st = _get(pid)
    if frame_bgr is not None and frame_bgr.size > 0:
        st["frame_buf"].append(frame_bgr)
        prnu = _prnu_score(frame_bgr, face_bbox)
        if prnu is not None:
            st["prnu_residuals"].append(prnu)
    if identity_coeffs is not None and identity_coeffs.size > 0:
        st["identity_coeffs"].append(identity_coeffs.astype(np.float32))


def score(pid: int,
          face_crop_bgr: Optional[np.ndarray] = None) -> BranchScore:
    t0 = time.monotonic()
    st = _get(pid)
    feats: Dict[str, float] = {}

    if face_crop_bgr is not None:
        xr = _xray_score(face_crop_bgr)
        if xr is not None:
            feats["xray"] = xr
            st["xray_history"].append(xr)

    tmp = _temporal_score(st["frame_buf"])
    if tmp is not None:
        feats["temporal"] = tmp

    td = _threedmm_score(st["identity_coeffs"])
    if td is not None:
        feats["threedmm"] = td

    if st["prnu_residuals"]:
        feats["prnu"] = float(np.mean(list(st["prnu_residuals"])))

    # Fused p_attack — prefer a trained MLP head, fall back to log-odds.
    p_attack = 0.5
    fusion_sess = _load_fusion()
    if fusion_sess is not None and feats:
        try:
            feat_vec = np.asarray(
                [[feats.get("xray", 0.5),
                  feats.get("temporal", 0.5),
                  feats.get("threedmm", 0.5),
                  feats.get("prnu", 0.5)]],
                dtype=np.float32,
            )
            out = fusion_sess.run(None, {fusion_sess.get_inputs()[0].name: feat_vec})
            p_attack = float(np.asarray(out[0]).reshape(-1)[0])
        except Exception:
            p_attack = _fuse(feats)
    else:
        p_attack = _fuse(feats) if feats else 0.5

    # Reliability — at least the x-ray head must be up, else the branch
    # is very weak. 3DMM + PRNU alone count half-weight.
    has_strong = any(k in feats for k in ("xray", "temporal"))
    n_feats = len(feats)
    reliability = min(1.0, (0.5 if has_strong else 0.2) + 0.15 * n_feats)

    p_attack = apply_calibration("deepfake", max(0.0, min(1.0, p_attack)))
    # v9.8.1 2026-05-12 (9th-pass deep audit): tighten the
    # DEEPFAKE_STREAM gate. Previous gate was just `p_attack > 0.80`
    # but the reliability formula
    # `(0.5 if has_strong else 0.2) + 0.15 * n_feats` caps at 1.0
    # even when individual feature scores are all marginal (e.g.
    # xray=0.5, temporal=0.55, threedmm=None, prnu=0.5 -> p_attack
    # ~0.78 raw, calibrator+prior-shift can push past 0.80). Require
    # BOTH:
    #   - reliability >= 0.50 (more than just "xray loaded successfully")
    #   - p_attack >= 0.85 (real spoof confidence, post-calibration)
    # so the alarming UX of DEEPFAKE_STREAM only fires when the branch
    # is genuinely confident. p_attack still flows to fusion unchanged.
    dominant = None
    if reliability >= 0.50 and p_attack >= 0.85:
        dominant = "DEEPFAKE_STREAM"
    return BranchScore(
        name="deepfake",
        p_attack=round(p_attack, 4),
        reliability=round(reliability, 3),
        latency_ms=(time.monotonic() - t0) * 1000.0,
        meta={k: round(v, 4) for k, v in feats.items()},
        dominant_class=dominant,
    )
