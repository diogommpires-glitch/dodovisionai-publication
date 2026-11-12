"""
DODO VISION AI — Detectors
MediaPipe Face Landmarker, DeepFace (age/gender/race), HSEmotion (emotion).
InsightFace removed: requires Visual C++ Build Tools on Windows, not worth the hassle.
open-rppg removed: JAX dependency fails on Windows, custom POS algorithm is sufficient.

v8.6: GPU/CPU saturation
  - HSEmotion ONNX: forced CUDAExecutionProvider with CPU fallback
  - TensorFlow (DeepFace): enabled per-process memory growth
  - ThreadPoolExecutor: scaled to logical cores via cfg.THREAD_POOL_WORKERS
  - Pre-warm: every model JIT-compiled before the first real frame

v8.7 BUGFIX: probe CUDA at import time so we don't ask ORT for a provider
the system can't load (which spams a long stack trace on Windows when cuDNN
isn't installed). Falls back to CPU silently and only logs once.
"""
import os
# v8.7: silence TF noise BEFORE we import anything that imports TF.
# DeepFace imports TF lazily, so setting these here is reliable.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")           # ERROR-only TF logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_LOGGING_VERBOSITY", "3")
# ORT also chats; tone it down (level 3 = ERROR).
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
import logging as _logging
_logging.getLogger("tensorflow").setLevel(_logging.ERROR)
_logging.getLogger("absl").setLevel(_logging.ERROR)

import cv2
import numpy as np
import mediapipe as mp
import threading
from concurrent.futures import ThreadPoolExecutor
import config as cfg

# v9.7.6 BUGFIX: Serialize HSEmotion ONNX inference across threads.
# The ORT DirectML provider is NOT thread-safe for concurrent .run()
# calls on a shared session. We were calling _fer.predict_emotions
# from two threads simultaneously:
#   1. _bg_deepface  (main.py:539) → run_deepface → run_hsemotion
#   2. _process_frame_fast (main.py:1228) → run_hsemotion
# Crash signature was a STATUS_ACCESS_VIOLATION (rc=-1073741819) inside
# onnxruntime_inference_collection.py:275, ~30s after first frame —
# the moment the bg DeepFace pool fired alongside a fast-path inference.
# This lock makes those calls serialize cheaply (HSEmotion is ~5-10ms
# per inference, so the contention is a non-issue compared to crashing).
_hsemo_lock = threading.Lock()

# ── Thread pool for parallel GPU work ──
# v8.6: workers are sized in config.py from os.cpu_count() with a hard cap so
# we don't oversubscribe the kernel scheduler on huge desktops.
pool = ThreadPoolExecutor(max_workers=cfg.THREAD_POOL_WORKERS,
                          thread_name_prefix="dodo-infer")
print(f"[OK] ThreadPoolExecutor: {cfg.THREAD_POOL_WORKERS} workers")

# ── v8.8: probe ONNX Runtime providers ONCE with explicit CUDA provider
# options (device_id, memory limits, cuDNN algo search). This prevents
# fallback to integrated GPU 0 and eliminates Error 126 / silent CPU
# fallback on multi-GPU systems.
_CUDA_PROVIDER_OPTS = getattr(cfg, "ONNX_CUDA_PROVIDER_OPTIONS", {})
_USABLE_ONNX_PROVIDERS = ["CPUExecutionProvider"]
try:
    import onnxruntime as _ort_probe
    _avail = _ort_probe.get_available_providers()
    if "CUDAExecutionProvider" in _avail:
        try:
            import io as _io, contextlib as _ctx
            from onnx import helper as _oh, TensorProto as _TP
            _node = _oh.make_node("Identity", inputs=["x"], outputs=["y"])
            _g = _oh.make_graph([_node], "probe",
                [_oh.make_tensor_value_info("x", _TP.FLOAT, [1])],
                [_oh.make_tensor_value_info("y", _TP.FLOAT, [1])])
            # v9.1.4: opset 11 + explicit IR 7 — ORT 1.18.x only supports
            # max IR version 10, and the default IR generated from opset 13
            # is 9 on new onnx releases but can jump to 11+ on older ones.
            # Pinning IR=7 is the safest lowest-common-denominator — every
            # ORT build since 1.4 accepts it, so the probe outcome reflects
            # the actual CUDA availability, not a model-format mismatch.
            _m = _oh.make_model(_g, opset_imports=[_oh.make_opsetid("", 11)])
            _m.ir_version = 7
            _buf = _m.SerializeToString()
            # v8.8: pass explicit provider options to bind to the correct GPU
            _probe_providers = [
                ("CUDAExecutionProvider", _CUDA_PROVIDER_OPTS),
                "CPUExecutionProvider",
            ]
            # v9.1.4: capture stderr so we can extract the REAL reason out
            # of ORT's noisy native-side log if the provider falls back.
            _probe_stderr = _io.StringIO()
            with _ctx.redirect_stderr(_probe_stderr):
                _sess = _ort_probe.InferenceSession(_buf,
                    providers=_probe_providers)
            _used = _sess.get_providers()
            if "CUDAExecutionProvider" in _used:
                _USABLE_ONNX_PROVIDERS = [
                    ("CUDAExecutionProvider", _CUDA_PROVIDER_OPTS),
                    "CPUExecutionProvider",
                ]
                print(f"[OK] ONNX Runtime CUDA probe passed (device_id={_CUDA_PROVIDER_OPTS.get('device_id', 'default')}): {_used}")
            else:
                # v9.1.4: try DirectML as a fallback before giving up —
                # works on Windows 11 with any GPU (including Intel UHD /
                # AMD) without a CUDA install.
                if "DmlExecutionProvider" in _avail:
                    try:
                        _sess2 = _ort_probe.InferenceSession(
                            _buf, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
                        if "DmlExecutionProvider" in _sess2.get_providers():
                            _USABLE_ONNX_PROVIDERS = [
                                "DmlExecutionProvider",
                                "CPUExecutionProvider",
                            ]
                            print("[OK] ONNX Runtime using DirectML (works on any Windows 11 GPU)")
                            _skip_cuda_warnings = True
                        else:
                            _skip_cuda_warnings = False
                    except Exception:
                        _skip_cuda_warnings = False
                else:
                    _skip_cuda_warnings = False
                if _skip_cuda_warnings:
                    pass
                else:
                    # CUDA is listed as available but the session fell
                    # back to CPU, which almost always means the Windows
                    # CUDA runtime DLL load failed (error 126: dependency
                    # missing). ORT writes that error to NATIVE stderr
                    # before the Python redirect_stderr has a chance to
                    # grab it — the fallback itself is the signal.
                    print("[--] ONNX Runtime CUDA registered but fell back to CPU")
                    print("     -> Most common cause on Windows: CUDA Toolkit + cuDNN not on PATH.")
                    print("     -> ORT 1.18 GPU needs CUDA 12.x and cuDNN 9.x.")
                    print("     -> Fix option 1 (simplest — no CUDA install, runs on any GPU):")
                    print("          pip uninstall onnxruntime-gpu")
                    print("          pip install onnxruntime-directml")
                    print("     -> Fix option 2 (bundles CUDA via PyTorch wheel):")
                    print("          pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121")
                    print("     -> Fix option 3 (system-wide CUDA Toolkit):")
                    print("          https://developer.nvidia.com/cuda-downloads")
                    print("          https://developer.nvidia.com/cudnn-downloads")
        except Exception as _ce:
            # v9.1.4: print full error, not just class name, so the root
            # cause (missing CUDA DLL, bad opset, etc.) is visible.
            print(f"[--] ONNX Runtime CUDA probe failed ({type(_ce).__name__}): {str(_ce)[:200]}")
            print("     -> using CPU. If you expected GPU, check CUDA_PATH and PATH env vars.")
    elif "DmlExecutionProvider" in _avail:
        # v9.1.5/6: onnxruntime-directml is installed. DirectX 12 route to
        # any Windows 11 GPU — but device_id MUST be passed or DirectML
        # defaults to adapter 0 (integrated GPU on hybrid-graphics laptops).
        _DML_OPTS = getattr(cfg, "DML_PROVIDER_OPTIONS", {"device_id": 1})
        try:
            import io as _io, contextlib as _ctx
            from onnx import helper as _oh, TensorProto as _TP
            _node = _oh.make_node("Identity", inputs=["x"], outputs=["y"])
            _g = _oh.make_graph([_node], "probe",
                [_oh.make_tensor_value_info("x", _TP.FLOAT, [1])],
                [_oh.make_tensor_value_info("y", _TP.FLOAT, [1])])
            _m = _oh.make_model(_g, opset_imports=[_oh.make_opsetid("", 11)])
            _m.ir_version = 7
            _buf = _m.SerializeToString()
            # v9.1.6: walk device_ids (preferred first, then fallbacks) so
            # we pick the discrete GPU even if adapter ordering differs.
            _dml_try_order = [_DML_OPTS.get("device_id", 1), 0, 2]
            _dml_try_order = list(dict.fromkeys(_dml_try_order))  # dedupe, preserve order
            _dml_chosen = None
            for _did in _dml_try_order:
                try:
                    with _ctx.redirect_stderr(_io.StringIO()):
                        _sess = _ort_probe.InferenceSession(
                            _buf,
                            providers=[
                                ("DmlExecutionProvider", {"device_id": _did}),
                                "CPUExecutionProvider",
                            ],
                        )
                    if "DmlExecutionProvider" in _sess.get_providers():
                        _dml_chosen = _did
                        break
                except Exception:
                    continue
            if _dml_chosen is not None:
                _USABLE_ONNX_PROVIDERS = [
                    ("DmlExecutionProvider", {"device_id": _dml_chosen}),
                    "CPUExecutionProvider",
                ]
                # Label the chosen adapter for the startup log so the user
                # can confirm which GPU Task Manager should show activity on.
                _label = "NVIDIA discrete (GPU 1)" if _dml_chosen == 1 else (
                    "primary adapter (GPU 0)" if _dml_chosen == 0 else f"adapter {_dml_chosen}")
                print(f"[OK] ONNX Runtime using DirectML device_id={_dml_chosen} ({_label})")
                if _dml_chosen != 1:
                    print("     NOTE: expected device_id=1 for NVIDIA on hybrid graphics.")
                    print("           If Task Manager shows activity on GPU 0 (integrated)")
                    print("           instead of GPU 1 (NVIDIA), set DML_PROVIDER_OPTIONS")
                    print("           in config.py to {'device_id': 1} and restart.")
            else:
                print("[--] DirectML listed but no device_id worked; using CPU")
        except Exception as _de:
            print(f"[--] DirectML probe failed ({type(_de).__name__}): {str(_de)[:200]}")
            print("     -> using CPU.")
    else:
        print("[--] ONNX Runtime: no GPU provider available in this wheel; using CPU")
        print("     -> Recommended for Windows 11: pip install onnxruntime-directml")
        print("     -> Or with CUDA 12.x installed:  pip install onnxruntime-gpu")
except Exception as _pe:
    print(f"[--] ONNX Runtime not importable ({_pe}); using CPU only")

# Mirror the probe result into the config knob so other modules use it too.
cfg.ONNX_PROVIDERS_PREFERRED = list(_USABLE_ONNX_PROVIDERS)

# ── v8.6: TensorFlow GPU memory growth (DeepFace backend) ──
# Without this, TF allocates the entire 6 GB of VRAM on first DeepFace call,
# leaving nothing for ONNX/CUDA. With memory_growth, both libraries coexist.
# v8.7: on native Windows, TF >= 2.11 is CPU-only — list_physical_devices('GPU')
# returns []. Don't whine about it.
if getattr(cfg, "TF_GPU_MEMORY_GROWTH", True):
    try:
        import tensorflow as _tf
        _gpus = _tf.config.list_physical_devices("GPU")
        for _g in _gpus:
            try:
                _tf.config.experimental.set_memory_growth(_g, True)
            except Exception:
                pass
        if _gpus:
            print(f"[OK] TensorFlow GPU memory growth enabled ({len(_gpus)} GPU)")
        else:
            print("[--] TensorFlow on CPU (TF >= 2.11 has no native-Windows GPU; ONNX still uses CUDA if available)")
    except Exception:
        pass

# ══════════════════════════════════════════════
#  MediaPipe Face Landmarker (cached singleton)
# ══════════════════════════════════════════════
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# v9.2 (plan steps 1-4): high-throughput landmarker pipeline.
#
# Step 1 — GPU delegate: try BaseOptions.Delegate.GPU first; fall back to
#          CPU if the host MediaPipe build doesn't include OpenGL/Metal
#          backends (common on stock Windows pip wheels).
# Step 2 — VIDEO running mode: pass monotonic timestamps to
#          detect_for_video() so the landmarker can use temporal tracking
#          between frames. Removes the per-frame warm-up cost IMAGE mode pays.
# Step 3 — Adaptive num_faces: two landmarkers (single + multi). Start
#          with multi; after 30 frames of seeing exactly one face, switch
#          to single (1-face detection is ~3× faster). Any frame seeing
#          zero faces or >1 triggers a retry at multi.
# Step 4 — Input downscale: landmarks are normalised [0,1] so resolution
#          doesn't affect the output. Downscaling the input to ~320px
#          short-side cuts landmark-detection cost roughly 3× with no
#          perceivable accuracy loss for face mesh. The full-resolution
#          frame is still available to rPPG + DeepFace which need it.
_LM_TARGET_SHORT = 320           # Step 4 downscaled resolution
_LM_SWITCH_STREAK = 30           # Step 3 frames of single-face observation before downshifting
# v9.7.2: periodic multi-face probe while in single mode. Without this the
# adaptive switcher gets stuck in single mode forever — the single-face
# landmarker returns at most 1 face, so n_faces stays at 1, and the
# "switch back to multi" branch (triggered by n_faces != 1) never fires
# even when a 2nd face enters the frame. Every _LM_PROBE_INTERVAL frames
# we run the multi-face landmarker once to count actual faces; if >1, we
# re-engage multi mode. ~2 s at 30 fps; negligible overhead.
_LM_PROBE_INTERVAL = int(getattr(cfg, "LM_PROBE_INTERVAL", 60))
_LM_MAX_FACES = int(getattr(cfg, "MAX_FACES", 5))

_lm_multi = None
_lm_single = None
_lm_active_mode = "multi"        # "multi" or "single"
_lm_single_face_streak = 0
_lm_gpu_ok = False
_lm_timestamp_ms = 0             # monotonic counter for VIDEO mode
_lm_probe_counter = 0            # v9.7.2 periodic multi-face probe counter

def _try_make_landmarker(num_faces, delegate=None):
    """Build a FaceLandmarker with the requested delegate, fall back to CPU
    if GPU delegate isn't compiled in this MediaPipe build."""
    # VIDEO mode — receives monotonic timestamps, enables temporal tracking.
    opts_kwargs = dict(
        base_options=BaseOptions(model_asset_path=cfg.MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=num_faces,
    )
    if delegate is not None:
        try:
            opts_kwargs["base_options"] = BaseOptions(
                model_asset_path=cfg.MODEL_PATH,
                delegate=delegate,
            )
        except Exception:
            pass
    return FaceLandmarker.create_from_options(FaceLandmarkerOptions(**opts_kwargs))

def _build_landmarkers():
    """Warm-build both single- and multi-face landmarkers with the best
    available delegate (GPU if MediaPipe has it, otherwise CPU)."""
    global _lm_multi, _lm_single, _lm_gpu_ok
    chosen_delegate = None
    try:
        # Step 1 — try GPU delegate if MediaPipe exposes it
        GPU = getattr(BaseOptions.Delegate, "GPU", None)
        if GPU is not None:
            try:
                _probe = _try_make_landmarker(1, delegate=GPU)
                _probe.close()
                chosen_delegate = GPU
                _lm_gpu_ok = True
                print("[OK] MediaPipe FaceLandmarker using GPU delegate")
            except Exception as e:
                print(f"[--] MediaPipe GPU delegate not usable ({type(e).__name__}); using CPU")
    except Exception:
        pass
    if not _lm_gpu_ok:
        chosen_delegate = getattr(BaseOptions.Delegate, "CPU", None)
        print("[OK] MediaPipe FaceLandmarker using CPU delegate")
    _lm_multi = _try_make_landmarker(_LM_MAX_FACES, delegate=chosen_delegate)
    if _LM_MAX_FACES > 1:
        _lm_single = _try_make_landmarker(1, delegate=chosen_delegate)
    return _lm_multi

def get_landmarker():
    """Legacy accessor — returns the currently-active landmarker."""
    if _lm_multi is None:
        _build_landmarkers()
    if _lm_active_mode == "single" and _lm_single is not None:
        return _lm_single
    return _lm_multi

def detect_faces(rgb_image):
    """Run MediaPipe on an RGB numpy array. Returns result object.

    v9.2: auto-downscales to _LM_TARGET_SHORT on the short side before
    inference, auto-switches between single/multi landmarker based on
    the last 30 frames, and uses VIDEO-mode detect_for_video so the
    landmarker can exploit temporal continuity."""
    global _lm_active_mode, _lm_single_face_streak, _lm_timestamp_ms, _lm_probe_counter
    if _lm_multi is None:
        _build_landmarkers()

    # Step 4 — downscale for landmark inference (preserve aspect ratio)
    h, w = rgb_image.shape[:2]
    short = min(h, w)
    if short > _LM_TARGET_SHORT:
        scale = _LM_TARGET_SHORT / short
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        # INTER_AREA is the correct choice for downscaling
        inf_img = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        inf_img = rgb_image

    # Step 2 — VIDEO mode needs a monotonic timestamp in milliseconds
    _lm_timestamp_ms += 1  # simple monotonic tick; MediaPipe only cares about strict ordering
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=inf_img)

    # v9.7.2: periodic multi-face probe. While in single mode, every
    # _LM_PROBE_INTERVAL frames we use the multi landmarker for one frame
    # to detect any newly-entered second face. This breaks the
    # single-mode lock-in described above _LM_PROBE_INTERVAL.
    _use_multi_probe = False
    if _lm_active_mode == "single" and _lm_single is not None:
        _lm_probe_counter += 1
        if _lm_probe_counter >= _LM_PROBE_INTERVAL:
            _lm_probe_counter = 0
            _use_multi_probe = True

    if _use_multi_probe:
        lm = _lm_multi
    else:
        lm = _lm_single if (_lm_active_mode == "single" and _lm_single is not None) else _lm_multi
    try:
        result = lm.detect_for_video(mp_image, _lm_timestamp_ms)
    except Exception:
        # Fallback: IMAGE-mode landmarker might have been built — recreate
        _build_landmarkers()
        lm = get_landmarker()
        result = lm.detect_for_video(mp_image, _lm_timestamp_ms)

    # Step 3 — adaptive num_faces feedback
    n_faces = len(getattr(result, "face_landmarks", []) or [])
    if _lm_active_mode == "multi":
        if n_faces == 1:
            _lm_single_face_streak += 1
            if _lm_single_face_streak >= _LM_SWITCH_STREAK and _lm_single is not None:
                _lm_active_mode = "single"
                _lm_single_face_streak = 0
        else:
            _lm_single_face_streak = 0
    else:  # single mode
        if n_faces != 1:
            _lm_active_mode = "multi"
            _lm_single_face_streak = 0

    return result

# ══════════════════════════════════════════════
#  HSEmotion ONNX (fast emotion recognition)
# ══════════════════════════════════════════════
HSEMOTION_ENABLED = False
_fer = None


def _maybe_quantize_onnx_fp16(src_path):
    """v8.7 (Step 6): if the FP32 HSEmotion ONNX exists and a sibling .fp16.onnx
    does not, build a half-precision copy on first run. CUDA ORT runs FP16
    ~1.6x faster on Ampere/Ada and the emotion model accuracy delta is <0.5%.

    v8.7 BUGFIX: only attempt this if `onnxconverter_common` is importable
    AND we actually have CUDA usable — there's no point quantising for CPU
    (CPU FP16 is *slower* than FP32 in ORT). Print a single, friendly hint
    instead of an "[--] skipped" line that looks alarming."""
    if not src_path or not os.path.exists(src_path):
        return src_path
    if src_path.endswith(".fp16.onnx"):
        return src_path
    fp16_path = src_path.replace(".onnx", ".fp16.onnx")
    if os.path.exists(fp16_path):
        return fp16_path
    # Skip quietly if CUDA isn't usable — FP16 only pays off on CUDA.
    if "CUDAExecutionProvider" not in _USABLE_ONNX_PROVIDERS:
        return src_path
    try:
        from onnxconverter_common import float16
        import onnx as _onnx
        model = _onnx.load(src_path)
        # keep_io_types=True so the FP32 input/output API stays unchanged
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        _onnx.save(model_fp16, fp16_path)
        print(f"[OK] HSEmotion FP16 model written -> {os.path.basename(fp16_path)}")
        return fp16_path
    except ImportError:
        # Friendly one-liner — opt-in for FP16 perf boost, completely optional.
        print("[INFO] HSEmotion FP16 not enabled (optional: pip install onnxconverter_common onnx)")
        return src_path
    except Exception as e:
        print(f"[--] FP16 quantization skipped ({type(e).__name__}: {e})")
        return src_path


try:
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    # v8.7: use the CUDA-probed provider list. We never hand a provider to
    # ORT that hasn't passed the boot probe, so no more "LoadLibrary failed"
    # noise on every recognizer construction.
    _preferred = list(_USABLE_ONNX_PROVIDERS)
    try:
        _fer = HSEmotionRecognizer(model_name=cfg.HSEMOTION_MODEL,
                                   providers=_preferred)
    except TypeError:
        # Older hsemotion-onnx versions don't accept `providers` kwarg
        _fer = HSEmotionRecognizer(model_name=cfg.HSEMOTION_MODEL)
        try:
            # Force the underlying ORT session onto our chosen providers
            _sess = getattr(_fer, "ort_session", None) or getattr(_fer, "session", None)
            if _sess is not None and hasattr(_sess, "set_providers"):
                _sess.set_providers(_preferred)
        except Exception:
            pass

    # v8.7 (Step 6): hot-swap the underlying ORT session to a FP16 one if we
    # can build it. The recognizer object's preprocessing layer is unchanged.
    # v8.8: _preferred may contain (provider, opts) tuples; check name only
    _has_cuda = any(
        (p == "CUDAExecutionProvider" if isinstance(p, str) else p[0] == "CUDAExecutionProvider")
        for p in _preferred
    )
    if getattr(cfg, "HSEMOTION_FP16", True) and _has_cuda:
        try:
            import onnxruntime as _ort
            _sess = getattr(_fer, "ort_session", None) or getattr(_fer, "session", None)
            if _sess is not None:
                model_path = None
                # Best-effort: grab the original model path from the session's input model
                try:
                    model_path = _sess._model_path  # private but works on most ORT builds
                except Exception:
                    model_path = None
                if not model_path:
                    # Fallback: try the package-level path the recognizer uses
                    import hsemotion_onnx, glob as _glob
                    pkg_dir = os.path.dirname(hsemotion_onnx.__file__)
                    candidates = _glob.glob(os.path.join(pkg_dir, "models", "*.onnx"))
                    candidates = [c for c in candidates if cfg.HSEMOTION_MODEL in c and ".fp16" not in c]
                    model_path = candidates[0] if candidates else None
                if model_path:
                    fp16_path = _maybe_quantize_onnx_fp16(model_path)
                    if fp16_path and fp16_path != model_path and os.path.exists(fp16_path):
                        new_sess = _ort.InferenceSession(fp16_path, providers=_preferred)
                        # Swap into the recognizer if attribute exists
                        if hasattr(_fer, "ort_session"):
                            _fer.ort_session = new_sess
                        elif hasattr(_fer, "session"):
                            _fer.session = new_sess
                        print("[OK] HSEmotion ONNX session swapped to FP16")
        except Exception as _e:
            print(f"[--] FP16 swap failed ({_e}); staying on FP32")

    HSEMOTION_ENABLED = True
    try:
        _used = (_fer.ort_session.get_providers()
                 if hasattr(_fer, "ort_session")
                 else getattr(_fer, "session", None).get_providers())
        print(f"[OK] HSEmotion ONNX  providers={_used}")
    except Exception:
        print("[OK] HSEmotion ONNX")
except Exception:
    print("[--] HSEmotion unavail (pip install hsemotion-onnx)")

# Map HSEmotion 8-class to DeepFace-compatible 7-class
_HSEMO_MAP = {
    "Anger": "angry", "Contempt": "neutral", "Disgust": "disgust",
    "Fear": "fear", "Happiness": "happy", "Neutral": "neutral",
    "Sadness": "sad", "Surprise": "surprise",
}
_HSEMO_LABELS_8 = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
_HSEMO_LABELS_7 = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

# v9.10: per-pid EMA smoothing on the 7- or 8-class HSEmotion score
# vector. α=0.20 -> ~5-frame effective window @ 30 FPS, kills the
# per-frame argmax flicker (HAPPY -> NEUTRAL -> HAPPY oscillation
# users see in v9.4-v9.9) without lagging more than ~150 ms behind
# the actual emotion. Keyed by integer pid so multi-face scenes
# don't pollute each other.
_HSEMO_EMA_ALPHA = 0.20
_hsemo_ema_state: "Dict[int, np.ndarray]" = {}


def _hsemo_smooth(pid: int, scores: np.ndarray) -> np.ndarray:
    """EMA-smooth the per-class score vector for one face. Returns the
    smoothed scores; updates the per-pid state in place."""
    prev = _hsemo_ema_state.get(int(pid))
    if prev is None or prev.shape != scores.shape:
        _hsemo_ema_state[int(pid)] = scores.astype(np.float32, copy=True)
        return scores
    smoothed = _HSEMO_EMA_ALPHA * scores + (1.0 - _HSEMO_EMA_ALPHA) * prev
    _hsemo_ema_state[int(pid)] = smoothed
    return smoothed


def reset_hsemo_smoothing(pid: int = None) -> None:
    """Drop the EMA state for one pid (face left frame) or all pids
    (e.g. session reset). Public hook so main.py can call it on
    track-id changes."""
    if pid is None:
        _hsemo_ema_state.clear()
    else:
        _hsemo_ema_state.pop(int(pid), None)


def run_hsemotion(crop, pid: int = 0):
    """Run HSEmotion on a BGR face crop. Returns
    {'scores': {...}, 'dominant': str, 'logits': np.ndarray} or None.

    v9.10 changes:
      - removed the eager `cv2.resize(rgb, (224, 224))` that was
        dropping resolution for any model with a different native
        input size (enet_b2_* needs 260x260; the library does its
        own resize, so our pre-resize was just throwing pixels away)
      - added per-pid EMA smoothing via `_hsemo_smooth` with α=0.20
        to kill per-frame argmax flicker
      - auto-detects 7-class vs 8-class label set from cfg.HSEMOTION_MODEL
        so the swap from enet_b0_8_* to enet_b2_7 works without code
        changes elsewhere
    """
    if not HSEMOTION_ENABLED or _fer is None:
        return None
    try:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # No pre-resize -- let HSEmotion's preprocess use its model's
        # native input size (224 for B0, 260 for B2). This gives the
        # B2 model the resolution it was trained for.
        with _hsemo_lock:
            emotion, scores = _fer.predict_emotions(rgb, logits=False)
        # 7-class vs 8-class: pick the right label list from len(scores)
        scores_arr = np.asarray(scores, dtype=np.float32)
        labels = _HSEMO_LABELS_7 if len(scores_arr) == 7 else _HSEMO_LABELS_8
        # EMA smooth before argmax to kill flicker
        smoothed = _hsemo_smooth(pid, scores_arr)
        smoothed_dominant = labels[int(np.argmax(smoothed))]
        mapped = _HSEMO_MAP.get(smoothed_dominant, "neutral")
        score_dict = {}
        for i, lab in enumerate(labels):
            k = _HSEMO_MAP.get(lab, "neutral")
            score_dict[k] = score_dict.get(k, 0) + round(float(smoothed[i]), 3)
        return {
            "scores": score_dict,
            "dominant": mapped,
            "logits": smoothed.tolist(),  # downstream debug / paper figs
            "raw_dominant": _HSEMO_MAP.get(emotion, "neutral"),  # pre-EMA
        }
    except Exception:
        return None

# ══════════════════════════════════════════════
#  DeepFace (age, gender, race — emotion only if HSEmotion unavailable)
# ══════════════════════════════════════════════
from deepface import DeepFace

def run_deepface(crop):
    """Run DeepFace on a BGR face crop with improved preprocessing.
    Larger input, histogram equalization, and better normalization for
    more accurate age/race/gender estimation."""
    try:
        if crop is None or crop.shape[0] < 50 or crop.shape[1] < 50:
            return None

        # Preprocessing: normalize lighting (CLAHE on L channel)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        resized = cv2.resize(enhanced, (cfg.DEEPFACE_RESIZE, cfg.DEEPFACE_RESIZE))

        if HSEMOTION_ENABLED and cfg.DEEPFACE_SKIP_EMOTION_IF_HSEMOTION:
            actions = ["age", "gender", "race"]
        else:
            actions = ["emotion", "age", "gender", "race"]

        r = DeepFace.analyze(
            img_path=resized, actions=actions,
            enforce_detection=False, silent=True, detector_backend="skip",
        )
        r = r[0] if isinstance(r, list) else r

        gender_raw = r.get("gender", {})
        gd = max(gender_raw, key=gender_raw.get, default="?") if gender_raw else "?"
        gc = round(gender_raw.get(gd, 0) / 100, 3) if gender_raw else 0
        race_raw = r.get("race", {})
        rd = r.get("dominant_race", "?")
        rc = round(race_raw.get(rd, 0) / 100, 3) if race_raw else 0

        result = {
            "raw_age": int(r.get("age", 0)),
            "gender": {
                "dominant": gd, "confidence": gc,
                "scores": {k: round(v / 100, 3) for k, v in gender_raw.items()},
            },
            "race": {
                "dominant": rd, "confidence": rc,
                "scores": {k: round(v / 100, 3) for k, v in race_raw.items()},
            },
        }

        # Emotion: HSEmotion if available, else DeepFace
        if HSEMOTION_ENABLED:
            hse = run_hsemotion(crop)
            result["emotion"] = hse if hse else {"scores": {}, "dominant": "neutral"}
        else:
            emo_raw = r.get("emotion", {})
            result["emotion"] = {
                "scores": {k: round(v / 100, 3) for k, v in emo_raw.items()},
                "dominant": r.get("dominant_emotion", "?"),
            }
        return result
    except Exception as e:
        # 2026-04-25 (v9.7.1): surface DeepFace failures instead of silently
        # returning None. Without this, _df_cache stays empty forever and
        # the AGE/GENDER/ETHNICITY tiles render '?'. Rate-limited via the
        # caller (_bg_deepface only submits when bbox shifted + interval
        # elapsed) so log volume stays bounded.
        try:
            print(f"[deepface] analyze failed: {type(e).__name__}: {e}")
        except Exception:
            pass
        return None

def get_face_crop(image, landmarks, h, w):
    """Extract face crop with generous padding for better demographic analysis.
    Wider padding captures forehead/hair which improves age and ethnicity accuracy."""
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    face_w = max(xs) - min(xs)
    face_h = max(ys) - min(ys)
    # Padding proportional to face size (50% of face dimension)
    pad_x = int(face_w * 0.5)
    pad_y = int(face_h * 0.5)
    x1 = max(0, int(min(xs)) - pad_x)
    y1 = max(0, int(min(ys)) - pad_y)
    x2 = min(w, int(max(xs)) + pad_x)
    y2 = min(h, int(max(ys)) + pad_y)
    crop = image[y1:y2, x1:x2]
    return crop if crop is not None and crop.size > 0 else None

# ══════════════════════════════════════════════
#  Optional modules status
# ══════════════════════════════════════════════
# v8.7: providers list is now the *probed* list (what actually works), not
# what onnxruntime claims it has. Avoids misleading "[OK] CUDAExecutionProvider"
# lines on systems where the cuDNN DLL isn't installed.
ONNX_PROVIDERS = list(_USABLE_ONNX_PROVIDERS)
print(f"[OK] ONNX Runtime active providers: {ONNX_PROVIDERS}")

# PyTorch: only check, don't worry about CUDA —- ONNX Runtime handles GPU
try:
    import torch
    if torch.cuda.is_available():
        print(f"[OK] PyTorch CUDA: {torch.cuda.get_device_name(0)}")
        # v8.6: hint torch to allow tf32 + cudnn benchmark for fastest convs
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
    else:
        # This is NORMAL if you installed torch without CUDA.
        # ONNX Runtime uses its own CUDAExecutionProvider independently.
        if "CUDAExecutionProvider" in ONNX_PROVIDERS:
            print("[OK] PyTorch CPU-only, but ONNX Runtime has CUDA (this is fine)")
        else:
            print("[--] PyTorch CPU-only, ONNX CPU-only")
except Exception:
    print("[--] PyTorch not installed")


# ══════════════════════════════════════════════
#  v8.6 — Pre-warm models so the first real frame is fast
# ══════════════════════════════════════════════
def warmup_models():
    """Run a tiny synthetic frame through every model to JIT-compile kernels.
    Without this, the first user-facing inference pays a 200-800ms cost that
    would otherwise miss the 1Hz deadline on the first analysis cycle.
    Safe to call multiple times — short-circuits after first run."""
    if getattr(warmup_models, "_done", False):
        return
    try:
        dummy = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        rgb = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
        # MediaPipe
        try: detect_faces(rgb)
        except Exception: pass
        # HSEmotion
        if HSEMOTION_ENABLED and _fer is not None:
            try: run_hsemotion(dummy)
            except Exception: pass
        # DeepFace
        try: run_deepface(dummy)
        except Exception: pass
        # v9.5.x — warm PAD ONNX / LightGBM sessions. Without this, the
        # first user-facing PAD score paid 2-9 s of DirectML JIT-compile
        # latency, which read as "3 second lag" in the UI on the first
        # real face. Branches silently no-op when their weights are absent.
        _pad_dummy_160 = (np.random.rand(160, 160, 3) * 255).astype(np.uint8)
        _pad_dummy_128 = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        _pad_dummy_224 = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        try:
            from pad.branches.patch_depth_branch import score as _p
            _p(_pad_dummy_160)
        except Exception: pass
        try:
            from pad.branches.moire_branch import score as _m
            _m(_pad_dummy_128)
        except Exception: pass
        try:
            from pad.branches.deepfake_branch import score as _d
            _d(0, face_crop_bgr=_pad_dummy_224)
        except Exception: pass
        warmup_models._done = True
        print("[OK] Models warmed up (MediaPipe + HSEmotion + DeepFace + PAD)")
    except Exception as _e:
        print(f"[WARN] Warmup failed (non-fatal): {_e}")


# Auto-warm on import if the config flag is set. This adds ~1-3s to startup
# but eliminates the first-frame stutter that otherwise hits the 1Hz pipeline.
if getattr(cfg, "WARMUP_MODELS_ON_START", False):
    try:
        warmup_models()
    except Exception:
        pass
