"""
DODO VISION AI v5 — Configuration
Together AI (full prompt) + Groq (compressed fallback). Keys from .env file.
"""
import os

# ── Load .env file (no dependency needed) ──
def _load_env(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_env()

# ── API Keys (from .env, never hardcoded) ──
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")

# ── Server ──
HOST = "127.0.0.1"
PORT = 8000

# ── MediaPipe ──
MODEL_PATH = "face_landmarker_v2_with_blendshapes.task"
MAX_FACES = 5

# ── DeepFace / HSEmotion ──
DEEPFACE_RESIZE = 320  # larger input = better age/race accuracy (was 224)
DEEPFACE_SKIP_EMOTION_IF_HSEMOTION = True
# v9.10 (2026-05-08): swap from enet_b0_8_best_afew to enet_b2_7.
#   - "afew" was fine-tuned on AFEW (acted facial expressions) -- great
#     for theatrical clips, mediocre for subtle webcam expressions.
#   - "_b2_7" is the B2 EfficientNet backbone (260x260 input, ~9 MB ONNX
#     vs B0's 5 MB) trained directly on AffectNet, 7 classes.
#   - 7-class drops "Contempt" which has ~30% inter-rater disagreement
#     on AffectNet; Ekman's 7-emotion taxonomy is the 2024-25 norm.
#   - Expected: +2-4% top-1 on still images, less per-frame flicker.
# To revert: HSEMOTION_MODEL = "enet_b0_8_best_afew"   # v9.4 original
# v9.11 ablation (2026-05-10): tested enet_b2_8 (8-class, adds Contempt).
# Result: NET WORSE on FER2013 (-2.9%) and NHFI (-1.7%); slight WIN on
# JAFFE (+0.9%) and EMOTIC v2 (+0.8%). The 8-class model "steals"
# Anger/Neutral predictions for Contempt, costing ~3% on those classes.
# 8-class wins only when the dataset HAS Contempt-labeled samples (NHFI 0->0.31).
# Reports of both 7-class and 8-class are on disk in reports/ for paper
# Tier-3 ablation. Production stays on 7-class because the live PAD overlay
# needs stable predictions; 8-class adds per-frame flicker between Anger/
# Contempt/Neutral that hurts UX for marginal accuracy gain.
HSEMOTION_MODEL = "enet_b2_7"

# ── rPPG ──
# v9.2 (plan step 13): shrink rPPG buffer.
# 1024 samples is a power-of-two (cheaper rfft), covers ~34s at 30 fps or
# ~8.5s at 120 fps — still generous headroom for heart-rate lock-in
# (needs ~10s) and respiration (needs ~30s) without paying the FFT cost
# of the old 1200-sample buffer. Confirmed-BPM lock-in at 60s still
# works because hr_samples retains 300 samples independently.
# v9.8: env-overridable. Higher = more frequency resolution but slower
# convergence; lower = faster lock but more BPM noise. 1024 (~17 s @
# 60 fps, ~34 s @ 30 fps) is the sweet spot. Override via
# `set DODO_RPPG_BUFFER_SIZE=2048` for paper-figure-quality FFT bins.
try:
    RPPG_BUFFER_SIZE = int(os.environ.get("DODO_RPPG_BUFFER_SIZE", "1024"))
except (TypeError, ValueError):
    RPPG_BUFFER_SIZE = 1024
RPPG_BUFFER_SIZE = max(256, min(8192, RPPG_BUFFER_SIZE))  # safety clamp
RPPG_MIN_SAMPLES = 180    # ~1.5 seconds at 120fps
RPPG_BANDPASS_LO = 0.75
RPPG_BANDPASS_HI = 3.5
RPPG_CONFIRM_AFTER_SECS = 45  # confirm faster with 120fps data (was 60)
RPPG_MOTION_THRESH = 0.020   # landmark displacement threshold for artifact rejection (normalized units; 0.020 ≈ 26 px on a 1280-wide frame — loose enough for normal sitting, tight enough to drop real head-turns)
RPPG_HR_SNR_GATE = 2.0       # peak/background power ratio needed to publish HR (was implicit 2.5; 2.0 trades a little noise for faster lock in real-world lighting)
# 2026-04-25: Resp-rate SNR gate. Was implicit 2.0 in rppg.py; bumped to
# 3.5 because 2.0 was admitting motion / DC-drift artefacts that produce
# resp_rate = 6 br/min (band low edge) or 30 (band high edge) regardless
# of actual breathing. With 3.5, resp shows "--" until a clean
# physiological respiration peak is locked, then publishes a stable value.
RPPG_RESP_SNR_GATE = 3.5

# ── EMA smoothing ──
EMA_HR = 0.2
EMA_VALENCE = 0.25    # faster response to emotion changes (was 0.15)
EMA_AROUSAL = 0.25    # faster response to arousal changes (was 0.15)
EMA_STRESS = 0.12

# ── Emotion fusion ──
DEEPFACE_WEIGHT = 0.55  # 55% CNN + 45% blendshapes (was 65/35, gives blendshapes more real-time influence)

# ── PERCLOS ──
PERCLOS_WINDOW = 900
PERCLOS_DROWSY_THRESH = 0.40
PERCLOS_FATIGUE_THRESH = 0.25
EYE_CLOSED_THRESH = 0.5

# ── Micro-expression ──
MICRO_VELOCITY_THRESH = 0.3
MICRO_WINDOW = 5

# ── Blink / Age ──
BLINK_HISTORY_SIZE = 600
AGE_SMOOTH_WINDOW = 90  # frames (more = more stable age, was 60)

# ── Auto monitor ──
AUTO_MONITOR_INTERVAL = 30

# ── Parquet ──
PARQUET_LOG_DIR = "logs"
PARQUET_FLUSH_INTERVAL = 30
os.makedirs(PARQUET_LOG_DIR, exist_ok=True)

# ── mmWave Radar (Seeed MR60BHA2) ──
# v9.7.5: env-overridable to keep all sensor port pins in start_inst.cmd.
RADAR_ENABLED = (os.environ.get("DODO_RADAR_ENABLED", "1") not in ("0", "false", "False"))
RADAR_PORT    = os.environ.get("DODO_RADAR_PORT",   "COM5")
RADAR_B_PORT  = os.environ.get("DODO_RADAR_B_PORT", "COM6")
try:
    RADAR_BAUD = int(os.environ.get("DODO_RADAR_BAUD", "115200"))
except Exception:
    RADAR_BAUD = 115200
RADAR_MODE = "json"          # matches the new radar_sketch.ino (v7.1)
RADAR_DEBUG = True           # print first 20 raw lines on connect
RADAR_STALE_SECS = 6.0       # suppress readings older than this

# ── Face Recognition ──
FACES_DIR = "faces"
FACE_RECOGNITION_THRESHOLD = 0.42
os.makedirs(FACES_DIR, exist_ok=True)

# ── Dodo LLM ──
DODO_PROMPT_FILE = "dodo_prompt.md"
DODO_PROMPT_FULL_FILE = "dodo_prompt_FULL.md"
# "together" = Together AI (full prompt, ~$0.017/req)
# "groq" = Groq free (compressed prompt, $0)
# "ollama" = local
#
# v9.7.6: when LLM_BACKEND is EXPLICITLY pinned via env (launcher does
# `set LLM_BACKEND=together`), honor it strictly. The pinned flag turns
# off auto-fallbacks in llm.py — without it, Together failures silently
# routed to Groq and polluted QLoRA training pairs with weaker outputs.
# When env is unset we keep the legacy auto-derive from key presence.
_LLM_BACKEND_ENV = os.environ.get("LLM_BACKEND", "").strip().lower()
# API_up v9.8: `vllm_local` joins the backend roster. Talks to a local
# OpenAI-compatible endpoint (vLLM, llama.cpp, lmstudio, tgi) at cfg.VLLM_URL.
# Useful for fully-offline deception narrative without external API deps.
if _LLM_BACKEND_ENV in ("together", "groq", "ollama", "vllm_local"):
    LLM_BACKEND = _LLM_BACKEND_ENV
    LLM_BACKEND_PINNED = True
else:
    LLM_BACKEND = "together" if os.environ.get("TOGETHER_API_KEY") else "groq"
    LLM_BACKEND_PINNED = False

# ── LLM Rolling Context ──
LLM_CONTEXT_WINDOW = 3  # number of previous analyses to include in context

# ── Whisper STT ──
WHISPER_MODEL = "small"  # tiny/base/small/medium (small = best balance for 6GB VRAM)

# ── GSR Sensor (Grove GSR v1.2 via Arduino) ──
# v9.7.5: env-overridable. `auto` scans for any Arduino-like serial
# device that isn't already claimed by RADAR_PORT or OXIMETER_PORT.
# When auto-detect fails (rare — happens if CH340 driver name shifts),
# pin via DODO_GSR_PORT=COM7 in start_inst.cmd.
GSR_ENABLED = (os.environ.get("DODO_GSR_ENABLED", "1") not in ("0", "false", "False"))
GSR_PORT    = os.environ.get("DODO_GSR_PORT", "auto")
try:
    GSR_BAUD = int(os.environ.get("DODO_GSR_BAUD", "9600"))
except Exception:
    GSR_BAUD = 9600

# ── Voice Monitor ──
VOICE_CHUNK_SECONDS = 3
VOICE_SAMPLE_RATE = 16000

# ── Dual Camera ──
DUAL_CAMERA_ENABLED = True  # auto-detect 120fps camera for fast biometrics

# ── Ollama (local fallback) ──
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_NUM_GPU = 99
OLLAMA_NUM_CTX = 32768
OLLAMA_TEMPERATURE = 0.6

# ── API_up v9.8: vLLM / OpenAI-compatible local server ──
# Default targets meta-llama/Meta-Llama-3.1-8B-Instruct on 127.0.0.1:8001.
# Start the server via `python serve_llm_local.py` (helper prints exact
# vLLM and Ollama commands sized for the 5090 Laptop's 24 GB VRAM).
VLLM_URL          = os.environ.get("VLLM_URL", "http://127.0.0.1:8001")
VLLM_MODEL        = os.environ.get("VLLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
VLLM_NUM_CTX      = int(os.environ.get("VLLM_NUM_CTX", "8192"))
VLLM_TEMPERATURE  = float(os.environ.get("VLLM_TEMPERATURE", "0.6"))
VLLM_TIMEOUT_S    = int(os.environ.get("VLLM_TIMEOUT_S", "60"))
VLLM_MAX_TOKENS   = int(os.environ.get("VLLM_MAX_TOKENS", "200"))

# ── Oximeter ──
OXIMETER_ENABLED = True
OXIMETER_PORT = "auto"          # v8.5: HID primary (VID=0x28E9 auto-detected), serial fallback
OXIMETER_BAUD = 19200
# HID VID/PID for Pulox PO-300 (GigaDevice 0x28E9:0x028A) — auto-detected, no manual config needed

# v8.9 Step 10: I2C baud rate for direct MAX30102 connections.
# Standard 100kHz causes FIFO overruns; 400kHz (Fast Mode) reduces clock
# period from 10us to 2.5us, keeping pace with the 32-sample internal FIFO.
OXIMETER_I2C_BAUD = 400000  # Hz (400kHz Fast Mode)

# v8.9 Step 11: USB UART bridge for microcontroller-buffered oximeter.
# When a dedicated MCU (Arduino/ESP32) handles I2C polling and streams
# aggregated packets over USB UART, use this higher baud rate.
OXIMETER_UART_BRIDGE_BAUD = 460800  # bps (recommended for MCU bridge)
OXIMETER_UART_BRIDGE_ENABLED = False  # set True when MCU bridge is connected

# ── WiFi CSI (v8.3 stub — planned v8.4) ──
WIFI_CSI_ENABLED = False

# ══════════════════════════════════════════════
#  v9.0: Fault-Tolerant Multimodal Sensor Fusion (10-Step Plan)
# ══════════════════════════════════════════════

# Step 2: Zero-Order Hold mediator — 50 Hz rigid tick rate for the sensor
# fusion mediator.  If any sensor misses its 20 ms window, ZOH extrapolates
# the last known value while decaying confidence exponentially.
ZOH_MEDIATOR_HZ = 50            # mediator tick rate (50 Hz = 20 ms window)
ZOH_AGE_DECAY_RATE = 0.5        # confidence halves every ~1.4 s after staleness

# Step 3: WSL2 containerization — auto-detect WSL2 runtime so gpu_enforce
# can switch from registry-based GPU pinning to NVIDIA Container Toolkit.
WSL2_CONTAINER_MODE = False     # set True when running inside WSL2/Docker
WSL2_GPU_MEM_LIMIT_GB = 4      # VRAM cap for nvidia-container-toolkit

# Step 4: PRISM rPPG — Projection-based Robust and Interpretable Signal
# Mixing (CMU) for remote photoplethysmography.  Offloads heavy matrix ops
# to CUDA via CuPy when available.
PRISM_ENABLED = True            # enable PRISM adaptive color-mixing
PRISM_WINDOW_S = 1.6            # temporal window for PRISM projection (seconds)
PRISM_GPU_OFFLOAD = True        # offload PRISM matrix ops to CUDA

# Step 5: Time-of-Flight seat validation for oximeter.
# When a ToF / proximity sensor is paired with the MAX30102, halt I2C
# polling if the finger is improperly seated (distance > threshold mm).
TOF_SEAT_ENABLED = True
TOF_SEAT_THRESHOLD_MM = 5.0    # max finger-to-sensor gap for valid reading
TOF_HALT_ON_UNSEATED = True    # stop I2C polling when finger is absent

# Step 6: Hardware timestamping — the MCU stamps each sample with
# millisecond-precision timestamps BEFORE UART transmission so Python
# never introduces OS-level network jitter into HRV analysis.
HW_TIMESTAMP_ENABLED = True
HW_TIMESTAMP_DRIFT_WARN_MS = 50  # warn if MCU-vs-host drift exceeds this

# Step 9: Lempel-Ziv Complexity (LZC) photic habituation monitor.
# Continuously analyze EEG complexity during visual stimulation at
# 27.3-28.9 Hz.  If LZC detects excessive habituation (distress),
# immediately suppress the visual stimulus.
LZC_MONITOR_ENABLED = True
LZC_WINDOW_S = 10.0            # LZC analysis window (seconds)
LZC_DISTRESS_THRESHOLD = 0.3   # LZC below this → suppress stimulus
LZC_VISUAL_HZ_LOW = 27.3       # visual entrainment band (lower bound)
LZC_VISUAL_HZ_HIGH = 28.9      # visual entrainment band (upper bound)
WCAG_231_SAFE_HZ = 3.0         # WCAG 2.3.1 max flash rate (Hz)

# ── Calibration / Training ──
CALIBRATION_DIR = "calibration"
os.makedirs(CALIBRATION_DIR, exist_ok=True)

# ── LSL ──
os.environ["PYLSL_DISABLE_IPV6"] = "1"

# ── Suppress TF noise ──
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# v9.1.3: CUDA device selection corrected.
#
# The v8.5 comment below conflated Windows Task Manager's "GPU 0 / GPU 1"
# (DirectX adapter order) with CUDA's device enumeration — they are NOT
# the same. CUDA only sees NVIDIA devices; the Intel/AMD integrated GPU
# is invisible to CUDA regardless of Task Manager ordering. Setting
# CUDA_VISIBLE_DEVICES=1 on a laptop with one NVIDIA dGPU therefore
# *hides* the only CUDA-capable device and forces CPU fallback, which
# is exactly what was causing the "GPU 0 / integrated" symptom: with
# CUDA disabled, every framework (ONNX, TF, PyTorch) runs on CPU and
# Windows attributes the matching CPU→iGPU compositing work to "GPU 0".
#
# Correct setup:
#   CUDA_DEVICE_ORDER=PCI_BUS_ID  — deterministic enumeration
#   CUDA_VISIBLE_DEVICES=0        — the one NVIDIA device is cuda:0
# The Windows OS-level dGPU preference is handled separately by
# gpu_enforce.enforce_discrete_gpu() via the DirectX registry.
CUDA_DEVICE_ID = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE_ID)

# ══════════════════════════════════════════════
#  v7 ADDITIONS
# ══════════════════════════════════════════════
# Step 4: Face Tracking
TRACKER_IOU_THRESHOLD = 0.3
TRACKER_MAX_MISSED_FRAMES = 15
TRACKER_MAX_PERSONS = 5

# Step 5: rPPG Calibration Regression
CALIBRATION_REGRESSION_FILE = os.path.join(CALIBRATION_DIR, "rppg_correction.json")
CALIBRATION_MIN_PAIRS_FOR_FIT = 10

# Step 7: Session Recording / Replay
SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)
SESSION_RECORD = True

# Step 9: Frame Quality
FRAME_QUALITY_ENABLED = True
FRAME_QUALITY_MIN_BRIGHTNESS = 30
FRAME_QUALITY_MAX_BRIGHTNESS = 230

# Step 10: App metadata
# v9.8: bumped from 9.7.1 -- this is the API_up upgrade fork's first
# user-visible feature surface (FF_USE_FP16, FF_BLINK_CNN, FF_AUDIO_NPU,
# vllm_local LLM, dual-camera UI pill, patch_depth EMA baseline,
# truth_engine vocal-indicator baseline subtraction).
APP_VERSION = "9.8.0"

# ══════════════════════════════════════════════
#  MAXIMUM POWER SETTINGS
# ══════════════════════════════════════════════
# v8.6: scale worker pool with the actual CPU. On a 6c/12t Ryzen 5 5600H this
# yields 12 workers; on a high-core desktop it scales up to a hard ceiling of
# 24 (beyond that we get TF/ONNX context-switch thrash). We deliberately pick
# logical-cores rather than physical so DeepFace's TF backend can keep one
# inference in flight per face while MediaPipe + rPPG continue on the side.
import multiprocessing as _mp
try:
    _LOGICAL_CORES = max(4, _mp.cpu_count())
except Exception:
    _LOGICAL_CORES = 8
THREAD_POOL_WORKERS = min(24, max(8, _LOGICAL_CORES))
# v8.6: with the new 1Hz-anchored full-frame cadence, DeepFace can run on
# every full frame without falling behind. Was 6 (every 6th frame); now 1
# means "run on every full-mode frame the UI sends" which is exactly 1Hz.
DEEPFACE_EVERY_N = 1
FACE_RECOG_EVERY_N = 4            # face recognition twice per second
MAX_INFLIGHT = max(4, _LOGICAL_CORES // 2)
FULL_MESH_ALWAYS = True
JPEG_QUALITY = 0.55               # higher quality so DeepFace gets cleaner pixels
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480

# v8.6: GPU acceleration knobs. ONNX Runtime always tries CUDA → CPU in that
# order. TensorFlow (DeepFace backend) needs explicit memory growth so it
# doesn't lock the entire 6GB of VRAM on first use.
# v9.1.3: explicit CUDA provider options for deterministic GPU binding.
# device_id=0 targets the one NVIDIA GPU visible to CUDA on this machine
# (see the block above — the iGPU is not a CUDA device, so the 3060 is
# always cuda:0 with CUDA_VISIBLE_DEVICES=0 / PCI_BUS_ID ordering).
ONNX_CUDA_PROVIDER_OPTIONS = {
    "device_id": 0,  # the discrete NVIDIA GPU (only CUDA device on this host)
    "arena_extend_strategy": "kSameAsRequested",  # v8.9: contiguous allocation — no fragmentation
    # v9.7.2 Blackwell tune: 2 GB -> 16 GB. Original cap sized for the
    # 3060 Laptop's 6 GB VRAM. On the 5090 Laptop (24 GB GDDR7) 2 GB
    # is 8 % utilization and forces ORT to fragment + evict when
    # multiple PAD branches inference concurrently. 16 GB leaves
    # 8 GB headroom for torch (deepfake_temporal MC3-18 ~1 GB),
    # TensorFlow CPU buffers, and OS overhead.
    "gpu_mem_limit": 16 * 1024 * 1024 * 1024,  # 16 GB VRAM cap
    "cudnn_conv_algo_search": "EXHAUSTIVE",
    "do_copy_in_default_stream": True,
    "gpu_external_alloc": "0",   # v8.9: let ORT manage its own arena (no external allocator)
    "gpu_external_free": "0",
    "cudnn_conv_use_max_workspace": "1",  # v8.9: max cuDNN workspace for faster conv
}

# v8.9 Step 4: rPPG GPU offload config — use CuPy for FFT/bandpass when available
# Falls back to NumPy/SciPy transparently if CuPy is not installed.
RPPG_GPU_OFFLOAD = True     # attempt GPU-accelerated FFT via CuPy
RPPG_GPU_FFT_MIN_LEN = 256  # minimum buffer length to justify GPU transfer overhead
ONNX_PROVIDERS_PREFERRED = [
    ("CUDAExecutionProvider", ONNX_CUDA_PROVIDER_OPTIONS),
    "CPUExecutionProvider",
]

# v9.1.6: DirectML provider options — device_id picks the DXGI adapter.
# On Windows 11 laptops with Intel/AMD integrated graphics + NVIDIA
# discrete, DXGI enumerates:
#   adapter 0  = integrated GPU (Task Manager "GPU 0")
#   adapter 1  = NVIDIA discrete (Task Manager "GPU 1")
# DirectML defaults to device 0 which is why the integrated GPU was
# pinned at 100% while the 3060 sat at 0% (matches your Task Manager
# screenshot). Setting device_id=1 routes DirectML straight to the
# NVIDIA. If your machine happens to have the NVIDIA as adapter 0
# (rare — usually a BIOS "discrete only" mode), set this to 0.
DML_PROVIDER_OPTIONS = {"device_id": 1}
ONNX_DML_PROVIDER_OPTIONS = DML_PROVIDER_OPTIONS
TF_GPU_MEMORY_GROWTH = True
WARMUP_MODELS_ON_START = True     # pre-run a tiny inference per model to JIT-compile

# v8.7 (Step 6): build & use a FP16 copy of the HSEmotion ONNX on first run.
# CUDA ORT runs FP16 ~1.6x faster on Ampere/Ada with <0.5% accuracy delta.
HSEMOTION_FP16 = True

# ── DeepFace cache expiry (seconds) ──
# Must be > DF_DEMOGRAPHICS_MIN_INTERVAL_S + typical DeepFace.analyze wall time
# (~300-500 ms on CPU), otherwise the cache expires ~0.5 s before the next
# demographics run writes, and AGE/GENDER/RACE briefly render as "?" every
# cycle. 3.5 s covers 2 s min-interval + 1 s DF runtime + 0.5 s safety margin.
DF_CACHE_MAX_AGE = 3.5

# v9.2 (plan steps 10-11): split cadences for DeepFace components.
#   - Emotion is handled every fast frame by HSEmotion ONNX (plan step 9).
#   - Age/gender/race rarely change; re-run at 2 s even if bbox moved.
#   - If the bbox has barely shifted (≤ DF_BBOX_STABLE_PX on the 10 px
#     rounded grid used by _bbox_moved), skip DeepFace entirely and
#     reuse the cached result.
DF_DEMOGRAPHICS_MIN_INTERVAL_S = 2.0
DF_BBOX_STABLE_PX = 20

# v9.2 plan step 9: HSEmotion on every fast frame.
HSEMOTION_EVERY_FAST_FRAME = True
HSEMOTION_COALESCE_MS = 33  # dedupe duplicate async requests inside 33 ms
# v9.7.2: cap concurrent HSEmotion calls per frame to keep the fast-frame
# budget (~33 ms at 30 fps). Each face costs ~8 ms on the 3060 (DML);
# 3 = ~24 ms worst case, leaves headroom for the rest of the pipeline.
HSEMOTION_MAX_FACES = 3

# ── Valence-Arousal ──
EMOTION_VA = {
    "happy": (0.8, 0.6), "sad": (-0.7, -0.3), "angry": (-0.6, 0.8),
    "surprise": (0.2, 0.9), "fear": (-0.8, 0.7), "disgust": (-0.6, 0.3),
    "neutral": (0.0, 0.0),
}


# ══════════════════════════════════════════════════════════════════════
#  v9.5 runtime env-var documentation (single source of truth, 2026-04-22)
# ══════════════════════════════════════════════════════════════════════
# All variables below are env-vars read by various submodules. This block
# documents the surface so you don't have to grep. Defaults match each
# module's internal fallback.
#
#  --- Logging (log_setup.py) ---
#  DODO_LOG_FORMAT       text | json                  default: text
#  DODO_LOG_LEVEL        DEBUG | INFO | WARNING|...   default: INFO
#  DODO_LOG_FILE         path (enables RotatingFileHandler) default: unset
#  DODO_LOG_FILE_BYTES   rotate when file exceeds     default: 10485760 (10 MB)
#  DODO_LOG_FILE_BACKUPS number of backups to keep    default: 5
#
#  --- Auth + TLS + Replay (auth.py / tls.py / session_replay.py) ---
#  DODO_AUTH_TOKEN       comma-separated bearer tokens; off when unset
#  DODO_TLS_CERT         PEM path
#  DODO_TLS_KEY          PEM path
#  DODO_TLS_SELFSIGN     1 -> dev cert at certs/localhost.{crt,key}
#  DODO_REPLAY_RECORD    1 -> record each WS session to sessions/replay/*.mpk
#
#  --- Rate limiting (rate_limit.py) ---
#  DODO_WS_RATE_PER_SEC  requests/sec per (token, ip)  default: 30
#  DODO_WS_RATE_BURST    burst capacity                default: 60
#
#  --- PAD thread pool + heavy-branch throttle ---
#  DODO_ONNX_WORKERS     PAD ORT thread pool size      default: 4
#  DODO_HEARTBEAT_S      crash-diag stack-dump interval default: 30
#  PAD_HEAVY_INTERVAL_MS heavy-branch throttle (moire/patch_depth/deepfake)
#                                                       default: 500
#
#  --- Deepfake branch (temporal head) ---
#  PAD_DEEPFAKE_HEAVY    1|true|yes -> enable         default: auto (if
#                        pad/models/deepfake_temporal.onnx exists)
#                        0|false|no -> force disable
#  PAD_DEEPFAKE_FAKE_INDEX  0|1 override for HF models with swapped labels
#
#  --- Feature flags (feature_flags.py) ---
#  FF_PAD_FUSION, FF_PAD_RPPG, FF_PAD_MOIRE, FF_PAD_CNN,
#  FF_PAD_DYNAMICS, FF_PAD_DEEPFAKE, FF_PAD_AUDIO,
#  FF_PAD_RADAR_PRESENCE, FF_PAD_BLINK_GAZE   (2026-04-22 additions)
#  FF_PAD_CALIBRATE, FF_PAD_META_FUSE, FF_PAD_UI
#  FF_PYANNOTE         1 -> enable optional pyannote diarisation path
#
#  --- v9.6 additions (2026-04-22 evening) ---
#  SAFE_MODE             1 -> start with SAFE_MODE.md constraints applied
#                        (sensors gated, heavy LLM off, forensic rotation)
#  DODO_SESSION_ID       override auto-generated session id (forensic log)
#  DODO_CLINICAL_SECRET  bearer for /clinical/* endpoints (off when unset)
#  PAD_PIPELINE_INTERIM_DIR  scratch dir for /models/deploy swaps
#                            default: <repo>/.pad_stage
#  HF_TOKEN / HUGGING_FACE_HUB_TOKEN   HF Hub auth for model pulls
#  GROQ_API_KEY / TOGETHER_API_KEY     fallback LLM providers for llm.py
#                            governor (Anthropic is primary; these fire
#                            when the governor degrades — see llm_governor)
#
#  --- Queue / training (queue_master.cmd, queue_*.cmd) ---
#  These are read by the cmd scripts, not the runtime server, but are
#  documented here as a single source of truth:
#    KILL_EXISTING         1 -> taskkill /F any running python.exe at start
#    MASTER_PATCH_DEPTH    1 -> enable stage 6 patch_depth retrain
#    MASTER_FXR_DIVERSE    1 -> enable stage 7 face_xray diverse retrain
#    MASTER_FXR_EFFNET     1 -> enable stage 8 face_xray EfficientNet-B4
#    MASTER_AUDIO_ADV      1 -> enable stage 9 audio AASIST adversarial
#    TRAIN_*               legacy gates used by queue_calib_sprint.cmd
#
# (end of v9.5/v9.6 env-var documentation block)
