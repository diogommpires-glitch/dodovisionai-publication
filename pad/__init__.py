"""
DODO VISION AI v9.4 — Presentation Attack Detection (PAD) subsystem.

Defense-in-depth PAD stack covering print, cutout, screen replay, 2D/3D
mask, deepfake video, adversarial patch, and audio spoofing attacks.

Design principles (see THREAT_MODEL.md for the attack → countermeasure
matrix and PAD_PRIVACY.md for data-minimisation posture):

  1. Defense-in-depth — no single modality trusted alone.
  2. ISO/IEC 30107-3:2023 reportable (APCER / BPCER / ACER).
  3. Calibrated per-branch probabilities, log-odds fusion.
  4. PersonState isolation preserved verbatim.
  5. INSTANT-tier latency contract inviolable (client-side fast guards only).

All branches produce a `BranchScore` consumed by `pad.fusion.fuse`.
A single `PADScore` per person lives inside `PersonState.pad_state`.
"""
from pathlib import Path

# v9.8.1 2026-05-12: silence ORT's global "Default" logger at import.
# Without this, loading deepfake_temporal.onnx prints 60+ lines of
# ScatterND warnings (every sub-graph emits one). These are not
# errors -- ORT is informing us that the CUDA kernel for ScatterND
# can't guarantee correctness with duplicated indices, but the model
# doesn't produce duplicates. Keeping ERROR/FATAL visible.
# 0=verbose 1=info 2=warning 3=error 4=fatal.
try:
    import onnxruntime as _ort
    _ort.set_default_logger_severity(3)
except Exception:
    pass

from pad.types import (
    AttackClass, BranchScore, PADScore, PADState,
    ATTACK_CLASSES, VerdictLabel,
)
from pad.fusion import fuse, fuse_with_history
from pad.thresholds import PADThresholds, DEFAULT_THRESHOLDS

# ── v9.4.1+: Model-fetch bundle ────────────────────────────────────
# Submodules:
#   fetch_patch_depth : stub or prints training instructions
#   fetch_face_xray   : downloads Face X-ray U-Net weights from HF
#   fetch_audio_pad   : downloads AASIST-L / RawNet2 weights
#   train_moire_gbt   : auto-trains the LightGBM moire classifier
#   verify_3dmm       : initializes face-alignment 3DMM basis
#   setup_all         : one-shot orchestrator
PAD_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PAD_ROOT / "models"
CACHE_DIR = PAD_ROOT / "cache"
MODEL_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

__all__ = [
    "AttackClass", "BranchScore", "PADScore", "PADState",
    "ATTACK_CLASSES", "VerdictLabel",
    "fuse", "fuse_with_history",
    "PADThresholds", "DEFAULT_THRESHOLDS",
    "PAD_ROOT", "MODEL_DIR", "CACHE_DIR",
]

__version__ = "0.9.4"          # legacy -- kept for backwards compat

# Repo-aligned PKG version. Mirrors `config.APP_VERSION` at build time.
# If you bump one, bump the other (the CITATION.cff also tracks).
PKG_VERSION = "9.8.0"
