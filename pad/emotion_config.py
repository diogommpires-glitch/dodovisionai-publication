"""pad/emotion_config.py -- shared constants for the Tier-A / Tier-B / Tier-C
emotion evaluation pipeline.

Single source of truth for label orders, dataset paths, dataset->HSE class
mappings, and the BEST_TIER_A_CONFIGS lookup. Imported by:
  - pad/emotion_features.py
  - scripts/emotion_tier_a_sweep.py
  - scripts/emotion_tier_b_finetune.py
  - scripts/emotion_confusion_matrices.py
  - scripts/emotion_balanced_metrics.py
  - scripts/emotion_deployment_curves.py
  - scripts/emotion_external_ensemble.py
"""
from __future__ import annotations
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ── Cache + report layout ──────────────────────────────────────────
TIER_A_CACHE_DIR = REPO / "reports" / "_tier_a_cache"

# ── Hyperparameter defaults shared across the sweep + downstream ───
CV_N_FOLDS = 5
CV_SEED = 42
MLP_HIDDEN_UNITS = 256
SVM_C = 10.0
LR_C = 1.0

# ── Label orders ────────────────────────────────────────────────────
# HSEmotion's 7-class output. Sorted alphabetically (matches sklearn label_set).
HSE_LABELS_7 = ["Anger", "Disgust", "Fear", "Happiness",
                "Neutral", "Sadness", "Surprise"]
# HSEmotion's 8-class output. Adds Contempt at index 1.
HSE_LABELS_8 = ["Anger", "Contempt", "Disgust", "Fear", "Happiness",
                "Neutral", "Sadness", "Surprise"]

# AffectNet-canonical order used by DAN, POSTER, EmoNet (8-cls).
AFFECTNET_LABELS_8 = ["Neutral", "Happiness", "Sadness", "Surprise",
                       "Fear", "Disgust", "Anger", "Contempt"]
AFFECTNET_LABELS_7 = AFFECTNET_LABELS_8[:-1]
# Aliases for legacy script names.
POSTER_LABELS_7 = AFFECTNET_LABELS_7
POSTER_LABELS_8 = AFFECTNET_LABELS_8
DAN_LABELS_7 = AFFECTNET_LABELS_7
DAN_LABELS_8 = AFFECTNET_LABELS_8

# ── Dataset path roots ──────────────────────────────────────────────
JAFFE_DIR = Path(r"E:\pad_datasets\JAFFE\jaffe\jaffe")
NHFI_DIR = Path(r"E:\pad_datasets\NHFI")
EMOTIC_DIR = Path(r"E:\pad_datasets\EMOTIC")
EMOTIC_ANN = EMOTIC_DIR / "Annotations" / "Annotations.mat"
EMOTIC_IMG_ROOT = EMOTIC_DIR / "emotic"
YUNET_PATH = REPO / "pad" / "models" / "yunet_face_detection_2023mar.onnx"

# ── Dataset-specific label mappings ─────────────────────────────────
# JAFFE filename code -> HSE-canonical class name.
JAFFE_CODE_TO_LABEL = {
    "AN": "Anger", "DI": "Disgust", "FE": "Fear", "HA": "Happiness",
    "NE": "Neutral", "SA": "Sadness", "SU": "Surprise",
}
# JAFFE filename: <subject 2 letters>.<emotion 2><take>.<frame>.tiff
JAFFE_FNAME_RE = re.compile(r"^([A-Z]{2})\.([A-Z]{2})\d+\.\d+\.tiff?$",
                             re.IGNORECASE)

# NHFI folder name (lowercase) -> HSE-canonical class name.
NHFI_TO_HSE = {"anger": "Anger", "contempt": "Contempt", "disgust": "Disgust",
               "fear": "Fear", "happiness": "Happiness",
               "neutrality": "Neutral", "sadness": "Sadness",
               "surprise": "Surprise"}

# HuggingFace Jeneral/fer-2013 label -> HSE-canonical class name.
HF_TO_HSE = {"angry": "Anger", "disgust": "Disgust", "fear": "Fear",
             "happy": "Happiness", "neutral": "Neutral", "sad": "Sadness",
             "surprise": "Surprise"}

# EMOTIC categories that map cleanly to HSE 7-class space (5 classes).
EMOTIC_TO_HSE = {"Anger": "Anger", "Fear": "Fear", "Happiness": "Happiness",
                 "Sadness": "Sadness", "Surprise": "Surprise"}

# ── Tier-A best (variant, classifier) per dataset ─────────────────
# From `reports/emotion_tier_a_sweep_v9.11.md`. Used by confmat / balmetric /
# deploy / Tier-B as the canonical "best Tier-A configuration".
BEST_TIER_A_CONFIGS = [
    ("JAFFE",   "JAFFE_c7_tta1.npz",   "mlp"),
    ("NHFI",    "NHFI_c8_tta1.npz",    "svm"),
    ("FER2013", "FER2013_c7_tta1.npz", "svm"),
    ("EMOTIC",  "EMOTIC_c7_tta1.npz",  "svm"),
]

# Zero-shot HSEmotion + Tier-A linear-probe baselines (for delta columns
# in downstream reports).
HSEMOTION_ZS_BASELINE = {"JAFFE": 0.5352, "NHFI": 0.6213,
                          "FER2013": 0.3596, "EMOTIC": 0.3891}
HSEMOTION_LP_BASELINE = {"JAFFE": 0.9297, "NHFI": 0.6583,
                          "FER2013": 0.6142, "EMOTIC": 0.7887}
