"""
DODO VISION AI v9.4 PAD Step 3 — Moiré pattern branch.

Detects screen-replay attacks by measuring the moiré aliasing between
a camera's Bayer grid and a display's subpixel grid. Two surfaces:

  - Server verifier: 2-D DCT low/high ratios per channel + LBP on
    magnitude spectrum + per-channel high-freq cross-correlation,
    scored by a LightGBM gradient-boosted tree when weights are
    available. When the weights file is missing we fall back to a
    hand-engineered heuristic sized against MSU-MFSD observations.
  - Client fast-guard (lives in ui.py): runs a 64×64 |FFT| on each RGB
    channel in WebGL/WebGPU at ~5 Hz and forwards `padFast.moire_rgb`.

Refs:
  - Patel, Han, Jain, Ott, *Live Face Video vs. Spoof Face Video: Use
    of Moiré Patterns to Detect Replay Video Attacks*, ICB 2015 (MSU).
  - CVPR 2024 Unified Physical-Digital Attack Detection Challenge —
    moiré augmentation as a training staple.
"""
from __future__ import annotations
import os
import time
import json
from collections import deque

# v9.7.5: rolling window of recent raw outputs. When the last
# `_MOIRE_SAT_WIN` outputs are all >0.995 the LightGBM model is in
# degenerate-saturation mode for the current subject and reliability
# collapses to ≤0.05 — branch still appears in the panel but can't
# dominate fusion. Mirrors the patch_depth_branch._saturate_buf logic.
_MOIRE_SAT_WIN = 5

# v9.8.1 2026-05-12 (Phase 2 #1: per-subject state isolation).
# Previous design used a single module-level `_moire_saturate_buf =
# deque()` shared by ALL persons. With multi-person tracking, person A
# and person B's frames would interleave into the same buffer,
# corrupting the saturated-run detection (a flicker on one subject
# could blank the detector for the other). Replace with one deque per
# pid. pid=None routes to a shared default bucket so legacy call sites
# stay backward-compatible.
_moire_saturate_state: "dict[int, deque[float]]" = {}


def _get_sat_buf(pid: "Optional[int]") -> "deque[float]":
    key = 0 if pid is None else int(pid)
    buf = _moire_saturate_state.get(key)
    if buf is None:
        buf = deque(maxlen=_MOIRE_SAT_WIN)
        _moire_saturate_state[key] = buf
    return buf


def prune_subject(pid: int) -> None:
    """Drop per-subject saturate state when a person is no longer
    being tracked. Idempotent."""
    _moire_saturate_state.pop(int(pid), None)
from typing import Optional, Dict

import numpy as np

from pad.types import BranchScore
from pad.calibrate import apply_calibration

_MOD_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_TXT_PATH = os.path.join(os.path.dirname(_MOD_DIR), "models", "moire_gbt.txt")
# v9.4.1+: the bundled `pad.train_moire_gbt` trainer writes a 20-D-feature
# LightGBM classifier pickled via joblib. When present, it takes priority
# over the legacy native `.txt` booster (which used a 9-D hand-crafted
# feature vector). _MODEL_FORMAT is set by _load_model() to one of
# {"pkl20", "txt9", None} so `score()` can dispatch to the right
# featurizer.
_MODEL_PKL_PATH = os.path.join(os.path.dirname(_MOD_DIR), "models", "moire_gbt.pkl")

_LGBM_MODEL = None
_LGBM_LOADED = False
_MODEL_FORMAT = None


_MODEL_FEATURE_MODE = "legacy"  # "legacy" (20-D) or "enhanced" (374-D, HSV/YCbCr LBP)


def _load_model():
    global _LGBM_MODEL, _LGBM_LOADED, _MODEL_FORMAT, _MODEL_FEATURE_MODE
    if _LGBM_LOADED:
        return _LGBM_MODEL
    _LGBM_LOADED = True
    # Prefer the pkl bundle if it ships. API_unq: bundles may declare
    # feature_mode="enhanced" (374-D HSV/YCbCr LBP) — dispatch via metadata.
    if os.path.exists(_MODEL_PKL_PATH):
        try:
            import joblib
            bundle = joblib.load(_MODEL_PKL_PATH)
            if isinstance(bundle, dict):
                _LGBM_MODEL = bundle.get("model")
                _MODEL_FEATURE_MODE = str(bundle.get("feature_mode", "legacy")).lower()
                feat_dim = bundle.get("feature_dim", 20)
            else:
                _LGBM_MODEL = bundle
                _MODEL_FEATURE_MODE = "legacy"
                feat_dim = 20
            _MODEL_FORMAT = "pkl_enhanced" if _MODEL_FEATURE_MODE == "enhanced" else "pkl20"
            print(f"[PAD moire] LightGBM (pkl, {feat_dim}-D, mode={_MODEL_FEATURE_MODE}) "
                  f"loaded from {_MODEL_PKL_PATH}")
            return _LGBM_MODEL
        except Exception as e:
            print(f"[PAD moire] pkl load failed ({e}); trying legacy .txt")
    if os.path.exists(_MODEL_TXT_PATH):
        try:
            import lightgbm as lgb
            _LGBM_MODEL = lgb.Booster(model_file=_MODEL_TXT_PATH)
            _MODEL_FORMAT = "txt9"
            print(f"[PAD moire] LightGBM (txt, 9-D) loaded from {_MODEL_TXT_PATH}")
        except Exception as e:
            print(f"[PAD moire] LightGBM load failed ({e}); falling back to heuristic")
            _LGBM_MODEL = None
            _MODEL_FORMAT = None
    return _LGBM_MODEL


def _dct2_features(img_gray: np.ndarray) -> Dict[str, float]:
    """2D DCT low/mid/high energy ratios — cheap moiré tell."""
    import cv2
    h, w = img_gray.shape
    # Pad to even sides
    if h % 2 or w % 2:
        h = h // 2 * 2
        w = w // 2 * 2
        img_gray = img_gray[:h, :w]
    f32 = np.float32(img_gray) / 255.0
    dct = cv2.dct(f32)
    # Partition into low/mid/high bands
    ql = min(h, w) // 8
    low = np.abs(dct[:ql, :ql]).sum()
    total = np.abs(dct).sum() + 1e-9
    mid_mask = np.zeros_like(dct, dtype=bool)
    mid_mask[:h // 2, :w // 2] = True
    mid_mask[:ql, :ql] = False
    mid = np.abs(dct[mid_mask]).sum()
    high = total - low - mid
    return {
        "dct_low_ratio": float(low / total),
        "dct_mid_ratio": float(mid / total),
        "dct_high_ratio": float(high / total),
    }


def _lbp_on_fft_mag(img_gray: np.ndarray) -> Dict[str, float]:
    """Compute LBP histogram on the log-magnitude FFT of the face
    grayscale — a moiré-specific texture signature per the MSU paper.
    """
    try:
        from skimage.feature import local_binary_pattern
    except Exception:
        return {"lbp_unif_ratio": 0.0, "lbp_var": 0.0}
    img_gray = np.float32(img_gray) / 255.0
    f = np.fft.fftshift(np.fft.fft2(img_gray))
    mag = np.log1p(np.abs(f))
    # Normalise to 0-255 for LBP
    mn, mx = mag.min(), mag.max()
    if mx - mn < 1e-9:
        return {"lbp_unif_ratio": 0.0, "lbp_var": 0.0}
    mag_u8 = ((mag - mn) / (mx - mn) * 255.0).astype(np.uint8)
    lbp = local_binary_pattern(mag_u8, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    hist = hist.astype(np.float64)
    hist /= (hist.sum() + 1e-9)
    # Uniform (low-information) spectra score high on the first bin;
    # moiré-laden spectra spread into higher bins.
    unif_ratio = float(hist[0])
    lbp_var = float(np.var(hist))
    return {"lbp_unif_ratio": unif_ratio, "lbp_var": lbp_var}


def _channel_cross_corr(img_bgr: np.ndarray) -> Dict[str, float]:
    """High-frequency cross-correlation across R, G, B channels.
    Moiré couples across channels because the Bayer/display grids
    co-align across colour channels when the recapture is off-axis.
    """
    import cv2
    b, g, r = cv2.split(img_bgr)
    # High-pass each channel
    def hp(ch):
        blur = cv2.GaussianBlur(ch, (7, 7), 1.5)
        return ch.astype(np.float32) - blur.astype(np.float32)
    hp_r, hp_g, hp_b = hp(r), hp(g), hp(b)

    def corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        num = float((a * b).sum())
        den = math.sqrt(float((a * a).sum()) * float((b * b).sum()) + 1e-9)
        return num / den
    import math
    rg = corr(hp_r, hp_g)
    rb = corr(hp_r, hp_b)
    gb = corr(hp_g, hp_b)
    avg = (rg + rb + gb) / 3.0
    return {
        "hp_corr_rg": rg, "hp_corr_rb": rb, "hp_corr_gb": gb,
        "hp_corr_avg": avg,
    }


def _heuristic_p_attack(feats: Dict[str, float]) -> float:
    """Fallback scorer used when no trained GBT is present.

    Intuition drawn from the MSU moiré paper: screen recaptures show
    (1) elevated mid-frequency DCT energy from the Bayer/display beat,
    (2) low LBP uniformity on the FFT magnitude (moiré creates
    structured artefacts), (3) high cross-channel high-freq correlation
    (the display subpixels move together across RGB).
    """
    mid = feats.get("dct_mid_ratio", 0.0)
    unif = feats.get("lbp_unif_ratio", 0.5)
    ccorr = feats.get("hp_corr_avg", 0.0)
    # Expected live ranges (webcam, flat forehead crop):
    #   mid ≈ 0.15–0.30, unif ≈ 0.35–0.55, ccorr ≈ 0.15–0.35
    # Replay ranges (phone on a 60 Hz monitor):
    #   mid > 0.35, unif < 0.25, ccorr > 0.5
    mid_s = max(0.0, min(1.0, (mid - 0.30) / 0.15))
    unif_s = max(0.0, min(1.0, (0.30 - unif) / 0.15))
    ccorr_s = max(0.0, min(1.0, (ccorr - 0.35) / 0.20))
    return float(0.4 * mid_s + 0.3 * unif_s + 0.3 * ccorr_s)


def score(face_crop_bgr: Optional[np.ndarray],
          client_moire_rgb: Optional[float] = None,
          pid: "Optional[int]" = None) -> BranchScore:
    """Run the moiré branch on a face crop (BGR). Returns reliability=0
    when the crop is too small / degenerate.

    v9.4.1 step 3: the optional `client_moire_rgb` argument carries the
    fresh high-frequency RGB energy ratio computed by the browser-side
    WebGL2 fast-guard (ui.py::_padFastComputeMoire). When both server
    and client agree (both high, both low) we boost reliability;
    disagreement dampens it. Value is in roughly [0, 1].

    v9.8.1 2026-05-12 (Phase 2 #1): added `pid` arg for per-subject
    state isolation. Defaults to None (= shared default bucket) so
    older call sites continue to work; new call sites should pass
    `pid=pid` explicitly to keep multi-person sessions isolated.
    """
    t0 = time.monotonic()
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return BranchScore(name="moire", p_attack=0.5, reliability=0.0,
                           latency_ms=(time.monotonic() - t0) * 1000.0)
    h, w = face_crop_bgr.shape[:2]
    if h < 48 or w < 48:
        return BranchScore(name="moire", p_attack=0.5, reliability=0.0,
                           latency_ms=(time.monotonic() - t0) * 1000.0)
    try:
        import cv2
        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = face_crop_bgr if face_crop_bgr.ndim == 2 else face_crop_bgr[:, :, 0]

    feats: Dict[str, float] = {}
    try:
        feats.update(_dct2_features(gray))
        feats.update(_lbp_on_fft_mag(gray))
        feats.update(_channel_cross_corr(face_crop_bgr) if face_crop_bgr.ndim == 3
                     else {"hp_corr_avg": 0.0})
    except Exception:
        # Fall through with whatever we got
        pass

    model = _load_model()
    if model is not None and _MODEL_FORMAT in ("pkl20", "pkl_enhanced"):
        # API_unq: dispatch featurizer on metadata flag. The enhanced 374-D
        # path adds HSV/YCbCr uniform LBP histograms — discriminative when
        # OLPF (e.g. Canon R10 EOS Webcam Utility) blurs high-freq Moire
        # below FFT detection.
        try:
            from pad.train_moire_gbt import extract_features_for_mode as _mg_extract
            import cv2
            rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB) \
                if face_crop_bgr.ndim == 3 else face_crop_bgr
            x = _mg_extract(rgb, _MODEL_FEATURE_MODE).reshape(1, -1)
            # scikit-learn LGBMClassifier exposes predict_proba
            p_attack = float(model.predict_proba(x)[0, 1])
            reliability = 1.0
        except Exception:
            p_attack = _heuristic_p_attack(feats)
            reliability = 0.5
    elif model is not None and _MODEL_FORMAT == "txt9":
        try:
            feature_names = [
                "dct_low_ratio", "dct_mid_ratio", "dct_high_ratio",
                "lbp_unif_ratio", "lbp_var",
                "hp_corr_rg", "hp_corr_rb", "hp_corr_gb", "hp_corr_avg",
            ]
            x = np.asarray([[feats.get(k, 0.0) for k in feature_names]],
                           dtype=np.float64)
            p_attack = float(model.predict(x)[0])
            reliability = 1.0
        except Exception:
            p_attack = _heuristic_p_attack(feats)
            reliability = 0.5
    else:
        p_attack = _heuristic_p_attack(feats)
        reliability = 0.5   # heuristic — half-weight in fusion

    # v9.4.1 step 3: cross-validate with the client fast-guard cue.
    if client_moire_rgb is not None:
        try:
            c = max(0.0, min(1.0, float(client_moire_rgb)))
            # Agreement: both high (>0.4) or both low (<0.2) → boost
            # reliability by +0.2 capped at 1.0. Disagreement (|Δ|>0.3)
            # → dampen by -0.2. Small nudges preserve model primacy.
            diff = abs(c - p_attack)
            if (c > 0.4 and p_attack > 0.4) or (c < 0.2 and p_attack < 0.2):
                reliability = min(1.0, reliability + 0.2)
                # Mild score nudge toward agreement
                p_attack = 0.8 * p_attack + 0.2 * c
            elif diff > 0.3:
                reliability = max(0.0, reliability - 0.2)
            feats["client_moire_rgb"] = round(c, 4)
            feats["client_server_diff"] = round(diff, 4)
        except Exception:
            pass

    raw_score = float(max(0.0, min(1.0, p_attack)))

    # v9.7.5: same degenerate-saturation guard as patch_depth_branch.
    # The LightGBM moiré model occasionally outputs raw=1.0 for real
    # faces with strong glasses (lens DCT mid-band mimics screen-pixel
    # patterns at saturation), and the Platt sigmoid asymptotes at
    # p_attack=1 regardless of personal calibration. Detect via a
    # rolling window: if the last `_MOIRE_SAT_WIN` raw outputs are all
    # >0.995, the model is in degenerate mode and reliability collapses
    # so the branch can't dominate fusion alone. Same principle as
    # `pad/branches/patch_depth_branch.py:_saturate_buf`.
    # v9.8.1 2026-05-12 (Phase 2 #1): per-subject saturate buffer
    # lookup. deque(maxlen=N) auto-evicts on append; no explicit
    # popleft needed.
    _sat_buf = _get_sat_buf(pid)
    _sat_buf.append(raw_score)
    saturated_run = (len(_sat_buf) >= _MOIRE_SAT_WIN
                     and all(x > 0.995 for x in _sat_buf))

    # Clip raw at 0.999 BEFORE calibration so Platt has finite logit
    # headroom. raw=1.0 maps to sigmoid(∞)=1 regardless of fit.
    raw_clipped = max(0.001, min(0.999, raw_score))
    p_attack = apply_calibration("moire", raw_clipped)

    # API_unq 2026-05-13 (P1.5): removed inverse `conf_factor = 0.3 +
    # 0.7 * margin` that REWARDED confidence rather than damping it.
    # margin = |p-0.5|*2 is the post-calibration confidence; multiplying
    # reliability up by margin double-counts the same signal that drove
    # the score in the first place. Sensoy et al. NeurIPS 2018 (EDL
    # Dirichlet vacuity) and Lin et al. CVPR 2024 (MMDG U-Adapter) both
    # downweight low-uncertainty signals, not upweight them. Reliability
    # is now passed through unchanged from the upstream signal-quality
    # estimator. A future API_unq pass will replace this with a proper
    # ReliabilityCalibrator (P1.1 in the literature brief).
    reliability = float(reliability)

    # Degenerate-saturation override.
    if saturated_run:
        reliability = min(reliability, 0.05)

    feats["raw_pre_calib"] = round(raw_clipped, 4)
    feats["saturated_run"] = float(saturated_run)
    # v9.8.1 2026-05-12 (Phase 2 #1): surface the pid bucket so
    # diagnostics + tests can verify the per-subject isolation is
    # routing correctly.
    feats["pid"] = float(0 if pid is None else int(pid))

    # v9.8: same dominant_class gate as patch_depth_branch -- when the
    # branch is in saturated_run mode the panel should NOT flash
    # "REPLAY_SCREEN" because the underlying model is in a degenerate
    # regime (reliability already collapsed to 0.05). Pre-fix this
    # leaked the wrong attack class to the UI even though the score
    # was already discounted by fusion math.
    #
    # v9.8.1 2026-05-12 (9th-pass deep audit): also gate on reliability.
    # The calibrator can saturate p_attack near 1.0 on a glasses-on-real-
    # subject combination where the moire LightGBM produces a high raw
    # score; without the rel gate the UI would render "REPLAY_SCREEN"
    # even though the branch is contributing weakly to fusion (rel<0.3).
    # Match the deepfake gate: BOTH p_attack > 0.85 AND reliability >= 0.3.
    dominant = None
    if (not saturated_run) and p_attack >= 0.85 and reliability >= 0.30:
        dominant = "REPLAY_SCREEN"

    return BranchScore(
        name="moire",
        p_attack=max(0.0, min(1.0, p_attack)),
        reliability=max(0.0, min(1.0, reliability)),
        latency_ms=(time.monotonic() - t0) * 1000.0,
        meta=feats,
        dominant_class=dominant,
    )
