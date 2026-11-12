"""
DODO VISION AI v9.4 PAD Step 4 — Patch + pseudo-depth CNN branch.

A generalizable CNN that fuses local texture patches with an estimated
pseudo-depth map. Liu et al. (CVPR 2018, "Learning Deep Models for
Face Anti-Spoofing") established that patch-based CNNs with an auxiliary
depth task generalize far better than binary classifiers. The 2024
CVPR line (CFPL-FAS, Test-Time Domain Generalization) improves
cross-domain EER but keeps the patch-depth decomposition.

This module runs INFERENCE only, against a preprocessed face crop at
224×224. Training lives outside the repo (see `pad/TRAINING.md`); the
expected ONNX weights file is:

    pad/models/patch_depth_fas.onnx

Expected ONNX I/O:
    inputs  : "image"  Float32 [1, 3, 224, 224]  RGB, mean/std-normalised
    outputs : "p_attack"   Float32 [1]           (sigmoid applied)
              "depth_l1"   Float32 [1]           per-frame mean |pred-pseudo|

Fallback when the weights file is missing: returns reliability=0.0 so
fusion ignores this branch. No false LIVE/SPOOF verdicts emitted.

References:
  - Liu et al., Learning Deep Models for Face Anti-Spoofing, CVPR 2018.
  - CFPL-FAS, CVPR 2024.
  - Test-Time Domain Generalization for Face AS, CVPR 2024.
  - Quality-Invariant DG for Face AS, IJCV 2024.
"""
from __future__ import annotations
import os
import time
from collections import deque

# v9.7.4: rolling window of recent raw outputs to detect the
# degenerate-saturation regime (model stuck at 1.0 on a real face).
# `_SATURATE_WIN` frames is ~2.5 s at the 2 Hz heavy-throttle, plenty
# to confirm a stuck model without overreacting to a single frame.
_SATURATE_WIN = 5

# v9.8.1 2026-05-12 (Phase 2 #1: per-subject state isolation).
# Previous design used module-level globals (`_subject_baseline = [0.5]`,
# `_saturate_buf = deque()`) -- if the API processed two persons in the
# same session, OR a subject reconnected under a different pid, ALL of
# their raw scores leaked into a shared baseline. With multi-person
# tracking, person A's high score could corrupt person B's EMA, etc.
#
# Replace with per-pid SubjectState. `score()` looks up (or creates)
# the bucket for the supplied `pid` and operates on that bucket alone.
# When `pid` is None (legacy single-subject mode -- old call sites
# that haven't migrated yet) we use pid=0 as the default bucket so
# behaviour is identical to the pre-Phase-2 globals.
#
# v9.8.1 2026-05-12 (Phase 2 #2: self-healing auto-recovery). The
# `_AUTO_DISABLE_SECS` flag used to be one-way: once flipped, the
# branch stayed dead until `reset_baseline()` was called. A transient
# 90-second lighting blip (someone walking behind the operator,
# room AC cycling shadow on the wall, etc.) permanently killed
# patch_depth for the rest of the session. New behaviour: once the
# branch is disabled, watch raw_score. If it falls back to a clearly-
# LIVE zone (< _AUTO_RECOVER_THRESHOLD = 0.40) for a sustained
# `_AUTO_RECOVER_SECS` window, re-enable. Threshold deliberately
# conservative so a real attack (raw stuck at 0.7-0.9) never recovers.
_BASELINE_ALPHA = 0.10
_AUTO_DISABLE_SECS = 60.0
_AUTO_RECOVER_THRESHOLD = 0.40   # raw_score safely LIVE
_AUTO_RECOVER_SECS = 5.0         # sustain window before un-disabling


class _SubjectState:
    """Per-pid scratchpad for patch_depth_branch. One instance per
    tracked person; the dict below is keyed by pid."""
    __slots__ = ("subject_baseline", "baseline_n",
                 "saturate_buf", "disabled_for_session",
                 "first_high_t", "first_recover_t")

    def __init__(self):
        self.subject_baseline: float = 0.5
        self.baseline_n: int = 0
        self.saturate_buf: "deque[float]" = deque(maxlen=_SATURATE_WIN)
        self.disabled_for_session: bool = False
        self.first_high_t: "Optional[float]" = None
        # v9.8.1 Phase 2 #2: timestamp of first frame where raw_score
        # fell below the recovery threshold while we were disabled.
        self.first_recover_t: "Optional[float]" = None


_subject_states: "dict[int, _SubjectState]" = {}
# Lock not needed under CPython GIL for the simple setdefault + per-
# bucket mutation pattern we use, but expose a helper for tests that
# want a deterministic reset.

def _get_state(pid: "Optional[int]") -> _SubjectState:
    """Return the SubjectState for `pid`, creating it on first sight.
    pid=None routes to the legacy default-bucket (pid=0)."""
    key = 0 if pid is None else int(pid)
    st = _subject_states.get(key)
    if st is None:
        st = _SubjectState()
        _subject_states[key] = st
    return st


def reset_baseline(pid: "Optional[int]" = None) -> None:
    """Reset the per-subject EMA baseline. Called when a new face binds
    or the user explicitly recalibrates. Also clears the auto-disable
    state so the branch comes back online for the new subject.

    pid=None (back-compat): reset the default bucket only.
    pid=int: reset that specific person's bucket.
    """
    key = 0 if pid is None else int(pid)
    _subject_states[key] = _SubjectState()


def prune_subject(pid: int) -> None:
    """Free the per-subject state for a pid that's no longer being
    tracked (e.g. WS disconnect / track lost). Idempotent."""
    _subject_states.pop(int(pid), None)
from typing import Optional, Dict

import numpy as np

from pad.types import BranchScore
from pad.calibrate import apply_calibration

_MOD_DIR = os.path.dirname(os.path.abspath(__file__))

# API_up v9.8 conductor phase 1: production patch_depth_fas.onnx is now v7
# (retrained 2026-05-07 on the full 444K-image mmap dataset; val_acc
# 0.998 on its own SiW-Mv2 held-out split; fixes the 18% live-FP-rate
# bug v6 had).
#
# Calibrator was refit against v7 outputs (ECE 0.0253 -> 0.0131) so
# scores are well-shaped for the live deployment distribution.
#
# Two escape hatches:
#   FF_USE_PATCH_DEPTH_V7=0  -> load patch_depth_fas_v6.onnx (legacy
#                              specialist; better cross-domain AUC 0.75
#                              but the live-FP bug returns)
#   FF_USE_PATCH_DEPTH_V7=1 (default)  -> load patch_depth_fas.onnx (v7)
_FF_USE_PATCH_DEPTH_V7 = os.environ.get(
    "FF_USE_PATCH_DEPTH_V7", "1").lower() in ("1", "true", "yes")
_DEFAULT_MODEL = "patch_depth_fas.onnx" if _FF_USE_PATCH_DEPTH_V7 else "patch_depth_fas_v6.onnx"
_MODEL_PATH = os.path.join(
    os.path.dirname(_MOD_DIR), "models", _DEFAULT_MODEL)

# API_up v9.8: FF_USE_FP16=1 picks `patch_depth_fas_fp16.onnx` if present
# (~1.2x faster on Blackwell, fp32-noise parity). See deepfake_branch._resolve_fp16
# for the same pattern.
_FF_USE_FP16 = os.environ.get("FF_USE_FP16", "0").lower() in ("1", "true", "yes")


from functools import lru_cache as _lru_cache

@_lru_cache(maxsize=8)
def _resolve_fp16(path: str) -> str:
    """v9.8: cached. Called once per session at session-load time, but
    the path-stat-and-stringop cost is unnecessary overhead. Cache is
    bounded so different model paths still resolve correctly."""
    if not _FF_USE_FP16:
        return path
    head, tail = os.path.split(path)
    name, ext = os.path.splitext(tail)
    if name.endswith("_fp16"):
        return path
    fp16_path = os.path.join(head, f"{name}_fp16{ext}")
    return fp16_path if os.path.isfile(fp16_path) else path

_ORT_SESSION = None
_ORT_LOADED = False
_INPUT_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_INPUT_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _load_session():
    global _ORT_SESSION, _ORT_LOADED
    if _ORT_LOADED:
        return _ORT_SESSION
    _ORT_LOADED = True
    path = _resolve_fp16(_MODEL_PATH)
    if not os.path.exists(path):
        print(f"[PAD patch_depth] ONNX model not found at {path}")
        print("                  -> branch will no-op until you fetch weights.")
        print("                  -> run: python pad/fetch_patch_depth.py")
        _ORT_SESSION = None
        return None
    if path != _MODEL_PATH:
        print(f"[PAD patch_depth] FF_USE_FP16: loading {os.path.basename(path)} (FP16, ~1.2x faster)")
    try:
        import onnxruntime as ort
        # Use DirectML when available (matches v9.1.3 device=1 selection).
        provs = [p for p in ort.get_available_providers()
                 if p in ("DmlExecutionProvider", "CUDAExecutionProvider")]
        if not provs:
            provs = ["CPUExecutionProvider"]
        else:
            provs = provs + ["CPUExecutionProvider"]
        _ORT_SESSION = ort.InferenceSession(path, providers=provs)
        print(f"[PAD patch_depth] ONNX loaded: providers={_ORT_SESSION.get_providers()}")
    except Exception as e:
        print(f"[PAD patch_depth] ONNX load failed ({type(e).__name__}: {e}); no-op")
        _ORT_SESSION = None
    return _ORT_SESSION


def _preprocess(face_crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Resize + BGR→RGB + normalise for the standard ImageNet stats the
    patch-depth CNN expects during inference."""
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return None
    try:
        import cv2
        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    except Exception:
        return None
    x = rgb.astype(np.float32) / 255.0
    x = (x - _INPUT_MEAN) / _INPUT_STD
    # CHW + batch
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x


def _blur_quality(img: np.ndarray) -> float:
    """Laplacian-variance-based sharpness as a proxy for reliability.
    A face crop that's motion-blurred or badly focused has Var(ΔL) ≪ 100;
    a sharp crop is > 300. Clamp linearly into [0.2, 1.0].
    """
    try:
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        v = float(lap.var())
    except Exception:
        return 0.5
    # map v=50 → 0.2, v=300 → 1.0
    x = (v - 50.0) / 250.0
    return max(0.2, min(1.0, x))


def score(face_crop_bgr: Optional[np.ndarray],
          pid: "Optional[int]" = None) -> BranchScore:
    """Score one face crop for print-attack-likeness.

    v9.8.1 2026-05-12 (Phase 2 #1): added `pid` arg for per-subject
    state isolation. Defaults to None (= shared default bucket) so
    older call sites continue to work; new call sites should pass
    `pid=pid` explicitly to keep multi-person sessions isolated.
    """
    t0 = time.monotonic()
    sess = _load_session()
    if sess is None:
        return BranchScore(
            name="patch_depth",
            p_attack=0.5,
            reliability=0.0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            meta={"status": 0.0},  # 0 = model missing
        )
    x = _preprocess(face_crop_bgr)
    if x is None:
        return BranchScore(
            name="patch_depth",
            p_attack=0.5,
            reliability=0.0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            meta={"status": -1.0},
        )
    # v9.4.1+: auto-detect input name. Original Atoum et al. code uses
    # `image`; our `pad.train_patch_depth` trainer and most HF exports
    # use `input` or `pixel_values`. Resolve from the session.
    try:
        input_name = sess.get_inputs()[0].name
    except Exception:
        input_name = "image"
    try:
        outputs = sess.run(None, {input_name: x})
    except Exception:
        return BranchScore(
            name="patch_depth",
            p_attack=0.5,
            reliability=0.0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            meta={"status": -2.0, "err": 0.0},
        )
    # v9.4.1+: accept either a scalar (legacy Atoum output) OR
    # [N, 2] logits (modern classifier-head exports).
    first = np.asarray(outputs[0])
    if first.ndim == 2 and first.shape[1] >= 2:
        logits = first[0]
        ex = np.exp(logits - logits.max())
        probs = ex / ex.sum()
        p_attack = float(probs[1])   # index 1 = spoof (matches trainer)
    else:
        p_attack = float(first.reshape(-1)[0])
    depth_l1 = float(np.asarray(outputs[1]).reshape(-1)[0]) if len(outputs) > 1 else 0.0

    reliability = _blur_quality(face_crop_bgr)
    raw_score = float(p_attack)

    # v9.8.1 2026-05-12 (Phase 2 #1): per-subject state lookup. All
    # mutations below operate on `st`, not module globals.
    st = _get_state(pid)

    # v9.7.4 — degenerate-saturation detector. The patch_depth ONNX
    # has been observed to saturate at exactly 1.0 for some live
    # subjects (specific lighting + glasses + skin tone combinations
    # that weren't in the CelebA-Spoof / SiW training distribution).
    # When that happens the Platt calibrator can't help: sigmoid(∞)=1
    # is mathematically inescapable, so the calibrated p_attack stays
    # at 1.0 and dominates fusion despite the user being clearly live.
    # Detect this via a rolling window of recent raw outputs — if ALL
    # of the last N outputs are above the saturation threshold, the
    # model is in degenerate mode for THIS subject, and we drop the
    # branch's reliability dramatically so it can't dominate.
    st.saturate_buf.append(raw_score)
    saturated_run = (len(st.saturate_buf) >= _SATURATE_WIN
                     and all(x > 0.995 for x in st.saturate_buf))

    # v9.8 — per-subject EMA baseline. Tracks the user's "resting"
    # raw_score so the dominant_class rule can fire on RELATIVE jumps
    # rather than the absolute > 0.65 threshold (which the model's
    # own LIVE distribution happens to straddle). First 8 frames seed
    # the EMA without firing dominance.
    if st.baseline_n < 1000:
        st.baseline_n += 1
    a = _BASELINE_ALPHA if st.baseline_n >= 8 else 0.5  # warm fast on first 8
    st.subject_baseline = (1 - a) * st.subject_baseline + a * raw_score
    baseline = st.subject_baseline
    raw_jump = raw_score - baseline   # +ve = score is ABOVE personal resting

    # v9.8: auto-disable tracking. The branch is "stuck high" when
    # raw_score stays above 0.6 -- that's the model's typical bias on
    # certain skin/lighting/glasses combinations where it produces
    # nonsense for THIS subject. After AUTO_DISABLE_SECS of continuous
    # high readings, give up and mark the branch session-disabled so
    # it stops dragging the meta-fuser.
    now_t = time.monotonic()
    if raw_score > 0.6:
        if st.first_high_t is None:
            st.first_high_t = now_t
        elif (now_t - st.first_high_t) > _AUTO_DISABLE_SECS:
            st.disabled_for_session = True
        # If we were tracking a recovery window but the score went
        # back high, abort the recovery attempt.
        st.first_recover_t = None
    else:
        # Score recovered -- reset the high-streak timer (handles
        # transient lighting / occlusion blips that briefly spike).
        st.first_high_t = None
        # v9.8.1 2026-05-12 (Phase 2 #2: SELF-HEALING). If we're
        # currently disabled AND the raw score is in the clearly-LIVE
        # zone, accumulate a sustain timer. Once it crosses
        # _AUTO_RECOVER_SECS, dynamically re-enable the branch so we
        # rejoin fusion. Threshold deliberately conservative (0.40)
        # so a real attack -- which keeps raw stuck in the 0.7-0.95
        # range -- never recovers; only sustained LIVE evidence flips
        # us back on.
        if st.disabled_for_session and raw_score < _AUTO_RECOVER_THRESHOLD:
            if st.first_recover_t is None:
                st.first_recover_t = now_t
            elif (now_t - st.first_recover_t) >= _AUTO_RECOVER_SECS:
                # Phoenix path: re-enable + reset the high-streak
                # timer so we can be re-disabled later if needed.
                st.disabled_for_session = False
                st.first_recover_t = None
                st.first_high_t = None
        elif raw_score >= _AUTO_RECOVER_THRESHOLD:
            # Score recovered but still in the "uncertain" zone
            # (0.40 ≤ raw < 0.6). Don't accumulate recovery time --
            # require CLEAR live evidence before un-disabling.
            st.first_recover_t = None

    # Dominant class fires only when ALL of:
    #   (a) baseline established (>= 8 frames seen)
    #   (b) NOT in degenerate saturation mode (would fire constantly
    #       on stuck-saturated subjects)
    #   (c) NOT auto-disabled for this session
    #   (d) baseline-warmup also gates the absolute >= 0.85 trigger,
    #       not just the relative jump (v9.8 fix: previously the
    #       absolute trigger could fire on the very first frame
    #       before EMA had any history to compare against)
    #   (e) EITHER raw_score >= 0.85 (very high absolute confidence)
    #       OR raw_score - personal baseline >= 0.20 (real spike)
    # v9.8.1 2026-05-12 (9th-pass deep audit): the dominant_class
    # decision used to be made on `raw_score` (PRE-calibration). That
    # meant a real subject who happened to produce raw=0.86 would emit
    # `dominant_class=PRINT` even though `apply_calibration` then
    # shifts the score to p_attack=0.32 (UNCERTAIN, leaning LIVE).
    # Result: the UI's PAD AGREEMENT panel shows "patch_depth: PRINT"
    # on a real subject when the actual fused verdict is LIVE. Move
    # the gate to POST-calibration p_attack so the dominant_class
    # reflects what the rest of the system believes about this branch.
    # Also tighten the baseline-warmup gate from 8 frames -> 30 frames
    # (~15 s at the 2 Hz heavy tick) so the EMA has had time to
    # stabilise on this subject before claiming PRINT.
    raw_clipped = max(0.001, min(0.999, raw_score))
    p_attack = apply_calibration("patch_depth", raw_clipped)

    dominant = None
    if (st.baseline_n >= 30
            and not saturated_run
            and not st.disabled_for_session
            and p_attack >= 0.75            # POST-calibration gate
            and (raw_score >= 0.85 or raw_jump >= 0.20)):
        # The patch-depth CNN is most discriminative for print/mask
        # classes; screen replay is covered by moiré + rPPG.
        dominant = "PRINT"

    # API_unq 2026-05-13 (P1.5): removed inverse `conf_factor = 0.3 +
    # 0.7 * margin` for the same reason as moire_branch.py — it
    # double-counted confidence (margin already drove the score, then
    # boosted reliability proportional to margin). Per Sensoy 2018
    # (EDL) and Lin 2024 (MMDG), uncertainty should DOWN-weight, not
    # up-weight. Reliability passes through; ReliabilityCalibrator
    # (P1.1) will eventually replace this with a proper rescale.
    reliability = float(reliability)

    # Degenerate-saturation override — collapse reliability to near 0
    # so the branch shows up in the panel for forensic visibility but
    # contributes effectively nothing to fusion math.
    if saturated_run:
        reliability = min(reliability, 0.05)
    # v9.8: same treatment when the auto-disable trigger fires. The
    # branch effectively stops voting in fusion; UI shows the disabled
    # state via meta["disabled_for_session"].
    if st.disabled_for_session:
        reliability = min(reliability, 0.02)

    # v9.8 — also clamp reliability when the score is close to the
    # user's personal baseline (no meaningful signal). Within +/- 0.05
    # of baseline, the branch is essentially saying "I have no opinion";
    # let the meta-fuser ignore it instead of letting a 0.6 baseline
    # leak through as a 0.6 weighted vote.
    if abs(raw_jump) < 0.05 and st.baseline_n >= 8:
        reliability = min(reliability, 0.20)

    # v9.8.1 2026-05-12 (Phase 2 #2): expose recovery state in meta so
    # the UI / forensic log can render "patch_depth recovering 2.3/5.0s"
    # when the auto-disable has flipped back off and the branch is
    # accumulating sustained-LIVE evidence to rejoin fusion.
    _recover_progress = 0.0
    if st.first_recover_t is not None and st.disabled_for_session:
        _recover_progress = round(now_t - st.first_recover_t, 1)

    return BranchScore(
        name="patch_depth",
        p_attack=max(0.0, min(1.0, p_attack)),
        reliability=round(max(0.0, min(1.0, reliability)), 3),
        latency_ms=(time.monotonic() - t0) * 1000.0,
        meta={"depth_l1": depth_l1, "status": 1.0,
              "raw_pre_calib": round(raw_clipped, 4),
              "saturated_run": float(saturated_run),
              "subject_baseline": round(baseline, 4),
              "raw_jump": round(raw_jump, 4),
              "disabled_for_session": float(st.disabled_for_session),
              "secs_high": (round(now_t - st.first_high_t, 1)
                            if st.first_high_t is not None else 0.0),
              "secs_recover": _recover_progress,
              "baseline_n": float(st.baseline_n),
              "pid": float(0 if pid is None else int(pid))},
        dominant_class=dominant,
    )
