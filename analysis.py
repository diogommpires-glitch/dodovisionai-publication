"""
DODO VISION AI — Analysis
FACS AUs, emotion fusion, blink/PERCLOS, gaze, head pose, micro-expressions, affect vector.

v8.7 (Step 3): the EMA smoothers are upgraded to constant-velocity Kalman
filters. The class name `EMA` is kept (and still works) for any external
caller, but the global ema_hr / ema_val / etc. instances are now Kalman1D.
This gives faster response to genuine trends, harder rejection of single-
frame spikes, and predict-on-missing so the UI never blanks during a
low-SNR frame.
"""
import math, time
import numpy as np
from collections import deque
import config as cfg
from kalman import (
    Kalman1D, KalmanDict,
    KALMAN_HR_Q, KALMAN_HR_R,
    KALMAN_AFFECT_Q, KALMAN_AFFECT_R,
    KALMAN_STRESS_Q, KALMAN_STRESS_R,
    KALMAN_HRV_Q, KALMAN_HRV_R,
    KALMAN_RESP_Q, KALMAN_RESP_R,
    KALMAN_EMO_Q, KALMAN_EMO_R,
)

# ══════════════════════════════════════════════
#  EMA Smoothing (legacy class kept for compatibility)
# ══════════════════════════════════════════════
class EMA:
    """Legacy fixed-alpha exponential moving average. Kept so any module
    that imports `from analysis import EMA` still works. The global vital
    smoothers below now use Kalman1D for better behavior."""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.val = None

    def update(self, v):
        if v is None:
            return self.val
        if self.val is None:
            self.val = v
            return v
        self.val = self.alpha * v + (1 - self.alpha) * self.val
        return self.val

    def get(self):
        return self.val

    def reset(self):
        self.val = None

# v8.7: Kalman replaces EMA for the global vital smoothers. The .update()
# and .get() / .reset() surface is identical so callers don't change.
ema_hr = Kalman1D(q=KALMAN_HR_Q, r=KALMAN_HR_R)
ema_val = Kalman1D(q=KALMAN_AFFECT_Q, r=KALMAN_AFFECT_R)
ema_aro = Kalman1D(q=KALMAN_AFFECT_Q, r=KALMAN_AFFECT_R)
ema_stress = Kalman1D(q=KALMAN_STRESS_Q, r=KALMAN_STRESS_R)
ema_hrv = Kalman1D(q=KALMAN_HRV_Q, r=KALMAN_HRV_R)
ema_resp = Kalman1D(q=KALMAN_RESP_Q, r=KALMAN_RESP_R)

# ══════════════════════════════════════════════
#  FACS Action Units from blendshapes
# ══════════════════════════════════════════════
FACS_MAP = {
    "AU1_inner_brow_raise": ["browInnerUp"],
    "AU2_outer_brow_raise": ["browOuterUpLeft", "browOuterUpRight"],
    "AU4_brow_lower": ["browDownLeft", "browDownRight"],
    "AU5_upper_lid_raise": ["eyeWideLeft", "eyeWideRight"],
    "AU6_cheek_raise": ["cheekSquintLeft", "cheekSquintRight"],
    "AU7_lid_tighten": ["eyeSquintLeft", "eyeSquintRight"],
    "AU9_nose_wrinkle": ["noseSneerLeft", "noseSneerRight"],
    "AU10_upper_lip_raise": ["mouthUpperUpLeft", "mouthUpperUpRight"],
    "AU12_lip_corner_pull": ["mouthSmileLeft", "mouthSmileRight"],
    "AU14_dimpler": ["mouthDimpleLeft", "mouthDimpleRight"],
    "AU15_lip_corner_depress": ["mouthFrownLeft", "mouthFrownRight"],
    "AU17_chin_raise": ["mouthShrinkLower", "mouthShrinkUpper"],
    "AU20_lip_stretch": ["mouthStretchLeft", "mouthStretchRight"],
    "AU22_lip_funnel": ["mouthFunnel"],
    "AU23_lip_tighten": ["mouthPressLeft", "mouthPressRight"],
    "AU24_lip_pressor": ["mouthPressLeft", "mouthPressRight"],
    "AU25_lips_part": ["jawOpen"],
    "AU26_jaw_drop": ["jawOpen"],
    "AU28_lip_suck": ["mouthRollLower", "mouthRollUpper"],
    "AU43_eyes_closed": ["eyeBlinkLeft", "eyeBlinkRight"],
    "AU45_blink": ["eyeBlinkLeft", "eyeBlinkRight"],
    "AU46_wink": ["eyeBlinkLeft", "eyeBlinkRight"],
}

def compute_aus(bs):
    """Map MediaPipe blendshape activations onto Action Unit (AU) scores
    via FACS_MAP. v9.8 perf: avoid the np.mean(list-of-floats) round-trip
    inside the FACS_MAP loop by averaging in pure Python. For the typical
    n=2 keys-per-AU this is ~3x faster (no numpy array allocation per
    call) and called per face per frame at 30 FPS."""
    aus = {}
    for au_name, blend_keys in FACS_MAP.items():
        n = len(blend_keys)
        if not n:
            aus[au_name] = 0
            continue
        s = 0.0
        for k in blend_keys:
            s += bs.get(k, 0)
        aus[au_name] = round(s / n, 3)
    return aus

# ══════════════════════════════════════════════
#  Blendshape-based emotion (FACS rules)
# ══════════════════════════════════════════════
def blendshape_emotion(bs):
    smile = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
    frown = (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
    brow_down = (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2
    brow_up = bs.get("browInnerUp", 0)
    brow_outer = (bs.get("browOuterUpLeft", 0) + bs.get("browOuterUpRight", 0)) / 2
    jaw = bs.get("jawOpen", 0)
    eye_wide = (bs.get("eyeWideLeft", 0) + bs.get("eyeWideRight", 0)) / 2
    eye_squint = (bs.get("eyeSquintLeft", 0) + bs.get("eyeSquintRight", 0)) / 2
    nose_sneer = (bs.get("noseSneerLeft", 0) + bs.get("noseSneerRight", 0)) / 2
    mouth_press = (bs.get("mouthPressLeft", 0) + bs.get("mouthPressRight", 0)) / 2
    cheek_sq = (bs.get("cheekSquintLeft", 0) + bs.get("cheekSquintRight", 0)) / 2
    lip_stretch = (bs.get("mouthStretchLeft", 0) + bs.get("mouthStretchRight", 0)) / 2
    mouth_pucker = bs.get("mouthPucker", 0)
    mouth_roll = (bs.get("mouthRollLower", 0) + bs.get("mouthRollUpper", 0)) / 2

    # Suppress eye_wide during blinks — MediaPipe briefly reports wide-eye
    # as eyelids reopen, which is a measurement artifact not genuine surprise
    blink_active = max(bs.get("eyeBlinkLeft", 0), bs.get("eyeBlinkRight", 0))
    eye_wide_adj = eye_wide * max(0.0, 1.0 - blink_active * 2.5)

    e = {}
    # Happy: smile + cheek raise + eye crinkle (Duchenne markers)
    e["happy"] = min(1, smile * 1.6 + cheek_sq * 0.6 + eye_squint * 0.4)
    # Sad: frown + inner brow raise + lip press + mouth down
    e["sad"] = min(1, frown * 1.3 + brow_up * 0.4 + mouth_press * 0.2 + mouth_pucker * 0.2)
    # Angry: brow lower + nose wrinkle + lip tighten + jaw clench
    e["angry"] = min(1, brow_down * 1.4 + nose_sneer * 0.9 + mouth_press * 0.5 + lip_stretch * 0.2)
    # Surprise: brow raise + wide eyes + jaw drop — all three must be above
    # resting-level thresholds to prevent normal eye opening / relaxed jaw
    # from false-triggering surprise
    _s_brow = max(0, (brow_up + brow_outer) * 0.5 - 0.08)
    _s_eyes = max(0, eye_wide_adj - 0.25)
    _s_jaw = max(0, jaw - 0.20)
    if _s_brow > 0.05 and _s_eyes > 0 and _s_jaw > 0:
        e["surprise"] = min(1, _s_brow * 1.2 + _s_eyes * 1.0 + _s_jaw * 0.8)
    else:
        e["surprise"] = min(1, (_s_brow + _s_eyes + _s_jaw) * 0.2)
    # Fear: wide eyes + brow raise + lip stretch — lip stretch (AU20) is the
    # key FACS differentiator from surprise; without it, penalize heavily
    _f_eyes = max(0, eye_wide_adj - 0.20)
    _f_lip = max(0, lip_stretch - 0.10)
    _f_raw = _f_eyes * 0.8 + max(0, brow_up - 0.1) * 0.6 + _f_lip * 1.2 + max(0, jaw - 0.15) * 0.2
    e["fear"] = min(1, _f_raw if _f_lip > 0.05 else _f_raw * 0.15)
    # Disgust: nose wrinkle + upper lip raise + frown
    e["disgust"] = min(1, nose_sneer * 1.6 + frown * 0.4 + brow_down * 0.2 + mouth_roll * 0.3)

    peak = max(e.values())
    e["neutral"] = max(0, 1.0 - peak * 1.2)

    return {k: round(v, 3) for k, v in e.items()}


# ── Per-frame emotion smoother for temporal stability ──
# v8.7: KalmanDict replaces the per-class EMA. Each emotion class is a
# Kalman1D with the recommended Q/R in kalman.py; the dict exposes the
# same shape (`{label: prob}`) so downstream consumers don't change.
_emo_kalman = KalmanDict(q=KALMAN_EMO_Q, r=KALMAN_EMO_R)
# Compatibility shim — anything still reading _emo_ema directly will see
# the current Kalman estimates.
class _EmoEmaProxy:
    def __getitem__(self, k): return _emo_kalman.get().get(k, 0.0)
    def get(self, k, default=0.0): return _emo_kalman.get().get(k, default)
    def __contains__(self, k): return k in _emo_kalman.get()
    def items(self): return _emo_kalman.get().items()
    def keys(self): return _emo_kalman.get().keys()
    def values(self): return _emo_kalman.get().values()
    def clear(self): _emo_kalman.reset()
_emo_ema = _EmoEmaProxy()
_EMO_ALPHA = 0.35  # legacy export

# v7.5: full EmotionSmoother (argmax + margin + 2-frame hysteresis) for a
# display-stable dominant emotion. The per-class EMA above is still used
# to fuse DF and blendshape — this wraps the fused distribution.
try:
    from emotion_smoother import EmotionSmoother as _EmotionSmoother
    _emotion_smoother = _EmotionSmoother()
except Exception:
    _emotion_smoother = None

# v7.5: gaze stability + proper EAR/PERCLOS wrapper for display consistency.
try:
    from gaze_stability import GazeStability, BlinkCounter
    _gaze_stab = GazeStability()
    _blink_cnt = BlinkCounter()
except Exception:
    _gaze_stab = None
    _blink_cnt = None

# v7.6: cognitive-state proxy estimator + CT2WS-style attention gate.
# These are inspired by DARPA AugCog + CT2WS and use ONLY webcam-derived
# signals. Not EEG, not neural recording.
try:
    from cognitive_state import estimator as _cog_estimator
    from attention_gate import gate as _attn_gate
except Exception:
    _cog_estimator = None
    _attn_gate = None

# v9.1 (optimization step 11): One-Euro landmark smoothing — adaptive
# low-pass that passes fast head turns through unchanged but kills the
# per-landmark shimmer that made overlays twitch at rest.
#
# v9.1.5: DISABLED for real-time tracking. Even with aggressive tuning
# (MIN_CUTOFF=5 Hz, BETA=0.2) the filter gives only ~50% pass-through on
# the first 33 ms frame of fast motion, which the user perceives as
# "face mask trailing the face". At 30+ fps capture the raw landmarks
# from MediaPipe are already sub-pixel stable; the filter was fighting
# a problem we no longer have. Kept the import guard so the module
# can be re-enabled by flipping `_LM_SMOOTH_ENABLED` to True if a use
# case for heavy temporal smoothing comes back.
_LM_SMOOTH_ENABLED = False
try:
    from landmark_smoother import smooth_landmarks as _smooth_lms_impl
    _smooth_lms = _smooth_lms_impl if _LM_SMOOTH_ENABLED else None
except Exception:
    _smooth_lms = None


def feed_cognitive_proxies(dilation=None, gaze=None, blink=None, rr_ms=None,
                           hr_bpm=None, event_marker=False):
    """Single entry-point the orchestrator (main.py) calls each frame.
    Signals are optional; missing values are ignored. Matches AugCog's
    'multiple sensors, graceful degradation' design pattern.
    """
    if _cog_estimator is None:
        return
    est = _cog_estimator()
    gate = _attn_gate() if _attn_gate else None
    t = None
    if dilation is not None:
        est.push_pupil(dilation, t)
        if gate: gate.push_pupil(dilation, t)
    if gaze and gaze.get("horizontal") is not None:
        prev_saccades = est._microsaccade_count
        est.push_gaze(gaze.get("horizontal"), gaze.get("vertical"), t)
        if gate and est._microsaccade_count > prev_saccades:
            gate.push_saccade(t)
    if blink and blink.get("blinked_this_frame"):
        est.push_blink(True, t)
    if rr_ms is not None:
        est.push_rr_ms(rr_ms)
    if hr_bpm is not None and gate:
        gate.push_hr(hr_bpm, t)
    if event_marker:
        est.push_event(t)
        if gate: gate.submit_event(t)


def stable_dominant_emotion(fused_probs):
    """Given fused emotion scores, return (label, confidence) stabilized."""
    if _emotion_smoother is None or not fused_probs:
        if not fused_probs:
            return "neutral", 0.0
        top = max(fused_probs.items(), key=lambda kv: kv[1])
        return top[0], float(top[1])
    label, conf, _ = _emotion_smoother.push(fused_probs)
    return label, round(conf, 3)


def stable_gaze(gaze_dict):
    """Apply EMA + steady detection to horizontal/vertical gaze."""
    if _gaze_stab is None or not gaze_dict:
        return gaze_dict or {}
    steady, sh, sv = _gaze_stab.push(
        float(gaze_dict.get("horizontal", 0.0) or 0.0),
        float(gaze_dict.get("vertical", 0.0) or 0.0),
    )
    out = dict(gaze_dict)
    out["horizontal"] = sh
    out["vertical"] = sv
    out["steady"] = bool(steady)
    return out

def fuse_emotions(df_scores, bs_scores, df_weight=None):
    """Fuse DeepFace/HSEmotion and blendshape emotions with Kalman smoothing
    (v8.7). Behaves like a per-class smoother with adaptive gain — the same
    `{label: prob}` shape as before, just steadier.

    v9.1 (optimization step 18): dynamic fusion weight. When blendshape
    confidence is low (faint activations across all classes), the FACS
    rules are unreliable — we fall back toward the CNN. When blendshapes
    are strong, we trust them more. This fixes the false-neutral bias
    when landmarks are stable but the CNN is uncertain, and also
    prevents FACS noise from dominating when the face is partially
    occluded or near-frontal.
    """
    if df_weight is None:
        df_weight = cfg.DEEPFACE_WEIGHT
    # v9.1: compute blendshape "confidence" as peak non-neutral score.
    bs_peak = 0.0
    for k, v in bs_scores.items():
        if k != "neutral" and v > bs_peak:
            bs_peak = v
    # When bs_peak < 0.3, raise DF weight by up to +0.10 (cap at 0.85).
    # When bs_peak > 0.6, lower DF weight by up to −0.10 (floor at 0.45).
    if bs_peak < 0.30:
        df_weight = min(0.85, df_weight + (0.30 - bs_peak) * 0.33)
    elif bs_peak > 0.60:
        df_weight = max(0.45, df_weight - (bs_peak - 0.60) * 0.25)
    raw = {}
    all_keys = set(list(df_scores.keys()) + list(bs_scores.keys()))
    for k in all_keys:
        dv = df_scores.get(k, 0)
        bv = bs_scores.get(k, 0)
        raw[k] = dv * df_weight + bv * (1 - df_weight)
    smoothed = _emo_kalman.update(raw)
    return {k: round(v, 3) for k, v in smoothed.items()}

# ══════════════════════════════════════════════
#  Head Pose (pitch, yaw, roll)
# ══════════════════════════════════════════════
def head_pose(fl, aspect: float = 16.0 / 9.0):
    # MediaPipe normalizes x by image width and y by image height; z is per
    # MediaPipe docs on the same scale as x. Using (y - y) and (x - x) in
    # atan2 directly inflates roll/pitch by the image aspect ratio (on a
    # 16:9 webcam a real 17° roll reads as ~30°). Rescale y into x-units by
    # dividing by `aspect` (= width/height) so atan2 returns pixel-space
    # angles. aspect defaults to 16:9 for modern webcams; callers with a
    # different resolution can override.
    chin = fl[152]; left_eye = fl[33]; right_eye = fl[263]; forehead = fl[10]
    eye_dx = right_eye.x - left_eye.x
    eye_dz = right_eye.z - left_eye.z
    yaw = math.degrees(math.atan2(eye_dz, eye_dx + 1e-6))
    vert_dy = (chin.y - forehead.y) / aspect
    vert_dz = chin.z - forehead.z
    pitch = math.degrees(math.atan2(vert_dz, vert_dy + 1e-6))
    roll_dy = (right_eye.y - left_eye.y) / aspect
    roll_dx = right_eye.x - left_eye.x
    roll = math.degrees(math.atan2(roll_dy, roll_dx + 1e-6))
    return {"pitch": round(pitch, 1), "yaw": round(yaw, 1), "roll": round(roll, 1)}

# ══════════════════════════════════════════════
#  Eye Gaze from iris (landmarks 468-477)
# ══════════════════════════════════════════════
def eye_gaze(fl):
    try:
        l_inner, l_outer, l_iris = fl[133], fl[33], fl[468]
        r_inner, r_outer, r_iris = fl[362], fl[263], fl[473]
        l_range = l_inner.x - l_outer.x
        l_pos = (l_iris.x - l_outer.x) / (l_range + 1e-6)
        r_range = r_inner.x - r_outer.x
        r_pos = (r_iris.x - r_outer.x) / (r_range + 1e-6)
        h_gaze = round(((l_pos + r_pos) / 2 - 0.5) * 2, 2)
        l_top, l_bot = fl[159], fl[145]
        l_v_range = l_bot.y - l_top.y
        l_v_pos = (l_iris.y - l_top.y) / (l_v_range + 1e-6)
        r_top, r_bot = fl[386], fl[374]
        r_v_range = r_bot.y - r_top.y
        r_v_pos = (r_iris.y - r_top.y) / (r_v_range + 1e-6)
        v_gaze = round(((l_v_pos + r_v_pos) / 2 - 0.5) * 2, 2)
        return {"horizontal": h_gaze, "vertical": v_gaze}
    except Exception:
        return {"horizontal": 0, "vertical": 0}

# ══════════════════════════════════════════════
#  Iris tracking + pupil dilation
# ══════════════════════════════════════════════
def get_iris_data(fl):
    try:
        l_iris = {"x": round(fl[468].x, 4), "y": round(fl[468].y, 4)}
        r_iris = {"x": round(fl[473].x, 4), "y": round(fl[473].y, 4)}
        l_ring = [{"x": round(fl[i].x, 4), "y": round(fl[i].y, 4)} for i in [468, 469, 470, 471, 472]]
        r_ring = [{"x": round(fl[i].x, 4), "y": round(fl[i].y, 4)} for i in [473, 474, 475, 476, 477]]
        l_eye = [{"x": round(fl[i].x, 4), "y": round(fl[i].y, 4)} for i in
                 [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]]
        r_eye = [{"x": round(fl[i].x, 4), "y": round(fl[i].y, 4)} for i in
                 [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]]
        l_eye_w = abs(fl[33].x - fl[133].x)
        l_iris_r = abs(fl[469].x - fl[471].x) / 2
        r_eye_w = abs(fl[362].x - fl[263].x)
        r_iris_r = abs(fl[474].x - fl[476].x) / 2
        l_dilation = round(l_iris_r / (l_eye_w + 1e-6), 3)
        r_dilation = round(r_iris_r / (r_eye_w + 1e-6), 3)
        return {
            "l_iris": l_iris, "r_iris": r_iris,
            "l_ring": l_ring, "r_ring": r_ring,
            "l_eye": l_eye, "r_eye": r_eye,
            "l_dilation": l_dilation, "r_dilation": r_dilation,
            "avg_dilation": round((l_dilation + r_dilation) / 2, 3),
        }
    except Exception:
        return None

# ══════════════════════════════════════════════
#  Blink Detection + PERCLOS
# ══════════════════════════════════════════════
blink_history = deque(maxlen=cfg.BLINK_HISTORY_SIZE)
perclos_history = deque(maxlen=cfg.PERCLOS_WINDOW)
bs_history = deque(maxlen=30)
prev_blink_state = {}

def _ear_from_landmarks(fl):
    """Eye Aspect Ratio computed from MediaPipe FaceMesh eyelid points.

    Returns None if landmarks aren't available or computation fails.
    Open eye ~ 0.30, closed eye ~ 0.10. Robust to glasses where the
    blendshape `eyeBlinkLeft/Right` classifier gets confused by lens
    edges / reflections.

    v9.8: vectorized via numpy (matches `dual_camera._ear`). 4 pairwise
    distances computed in a single np.linalg.norm call. ~3x faster than
    the Python sqrt loop, runs in the per-face hot path so the saving
    compounds at 30-60 fps.
    """
    try:
        if fl is None or len(fl) < 387:
            return None
        # Stack 8 landmarks into a [4, 2, 2] array
        # axis 0 = pair (left_v, left_h, right_v, right_h)
        # axis 1 = endpoint (a, b)
        # axis 2 = (x, y)
        pts = np.array([
            [(fl[159].x, fl[159].y), (fl[145].x, fl[145].y)],   # left vertical
            [(fl[33].x,  fl[33].y),  (fl[133].x, fl[133].y)],    # left horizontal
            [(fl[386].x, fl[386].y), (fl[374].x, fl[374].y)],    # right vertical
            [(fl[362].x, fl[362].y), (fl[263].x, fl[263].y)],    # right horizontal
        ], dtype=np.float32)
        d = np.linalg.norm(pts[:, 0] - pts[:, 1], axis=1)
        # d = [l_v, l_h, r_v, r_h]
        if d[1] <= 1e-6 or d[3] <= 1e-6:
            return None
        return float(((d[0] / d[1]) + (d[2] / d[3])) / 2)
    except Exception:
        return None


# v9.7.4 full rewrite: per-face subject-adaptive blink detector.
# Previous implementation used absolute blink_val thresholds with an
# EAR override that engaged within a narrow band — failed for
# glasses-wearers because their EAR sticks at ~0.25 (landmarks on
# the lens frame, not the eyelid). The new detector uses RELATIVE
# drops from a rolling personal baseline so the absolute value
# never matters. See `blink_detector.BlinkDetector` for the math.
from blink_detector import BlinkDetector
_per_face_blink: "dict[int, BlinkDetector]" = {}


def detect_blink(bs, face_idx, fl=None):
    """Blink + PERCLOS + drowsiness via the subject-adaptive
    `BlinkDetector`. One detector instance per face_idx so multi-face
    scenes don't cross-contaminate baselines.

    Returns the same dict shape as the previous implementation — all
    upstream callers keep working without changes."""
    blink_val = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
    ear = _ear_from_landmarks(fl) if fl is not None else None
    det = _per_face_blink.get(face_idx)
    if det is None:
        det = BlinkDetector()
        _per_face_blink[face_idx] = det
    out = det.update(ear=ear, bs_blink=blink_val, t=time.time())
    # Mirror to legacy histories so callers that read these globals
    # (forensic_export etc.) keep seeing data.
    if out["blinked"]:
        blink_history.append(time.time())
    perclos_history.append(out["_state"] in (BlinkDetector.CLOSING,
                                             BlinkDetector.CLOSED,
                                             BlinkDetector.OPENING))
    return {
        "blinked":       out["blinked"],
        "blink_rate_pm": out["blink_rate_pm"],
        "eye_openness":  out["eye_openness"],
        "perclos":       out["perclos"],
        "drowsiness":    out["drowsiness"],
    }

# ══════════════════════════════════════════════
#  Micro-Expression Detection
# ══════════════════════════════════════════════
def detect_micro_expressions(bs):
    bs_history.append(bs)
    if len(bs_history) < cfg.MICRO_WINDOW:
        return {"detected": False}
    prev = bs_history[-cfg.MICRO_WINDOW]
    key_bs = ["mouthSmileLeft", "mouthSmileRight", "browDownLeft", "browDownRight",
              "eyeWideLeft", "eyeWideRight", "noseSneerLeft", "noseSneerRight"]
    max_delta = 0
    max_key = ""
    for k in key_bs:
        delta = abs(bs.get(k, 0) - prev.get(k, 0))
        if delta > max_delta:
            max_delta = delta
            max_key = k
    if max_delta > cfg.MICRO_VELOCITY_THRESH:
        return {"detected": True, "muscle": max_key, "intensity": round(max_delta, 3)}
    return {"detected": False}

# ══════════════════════════════════════════════
#  Age smoothing (increased window + trimmed mean)
# ══════════════════════════════════════════════
age_history = {}

def smooth_age(face_idx, raw_age):
    if face_idx not in age_history:
        age_history[face_idx] = deque(maxlen=cfg.AGE_SMOOTH_WINDOW)
    age_history[face_idx].append(raw_age)
    arr = list(age_history[face_idx])
    if len(arr) >= 5:
        # Trimmed mean: remove top/bottom 15% to eliminate outliers
        arr_sorted = sorted(arr)
        trim = max(1, len(arr_sorted) // 7)
        trimmed = arr_sorted[trim:-trim] if trim > 0 and len(arr_sorted) > 2 * trim else arr_sorted
        return int(round(np.mean(trimmed)))
    return int(round(np.median(arr)))

# ══════════════════════════════════════════════
#  Race/Ethnicity smoothing (majority voting over N frames)
# ══════════════════════════════════════════════
race_history = {}
RACE_SMOOTH_WINDOW = 60

def smooth_race(face_idx, race_scores):
    """Smooth ethnicity prediction using accumulated score voting.
    Instead of picking dominant race per-frame, accumulate scores across
    frames and pick the highest total. Much more stable."""
    if face_idx not in race_history:
        race_history[face_idx] = {"scores": {}, "count": 0}
    rh = race_history[face_idx]
    rh["count"] += 1
    # Accumulate scores
    for race, score in race_scores.items():
        rh["scores"][race] = rh["scores"].get(race, 0) + score
    # Decay old data (slower decay = more stable)
    if rh["count"] > RACE_SMOOTH_WINDOW:
        decay = 0.97
        for k in rh["scores"]:
            rh["scores"][k] *= decay
    # Pick dominant from accumulated scores
    if rh["scores"]:
        total = sum(rh["scores"].values())
        dominant = max(rh["scores"], key=rh["scores"].get)
        confidence = rh["scores"][dominant] / (total + 1e-6)
        normalized = {k: round(v / (total + 1e-6), 3) for k, v in rh["scores"].items()}
        return {"dominant": dominant, "confidence": round(confidence, 3), "scores": normalized}
    return {"dominant": "?", "confidence": 0, "scores": {}}

# ══════════════════════════════════════════════
#  Build face data (fast + full modes)
# ══════════════════════════════════════════════
def build_face_fast(fi, fl, bs_dict):
    # v9.1: One-Euro smooth the landmarks before derived features read them.
    # head_pose, eye_gaze, and get_iris_data all index into fl, so smoothing
    # here propagates to all downstream metrics without separate tuning.
    if _smooth_lms is not None:
        try:
            fl = _smooth_lms(fi, fl)
        except Exception:
            pass
    # v9.2 (plan step 21): 3-decimal precision on fast frames instead of 4.
    # 3 decimals = 0.001 = 0.6 px on a 640px canvas — sub-pixel already,
    # and the visual face mask is drawn with 1.4 px GL points so extra
    # precision is invisible. Shaves ~15% off JSON payload size per frame.
    lms = [[round(lm.x, 3), round(lm.y, 3)] for lm in fl]
    aus = compute_aus(bs_dict)
    pose = head_pose(fl)
    gaze = eye_gaze(fl)
    blink = detect_blink(bs_dict, fi, fl=fl)  # v9.7.4: EAR fallback for glasses
    micro = detect_micro_expressions(bs_dict) if fi == 0 else {"detected": False}
    bs_emo = blendshape_emotion(bs_dict)
    iris = get_iris_data(fl)
    return {
        "i": fi, "lm": lms, "aus": aus, "pose": pose, "gaze": gaze,
        "blink": blink, "micro": micro, "bs_emotion": bs_emo, "iris": iris,
    }

def build_face_full(fi, fl, bs_dict):
    if _smooth_lms is not None:
        try:
            fl = _smooth_lms(fi, fl)
        except Exception:
            pass
    lms = [{"x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4)} for lm in fl]
    aus = compute_aus(bs_dict)
    pose = head_pose(fl)
    gaze = eye_gaze(fl)
    blink = detect_blink(bs_dict, fi, fl=fl)  # v9.7.4: EAR fallback for glasses
    micro = detect_micro_expressions(bs_dict)
    bs_emo = blendshape_emotion(bs_dict)
    iris = get_iris_data(fl)
    return {
        "i": fi, "lm": lms, "aus": aus, "pose": pose, "gaze": gaze,
        "blink": blink, "micro": micro, "bs_emotion": bs_emo, "iris": iris,
    }

# ══════════════════════════════════════════════
#  Affect Vector (global state)
# ══════════════════════════════════════════════
affect_vector = {
    "valence": 0.0, "arousal": 0.0, "dominant_emotion": "neutral", "emotion_scores": {},
    "heart_rate_bpm": None, "confirmed_bpm": None, "hrv_rmssd": None,
    "resp_rate": None, "stress_index": None, "blink_rate": None,
    "perclos": 0.0, "drowsiness": "alert",
    "head_pose": None, "gaze": None, "micro_expression": None,
    "pupil_dilation": None,
    "age": None, "gender": None, "race": None,
    # Radar vitals (from MR60BHA2 — Radar A)
    "radar_hr": None, "radar_breath": None, "radar_presence": None, "radar_distance": None,
    "radar_people_count": None,  # v7.7: people count
    "radar_breath_phase": None, "radar_heart_phase": None,  # v7.7 live EKG/breath
    # Radar B (dual radar)
    "radar_b_hr": None, "radar_b_breath": None, "radar_b_distance": None,
    "radar_b_people_count": None,  # v7.7: people count for radar B
    # v7.9: provenance tags so UI can show (cam) / (radar) next to values
    # when the radar hardware is not attached and MediaPipe is filling in.
    "radar_distance_source": None, "radar_people_source": None,
    "signal_quality": {"face": 0, "rppg": 0}, "confidence": 0.0, "timestamp": 0,
    "identity": None, "oxi_hr": None, "oxi_spo2": None, "oxi_signal": None,
    # v9.1 (optimization step 19): end-to-end oximeter state for UI consumption.
    # UI staleness badge compares oxi_last_update against now() to decide if
    # the oximeter panel should flash red. oxi_connected tracks HID/serial
    # link state so the UI shows Searching… / Connected correctly even when
    # a finger isn't seated yet (heart_rate will be None but link is live).
    "oxi_connected": False, "oxi_last_update": None, "oxi_mode": None,
    "oxi_age_s": None, "oxi_motion_gated": None, "oxi_pleth": None,
    "hr_source": None, "rppg_quality": None,
}

def update_affect_vector(df, hr, hrv, stress, quality, resp, pose, gaze, blink, micro, fused_emo, iris=None, radar_data=None):
    if fused_emo:
        dom, _conf = stable_dominant_emotion(fused_emo)
        v_s, a_s, w_s = 0, 0, 0
        for e, s in fused_emo.items():
            ev, ea = cfg.EMOTION_VA.get(e, (0, 0))
            v_s += ev * s; a_s += ea * s; w_s += s
        if w_s > 0:
            affect_vector["valence"] = round(ema_val.update(v_s / w_s), 3)
            affect_vector["arousal"] = round(ema_aro.update(a_s / w_s), 3)
        affect_vector["dominant_emotion"] = dom
        affect_vector["emotion_scores"] = fused_emo
    if df:
        affect_vector["age"] = df.get("age", {}).get("value") if isinstance(df.get("age"), dict) else df.get("raw_age")
        # 2026-04-25: also expose age_range so the UI fallback path
        # (`av?.age_range`) resolves when face['df'] hasn't propagated into
        # the WS payload yet (e.g. timing race during DeepFace cache miss).
        # Backend stores age as {"value": int, "range": "20-29"} in main.py:414.
        if isinstance(df.get("age"), dict):
            affect_vector["age_range"] = df.get("age", {}).get("range")
        affect_vector["gender"] = df.get("gender", {}).get("dominant")
        affect_vector["race"] = df.get("race", {}).get("dominant")
        # 2026-04-25: also expose the per-class score dicts so the UI can
        # render bar graphs alongside the dominant string. Without these
        # the gB / rB blocks in ui.py:718 stay empty.
        affect_vector["gender_scores"] = df.get("gender", {}).get("scores") or {}
        affect_vector["race_scores"]   = df.get("race", {}).get("scores")   or {}
        # Mirror the dominant strings under "ethnicity" alias for the
        # 2026-04-25 UI rename (Race -> Ethnicity).
        affect_vector["ethnicity"]        = affect_vector["race"]
        affect_vector["ethnicity_scores"] = affect_vector["race_scores"]
        affect_vector["signal_quality"]["face"] = round(max(fused_emo.values()) if fused_emo else 0, 2)
    if hr:
        smoothed_hr = ema_hr.update(hr)
        affect_vector["heart_rate_bpm"] = round(smoothed_hr, 1)
        if smoothed_hr > 90:
            affect_vector["arousal"] = min(1.0, affect_vector["arousal"] + 0.05)
    affect_vector["confirmed_bpm"] = None  # set by rppg module
    # v7.7: extra display-layer smoothing so UI text doesn't jitter even
    # during a single transient None from estimate_hrv().
    if hrv is not None:
        smoothed = ema_hrv.update(hrv)
        affect_vector["hrv_rmssd"] = round(smoothed, 1) if smoothed is not None else hrv
    # else: leave previous value on the affect_vector; do NOT blank.
    affect_vector["stress_index"] = round(ema_stress.update(stress), 2) if stress else affect_vector.get("stress_index")
    # v7.9: display-layer EMA on top of rppg's smoothed resp so a single
    # low-SNR frame returning None doesn't blank the UI. Previous value
    # is kept in affect_vector until a new accepted value arrives.
    if resp is not None:
        smoothed_resp = ema_resp.update(resp)
        affect_vector["resp_rate"] = round(smoothed_resp, 1) if smoothed_resp is not None else resp
    affect_vector["blink_rate"] = blink.get("blink_rate_pm") if blink else None
    affect_vector["perclos"] = blink.get("perclos", 0) if blink else 0
    affect_vector["drowsiness"] = blink.get("drowsiness", "alert") if blink else "alert"
    # v9.8.1 (2026-05-12): propagate the edge-triggered `blinked` boolean
    # so pad/branches/dynamics_branch.feed_blink() gets called. RCA: until
    # now `av["blinked"]` was never set, so `_pad_run_once_unlocked` saw
    # bool(None)==False every tick, dynamics_branch never recorded any
    # blink event, its blink_rate_per_min came back None after 15s, the
    # branch fell into the "no blinks after 15 s = suspicious" gate
    # (blink feature pinned to 0.1), and the fused PAD verdict came out
    # SPOOF with attack_class=PRINT every frame for a real subject.
    affect_vector["blinked"] = bool(blink.get("blinked")) if blink else False
    affect_vector["eye_openness"] = blink.get("eye_openness") if blink else None
    affect_vector["head_pose"] = pose
    affect_vector["gaze"] = gaze
    affect_vector["micro_expression"] = micro if micro and micro.get("detected") else None
    if iris:
        affect_vector["pupil_dilation"] = iris.get("avg_dilation")
    affect_vector["signal_quality"]["rppg"] = quality
    affect_vector["confidence"] = round(
        affect_vector["signal_quality"].get("face", 0) * 0.6 +
        affect_vector["signal_quality"].get("rppg", 0) * 0.4, 2
    )
    affect_vector["timestamp"] = time.time()
    # Merge radar data if available
    if radar_data:
        affect_vector["radar_hr"] = radar_data.get("heart_rate")
        affect_vector["radar_breath"] = radar_data.get("breath_rate")
        affect_vector["radar_presence"] = radar_data.get("presence")
        affect_vector["radar_distance"] = radar_data.get("distance")
        affect_vector["radar_people_count"] = radar_data.get("people_count")  # v7.7
        # v7.7: pass raw breath phase through so the UI breath animation
        # can mirror actual chest-wall motion rather than a synthesized
        # sinusoid at the estimated rate.
        affect_vector["radar_breath_phase"] = radar_data.get("breath_phase")
        affect_vector["radar_heart_phase"] = radar_data.get("heart_phase")
    # v8.7 (Step 4): attach trust gauge to global affect_vector too
    try:
        from trust_gauge import score_trust as _gv_score_trust
        affect_vector["trust"] = _gv_score_trust(affect_vector)
    except Exception:
        pass

def reset_all():
    """Clear all analysis buffers."""
    blink_history.clear()
    perclos_history.clear()
    bs_history.clear()
    prev_blink_state.clear()
    age_history.clear()
    race_history.clear()
    _emo_kalman.reset()
    ema_hr.reset()
    ema_val.reset()
    ema_aro.reset()
    ema_stress.reset()
    ema_hrv.reset()
    ema_resp.reset()

# ══════════════════════════════════════════════
#  Deception-relevant facial metrics (research-backed)
# ══════════════════════════════════════════════
def detect_duchenne_smile(bs):
    """Detect genuine (Duchenne) vs fake smile.
    Duchenne = AU6 (cheek raise) + AU12 (lip corner pull)
    Fake = AU12 only, no AU6.
    Source: Ekman et al. (1991), Frank & Ekman (1997)
    Returns: {"smiling": bool, "duchenne": bool, "fake_smile": bool, "au6": float, "au12": float}
    """
    au12 = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
    au6 = (bs.get("cheekSquintLeft", 0) + bs.get("cheekSquintRight", 0)) / 2
    smiling = au12 > 0.15
    duchenne = smiling and au6 > 0.1
    fake = smiling and au6 < 0.05
    return {"smiling": smiling, "duchenne": duchenne, "fake_smile": fake,
            "au6": round(au6, 3), "au12": round(au12, 3)}

def detect_facial_asymmetry(bs):
    """Measure left-right facial asymmetry. Higher = more likely faked expression.
    Genuine emotions produce symmetric expressions; deliberate ones are asymmetric.
    Source: Ekman, Hager & Friesen (1981)
    """
    pairs = [
        ("mouthSmileLeft", "mouthSmileRight"),
        ("browDownLeft", "browDownRight"),
        ("eyeSquintLeft", "eyeSquintRight"),
        ("cheekSquintLeft", "cheekSquintRight"),
        ("mouthFrownLeft", "mouthFrownRight"),
        ("eyeWideLeft", "eyeWideRight"),
        ("noseSneerLeft", "noseSneerRight"),
    ]
    asym_scores = []
    for left_key, right_key in pairs:
        l = bs.get(left_key, 0)
        r = bs.get(right_key, 0)
        if l + r > 0.05:  # only count if there's meaningful activation
            asym = abs(l - r) / (max(l, r) + 1e-6)
            asym_scores.append(asym)
    if not asym_scores:
        return {"asymmetry": 0.0, "pairs_checked": 0}
    return {"asymmetry": round(float(np.mean(asym_scores)), 3),
            "max_asymmetry": round(float(max(asym_scores)), 3),
            "pairs_checked": len(asym_scores)}

def detect_fear_leakage(bs):
    """Detect fear expression pattern (deception leakage indicator).
    Fear = AU1 (inner brow raise) + AU2 (outer brow raise) + AU4 (brow lower) + AU5 (upper lid raise)
    Source: Su & Levine (2016), PMC8217652
    Key AUs for deception: AU1, AU2, AU4, AU12, AU15, AU45
    """
    au1 = bs.get("browInnerUp", 0)
    au2 = (bs.get("browOuterUpLeft", 0) + bs.get("browOuterUpRight", 0)) / 2
    au4 = (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2
    au5 = (bs.get("eyeWideLeft", 0) + bs.get("eyeWideRight", 0)) / 2
    au15 = (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
    # Fear pattern: raised brows + wide eyes
    fear_score = (au1 * 0.3 + au2 * 0.2 + au5 * 0.3 + au15 * 0.2)
    # Deception AU pattern (Su & Levine 2016)
    deception_aus = au1 * 0.2 + au2 * 0.15 + au4 * 0.25 + au15 * 0.2 + (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2 * 0.2
    return {
        "fear_score": round(fear_score, 3),
        "deception_au_score": round(deception_aus, 3),
        "au1": round(au1, 3), "au2": round(au2, 3),
        "au4": round(au4, 3), "au5": round(au5, 3),
    }
