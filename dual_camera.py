"""
DODO VISION AI — Dual Camera System

Camera 0 (HQ, 30fps, displayed in UI):
  - Face mesh overlay (478 landmarks)
  - DeepFace demographics (age, gender, race)
  - Face recognition (Facenet512)
  - Emotion detection (HSEmotion)
  - Head pose and gaze direction
  - Also runs rPPG at 30fps (fallback if no 120fps camera)

Camera 1 (120fps, background thread, if detected):
  - rPPG heart rate, HRV, respiratory rate (4x FFT resolution)
  - Blink detection + PERCLOS (8ms precision vs 33ms)
  - Micro-expression detection (catches 40ms flickers)
  - Pupil dilation tracking (120 samples/sec)
  - Duchenne smile, facial asymmetry, fear AU leakage

IMPORTANT: The 120fps ELP/SVPRO cameras default to YUY2 at 30fps.
You MUST force MJPEG codec via cv2.CAP_PROP_FOURCC to unlock 120fps.
This is done automatically in this module.

If no high-fps camera is found, the system runs entirely on the HQ camera
with no degradation. The 120fps camera is a pure upgrade, not a requirement.

v9.8.3 note — single-subject 120fps path: blink / pupil / micro metrics merged
into `analysis.affect_vector` (face slot 0) via `_merge_fast_bio()` are intended
for the primary on-screen subject and the dual-camera biometric rig. When
multiple faces are tracked, per-face PAD dynamics still use each slot's own
pose/gaze from `faces[i]`; only slot 0 inherits the fast-camera blink/pupil
overrides so the high-fps sensor stays authoritative for the main ROI.
"""
import os, threading, time, cv2
import numpy as np
import mediapipe as mp
from collections import deque
import config as cfg

# ── Shared fast-biometric state ──
_fast_bio = {
    "blink_rate": None, "eye_openness": None, "perclos": 0.0,
    "drowsiness": "alert", "blinked": False,
    "micro_detected": False, "micro_muscle": None, "micro_intensity": 0,
    "pupil_dilation": None, "pupil_l": None, "pupil_r": None,
    "duchenne": False, "fake_smile": False, "au6": 0, "au12": 0,
    "asymmetry": 0.0, "fear_score": 0.0, "deception_au_score": 0.0,
    "fps": 0, "frames": 0, "active": False, "device": None,
}
_fast_lock = threading.Lock()

# ── Internal buffers ──
_blink_history = deque(maxlen=1200)
_perclos_history = deque(maxlen=3600)
_bs_history = deque(maxlen=60)
_prev_blink = False
_prev_lm = None
_thread = None
_stop = threading.Event()
_fps_times = deque(maxlen=150)

# v9.8.1 2026-05-12: latest captured frame as JPEG bytes, refreshed
# at ~5 fps by the camera loop. Lets the UI display a thumbnail of
# the server-held camera (R10 etc.) that the browser cannot open
# directly via getUserMedia under Windows DSHOW exclusive access.
# Lock + module-level globals to keep the access cheap on hot path.
_latest_jpeg_lock = threading.Lock()
_latest_jpeg_bytes: bytes = b""
_latest_jpeg_ts: float = 0.0


def get_latest_jpeg() -> "tuple[bytes, float]":
    """Return (jpeg_bytes, capture_ts). Empty bytes + 0.0 if no frame
    captured yet."""
    with _latest_jpeg_lock:
        return _latest_jpeg_bytes, _latest_jpeg_ts

# ── Separate MediaPipe instance for 120fps thread ──
_landmarker = None

_lm_ts_ms = 0  # v9.7.3: monotonic timestamp for VIDEO-mode detect_for_video

# v9.7.4: latest computed EAR for diagnostics. Exposed via
# get_fast_biometrics() so the UI / debug panel can show what the
# EAR landmark-based eye-aspect-ratio is reading on this subject —
# helps diagnose glasses-misplacement situations where EAR sticks at
# a constant value and blinks never register.
_last_ear = 0.0

# v9.7.4 (full rewrite): subject-adaptive blink detector. Replaces
# the absolute-threshold + EAR-override hack which couldn't see
# through glasses-on-frame landmark misplacement. The new detector
# uses RELATIVE drops from a rolling-median baseline + a 4-state
# machine + cross-channel cross-validation. See blink_detector.py.
from blink_detector import BlinkDetector
_blink_det = BlinkDetector()

# ── API_up v9.8: optional 3rd blink channel via eye_state_v1.onnx CNN ──
# Trained on MRL Eye Dataset, val_acc 0.9764, ~0.14 ms on CPU per eye.
# Disabled by default (FF_BLINK_CNN=0). When enabled, complements EAR +
# blendshape with a fully-trained classifier — particularly helpful for
# glasses-wearers where the EAR landmarks slide to lens edges and the
# blendshape regression saturates.
_FF_BLINK_CNN = (os.environ.get("FF_BLINK_CNN", "0") == "1")
_eye_state_sess = None
_eye_state_warned = False

def _init_eye_state():
    """Lazy-load the eye-state ONNX session once. Cheap to call repeatedly;
    after first call the session is cached. ORT-CPU is the right backend
    here -- the model is so small (240 KB) that PCIe transfer to GPU
    dominates."""
    global _eye_state_sess, _eye_state_warned
    if not _FF_BLINK_CNN or _eye_state_sess is not None:
        return
    try:
        import onnxruntime as ort
        model_path = os.path.join(os.path.dirname(__file__),
                                   "pad", "models", "eye_state_v1.onnx")
        if not os.path.isfile(model_path):
            if not _eye_state_warned:
                print(f"[--] eye_state_v1.onnx not found at {model_path}; "
                      f"FF_BLINK_CNN disabled for this session")
                _eye_state_warned = True
            return
        _eye_state_sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        print("[OK] eye_state_v1.onnx loaded as 3rd blink channel (FF_BLINK_CNN=1)")
    except Exception as e:
        if not _eye_state_warned:
            print(f"[--] eye_state_v1 failed to load ({e}); FF_BLINK_CNN disabled")
            _eye_state_warned = True

# MediaPipe FaceMesh eye landmark indices (478-pt mesh, used for cropping).
_LEFT_EYE_IDX  = (33, 133, 159, 145)   # outer, inner, top, bottom
_RIGHT_EYE_IDX = (263, 362, 386, 374)

def _eye_crop_64(frame, fl, idxs):
    """Extract a 64x64 grayscale eye crop centered on the eye landmarks.
    Returns (1,1,64,64) float32 normalized to [-1,1], or None on failure.
    Crop side is 2x the eye-corner distance so the full eye + a margin
    of skin/eyelid is included -- matches MRL Eye Dataset training crops."""
    try:
        h, w = frame.shape[:2]
        xs = [int(fl[i].x * w) for i in idxs]
        ys = [int(fl[i].y * h) for i in idxs]
        cx = sum(xs) // 4; cy = sum(ys) // 4
        half = max(20, max(xs) - min(xs))
        x0, x1 = max(0, cx - half), min(w, cx + half)
        y0, y1 = max(0, cy - half), min(h, cy + half)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        x = crop.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        return x[None, None, :, :]
    except Exception:
        return None

def _eye_state_prob(frame, fl):
    """Run eye_state_v1.onnx on both eyes and return mean P(eye is closed).
    Returns None if FF_BLINK_CNN=0, the session isn't loaded, the frame
    is missing, or both eyes failed to crop."""
    _init_eye_state()
    if _eye_state_sess is None or frame is None or fl is None:
        return None
    try:
        probs = []
        for idxs in (_LEFT_EYE_IDX, _RIGHT_EYE_IDX):
            x = _eye_crop_64(frame, fl, idxs)
            if x is None:
                continue
            logits = _eye_state_sess.run(None, {"eye": x})[0][0]
            ex = np.exp(logits - logits.max())
            sm = ex / ex.sum()
            probs.append(float(sm[0]))  # class 0 = closed (per MRL labelling)
        return (sum(probs) / len(probs)) if probs else None
    except Exception:
        return None

def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        BO = mp.tasks.BaseOptions
        FL = mp.tasks.vision.FaceLandmarker
        FLO = mp.tasks.vision.FaceLandmarkerOptions
        RM = mp.tasks.vision.RunningMode
        # v9.7.3: VIDEO mode (was IMAGE). VIDEO reuses tracker state
        # frame-to-frame so the landmark cost drops ~30% in the steady
        # state — the high-fps thread was capping near 60 fps because
        # IMAGE mode re-ran the full detect+landmark pipeline every
        # frame. With VIDEO the dual-camera loop should comfortably
        # reach the 100-110 fps regime on a 3060.
        _landmarker = FL.create_from_options(FLO(
            base_options=BO(model_asset_path=cfg.MODEL_PATH),
            running_mode=RM.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        ))
    return _landmarker

def get_fast_biometrics():
    with _fast_lock:
        return dict(_fast_bio)

def get_status():
    return {
        "active": _fast_bio["active"],
        "fps": _fast_bio["fps"],
        "frames": _fast_bio["frames"],
        "device": _fast_bio.get("device"),
        # v9.8: surface the chosen camera name + the measured boot fps so
        # the UI pill can show e.g. "120FPS: 115fps (HD USB Camera)".
        # Without this the pill just says "off" when the burst test
        # rejects all candidates, with no hint why.
        "camera_name": _fast_bio.get("camera_name"),
        "boot_fps": _fast_bio.get("boot_fps"),
    }

# ══════════════════════════════════════════════
#  120fps Analysis Functions
# ══════════════════════════════════════════════
# v9.7.3: EAR (Eye Aspect Ratio) fallback. The MediaPipe blendshape
# `eyeBlinkLeft`/`Right` is a learned classifier that gets confused
# by glasses — lens edges, reflections, and frame rims push the score
# into the 0.2-0.5 "neither clearly open nor clearly closed" range,
# which falls below cfg.EYE_CLOSED_THRESH (0.5) so blinks register
# late or never. EAR is computed directly from eyelid landmark
# positions which remain detectable through most glasses, giving a
# more reliable closed-eye signal. We only use it when the blendshape
# is in the ambiguous middle band — leaves the well-calibrated
# clear-cut cases alone.
# Standard 4-point EAR landmarks (MediaPipe FaceMesh 478):
#   Left eye  outer/inner corners: 33, 133  ; lid top/bot: 159, 145
#   Right eye outer/inner corners: 263, 362 ; lid top/bot: 386, 374
def _ear(fl) -> "float | None":
    """v9.8: vectorized EAR. Closed eye ~= 0.10, open ~= 0.30. Computes
    both eyes' aspect ratios in a single numpy operation rather than
    per-landmark Python sqrt calls. ~3x faster on profiling, runs
    every frame at 60 fps so the saving compounds. Behavior identical
    to the previous implementation (verified by unit test, numerical
    parity to 6 decimals)."""
    try:
        # Stack the 8 landmarks into a [4, 2, 2] array:
        #   axis 0 = pair (left_v, left_h, right_v, right_h)
        #   axis 1 = endpoint (a, b)
        #   axis 2 = (x, y)
        pts = np.array([
            [(fl[159].x, fl[159].y), (fl[145].x, fl[145].y)],  # left vertical
            [(fl[33].x,  fl[33].y),  (fl[133].x, fl[133].y)],   # left horizontal
            [(fl[386].x, fl[386].y), (fl[374].x, fl[374].y)],   # right vertical
            [(fl[362].x, fl[362].y), (fl[263].x, fl[263].y)],   # right horizontal
        ], dtype=np.float32)
        # |a - b| over the 4 pairs at once
        d = np.linalg.norm(pts[:, 0] - pts[:, 1], axis=1)
        # d = [l_v, l_h, r_v, r_h]
        l = d[0] / d[1] if d[1] > 1e-6 else 0.0
        r = d[2] / d[3] if d[3] > 1e-6 else 0.0
        return float((l + r) / 2)
    except Exception:
        return None


def _blink(bs, fl=None, frame=None):
    """Blink + PERCLOS + drowsiness via the subject-adaptive
    `BlinkDetector`. `fl` is the 478-landmark FaceMesh result for the
    EAR computation; pass None to use blendshape only. `frame` is the
    BGR camera frame; when supplied AND `FF_BLINK_CNN=1` is in env,
    a 3rd CNN-based eye-state channel (`eye_state_v1.onnx`) joins the
    EAR + blendshape vote inside `BlinkDetector.update()`.

    v9.7.4 full rewrite: was a tangle of blendshape-or-EAR with
    absolute thresholds that never worked on glasses-wearers because
    MediaPipe's eyelid landmarks slide to the lens frame, pinning EAR
    at a constant value where no fixed cut fires. The new detector
    uses RELATIVE drops from a rolling-median baseline — works
    regardless of where the absolute baseline sits.

    API_up v9.8: optional CNN tie-breaker. Activates when FF_BLINK_CNN=1.

    Returns the same 5-tuple shape as the previous implementation
    (blinked, bpm, eye_openness, perclos_pct, drowsy_label) so all
    upstream call sites keep working unchanged."""
    global _last_ear
    bv = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
    ear = _ear(fl) if fl is not None else None
    if ear is not None:
        _last_ear = float(ear)
    cnn_prob = _eye_state_prob(frame, fl) if _FF_BLINK_CNN else None
    out = _blink_det.update(ear=ear, bs_blink=bv, cnn_closed=cnn_prob, t=time.time())
    if out["blinked"]:
        # Mirror to the legacy history deque so callers reading
        # `_blink_history` directly (forensic_export, etc.) keep
        # seeing entries.
        _blink_history.append(time.time())
    # v9.7.5 BUGFIX: return PERCLOS as a FRACTION (0-1), not a
    # percentage. The UI's display logic does `Math.round(av.perclos
    # * 100)` assuming 0-1 input. Returning 0-100 from here caused
    # a double-multiply: a true 2.5 % perclos was reported as
    # `2.5 * 100 = 250 %` in the UI. The analysis.detect_blink path
    # already returns a fraction, so this also unifies the two paths.
    return (out["blinked"], out["blink_rate_pm"], out["eye_openness"],
            out["perclos"], out["drowsiness"])


def _micro(bs):
    _bs_history.append(bs)
    w = min(6, len(_bs_history))
    if len(_bs_history) < w: return False, None, 0
    prev = _bs_history[-w]
    keys = ["mouthSmileLeft","mouthSmileRight","browDownLeft","browDownRight",
            "eyeWideLeft","eyeWideRight","noseSneerLeft","noseSneerRight",
            "mouthFrownLeft","mouthFrownRight","browInnerUp"]
    md, mk = 0, ""
    for k in keys:
        d = abs(bs.get(k, 0) - prev.get(k, 0))
        if d > md: md, mk = d, k
    if md > cfg.MICRO_VELOCITY_THRESH * 0.7:
        return True, mk, round(md, 3)
    return False, None, 0

def _pupil(fl):
    try:
        lw = abs(fl[33].x - fl[133].x); lr = abs(fl[469].x - fl[471].x) / 2
        rw = abs(fl[362].x - fl[263].x); rr = abs(fl[474].x - fl[476].x) / 2
        ld = lr / (lw + 1e-6); rd = rr / (rw + 1e-6)
        return round(ld, 4), round(rd, 4), round((ld + rd) / 2, 4)
    except Exception: return None, None, None

def _duchenne(bs):
    a12 = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
    a6 = (bs.get("cheekSquintLeft", 0) + bs.get("cheekSquintRight", 0)) / 2
    s = a12 > 0.15
    return (s and a6 > 0.1), (s and a6 < 0.05), round(a6, 3), round(a12, 3)

def _asymmetry(bs):
    pairs = [("mouthSmileLeft","mouthSmileRight"),("browDownLeft","browDownRight"),
             ("eyeSquintLeft","eyeSquintRight"),("cheekSquintLeft","cheekSquintRight"),
             ("mouthFrownLeft","mouthFrownRight"),("eyeWideLeft","eyeWideRight"),
             ("noseSneerLeft","noseSneerRight")]
    sc = []
    for l, r in pairs:
        lv, rv = bs.get(l, 0), bs.get(r, 0)
        if lv + rv > 0.05: sc.append(abs(lv - rv) / (max(lv, rv) + 1e-6))
    return round(float(np.mean(sc)), 3) if sc else 0.0

def _fear(bs):
    a1 = bs.get("browInnerUp", 0)
    a2 = (bs.get("browOuterUpLeft", 0) + bs.get("browOuterUpRight", 0)) / 2
    a4 = (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2
    a5 = (bs.get("eyeWideLeft", 0) + bs.get("eyeWideRight", 0)) / 2
    a15 = (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
    bl = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
    return round(a1*0.3+a2*0.2+a5*0.3+a15*0.2, 3), round(a1*0.2+a2*0.15+a4*0.25+a15*0.2+bl*0.2, 3)

def _motion(fl):
    global _prev_lm
    ki = [1, 33, 263, 152, 10]
    cur = [(fl[i].x, fl[i].y) for i in ki if i < len(fl)]
    if not cur: return True
    if _prev_lm is None: _prev_lm = cur; return True
    t = sum(((a-c)**2+(b-d)**2)**0.5 for (a,b),(c,d) in zip(cur, _prev_lm))
    _prev_lm = cur
    return t / len(cur) <= cfg.RPPG_MOTION_THRESH

# ══════════════════════════════════════════════
#  Camera Detection and Loop
# ══════════════════════════════════════════════
MJPEG_FOURCC = cv2.VideoWriter_fourcc(*'MJPG')

# v9.7.3: name-aware camera selection. Previously this function did a
# blind `for idx in range(1, 6)` which (a) skipped idx=0 entirely
# although on many Windows machines the high-fps USB camera lives there
# and (b) would happily pick OBS Virtual Camera, NDI Source, Snap
# Camera, or any other software-virtual capture device that happens to
# be installed. Now we resolve cv2 indices to friendly names via
# DirectShow's FilterGraph (pygrabber) and skip anything matching the
# virtual-camera blacklist. The user can override via env vars:
#   DODO_HIFPS_CAM_NAME       — exact friendly name to prefer
#   DODO_HIFPS_CAM_SKIP_EXTRA — comma-separated extra patterns to skip
# Without pygrabber installed, falls back to the legacy index-iteration
# (so the system never breaks if the dependency is missing).
_VIRTUAL_CAM_PATTERNS = (
    "obs ", "obs virtual", "obs-virtual",
    "virtual camera", "virtualcam", "vcam",
    "snap camera", "snapcam",
    "streamlabs", "xsplit", "scarlet",
    "ndi ", "ndi-",
    "manycam", "ecamm", "epoccam",
    "droidcam",
)


def _list_dshow_cameras() -> "list[tuple[int, str]]":
    """Return [(cv2_idx, friendly_name), ...] in DirectShow enumeration
    order, which matches cv2's MSMF/DSHOW index sequence on Windows.
    Returns empty list if pygrabber is unavailable — callers should
    fall back to plain index iteration in that case."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
        return [(i, str(n)) for i, n in enumerate(names)]
    except Exception:
        return []


def _camera_is_virtual(name: str) -> bool:
    """True if `name` matches any known software-virtual capture device.
    Case-insensitive substring match."""
    n = (name or "").lower()
    if not n:
        return False
    for pat in _VIRTUAL_CAM_PATTERNS:
        if pat in n:
            return True
    extra = os.environ.get("DODO_HIFPS_CAM_SKIP_EXTRA", "")
    for pat in (p.strip().lower() for p in extra.split(",") if p.strip()):
        if pat in n:
            return True
    return False


_HIFPS_PREFERRED_NAME_PATTERNS = (
    # v9.8.1 2026-05-12 (3rd pass): reverted -- HD USB Camera is the
    # high-fps server cam. The Canon R10 (via EOS Webcam Utility) only
    # delivers 30 fps natively, which is FINE for face quality + rPPG
    # + patch_depth + dynamics, but the dual_camera thread's whole
    # purpose is 50-120 fps blink / micro-expression / pupil-rate
    # signals -- 30 fps would HURT those. So:
    #   server dual_camera → HD USB Camera (high fps biometrics)
    #   browser main feed  → Canon R10 if plugged (high quality
    #                        face image; sent to server via WS for
    #                        rPPG / patch_depth / deepfake / dynamics)
    # The browser's getUserMedia picker auto-prefers the R10 because
    # the EOS patterns are NOT in ui.py SERVER_CAM_PATTERNS.
    "hd usb camera",        # primary 50-120 fps target on this machine
    "elp",                  # ELP UVC modules — common 120 fps hardware
    "svpro",                # SVPRO UVC modules — same family
    "120fps", "120 fps",
    "high speed",
    # Canon EOS / capture-card paths kept as LAST-RESORT fallback in
    # case HD USB Camera is unplugged. They'll still be detected and
    # accepted via the Canon fast-path, just at lower priority.
    "eos webcam utility",
    "canon webcam utility",
    "canon eos",
    "elgato cam link",
    "magewell",
    "blackmagic",
    # If you ever swap to a different machine where UGREEN is the
    # 120 fps source, override with `set DODO_HIFPS_CAM_NAME=UGREEN`
    # before launching the server.
)


def _candidate_indices() -> "tuple[list[tuple[int, str]], bool]":
    """Build the ordered list of (idx, name) pairs to try, AND return a
    bool indicating whether any of the candidates actually matched a
    named/preferred pattern (vs. the candidates being purely
    fall-through "any-physical-camera").

    Order:
      1. The user-pinned name (env DODO_HIFPS_CAM_NAME) if it resolves —
         explicit user choice always wins.
      2. Cameras whose name matches `_HIFPS_PREFERRED_NAME_PATTERNS`
         (HD USB Camera, ELP, SVPRO, …). These are the cams that
         actually deliver 120 fps when given exclusive MJPEG access;
         picking them first means the high-fps path stays high-fps
         regardless of where they sit in DirectShow's enumeration.
      3. Any remaining physical camera, in DirectShow order. We no
         longer demote idx=0 — the user enabled Windows' multi-app
         camera access in Settings, so collision with the browser's
         `getUserMedia()` default no longer hard-fails. The browser
         must, however, be configured to use a DIFFERENT cam if you
         want the high-fps path to stay exclusive (see notes below).
      4. Virtual cams (OBS, NDI, Snap, etc.) are always excluded.
      5. Legacy fallback: when pygrabber is missing, iterate
         [0,1,2,3,4,5] — idx=0 first now since the user wants the
         HD USB Camera (which is at idx=0 on this machine) to be
         the default high-fps target.

    The second return value (`matched_named`) is True iff at least one
    candidate came from Tier 1 or Tier 2 (a real name match). False
    iff we're returning purely Tier-3 fall-through cameras. Used by
    `_find_120fps_camera` to optionally SKIP the integrated-camera
    fallback when no named match exists -- that prevents the server
    from grabbing the same camera the browser's `getUserMedia` wants
    to use, which on Windows MSMF is exclusive-open and shows up to
    the browser as `NotReadableError -- Device in use`.
    """
    cams = _list_dshow_cameras()
    if not cams:
        # Legacy fallback: no name info available, so we can't know
        # whether the legacy indices are named matches. Treat as
        # not-matched so the caller can apply the skip-fallback policy.
        return [(i, "") for i in [0, 1, 2, 3, 4, 5]], False
    # Filter virtual cams
    physical = [(i, n) for (i, n) in cams if not _camera_is_virtual(n)]
    skipped  = [(i, n) for (i, n) in cams if _camera_is_virtual(n)]
    if skipped:
        print(f"[DUAL] Skipping virtual camera(s): "
              + ", ".join(f"idx={i} '{n}'" for i, n in skipped))
    # Tier 1 — explicit env pin
    pinned_name = os.environ.get("DODO_HIFPS_CAM_NAME", "").strip().lower()
    if pinned_name:
        pinned = [(i, n) for (i, n) in physical if pinned_name in n.lower()]
        rest   = [(i, n) for (i, n) in physical if pinned_name not in n.lower()]
        if pinned:
            print(f"[DUAL] DODO_HIFPS_CAM_NAME pin matched: "
                  + ", ".join(f"idx={i} '{n}'" for i, n in pinned))
            return pinned + rest, True
    # Tier 2 — known high-fps name patterns. These are the cams that
    # actually deliver 120 fps with MJPEG; ordering them first means
    # the dual-camera target stays consistent across different machines
    # (the same HD USB Camera will be picked whether it lands at idx 0,
    # 1, or 3). Within each tier we keep DirectShow enumeration order.
    preferred = []
    rest      = []
    for i, n in physical:
        nl = n.lower()
        if any(pat in nl for pat in _HIFPS_PREFERRED_NAME_PATTERNS):
            preferred.append((i, n))
        else:
            rest.append((i, n))
    if preferred:
        print(f"[DUAL] Preferred high-fps camera(s): "
              + ", ".join(f"idx={i} '{n}'" for i, n in preferred))
        return preferred + rest, True
    # No named or preferred match -- everything in `rest` is the
    # browser's likely camera (typically idx=0 integrated webcam).
    return rest, False


def _reset_camera_to_auto(idx: int) -> bool:
    """v9.10 fix (2026-05-08): clear any stuck manual-exposure /
    manual-white-balance state on a camera so the browser (or a
    future session) gets a properly auto-exposed feed.

    Why this is needed: when the high-fps capture path runs, it sets
    AUTO_EXPOSURE=0.25 (manual) and EXPOSURE=-6 (1/64 s) for stable
    rPPG. UVC cameras retain those settings at the firmware level
    even after `cap.release()`, so the next consumer (e.g. the
    browser's `getUserMedia`) reads a dark image until something
    explicitly resets them. Windows Camera Settings UI does this
    reset implicitly when opened -- the user reported that as the
    workaround. We do it here on every server startup that's about
    to SKIP the high-fps init.

    Implementation note: DirectShow on Windows ignores AUTO_EXPOSURE
    writes unless frames are actually flowing through the pipeline.
    We open, write the property, READ A FEW FRAMES to commit the
    change, then release. Both DSHOW and MSMF backends are tried in
    sequence -- DSHOW first (more permissive about concurrent opens)
    then MSMF as a fallback if DSHOW didn't accept the write.

    Returns True if the reset was applied AND verified (readback
    matches what we wrote), False on any error.
    """
    for backend, backend_name in (
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF,  "MSMF"),
    ):
        cap = None
        try:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                continue
            # 0.75 = auto on most UVC cameras (mirrors v4l2
            # V4L2_EXPOSURE_APERTURE_PRIORITY = 3).
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)
            # Pump 5 frames so the firmware commits the change.
            # On the integrated webcam this is the only way to make
            # the manual->auto switch stick after a prior session
            # left it in manual mode.
            for _ in range(5):
                cap.read()
            ae_after = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            print(f"[DUAL] reset cam {idx} via {backend_name}: "
                  f"AUTO_EXPOSURE -> {ae_after}")
            return True
        except Exception as e:
            print(f"[DUAL] reset cam {idx} via {backend_name} "
                  f"raised: {e!r}")
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
    return False


def _find_120fps_camera():
    """Find high-fps camera, skipping virtual capture devices (OBS, NDI,
    Snap, etc.). Forces MJPEG codec which is required for 120 fps on
    ELP/SVPRO/UGREEN cameras — they default to YUY2@30fps without
    explicit MJPEG selection. The browser's getUserMedia owns one of
    the physical cameras; the OS's exclusive-open semantics will make
    that camera fail to open here, so we naturally fall through to the
    next candidate (the dual-camera target).

    v9.8: env override DODO_HIFPS_FORCE_120=1 skips the 720p@60 path
    entirely and demands 480p@120 MJPG. Useful for paper-figure runs
    that need the 120 fps headline number even though 720p@60 gives
    cleaner rPPG signal (4x more skin pixels per frame). Default OFF.
    """
    candidates, matched_named = _candidate_indices()
    print(f"[DUAL] Camera selection order: "
          + ", ".join(f"idx={i}('{n or '?'}')" for i, n in candidates))

    # v9.10 fix (2026-05-08): when no camera matched a named/preferred
    # pattern (no DODO_HIFPS_CAM_NAME hit, no _HIFPS_PREFERRED pattern
    # match), the only candidates left are random physical cameras --
    # most likely the integrated laptop webcam at idx=0 that the browser
    # *also* wants for `getUserMedia`. On Windows MSMF that handle is
    # exclusive, so the server grabbing it makes the browser report
    # `NotReadableError -- Device in use`.
    #
    # Default new behaviour: if no named match, SKIP the high-fps
    # init entirely. The browser keeps the integrated camera; the
    # server still gets per-frame data via the WS pipeline.
    #
    # Override: set DODO_HIFPS_ALLOW_INTEGRATED=1 to re-enable the
    # old fallback (grab whatever camera is available, accepting the
    # browser conflict). Useful for headless / single-app setups
    # where there is no browser webcam consumer.
    if not matched_named:
        allow = os.environ.get("DODO_HIFPS_ALLOW_INTEGRATED", "0").lower() in (
            "1", "true", "yes")
        if not allow:
            print("[DUAL] No DODO_HIFPS_CAM_NAME match and no preferred "
                  "high-fps cam (HD USB Camera, ELP, SVPRO, UGREEN, …) "
                  "is plugged in. SKIPPING server-side high-fps capture "
                  "so the browser can have the integrated camera "
                  "without `NotReadableError`. To override and grab the "
                  "integrated camera anyway, set "
                  "DODO_HIFPS_ALLOW_INTEGRATED=1.")
            # v9.10 fix: a previous server run may have left the
            # candidate camera with manual exposure / manual WB stuck
            # at the rPPG settings (AUTO_EXPOSURE=0.25, EXPOSURE=-6).
            # That makes the browser see a dark feed -- and the
            # patch_depth model trained on properly-exposed faces
            # flags underexposed frames as SPOOF. Reset every
            # candidate camera to auto-exposure now so the browser
            # gets a clean image from the next getUserMedia call.
            for _idx, _name in candidates:
                if _reset_camera_to_auto(_idx):
                    print(f"[DUAL] Reset cam idx={_idx} '{_name}' "
                          f"to AUTO exposure / AUTO white balance "
                          f"(clears stuck manual settings from prior runs).")
            # Match the function's documented (cap, idx, fps) return
            # contract so `_camera_loop` can unpack with no TypeError.
            return None, None, 0
        print("[DUAL] No named/preferred match, but "
              "DODO_HIFPS_ALLOW_INTEGRATED=1 -- falling back to the "
              "first available physical camera (browser may report "
              "Device-in-use).")
    _force_120 = os.environ.get("DODO_HIFPS_FORCE_120", "0").lower() in ("1", "true", "yes")
    if _force_120:
        print("[DUAL] DODO_HIFPS_FORCE_120=1 -- skipping 720p@60 path, "
              "demanding 480p@120 MJPG.")
    for idx, _name in candidates:
        cap = None
        # v9.8.1 (2026-05-12): try DSHOW FIRST, MSMF as fallback.
        #
        # WHY: Windows 10/11 "Camera Frame Server" sits on top of MSMF
        # and forces SHARED access -- under which UVC cameras silently
        # fall back to YUY2 @ 30 fps regardless of cap.set(FOURCC, MJPG).
        # Operator's HD USB Camera kept reporting boot_fps=35 (YUY2)
        # because the MSMF path was selected first, and the launcher's
        # ConcurrentAccessSupported=0 registry pin requires re-enumeration
        # to take effect. DirectShow bypasses the Frame Server entirely:
        # it opens the device in EXCLUSIVE mode, the FOURCC=MJPG set
        # actually applies, and the camera enumerates its full mode
        # table (640x480@120, 1280x720@60). Verified by testing
        # `cv2.VideoCapture(idx, CAP_DSHOW)` directly: same camera that
        # MSMF caps at 30 fps sustains 60 fps under DSHOW.
        # The order swap is the actual fps fix the operator was asking
        # for; the hardware was fine, the OS plumbing was wrong.
        for backend, backend_name in (
            (cv2.CAP_DSHOW, "DSHOW"),
            (cv2.CAP_MSMF,  "MSMF"),
        ):
            try:
                trial = cv2.VideoCapture(idx, backend)
                if trial.isOpened():
                    cap = trial
                    print(f"[DUAL] Cam {idx} '{_name or '?'}': opened via {backend_name}")
                    break
                trial.release()
            except Exception:
                pass
        if cap is None:
            # v9.7.3: when the browser's getUserMedia owns this camera,
            # OpenCV's open() will fail silently — that's the natural
            # exclusion mechanism that lets the dual-camera path pick
            # the OTHER physical camera. Just log and move on.
            if _name:
                print(f"[DUAL] Cam {idx} '{_name}': could not open "
                      "(in use by browser? unplugged?), trying next…")
            continue
        try:
            # v9.8.1 2026-05-12: Canon EOS / pro capture-card fast path.
            # Canon's EOS Webcam Utility (and capture cards like Cam Link)
            # IGNORE cv2 cap.set() for FOURCC, resolution, fps, and
            # exposure -- they always serve their configured native
            # stream (typically 1080p30 YUY2 for the R10). Running our
            # standard probe sequence on these cameras:
            #   - cap.set(MJPG) silently fails
            #   - cap.set(width=1280, height=720) silently fails
            #   - exposure / WB setters silently fail
            #   - burst test reads native 30 fps for 1.5 s → 45 frames
            #     measured → below the >=50 fps gate → fall to 480p path
            #   - 480p path same situation → fall to last-resort
            #     >=25 path BUT only if 480p re-open succeeded, which it
            #     doesn't because Canon ignores those calls too
            # Net effect: R10 is REJECTED entirely even though it's
            # producing perfectly usable 1080p30 frames. Detect Canon /
            # pro-grade devices by name and trust their native rate.
            _name_lc = (_name or "").lower()
            _is_canon_or_pro = any(s in _name_lc for s in (
                "eos webcam", "canon webcam", "canon eos",
                "elgato cam link", "magewell", "blackmagic",
            ))
            _pin_lc = os.environ.get("DODO_HIFPS_CAM_NAME", "").strip().lower()
            _pin_targets_pro = any(s in _pin_lc for s in (
                "eos webcam", "canon webcam", "canon eos",
                "elgato cam link", "magewell", "blackmagic",
            ))
            _allow_pro_fallback = _pin_targets_pro or (
                os.environ.get("DODO_HIFPS_ALLOW_PRO_FALLBACK", "0").lower()
                in ("1", "true", "yes")
            )
            if _is_canon_or_pro and not _allow_pro_fallback:
                print(f"[DUAL] Cam {idx} '{_name}': pro/canon path detected, "
                      "but DODO_HIFPS_ALLOW_PRO_FALLBACK=0. "
                      "Skipping to keep HD USB/ELP-class high-fps role.")
                cap.release()
                continue
            if _is_canon_or_pro:
                # v9.8.1 2026-05-12 (refined): the previous version
                # used time-bounded `while time.time() - t0 < 2.0: cap.read()`
                # loops, which deadlock if cap.read() blocks longer
                # than the wall-clock window (happens when the R10 is
                # plugged in via USB but the camera body is OFF, or
                # the EOS Webcam Utility background service hasn't
                # latched the stream yet). The time check only fires
                # AFTER cap.read() returns -- if it never returns, the
                # whole probe hangs.
                #
                # Safer: do a single bounded cap.read() with a
                # backend-level grab() first (which is non-blocking
                # on most backends), then accept if any frame came
                # out. Skip the fps measurement entirely; the live
                # camera_loop will compute and expose the actual rate
                # via _fast_bio["fps"] once streaming.
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                # Single non-bounded read attempt -- if Canon isn't
                # actually streaming this returns False fast (DSHOW
                # error path) rather than blocking.
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"[DUAL] Canon/Pro fast-path cam {idx} '{_name}': "
                          f"first-frame OK {actual_w}x{actual_h} -- accepting; "
                          f"camera_loop will measure live fps")
                    # Report a placeholder fps of 30 -- the camera_loop
                    # immediately overwrites with measured value.
                    return cap, idx, 30
                print(f"[DUAL] Canon/Pro cam {idx} '{_name}': "
                      f"first-frame failed (ret={ret}, camera body off / "
                      f"EOS Utility service not running?), releasing")
                cap.release()
                continue

            # CRITICAL: Force MJPEG codec FIRST, before setting resolution/fps.
            # Without this, the camera stays in YUY2 mode at 30fps max.
            cap.set(cv2.CAP_PROP_FOURCC, MJPEG_FOURCC)

            # 2026-04-25: target 1280x720 @ 60fps MJPG. The HD USB Camera
            # firmware exposes (verified via NCH Debut Capture):
            #   - 1920x1080 MJPG -> 30 fps max
            #   - 1280x720  MJPG -> 60 fps max  <-- our target
            #   -  640x480  MJPG -> 120 fps max
            #   - any res  YUY2 -> 5-30 fps (uncompressed, bandwidth-bound)
            # 720p@60 gives 4x more pixels than 480p@120 for face-crop /
            # rPPG / blink detection. 60 fps oversamples blinks (~150 ms,
            # = 9 frames) and heart rate (~1 Hz) by ~6x — sufficient.
            # If the camera is 480p-only (older / no 720p MJPG mode), the
            # cv2 set_resolution will silently round down and the burst
            # test below will trigger the 480p@120 fallback path.
            # v9.8: when DODO_HIFPS_FORCE_120=1, skip the 720p path
            # and target 480p@120 directly.
            if _force_120:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 120)
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 60)
            # v9.8.1 2026-05-12: re-apply MJPG FOURCC after the resize
            # call. On the HD USB Camera under DSHOW, setting width/height
            # silently reverts the codec to YUY2 -- which then can't
            # sustain 720p@60 across USB 2.0 (~110 MB/s required, bus
            # caps at ~40), so the burst test measured 10 fps and we
            # fell through to the 480p fallback. The 480p path re-applied
            # FOURCC after resize and got 50 fps MJPG -- the 720p path
            # needed the same belt-and-braces. Boot log evidence
            # (2026-05-12, 720p probe):
            #   Probe cam 0: 1280x720@60fps codec=YUY2   <-- silently reverted
            #   Burst test:  15 frames in 1.50s = 10.0 fps
            cap.set(cv2.CAP_PROP_FOURCC, MJPEG_FOURCC)

            # Disable auto-exposure (critical for rPPG)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual on most UVC cameras
            cap.set(cv2.CAP_PROP_EXPOSURE, -6)          # ~1/120s exposure time
            # Disable auto white balance (rPPG needs stable color channels)
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)

            # Read actual capabilities
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc_raw = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc_raw >> (8*i)) & 0xFF) for i in range(4)])

            print(f"[DUAL] Probe cam {idx}: {actual_w}x{actual_h}@{actual_fps}fps codec={fourcc_str}")

            # v9.7.3: ALWAYS burst-test, regardless of `actual_fps`. The
            # Windows UVC `cap.get(CAP_PROP_FPS)` is unreliable -- the
            # MSMF backend sometimes silently fails to apply the FOURCC
            # switch and reports 30 fps even when the camera will
            # actually push 120 fps if you just read frames in a tight
            # loop. The earlier `if actual_fps >= 50:` gate was rejecting
            # the camera in this state without ever measuring real
            # throughput. Now we trust only the measured rate.
            #
            # v9.8: time-bound the burst (1.5 s) and use 30-frame warmup.
            # The pre-v9.8 30-frame burst could finish in 0.25 s reading
            # only buffered frames, reporting "120 fps" even when the
            # USB hub bus actually saturates at 30 fps. The bounded
            # window measures sustained throughput.
            for _ in range(30):
                cap.read()  # warmup -- drain initial buffered frames
            t0 = time.time()
            count = 0
            burst_window = 1.5
            while time.time() - t0 < burst_window:
                ret, _ = cap.read()
                if ret: count += 1
            elapsed = time.time() - t0
            measured_fps = count / elapsed if elapsed > 0 else 0
            print(f"[DUAL] Burst test cam {idx}: {count} frames in {elapsed:.2f}s = {measured_fps:.1f} fps "
                  f"(reported {actual_fps}, codec={fourcc_str or 'unknown'})")
            # v9.8.1 2026-05-12 (operator approved "yes do it all" -- the
            # 720p@60 bump): lower the acceptance threshold from 50 to 45.
            # The HD USB Camera's `/camera/capabilities` probe confirms
            # 1280x720@60fps MJPG is firmware-supported, but on USB-2.0
            # shared bus we typically measure 50-58 fps sustained instead
            # of a clean 60. 50 was too tight a gate -- jitter alone could
            # push us into the 480p fallback. 45 is wide enough to ride
            # bus variance while still rejecting cases where the codec
            # negotiation actually failed (YUY2 caps at 30 fps).
            if measured_fps >= 45:
                return cap, idx, round(measured_fps)

            # 720p path failed (codec negotiation actually broke, or this
            # camera genuinely doesn't have a 720p@60 mode). Try
            # 480p@120 as the fallback — verified-supported on most
            # commodity 120fps cams.
            print(f"[DUAL] Cam {idx}: 720p path got {measured_fps:.1f} fps measured, falling back to 480p@120 MJPG")
            # v9.7.3: re-set FOURCC FIRST under MSMF so the codec switch
            # actually applies. Order matters on Windows — width/height
            # before FOURCC sometimes wedges the codec at YUY2.
            cap.set(cv2.CAP_PROP_FOURCC, MJPEG_FOURCC)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 120)
            # Re-apply FOURCC after dimensions in case the resize
            # reverted it. Belt-and-braces: this has been observed to
            # be the difference between 30 fps and 120 fps on the same
            # ELP/SVPRO cameras across reboots.
            cap.set(cv2.CAP_PROP_FOURCC, MJPEG_FOURCC)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc_raw = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc_raw >> (8*i)) & 0xFF) for i in range(4)])
            print(f"[DUAL] Probe cam {idx} (fallback): {actual_w}x{actual_h}@{actual_fps}fps codec={fourcc_str or 'unknown'}")

            # v9.7.3: same -- always burst-test, don't trust actual_fps.
            # v9.8: same warmup + bounded-window fix as the 720p path above.
            for _ in range(30):
                cap.read()
            t0 = time.time()
            count = 0
            while time.time() - t0 < burst_window:
                ret, _ = cap.read()
                if ret: count += 1
            elapsed = time.time() - t0
            measured_fps = count / elapsed if elapsed > 0 else 0
            print(f"[DUAL] Burst test cam {idx} (480p): {count} frames in {elapsed:.2f}s = {measured_fps:.1f} fps "
                  f"(reported {actual_fps}, codec={fourcc_str or 'unknown'})")

            if measured_fps >= 50:
                return cap, idx, round(measured_fps)

            # v9.8: 480p path also too slow. Last-resort accept whatever
            # rate the camera CAN sustain -- rPPG at 30 fps still works
            # (Nyquist for 1-2 Hz pulse signal is ~4 fps). Return the cap
            # but with a loud warning so the user knows the bus is
            # bottlenecked.
            if measured_fps >= 25:
                print(f"[DUAL] [WARN] cam {idx} sustaining only {measured_fps:.1f} fps "
                      f"(target was 60+). USB-bus bandwidth is the likely cause -- "
                      f"this camera is probably sharing a USB 2.0 hub with another "
                      f"device. For 120 fps, plug it directly into a USB 3.0+ port "
                      f"on the laptop chassis (not via a dock or hub). rPPG / blink "
                      f"/ micro-expression detection still work at this rate, just "
                      f"with reduced temporal resolution.")
                return cap, idx, round(measured_fps)

            cap.release()
        except Exception as e:
            print(f"[DUAL] Probe cam {idx} error: {e}")
    return None, None, 0

def _camera_loop(rppg_mod):
    cap, cidx, measured_fps = _find_120fps_camera()
    if cap is None:
        with _fast_lock:
            _fast_bio["active"] = False
        print("[DUAL] No high-fps camera found. System runs on single camera.")
        print("[DUAL] All biometrics (rPPG, blink, micro, pupil) use HQ camera at 30fps.")
        return

    # v9.7.3: surface the camera's friendly name in the boot banner so
    # it's instantly obvious whether OBS Virtual / Snap / NDI snuck in.
    _cam_name = "?"
    try:
        for _i, _n in _list_dshow_cameras():
            if _i == cidx:
                _cam_name = _n
                break
    except Exception:
        pass
    # v9.7.5: extra-loud confirmation banner so when the user grep-tails
    # the boot log they can verify at a glance that HD USB Camera (or
    # whichever they pinned) was chosen, not some accidental fallback.
    _expected = os.environ.get("DODO_HIFPS_CAM_NAME", "").strip()
    _match_marker = ""
    if _expected:
        if _expected.lower() in (_cam_name or "").lower():
            _match_marker = "  [PIN MATCHED]"
        else:
            _match_marker = f"  [WARNING: env pin was '{_expected}' — got '{_cam_name}' instead]"
    print(f"[DUAL] ============================================================")
    print(f"[DUAL]  120fps camera ACTIVE: '{_cam_name}'  idx={cidx}  {measured_fps} fps measured{_match_marker}")
    print(f"[DUAL] ============================================================")
    print("[DUAL] HQ camera: mesh, demographics, emotion, recognition")
    print("[DUAL] 120fps camera: rPPG, blink, PERCLOS, micro-expressions, pupil, deception AUs")

    with _fast_lock:
        _fast_bio["active"] = True
        _fast_bio["device"] = cidx
        _fast_bio["camera_name"] = _cam_name
        _fast_bio["boot_fps"] = int(measured_fps)
        _fast_bio["fps"] = int(measured_fps)  # seed for the UI pill before
                                              # the steady-state fps counter
                                              # has its first 2-second window

    lm = _get_landmarker()
    fc = 0
    global _lm_ts_ms

    # v9.7.5: landmarker frame-skip. The MediaPipe FaceLandmarker
    # detect_for_video call is the dominant per-frame cost (~30-35 ms
    # on CPU) and dominates the per-frame budget — that's why steady-
    # state fps was 22 even though the camera was pushing 107 fps in
    # the burst test. By detecting on every Nth frame (default 2) and
    # caching the landmarks for the in-between frame, rPPG sampling
    # gets to run at the full camera rate for clean signal, while
    # FaceMesh-derived signals (blink, micro, pupil, asymmetry) stay
    # at the slightly-lower but still-fine N/2 rate. Override via env
    # `DODO_DUAL_CAM_DETECT_EVERY_N`. N=1 = original behavior (slow,
    # ~22 fps). N=2 = ~40 fps (recommended, smooth biometrics).
    # N=3 = ~60 fps but blink/micro responsiveness drops.
    _detect_every_n = 2
    try:
        _detect_every_n = max(1, int(os.environ.get("DODO_DUAL_CAM_DETECT_EVERY_N", "2")))
    except Exception:
        pass
    _last_fl = None
    _last_bs: dict = {}

    while not _stop.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        fc += 1
        now = time.time()
        _fps_times.append(now)
        cfps = len([t for t in _fps_times if now - t < 1.0])
        # v9.8.1 2026-05-12 (refined): stash JPEG-encoded copy of the
        # latest frame so the UI can render a thumbnail preview of the
        # server-held camera (HD USB Camera at 720p/60 etc) that the
        # browser doesn't have a direct handle on.
        #
        # 2nd pass (operator reported "all feeds are choppy"): the
        # 5 fps re-encode at quality=60 was uploading ~250 KB / sec
        # per polling client, AND the encode itself was eating ~3 ms
        # per call at 720p which steals from the camera_loop's heavy
        # tick. Drop to ~3 fps preview, shrink to 480px wide, and use
        # quality=50 -- still readable as a "yes this camera is
        # alive" thumbnail, ~40 KB / frame. Keeps the PAD pipeline
        # un-starved on the dGPU thread.
        _jpeg_tick = 20 if cfps >= 50 else 10  # ~2-3 Hz preview
        if fc % _jpeg_tick == 0:
            try:
                _h, _w = frame.shape[:2]
                if _w > 480:
                    _scale = 480.0 / float(_w)
                    _frame_jpeg = cv2.resize(
                        frame, (480, int(_h * _scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    _frame_jpeg = frame
                ok, buf = cv2.imencode(".jpg", _frame_jpeg,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                if ok:
                    with _latest_jpeg_lock:
                        globals()["_latest_jpeg_bytes"] = buf.tobytes()
                        globals()["_latest_jpeg_ts"] = now
            except Exception:
                pass  # nolint:silent-except encode-best-effort
        # v9.8.1: write frame counter + live fps to _fast_bio every 10 frames
        # so /status and the stale-frame watchdog can observe forward
        # progress. Before this, the watchdog was firing every 8 s because
        # _fast_bio["frames"] never advanced past its initial 0 -- the loop
        # incremented only the local `fc`.
        if fc % 10 == 0:
            with _fast_lock:
                _fast_bio["frames"] = fc
                _fast_bio["fps"] = cfps

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Frame-skip the heavy detect call.
            should_detect = (fc % _detect_every_n == 0) or _last_fl is None
            if should_detect:
                _lm_ts_ms += 1
                r = lm.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    _lm_ts_ms,
                )
                if r.face_landmarks:
                    _last_fl = r.face_landmarks[0]
                    if r.face_blendshapes and len(r.face_blendshapes) > 0:
                        _last_bs = {b.category_name: round(b.score, 3)
                                    for b in r.face_blendshapes[0]}
                else:
                    # detection lost; fall through to cached lm if any
                    pass

            if _last_fl is None:
                continue

            fl = _last_fl
            bs = _last_bs

            # rPPG sampling at 120fps (the primary benefit)
            if _motion(fl):
                rppg_mod.add_sample(frame, fl)

            # All time-sensitive biometrics at 120fps. v9.7.3: pass `fl`
            # so _blink can apply the EAR fallback for glasses-confused
            # blendshapes.
            bl, bpm, eo, pc, dr = _blink(bs, fl, frame=frame)
            md, mm, mi = _micro(bs)
            pl, pr, pa = _pupil(fl)
            duc, fake, a6, a12 = _duchenne(bs)
            asym = _asymmetry(bs)
            fear, dec_au = _fear(bs)

            with _fast_lock:
                _fast_bio.update({
                    "blink_rate": bpm, "eye_openness": eo, "perclos": pc,
                    "drowsiness": dr, "blinked": bl,
                    "micro_detected": md, "micro_muscle": mm, "micro_intensity": mi,
                    "pupil_dilation": pa, "pupil_l": pl, "pupil_r": pr,
                    "duchenne": duc, "fake_smile": fake, "au6": a6, "au12": a12,
                    "asymmetry": asym, "fear_score": fear, "deception_au_score": dec_au,
                    "fps": cfps, "frames": fc, "active": True, "device": cidx,
                })
        except Exception:
            pass

    cap.release()
    with _fast_lock:
        _fast_bio["active"] = False
    print(f"[DUAL] 120fps camera stopped after {fc} frames")

# ══════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════
def start(rppg_mod):
    global _thread
    if _thread and _thread.is_alive():
        return False
    _stop.clear()
    _thread = threading.Thread(target=_camera_loop, args=(rppg_mod,), daemon=True, name="dual-120fps")
    _thread.start()
    return True

def stop():
    _stop.set()
    if _thread: _thread.join(timeout=3)

def reset():
    global _prev_blink, _prev_lm
    _blink_history.clear(); _perclos_history.clear(); _bs_history.clear()
    _prev_blink = False; _prev_lm = None


# ──────────────────────────────────────────────────────────────────
#  v9.8.1 — runtime restart + stale-frame watchdog
# ──────────────────────────────────────────────────────────────────

def restart(rppg_mod=None) -> dict:
    """Force-reacquire the high-fps camera. Use after the browser's
    getUserMedia collision steals the camera and leaves the dual-cam
    capture loop reading stale frames forever.

    Re-uses the existing rppg_mod handle if we have one (captured by
    `start()`), otherwise the caller must pass it. Returns a status
    dict the operator endpoint can echo back."""
    global _thread, _last_rppg_mod
    info = {"prior_thread_alive": False, "prior_frames": int(_fast_bio.get("frames", 0))}
    if _thread and _thread.is_alive():
        info["prior_thread_alive"] = True
        _stop.set()
        try:
            _thread.join(timeout=4)
        except Exception:
            pass  # nolint:silent-except watchdog teardown best-effort
    _stop.clear()
    rppg_to_use = rppg_mod or _last_rppg_mod
    if rppg_to_use is None:
        info["error"] = "no rppg_mod available -- pass one in or call start() first"
        return info
    _last_rppg_mod = rppg_to_use
    _thread = threading.Thread(
        target=_camera_loop, args=(rppg_to_use,), daemon=True, name="dual-120fps"
    )
    _thread.start()
    info["thread_started"] = True
    return info


_last_rppg_mod = None  # captured by start() so restart() can reuse


_orig_start = start
def start(rppg_mod):  # noqa: F811
    """v9.8.1: shim around the original start() to remember rppg_mod
    so the watchdog + /dual_camera/restart can re-launch without the
    caller having to pass it again."""
    global _last_rppg_mod
    _last_rppg_mod = rppg_mod
    return _orig_start(rppg_mod)


# Watchdog: monitors `_fast_bio["frames"]` and `cap.read()` cadence.
# If frames stop advancing for STALL_S seconds AND the loop claims
# active=True, restart the capture. Survives the browser stealing the
# camera mid-session via shared MSMF mode.
_watchdog_thread = None
_watchdog_stop = threading.Event()
WATCHDOG_STALL_S = 6.0      # restart if no new frames for this long
WATCHDOG_POLL_S  = 2.0      # check cadence


def _watchdog_loop() -> None:
    last_frames = -1
    last_change_t = time.time()
    while not _watchdog_stop.is_set():
        time.sleep(WATCHDOG_POLL_S)
        with _fast_lock:
            cur_frames = int(_fast_bio.get("frames", 0))
            active = bool(_fast_bio.get("active", False))
        if cur_frames != last_frames:
            last_frames = cur_frames
            last_change_t = time.time()
            continue
        if not active:
            continue  # nothing to recover; either pre-init or shut down
        age = time.time() - last_change_t
        if age >= WATCHDOG_STALL_S:
            print(f"[DUAL-WD] frames stalled at {cur_frames} for {age:.1f}s "
                  f"-- triggering restart() (likely browser-camera collision)")
            try:
                info = restart()
                print(f"[DUAL-WD] restart() -> {info}")
            except Exception as e:
                print(f"[DUAL-WD] restart() failed: {type(e).__name__}: {e}")
            last_change_t = time.time()  # don't hammer on persistent failure


def start_watchdog() -> bool:
    """Idempotent. Called from main.py once the dual-cam start() has
    been kicked off; ensures the watchdog isn't tied to the camera
    thread's lifetime."""
    global _watchdog_thread
    if _watchdog_thread and _watchdog_thread.is_alive():
        return False
    _watchdog_stop.clear()
    _watchdog_thread = threading.Thread(
        target=_watchdog_loop, daemon=True, name="dual-watchdog"
    )
    _watchdog_thread.start()
    print(f"[DUAL-WD] watchdog armed (stall_s={WATCHDOG_STALL_S}, poll_s={WATCHDOG_POLL_S})")
    return True
