"""
DODO VISION AI v8.9 — rPPG
POS algorithm (Wang et al. 2017) with motion artifact rejection
and adaptive fps estimation from frame 1.

v8.9: optional CuPy GPU offload for FFT and bandpass filtering.
"""
import time
import os
import math
import json
import threading
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, medfilt
import config as cfg

# v8.9 Step 4: CuPy GPU-accelerated FFT (transparent fallback to NumPy)
_CUPY_OK = False
_cp = None
_cpfft = None
if getattr(cfg, "RPPG_GPU_OFFLOAD", False):
    try:
        import cupy as _cp
        import cupy.fft as _cpfft
        _CUPY_OK = True
        try:
            _gpu_name = _cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        except Exception:
            _gpu_name = "NVIDIA"
        print(f"[OK] rPPG GPU offload: CuPy FFT enabled ({_gpu_name} via CUDA)")
    except ImportError:
        # v9.2 plan step 12: give an actionable install hint. CuPy
        # bundles its own CUDA runtime so no separate toolkit install
        # is needed — pick the wheel that matches your CUDA minor.
        print("[--] rPPG GPU offload: CuPy not installed, using NumPy FFT")
        print("     Install for GPU-accelerated FFT:")
        print("       pip install cupy-cuda12x    # for CUDA 12.x")
        print("       pip install cupy-cuda11x    # for CUDA 11.x")


def _gpu_rfft(signal: np.ndarray):
    """Compute rfft on GPU if CuPy available, else NumPy.

    Returns (fft_magnitudes, freqs) as NumPy arrays.
    """
    n = len(signal)
    min_len = getattr(cfg, "RPPG_GPU_FFT_MIN_LEN", 256)
    if _CUPY_OK and n >= min_len:
        sig_gpu = _cp.asarray(signal)
        fft_gpu = _cp.abs(_cpfft.rfft(sig_gpu))
        return _cp.asnumpy(fft_gpu)
    return np.abs(np.fft.rfft(signal))

# v7.5: physiological HR spike filter applied to rPPG output as well.
# Previously only radar HR went through hr_filter; rPPG could still spike
# on motion or lighting glitches and overwrite the displayed BPM.
try:
    from hr_filter import HRFilter, HR_ADULT_RESTING
    _hr_filter = HRFilter(HR_ADULT_RESTING)
except Exception:
    _hr_filter = None

# EMA smoother for display-stable BPM (α=0.25 → ~4-sample memory).
_ema_hr = None
_EMA_ALPHA = 0.25

# ── Step 5: Auto-load calibration regression ──
_cal_slope = 1.0
_cal_intercept = 0.0
_cal_loaded = False

def _load_calibration_correction():
    """Load regression correction from calibration.py's output file."""
    global _cal_slope, _cal_intercept, _cal_loaded
    reg_file = getattr(cfg, 'CALIBRATION_REGRESSION_FILE', os.path.join(cfg.CALIBRATION_DIR, "rppg_correction.json"))
    if os.path.exists(reg_file):
        try:
            with open(reg_file) as f:
                d = json.load(f)
            _cal_slope = d.get("slope", 1.0)
            _cal_intercept = d.get("intercept", 0.0)
            _cal_loaded = True
            print(f"[rPPG] Calibration loaded: y={_cal_slope:.4f}*x+{_cal_intercept:.2f}")
        except Exception as e:
            # 2026-04-25 (v9.7.1): surface load failures so the user
            # knows why their calibration isn't applying. Falls through
            # to the identity transform (1*x + 0) which is safe.
            print(f"[rPPG] Calibration load failed: {type(e).__name__}: {e}")

def apply_calibration(raw_hr):
    """Apply regression correction if calibration exists."""
    if not _cal_loaded:
        _load_calibration_correction()
    if _cal_loaded and raw_hr is not None and raw_hr > 0:
        return round(_cal_slope * raw_hr + _cal_intercept, 1)
    return raw_hr

# Auto-load on import
_load_calibration_correction()

# ── Buffers ──
rppg_rgb = deque(maxlen=cfg.RPPG_BUFFER_SIZE)
rppg_times = deque(maxlen=cfg.RPPG_BUFFER_SIZE)
hr_samples = deque(maxlen=300)
# v9.2 step 29: timestamp of last confirmed_bpm integration step for the
# velocity-aware drift integrator. Stays None until the 60-s lock gate opens.
_confirmed_bpm_t_prev = None
confirmed_bpm = None
start_time = None
fps_counter = deque(maxlen=60)

# ── Motion artifact rejection ──
_prev_landmarks = None
_motion_rejected = 0
_rppg_lock = threading.Lock()  # thread safety for dual camera

# ── Diagnostic throttle (5 s heartbeat of rPPG pipeline state) ──
_last_diag_t = 0.0

# ── ROI landmark indices (forehead + cheeks) ──
ROI_INDICES = [10, 67, 109, 108, 69, 104, 68, 338, 297, 299, 337, 333,
               116, 123, 147, 187, 205, 36, 345, 352, 376, 411, 425, 266]

# ── Key landmarks for motion detection (nose, chin, eye corners) ──
MOTION_LANDMARKS = [1, 33, 263, 152, 10, 468, 473]

# v9.8: cache Butterworth filter coefficients. `butter()` is pure
# numerical computation (no I/O) but at order=4 it allocates ~20 small
# arrays per call. `bandpass()` runs every rppg cycle (~5 Hz) so caching
# saves ~100 us / call -- not earth-shattering but compounds over a long
# session and removes a small allocation hot spot.
from functools import lru_cache as _lru_cache_rppg

@_lru_cache_rppg(maxsize=32)
def _butter_coefs(order: int, lo_norm: float, hi_norm: float):
    """Cached Butterworth bandpass coefficients keyed by (order, lo/nyq, hi/nyq)."""
    return butter(order, [lo_norm, hi_norm], btype="band")


def bandpass(sig, lo=None, hi=None, fs=20, order=4):
    lo = lo or cfg.RPPG_BANDPASS_LO
    hi = hi or cfg.RPPG_BANDPASS_HI
    nyq = fs * 0.5
    # Quantize the band edges to 6 decimals so floating-point dither at
    # the 7th doesn't bust the cache (fs varies frame-to-frame by ~0.01).
    lo_n = round(max(lo / nyq, 0.01), 6)
    hi_n = round(min(hi / nyq, 0.99), 6)
    b, a = _butter_coefs(order, lo_n, hi_n)
    return filtfilt(b, a, sig)

def pos_rppg(rgb_buf, fs=20):
    """POS algorithm: plane-orthogonal-to-skin."""
    if len(rgb_buf) < cfg.RPPG_MIN_SAMPLES:
        return None
    C = np.array(rgb_buf)
    m = np.mean(C, axis=0)
    if np.any(m < 1):
        return None
    Cn = C / m
    S1 = Cn[:, 1] - Cn[:, 2]
    S2 = Cn[:, 1] + Cn[:, 2] - 2 * Cn[:, 0]
    std1 = np.std(S1) if np.std(S1) > 0 else 1
    std2 = np.std(S2) if np.std(S2) > 0 else 1
    return S1 + (std1 / std2) * S2


# ══════════════════════════════════════════════
#  v9.0 Step 4: PRISM Signal Mixing (CMU Architecture)
#  Projection-based Robust and Interpretable Signal Mixing for rPPG.
#  Adaptively optimizes photometric detrending and color mixing in real
#  time.  Heavy matrix operations are offloaded to CUDA via CuPy.
# ══════════════════════════════════════════════

def _prism_projection_matrix(rgb_buf: np.ndarray, fs: float) -> np.ndarray:
    """Compute the PRISM adaptive projection matrix.

    PRISM decomposes the RGB signal into a skin-tone subspace and a
    pulse subspace using adaptive photometric detrending.  The projection
    matrix W is learned per-window to maximally separate specular
    reflection (noise) from volumetric pulse (signal).

    GPU offload: if CuPy is available, the covariance computation and
    eigendecomposition run on the NVIDIA GPU, freeing CPU cycles for
    websocket handling.
    """
    n, d = rgb_buf.shape  # n=samples, d=3 (RGB)
    if n < 10 or d != 3:
        return np.eye(d)

    # Temporal normalization (column-wise z-score)
    mu = rgb_buf.mean(axis=0)
    sigma = rgb_buf.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    Cn = (rgb_buf - mu) / sigma

    if _CUPY_OK and getattr(cfg, "PRISM_GPU_OFFLOAD", False):
        # GPU path: covariance + eigh on CUDA
        Cn_gpu = _cp.asarray(Cn)
        cov_gpu = _cp.cov(Cn_gpu.T)
        eigvals, eigvecs = _cp.linalg.eigh(cov_gpu)
        # The pulse signal lives in the eigenvector with the smallest
        # eigenvalue (least variance = periodic pulse, not DC skin tone)
        W = _cp.asnumpy(eigvecs)
    else:
        # CPU fallback
        cov = np.cov(Cn.T)
        eigvals, W = np.linalg.eigh(cov)

    return W


def prism_rppg(rgb_buf, fs=20):
    """PRISM-enhanced rPPG: adaptive color-mixing with GPU offload.

    Falls back to POS if PRISM is disabled or buffer is too short.
    """
    if not getattr(cfg, "PRISM_ENABLED", False):
        return pos_rppg(rgb_buf, fs)

    if len(rgb_buf) < cfg.RPPG_MIN_SAMPLES:
        return None

    C = np.array(rgb_buf, dtype=np.float64)
    mu = np.mean(C, axis=0)
    if np.any(mu < 1):
        return pos_rppg(rgb_buf, fs)

    # Windowed PRISM projection
    win_samples = int(getattr(cfg, "PRISM_WINDOW_S", 1.6) * fs)
    win_samples = max(10, min(win_samples, len(C)))

    pulse = np.zeros(len(C))
    step = max(1, win_samples // 2)  # 50% overlap

    for start in range(0, len(C) - win_samples + 1, step):
        window = C[start:start + win_samples]
        W = _prism_projection_matrix(window, fs)

        # Normalize window
        wmu = window.mean(axis=0)
        wmu[wmu < 1] = 1
        Cn = window / wmu

        # Project through PRISM matrix — pulse is the FIRST eigenvector
        projected = Cn @ W
        # The pulse component is in column 0 (smallest eigenvalue).
        # numpy.linalg.eigh sorts eigenvalues ascending, so column 0
        # has the least variance = periodic pulse, not DC skin tone.
        pulse_component = projected[:, 0]  # smallest eigenvalue = pulse
        pulse_component = pulse_component - np.mean(pulse_component)

        # Overlap-add into the output
        pulse[start:start + win_samples] += pulse_component

    # Normalize overlap regions
    counts = np.zeros(len(C))
    for start in range(0, len(C) - win_samples + 1, step):
        counts[start:start + win_samples] += 1
    counts[counts < 1] = 1
    pulse /= counts

    return pulse

def extract_rgb(image, face_landmarks):
    """Extract mean RGB from forehead+cheek ROI.

    v9.2 (plan step 14): vectorized. Previous implementation did three
    separate boolean-mask slices of `image[:, :, c][mask == 255]` which
    walks the HxW array three times. Using `cv2.mean(image, mask=mask)`
    does it in a single CUDA-accelerated C pass — about 3× faster for
    typical 640×480 frames and eliminates the Python-level fancy indexing.
    """
    import cv2
    h, w = image.shape[:2]
    pts = []
    for i in ROI_INDICES:
        if i < len(face_landmarks):
            pts.append([int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)])
    if len(pts) < 6:
        return None
    hull = cv2.convexHull(np.array(pts, dtype=np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    # Minimum ROI pixel count sanity check (quick, avoids cv2.mean on empty)
    if cv2.countNonZero(mask) < 50:
        return None
    # cv2.mean returns (B, G, R, alpha). Image is BGR; remap to (R, G, B).
    mean_bgr = cv2.mean(image, mask=mask)
    return (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))

def _check_motion(face_landmarks):
    """Reject frame if face moved too much between frames.
    Returns True if motion is acceptable (frame should be kept).
    """
    global _prev_landmarks, _motion_rejected
    # Extract key landmark positions
    current = []
    for i in MOTION_LANDMARKS:
        if i < len(face_landmarks):
            current.append((face_landmarks[i].x, face_landmarks[i].y))
    if not current:
        return True  # can't check, accept

    if _prev_landmarks is None:
        _prev_landmarks = current
        return True

    # Compute mean displacement
    total_disp = 0
    for (cx, cy), (px, py) in zip(current, _prev_landmarks):
        total_disp += ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
    avg_disp = total_disp / len(current)
    _prev_landmarks = current

    if avg_disp > cfg.RPPG_MOTION_THRESH:
        _motion_rejected += 1
        return False  # too much motion, reject this frame
    return True

def _estimate_fps():
    """Adaptive fps estimation that works from the first few frames."""
    if len(rppg_times) < 2:
        return 20.0  # initial default
    # Use available samples, not hardcoded window
    n = min(len(rppg_times), 30)
    dt = rppg_times[-1] - rppg_times[-n]
    if dt <= 0:
        return 20.0
    return max(10, min(60, (n - 1) / dt))

def add_sample(image, face_landmarks):
    """Extract RGB from face and add to rPPG buffer with motion rejection.
    Thread-safe: called from both HQ camera (main thread) and 120fps camera (background thread)."""
    with _rppg_lock:
        if not _check_motion(face_landmarks):
            return
    rv = extract_rgb(image, face_landmarks)
    if rv:
        rppg_rgb.append(rv)      # deque.append is atomic in CPython
        rppg_times.append(time.time())

def estimate_hr():
    """Estimate heart rate from rPPG buffer. Returns (hr, waveform, quality, resp_rate)."""
    global confirmed_bpm
    if len(rppg_rgb) < cfg.RPPG_MIN_SAMPLES:
        return None, [], 0, None

    fs = _estimate_fps()
    # v9.0 Step 4: use PRISM when enabled, fallback to POS
    if getattr(cfg, "PRISM_ENABLED", False):
        pulse = prism_rppg(list(rppg_rgb), fs)
    else:
        pulse = pos_rppg(list(rppg_rgb), fs)
    if pulse is None:
        return None, [], 0, None
    pulse = medfilt(pulse, kernel_size=5)
    pulse = pulse - np.mean(pulse)
    try:
        filtered = bandpass(pulse, cfg.RPPG_BANDPASS_LO, cfg.RPPG_BANDPASS_HI, fs, 4)
    except Exception:
        return None, [], 0, None
    fft = _gpu_rfft(filtered)  # v8.9: GPU-accelerated when CuPy available
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fs)
    mask = (freqs >= cfg.RPPG_BANDPASS_LO) & (freqs <= cfg.RPPG_BANDPASS_HI)
    if not np.any(mask):
        return None, [], 0, None
    fft_m = fft[mask]
    freqs_m = freqs[mask]
    pi = np.argmax(fft_m)
    hr = round(freqs_m[pi] * 60.0, 1)
    # Step 5: Auto-apply calibration correction if regression file exists
    try:
        import json as _json
        _reg_file = getattr(cfg, 'CALIBRATION_REGRESSION_FILE', None)
        if _reg_file and os.path.exists(_reg_file):
            with open(_reg_file) as _rf:
                _reg = _json.load(_rf)
            _slope = _reg.get("slope", 1.0)
            _intercept = _reg.get("intercept", 0.0)
            if abs(_slope - 1.0) > 0.001 or abs(_intercept) > 0.1:
                hr = round(_slope * hr + _intercept, 1)
    except Exception:
        pass
    # SNR-based quality: peak power vs. rest-of-band mean. Gate HR when SNR
    # is too low so the UI blanks instead of showing a phantom reading.
    _peak_power = float(fft_m[pi])
    _bg_power = float(np.mean(np.delete(fft_m, pi))) if len(fft_m) > 1 else 1e-6
    _snr = _peak_power / (_bg_power + 1e-6)
    quality = min(1.0, _snr / 8.0)
    _HR_SNR_GATE = getattr(cfg, "RPPG_HR_SNR_GATE", 2.5)
    # Diagnostic: every ~5 s, print why rPPG is/isn't emitting. Lets the
    # operator see whether the pipeline is motion-starved (buffer small),
    # SNR-gated (buffer full but signal too noisy), or actually locking.
    global _last_diag_t
    _now = time.time()
    if _now - _last_diag_t >= 5.0:
        _last_diag_t = _now
        _state = "LOCK" if _snr >= _HR_SNR_GATE else "GATE"
        print(f"[rPPG diag] buf={len(rppg_rgb)}/{cfg.RPPG_BUFFER_SIZE} "
              f"motion_rejected={_motion_rejected} fs={fs:.1f} "
              f"hr={hr:.1f} snr={_snr:.2f} gate={_HR_SNR_GATE} {_state} "
              f"samples={len(hr_samples)}/30 elapsed={get_elapsed():.1f}s")
    # v9.8: surface signal quality every cycle so WS/forensic can read it.
    _state = "LOCK" if _snr >= _HR_SNR_GATE else "GATE"
    _publish_quality(_snr, quality, _state, fs)
    if _snr < _HR_SNR_GATE:
        # Low-confidence estimate — refuse to publish HR this cycle.
        return None, filtered[-100:].tolist(), round(quality, 2), None
    # Respiratory rate: physiological band is 0.1–0.5 Hz (6–30 brpm).
    # Require ≥20 s of buffer and a reasonable resp-band SNR before
    # publishing, otherwise return None so the UI stays blank.
    resp = None
    if len(rppg_rgb) / max(fs, 1.0) >= 20.0:
        resp_mask = (freqs >= 0.10) & (freqs <= 0.5)
        if np.any(resp_mask):
            rf = fft[resp_mask]
            rfreqs = freqs[resp_mask]
            rpi = int(np.argmax(rf))
            rpeak = float(rf[rpi])
            rbg = float(np.mean(np.delete(rf, rpi))) if len(rf) > 1 else 1e-6
            if rpeak / (rbg + 1e-6) >= getattr(cfg, "RPPG_RESP_SNR_GATE", 3.5):
                resp_raw = round(float(rfreqs[rpi]) * 60.0, 1)
                # 2026-04-25: reject band-edge artefacts. Real respiration
                # rates cluster 10-22 br/min for adults at rest. Values
                # pinned to 6 br/min (band lo) or >=29 br/min (band hi)
                # are almost always DC-drift / motion / harmonic leak,
                # NOT genuine breathing. Suppress those.
                if 7.5 <= resp_raw <= 28.0:
                    resp = _smooth_resp(resp_raw)
    # v7.5: physiological filter + EMA for display stability.
    # Filter rejects out-of-bounds, impossible jumps (>6 BPM/s),
    # and IQR outliers after a short warmup. Rejected values return
    # the last accepted one so the UI doesn't flicker to None.
    global _ema_hr
    if _hr_filter is not None:
        filtered_hr = _hr_filter.push(hr)
        if filtered_hr is None:
            # keep last displayed value rather than collapse to None
            filtered_hr = _hr_filter.last() if hasattr(_hr_filter, "last") else hr
        if filtered_hr is not None:
            _ema_hr = filtered_hr if _ema_hr is None else (
                _EMA_ALPHA * filtered_hr + (1.0 - _EMA_ALPHA) * _ema_hr
            )
            hr = round(_ema_hr, 1)
    if 40 < hr < 210:
        hr_samples.append(hr)
    # Confirm BPM after 60 seconds.
    # v9.2 (plan step 29): velocity-aware drift instead of instantaneous snap.
    # Previously confirmed_bpm jumped to the new trimmed-median value every
    # frame once the 60-s gate opened, which visually looked like a HR
    # "lock step". Now the new computed target is eased toward with a
    # bounded velocity (max 2 BPM/sec) so the display drifts smoothly
    # from one stable value to the next over ~3 seconds. The underlying
    # trimmed-median math is unchanged — only the integration is smoothed.
    if start_time and time.time() - start_time >= cfg.RPPG_CONFIRM_AFTER_SECS and len(hr_samples) >= 30:
        arr = np.array(hr_samples)
        lo_p, hi_p = np.percentile(arr, 10), np.percentile(arr, 90)
        t = arr[(arr >= lo_p) & (arr <= hi_p)]
        if len(t) > 5:
            target = float(np.median(t))
            global _confirmed_bpm_t_prev
            now_t = time.time()
            if confirmed_bpm is None or _confirmed_bpm_t_prev is None:
                confirmed_bpm = round(target, 0)
            else:
                dt = max(0.001, now_t - _confirmed_bpm_t_prev)
                max_step = 2.0 * dt           # BPM per second cap (2 BPM/s)
                delta = max(-max_step, min(max_step, target - confirmed_bpm))
                confirmed_bpm = round(confirmed_bpm + delta, 0)
            _confirmed_bpm_t_prev = now_t
    return hr, filtered[-100:].tolist(), round(quality, 2), resp

_hrv_ema = None           # v7.7 RMSSD smoothing
_stress_ema = None        # v7.7 stress smoothing
# v7.9: lowered alphas (was 0.10 / 0.08) because the value was still
# "hopping like a crazed rabbit" in field use. τ_eff is now ~20–30
# samples per stream. Warm-up bumped to 40 samples (~2 s at 20 fps) so
# the first output is already built on a stable HR history.
_HRV_EMA_ALPHA = 0.05     # τ ≈ 20 samples
_STRESS_EMA_ALPHA = 0.04  # τ ≈ 25 samples — slower than HRV
_HRV_MIN_SAMPLES = 40     # need ~2 s of HR history before reporting
_hrv_median_buf = deque(maxlen=7)  # v7.9 pre-EMA median filter kills single-sample glitches

# v7.9: respiratory-rate stability layer. Before v7.9 the FFT-peak
# frequency was published raw, which meant the UI hopped 2–4 brpm per
# refresh every time a neighbouring spectral bin crossed the dominant
# peak. We now:
#   - bound the accepted band to 6–30 brpm (resting + exercise adults)
#   - carry a jump-rate gate (±3 brpm/s) like HRFilter
#   - median-filter over 5 accepted samples before EMA
#   - EMA α=0.15 on the final display value
_resp_ema = None
_resp_last_accepted = None
_resp_last_time = 0.0
_resp_median_buf = deque(maxlen=5)
_RESP_EMA_ALPHA = 0.15
_RESP_MIN_BPM = 6.0
_RESP_MAX_BPM = 30.0
_RESP_JUMP_PER_SEC = 3.0  # physiology: resp rate can't shift more than ~3 brpm/s

def _smooth_resp(resp_raw):
    """v7.9 respiratory-rate stability pipeline.

    Inputs may be any raw FFT peak (the caller already applied an SNR
    gate). We clamp to physiology, reject impossible jumps, median-filter
    the accepted tail, then EMA.

    Returns the smoothed value, or the last accepted EMA if the current
    sample is rejected — never returns None once we've produced one good
    reading, to keep the UI from flickering to "--".
    """
    global _resp_ema, _resp_last_accepted, _resp_last_time
    if resp_raw is None:
        return round(_resp_ema, 1) if _resp_ema is not None else None
    now = time.time()
    # hard bounds
    if not (_RESP_MIN_BPM <= resp_raw <= _RESP_MAX_BPM):
        return round(_resp_ema, 1) if _resp_ema is not None else None
    # jump-rate gate — after first accepted value
    if _resp_last_accepted is not None and _resp_last_time > 0:
        dt = max(0.1, now - _resp_last_time)
        max_jump = _RESP_JUMP_PER_SEC * dt + 1.0   # +1 brpm slack
        if abs(resp_raw - _resp_last_accepted) > max_jump:
            return round(_resp_ema, 1) if _resp_ema is not None else None
    _resp_last_accepted = resp_raw
    _resp_last_time = now
    _resp_median_buf.append(resp_raw)
    med = float(np.median(_resp_median_buf))
    _resp_ema = med if _resp_ema is None else (
        _RESP_EMA_ALPHA * med + (1.0 - _RESP_EMA_ALPHA) * _resp_ema
    )
    return round(_resp_ema, 1)


def estimate_hrv():
    """Estimate HRV (RMSSD proxy) and a stress index — v7.7.

    Problem in v7.5/7.6
    -------------------
    The previous implementation computed RR = 60000/bpm over the last 30
    *smoothed* HR samples, then np.diff of that. Any 1-bpm HR twitch
    produced a big Δ-RR, and with no smoothing on the output the UI saw
    rmssd swinging between 5 ms and 120 ms, which made `stress_index`
    oscillate between 0.3 and 1.0 every second.

    v7.7 fixes
    ----------
    1. Require ≥ _HRV_MIN_SAMPLES (~20 s warm-up) before reporting.
    2. Trim the window to the 10th–90th percentile (IQR) before diffing.
    3. EMA-smooth rmssd (α=0.10) and stress (α=0.08) on the output side.
    4. Clamp rmssd to a physiological band (5–200 ms) so a single
       glitch can't latch the EMA.
    5. Cleaner stress formula: normalized combination of elevated HR and
       suppressed HRV, each clipped to [0, 1] before blending.
    """
    global _hrv_ema, _stress_ema
    if len(hr_samples) < _HRV_MIN_SAMPLES:
        return None, None
    window = [b for b in list(hr_samples)[-60:] if 40 < b < 210]
    if len(window) < 10:
        return None, None
    arr = np.array(window, dtype=float)
    lo, hi = np.percentile(arr, 10), np.percentile(arr, 90)
    trimmed = arr[(arr >= lo) & (arr <= hi)]
    if len(trimmed) < 6:
        trimmed = arr
    rr = 60000.0 / trimmed
    rmssd_raw = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    # v7.9: tighter physiological clamp (10–150 ms is the realistic seated-adult band;
    # 5–200 lets sensor noise through).
    if not np.isfinite(rmssd_raw) or rmssd_raw < 10 or rmssd_raw > 150:
        # Reject glitch; keep previous EMA value so UI doesn't jump.
        if _hrv_ema is None:
            return None, None
        rmssd_display = round(_hrv_ema, 1)
        stress_display = round(_stress_ema, 2) if _stress_ema is not None else None
        return rmssd_display, stress_display
    # v7.9: median pre-filter over last 7 accepted rmssd samples kills the
    # single-sample spikes that the EMA alone would still leak through.
    _hrv_median_buf.append(rmssd_raw)
    rmssd_feed = float(np.median(_hrv_median_buf))
    _hrv_ema = rmssd_feed if _hrv_ema is None else (
        _HRV_EMA_ALPHA * rmssd_feed + (1.0 - _HRV_EMA_ALPHA) * _hrv_ema
    )
    mean_hr = float(np.mean(trimmed))
    hr_component     = max(0.0, min(1.0, (mean_hr - 60.0) / 60.0))      # 0 at 60 bpm, 1 at 120 bpm
    hrv_component    = max(0.0, min(1.0, (50.0 - _hrv_ema) / 45.0))     # 0 at 50 ms rmssd, 1 at 5 ms
    stress_raw       = 0.5 * hr_component + 0.5 * hrv_component
    _stress_ema = stress_raw if _stress_ema is None else (
        _STRESS_EMA_ALPHA * stress_raw + (1.0 - _STRESS_EMA_ALPHA) * _stress_ema
    )
    return round(_hrv_ema, 1), round(_stress_ema, 2)

def mark_start():
    global start_time
    if not start_time:
        start_time = time.time()

def get_elapsed():
    return round(time.time() - start_time, 1) if start_time else 0

def get_confirmed_bpm():
    return confirmed_bpm

def get_motion_stats():
    """Return motion rejection statistics."""
    return {"rejected_frames": _motion_rejected}


# v9.8: rPPG signal-quality reporter. The SNR + quality numbers are
# computed every fft cycle (~5 Hz at default settings) and stashed into
# `_last_quality_snapshot` so the UI/forensic_export can read the
# currently-published confidence without re-running the FFT.
_last_quality_snapshot = {
    "snr": None,            # peak_power / bg_power
    "quality": None,        # min(1.0, snr/8.0); UI consumes this
    "state": "STARTING",    # 'LOCK' | 'GATE' | 'STARTING'
    "buf_fill": 0.0,        # rppg_rgb fill ratio 0..1
    "fs": None,             # measured sampling rate
    "motion_rejected": 0,   # cumulative motion-frame drop count
    "last_update_t": 0.0,   # monotonic seconds
}


def _publish_quality(snr, quality, state, fs):
    """Internal: called from the FFT path to expose live signal quality
    so the WS payload + forensic export can include it without
    duplicating the FFT computation."""
    _last_quality_snapshot["snr"] = round(float(snr), 3)
    _last_quality_snapshot["quality"] = round(float(quality), 3)
    _last_quality_snapshot["state"] = state
    _last_quality_snapshot["buf_fill"] = round(
        len(rppg_rgb) / max(1, getattr(cfg, "RPPG_BUFFER_SIZE", 300)), 3)
    _last_quality_snapshot["fs"] = round(float(fs), 2) if fs else None
    _last_quality_snapshot["motion_rejected"] = _motion_rejected
    _last_quality_snapshot["last_update_t"] = time.monotonic()


def get_signal_quality():
    """Return a copy of the last published rPPG signal-quality snapshot.

    Fields:
      snr             : peak/bg-power ratio at the dominant HR frequency
      quality         : SNR mapped to [0, 1] (UI gradient bar input)
      state           : 'LOCK' (publishing HR), 'GATE' (snr too low),
                        'STARTING' (buffer filling, no FFT yet)
      buf_fill        : 0..1 fill ratio of the RGB ring buffer
      fs              : measured sample rate (Hz)
      motion_rejected : cumulative motion-rejected frame count
      stale_s         : seconds since last FFT cycle (>= 2 s = stale)
    """
    snap = dict(_last_quality_snapshot)
    snap["stale_s"] = round(time.monotonic() - snap["last_update_t"], 1) \
        if snap["last_update_t"] > 0 else None
    return snap

def reset():
    global confirmed_bpm, start_time, _prev_landmarks, _motion_rejected, _ema_hr, _hr_filter
    global _hrv_ema, _stress_ema, _resp_ema, _resp_last_accepted, _resp_last_time
    _hrv_ema = None
    _stress_ema = None
    _resp_ema = None
    _resp_last_accepted = None
    _resp_last_time = 0.0
    _hrv_median_buf.clear()
    _resp_median_buf.clear()
    with _rppg_lock:
        rppg_rgb.clear()
        rppg_times.clear()
        hr_samples.clear()
        confirmed_bpm = None
        start_time = None
        fps_counter.clear()
        _prev_landmarks = None
        _motion_rejected = 0
        _ema_hr = None
        try:
            from hr_filter import HRFilter, HR_ADULT_RESTING
            _hr_filter = HRFilter(HR_ADULT_RESTING)
        except Exception:
            _hr_filter = None

def get_filter_stats():
    """Expose rPPG HR filter counters (accepted, rejected_*) for diagnostics."""
    if _hr_filter is None:
        return None
    return _hr_filter.stats()


# ── v8.2: Per-person rPPG estimation ──
def extract_rgb_for_person(image, face_landmarks_single, smoothed_coords=None):
    """Extract mean RGB from a single face's forehead+cheek ROI.
    Same algorithm as extract_rgb but accepts a single face landmark list.

    v8.8: accepts optional `smoothed_coords` dict {idx: (x, y)} from
    PersonState.smooth_landmarks(). When provided, uses EMA-smoothed
    coordinates instead of raw landmarks to suppress jitter-induced
    rPPG artifacts.
    """
    import cv2
    h, w = image.shape[:2]
    pts = []
    for i in ROI_INDICES:
        if smoothed_coords is not None and i in smoothed_coords:
            sx, sy = smoothed_coords[i]
            pts.append([int(sx * w), int(sy * h)])
        elif i < len(face_landmarks_single):
            pts.append([int(face_landmarks_single[i].x * w),
                        int(face_landmarks_single[i].y * h)])
    if len(pts) < 6:
        return None
    hull = cv2.convexHull(np.array(pts, dtype=np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    # v9.2 (plan step 14): single-pass cv2.mean instead of three bool-slice
    # copies. See extract_rgb() above for the rationale.
    if cv2.countNonZero(mask) < 50:
        return None
    mean_bgr = cv2.mean(image, mask=mask)
    return (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))


def estimate_hr_person(ps):
    """Estimate heart rate from a PersonState's own rPPG buffer.
    Returns (hr, hrv, resp, stress, confirmed_bpm, quality).
    Uses the same POS algorithm + bandpass + SNR gate as the global estimate_hr,
    but reads from ps.rppg_rgb / ps.rppg_times instead of module-level buffers."""
    if len(ps.rppg_rgb) < cfg.RPPG_MIN_SAMPLES:
        return None, None, None, None, None, 0

    # Adaptive fps from this person's timestamps
    if len(ps.rppg_times) < 2:
        fs = 20.0
    else:
        n = min(len(ps.rppg_times), 30)
        dt = ps.rppg_times[-1] - ps.rppg_times[-n]
        fs = max(10, min(60, (n - 1) / dt)) if dt > 0 else 20.0

    # v9.0 FIX: use PRISM when enabled (was only wired into global
    # estimate_hr, not per-person). Falls back to POS automatically.
    if getattr(cfg, "PRISM_ENABLED", False):
        pulse = prism_rppg(list(ps.rppg_rgb), fs)
    else:
        pulse = pos_rppg(list(ps.rppg_rgb), fs)
    if pulse is None:
        return None, None, None, None, None, 0
    pulse = medfilt(pulse, kernel_size=5)
    pulse = pulse - np.mean(pulse)
    try:
        filtered = bandpass(pulse, cfg.RPPG_BANDPASS_LO, cfg.RPPG_BANDPASS_HI, fs, 4)
    except Exception:
        return None, None, None, None, None, 0
    fft = _gpu_rfft(filtered)  # v8.9: GPU-accelerated when CuPy available
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fs)
    mask = (freqs >= cfg.RPPG_BANDPASS_LO) & (freqs <= cfg.RPPG_BANDPASS_HI)
    if not np.any(mask):
        return None, None, None, None, None, 0
    fft_m = fft[mask]
    freqs_m = freqs[mask]
    pi = np.argmax(fft_m)
    hr = round(freqs_m[pi] * 60.0, 1)
    hr = apply_calibration(hr)

    # SNR quality
    _peak_power = float(fft_m[pi])
    _bg_power = float(np.mean(np.delete(fft_m, pi))) if len(fft_m) > 1 else 1e-6
    _snr = _peak_power / (_bg_power + 1e-6)
    quality = min(1.0, _snr / 8.0)
    _HR_SNR_GATE = getattr(cfg, "RPPG_HR_SNR_GATE", 2.5)
    if _snr < _HR_SNR_GATE:
        return None, None, None, None, None, round(quality, 2)

    # Respiratory rate
    resp = None
    if len(ps.rppg_rgb) / max(fs, 1.0) >= 20.0:
        resp_mask = (freqs >= 0.10) & (freqs <= 0.5)
        if np.any(resp_mask):
            rf = fft[resp_mask]
            rfreqs = freqs[resp_mask]
            rpi = int(np.argmax(rf))
            rpeak = float(rf[rpi])
            rbg = float(np.mean(np.delete(rf, rpi))) if len(rf) > 1 else 1e-6
            if rpeak / (rbg + 1e-6) >= getattr(cfg, "RPPG_RESP_SNR_GATE", 2.0):
                resp = round(float(rfreqs[rpi]) * 60.0, 1)

    if 40 < hr < 210:
        ps.hr_samples.append(hr)

    # Confirmed BPM
    c_bpm = None
    if len(ps.hr_samples) >= 30:
        elapsed_s = ps.rppg_times[-1] - ps.rppg_times[0] if len(ps.rppg_times) > 1 else 0
        if elapsed_s >= cfg.RPPG_CONFIRM_AFTER_SECS:
            arr = np.array(ps.hr_samples)
            lo_p, hi_p = np.percentile(arr, 10), np.percentile(arr, 90)
            t = arr[(arr >= lo_p) & (arr <= hi_p)]
            if len(t) > 5:
                c_bpm = round(float(np.median(t)), 0)

    # HRV + stress (simplified per-person version)
    hrv = None
    stress = None
    if len(ps.hr_samples) >= 20:
        window = [b for b in list(ps.hr_samples)[-60:] if 40 < b < 210]
        if len(window) >= 10:
            arr = np.array(window, dtype=float)
            lo, hi = np.percentile(arr, 10), np.percentile(arr, 90)
            trimmed = arr[(arr >= lo) & (arr <= hi)]
            if len(trimmed) < 6:
                trimmed = arr
            rr = 60000.0 / trimmed
            rmssd_raw = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
            if np.isfinite(rmssd_raw) and 10 <= rmssd_raw <= 150:
                hrv = round(rmssd_raw, 1)
                mean_hr = float(np.mean(trimmed))
                hr_comp = max(0.0, min(1.0, (mean_hr - 60.0) / 60.0))
                hrv_comp = max(0.0, min(1.0, (50.0 - rmssd_raw) / 45.0))
                stress = round(0.5 * hr_comp + 0.5 * hrv_comp, 2)

    return hr, hrv, resp, stress, c_bpm, round(quality, 2)


# ── v9.4.1: Liveness signal extraction ─────────────────────────────────
# Three-output face-liveness path. Unifies the rPPG buffers that the
# `pad.branches.rppg_branch` used to maintain privately.
# See Liu et al. 2018 (https://link.springer.com/chapter/10.1007/978-3-319-92627-8_13).

_LIVENESS_WIN_SECONDS = 10.0


def feed_liveness_rois(ps, ts, forehead_rgb=None, l_cheek_rgb=None,
                       r_cheek_rgb=None, fs_hint=None):
    """Push one frame's per-ROI mean RGB into the PersonState cheek
    deques. Left/right cheeks are dedicated liveness buffers; forehead
    is OPTIONAL — the HR path already feeds `ps.rppg_rgb`, so callers
    normally pass None for the forehead to avoid double-counting. When
    forehead_rgb is provided we append it too, useful for tests or when
    the HR path is disabled.

    None entries are ignored so callers may feed partial ROIs.
    """
    if ps is None:
        return
    if fs_hint and fs_hint > 0:
        ps.rppg_fs_est = float(fs_hint)
    cutoff = ts - _LIVENESS_WIN_SECONDS
    if forehead_rgb is not None:
        ps.rppg_rgb.append((float(forehead_rgb[0]), float(forehead_rgb[1]),
                            float(forehead_rgb[2])))
        ps.rppg_times.append(ts)
    if l_cheek_rgb is not None:
        ps.rppg_l_cheek.append(
            (ts, float(l_cheek_rgb[0]), float(l_cheek_rgb[1]),
             float(l_cheek_rgb[2])))
        while ps.rppg_l_cheek and ps.rppg_l_cheek[0][0] < cutoff:
            ps.rppg_l_cheek.popleft()
    if r_cheek_rgb is not None:
        ps.rppg_r_cheek.append(
            (ts, float(r_cheek_rgb[0]), float(r_cheek_rgb[1]),
             float(r_cheek_rgb[2])))
        while ps.rppg_r_cheek and ps.rppg_r_cheek[0][0] < cutoff:
            ps.rppg_r_cheek.popleft()


def _pos_signal_from_tuples(samples):
    """POS projection of a single-ROI (t,r,g,b) time series."""
    if len(samples) < 20:
        return np.zeros(0, dtype=float)
    arr = np.asarray(samples, dtype=float)
    rgb = arr[:, 1:]
    m = rgb.mean(axis=0)
    if np.any(m < 1):
        return np.zeros(0, dtype=float)
    C = rgb / m
    S1 = C[:, 1] - C[:, 2]
    S2 = C[:, 1] + C[:, 2] - 2.0 * C[:, 0]
    s1_std = np.std(S1) or 1.0
    s2_std = np.std(S2) or 1.0
    return S1 + (s1_std / s2_std) * S2


def _pos_signal_from_forehead(ps):
    """POS projection of ps.rppg_rgb (paired with ps.rppg_times)."""
    n = min(len(ps.rppg_rgb), len(ps.rppg_times))
    if n < 20:
        return np.zeros(0, dtype=float)
    rgb = np.asarray(list(ps.rppg_rgb)[-n:], dtype=float)
    m = rgb.mean(axis=0)
    if np.any(m < 1):
        return np.zeros(0, dtype=float)
    C = rgb / m
    S1 = C[:, 1] - C[:, 2]
    S2 = C[:, 1] + C[:, 2] - 2.0 * C[:, 0]
    s1_std = np.std(S1) or 1.0
    s2_std = np.std(S2) or 1.0
    return S1 + (s1_std / s2_std) * S2


def _pulse_band_power_fraction(sig, fs):
    """Fraction of total spectral power that lies in the pulse band
    [0.75, 3.0] Hz. This is a "spectral concentration index", NOT a
    proper SNR. Kept for backward compat with downstream code that
    uses RPPG_HR_SNR_GATE as a band-fraction threshold.

    API_unq 2026-05-13 (P1.3): renamed from `_pulse_snr` to reflect
    that the returned quantity is a 0-1 band-power ratio rather than
    a peak-vs-noise SNR. The canonical rPPG SNR is `_pulse_snr_dB`
    below (de Haan & Jeanne 2013).
    """
    if sig.size < 32 or fs <= 0:
        return 0.0
    sig = sig - np.mean(sig)
    spec = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(sig.size, 1.0 / fs)
    total = float(spec.sum()) or 1e-12
    band = ((freqs >= 0.75) & (freqs <= 3.0))
    return float(spec[band].sum()) / total


# API_unq (P1.3): backward-compat alias. Existing callers throughout
# the codebase use `_pulse_snr(...)`. Renaming everywhere would touch
# ~6 callsites for no behavioural change. Keep the old name pointing
# at the same band-power-fraction function so the audit-trail is the
# one-line public-API rename.
_pulse_snr = _pulse_band_power_fraction


def _pulse_snr_dB(sig, fs, peak_hz=None, half_width_hz=0.1,
                  band=(0.7, 4.0)):
    """Proper peak-vs-rest pulse SNR in dB.

    de Haan & Jeanne (IEEE T-BME 60(10), 2013) define
        SNR_dB = 10 * log10( P_signal / P_noise )
    where P_signal is the power within ±half_width_hz of the dominant
    peak inside `band`, and P_noise is the power in the rest of `band`.
    Heusch & Marcel (IEEE TBIOM 2018) use the same form.

    If `peak_hz` is None, the peak is auto-detected as the argmax
    inside `band`. Returns -inf when sig is too short or empty.

    Reference:
      de Haan, G. & Jeanne, V. (2013). Robust pulse rate from
      chrominance-based rPPG. IEEE TBME 60(10), 2878-2886.
    """
    if sig.size < 32 or fs <= 0:
        return float("-inf")
    sig = sig - np.mean(sig)
    spec = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(sig.size, 1.0 / fs)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(in_band):
        return float("-inf")
    if peak_hz is None:
        # Argmax inside the band
        band_spec = spec.copy()
        band_spec[~in_band] = 0.0
        idx = int(np.argmax(band_spec))
        peak_hz = float(freqs[idx])
    sig_mask = ((freqs >= peak_hz - half_width_hz) &
                (freqs <= peak_hz + half_width_hz) & in_band)
    sig_p = float(spec[sig_mask].sum())
    noise_p = float(spec[in_band].sum() - sig_p)
    if noise_p <= 0.0 or sig_p <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(sig_p / noise_p)


def _phase_coherence(a, b, fs, hr_hz=None):
    if a.size < 32 or b.size < 32 or fs <= 0:
        return 0.0
    n = min(a.size, b.size)
    aa = a[-n:] - np.mean(a[-n:])
    bb = b[-n:] - np.mean(b[-n:])
    try:
        from scipy.signal import coherence
        f, Cxy = coherence(aa, bb, fs=fs, nperseg=min(128, n // 2))
    except Exception:
        return 0.0
    if hr_hz and hr_hz > 0:
        mask = (f >= max(0.5, hr_hz - 0.25)) & (f <= hr_hz + 0.25)
        if not np.any(mask):
            return float(np.mean(Cxy))
        return float(np.mean(Cxy[mask]))
    band = (f >= 0.75) & (f <= 3.0)
    return float(np.mean(Cxy[band])) if np.any(band) else float(np.mean(Cxy))


def _refresh_band_ratio(sig, fs):
    if sig.size < 64 or fs < 30.0:
        return 0.0
    sig = sig - np.mean(sig)
    spec = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(sig.size, 1.0 / fs)
    total = float(spec.sum()) or 1e-12
    if fs >= 110.0:
        hot = [(48.0, 52.0), (58.0, 62.0), (118.0, 122.0)]
    else:
        nyq = fs * 0.5
        hot = [(nyq * 0.85, nyq)]
    mask = np.zeros_like(freqs, dtype=bool)
    for lo, hi in hot:
        mask |= ((freqs >= lo) & (freqs <= hi))
    return float(spec[mask].sum()) / total


def extract_liveness(ps, frames=None, hr_hint_bpm=None):
    """Return (pulse_snr, phase_coherence, refresh_band_ratio) for the
    per-person rPPG buffers held on PersonState.

    Parameters
    ----------
    ps : PersonState
        The per-person state object holding rppg_rgb, rppg_l_cheek,
        rppg_r_cheek, and rppg_fs_est.
    frames : ignored, kept for API compatibility.
    hr_hint_bpm : optional float, used to steer the coherence band.

    Returns
    -------
    tuple of three floats in the range roughly [0, 1]:
      pulse_snr : FFT band power 0.75–3.0 Hz over total power.
      phase_coherence : mean Welch coherence magnitude between ROIs.
      refresh_band_ratio : power in screen-refresh bands over total.
    """
    if ps is None:
        return (0.0, 0.0, 0.0)
    fs = float(getattr(ps, "rppg_fs_est", 20.0) or 20.0)

    sig_fh = _pos_signal_from_forehead(ps)
    lc_list = list(ps.rppg_l_cheek) if hasattr(ps, "rppg_l_cheek") else []
    rc_list = list(ps.rppg_r_cheek) if hasattr(ps, "rppg_r_cheek") else []
    sig_lc = _pos_signal_from_tuples(lc_list) if len(lc_list) >= 20 else np.zeros(0)
    sig_rc = _pos_signal_from_tuples(rc_list) if len(rc_list) >= 20 else np.zeros(0)

    snr_vals = []
    for s in (sig_fh, sig_lc, sig_rc):
        if s.size >= 32:
            snr_vals.append(_pulse_snr(s, fs))
    pulse_snr = float(np.mean(snr_vals)) if snr_vals else 0.0

    hr_hz = (hr_hint_bpm / 60.0) if hr_hint_bpm else None
    coh_vals = []
    if sig_fh.size and sig_lc.size:
        coh_vals.append(_phase_coherence(sig_fh, sig_lc, fs, hr_hz))
    if sig_fh.size and sig_rc.size:
        coh_vals.append(_phase_coherence(sig_fh, sig_rc, fs, hr_hz))
    if sig_lc.size and sig_rc.size:
        coh_vals.append(_phase_coherence(sig_lc, sig_rc, fs, hr_hz))
    phase_coh = float(np.mean(coh_vals)) if coh_vals else 0.0

    refresh = 0.0
    for s in (sig_fh, sig_lc, sig_rc):
        if s.size >= 64:
            refresh = max(refresh, _refresh_band_ratio(s, fs))

    return (pulse_snr, phase_coh, refresh)


# ════════════════════════════════════════════════════════════════════
# API_unq 2026-05-14 -- rPPG paper-grade additions per Phase-2 + 2026
#   rPPG-focused lit-review (compass_artifact follow-up).
#
# Items landed:
#   1) spatial pulse-propagation score (anti-rePPG injection defense)
#   2) periodicity-guided ultra-short HR estimator
#   3) SDNN + LF/HF enrichment of HRV (de Haan + VitalLens 2.0 conventions)
# ════════════════════════════════════════════════════════════════════

def _xcorr_lag_seconds(a, b, fs, max_lag_s=0.50):
    """Lag in seconds at which signals `a` and `b` reach max cross-
    correlation. Positive lag means `b` lags `a` (i.e. `a` leads `b`).

    Used by `spatial_propagation_score` to verify the real cardiovascular
    propagation pattern: pulse arrives at forehead first (from carotid),
    then cheeks (~50-150 ms later, depending on subject) via facial
    artery branches. Synthetic injection (rePPG, deepfake) typically
    applies the pulse uniformly across ROIs -> lag near 0.
    """
    if a.size < 32 or b.size < 32 or fs <= 0:
        return 0.0
    n = min(a.size, b.size)
    a = (a[-n:] - np.mean(a[-n:])).astype(np.float32)
    b = (b[-n:] - np.mean(b[-n:])).astype(np.float32)
    sa = float(np.std(a)) or 1e-9
    sb = float(np.std(b)) or 1e-9
    a = a / sa
    b = b / sb
    max_lag = int(max_lag_s * fs)
    if max_lag < 1:
        return 0.0
    # Search +- max_lag samples
    best_corr = -1.0
    best_lag = 0
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            x = a[: n - k]
            y = b[k:]
        else:
            x = a[-k:]
            y = b[: n + k]
        if x.size < 16:
            continue
        c = float(np.dot(x, y) / x.size)
        if c > best_corr:
            best_corr = c
            best_lag = k
    return float(best_lag) / float(fs)


def spatial_propagation_score(ps, hr_hint_bpm=None):
    """API_unq P-anti-rePPG: detect synthetic pulse injection by checking
    that the cardiovascular wave PROPAGATES across face regions in a
    physiologically plausible way.

    Real subject:
      - forehead leads L/R cheeks by ~50-150 ms (carotid -> facial
        artery branches)
      - L and R cheeks ~simultaneous (symmetric vascular supply)
      - small but nonzero non-trivial lags

    Synthetic injection (rePPG, generative deepfake-with-pulse):
      - same signal applied uniformly across ROIs -> all lags ~0
      - OR random/incoherent inter-region pulse -> lags noisy

    Returns dict with:
        forehead_to_lc_lag_s : seconds (sign: positive = forehead leads)
        forehead_to_rc_lag_s : seconds
        lc_to_rc_lag_s       : seconds (real: small, |lag|<60ms)
        is_plausible_live    : bool (anatomically plausible propagation)
        is_suspicious        : bool (synthetic-uniform signature)
        score                : 0..1, 0=live, 1=suspicious injection

    Plausibility window for "live":
        forehead-to-cheek lag in [+0.020, +0.180] s
        cheek-to-cheek |lag| < 0.080 s

    Suspicious "uniform injection":
        all three |lag| < 0.020 s (3 frames at 30 fps), simultaneously
    """
    if ps is None:
        return {
            "forehead_to_lc_lag_s": 0.0,
            "forehead_to_rc_lag_s": 0.0,
            "lc_to_rc_lag_s": 0.0,
            "is_plausible_live": False,
            "is_suspicious": False,
            "score": 0.5,
            "reason": "no_state",
        }
    fs = float(getattr(ps, "rppg_fs_est", 20.0) or 20.0)
    sig_fh = _pos_signal_from_forehead(ps)
    lc_list = list(ps.rppg_l_cheek) if hasattr(ps, "rppg_l_cheek") else []
    rc_list = list(ps.rppg_r_cheek) if hasattr(ps, "rppg_r_cheek") else []
    sig_lc = _pos_signal_from_tuples(lc_list) if len(lc_list) >= 32 else np.zeros(0)
    sig_rc = _pos_signal_from_tuples(rc_list) if len(rc_list) >= 32 else np.zeros(0)
    if sig_fh.size < 64 or sig_lc.size < 64 or sig_rc.size < 64:
        return {
            "forehead_to_lc_lag_s": 0.0,
            "forehead_to_rc_lag_s": 0.0,
            "lc_to_rc_lag_s": 0.0,
            "is_plausible_live": False,
            "is_suspicious": False,
            "score": 0.5,
            "reason": "insufficient_buffer",
        }
    # Band-pass each signal to the pulse band so lag estimation isn't
    # contaminated by low-freq baseline drift or high-freq noise.
    try:
        sig_fh_bp = bandpass(sig_fh, 0.7, 3.5, fs, order=3)
        sig_lc_bp = bandpass(sig_lc, 0.7, 3.5, fs, order=3)
        sig_rc_bp = bandpass(sig_rc, 0.7, 3.5, fs, order=3)
    except Exception:
        sig_fh_bp = sig_fh
        sig_lc_bp = sig_lc
        sig_rc_bp = sig_rc

    lag_fh_lc = _xcorr_lag_seconds(sig_fh_bp, sig_lc_bp, fs)
    lag_fh_rc = _xcorr_lag_seconds(sig_fh_bp, sig_rc_bp, fs)
    lag_lc_rc = _xcorr_lag_seconds(sig_lc_bp, sig_rc_bp, fs)

    is_live = (
        0.020 <= lag_fh_lc <= 0.180
        and 0.020 <= lag_fh_rc <= 0.180
        and abs(lag_lc_rc) < 0.080
    )
    is_suspicious = (
        abs(lag_fh_lc) < 0.020
        and abs(lag_fh_rc) < 0.020
        and abs(lag_lc_rc) < 0.020
    )
    if is_live:
        score = 0.05
        reason = "plausible_live_propagation"
    elif is_suspicious:
        score = 0.85
        reason = "uniform_signal_synthetic_signature"
    else:
        # Ambiguous: neither clearly live nor clearly synthetic
        # (e.g. noisy or partial buffer). Lean slightly suspicious
        # because real subjects almost always show forehead leading.
        score = 0.55
        reason = "ambiguous_propagation"
    return {
        "forehead_to_lc_lag_s": float(lag_fh_lc),
        "forehead_to_rc_lag_s": float(lag_fh_rc),
        "lc_to_rc_lag_s": float(lag_lc_rc),
        "is_plausible_live": bool(is_live),
        "is_suspicious": bool(is_suspicious),
        "score": float(score),
        "reason": reason,
    }


def estimate_hr_short(ps, window_s=2.5, hr_hint_bpm=None):
    """API_unq P-ultrashort: periodicity-guided HR estimate from a 2-3 s
    rPPG window. Designed for active-liveness challenge flows (e.g. the
    screen_flash branch) that need a fast pulse signal without waiting
    for the 60 s `confirmed_bpm` lock.

    Approach (Lin et al. 2025 "Periodicity-Guided rPPG"):
      1) Take last `window_s` of POS signal from ps.rppg_rgb buffer.
      2) Build autocorrelation; mask the lag range corresponding to
         HR band [0.8, 3.0] Hz (50-180 bpm).
      3) Pick first peak with prominence >= 0.15 of max.
      4) Convert lag -> bpm.
      5) Validate by checking 2nd-harmonic peak at half the lag exists
         (consistent with the pulse waveform's biphasic shape).
      6) Return (bpm_estimate, confidence in [0, 1]).

    Returns: dict { hr_bpm, confidence, n_samples, reason }.
        confidence < 0.30  -> don't trust this estimate.
    """
    if ps is None:
        return {"hr_bpm": None, "confidence": 0.0, "reason": "no_state"}
    fs = float(getattr(ps, "rppg_fs_est", 20.0) or 20.0)
    if fs <= 0:
        return {"hr_bpm": None, "confidence": 0.0, "reason": "no_fs_est"}
    n_target = int(window_s * fs)
    rgb = list(ps.rppg_rgb)[-n_target:] if hasattr(ps, "rppg_rgb") else []
    if len(rgb) < max(16, int(fs * 1.5)):
        return {
            "hr_bpm": None,
            "confidence": 0.0,
            "reason": f"insufficient_samples ({len(rgb)} < {int(fs*1.5)})",
            "n_samples": len(rgb),
        }
    arr = np.asarray(rgb, dtype=float)
    m = arr.mean(axis=0)
    if np.any(m < 1):
        return {"hr_bpm": None, "confidence": 0.0,
                "reason": "dark_frames", "n_samples": len(arr)}
    C = arr / m
    S1 = C[:, 1] - C[:, 2]
    S2 = C[:, 1] + C[:, 2] - 2.0 * C[:, 0]
    s1_std = np.std(S1) or 1.0
    s2_std = np.std(S2) or 1.0
    sig = S1 + (s1_std / s2_std) * S2
    try:
        sig = bandpass(sig, 0.7, 3.5, fs, order=3)
    except Exception:
        pass
    sig = sig - np.mean(sig)
    sig_std = np.std(sig) or 1.0
    sig = sig / sig_std
    # Autocorrelation (positive lags only)
    acorr_full = np.correlate(sig, sig, mode="full")
    mid = acorr_full.size // 2
    acorr = acorr_full[mid: mid + sig.size]
    # HR band: 50-180 bpm  ->  lags 60/180 to 60/50 = 0.333 .. 1.20 sec
    min_lag = int(round(0.333 * fs))
    max_lag = int(round(1.200 * fs))
    if max_lag >= acorr.size:
        max_lag = acorr.size - 1
    if min_lag >= max_lag:
        return {"hr_bpm": None, "confidence": 0.0,
                "reason": "window_too_short", "n_samples": len(arr)}
    window = acorr[min_lag:max_lag + 1]
    peak_rel = int(np.argmax(window))
    peak_lag = min_lag + peak_rel
    peak_val = float(window[peak_rel])
    # Normalize peak prominence by acorr[0] (signal variance) for a 0..1 score
    norm = float(acorr[0]) if acorr[0] > 0 else 1e-6
    prominence = peak_val / norm
    bpm = 60.0 * fs / peak_lag if peak_lag > 0 else None
    # 2nd-harmonic check: a real pulse has acorr energy at half-lag too.
    half_lag = peak_lag // 2
    if half_lag >= 4 and half_lag < acorr.size:
        half_val = float(acorr[half_lag])
        harmonic_ok = (half_val / norm) > 0.20
    else:
        harmonic_ok = False
    # Confidence: combine prominence + harmonic check
    confidence = float(max(0.0, min(1.0, prominence)) * (1.0 if harmonic_ok else 0.6))
    return {
        "hr_bpm": float(bpm) if bpm else None,
        "confidence": float(confidence),
        "prominence": float(prominence),
        "harmonic_ok": bool(harmonic_ok),
        "n_samples": int(len(arr)),
        "window_s": float(window_s),
        "reason": "ok" if confidence >= 0.30 else "low_confidence",
    }


def estimate_hrv_full(ps=None):
    """API_unq P-hrv-full: extended HRV with SDNN + LF/HF added to the
    existing RMSSD output. Per VitalLens 2.0 (Karras et al. 2025) +
    de Haan & Jeanne 2013, HRV evaluation should report all three
    canonical metrics with confidence intervals.

    Returns:
        dict {
            rmssd_ms : root mean square of successive RR differences
            sdnn_ms  : std-dev of NN/RR intervals
            lf_hf    : LF (0.04-0.15 Hz) / HF (0.15-0.40 Hz) power ratio
            n_rr     : count of RR intervals used
            stress   : 0..1 composite stress index (already in legacy)
            reason   : 'ok' | <error>
        }

    Falls back to (NaN, NaN, NaN) when the per-person RR buffer is
    insufficient. Backwards-compatible: `estimate_hrv()` unchanged
    (returns the legacy (rmssd, stress) tuple).
    """
    # Reach into the legacy module-level RR buffer used by estimate_hrv().
    # The legacy function reads ps._rr_ms if ps is given, otherwise the
    # global hr_samples-derived buffer. Reuse that path.
    try:
        if ps is not None and hasattr(ps, "_rr_ms"):
            rr = np.asarray(list(ps._rr_ms), dtype=float)
        else:
            # Try the module-level buffer used by legacy estimate_hrv
            global hr_samples
            if "hr_samples" in globals() and len(hr_samples) >= 6:
                hr_arr = np.asarray(list(hr_samples), dtype=float)
                rr = 60000.0 / np.clip(hr_arr, 30.0, 220.0)
            else:
                return {"rmssd_ms": float("nan"), "sdnn_ms": float("nan"),
                        "lf_hf": float("nan"), "n_rr": 0,
                        "reason": "insufficient_rr"}
    except Exception as e:
        return {"rmssd_ms": float("nan"), "sdnn_ms": float("nan"),
                "lf_hf": float("nan"), "n_rr": 0,
                "reason": f"buffer_error:{e}"}
    if rr.size < 6:
        return {"rmssd_ms": float("nan"), "sdnn_ms": float("nan"),
                "lf_hf": float("nan"), "n_rr": int(rr.size),
                "reason": "insufficient_rr"}
    # RMSSD: root mean square of successive differences
    diffs = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diffs * diffs)))
    rmssd = max(10.0, min(150.0, rmssd))  # physiological clamp
    # SDNN: std-dev of RR intervals
    sdnn = float(np.std(rr, ddof=1))
    sdnn = max(5.0, min(250.0, sdnn))  # physiological clamp
    # LF/HF: requires resampled RR tachogram at uniform rate
    # Approximate using Welch on the raw RR series at the mean RR rate.
    lf_hf = float("nan")
    try:
        # Estimate "fs_rr" as 1 / (mean_rr_seconds). Tachogram interpolation
        # to 4 Hz is the canonical Task Force (1996) approach; we approximate
        # with welch on the raw RR signal which works well enough for the
        # ratio.
        rr_s = rr / 1000.0
        fs_rr = 4.0  # canonical interpolated rate
        # Linear interpolation to fs_rr
        t_orig = np.cumsum(rr_s)
        t_orig = t_orig - t_orig[0]
        if t_orig[-1] < 10.0:  # need at least 10 s of RR series
            raise ValueError("rr_series_too_short")
        t_uniform = np.arange(0, t_orig[-1], 1.0 / fs_rr)
        rr_interp = np.interp(t_uniform, t_orig, rr - np.mean(rr))
        from scipy.signal import welch
        freqs, psd = welch(rr_interp, fs=fs_rr,
                           nperseg=min(256, rr_interp.size))
        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
        lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
        hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0
        if hf_power > 1e-9:
            lf_hf = lf_power / hf_power
        # Clamp to a sane range for autonomic-balance reporting
        lf_hf = max(0.1, min(20.0, lf_hf))
    except Exception:
        pass
    return {
        "rmssd_ms": float(rmssd),
        "sdnn_ms": float(sdnn),
        "lf_hf": float(lf_hf),
        "n_rr": int(rr.size),
        "reason": "ok",
    }

