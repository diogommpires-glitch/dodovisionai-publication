"""
DODO VISION AI v7 — Per-Person State (Step 1)
Each tracked face owns isolated biometric state: rPPG buffer, vitals,
emotion history, demographics, gaze/blink, micro-expressions,
deception indicators, vocal state, LLM analysis.

v8.8: RBAC data isolation — raw physiological telemetry is classified as
MEDICAL_RESTRICTED and stripped from public API/WebSocket output unless a
valid clinical auth token is verified. Prevents sphere transgression.

Integrates with original analysis.py patterns (EMA, blendshape names,
FACS AUs, affect_vector format).
"""

import time, threading, json, os, hashlib, hmac
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Any

# ── v8.8: Data classification levels ──
ACCESS_PUBLIC = "PUBLIC"                   # safe for any consumer
ACCESS_MEDICAL_RESTRICTED = "MEDICAL_RESTRICTED"  # raw waveforms, spectrograms, proxy data

# Fields classified as MEDICAL_RESTRICTED (never sent over public WebSocket)
_RESTRICTED_FIELDS = frozenset({
    "rppg_rgb", "rppg_times", "hr_samples",
    "rppg_l_cheek", "rppg_r_cheek",   # v9.4.1
    "_phase_buffer", "_smoothed_lm",
    "mmwave_spectrogram", "source_space_proxy",
    "cold_baseline", "warm_baseline",
})

# v8.8: Clinical auth token verification. In production, use asymmetric
# keys or OAuth2 scopes. This is a HMAC-SHA256 stub.
_CLINICAL_AUTH_SECRET = os.environ.get("DODO_CLINICAL_SECRET", "")


def verify_clinical_token(token: str) -> bool:
    """Verify a clinical authentication token for MEDICAL_RESTRICTED access.
    Returns True only if the token matches the HMAC signature."""
    if not _CLINICAL_AUTH_SECRET or not token:
        return False
    try:
        # Expected format: "timestamp:signature"
        parts = token.split(":", 1)
        if len(parts) != 2:
            return False
        ts_str, sig = parts
        expected = hmac.new(
            _CLINICAL_AUTH_SECRET.encode(),
            ts_str.encode(),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(sig, expected)
    except Exception:
        return False

try:
    from deception_ensemble import score as _ensemble_score
except ImportError:
    _ensemble_score = None

try:
    from trust_gauge import score_trust as _score_trust   # v8.7 (Step 4)
except ImportError:
    _score_trust = None

# v8.7 (Step 3): Kalman replaces per-class EMA on per-person emotion smoothing.
try:
    from kalman import (
        Kalman1D, KalmanDict,
        KALMAN_HR_Q, KALMAN_HR_R,
        KALMAN_AFFECT_Q, KALMAN_AFFECT_R,
        KALMAN_STRESS_Q, KALMAN_STRESS_R,
        KALMAN_HRV_Q, KALMAN_HRV_R,
        KALMAN_RESP_Q, KALMAN_RESP_R,
        KALMAN_EMO_Q, KALMAN_EMO_R,
    )
    _KALMAN_OK = True
except Exception:
    _KALMAN_OK = False

# Deception indicator weights (total ~1.0)
DECEPTION_WEIGHTS = {
    "pupil_dilation": 0.15,       # #1 Rashid et al. 2024
    "blink_suppression": 0.12,    # #2 Mann et al. 2002
    "duchenne_smile": 0.10,       # #3 Ekman et al. 1991
    "facial_asymmetry": 0.08,     # #4 Ekman, Hager & Friesen 1981
    "fear_leakage": 0.10,         # #5 Su & Levine 2016
    "heart_rate_change": 0.10,    # #6 Rashid et al. 2024
    "vocal_f0_shift": 0.08,       # #7 small but reliable
    "response_latency": 0.07,     # #8 moderate
    "micro_expression": 0.08,     # facial micro flashes
    "gsr_change": 0.06,           # galvanic skin response
    "hrv_drop": 0.06,             # autonomic stress
}


class PersonState:
    """Complete per-person biometric state."""

    def __init__(self, track_id: int, identity: Optional[str] = None):
        self.track_id = track_id
        self.identity = identity
        self.created_at = time.time()
        self._lock = threading.Lock()

        # ── rPPG (per-person buffer) ──
        self.rppg_rgb = deque(maxlen=1200)
        self.rppg_times = deque(maxlen=1200)
        self.hr_samples = deque(maxlen=300)
        # v9.4.1: per-ROI cheek buffers for the rppg.extract_liveness()
        # three-output signal path (pulse_snr, phase_coherence,
        # refresh_band_ratio). Each entry is (t, r, g, b). Evicted by
        # rppg.feed_liveness_rois() when older than RPPG_LIVENESS_WINDOW s.
        self.rppg_l_cheek = deque(maxlen=1200)
        self.rppg_r_cheek = deque(maxlen=1200)
        self.rppg_fs_est = 20.0

        # ── Vitals ──
        self.heart_rate = None
        self.hr_confidence = 0
        self.hrv_rmssd = None
        self.respiratory_rate = None
        self.stress_index = None
        self.confirmed_bpm = None
        self.calibrated_hr = None
        self.rppg_quality = 0  # v8.2: per-person rPPG signal quality

        # ── External sensors ──
        self.oxi_hr = None
        self.oxi_spo2 = None
        self.radar_hr = None
        self.radar_breath = None
        self.radar_distance = None
        self.radar_presence = None
        self.gsr_conductance = None
        self.gsr_scr = None

        # ── v9.4 PAD (Presentation Attack Detection) per-person state ──
        # Holds the latest fused PADScore + bounded history for
        # hysteresis + per-person evidence chain. See pad/types.py.
        try:
            from pad.types import PADState as _PADStateCls
            self.pad_state = _PADStateCls()
        except Exception:
            self.pad_state = None

        # v9.8.3 — per-track face-slot metrics (pose/gaze/blink/pupil from
        # faces[face_idx]), not the global analysis.affect_vector (face 0 +
        # dual-cam merge). Consumers use this for audit / PAD AGREEMENT
        # cross-checks; top-level head_pose/gaze may still mirror legacy paths.
        self.face_affect_snapshot: Optional[Dict[str, Any]] = None

        # ── Emotion (Kalman-smoothed in v8.7) ──
        self._emo_kalman = KalmanDict(q=KALMAN_EMO_Q, r=KALMAN_EMO_R) if _KALMAN_OK else None
        self._emo_ema: Dict[str, float] = {}   # legacy fallback if kalman.py missing
        self._emo_alpha = 0.35
        self.emotion_scores: Dict[str, float] = {}
        self.dominant_emotion = "neutral"
        self.valence = 0.0
        self.arousal = 0.0
        self.bs_emotion: Dict[str, float] = {}
        # Per-person Kalman smoothers for vitals (so each face has its own
        # tracker instead of sharing the global ema_hr / ema_stress)
        if _KALMAN_OK:
            self._kf_hr     = Kalman1D(q=KALMAN_HR_Q,     r=KALMAN_HR_R)
            self._kf_val    = Kalman1D(q=KALMAN_AFFECT_Q, r=KALMAN_AFFECT_R)
            self._kf_aro    = Kalman1D(q=KALMAN_AFFECT_Q, r=KALMAN_AFFECT_R)
            self._kf_stress = Kalman1D(q=KALMAN_STRESS_Q, r=KALMAN_STRESS_R)
            self._kf_hrv    = Kalman1D(q=KALMAN_HRV_Q,    r=KALMAN_HRV_R)
            self._kf_resp   = Kalman1D(q=KALMAN_RESP_Q,   r=KALMAN_RESP_R)
        else:
            self._kf_hr = self._kf_val = self._kf_aro = None
            self._kf_stress = self._kf_hrv = self._kf_resp = None

        # ── Demographics (smoothed) ──
        self.age = None
        self.gender = None
        self.race = None
        self._age_buf = deque(maxlen=90)
        self._race_scores: Dict[str, float] = {}
        self._race_count = 0
        self._gender_counts: Dict[str, int] = {}

        # ── Pose / Gaze / Blink ──
        self.head_pose: Optional[Dict] = None
        self.gaze: Optional[Dict] = None
        self.iris_data: Optional[Dict] = None
        self.blink_data: Optional[Dict] = None
        self.micro_data: Optional[Dict] = None
        self.blink_history = deque(maxlen=600)   # (ts, bool) -- used by pad.blink_gaze_branch
        self.perclos_history = deque(maxlen=900)
        self._prev_blink_closed = False
        # v9.8.1 (2026-05-12): fields required by pad/branches/blink_gaze_branch.score().
        # Without them, getattr() returns None -> branch abstains forever
        # (reliability=0, status='warmup'), so the LIVE temporal signal
        # from blink regularity + gaze micro-saccades is permanently
        # absent from fusion.
        # v9.8.1 BUGFIX 2026-05-12 02:50: session_start MUST be time.time()
        # (epoch) not time.monotonic() because blink_gaze_branch.score()
        # computes `duration = time.time() - session_start`. Using the
        # monotonic clock made duration ~= 1.7 BILLION seconds (the
        # epoch-vs-monotonic gap), pushing the computed bpm to 1e-7 and
        # firing the "NO_BLINK" SPOOF detector on every real subject.
        # Self-inflicted regression from earlier in this session.
        self.session_start = time.time()
        self.gaze_history = deque(maxlen=900)

        # ── Deception state ──
        self.baseline_hr = None
        self.baseline_stress = None
        self.baseline_blink_rate = None
        self.baseline_pupil = None
        self.baseline_f0 = None
        self.baseline_gsr = None
        self.baseline_hrv = None
        # 2026-04-29 (v9.7.3): baseline rest-state facial asymmetry. Without
        # this every face shows ~100% asymmetry because natural neutral
        # faces have AU L/R imbalance ~0.3-0.5 (mouth dominant side, eye
        # squint micro-differences), and the previous `min(1, asym * 3)`
        # mapping saturated on rest-state noise alone.
        self.baseline_facial_asymmetry = None
        # EMA-smoothed pupil reading used inside update_deception so a
        # single noisy iris frame doesn't spike the pupil_dilation
        # indicator to 100%. ~0.7 keeps ≈3-frame effective window at 1Hz.
        self._pupil_ema: float = 0.0
        self._pupil_ema_init: bool = False
        self.is_baselined = False
        self.deception_indicators: Dict[str, float] = {k: 0.0 for k in DECEPTION_WEIGHTS}
        self.deception_score = 0.0

        # ── v8.7 (Step 5): Two-stage baseline ──
        # Cold baseline: 30-second auto-capture immediately after track start
        # Warm baseline: 60-second sliding window, recomputed continuously
        # Drift between cold and warm reveals slow autonomic shifts (mounting
        # tension) that a single-shot baseline misses.
        self.cold_window_secs = 30.0
        self.warm_window_secs = 60.0
        self._cold_buf: Dict[str, deque] = {
            "hr": deque(maxlen=900), "stress": deque(maxlen=900),
            "blink": deque(maxlen=900), "pupil": deque(maxlen=900),
            "hrv": deque(maxlen=900), "gsr": deque(maxlen=900), "f0": deque(maxlen=900),
            # v9.7.3: track rest-state facial asymmetry so the deception
            # indicator subtracts personal baseline before scaling.
            "asym": deque(maxlen=900),
        }
        self._warm_buf: Dict[str, deque] = {
            "hr": deque(maxlen=1800), "stress": deque(maxlen=1800),
            "blink": deque(maxlen=1800), "pupil": deque(maxlen=1800),
            "hrv": deque(maxlen=1800), "gsr": deque(maxlen=1800), "f0": deque(maxlen=1800),
            "asym": deque(maxlen=1800),
        }
        self._cold_started = time.time()
        self._cold_locked = False
        self.cold_baseline: Dict[str, float] = {}
        self.warm_baseline: Dict[str, float] = {}
        self.baseline_drift: Dict[str, float] = {}     # warm - cold per metric
        self.baseline_drift_score: float = 0.0          # 0..1, overall drift
        self._last_baseline_sample: float = 0            # throttle for build_affect_dict

        # ── Voice ──
        self.vocal_f0 = None
        self.vocal_jitter = None
        self.vocal_stress = 0.0
        self.vocal_pauses: List[float] = []

        # ── LLM ──
        self.llm_analysis = ""
        self.llm_last_update = 0.0

        # ── v8.8: EMA landmark smoothing for jitter suppression ──
        # Smooths the raw MediaPipe ROI coordinates before rPPG extraction.
        # Per-person state ensures multi-person isolation.
        self._ema_alpha_lm = 0.4  # 0.4 balances jitter suppression vs. responsiveness
        self._smoothed_lm: Optional[Dict[int, tuple]] = None  # idx -> (x, y)

        # ── 120fps fast biometrics ──
        self.duchenne_smile = False
        self.fake_smile = False
        self.facial_asymmetry = 0.0
        self.fear_au_score = 0.0
        self.deception_au_score = 0.0

    def update_emotions(self, fused: Dict[str, float]):
        """Apply per-emotion Kalman smoothing (v8.7).
        Falls back to legacy EMA if kalman.py is unavailable."""
        if self._emo_kalman is not None:
            smoothed = self._emo_kalman.update(fused)
            self.emotion_scores = {k: round(v, 3) for k, v in smoothed.items()}
        else:
            for k, v in fused.items():
                if k in self._emo_ema:
                    self._emo_ema[k] = self._emo_alpha * v + (1 - self._emo_alpha) * self._emo_ema[k]
                else:
                    self._emo_ema[k] = v
            self.emotion_scores = {k: round(v, 3) for k, v in self._emo_ema.items()}
        if self.emotion_scores:
            self.dominant_emotion = max(self.emotion_scores, key=self.emotion_scores.get)

    def smooth_landmarks(self, face_landmarks, roi_indices):
        """Apply per-person EMA smoothing to the ROI landmark coordinates.

        v8.8: Suppresses high-frequency spatial jitter from MediaPipe before
        the coordinates reach the rPPG ROI extraction. Uses the recursive EMA:
            S_t = alpha * X_t + (1 - alpha) * S_{t-1}
        Operates on normalized (0..1) x,y coordinates.

        Returns a dict {landmark_index: (smoothed_x, smoothed_y)} for each
        ROI index that exists in the landmark list.
        """
        alpha = self._ema_alpha_lm
        smoothed = {}
        for idx in roi_indices:
            if idx >= len(face_landmarks):
                continue
            raw_x = face_landmarks[idx].x
            raw_y = face_landmarks[idx].y
            if self._smoothed_lm is not None and idx in self._smoothed_lm:
                prev_x, prev_y = self._smoothed_lm[idx]
                sx = alpha * raw_x + (1 - alpha) * prev_x
                sy = alpha * raw_y + (1 - alpha) * prev_y
            else:
                sx, sy = raw_x, raw_y
            smoothed[idx] = (sx, sy)
        self._smoothed_lm = smoothed
        return smoothed

    # v8.7: helper setters that route through per-person Kalman filters.
    # Callers in main.py / rppg.py can swap `ps.heart_rate = x` for
    # `ps.set_heart_rate(x)` to get smoothed, predict-on-missing behavior.
    def set_heart_rate(self, hr, confidence: float = None):
        if self._kf_hr is None or hr is None:
            self.heart_rate = hr
            return
        # Adaptive R: lower confidence -> wider measurement noise
        r_eff = None
        if confidence is not None and confidence < 0.5:
            r_eff = max(8.0, 32.0 * (1 - confidence))
        est = self._kf_hr.update(float(hr), r_override=r_eff)
        if est is not None:
            self.heart_rate = round(est, 1)
        if confidence is not None:
            self.hr_confidence = confidence

    def set_valence_arousal(self, valence, arousal):
        if self._kf_val is not None and valence is not None:
            v = self._kf_val.update(float(valence))
            self.valence = round(v, 3) if v is not None else valence
        elif valence is not None:
            self.valence = float(valence)
        if self._kf_aro is not None and arousal is not None:
            a = self._kf_aro.update(float(arousal))
            self.arousal = round(a, 3) if a is not None else arousal
        elif arousal is not None:
            self.arousal = float(arousal)

    def set_stress(self, stress):
        if self._kf_stress is not None and stress is not None:
            s = self._kf_stress.update(float(stress))
            self.stress_index = round(s, 3) if s is not None else stress
        elif stress is not None:
            self.stress_index = float(stress)

    def set_hrv(self, hrv):
        if self._kf_hrv is not None and hrv is not None:
            v = self._kf_hrv.update(float(hrv))
            self.hrv_rmssd = round(v, 1) if v is not None else hrv
        elif hrv is not None:
            self.hrv_rmssd = float(hrv)

    def set_resp_rate(self, resp):
        if self._kf_resp is not None and resp is not None:
            v = self._kf_resp.update(float(resp))
            self.respiratory_rate = round(v, 1) if v is not None else resp
        elif resp is not None:
            self.respiratory_rate = float(resp)

    def update_demographics(self, raw_age=None, gender_raw=None, race_scores=None):
        if raw_age is not None:
            self._age_buf.append(raw_age)
            arr = sorted(list(self._age_buf))
            trim = max(1, len(arr) // 7)
            trimmed = arr[trim:-trim] if trim > 0 and len(arr) > 2 * trim else arr
            self.age = int(round(np.mean(trimmed)))
        if gender_raw:
            gd = max(gender_raw, key=gender_raw.get, default="?")
            self._gender_counts[gd] = self._gender_counts.get(gd, 0) + 1
            self.gender = max(self._gender_counts, key=self._gender_counts.get)
        if race_scores:
            self._race_count += 1
            for r, s in race_scores.items():
                self._race_scores[r] = self._race_scores.get(r, 0) + s
            if self._race_count > 60:
                for k in self._race_scores: self._race_scores[k] *= 0.97
            total = sum(self._race_scores.values())
            if total > 0:
                self.race = max(self._race_scores, key=self._race_scores.get)

    def set_baseline(self):
        """Capture current state as deception baseline (manual/legacy entrypoint).
        v8.7: also seeds the cold/warm baseline so subsequent automatic drift
        tracking has something to compare against."""
        self.baseline_hr = self.heart_rate
        self.baseline_stress = self.stress_index
        self.baseline_blink_rate = self.blink_data.get("blink_rate_pm") if self.blink_data else None
        self.baseline_pupil = self.iris_data.get("avg_dilation") if self.iris_data else None
        self.baseline_gsr = self.gsr_conductance
        self.baseline_hrv = self.hrv_rmssd
        self.baseline_f0 = self.vocal_f0
        # v9.7.3: capture rest-state asymmetry from the current frame so
        # this legacy entrypoint matches what the cold-window path now
        # produces. None is fine — update_deception falls back to a 0.35
        # population prior when this is missing.
        self.baseline_facial_asymmetry = (
            self.facial_asymmetry if self.facial_asymmetry and self.facial_asymmetry > 0 else None
        )
        self.is_baselined = True
        # Mirror into cold baseline so the warm/drift logic is happy
        self.cold_baseline = {
            "hr": self.baseline_hr, "stress": self.baseline_stress,
            "blink": self.baseline_blink_rate, "pupil": self.baseline_pupil,
            "hrv": self.baseline_hrv, "gsr": self.baseline_gsr, "f0": self.baseline_f0,
            "asym": self.baseline_facial_asymmetry,
        }
        self._cold_locked = True

    # ── v8.7 (Step 5): Two-stage cold/warm baseline ──
    def feed_baseline_sample(self):
        """Push the current vital snapshot into both cold + warm windows.
        Call this at ~1Hz from the orchestrator (main.py). After
        cold_window_secs the cold baseline is locked; after that, the
        warm baseline keeps rolling and drift is computed."""
        now = time.time()
        sample = {
            "hr": self.heart_rate, "stress": self.stress_index,
            "blink": (self.blink_data or {}).get("blink_rate_pm"),
            "pupil": (self.iris_data or {}).get("avg_dilation"),
            "hrv": self.hrv_rmssd, "gsr": self.gsr_conductance,
            "f0": self.vocal_f0,
            # v9.7.3: include facial_asymmetry only when blendshapes were
            # actually active — `pairs_checked == 0` upstream returns 0.0
            # which would lower the rest-state median artificially. We
            # filter that case out by only sampling positive values.
            "asym": (self.facial_asymmetry if self.facial_asymmetry and self.facial_asymmetry > 0 else None),
        }
        # Push to cold while it's still warming up
        if not self._cold_locked:
            for k, v in sample.items():
                if v is not None:
                    self._cold_buf[k].append(float(v))
            if (now - self._cold_started) >= self.cold_window_secs:
                self._lock_cold_baseline()
        # Always push to warm window (rolling)
        for k, v in sample.items():
            if v is not None:
                self._warm_buf[k].append(float(v))
        if self._cold_locked and len(self._warm_buf["hr"]) > 30:
            self._compute_warm_drift()

    def _lock_cold_baseline(self):
        """Compute medians of the cold window and freeze them as the cold baseline."""
        baseline = {}
        for k, dq in self._cold_buf.items():
            if dq:
                arr = sorted(dq)
                baseline[k] = arr[len(arr) // 2]
        self.cold_baseline = baseline
        self._cold_locked = True
        # Backfill the legacy single baseline fields so downstream code still works
        if not self.is_baselined:
            self.baseline_hr = baseline.get("hr")
            self.baseline_stress = baseline.get("stress")
            self.baseline_blink_rate = baseline.get("blink")
            self.baseline_pupil = baseline.get("pupil")
            self.baseline_hrv = baseline.get("hrv")
            self.baseline_gsr = baseline.get("gsr")
            self.baseline_f0 = baseline.get("f0")
            # v9.7.3: rest-state facial asymmetry. When None (subject sat
            # perfectly neutral for the cold window so no AU pairs ever
            # passed the activation threshold), update_deception falls
            # back to a population prior so it doesn't divide-by-None.
            self.baseline_facial_asymmetry = baseline.get("asym")
            # Only mark baselined when we have enough useful metrics
            useful = sum(1 for v in baseline.values() if v is not None)
            self.is_baselined = useful >= 3
        # v8.7 (Step 9): forensic event log entry. Lazy-import so person_state
        # has no hard dependency on the SQLite log.
        try:
            import forensic_log as _fl
            _fl.log_event("baseline_cold_locked",
                person_id=getattr(self, "track_id", None),
                identity=getattr(self, "identity", None),
                payload={k: round(v, 3) for k, v in baseline.items()
                         if isinstance(v, (int, float))})
        except Exception:
            pass

    def _compute_warm_drift(self):
        """Recompute warm baseline (rolling median) and the cold->warm drift."""
        warm = {}
        for k, dq in self._warm_buf.items():
            if dq:
                arr = sorted(dq)
                warm[k] = arr[len(arr) // 2]
        self.warm_baseline = warm
        drift = {}
        # Per-metric drift normalized to "fraction of baseline"
        for k, cv in self.cold_baseline.items():
            if cv is None or cv == 0:
                continue
            wv = warm.get(k)
            if wv is None:
                continue
            drift[k] = round((wv - cv) / abs(cv), 3)
        self.baseline_drift = drift
        # Composite drift score: max absolute fractional drift across HR/stress/blink
        weights = {"hr": 0.35, "stress": 0.25, "blink": 0.15,
                   "pupil": 0.10, "hrv": 0.10, "gsr": 0.05}
        score = 0.0
        wsum = 0.0
        for k, w in weights.items():
            if k in drift:
                score += w * min(1.0, abs(drift[k]) * 3)  # ~33% drift -> full
                wsum += w
        self.baseline_drift_score = round(score / wsum, 3) if wsum > 0 else 0.0

    def update_deception(self):
        """Compute deception indicators vs baseline (Step 3 integration)."""
        if not self.is_baselined:
            return
        ind = self.deception_indicators
        # ── Pupil dilation change ──
        # v9.7.3: was `min(1, |curr-base|/base * 5)` which saturated at
        # 20% deviation. Iris-diameter measurements are noisy enough that
        # 20% drift happens every blink even at rest, pegging the bar at
        # 100% permanently. Two changes:
        #   1. EMA-smooth `curr_pupil` over recent frames (alpha=0.3) so
        #      single-frame outliers don't spike the indicator.
        #   2. Drop the multiplier from ×5 to ×1.5 + introduce a 5%
        #      dead-zone. Now: <5% drift → 0, 70% drift → 100%. That
        #      matches the literature on cognitive-load pupillometry
        #      (Beatty 1982 reports 0.3-0.5 mm dilation under load,
        #      ≈10-20% relative on a 3-4 mm baseline).
        curr_pupil_raw = self.iris_data.get("avg_dilation", 0) if self.iris_data else 0
        if curr_pupil_raw and curr_pupil_raw > 0:
            if not self._pupil_ema_init:
                self._pupil_ema = float(curr_pupil_raw)
                self._pupil_ema_init = True
            else:
                self._pupil_ema = 0.7 * self._pupil_ema + 0.3 * float(curr_pupil_raw)
        curr_pupil = self._pupil_ema if self._pupil_ema_init else 0.0
        if self.baseline_pupil and self.baseline_pupil > 0 and curr_pupil > 0:
            rel = abs(curr_pupil - self.baseline_pupil) / self.baseline_pupil
            # 5% dead zone for sensor noise, then linear ramp to ×1.5
            rel_active = max(0.0, rel - 0.05)
            ind["pupil_dilation"] = min(1.0, rel_active * 1.5)
        # ── Blink suppression ──
        # v9.7.5: was `min(1, max(0, (1 - ratio) * 2))` — when current
        # blink rate dropped to 50 % of baseline (a perfectly normal
        # bit of attention focus), the indicator pegged at 100 %.
        # Same family of bug as the pupil and asymmetry over-scaling.
        # Now: 30 % dead zone (allow normal attention-related blink-rate
        # variation), then linear ramp at ×1.2 instead of ×2. So a
        # rate at 70 % of baseline is the start of "suppression";
        # below 17 % of baseline pegs at 100 %. Matches the
        # Holland & Tarlow 1972 cognitive-load literature where
        # 30-50 % rate drops are normal and don't indicate suppression.
        curr_blink = self.blink_data.get("blink_rate_pm", 15) if self.blink_data else 15
        if self.baseline_blink_rate and self.baseline_blink_rate > 0:
            ratio = curr_blink / self.baseline_blink_rate
            drop = max(0.0, 1.0 - ratio)        # 0..1 (0 = same as baseline)
            drop_active = max(0.0, drop - 0.30) # 30 % dead zone
            ind["blink_suppression"] = min(1.0, drop_active * 1.2)
        # Duchenne vs fake
        ind["duchenne_smile"] = 0.8 if self.fake_smile else 0.0
        # ── Facial asymmetry ──
        # v9.7.3: was `min(1, self.facial_asymmetry * 3)`. Problem:
        # `self.facial_asymmetry` is the MEAN |L-R|/max(L,R) across active
        # blendshape pairs (mouthSmile, browDown, eyeSquint, etc). Even
        # neutral faces produce 0.3-0.5 from natural anatomical asymmetry
        # and pixel noise, so ×3 pegged the bar at 100% on rest state.
        # Now:
        #   1. Subtract personal `baseline_facial_asymmetry` (population
        #      prior of 0.35 if uncalibrated — empirical median across
        #      neutral capture).
        #   2. Drop multiplier from ×3 to ×1.5 so a 67% deviation above
        #      personal baseline = 100%.
        #   3. Floor at 0 so symmetry-better-than-baseline doesn't go
        #      negative.
        _asym_baseline = self.baseline_facial_asymmetry if self.baseline_facial_asymmetry else 0.35
        _asym_excess = max(0.0, float(self.facial_asymmetry) - _asym_baseline)
        ind["facial_asymmetry"] = min(1.0, _asym_excess * 1.5)
        # Fear AU leakage
        ind["fear_leakage"] = min(1, self.fear_au_score * 2)
        # Heart rate change
        if self.baseline_hr and self.baseline_hr > 0 and self.heart_rate:
            ind["heart_rate_change"] = min(1, abs(self.heart_rate - self.baseline_hr) / self.baseline_hr * 3)
        # Vocal
        if self.baseline_f0 and self.baseline_f0 > 0 and self.vocal_f0:
            ind["vocal_f0_shift"] = min(1, abs(self.vocal_f0 - self.baseline_f0) / self.baseline_f0 * 3)
        # ── Micro expressions ──
        # v9.7.3: was `min(1, intensity * 2)`. `intensity` is the MAX
        # blendshape delta over a 6-frame window at 120 fps from
        # `dual_camera._micro`. Natural facial noise (eye micro-tremor,
        # respiration-driven brow flutter) generates ~0.30-0.40 every
        # window, so `* 2` pegged the indicator at 60-80% permanently
        # even on a still subject. Two changes:
        #   1. 15% dead-zone for blendshape noise floor (empirical: a
        #      neutral subject in good lighting holds ~0.10-0.18 max).
        #   2. Multiplier dropped from ×2 to ×1.2 so a clear
        #      micro-expression at intensity=0.65 reaches ~60% rather
        #      than 100%.
        if self.micro_data and self.micro_data.get("detected"):
            _mi_raw = float(self.micro_data.get("intensity", 0) or 0)
            _mi_active = max(0.0, _mi_raw - 0.15)
            ind["micro_expression"] = min(1.0, _mi_active * 1.2)
        else:
            ind["micro_expression"] = 0.0
        # GSR change
        if self.baseline_gsr and self.baseline_gsr > 0 and self.gsr_conductance:
            ind["gsr_change"] = min(1, abs(self.gsr_conductance - self.baseline_gsr) / self.baseline_gsr * 2)
        # HRV drop
        if self.baseline_hrv and self.baseline_hrv > 0 and self.hrv_rmssd:
            drop = max(0, self.baseline_hrv - self.hrv_rmssd) / self.baseline_hrv
            ind["hrv_drop"] = min(1, drop * 2)
        # v8.1: calibrated ensemble (Platt sigmoid) when available
        if _ensemble_score is not None:
            face_q = max(self.emotion_scores.values()) if self.emotion_scores else 0
            vad_q = 1.0 if self.vocal_f0 and self.vocal_f0 > 0 else 0.0
            hr_q = 1.0 if self.hr_confidence and self.hr_confidence > 0.3 else 0.0
            gsr_q = 1.0 if self.gsr_conductance and self.gsr_conductance > 0 else 0.0
            hrv_q = 1.0 if self.hrv_rmssd and self.hrv_rmssd > 0 else 0.0
            modalities = {
                "micro_expr":    {"score": ind.get("micro_expression", 0), "quality": face_q},
                "gaze_flight":   {"score": ind.get("pupil_dilation", 0), "quality": face_q},
                "facial_asym":   {"score": ind.get("facial_asymmetry", 0), "quality": face_q},
                "vocal_stress":  {"score": ind.get("vocal_f0_shift", 0), "quality": vad_q},
                "hr_reactivity": {"score": ind.get("heart_rate_change", 0), "quality": hr_q},
                "gsr":           {"score": ind.get("gsr_change", 0), "quality": gsr_q},
                "hrv":           {"score": ind.get("hrv_drop", 0), "quality": hrv_q},
            }
            result = _ensemble_score(modalities)
            self.deception_score = result.probability
        else:
            self.deception_score = sum(ind.get(k, 0) * w for k, w in DECEPTION_WEIGHTS.items())

    def build_affect_dict(self) -> Dict[str, Any]:
        """Build affect vector in the same format as original analysis.affect_vector."""
        # v8.7 (Step 5): self-throttled baseline sampling (1 Hz max).
        # Frees callers from having to schedule it.
        now = time.time()
        if now - getattr(self, "_last_baseline_sample", 0) >= 1.0:
            self._last_baseline_sample = now
            try:
                self.feed_baseline_sample()
            except Exception:
                pass
        self.update_deception()
        # v9.7.4: HR display fallback chain. Was: `heart_rate_bpm =
        # self.heart_rate` (rPPG only) but `hr_source` followed
        # OXI → RADAR → rPPG priority. Result: when rPPG hadn't
        # locked yet but radar reported HR=97, the panel showed
        # `HR -- [RADAR]` (label correct, value missing). Now the
        # displayed HR follows the same priority as `hr_source`.
        _display_hr = (self.oxi_hr if self.oxi_hr else
                       (self.radar_hr if self.radar_hr else
                        (self.heart_rate if self.heart_rate else None)))
        out = {
            "track_id": self.track_id, "identity": self.identity,
            "valence": round(self.valence, 3), "arousal": round(self.arousal, 3),
            "dominant_emotion": self.dominant_emotion,
            "emotion_scores": self.emotion_scores,
            "heart_rate_bpm": _display_hr, "confirmed_bpm": self.confirmed_bpm,
            "calibrated_hr": self.calibrated_hr,
            "hrv_rmssd": self.hrv_rmssd, "stress_index": self.stress_index,
            "resp_rate": self.respiratory_rate,
            "blink_rate": self.blink_data.get("blink_rate_pm") if self.blink_data else None,
            "perclos": self.blink_data.get("perclos", 0) if self.blink_data else 0,
            "drowsiness": self.blink_data.get("drowsiness", "alert") if self.blink_data else "alert",
            "head_pose": self.head_pose, "gaze": self.gaze,
            "micro_expression": self.micro_data if self.micro_data and self.micro_data.get("detected") else None,
            "pupil_dilation": self.iris_data.get("avg_dilation") if self.iris_data else None,
            "age": self.age, "gender": self.gender, "race": self.race,
            "oxi_hr": self.oxi_hr, "oxi_spo2": self.oxi_spo2,
            "radar_hr": self.radar_hr, "radar_breath": self.radar_breath,
            "radar_distance": self.radar_distance, "radar_presence": self.radar_presence,
            "gsr_conductance": self.gsr_conductance, "gsr_scr": self.gsr_scr,
            "duchenne_smile": self.duchenne_smile, "fake_smile": self.fake_smile,
            "facial_asymmetry": round(self.facial_asymmetry, 3),
            "deception": {
                "score": round(self.deception_score, 3),
                "baselined": self.is_baselined,
                "indicators": {k: round(v, 3) for k, v in self.deception_indicators.items()},
                # v8.7 (Step 5): two-stage baseline + drift telemetry
                "cold_baseline": {k: round(v, 3) for k, v in self.cold_baseline.items() if v is not None},
                "warm_baseline": {k: round(v, 3) for k, v in self.warm_baseline.items() if v is not None},
                "baseline_drift": self.baseline_drift,
                "baseline_drift_score": self.baseline_drift_score,
                "cold_locked": self._cold_locked,
            },
            "vocal": {"f0": self.vocal_f0, "jitter": self.vocal_jitter, "stress": round(self.vocal_stress, 3)},
            "llm_analysis": self.llm_analysis,
            "rppg_quality": self.rppg_quality,
            "hr_source": "OXI" if self.oxi_hr else ("RADAR" if self.radar_hr else ("rPPG" if self.heart_rate else None)),
            "signal_quality": {"face": max(self.emotion_scores.values()) if self.emotion_scores else 0, "rppg": self.hr_confidence},
            "timestamp": time.time(),
        }
        if self.face_affect_snapshot:
            out["face_affect"] = self.face_affect_snapshot
        # v8.7 (Step 4): cross-modal trust gauge
        if _score_trust is not None:
            try:
                out["trust"] = _score_trust(out)
            except Exception:
                out["trust"] = {"trust": 0.0, "level": "none", "explain": "trust_gauge_error"}
        return out

    def build_affect_dict_sanitized(self, clinical_token: str = "") -> Dict[str, Any]:
        """v8.8: Public-safe affect dict with MEDICAL_RESTRICTED fields stripped.

        Exposes only high-level state strings (e.g. "Alert", "Fatigued") and
        scalar summaries. Raw rPPG waveforms, baseline buffers, and mmWave
        spectrograms are withheld unless a valid clinical auth token is provided.

        This method enforces the Neurorights data sovereignty constraint:
        third-party plugins cannot access raw physiological telemetry.
        """
        full = self.build_affect_dict()

        if verify_clinical_token(clinical_token):
            full["_access_level"] = ACCESS_MEDICAL_RESTRICTED
            return full

        # Sanitize: strip restricted fields and replace with high-level summaries
        sanitized = dict(full)
        sanitized["_access_level"] = ACCESS_PUBLIC

        # Replace deception detail with boolean summary
        if "deception" in sanitized:
            dec = sanitized["deception"]
            sanitized["deception"] = {
                "score": dec.get("score", 0),
                "baselined": dec.get("baselined", False),
                "baseline_drift_score": dec.get("baseline_drift_score", 0),
            }

        # Replace detailed emotion scores with dominant only
        sanitized.pop("emotion_scores", None)

        # Strip raw baseline data
        for key in ("cold_baseline", "warm_baseline", "baseline_drift"):
            if key in sanitized.get("deception", {}):
                sanitized["deception"].pop(key, None)

        # High-level alertness string (replaces raw cognitive metrics)
        if self.stress_index is not None and self.stress_index > 0.7:
            sanitized["alertness_state"] = "stressed"
        elif self.blink_data and self.blink_data.get("drowsiness") == "drowsy":
            sanitized["alertness_state"] = "fatigued"
        else:
            sanitized["alertness_state"] = "alert"

        return sanitized

    def summarize(self, output_dir: str = "sessions/persons") -> Optional[str]:
        """Write a brief JSON summary before pruning. Returns filepath or None.
        v8.4: preserves forensic data for people who left the frame."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            summary = {
                "track_id": self.track_id,
                "identity": self.identity,
                "created_at": self.created_at,
                "pruned_at": time.time(),
                "duration_s": round(time.time() - self.created_at, 1),
                "last_hr": self.heart_rate,
                "last_stress": self.stress_index,
                "last_emotion": self.dominant_emotion,
                "deception_score": round(self.deception_score, 3),
                "is_baselined": self.is_baselined,
                "age": self.age,
                "gender": self.gender,
                "race": self.race,
                "llm_analysis": self.llm_analysis[:500] if self.llm_analysis else None,
            }
            ts = int(time.time() * 1000)
            fname = f"person_{self.track_id}_{ts}.json"
            fpath = os.path.join(output_dir, fname)
            with open(fpath, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            return fpath
        except Exception:
            return None
