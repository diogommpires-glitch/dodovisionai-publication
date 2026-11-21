"""
DODO VISION AI — Truth Engine (Research-Corrected)
Multi-modal deception analysis based on peer-reviewed research.

INDICATOR RELIABILITY (from meta-analyses):
  #1 Pupil dilation (Rashid et al. 2024): MOST significant predictor
  #2 Blink suppression during lying (Mann et al. 2002): reliable
  #3 Duchenne vs fake smile (Ekman et al. 1991): diagnostic
  #4 Facial asymmetry (Ekman, Hager & Friesen 1981): moderate
  #5 Fear AU leakage (Su & Levine 2016): 78% ML accuracy
  #6 Heart rate increase: moderate (Rashid et al. 2024)
  #7 Vocal F0 increase: small but reliable
  #8 Response latency: moderate
  #9 Content analysis (CBCA): MOST reliable overall method
  
  NOT RELIABLE (DePaulo et al. 2003, d=0.10):
  - Gaze aversion (common myth, near zero predictive value)
  - General nervousness
  - Fidgeting
"""
import time
import numpy as np

BASELINE_DURATION = 20  # seconds (was 10, research needs more samples)

class DeceptionBaseline:
    """Stores physiological baseline for comparison."""
    def __init__(self):
        self.hr_samples = []
        self.stress_samples = []
        self.blink_samples = []
        self.pupil_samples = []
        self.gaze_h_samples = []
        self.gaze_v_samples = []
        self.pose_yaw_samples = []
        self.micro_count = 0
        # New: facial metrics
        self.asymmetry_samples = []
        self.duchenne_count = 0
        self.fake_smile_count = 0
        self.fear_scores = []
        self.deception_au_scores = []
        self.blendshapes_history = []
        self.start_time = time.time()
        self.ready = False
        self.frame_count = 0

    def add_frame(self, av, bs=None):
        self.frame_count += 1
        if av.get("heart_rate_bpm"):
            self.hr_samples.append(av["heart_rate_bpm"])
        if av.get("stress_index") is not None:
            self.stress_samples.append(av["stress_index"])
        if av.get("blink_rate") is not None:
            self.blink_samples.append(av["blink_rate"])
        if av.get("pupil_dilation") is not None:
            self.pupil_samples.append(av["pupil_dilation"])
        if av.get("gaze"):
            self.gaze_h_samples.append(av["gaze"].get("horizontal", 0))
            self.gaze_v_samples.append(av["gaze"].get("vertical", 0))
        if av.get("head_pose"):
            self.pose_yaw_samples.append(abs(av["head_pose"].get("yaw", 0)))
        if av.get("micro_expression") and av["micro_expression"].get("detected"):
            self.micro_count += 1
        # Facial deception metrics (computed from blendshapes if available)
        if bs:
            try:
                from analysis import detect_facial_asymmetry, detect_duchenne_smile, detect_fear_leakage
                asym = detect_facial_asymmetry(bs)
                self.asymmetry_samples.append(asym["asymmetry"])
                duc = detect_duchenne_smile(bs)
                if duc["duchenne"]: self.duchenne_count += 1
                if duc["fake_smile"]: self.fake_smile_count += 1
                fear = detect_fear_leakage(bs)
                self.fear_scores.append(fear["fear_score"])
                self.deception_au_scores.append(fear["deception_au_score"])
            except Exception:
                pass

        elapsed = time.time() - self.start_time
        if elapsed >= BASELINE_DURATION and len(self.hr_samples) >= 10:
            self.ready = True

    def get_summary(self):
        def _m(arr): return float(np.mean(arr)) if arr else None
        def _s(arr): return float(np.std(arr)) if len(arr) > 2 else None
        return {
            "hr_mean": _m(self.hr_samples), "hr_std": _s(self.hr_samples),
            "stress_mean": _m(self.stress_samples),
            "blink_mean": _m(self.blink_samples), "blink_std": _s(self.blink_samples),
            "pupil_mean": _m(self.pupil_samples), "pupil_std": _s(self.pupil_samples),
            "gaze_h_std": _s(self.gaze_h_samples), "gaze_v_std": _s(self.gaze_v_samples),
            "yaw_mean": _m(self.pose_yaw_samples),
            "micro_rate": self.micro_count / max(1, self.frame_count),
            "asymmetry_mean": _m(self.asymmetry_samples),
            "fear_mean": _m(self.fear_scores),
            "deception_au_mean": _m(self.deception_au_scores),
            "duchenne_rate": self.duchenne_count / max(1, self.frame_count),
            "fake_smile_rate": self.fake_smile_count / max(1, self.frame_count),
            "duration_s": round(time.time() - self.start_time, 1),
            "samples": len(self.hr_samples),
        }


class AnswerCapture:
    def __init__(self):
        self.hr_samples = []
        self.stress_samples = []
        self.blink_samples = []
        self.pupil_samples = []
        self.gaze_h_samples = []
        self.gaze_v_samples = []
        self.pose_yaw_samples = []
        self.micro_count = 0
        self.asymmetry_samples = []
        self.duchenne_count = 0
        self.fake_smile_count = 0
        self.fear_scores = []
        self.deception_au_scores = []
        self.start_time = time.time()
        self.frame_count = 0

    def add_frame(self, av, bs=None):
        self.frame_count += 1
        if av.get("heart_rate_bpm"):
            self.hr_samples.append(av["heart_rate_bpm"])
        if av.get("stress_index") is not None:
            self.stress_samples.append(av["stress_index"])
        if av.get("blink_rate") is not None:
            self.blink_samples.append(av["blink_rate"])
        if av.get("pupil_dilation") is not None:
            self.pupil_samples.append(av["pupil_dilation"])
        if av.get("gaze"):
            self.gaze_h_samples.append(av["gaze"].get("horizontal", 0))
            self.gaze_v_samples.append(av["gaze"].get("vertical", 0))
        if av.get("head_pose"):
            self.pose_yaw_samples.append(abs(av["head_pose"].get("yaw", 0)))
        if av.get("micro_expression") and av["micro_expression"].get("detected"):
            self.micro_count += 1
        if bs:
            try:
                from analysis import detect_facial_asymmetry, detect_duchenne_smile, detect_fear_leakage
                asym = detect_facial_asymmetry(bs)
                self.asymmetry_samples.append(asym["asymmetry"])
                duc = detect_duchenne_smile(bs)
                if duc["duchenne"]: self.duchenne_count += 1
                if duc["fake_smile"]: self.fake_smile_count += 1
                fear = detect_fear_leakage(bs)
                self.fear_scores.append(fear["fear_score"])
                self.deception_au_scores.append(fear["deception_au_score"])
            except Exception:
                pass

    def get_summary(self):
        def _m(arr): return float(np.mean(arr)) if arr else None
        def _s(arr): return float(np.std(arr)) if len(arr) > 2 else None
        return {
            "hr_mean": _m(self.hr_samples),
            "stress_mean": _m(self.stress_samples),
            "blink_mean": _m(self.blink_samples),
            "pupil_mean": _m(self.pupil_samples), "pupil_std": _s(self.pupil_samples),
            "gaze_h_std": _s(self.gaze_h_samples), "gaze_v_std": _s(self.gaze_v_samples),
            "yaw_mean": _m(self.pose_yaw_samples),
            "micro_count": self.micro_count,
            "micro_rate": self.micro_count / max(1, self.frame_count),
            "asymmetry_mean": _m(self.asymmetry_samples),
            "fear_mean": _m(self.fear_scores),
            "deception_au_mean": _m(self.deception_au_scores),
            "duchenne_count": self.duchenne_count,
            "fake_smile_count": self.fake_smile_count,
            "fake_smile_rate": self.fake_smile_count / max(1, self.frame_count),
            "duration_s": round(time.time() - self.start_time, 1),
            "samples": len(self.hr_samples),
        }


def compute_deception_indicators(bs, ans, voc):
    """Research-corrected indicator weights based on meta-analyses."""
    indicators = {}
    vocal = voc or {}

    # ═══ TIER 1: MOST RELIABLE (weight 3.0) ═══

    # 1. Pupil dilation (Rashid et al. 2024: #1 predictor)
    # v9.7.3 fix (sibling of hr_spike fix below): same `+1e-6` floor pitfall.
    # Pupil std is ~0.005-0.02 in normalized units; even a 0.005 jitter
    # delta saturated score=1.0 with a 0.005 baseline std. Three corrections
    # mirror the hr_spike treatment: dead-zone, denominator floor, positive-
    # only delta (cognitive load = pupil DILATION, not constriction).
    if bs.get("pupil_mean") and ans.get("pupil_mean"):
        delta = ans["pupil_mean"] - bs["pupil_mean"]
        std = bs.get("pupil_std") or 0.02
        delta_pos = max(0.0, delta - 0.01)            # 1 % dead-zone
        denom = max(std * 1.5, 0.03)                  # 3 % denominator floor
        indicators["pupil_load"] = {
            "score": round(min(1.0, delta_pos / denom), 3),
            "label": "Pupil Dilation (Cognitive Load)",
            "desc": (f"Pupil dilated +{delta:.3f} vs baseline"
                     if delta > 0.01 else "Pupil stable vs baseline"),
        }

    # 2. Fake smile detection (Ekman et al. 1991)
    ans_fake = ans.get("fake_smile_rate", 0)
    bs_fake = bs.get("fake_smile_rate", 0)
    if ans_fake > 0 or bs_fake > 0:
        indicators["fake_smile"] = {
            "score": round(min(1.0, ans_fake * 3), 3),
            "label": "Non-Duchenne (Fake) Smile",
            "desc": "Smiling without eye involvement" if ans_fake > 0.05 else "Genuine or no smile",
        }

    # 3. Facial asymmetry (Ekman, Hager & Friesen 1981)
    # v9.7.3 fix: previous code did `(ans * 2.5)` with NO baseline subtraction.
    # Every neutral face has 0.3-0.5 natural AU L/R imbalance from intrinsic
    # facial geometry; that's not deception, it's anatomy. Result: indicator
    # was stuck at 100 % for nearly every subject regardless of veracity.
    # SESSION_HANDOFF_2026-04-29 fixed this in person_state.py for the LIVE
    # deception path; truth_engine.py was missed. Match the fix here:
    # subtract baseline asymmetry and use a 5 % dead-zone.
    if ans.get("asymmetry_mean") is not None and bs.get("asymmetry_mean") is not None:
        asym_delta = (ans["asymmetry_mean"] or 0) - (bs["asymmetry_mean"] or 0)
        asym_delta_pos = max(0.0, asym_delta - 0.05)  # 5 % dead-zone
        indicators["facial_asymmetry"] = {
            "score": round(min(1.0, asym_delta_pos * 2.0), 3),
            "label": "Facial Asymmetry (Faked Expression)",
            "desc": (f"Asymmetry +{asym_delta:.2f} vs baseline"
                     if asym_delta > 0.05 else "Symmetric expressions"),
        }

    # ═══ TIER 2: MODERATELY RELIABLE (weight 2.0) ═══

    # 4. Fear AU leakage (Su & Levine 2016, >80% ML accuracy)
    # v9.8 fix: same dead-zone + baseline-subtraction pattern as the other
    # indicators. Pre-fix the `* 5` multiplier saturated on natural facial
    # noise (fear blendshape baseline ~ 0.02 with std ~0.01) and the
    # deception_au_mean term used the answer-only value (no baseline
    # subtraction), so any subject with anatomically high resting AU 4/7/14
    # got flagged as "fear leakage" regardless of veracity.
    if ans.get("deception_au_mean") is not None:
        fear_delta = (ans.get("fear_mean") or 0) - (bs.get("fear_mean") or 0)
        au_delta = ((ans.get("deception_au_mean") or 0)
                    - (bs.get("deception_au_mean") or 0))
        # 0.03 dead-zone on each channel; ~3 % above resting noise floor.
        fear_pos = max(0.0, fear_delta - 0.03)
        au_pos = max(0.0, au_delta - 0.03)
        indicators["fear_leakage"] = {
            "score": round(min(1.0, fear_pos * 4 + au_pos * 2), 3),
            "label": "Fear/Stress AU Leakage",
            "desc": ("Deception-associated AU pattern detected"
                     if fear_delta > 0.05 or au_delta > 0.05
                     else "No fear leakage"),
        }

    # 5. Blink SUPPRESSION (Mann et al. 2002: liars blink LESS)
    # v9.7.3 fix: same `+1e-6` denominator pitfall as hr_spike. With a low
    # baseline (e.g., 5 bpm), a 1-bpm decrease was scoring 0.67. Use a
    # 2-bpm dead-zone (smaller than typical 30 % drop literature reports)
    # plus a 4-bpm denominator floor.
    if bs.get("blink_mean") and ans.get("blink_mean"):
        blink_decrease = bs["blink_mean"] - ans["blink_mean"]  # positive = suppression
        decrease_pos = max(0.0, blink_decrease - 2.0)          # 2 bpm dead-zone
        denom = max(bs["blink_mean"] * 0.3, 4.0)               # 4 bpm floor
        indicators["blink_suppression"] = {
            "score": round(min(1.0, decrease_pos / denom), 3),
            "label": "Blink Suppression (Concentration)",
            "desc": (f"Blink rate -{blink_decrease:.1f} bpm vs baseline"
                     if blink_decrease > 2 else "Normal blink rate"),
        }

    # 6. Heart rate spike (Rashid et al. 2024: moderate predictor)
    # v9.7.3 fix: previous formula was hr_delta / (hr_std * 2 + 1e-6).
    # With a stable baseline (hr_std ~0.5 BPM, common with rPPG averaging
    # a calm sitting subject), a 1 BPM rise saturated score=1.0 -- making
    # the indicator fire on irrelevant noise. Three corrections:
    #   1) dead-zone: <3 BPM delta -> score 0 (sub-noise threshold)
    #   2) hard 5-BPM denominator floor (no matter how stable the
    #      baseline, rPPG noise is min ~3 BPM at the per-frame level)
    #   3) only count POSITIVE deltas (deception lit shows hr ELEVATED;
    #      a drop is not deception evidence -- max(hr_delta, 0))
    if bs.get("hr_mean") and ans.get("hr_mean"):
        hr_delta = ans["hr_mean"] - bs["hr_mean"]
        hr_std = bs.get("hr_std") or 5.0
        hr_delta_pos = max(0.0, hr_delta - 3.0)         # 3 BPM dead-zone
        hr_denom = max(hr_std * 2.0, 5.0)               # 5 BPM floor
        indicators["hr_spike"] = {
            "score": round(min(1.0, hr_delta_pos / hr_denom), 3),
            "label": "Heart Rate Increase",
            "desc": (f"HR +{hr_delta:.1f} BPM vs baseline"
                     if hr_delta > 3 else "HR stable vs baseline"),
        }

    # 7. Stress index delta
    # v9.7.3 fix: previous formula `* 3` saturated at delta=0.34. Stress index
    # noise floor is ~0.05-0.10 from rPPG/HRV jitter alone. Apply a 0.10
    # dead-zone before scaling.
    if bs.get("stress_mean") is not None and ans.get("stress_mean") is not None:
        stress_delta = ans["stress_mean"] - bs["stress_mean"]
        stress_delta_pos = max(0.0, stress_delta - 0.10)       # 10 % dead-zone
        indicators["stress_spike"] = {
            "score": round(min(1.0, stress_delta_pos * 3.0), 3),
            "label": "Physiological Stress",
            "desc": f"Stress {'elevated' if stress_delta > 0.1 else 'stable'}",
        }

    # 8. Micro-expressions (moderate, rare but diagnostic when present)
    indicators["micro_expressions"] = {
        "score": round(min(1.0, ans.get("micro_count", 0) * 0.25), 3),
        "label": "Micro-Expression Leakage",
        "desc": f"{ans.get('micro_count', 0)} micro-expressions during answer",
    }

    # ═══ TIER 3: VOCAL (weight 1.5) ═══

    # v9.8 fix: every vocal indicator below now subtracts the baseline
    # `bs.get(...)` value (collected during the calm-baseline phase) so
    # subjects who naturally have high pitch variability / jitter / pause
    # rates aren't flagged as deceptive purely on anatomy. Pattern
    # mirrors the facial-asymmetry fix from v9.7.3.

    # vocal_pitch: f0 standard deviation, baselined.
    if vocal.get("f0_std_hz"):
        bs_pitch = bs.get("vocal_f0_std", 0) or 0
        pitch_delta_pos = max(0.0, vocal["f0_std_hz"] - bs_pitch - 5.0)  # 5 Hz dead-zone
        indicators["vocal_pitch"] = {
            "score": round(min(1.0, pitch_delta_pos / 30.0), 3),  # 30 Hz delta = saturate
            "label": "Vocal Pitch Variability",
            "desc": f"Voice pitch +{pitch_delta_pos:.1f} Hz vs baseline" if pitch_delta_pos > 1 else "Pitch stable vs baseline",
        }

    # vocal_jitter: cycle-to-cycle period instability, baselined.
    if vocal.get("jitter"):
        bs_jit = bs.get("vocal_jitter", 0) or 0
        jit_delta_pos = max(0.0, vocal["jitter"] - bs_jit - 0.005)  # 0.5 % dead-zone
        indicators["vocal_jitter"] = {
            "score": round(min(1.0, jit_delta_pos / 0.04), 3),
            "label": "Voice Instability",
            "desc": (f"Jitter +{jit_delta_pos*100:.2f} pp vs baseline"
                     if jit_delta_pos > 0.005 else "Jitter at baseline"),
        }

    # response_latency: baseline subtraction so a slow-talking subject
    # isn't flagged. The measurement is per-question, baseline is the
    # calm preamble's natural pause duration.
    if vocal.get("response_latency_s") is not None:
        latency = vocal["response_latency_s"]
        bs_lat = bs.get("vocal_response_latency", 0) or 0
        lat_delta_pos = max(0.0, latency - bs_lat - 0.3)  # 300 ms dead-zone
        indicators["response_latency"] = {
            "score": round(min(1.0, lat_delta_pos / 1.5), 3),  # 1.5 s slower = saturate
            "label": "Response Delay",
            "desc": (f"Delay +{lat_delta_pos:.2f} s vs baseline"
                     if lat_delta_pos > 0.3 else "Prompt response"),
        }

    # speech_hesitation: pause-rate vs baseline pause-rate (per-second).
    if vocal.get("pause_count") is not None:
        dur = vocal.get("duration_s", 1) or 1
        rate = vocal["pause_count"] / max(dur, 1)
        bs_rate = bs.get("vocal_pause_rate", 0) or 0
        rate_delta_pos = max(0.0, rate - bs_rate - 0.05)  # 0.05 pause/s dead-zone
        indicators["speech_hesitation"] = {
            "score": round(min(1.0, rate_delta_pos * 4.0), 3),
            "label": "Speech Hesitations",
            "desc": (f"+{rate_delta_pos*60:.1f} pauses/min vs baseline"
                     if rate_delta_pos > 0.05 else f"{vocal['pause_count']} pauses (at baseline rate)"),
        }

    # ═══ TIER 4: LOW RELIABILITY (weight 0.3) ═══
    # Gaze aversion: DePaulo et al. 2003 meta-analysis d=0.10 (NOT reliable)
    # v9.7.3 fix: small `* 3` was saturating on natural saccadic noise.
    # 0.05 dead-zone matches the typical baseline gaze_h_std noise floor.
    if bs.get("gaze_h_std") is not None and ans.get("gaze_h_std") is not None:
        gaze_delta = (ans["gaze_h_std"] or 0) - (bs["gaze_h_std"] or 0)
        gaze_delta_pos = max(0.0, gaze_delta - 0.05)
        indicators["gaze_shift"] = {
            "score": round(min(1.0, gaze_delta_pos * 3.0), 3),
            "label": "Gaze Pattern (low reliability)",
            "desc": "Note: gaze aversion is NOT a reliable deception cue (DePaulo 2003)",
        }

    # ═══ COMPOSITE SCORE (research-corrected weights) ═══
    weights = {
        # Tier 1: most reliable
        "pupil_load": 3.0, "fake_smile": 3.0, "facial_asymmetry": 3.0,
        # Tier 2: moderately reliable
        "fear_leakage": 2.5, "blink_suppression": 2.0, "hr_spike": 2.0,
        "stress_spike": 1.5, "micro_expressions": 2.0,
        # Tier 3: vocal
        "vocal_pitch": 1.5, "vocal_jitter": 1.0, "response_latency": 1.5, "speech_hesitation": 1.0,
        # Tier 4: low reliability
        "gaze_shift": 0.3,
    }
    total_w, total_s = 0, 0
    for key, ind in indicators.items():
        w = weights.get(key, 1.0)
        total_w += w
        total_s += ind["score"] * w

    composite = round(total_s / (total_w + 1e-6), 3)
    if composite > 0.55:
        verdict = "HIGH DECEPTION PROBABILITY"
    elif composite > 0.35:
        verdict = "MODERATE DECEPTION SIGNALS"
    elif composite > 0.15:
        verdict = "SLIGHT INCONGRUENCE"
    else:
        verdict = "LOW DECEPTION PROBABILITY"

    return {
        "composite_score": composite,
        "verdict": verdict,
        "indicators": indicators,
        "indicator_count": len(indicators),
    }


# ── State management ──
_baseline = None
_test_active = False
_answer_capture = None

def start_baseline():
    global _baseline, _test_active
    _baseline = DeceptionBaseline()
    _test_active = True
    return {"status": "baseline_collecting", "msg": f"Look at camera calmly for {BASELINE_DURATION} seconds..."}

def feed_baseline(av, bs=None):
    if _baseline and not _baseline.ready:
        _baseline.add_frame(av, bs)
        return {"ready": _baseline.ready, "elapsed": round(time.time() - _baseline.start_time, 1)}
    return {"ready": True}

def get_baseline_status():
    if _baseline is None:
        return {"active": False}
    return {"active": True, "ready": _baseline.ready,
            "summary": _baseline.get_summary() if _baseline.ready else None}

def start_answer_capture():
    global _answer_capture
    _answer_capture = AnswerCapture()
    return {"status": "capturing"}

def feed_answer(av, bs=None):
    if _answer_capture:
        _answer_capture.add_frame(av, bs)

def stop_answer_capture():
    global _answer_capture
    if _answer_capture:
        summary = _answer_capture.get_summary()
        _answer_capture = None
        return summary
    return None

def get_full_analysis(baseline_summary, answer_summary, vocal_analysis, transcript):
    indicators = compute_deception_indicators(baseline_summary, answer_summary, vocal_analysis)
    return {
        "baseline": baseline_summary,
        "answer_biometrics": answer_summary,
        "vocal_analysis": vocal_analysis,
        "transcript": transcript,
        "deception_indicators": indicators,
    }

def reset():
    global _baseline, _test_active, _answer_capture
    _baseline = None
    _test_active = False
    _answer_capture = None
