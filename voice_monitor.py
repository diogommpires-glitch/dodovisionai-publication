"""
DODO VISION AI — Continuous Voice Monitor
Real-time vocal stress tracking from streaming audio chunks.
Uses lightweight librosa features (NOT Whisper) for speed.
Processes 3-second overlapping windows at ~1Hz.
"""
import numpy as np
import time
from collections import deque

# v7.5: adaptive VAD replaces the crude `energy_db > -35` gate.
# The old gate latched to True under any AC hum > ~40 dB SPL and
# caused the "constant microphone surveillance" flicker.
try:
    import vad as _vad
except Exception:
    _vad = None

# v7.5: EMA-smoothed stress score to stop the right-corner category
# from oscillating calm↔elevated every second on the same voice.
_STRESS_EMA_ALPHA = 0.3
_stress_ema = None
# Hysteresis thresholds for category (with 0.03 deadband around edges).
_CATEGORY_BAND = {
    "calm":     (-1e9, 0.08),
    "mild":     (0.05, 0.18),
    "elevated": (0.15, 0.33),
    "high":     (0.30,  1e9),
}
_last_category = "unknown"

# ── State ──
_active = False
_on_result_callback = None   # Step 3: callback to feed person states
_voice_metrics = {
    "f0_hz": None, "f0_std_hz": None, "jitter": None,
    "energy_db": None, "speech_active": False,
    "syllable_rate": None, "spectral_centroid_hz": None,
    "pause_count": None,   # v7.9: exposed so UI has a value to show
    "stress_level": "unknown",  # calm/mild/elevated/high
    "last_update": 0,
}
_f0_history = deque(maxlen=60)      # 1-minute of F0 readings
_jitter_history = deque(maxlen=60)
_energy_history = deque(maxlen=60)
_baseline_f0 = None
_baseline_jitter = None
_sample_count = 0

def get_voice_metrics():
    """Get current voice metrics. Returns None if inactive or stale."""
    if not _active:
        return None
    if time.time() - _voice_metrics["last_update"] > 5:
        return None
    return dict(_voice_metrics)

def get_voice_baseline():
    """Get the voice baseline (first 10 readings used as calm reference)."""
    return {
        "f0_baseline": _baseline_f0,
        "jitter_baseline": _baseline_jitter,
        "samples": _sample_count,
    }

def process_chunk(audio_bytes, sr=16000, mic_active=True):
    """Process a 3-second audio chunk and update voice metrics.
    Designed to be fast (<100ms) for real-time use.

    v9.2 (plan step 15): two short-circuits before the expensive pYIN path:
      1. If `mic_active=False`, return immediately (caller has the mic off).
      2. If the chunk is near-silent (RMS < 1e-3), skip librosa — the
         chunk is just background noise and pitch estimation would either
         fail or return meaningless values. Cheap NumPy RMS takes <0.5 ms.
    """
    global _baseline_f0, _baseline_jitter, _sample_count
    if not mic_active:
        return {"skipped": "mic_muted"}
    try:
        import librosa
    except ImportError:
        return {"error": "librosa not installed"}

    # v8.3: warmup guard — the very first chunk triggers librosa's lazy
    # internal imports (numba JIT, audioread probes). Skip processing on
    # chunk 0 so the caller gets a fast non-None response on chunk 1.
    # Note: must run BEFORE the silence gate so the warmup accounting
    # stays consistent even if the very first chunk happens to be silent.
    if _sample_count == 0:
        _sample_count += 1
        _voice_metrics["last_update"] = time.time()
        return dict(_voice_metrics)

    # v9.2 (plan step 15): silence gate — avoid loading librosa's pYIN
    # entirely when the chunk is just background noise. Cheap NumPy RMS.
    try:
        quick_np = np.frombuffer(audio_bytes, dtype=np.float32)
        if len(quick_np) > 0 and float(np.sqrt(np.mean(quick_np ** 2))) < 1e-3:
            return {"skipped": "silence", "rms": 0.0}
    except Exception:
        pass  # fall through to the full pipeline

    try:
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        if len(audio_np) < sr:  # less than 1 second
            return {"error": "chunk too short"}

        # Energy (RMS in dB)
        rms = librosa.feature.rms(y=audio_np, frame_length=1024, hop_length=256)[0]
        energy_db = float(20 * np.log10(np.mean(rms) + 1e-10))

        # v7.5: adaptive VAD (SNR vs noise floor + ZCR + spectral flatness
        # + 2-on / 4-off hysteresis). Falls back to old energy gate if
        # the vad module couldn't be imported.
        if _vad is not None:
            speech_active = _vad.is_speech(audio_np, sr=sr)
        else:
            speech_active = energy_db > -35

        # v7.9: pause-count is computed over every chunk — silence runs
        # longer than 300 ms are counted regardless of VAD verdict, so
        # the UI "Pauses" cell always has a real number.
        try:
            hop = 256
            frame = 1024
            rms_env = librosa.feature.rms(y=audio_np, frame_length=frame, hop_length=hop)[0]
            rms_db_env = 20 * np.log10(rms_env + 1e-10)
            silent_thresh = -45.0
            silent = rms_db_env < silent_thresh
            # Count silent runs ≥ 300 ms.
            min_run = int(round(0.3 * sr / hop))
            pause_count = 0
            run = 0
            for s in silent:
                if s:
                    run += 1
                else:
                    if run >= min_run:
                        pause_count += 1
                    run = 0
            if run >= min_run:
                pause_count += 1
        except Exception:
            pause_count = None

        # v7.9: if the gate says "not speech", we still compute F0 +
        # jitter on a best-effort basis so the UI cells populate. We
        # surface speech_active=False so downstream deception scoring
        # and the forensic log still know whether it's voiced, but the
        # numeric panel is no longer gated on VAD.
        if not speech_active:
            _voice_metrics["speech_active"] = False
            _voice_metrics["energy_db"] = round(energy_db, 1)
            _voice_metrics["pause_count"] = pause_count
            _voice_metrics["last_update"] = time.time()
            # Best-effort quiet-frame pitch — fast pYIN on a short window
            # is cheap. If it fails we leave previous values untouched.
            try:
                f0_q, _, _ = librosa.pyin(
                    audio_np, fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'), sr=sr,
                    frame_length=2048, hop_length=512
                )
                f0_q_clean = f0_q[~np.isnan(f0_q)] if f0_q is not None else np.array([])
                if len(f0_q_clean) >= 3:
                    f0_mean_q = float(np.mean(f0_q_clean))
                    f0_diffs_q = np.abs(np.diff(f0_q_clean))
                    jitter_q = float(np.mean(f0_diffs_q) / (f0_mean_q + 1e-6))
                    _voice_metrics["f0_hz"] = round(f0_mean_q, 1)
                    _voice_metrics["jitter"] = round(jitter_q, 4)
            except Exception:
                pass
            return _voice_metrics

        # Fundamental frequency (F0) via pYIN - fast pitch tracking
        f0, voiced_flag, _ = librosa.pyin(
            audio_np, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), sr=sr,
            frame_length=2048, hop_length=512
        )
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])

        if len(f0_clean) < 3:
            _voice_metrics["speech_active"] = True
            _voice_metrics["energy_db"] = round(energy_db, 1)
            _voice_metrics["last_update"] = time.time()
            return _voice_metrics

        f0_mean = float(np.mean(f0_clean))
        f0_std = float(np.std(f0_clean))

        # Jitter (pitch perturbation quotient)
        f0_diffs = np.abs(np.diff(f0_clean))
        jitter = float(np.mean(f0_diffs) / (f0_mean + 1e-6))

        # Syllable rate via onset detection
        onset_env = librosa.onset.onset_strength(y=audio_np, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        duration = len(audio_np) / sr
        syllable_rate = len(onsets) / duration if duration > 0 else 0

        # Spectral centroid (voice brightness)
        sc = float(np.mean(librosa.feature.spectral_centroid(y=audio_np, sr=sr)))

        # Update histories
        _f0_history.append(f0_mean)
        _jitter_history.append(jitter)
        _energy_history.append(energy_db)
        _sample_count += 1

        # Establish baseline from first 10 samples
        if _sample_count <= 10:
            if _sample_count == 10:
                _baseline_f0 = float(np.mean(list(_f0_history)))
                _baseline_jitter = float(np.mean(list(_jitter_history)))
                print(f"[VOICE] Baseline established: F0={_baseline_f0:.1f}Hz, jitter={_baseline_jitter:.4f}")

        # v7.5: EMA-smoothed stress score with hysteresis on category.
        # Fixes the right-corner stress label bouncing every second.
        global _stress_ema, _last_category
        stress_level = _last_category if _last_category != "unknown" else "calm"
        if _baseline_f0 is not None:
            f0_elevation = (f0_mean - _baseline_f0) / (_baseline_f0 + 1e-6)
            jitter_elevation = (jitter - _baseline_jitter) / (_baseline_jitter + 1e-6)
            vocal_stress = max(0.0, f0_elevation * 0.6 + jitter_elevation * 0.4)
            _stress_ema = vocal_stress if _stress_ema is None else (
                _STRESS_EMA_ALPHA * vocal_stress + (1.0 - _STRESS_EMA_ALPHA) * _stress_ema
            )
            # Hysteresis: stay in current band unless we fully cross the next one.
            score = _stress_ema
            if _last_category in _CATEGORY_BAND:
                lo, hi = _CATEGORY_BAND[_last_category]
                if lo <= score < hi:
                    stress_level = _last_category
                else:
                    # find strictly new band
                    for name, (a, b) in _CATEGORY_BAND.items():
                        if a <= score < b:
                            stress_level = name
                            break
            else:
                for name, (a, b) in _CATEGORY_BAND.items():
                    if a <= score < b:
                        stress_level = name
                        break
            _last_category = stress_level

        # Update global metrics
        _voice_metrics.update({
            "f0_hz": round(f0_mean, 1),
            "f0_std_hz": round(f0_std, 1),
            "jitter": round(jitter, 4),
            "energy_db": round(energy_db, 1),
            "speech_active": True,
            "syllable_rate": round(syllable_rate, 2),
            "spectral_centroid_hz": round(sc, 0),
            "pause_count": pause_count,  # v7.9
            "stress_level": stress_level,
            "last_update": time.time(),
        })

        # Step 3: Feed vocal metrics to person states via callback
        if _on_result_callback:
            try:
                _on_result_callback({
                    "f0": f0_mean, "jitter": jitter,
                    "energy_db": energy_db, "stress_level": stress_level,
                    "syllable_rate": syllable_rate,
                    "f0_baseline": _baseline_f0,
                    "jitter_baseline": _baseline_jitter,
                })
            except Exception:
                pass

        # v8.1: feed audio chunk to diarizer for speaker tracking
        try:
            from diarization import get_diarizer
            get_diarizer().ingest(audio_np, sr)
        except Exception:
            pass

        return dict(_voice_metrics)

    except Exception as e:
        return {"error": str(e)}

def start():
    global _active
    _active = True
    print("[OK] Voice monitor started")

def stop():
    global _active
    _active = False

def set_deception_callback(callback):
    """Step 3: Set callback for continuous voice -> deception feeding.
    callback(voice_result_dict) called after each processed chunk."""
    global _on_result_callback
    _on_result_callback = callback

def get_vad_stats():
    """Expose VAD diagnostics (noise floor, speech ratio) for the UI/API."""
    if _vad is None:
        return None
    return _vad.stats()

def reset():
    global _active, _baseline_f0, _baseline_jitter, _sample_count, _stress_ema, _last_category
    _active = False
    _baseline_f0 = None
    _baseline_jitter = None
    _sample_count = 0
    _stress_ema = None
    _last_category = "unknown"
    if _vad is not None:
        _vad.reset()
    _f0_history.clear()
    _jitter_history.clear()
    _energy_history.clear()
    _voice_metrics.update({
        "f0_hz": None, "f0_std_hz": None, "jitter": None,
        "energy_db": None, "speech_active": False,
        "syllable_rate": None, "spectral_centroid_hz": None,
        "stress_level": "unknown", "last_update": 0,
    })
