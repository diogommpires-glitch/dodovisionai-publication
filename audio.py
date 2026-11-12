"""
DODO VISION AI v6 — Audio Analysis
Whisper speech-to-text + vocal stress analysis (pitch, jitter, speech rate, hesitations).
Uses librosa for acoustic features and openai-whisper for transcription.
"""
import tempfile, os
import numpy as np
import config as cfg

# ── Whisper STT ──
_whisper_model = None
WHISPER_LOADED = False

def _load_whisper():
    global _whisper_model, WHISPER_LOADED
    if _whisper_model is not None:
        return True
    try:
        import whisper
        model_size = getattr(cfg, "WHISPER_MODEL", "small")
        print(f"[..] Loading Whisper {model_size} (first time takes ~30s)...")
        _whisper_model = whisper.load_model(model_size)
        WHISPER_LOADED = True
        print(f"[OK] Whisper {model_size} loaded")
        return True
    except ImportError:
        print("[--] Whisper not installed (pip install openai-whisper)")
        return False
    except Exception as e:
        print(f"[--] Whisper load error: {e}")
        return False

def transcribe(audio_bytes, sr=16000):
    """Transcribe audio bytes to text using Whisper.
    Returns: {"text": str, "language": str, "segments": list}
    """
    if not _load_whisper():
        return {"text": "", "error": "Whisper not available"}
    try:
        # Write to temp WAV file
        import soundfile as sf
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        if len(audio_np) < 1600:  # less than 0.1s
            return {"text": "", "error": "Audio too short"}
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_np, sr)
            tmp_path = f.name
        result = _whisper_model.transcribe(tmp_path, language=None, fp16=False)
        os.unlink(tmp_path)
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", "unknown"),
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in result.get("segments", [])
            ],
        }
    except Exception as e:
        return {"text": "", "error": str(e)}

def analyze_vocal_stress(audio_bytes, sr=16000):
    """Analyze vocal stress indicators from audio.
    Returns dict of deception-relevant vocal features.
    Research basis: F0 increase, jitter increase, speech rate changes during deception.
    """
    try:
        import librosa
    except ImportError:
        return {"error": "librosa not installed"}
    try:
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        if len(audio_np) < sr:  # less than 1 second
            return {"error": "Audio too short for analysis"}
        duration = len(audio_np) / sr

        # Fundamental frequency (F0) - pitch
        f0, voiced_flag, _ = librosa.pyin(
            audio_np, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), sr=sr
        )
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])

        if len(f0_clean) < 5:
            return {
                "duration_s": round(duration, 2),
                "error": "Not enough voiced segments",
            }

        f0_mean = float(np.mean(f0_clean))
        f0_std = float(np.std(f0_clean))
        f0_range = float(np.max(f0_clean) - np.min(f0_clean))

        # Jitter (pitch perturbation) - higher = more vocal instability
        f0_diffs = np.abs(np.diff(f0_clean))
        jitter = float(np.mean(f0_diffs) / (f0_mean + 1e-6))

        # Speech rate estimation
        # Use energy-based syllable detection
        rms = librosa.feature.rms(y=audio_np, frame_length=1024, hop_length=256)[0]
        rms_thresh = np.mean(rms) * 0.5
        speech_frames = np.sum(rms > rms_thresh)
        total_frames = len(rms)
        speech_ratio = speech_frames / (total_frames + 1e-6)

        # Onset detection for syllable-like events
        onset_env = librosa.onset.onset_strength(y=audio_np, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        syllable_rate = len(onsets) / duration  # syllables per second

        # Pause detection (gaps > 300ms)
        frame_dur = 256 / sr
        silent_runs = []
        current_run = 0
        for r in rms:
            if r < rms_thresh:
                current_run += 1
            else:
                if current_run * frame_dur > 0.3:
                    silent_runs.append(current_run * frame_dur)
                current_run = 0
        pause_count = len(silent_runs)
        total_pause_dur = sum(silent_runs)

        # Response latency (time before first speech)
        first_speech = 0
        for i, r in enumerate(rms):
            if r > rms_thresh:
                first_speech = i * frame_dur
                break

        # Shimmer (amplitude perturbation)
        rms_voiced = rms[rms > rms_thresh]
        shimmer = 0.0
        if len(rms_voiced) > 2:
            amp_diffs = np.abs(np.diff(rms_voiced))
            shimmer = float(np.mean(amp_diffs) / (np.mean(rms_voiced) + 1e-6))

        # Spectral features
        spectral_centroid = float(np.mean(
            librosa.feature.spectral_centroid(y=audio_np, sr=sr)
        ))

        return {
            "duration_s": round(duration, 2),
            "f0_mean_hz": round(f0_mean, 1),
            "f0_std_hz": round(f0_std, 1),
            "f0_range_hz": round(f0_range, 1),
            "jitter": round(jitter, 4),
            "shimmer": round(shimmer, 4),
            "syllable_rate": round(syllable_rate, 2),
            "speech_ratio": round(speech_ratio, 3),
            "pause_count": pause_count,
            "total_pause_s": round(total_pause_dur, 2),
            "response_latency_s": round(first_speech, 3),
            "spectral_centroid_hz": round(spectral_centroid, 1),
        }
    except Exception as e:
        return {"error": str(e)}
