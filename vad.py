"""
DODO VISION AI — Voice Activity Detection (v7.5)

Adaptive energy-based VAD with:
  * rolling noise-floor estimate (robust to AC hum, fan noise, etc.)
  * zero-crossing-rate gate (rejects pure-tone noise)
  * spectral-flatness gate (speech is structured, noise is flat)
  * on/off hysteresis (speech-on needs 2 consecutive frames,
    speech-off needs 4 — prevents the "right corner flicker")

Falls back gracefully if librosa is missing (returns energy-only).
The public surface is:
    is_speech(audio_f32, sr) -> bool
    reset()
    stats() -> dict
"""
from __future__ import annotations
import numpy as np
from collections import deque

# ── Tunables (values from Google WebRTC VAD + Silero defaults, adapted) ──
_NOISE_FLOOR_LEN = 60          # ~60 frames of history for noise floor
_SPEECH_SNR_DB = 8.0           # speech must exceed noise floor by >=8 dB
_HARD_FLOOR_DB = -55.0         # absolute dB floor — anything quieter is silence
_ZCR_MIN = 0.02                # zero-crossing rate floor (pure DC → 0)
_ZCR_MAX = 0.35                # too high → fricative / noise
_SPECTRAL_FLATNESS_MAX = 0.6   # flatness > 0.6 ≈ white noise, reject
_HYST_ON = 2                   # consecutive frames to latch ON
_HYST_OFF = 4                  # consecutive frames to latch OFF

_noise_db = deque(maxlen=_NOISE_FLOOR_LEN)
_on_count = 0
_off_count = 0
_active = False
_stats = {
    "frames_total": 0,
    "frames_speech": 0,
    "frames_rejected_zcr": 0,
    "frames_rejected_flat": 0,
    "frames_rejected_snr": 0,
    "noise_floor_db": None,
}


def _zcr(audio):
    s = np.sign(audio)
    s[s == 0] = 1
    return float(np.mean(np.abs(np.diff(s)) / 2.0))


def _spectral_flatness(audio):
    """Geometric/arithmetic mean of power spectrum.
    1.0 = white noise, ~0.0 = pure tone / voiced speech."""
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) ** 2 + 1e-12
    gm = np.exp(np.mean(np.log(spec)))
    am = np.mean(spec)
    return float(gm / am)


def is_speech(audio_f32, sr: int = 16000) -> bool:
    """Return True if the chunk contains speech.
    Robust to constant background noise — adapts to it over ~60 frames.
    """
    global _on_count, _off_count, _active
    _stats["frames_total"] += 1

    if audio_f32 is None or len(audio_f32) < sr // 4:
        _off_count += 1
        _on_count = 0
        if _active and _off_count >= _HYST_OFF:
            _active = False
        return _active

    rms = float(np.sqrt(np.mean(audio_f32 ** 2)) + 1e-12)
    energy_db = float(20.0 * np.log10(rms + 1e-12))

    # Bootstrap noise floor with quiet samples.
    noise_floor = np.median(_noise_db) if _noise_db else _HARD_FLOOR_DB
    _stats["noise_floor_db"] = round(float(noise_floor), 1)

    snr_db = energy_db - noise_floor
    frame_ok = True

    if energy_db < _HARD_FLOOR_DB:
        frame_ok = False
    else:
        if snr_db < _SPEECH_SNR_DB:
            _stats["frames_rejected_snr"] += 1
            frame_ok = False
        else:
            zcr = _zcr(audio_f32)
            if zcr < _ZCR_MIN or zcr > _ZCR_MAX:
                _stats["frames_rejected_zcr"] += 1
                frame_ok = False
            else:
                flat = _spectral_flatness(audio_f32)
                if flat > _SPECTRAL_FLATNESS_MAX:
                    _stats["frames_rejected_flat"] += 1
                    frame_ok = False

    # Non-speech frames update the noise floor.
    if not frame_ok:
        _noise_db.append(energy_db)

    # Hysteresis latch.
    if frame_ok:
        _on_count += 1
        _off_count = 0
        if not _active and _on_count >= _HYST_ON:
            _active = True
    else:
        _off_count += 1
        _on_count = 0
        if _active and _off_count >= _HYST_OFF:
            _active = False

    if _active:
        _stats["frames_speech"] += 1
    return _active


def reset():
    global _on_count, _off_count, _active
    _noise_db.clear()
    _on_count = 0
    _off_count = 0
    _active = False
    for k in _stats:
        if k.startswith("frames_"):
            _stats[k] = 0


def stats():
    s = dict(_stats)
    t = s["frames_total"] or 1
    s["speech_ratio"] = round(s["frames_speech"] / t, 3)
    return s
