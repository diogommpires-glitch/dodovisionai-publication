"""
DODO VISION AI v8.8 — mmWave Radar Fusion (CamWave-Emo Architecture)

Scaffolds the dual-modality CamWave-Emo framework for 120 GHz FMCW radar
integration alongside the existing optical camera pipeline. During
calibration, the camera provides ground-truth facial expression labels
via CCA alignment. At runtime, the radar operates independently for
privacy-preserving emotion detection.

Signal processing pipeline:
  1. Raw beat signal ingestion from FMCW radar
  2. Low-pass filtering for environmental noise removal
  3. Hamming windowing into overlapping segments
  4. Continuous Wavelet Transform (CWT) for time-frequency decomposition
  5. Power spectrogram generation: S(tau, f) = |X(tau, f)|^2
  6. Micro-Doppler signature extraction for emotion classification

Radar hardware parameters (optimized for micro-facial motion):
  - Operating frequency: 120 GHz (wavelength ~2.45 mm)
  - ADC sample rate: 4000 ksps
  - Chirp density: 128 chirps/frame
  - Range resolution: 3.8-5.0 cm (3-4 GHz bandwidth)

Contactless vital sign extraction:
  - Respiration: 0.1-0.5 Hz bandpass on unwrapped phase
  - Heart rate: 0.8-2.0 Hz bandpass on unwrapped phase
  - Phase-to-displacement: d = (lambda / 4*pi) * delta_phi

References:
  - CamWave-Emo (ACM MM 2024): dual-modality emotion recognition
  - FMCW radar micro-Doppler: IEEE Trans. Microwave Theory
"""
from __future__ import annotations
import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

# ── Radar hardware constants ──
OPERATING_FREQ_GHZ = 120.0
WAVELENGTH_M = 3e8 / (OPERATING_FREQ_GHZ * 1e9)  # ~2.5 mm
ADC_SAMPLE_RATE_KSPS = 4000
CHIRPS_PER_FRAME = 128
BANDWIDTH_GHZ = 4.0  # 120-124 GHz sweep
RANGE_RESOLUTION_M = 3e8 / (2 * BANDWIDTH_GHZ * 1e9)  # ~3.75 cm

# ── Vital sign bandpass ranges (Hz) ──
RESP_BAND = (0.1, 0.5)     # 6-30 breaths per minute
CARDIAC_BAND = (0.8, 2.0)  # 48-120 BPM

# ── CWT parameters ──
CWT_SCALES_MIN = 4
CWT_SCALES_MAX = 128


@dataclass
class RadarFrame:
    """Single radar frame: chirps x ADC samples."""
    beat_signal: np.ndarray   # shape: (chirps, adc_samples)
    timestamp: float = field(default_factory=time.time)


@dataclass
class MicroDopplerSignature:
    """Extracted micro-Doppler features from a time-frequency analysis."""
    power_spectrogram: Optional[np.ndarray] = None  # (time, freq)
    spectral_centroid: float = 0.0
    spectral_entropy: float = 0.0
    total_power: float = 0.0
    timestamp: float = 0.0


@dataclass
class ContactlessVitals:
    """Radar-derived vital signs from phase unwrapping."""
    heart_rate_bpm: Optional[float] = None
    respiratory_rate_bpm: Optional[float] = None
    displacement_mm: Optional[float] = None
    confidence: float = 0.0


class MmWaveRadarFusion:
    """Asynchronous 120 GHz FMCW radar fusion engine.

    Implements the CamWave-Emo dual-modality architecture:
    - Calibration mode: camera labels + radar spectrograms -> CCA alignment
    - Runtime mode: radar-only emotion classification (privacy-preserving)
    """

    def __init__(self, fs: float = None, window_size: int = 256):
        self.fs = fs or (ADC_SAMPLE_RATE_KSPS * 1000)
        self.window_size = window_size
        self._frame_buffer: Deque[RadarFrame] = deque(maxlen=512)
        self._phase_buffer: Deque[float] = deque(maxlen=4096)
        self._phase_times: Deque[float] = deque(maxlen=4096)
        self._calibration_pairs: List[Tuple[np.ndarray, str]] = []
        self._is_calibrating = False

    # ──────────────────────────────────────────────
    #  Signal ingestion
    # ──────────────────────────────────────────────

    def ingest_frame(self, beat_signal: np.ndarray) -> None:
        """Ingest a raw radar frame (chirps x ADC samples)."""
        frame = RadarFrame(beat_signal=np.asarray(beat_signal, dtype=np.float64))
        self._frame_buffer.append(frame)

    def ingest_phase(self, phase_rad: float) -> None:
        """Ingest a single unwrapped phase sample from the beat signal."""
        self._phase_buffer.append(float(phase_rad))
        self._phase_times.append(time.time())

    # ──────────────────────────────────────────────
    #  Signal processing: windowing & filtering
    # ──────────────────────────────────────────────

    @staticmethod
    def hamming_window(signal: np.ndarray) -> np.ndarray:
        """Apply Hamming window to a 1D signal segment.

        Smooths segment edges to preserve frequency resolution
        and minimize spectral leakage in subsequent FFT/CWT.
        """
        w = np.hamming(len(signal))
        return signal * w

    @staticmethod
    def lowpass_filter(signal: np.ndarray, cutoff_hz: float, fs: float,
                       order: int = 4) -> np.ndarray:
        """Butterworth low-pass filter for environmental noise removal."""
        from scipy.signal import butter, filtfilt
        nyq = fs / 2.0
        b, a = butter(order, min(cutoff_hz / nyq, 0.99), btype='low')
        return filtfilt(b, a, signal)

    @staticmethod
    def bandpass_filter(signal: np.ndarray, lo: float, hi: float,
                        fs: float, order: int = 4) -> np.ndarray:
        """Butterworth bandpass for isolating physiological bands."""
        from scipy.signal import butter, filtfilt
        nyq = fs / 2.0
        b, a = butter(order, [max(lo / nyq, 0.01), min(hi / nyq, 0.99)],
                       btype='band')
        return filtfilt(b, a, signal)

    # ──────────────────────────────────────────────
    #  Time-frequency analysis
    # ──────────────────────────────────────────────

    def compute_cwt(self, signal: np.ndarray,
                    scales: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Continuous Wavelet Transform using Morlet wavelet.

        Superior to STFT for capturing rapid, non-stationary micro-facial
        expressions due to multi-scale decomposition.

        Returns (coefficients, frequencies).
        """
        from scipy.signal import morlet2, cwt as scipy_cwt
        n = len(signal)
        if scales is None:
            scales = np.arange(CWT_SCALES_MIN, min(CWT_SCALES_MAX, n // 2))
        # Morlet wavelet (w=6 is standard)
        def _morlet(M, s):
            return morlet2(M, s, w=6.0)

        coeffs = scipy_cwt(signal, _morlet, scales)
        # Convert scales to frequencies
        freqs = 6.0 * self.fs / (2 * np.pi * scales)
        return coeffs, freqs

    def power_spectrogram(self, signal: np.ndarray,
                          hop: int = 32) -> MicroDopplerSignature:
        """Generate power spectrogram S(tau, f) = |X(tau, f)|^2.

        Segments the signal with Hamming windows and computes CWT
        on each segment, then takes squared magnitude.
        """
        n = len(signal)
        if n < self.window_size:
            return MicroDopplerSignature(timestamp=time.time())

        segments = []
        timestamps = []
        for start in range(0, n - self.window_size + 1, hop):
            seg = signal[start:start + self.window_size]
            seg = self.hamming_window(seg)
            segments.append(seg)
            timestamps.append(start / self.fs)

        if not segments:
            return MicroDopplerSignature(timestamp=time.time())

        # Compute CWT on first segment to determine shape
        coeffs0, freqs = self.compute_cwt(segments[0])
        spectrogram = np.zeros((len(segments), len(freqs)))

        for i, seg in enumerate(segments):
            c, _ = self.compute_cwt(seg)
            spectrogram[i, :] = np.mean(np.abs(c) ** 2, axis=1)

        # Extract scalar features
        total = float(spectrogram.sum())
        if total > 0:
            p = spectrogram.flatten() / total
            centroid = float((np.arange(len(p)) * p).sum() / len(p))
            ent = -float((p * np.log(p + 1e-12)).sum())
            ent_norm = ent / max(1e-12, math.log(len(p)))
        else:
            centroid, ent_norm = 0.0, 0.0

        return MicroDopplerSignature(
            power_spectrogram=spectrogram,
            spectral_centroid=round(centroid, 4),
            spectral_entropy=round(ent_norm, 4),
            total_power=round(total, 6),
            timestamp=time.time(),
        )

    # ──────────────────────────────────────────────
    #  Contactless vital sign extraction
    # ──────────────────────────────────────────────

    def extract_vitals(self) -> ContactlessVitals:
        """Extract heart rate and respiratory rate from phase buffer.

        Uses the unwrapped phase of the radar beat signal:
          displacement = (lambda / 4*pi) * delta_phi

        Bandpass filters isolate respiration (0.1-0.5 Hz) and
        cardiac activity (0.8-2.0 Hz).
        """
        if len(self._phase_buffer) < 256:
            return ContactlessVitals()

        phase = np.array(self._phase_buffer, dtype=np.float64)
        times = np.array(self._phase_times)

        # Estimate sampling rate from timestamps
        if len(times) > 1:
            dt = times[-1] - times[0]
            fs = len(times) / max(dt, 0.01)
        else:
            fs = 20.0

        # Phase to displacement (mm)
        displacement = (WAVELENGTH_M / (4 * np.pi)) * phase * 1000

        hr_bpm = None
        rr_bpm = None
        conf = 0.0

        # Respiratory rate extraction
        try:
            resp_sig = self.bandpass_filter(displacement, *RESP_BAND, fs)
            fft_r = np.abs(np.fft.rfft(resp_sig))
            freqs_r = np.fft.rfftfreq(len(resp_sig), 1.0 / fs)
            mask_r = (freqs_r >= RESP_BAND[0]) & (freqs_r <= RESP_BAND[1])
            if np.any(mask_r):
                peak_idx = np.argmax(fft_r[mask_r])
                rr_bpm = round(float(freqs_r[mask_r][peak_idx]) * 60.0, 1)
        except Exception:
            pass

        # Heart rate extraction
        try:
            cardiac_sig = self.bandpass_filter(displacement, *CARDIAC_BAND, fs)
            fft_c = np.abs(np.fft.rfft(cardiac_sig))
            freqs_c = np.fft.rfftfreq(len(cardiac_sig), 1.0 / fs)
            mask_c = (freqs_c >= CARDIAC_BAND[0]) & (freqs_c <= CARDIAC_BAND[1])
            if np.any(mask_c):
                fft_masked = fft_c[mask_c]
                peak_idx = np.argmax(fft_masked)
                hr_bpm = round(float(freqs_c[mask_c][peak_idx]) * 60.0, 1)
                # SNR-based confidence
                bg = float(np.mean(np.delete(fft_masked, peak_idx))) if len(fft_masked) > 1 else 1e-6
                snr = float(fft_masked[peak_idx]) / (bg + 1e-6)
                conf = min(1.0, snr / 8.0)
        except Exception:
            pass

        return ContactlessVitals(
            heart_rate_bpm=hr_bpm,
            respiratory_rate_bpm=rr_bpm,
            displacement_mm=round(float(np.std(displacement)), 3) if len(displacement) > 0 else None,
            confidence=round(conf, 2),
        )

    # ──────────────────────────────────────────────
    #  CamWave-Emo calibration (CCA alignment stub)
    # ──────────────────────────────────────────────

    def start_calibration(self) -> None:
        """Begin calibration phase: collect camera+radar pairs."""
        self._is_calibrating = True
        self._calibration_pairs.clear()

    def add_calibration_sample(self, spectrogram: np.ndarray,
                                camera_label: str) -> None:
        """Add a (radar_spectrogram, camera_emotion_label) pair."""
        if self._is_calibrating:
            self._calibration_pairs.append((spectrogram, camera_label))

    def finish_calibration(self) -> Dict:
        """Finish calibration and compute CCA alignment.

        Canonical Correlation Analysis aligns the latent features of
        radar spectrograms with visual emotion labels. Once trained,
        the camera can be disabled for privacy-preserving runtime.

        Returns calibration metrics.
        """
        self._is_calibrating = False
        n = len(self._calibration_pairs)
        if n < 50:
            return {"status": "insufficient_data", "pairs": n, "min_required": 50}

        # CCA alignment stub — requires sklearn or custom implementation
        # In production, this would:
        # 1. Flatten spectrograms into feature vectors
        # 2. One-hot encode emotion labels
        # 3. Apply sklearn.cross_decomposition.CCA
        # 4. Store the transformation matrices for runtime projection
        return {
            "status": "calibration_complete",
            "pairs": n,
            "note": "CCA alignment stub — integrate sklearn CCA for production",
        }

    # ──────────────────────────────────────────────
    #  Runtime emotion inference (stub)
    # ──────────────────────────────────────────────

    def classify_emotion(self, spectrogram: np.ndarray) -> Optional[Dict]:
        """Classify emotion from radar spectrogram alone (post-calibration).

        In production, this projects the spectrogram through the CCA-aligned
        transformation and feeds it into a CNN+RNN classifier.

        Returns None until a trained model is loaded.
        """
        # Stub: requires trained model weights
        return None

    def summary(self) -> Dict:
        """Diagnostic summary for the status endpoint."""
        vitals = self.extract_vitals()
        return {
            "frames_buffered": len(self._frame_buffer),
            "phase_samples": len(self._phase_buffer),
            "calibrating": self._is_calibrating,
            "calibration_pairs": len(self._calibration_pairs),
            "vitals": {
                "hr_bpm": vitals.heart_rate_bpm,
                "rr_bpm": vitals.respiratory_rate_bpm,
                "displacement_mm": vitals.displacement_mm,
                "confidence": vitals.confidence,
            },
            "config": {
                "freq_ghz": OPERATING_FREQ_GHZ,
                "wavelength_mm": round(WAVELENGTH_M * 1000, 3),
                "adc_ksps": ADC_SAMPLE_RATE_KSPS,
                "chirps_per_frame": CHIRPS_PER_FRAME,
                "range_res_cm": round(RANGE_RESOLUTION_M * 100, 2),
            },
        }
