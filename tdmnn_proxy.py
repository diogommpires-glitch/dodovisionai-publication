"""
DODO VISION AI v8.8 — Temporal-Difference Minimizing Neural Network Proxy

Integrates P300 Event-Related Potential proxy data with the existing
cognitive state estimator and MMD stability gate to produce a composite
cognitive assessment over 5-15 second rolling windows.

Architecture (TDMNN stub):
  - Multibranch convolutional recurrent network (stub for future model)
  - Maximum Mean Discrepancy (MMD) minimisation across temporal sections
  - Leverages inherent 5-15s temporal stability of human emotional states

Proxy data sources (all contactless):
  - Pupillary dilation dynamics (P300-analog latency)
  - Micro-saccade rate (cognitive workload marker)
  - HRV LF/HF ratio (sympathovagal balance)
  - Heart rate variability shifts (autonomic arousal)
  - mmWave radar micro-Doppler (when available)

References:
  - TDMNN: Multimedia Tools & Applications (2023)
  - Privitera et al. 2010: Pupil dilation during visual target detection
  - Engbert & Kliegl 2003: Microsaccades and covert attention
"""
from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Sequence

import numpy as np

from mmd_stability import TemporalStabilityGate


# ── Temporal window parameters ──
TDMNN_WINDOW_MIN_S = 5.0     # minimum temporal section (seconds)
TDMNN_WINDOW_MAX_S = 15.0    # maximum temporal section (seconds)
TDMNN_FEATURE_DIM = 8        # proxy feature vector dimensionality
MMD_SIGMA = 1.0               # RBF kernel bandwidth
MMD_STABILITY_THRESH = 0.05   # threshold for regime change detection


@dataclass
class P300ProxyEvent:
    """A contactless proxy for the P300 event-related potential.

    The P300 is a positive ERP deflection ~300ms after oddball stimulus
    recognition. We approximate it via:
      - Pupillary dilation spike latency (200-800ms post-stimulus)
      - Heart rate acceleration onset
      - Micro-saccade suppression window
    """
    timestamp: float = 0.0
    stimulus_time: float = 0.0
    pupil_latency_ms: Optional[float] = None   # time to peak dilation
    hr_acceleration_ms: Optional[float] = None  # time to HR increase
    saccade_suppressed: bool = False             # saccade-free window
    confidence: float = 0.0

    @property
    def analog_latency_ms(self) -> Optional[float]:
        """Best-estimate P300-analog latency from available proxies."""
        vals = [v for v in [self.pupil_latency_ms, self.hr_acceleration_ms] if v is not None]
        if not vals:
            return None
        return round(sum(vals) / len(vals), 1)


@dataclass
class CognitiveAssessment:
    """Composite cognitive state assessment from TDMNN proxy analysis."""
    cognitive_load: float = 0.0      # 0..1 composite workload estimate
    attention_level: float = 0.0     # 0..1 sustained attention proxy
    temporal_stability: float = 0.0  # 0..1 emotional state consistency
    p300_detected: bool = False
    state_label: str = "unknown"     # "alert", "fatigued", "overloaded", "neutral"
    mmd_score: Optional[float] = None
    timestamp: float = 0.0


class TDMNNProxy:
    """Temporal-Difference Minimizing Neural Network proxy.

    Collects multimodal proxy feature vectors over rolling windows
    and uses MMD to assess temporal stability of the cognitive state.
    In production, the MMD term would be integrated into the loss
    function of a multibranch convolutional recurrent network.
    """

    def __init__(self, window_s: float = 10.0, sigma: float = MMD_SIGMA):
        self.window_s = max(TDMNN_WINDOW_MIN_S, min(TDMNN_WINDOW_MAX_S, window_s))
        self._gate = TemporalStabilityGate(
            window=8, sigma=sigma, mmd_threshold=MMD_STABILITY_THRESH
        )
        self._feature_buf: Deque[tuple] = deque(maxlen=512)  # (timestamp, feature_vec)
        self._p300_events: Deque[P300ProxyEvent] = deque(maxlen=64)
        self._last_assessment = CognitiveAssessment()

    def push_feature_vector(self, features: Sequence[float],
                            t: Optional[float] = None) -> None:
        """Push a proxy feature vector (8-dim) into the temporal buffer.

        Expected feature order:
          [0] proxy_cognitive_load
          [1] proxy_pupil_spectrum_cognitive
          [2] proxy_pupil_spectral_entropy
          [3] proxy_hrv_lf_hf_ratio
          [4] proxy_microsaccade_rate (normalized)
          [5] heart_rate_bpm (normalized to 0-1)
          [6] hrv_rmssd (normalized)
          [7] stress_index
        """
        if t is None:
            t = time.time()
        fv = [float(x) if x is not None else 0.0 for x in features]
        # Pad or truncate to TDMNN_FEATURE_DIM
        fv = (fv + [0.0] * TDMNN_FEATURE_DIM)[:TDMNN_FEATURE_DIM]
        self._feature_buf.append((t, fv))
        self._gate.push(fv)

    def push_p300_event(self, event: P300ProxyEvent) -> None:
        """Record a P300-analog detection event."""
        self._p300_events.append(event)

    def assess(self) -> CognitiveAssessment:
        """Compute cognitive state assessment using MMD temporal stability.

        The TDMNN principle: human emotional and cognitive states are
        stable over 5-15 second windows. If the MMD between consecutive
        windows exceeds threshold, a regime change (state transition) is
        occurring.
        """
        now = time.time()

        if len(self._feature_buf) < 4:
            return CognitiveAssessment(timestamp=now)

        # Split buffer into temporal sections within the window
        cutoff = now - self.window_s
        recent = [fv for t, fv in self._feature_buf if t >= cutoff]
        if len(recent) < 4:
            return CognitiveAssessment(timestamp=now)

        # MMD-based temporal stability
        mmd_score = self._gate.score()
        stability = 1.0 - min(1.0, (mmd_score or 0.0) / (MMD_STABILITY_THRESH * 4))

        # Aggregate cognitive load from feature vectors (mean over window)
        arr = np.array(recent, dtype=float)
        mean_features = arr.mean(axis=0)
        cognitive_load = float(np.clip(mean_features[0], 0, 1))
        attention = float(np.clip(1.0 - mean_features[2], 0, 1))  # inverse entropy

        # P300 detection: check recent events
        p300_recent = [e for e in self._p300_events if e.timestamp >= cutoff]
        p300_detected = len(p300_recent) > 0

        # State label heuristic
        if cognitive_load > 0.8:
            label = "overloaded"
        elif cognitive_load > 0.5:
            label = "alert"
        elif attention < 0.3:
            label = "fatigued"
        else:
            label = "neutral"

        self._last_assessment = CognitiveAssessment(
            cognitive_load=round(cognitive_load, 3),
            attention_level=round(attention, 3),
            temporal_stability=round(stability, 3),
            p300_detected=p300_detected,
            state_label=label,
            mmd_score=round(mmd_score, 4) if mmd_score is not None else None,
            timestamp=now,
        )
        return self._last_assessment

    def from_cognitive_snapshot(self, snapshot: Dict) -> None:
        """Convenience: push from a CognitiveStateEstimator.snapshot() dict.

        Maps the proxy fields to the 8-dim feature vector expected by
        push_feature_vector().
        """
        fv = [
            snapshot.get("proxy_cognitive_load_0_1", 0.0),
            snapshot.get("proxy_pupil_spectrum_cognitive", 0.0),
            snapshot.get("proxy_pupil_spectral_entropy", 0.0),
            snapshot.get("proxy_hrv_lf_hf_ratio", 0.0),
            min(1.0, (snapshot.get("proxy_microsaccade_rate_per_min", 0.0) or 0.0) / 60.0),
            0.0,  # placeholder: normalized HR
            0.0,  # placeholder: normalized HRV
            0.0,  # placeholder: stress_index
        ]
        self.push_feature_vector(fv)

    def summary(self) -> Dict:
        """Diagnostic summary for status endpoint."""
        a = self._last_assessment
        return {
            "cognitive_load": a.cognitive_load,
            "attention_level": a.attention_level,
            "temporal_stability": a.temporal_stability,
            "p300_detected": a.p300_detected,
            "state_label": a.state_label,
            "mmd_score": a.mmd_score,
            "feature_buffer_size": len(self._feature_buf),
            "p300_events": len(self._p300_events),
            "window_s": self.window_s,
        }
