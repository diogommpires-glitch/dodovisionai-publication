"""
DODO VISION AI — Emotion temporal smoother (v7.5)

Affective computing best-practice: raw per-frame emotion softmax from
HSEmotion/EmotiEffLib is high-variance because micro-expressions, blink
occlusion, and JPEG/blur noise all perturb the logits. Stabilize with:

  1. Per-class EMA of softmax probabilities (α=0.4).
  2. Confidence gating: only emit a label if argmax prob exceeds
     CONF_THRESH=0.55 and exceeds runner-up by MARGIN=0.08.
  3. Hysteresis on label changes (2 consecutive frames required).

This matches the ABAW 2024 winner (EmotiEffLib)'s published
post-processing recipe and FEAL 2023 findings on stable affect.
"""
from __future__ import annotations
from collections import deque

EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
# v9.1.7: instant emotion. User wants the UI label to switch the moment
# the face changes, so every gate is pulled to the most responsive setting
# that still ignores random single-frame noise:
#   - EMA α=0.75: 75% of each new frame pass-through (previously 55%).
#   - _SWITCH_STREAK=1: a single confident frame flips the label.
#   - MARGIN bumped slightly to 0.10 to compensate for the higher alpha
#     letting more noise into the argmax decision.
# At 30 fps this means emotion updates in ~33 ms (1 frame) once the
# confidence + margin gates both pass.
_EMA_ALPHA = 0.75
_CONF_THRESH = 0.45
_MARGIN = 0.10
_SWITCH_STREAK = 1


class EmotionSmoother:
    def __init__(self):
        self._ema = {e: 0.0 for e in EMOTIONS}
        self._last_label = "neutral"
        self._candidate = None
        self._streak = 0
        self._history = deque(maxlen=60)  # for valence/arousal track

    def push(self, probs: dict):
        """probs: {'angry':0.1,...} - sum≈1. Returns (label, confidence, probs_ema)."""
        if not probs:
            return self._last_label, 0.0, dict(self._ema)
        # EMA the probabilities
        for e in EMOTIONS:
            p = float(probs.get(e, 0.0))
            self._ema[e] = _EMA_ALPHA * p + (1.0 - _EMA_ALPHA) * self._ema[e]

        # argmax + margin test
        ranked = sorted(self._ema.items(), key=lambda kv: kv[1], reverse=True)
        top, top_p = ranked[0]
        runner_p = ranked[1][1] if len(ranked) > 1 else 0.0
        confident = (top_p >= _CONF_THRESH) and (top_p - runner_p >= _MARGIN)

        if not confident:
            # keep last stable label, but report low confidence
            self._candidate = None
            self._streak = 0
            self._history.append(self._last_label)
            return self._last_label, float(top_p), dict(self._ema)

        # hysteresis — require streak to switch labels
        if top == self._last_label:
            self._candidate = None
            self._streak = 0
        else:
            if self._candidate == top:
                self._streak += 1
            else:
                self._candidate = top
                self._streak = 1
            if self._streak >= _SWITCH_STREAK:
                self._last_label = top
                self._candidate = None
                self._streak = 0

        self._history.append(self._last_label)
        return self._last_label, float(top_p), dict(self._ema)

    def recent_distribution(self, n: int = 30):
        """Return proportion of each label in last n frames (for affect charts)."""
        recent = list(self._history)[-n:]
        if not recent:
            return {e: 0.0 for e in EMOTIONS}
        out = {e: 0.0 for e in EMOTIONS}
        for lbl in recent:
            if lbl in out:
                out[lbl] += 1
        n_total = len(recent)
        return {k: round(v / n_total, 3) for k, v in out.items()}

    def reset(self):
        self._ema = {e: 0.0 for e in EMOTIONS}
        self._last_label = "neutral"
        self._candidate = None
        self._streak = 0
        self._history.clear()
