"""
DODO VISION AI — Attention gate (v7.6)

Inspired by DARPA CT2WS (Cognitive Technology Threat Warning System):
a computer-vision pipeline proposes candidate events, and a human-
response signature (P300 in the original; pupillary + HR reaction here)
confirms whether the observer's attention was actually captured.

In CT2WS, EEG-P300 reduced 810 false positives / 2304 targets to 5
false alarms per hour. We approximate the same effect without EEG:

  1. Upstream caller flags a "candidate" event (radar motion,
     voice onset, face-region anomaly, gaze divergence, etc.).
  2. We log the event timestamp into CognitiveStateEstimator.
  3. After the CT2WS attention window (200-800 ms), we measure:
       - pupil dilation delta > Δmin    (locus-coeruleus arousal)
       - instantaneous HR acceleration    (sympathetic trigger)
       - microsaccade rate suppression    (attention capture)
  4. The three votes produce a 0-1 confirmation score. The gate
     returns CONFIRMED / UNCONFIRMED / INSUFFICIENT_SIGNAL.

None of this is neural recording. It is autonomic/oculomotor
confirmation of attention, as used in operational DMS systems.
"""
from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

PUPIL_DELTA_MIN = 0.02        # normalized iris-diameter change
HR_ACCEL_BPM_MIN = 2.0        # BPM increase vs. 3-s pre-baseline
MICRO_SUPPRESS_RATIO = 0.7    # post/pre saccade-rate ratio for "capture"
WINDOW_LO_MS = 200
WINDOW_HI_MS = 800
PRE_BASELINE_S = 0.5


@dataclass
class AttentionConfirmation:
    event_t: float
    confirmed: bool
    score: float                          # 0-1
    votes: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


class AttentionGate:
    def __init__(self):
        self._events: Deque[float] = deque(maxlen=256)
        self._confirmed: List[AttentionConfirmation] = []
        # Small ring buffers of (t, value). Populated via push_* calls
        # from the orchestrator after each frame.
        self._pupil: Deque = deque(maxlen=2048)
        self._hr: Deque = deque(maxlen=2048)
        self._saccade_t: Deque[float] = deque(maxlen=2048)

    # ── ingestion ────────────────────────────────────────────────
    def push_pupil(self, dilation: Optional[float], t: Optional[float] = None):
        if dilation is None:
            return
        self._pupil.append((t if t is not None else time.time(), float(dilation)))

    def push_hr(self, hr_bpm: Optional[float], t: Optional[float] = None):
        if hr_bpm is None or hr_bpm <= 0:
            return
        self._hr.append((t if t is not None else time.time(), float(hr_bpm)))

    def push_saccade(self, t: Optional[float] = None):
        self._saccade_t.append(t if t is not None else time.time())

    def submit_event(self, t: Optional[float] = None) -> None:
        """Candidate event from an upstream detector (radar, voice, face)."""
        self._events.append(t if t is not None else time.time())

    # ── evaluation ───────────────────────────────────────────────
    def evaluate_latest(self, deadline_grace_s: float = 0.9) -> Optional[AttentionConfirmation]:
        """Resolve the most recent event once enough time has passed."""
        if not self._events:
            return None
        ev_t = self._events[-1]
        if (time.time() - ev_t) < WINDOW_HI_MS / 1000.0 + deadline_grace_s:
            return None  # still collecting data

        pupil_vote = self._pupil_delta_vote(ev_t)
        hr_vote = self._hr_accel_vote(ev_t)
        sacc_vote = self._saccade_suppression_vote(ev_t)

        votes = {}
        if pupil_vote is not None:
            votes["pupil"] = pupil_vote
        if hr_vote is not None:
            votes["hr_accel"] = hr_vote
        if sacc_vote is not None:
            votes["saccade_suppress"] = sacc_vote

        if not votes:
            conf = AttentionConfirmation(
                event_t=ev_t, confirmed=False, score=0.0,
                votes=votes, reason="insufficient_signal")
        else:
            score = sum(votes.values()) / len(votes)
            conf = AttentionConfirmation(
                event_t=ev_t, confirmed=score >= 0.5, score=round(score, 3),
                votes=votes,
                reason="confirmed" if score >= 0.5 else "below_threshold",
            )
        self._confirmed.append(conf)
        # consume
        try:
            self._events.remove(ev_t)
        except ValueError:
            pass
        return conf

    # ── voting helpers ───────────────────────────────────────────
    def _pupil_delta_vote(self, ev_t: float) -> Optional[float]:
        pre = [v for (t, v) in self._pupil if ev_t - PRE_BASELINE_S <= t < ev_t]
        post = [v for (t, v) in self._pupil
                if ev_t + WINDOW_LO_MS / 1000.0 <= t <= ev_t + WINDOW_HI_MS / 1000.0]
        if not pre or not post:
            return None
        baseline = sum(pre) / len(pre)
        peak = max(post)
        delta = peak - baseline
        # linear mapping, saturated at 3× Δmin
        return max(0.0, min(1.0, delta / (3.0 * PUPIL_DELTA_MIN)))

    def _hr_accel_vote(self, ev_t: float) -> Optional[float]:
        pre = [v for (t, v) in self._hr if ev_t - 3.0 <= t < ev_t]
        post = [v for (t, v) in self._hr
                if ev_t + WINDOW_LO_MS / 1000.0 <= t <= ev_t + 3.0]
        if not pre or not post:
            return None
        accel = max(post) - (sum(pre) / len(pre))
        return max(0.0, min(1.0, accel / (3.0 * HR_ACCEL_BPM_MIN)))

    def _saccade_suppression_vote(self, ev_t: float) -> Optional[float]:
        pre_n = sum(1 for t in self._saccade_t if ev_t - 1.0 <= t < ev_t)
        post_n = sum(1 for t in self._saccade_t
                     if ev_t + WINDOW_LO_MS / 1000.0 <= t <= ev_t + 1.0)
        if pre_n < 2:
            return None
        ratio = post_n / max(1, pre_n)
        # Lower ratio == more suppression == higher vote
        return max(0.0, min(1.0, (MICRO_SUPPRESS_RATIO - ratio) / MICRO_SUPPRESS_RATIO))

    # ── stats for /attention endpoint ────────────────────────────
    def stats(self) -> Dict:
        recent = self._confirmed[-50:]
        n_conf = sum(1 for c in recent if c.confirmed)
        return {
            "total_events_submitted": len(self._events) + len(self._confirmed),
            "total_evaluated": len(self._confirmed),
            "recent_confirm_rate": round(n_conf / max(1, len(recent)), 3),
            "last_decision": None if not recent else {
                "confirmed": recent[-1].confirmed,
                "score": recent[-1].score,
                "votes": recent[-1].votes,
                "reason": recent[-1].reason,
                "event_t": recent[-1].event_t,
            },
        }

    def reset(self):
        self._events.clear()
        self._confirmed.clear()
        self._pupil.clear()
        self._hr.clear()
        self._saccade_t.clear()


_default = AttentionGate()


def gate() -> AttentionGate:
    return _default
