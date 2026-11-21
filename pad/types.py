"""
DODO VISION AI v9.4 — PAD type contracts.

All PAD-internal data structures. No other module in the repo may
instantiate `PADScore` directly — it must come from `pad.fusion.fuse()`.
This is enforced by convention + a lint rule (see `pad/tests/test_types.py`).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Mapping, List, Optional
import time

# ── Attack taxonomy (ISO/IEC 30107-3 §6 "attack classes") ──
AttackClass = Literal[
    "PRINT",            # A: paper photo
    "CUTOUT",           # B: paper with eye-holes
    "REPLAY_SCREEN",    # C: phone/tablet/monitor replay
    "MASK_2D",          # D: partial silicone/latex
    "MASK_3D",          # E: full 3D mask
    "DEEPFAKE_STREAM",  # F: real-time face swap via virtual camera
    "ADV_PATCH",        # G: adversarial glasses / hat pattern.
                        #    API_unq 2026-05-13 (P2.6): no branch currently
                        #    emits this label. Placeholder for future patch-
                        #    detection branch per Liu et al. CVPR 2022
                        #    Segment-and-Complete (cf. Sharif et al. CCS 2016
                        #    eyeglass-frame attacks; Komkov & Petiushko ICPR
                        #    2021 AdvHat). Kept in the taxonomy so
                        #    BranchScore.dominant_class can carry the label
                        #    forward when that branch lands; tests in
                        #    pad/tests/test_fusion.py and test_wiring_v941.py
                        #    exercise the path via a synthetic stub.
    "AUDIO_DEEPFAKE",   # H: TTS / voice clone
    "REPLAY_LOOP",      # I: replay-to-replay loop
    "UNKNOWN",          # catchall
]
ATTACK_CLASSES: tuple = (
    "PRINT", "CUTOUT", "REPLAY_SCREEN",
    "MASK_2D", "MASK_3D",
    "DEEPFAKE_STREAM", "ADV_PATCH",
    "AUDIO_DEEPFAKE", "REPLAY_LOOP",
    "UNKNOWN",
)

VerdictLabel = Literal["LIVE", "SPOOF", "UNCERTAIN"]


@dataclass(slots=True)
class BranchScore:
    """Single PAD branch's opinion on one frame cluster.

    Contract:
      - `p_attack` in [0, 1] — calibrated probability the input is a spoof.
      - `reliability` in [0, 1] — how much weight this branch deserves
        right now (e.g., an rPPG branch has reliability=0 under heavy
        motion; an audio branch has reliability=0 during silence).
      - `latency_ms` — how long the branch took on this frame cluster.
      - `meta` — optional per-branch diagnostic floats for the HUD.

    v9.8.1 2026-05-12 (Audit H3): __post_init__ clamps p_attack and
    reliability into [0, 1] and replaces any NaN / +/-Inf with safe
    neutral values (p=0.5, rel=0.0). A branch that returns NaN due to
    a division-by-zero, log(0), or uncaught math error can no longer
    poison the fusion stage -- the contract is enforced at the type
    boundary. Same guard for `latency_ms` so a clock skew can't push
    a negative latency into Prometheus/OTel pipelines.
    """
    name: str
    p_attack: float
    reliability: float
    latency_ms: float = 0.0
    meta: Mapping[str, float] = field(default_factory=dict)
    # Per-branch preferred attack class (used only when this branch is
    # the dominant contributor to the fused verdict). Keep tight.
    dominant_class: Optional[AttackClass] = None

    def __post_init__(self):
        import math as _math
        # p_attack: must be a finite float in [0, 1]. Anything else
        # collapses to 0.5 (= "no opinion") + reliability=0 so this
        # branch's vote is silently dropped by fusion's emit floor.
        try:
            p = float(self.p_attack)
            if _math.isnan(p) or _math.isinf(p):
                p = 0.5
                # Anything that emitted NaN can't be trusted -- zero
                # out reliability so fusion ignores it.
                object.__setattr__(self, "reliability", 0.0)
        except (TypeError, ValueError):
            p = 0.5
            object.__setattr__(self, "reliability", 0.0)
        # Clip without further side-effects.
        if p < 0.0:
            p = 0.0
        elif p > 1.0:
            p = 1.0
        object.__setattr__(self, "p_attack", p)

        # reliability: same treatment.
        try:
            r = float(self.reliability)
            if _math.isnan(r) or _math.isinf(r):
                r = 0.0
        except (TypeError, ValueError):
            r = 0.0
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        object.__setattr__(self, "reliability", r)

        # latency_ms: non-negative finite float.
        try:
            lat = float(self.latency_ms)
            if _math.isnan(lat) or _math.isinf(lat) or lat < 0.0:
                lat = 0.0
        except (TypeError, ValueError):
            lat = 0.0
        object.__setattr__(self, "latency_ms", lat)


@dataclass(slots=True)
class PADScore:
    """Fused PAD verdict for one frame cluster, one person."""
    p_attack: float
    verdict: VerdictLabel
    attack_class: AttackClass
    branches: List[BranchScore]
    ts: float
    # Log-odds value before sigmoid — useful for debugging fusion math.
    log_odds: float = 0.0
    # v9.7.6 — fusion-level diagnostics. Populated by pad.fusion.fuse()
    # when the verdict is downgraded by a guard (e.g. the physical-
    # liveness gate that refuses confident LIVE without radar/rppg
    # confirmation). UI reads this to show DEGRADED warnings.
    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PADState:
    """Per-person bounded-history of PAD verdicts.

    Lives inside `PersonState.pad_state`. The bounded `history` ring
    supports short-window smoothing (e.g., require 3 consecutive SPOOF
    verdicts before surfacing the UI banner) without unbounded memory.
    """
    last: Optional[PADScore] = None
    history: List[PADScore] = field(default_factory=list)
    max_history: int = 120  # ~1 minute at 2 Hz fusion

    def push(self, score: PADScore) -> None:
        self.last = score
        self.history.append(score)
        if len(self.history) > self.max_history:
            # deque-like O(1) amortised pop from front
            del self.history[0]

    def recent_verdict_ratio(self, verdict: VerdictLabel,
                             window: int = 6) -> float:
        """Fraction of the last `window` verdicts matching `verdict`.
        Used by the UI hysteresis so a single spurious SPOOF frame
        doesn't flip the tile to red.
        """
        if not self.history:
            return 0.0
        tail = self.history[-window:]
        hits = sum(1 for s in tail if s.verdict == verdict)
        return hits / len(tail)

    def reset(self) -> None:
        self.last = None
        self.history.clear()


def utc_now() -> float:
    return time.time()
