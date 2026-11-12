"""screen_flash_protocol.py
=========================

API_unq active-liveness challenge state machine: random background color
pulse + skin-reflection delta verification.

Theory:
  Real-time deepfake renderers (DeepFaceLab / SimSwap real-time pipelines)
  have inherent ~150-400 ms generation latency. When the UI pulses the
  page background to a random color for ~80 ms, a *live* face's skin
  reflects the ambient light shift in CIELAB color space (Δa*/Δb* aligned
  to the flash hue). A *deepfake-injected* stream is generated upstream
  of the screen and shows ZERO ambient response. A *screen replay* (phone
  / tablet displaying a face) either shows no response (display ignores
  ambient) or a wrong-direction response (reflects its own emitted light,
  not the page flash).

Protocol (per-person, per-challenge):
  1. UI requests challenge: POST /pad/challenge/screen_flash/start?pid=N
  2. Server picks random color from {RED, GREEN, BLUE, WHITE} + t_start
  3. Server returns {challenge_id, color_hex, expected_dir, duration_ms}
  4. UI pulses document.body.style.backgroundColor for duration_ms
  5. WS frames continue uninterrupted. Branch reads PersonState rPPG
     forehead/cheek RGB buffers, samples ROI Lab values at:
       - pre  : [t_start - 200ms, t_start)
       - peak : [t_start + 30ms, t_start + duration_ms)
       - post : [t_start + duration_ms + 50ms, t_start + 500ms]
  6. Branch computes Δa*, Δb* deltas at peak vs baseline.
  7. Verdict:
       - Δ aligned with expected_dir AND |Δ| ≥ JND_THRESHOLD: LIVE
       - |Δ| < NOISE_FLOOR (no ambient response): SPOOF (likely deepfake)
       - Δ in wrong direction at high magnitude: SPOOF (likely screen replay)

The protocol module is stateless from the branch's POV; the branch reads
this module's `active_challenge(pid)` to know whether to score.

Off-the-shelf, requires NO retraining — calibrator can be fit either
from synthetic anchors (5 live deltas, 5 spoof zeros) or from a small
operator-collected set.
"""
from __future__ import annotations

import threading
import time
import secrets
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# ---- Constants ----------------------------------------------------------
# CIELAB skin JND (just-noticeable difference). 2.3 is the published average
# but skin tone interaction shifts this in practice — we use 2.0 as the
# detection floor (Sharma 2003, Luo et al. CIEDE2000).
JND_THRESHOLD = 2.0
# Below this magnitude we declare "no response" (likely deepfake injection
# or matte print absorbing the ambient).
NOISE_FLOOR = 0.7
# Default flash duration. Long enough to register across 2-3 frames at 30 fps,
# short enough that a slow real-time deepfake misses it entirely.
DEFAULT_FLASH_MS = 120

# Available challenge colors. Each entry: (hex, expected Δ direction signed)
#   expected_dir is a (dL, da, db) tuple of signs in {-1, 0, +1}.
#   0 = "don't care" (we don't gate on that channel).
COLORS = {
    "RED":   ("#FF0000", ( 0, +1,  0)),
    "GREEN": ("#00FF00", (+1, -1,  0)),
    "BLUE":  ("#0000FF", ( 0,  0, -1)),
    "WHITE": ("#FFFFFF", (+1,  0,  0)),
}


@dataclass
class ScreenFlashChallenge:
    challenge_id: str
    pid: int
    color_name: str
    color_hex: str
    expected_dir: Tuple[int, int, int]
    t_start: float            # monotonic seconds when flash begins
    duration_ms: int
    t_observe_end: float      # latest time the branch should sample
    consumed: bool = False    # True once branch has scored this challenge

    @property
    def t_peak_start(self) -> float:
        return self.t_start + 0.030     # +30 ms warm-up

    @property
    def t_peak_end(self) -> float:
        return self.t_start + self.duration_ms / 1000.0

    @property
    def t_baseline_start(self) -> float:
        return self.t_start - 0.200

    @property
    def t_baseline_end(self) -> float:
        return self.t_start


# Per-pid active challenge registry.
_LOCK = threading.RLock()
_ACTIVE: Dict[int, ScreenFlashChallenge] = {}
# Bounded history of completed challenges per pid for the calibrator path.
_HISTORY: Dict[int, list] = {}
_HISTORY_CAP = 32


def start_challenge(pid: int, duration_ms: int = DEFAULT_FLASH_MS,
                    color_name: Optional[str] = None) -> ScreenFlashChallenge:
    """Begin a fresh screen-flash challenge for `pid`. Overwrites any prior
    in-flight challenge. Returns the challenge state (also stored in the
    module registry so the branch can read it on subsequent frames).
    """
    import random as _r
    if color_name is None:
        color_name = _r.choice(list(COLORS.keys()))
    hex_, exp = COLORS[color_name]
    ch_id = secrets.token_hex(6)
    t0 = time.monotonic()
    obs_end = t0 + 0.500  # 500 ms observation window total
    challenge = ScreenFlashChallenge(
        challenge_id=ch_id,
        pid=int(pid),
        color_name=color_name,
        color_hex=hex_,
        expected_dir=exp,
        t_start=t0,
        duration_ms=int(duration_ms),
        t_observe_end=obs_end,
    )
    with _LOCK:
        _ACTIVE[int(pid)] = challenge
    return challenge


def active_challenge(pid: int) -> Optional[ScreenFlashChallenge]:
    """Return the in-flight challenge for `pid`, or None if none / expired."""
    with _LOCK:
        ch = _ACTIVE.get(int(pid))
        if ch is None:
            return None
        if time.monotonic() > ch.t_observe_end:
            # Expired — branch missed the window. Move to history as a fail.
            _ACTIVE.pop(int(pid), None)
            _push_history(int(pid), ch, p_attack=0.5, reason="expired_window")
            return None
        return ch


def consume_challenge(pid: int, p_attack: float, reason: str) -> None:
    """Mark a challenge as scored. Branch calls this after producing a verdict."""
    with _LOCK:
        ch = _ACTIVE.get(int(pid))
        if ch is None or ch.consumed:
            return
        ch.consumed = True
        _ACTIVE.pop(int(pid), None)
        _push_history(int(pid), ch, p_attack, reason)


def _push_history(pid: int, ch: ScreenFlashChallenge, p_attack: float, reason: str) -> None:
    hist = _HISTORY.setdefault(pid, [])
    hist.append({
        "challenge_id": ch.challenge_id,
        "color_name": ch.color_name,
        "p_attack": float(p_attack),
        "reason": reason,
        "t_start": ch.t_start,
        "duration_ms": ch.duration_ms,
    })
    if len(hist) > _HISTORY_CAP:
        del hist[: len(hist) - _HISTORY_CAP]


def get_history(pid: int) -> list:
    with _LOCK:
        return list(_HISTORY.get(int(pid), []))


def reset(pid: Optional[int] = None) -> None:
    """Clear in-flight + history for `pid`, or for all pids if None."""
    with _LOCK:
        if pid is None:
            _ACTIVE.clear()
            _HISTORY.clear()
        else:
            _ACTIVE.pop(int(pid), None)
            _HISTORY.pop(int(pid), None)
