"""pad/branches/blink_gaze_branch.py  --  Temporal liveness from blink + gaze.

Real humans blink 12-20 times per minute (mean ~15 bpm, SD ~5 bpm) with
saccadic gaze micro-movements (0.5-3 deg amplitude). Static photo attacks
never blink. Deepfake streams often have:
  - unnatural blink regularity (too periodic, or absent)
  - over-stable gaze (too-steady iris position, no micro-saccades)
  - fast/slow eyelid closure dynamics

This branch scores:
  p_attack = f(blink_rate_deviation, gaze_stability_z, eye_asymmetry)

Inputs (from PersonState or equivalent):
  - blink_history: list of (timestamp, bool_is_blink) recent events
  - gaze_history:  list of (t, gaze_x, gaze_y) iris-centroid in px or deg
  - session_duration_s: how long we've been observing this face

Integration: call `score(ps)` from main.py's PAD loop where `ps` is the
PersonState instance. Requires at least 10 s of history for stable output;
emits reliability=0 until that's reached.
"""
from __future__ import annotations
import time
from collections import deque
from typing import Deque, Optional, Tuple

from pad.types import BranchScore
from pad.calibrate import apply_calibration


MIN_WARMUP_S = 10.0         # need at least 10 s before emitting a verdict
MIN_BLINK_EVENTS = 3
HEALTHY_BPM_MEAN = 15.0
HEALTHY_BPM_SD = 5.0

# v9.8.1 2026-05-12 (9th-pass deep audit): gates for the strong-spoof
# NO_BLINK label. The branch was firing `score_blink=0.90 + NO_BLINK`
# on real subjects between t=10s (post-warmup) and ~t=30s because EAR-
# landmark detection through glasses delays the first server-side
# blink event. Raise the bar so NO_BLINK requires BOTH a sustained
# observation window AND a confirmed absence of blink events. Anything
# shorter / fewer events stays in the "weak suspicion" mid-range
# instead of pushing fusion toward SPOOF.
NO_BLINK_MIN_DURATION_S = 30.0   # require 30 s observation (was: any >=10s)
NO_BLINK_MIN_EVENTS_OBSERVED = 0  # blinks must be EXACTLY 0 over 30s
                                  # (sleep mode + 1 blink no longer triggers)


def _blink_rate_bpm(blinks: Deque, duration_s: float) -> Optional[float]:
    """Count blink-true events over duration; return bpm."""
    if duration_s <= 0:
        return None
    n_blinks = sum(1 for (_, is_blink) in blinks if bool(is_blink))
    return n_blinks * 60.0 / duration_s


def _gaze_stability(gaze: Deque) -> Optional[float]:
    """Standard deviation of gaze (x,y) position over the buffer.
    Real humans: small but non-zero SD (~2-5 px for iris in a 640 px frame).
    Static photo: SD = 0 (iris doesn't move).
    Too-stable deepfake: SD also very small.
    """
    if len(gaze) < 5:
        return None
    xs = [g[1] for g in gaze]
    ys = [g[2] for g in gaze]
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    var = sum((x - mx) ** 2 + (y - my) ** 2 for x, y in zip(xs, ys)) / len(xs)
    return var ** 0.5


def score(ps, now: Optional[float] = None) -> BranchScore:
    """Score a PersonState instance with blink_history + gaze_history deques.

    PersonState is duck-typed: we just need
      ps.blink_history : Deque[Tuple[float, bool]]
      ps.gaze_history  : Deque[Tuple[float, float, float]]
      ps.session_start : float (monotonic)  -- optional
    """
    t0 = time.monotonic()
    now = now if now is not None else time.time()

    blinks = getattr(ps, "blink_history", None)
    gaze = getattr(ps, "gaze_history", None)
    session_start = getattr(ps, "session_start", None)
    duration = (now - session_start) if session_start else None

    # Warmup: not enough observation yet -> abstain
    if blinks is None or gaze is None or duration is None or duration < MIN_WARMUP_S:
        return BranchScore(
            name="blink_gaze", p_attack=0.5, reliability=0.0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            meta={"status": "warmup",
                  "duration_s": round(duration, 2) if duration else None},
            dominant_class=None)

    bpm = _blink_rate_bpm(blinks, duration)
    gaze_std = _gaze_stability(gaze)

    # Score components -- each in [0, 1]
    score_blink = 0.5
    score_gaze = 0.5
    dom = None

    # v9.8.1 2026-05-12 (9th-pass deep audit): count blinks BEFORE the
    # scoring branch so we can guard NO_BLINK with the strict observation
    # window. blinks_count_now is the same calculation reliability uses
    # below, hoisted up here so both can read it.
    blinks_count_now = sum(1 for (_, b) in blinks if b) if blinks else 0

    if bpm is not None:
        # z-score vs healthy distribution
        z_bpm = abs(bpm - HEALTHY_BPM_MEAN) / HEALTHY_BPM_SD
        if bpm < 2.0:
            # Candidate for the strong NO_BLINK label, BUT gate it
            # behind:
            #   (a) at least 30 s of observation (not 10 s -- glasses-
            #       induced EAR jitter routinely delays the first
            #       server-side detected blink to t=20-30s on real
            #       subjects)
            #   (b) literally zero blink events observed (any single
            #       observed blink invalidates the photo-attack
            #       hypothesis -- a photo cannot blink even once)
            if (duration >= NO_BLINK_MIN_DURATION_S
                    and blinks_count_now <= NO_BLINK_MIN_EVENTS_OBSERVED):
                score_blink = 0.90
                dom = "NO_BLINK"
            else:
                # Sub-threshold "low blink rate" -- weak suspicion only,
                # let other branches drive the verdict.
                score_blink = 0.55
        elif z_bpm > 2.5:
            # Outside 2.5 SD -- suspicious
            score_blink = 0.70
        elif z_bpm > 1.5:
            score_blink = 0.55
        else:
            score_blink = 0.15       # healthy

    if gaze_std is not None:
        if gaze_std < 0.5:
            # Eye is frozen -- likely photo / perfect deepfake
            score_gaze = 0.85
            if dom is None:
                dom = "FROZEN_GAZE"
        elif gaze_std < 1.5:
            score_gaze = 0.55
        else:
            score_gaze = 0.15        # healthy micro-saccades

    # Weighted mix: blink is the stronger liveness cue
    raw_p = 0.65 * score_blink + 0.35 * score_gaze

    # Reliability: higher with longer observation + more blink events
    # (n_blink_events already computed above as blinks_count_now)
    n_blink_events = blinks_count_now
    reliability = min(1.0, 0.2 + 0.02 * min(duration, 30) + 0.1 * min(n_blink_events, 5))

    p_attack = apply_calibration("blink_gaze", raw_p)

    return BranchScore(
        name="blink_gaze",
        p_attack=max(0.0, min(1.0, p_attack)),
        reliability=round(reliability, 3),
        latency_ms=(time.monotonic() - t0) * 1000.0,
        meta={
            "bpm": round(bpm, 2) if bpm is not None else None,
            "gaze_std": round(gaze_std, 3) if gaze_std is not None else None,
            "duration_s": round(duration, 1),
            "n_blinks": n_blink_events,
            "score_blink": round(score_blink, 3),
            "score_gaze": round(score_gaze, 3),
        },
        dominant_class=dom,
    )
