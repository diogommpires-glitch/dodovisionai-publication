"""pad/branches/radar_presence_branch.py  --  Radar-based presence PAD branch.

Uses the MR60BHA2 60 GHz FMCW radar data already being streamed by
`radar.py` (COM5/COM6 ports). Real humans emit breathing + heart pulses
the radar picks up within ~2 m; photo/screen/paper attacks have none.

This branch:
  1. Reads recent radar state via `radar.get_radar_data()`.
  2. Emits p_attack = low if radar sees a live human (HR/BR fresh + plausible).
  3. Emits p_attack = high if radar is healthy but sees nothing.
  4. Emits reliability = 0 if radar is stale/offline (fusion ignores it).

Output is a `BranchScore(name="radar_presence", ...)` compatible with the
rest of the PAD fusion stack.

Key assumption: the radar reader is always running in the main event loop
and publishes to `radar._radar_data`. If the radar is not connected, this
branch degrades gracefully to reliability=0.
"""
from __future__ import annotations
import time
from typing import Optional

from pad.types import BranchScore
from pad.calibrate import apply_calibration


# Heuristic thresholds (tuned against MR60BHA2 typical sensitivity)
HR_PLAUSIBLE_MIN = 30    # bpm
HR_PLAUSIBLE_MAX = 180
BR_PLAUSIBLE_MIN = 4     # rpm
BR_PLAUSIBLE_MAX = 40
MAX_AGE_S = 5.0          # data older than this is stale

# API_unq P2.4: per-frame radar A/B picker debounce (2-of-3 vote).
_RADAR_CHOICE_HIST: list = []
_RADAR_LAST_COMMITTED: str = "A"


def score() -> BranchScore:
    """Evaluate presence from the current radar state snapshot."""
    t0 = time.monotonic()

    # v9.7.4 — read both Radar A (`radar.summary()`) and Radar B
    # (`radar.summary_b()`) and pick whichever has fresher vitals.
    # Earlier this branch relied on Radar A alone, but on dual-radar setups
    # Radar A often takes 30+ s to lock chest-wall HR/BR while Radar B
    # locks within a few seconds. Merging means the branch contributes
    # to fusion as soon as ANY radar is producing vitals — and stays
    # contributing if either radar holds the lock.
    try:
        from radar import summary as _radar_summary, summary_b as _radar_summary_b
    except Exception as e:
        return BranchScore(
            name="radar_presence", p_attack=0.5, reliability=0.0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            meta={"error": f"radar module unavailable: {e}"},
            dominant_class=None)

    try:
        d_a = _radar_summary()
    except Exception as e:
        d_a = None
    try:
        d_b = _radar_summary_b()
    except Exception as e:
        d_b = None

    def _quality_score(d):
        """Higher score = better radar to use. Reward connected,
        recent, and HR/BR-bearing data."""
        if d is None or not isinstance(d, dict):
            return -1.0
        if not d.get("connected"):
            return -0.5
        last_update = d.get("last_update", 0) or 0
        if not last_update:
            return -0.25
        age = time.time() - last_update
        if age > MAX_AGE_S:
            return -0.1
        score = 1.0 - (age / MAX_AGE_S)  # 1.0 at fresh, 0.0 at MAX_AGE_S
        hr = d.get("heart_rate")
        br = d.get("breath_rate")
        if hr is not None and HR_PLAUSIBLE_MIN <= float(hr) <= HR_PLAUSIBLE_MAX:
            score += 0.5
        if br is not None and BR_PLAUSIBLE_MIN <= float(br) <= BR_PLAUSIBLE_MAX:
            score += 0.5
        return score

    qa, qb = _quality_score(d_a), _quality_score(d_b)
    # API_unq 2026-05-13 (P2.4): radar A/B debounce.
    # Pre-API_unq this picked the higher-quality radar per frame. When
    # the two are close (both healthy, qa ≈ qb), the picker flips A/B
    # every 1-3 frames as the score-tie tips on transient noise, and
    # downstream consumers see jittered HR/BR/dist. Fix: 2-of-3 vote
    # over the last 3 frames before switching from A→B or B→A.
    #   _radar_choice_history: deque[str] last 3 choices ("A"/"B")
    #   per-frame raw_choice ← max quality
    #   committed_choice ← _radar_choice_history.most_common (≥2 of 3)
    #   if no majority (alternating ties), STICKY: keep the previous
    #   committed_choice rather than flip.
    global _RADAR_CHOICE_HIST, _RADAR_LAST_COMMITTED
    raw_choice = "A" if qa >= qb else "B"
    try:
        _RADAR_CHOICE_HIST.append(raw_choice)
        while len(_RADAR_CHOICE_HIST) > 3:
            _RADAR_CHOICE_HIST.pop(0)
    except NameError:
        _RADAR_CHOICE_HIST = [raw_choice]
        _RADAR_LAST_COMMITTED = raw_choice
    if "_RADAR_LAST_COMMITTED" not in globals():
        _RADAR_LAST_COMMITTED = raw_choice
    n_a = _RADAR_CHOICE_HIST.count("A")
    n_b = _RADAR_CHOICE_HIST.count("B")
    if n_a >= 2:
        committed = "A"
    elif n_b >= 2:
        committed = "B"
    else:
        # No 2-of-3 majority (only happens at startup with <3 frames);
        # stick with the previous commit.
        committed = _RADAR_LAST_COMMITTED
    _RADAR_LAST_COMMITTED = committed
    d = d_a if committed == "A" else d_b
    radar_used = committed

    if d is None or not isinstance(d, dict):
        return BranchScore(
            name="radar_presence", p_attack=0.5, reliability=0.0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            meta={"status": "no_data", "radar_used": "none",
                  "qa": qa, "qb": qb},
            dominant_class=None)

    # Use the real `summary()` field names. Note `last_update` is the
    # canonical freshness timestamp (set by the radar reader thread on
    # every parsed JSON line).
    hr = d.get("heart_rate")
    br = d.get("breath_rate")
    presence = d.get("presence")
    connected = bool(d.get("connected"))
    last_update = d.get("last_update", 0) or 0
    age = time.time() - last_update if last_update else float("inf")

    # If radar isn't connected at all (no port open), don't contribute.
    if not connected:
        return BranchScore(
            name="radar_presence", p_attack=0.5, reliability=0.0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            meta={"status": "offline", "connected": False,
                  "error": d.get("error")},
            dominant_class=None)

    # Stale -> radar offline, don't contribute
    if age > MAX_AGE_S or last_update == 0:
        return BranchScore(
            name="radar_presence", p_attack=0.5, reliability=0.0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            meta={"age_s": round(age, 2), "status": "stale",
                  "connected": connected, "last_update": last_update},
            dominant_class=None)

    # Reliability drops with age (exponential decay, per v9.0 ZOH policy)
    reliability = max(0.0, min(1.0, 1.0 - (age / MAX_AGE_S) * 0.5))

    # Evaluate liveness signal
    has_hr = hr is not None and HR_PLAUSIBLE_MIN <= float(hr) <= HR_PLAUSIBLE_MAX
    has_br = br is not None and BR_PLAUSIBLE_MIN <= float(br) <= BR_PLAUSIBLE_MAX
    # v9.7.4: also count distance/people from the radar — the MR60BHA2
    # reports `dist > 0.0` and `people_count >= 1` for several seconds
    # BEFORE it locks onto chest-wall motion to extract HR/BR. During
    # that lock-on window the previous logic was incorrectly emitting
    # `raw_p = 0.80` ("NON_HUMAN_PRESENT") with `rel = 0.99`, which then
    # dominated the fused PAD verdict and false-positive'd the live
    # face as SPOOF. Distance + people count is a strong indicator of
    # SOMETHING physically present (radar reflection from a body) even
    # before vitals lock; treat it as ambiguous-toward-live, not spoof.
    dist = d.get("distance")
    people = d.get("people_count")
    has_dist  = dist is not None and float(dist) > 0.05
    has_people = people is not None and int(people) >= 1
    sees_someone = bool(presence) or has_hr or has_br or has_dist or has_people

    # Raw p_attack: low if radar confirms live; high if radar is healthy but sees nothing
    if has_hr and has_br:
        raw_p = 0.05            # strong live signal — vitals locked
        reliability = min(1.0, reliability + 0.2)
        dominant = None
    elif has_hr or has_br:
        raw_p = 0.15            # partial vitals lock
        dominant = None
    elif has_dist or has_people:
        # Radar sees a physical body (reflection + people-count tracker
        # active) but vitals haven't locked yet. This is the typical
        # 5-30 s warmup window for the MR60BHA2. Treat as moderate-live
        # AND drop reliability so the branch doesn't dominate fusion
        # while we wait for HR/BR to come online.
        raw_p = 0.30
        reliability = max(0.0, reliability * 0.4)
        dominant = None
    elif sees_someone:
        raw_p = 0.35            # presence flag only — softer ambiguous
        reliability = max(0.0, reliability * 0.5)
        dominant = None
    else:
        # Radar healthy, sees absolutely nothing — only NOW is it a
        # strong spoof cue. Even so, halve reliability vs the previous
        # rel=0.99 because radar field-of-view doesn't always cover the
        # camera's view (off-axis subjects vanish from radar but stay
        # in camera frame).
        raw_p = 0.80
        reliability = max(0.0, reliability * 0.5)
        dominant = "NON_HUMAN_PRESENT"

    # Apply per-branch calibration if available
    p_attack = apply_calibration("radar_presence", raw_p)

    return BranchScore(
        name="radar_presence",
        p_attack=max(0.0, min(1.0, p_attack)),
        reliability=round(reliability, 3),
        latency_ms=(time.monotonic() - t0) * 1000.0,
        meta={
            "radar_used": radar_used,
            "hr": hr, "br": br, "presence": presence,
            "age_s": round(age, 2),
            "has_hr": has_hr, "has_br": has_br,
        },
        dominant_class=dominant,
    )
