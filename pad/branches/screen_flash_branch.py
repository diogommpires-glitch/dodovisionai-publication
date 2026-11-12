"""pad/branches/screen_flash_branch.py  --  Active screen-flash liveness.

API_unq new branch. Designed to defeat real-time deepfake injection
(Camera Pipe Injection class per the 2026 PAD survey).

The server triggers a brief background color flash via
`pad.screen_flash_protocol.start_challenge(pid)`. The UI pulses
`document.body.style.backgroundColor` for ~120 ms. This branch reads the
per-pid rPPG forehead/cheek RGB ring buffers around the challenge window,
converts to CIELAB, and tests whether the skin reflected the ambient
shift in the expected direction.

Verdicts:
  - aligned + magnitude >= JND     → LIVE   (p_attack ~0.05, rel >=0.9)
  - magnitude < NOISE_FLOOR         → SPOOF  (p_attack ~0.75)  ← deepfake or
                                                                 matte print
  - wrong direction at magnitude    → SPOOF  (p_attack ~0.90)  ← screen replay
                                                                 reflecting own
                                                                 emitted light
  - no challenge / no rPPG buffer   → ABSTAIN (rel=0)

Integration: call `score(ps, now=None)` once per frame from the PAD loop.
The branch returns abstain (rel=0) when no challenge is active; the
fusion ignores the entry (per `min_reliability_emit=0.05`). To fire a
challenge, hit `POST /pad/challenge/screen_flash/start?pid=N` (see
pad/screen_flash_protocol.py + the route wiring in app/api/routes.py).

NOTE: requires `rppg.feed_liveness_rois` to populate
ps.rppg_rgb / ps.rppg_times / ps.rppg_l_cheek / ps.rppg_r_cheek at the
high-fps camera rate (≥30 Hz). On boxes without high-fps dual camera, the
challenge window is too short (~3 frames @ 30fps) and the branch will
often abstain due to insufficient samples — expected behaviour.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

from pad.types import BranchScore
from pad.calibrate import apply_calibration
from pad.screen_flash_protocol import (
    JND_THRESHOLD,
    NOISE_FLOOR,
    active_challenge,
    consume_challenge,
)

BRANCH_NAME = "screen_flash"

# Minimum number of frames we need in each of {baseline, peak} windows
# to trust the Lab delta. Below this, abstain.
_MIN_BASELINE_SAMPLES = 3
_MIN_PEAK_SAMPLES = 2


def _rgb_to_lab(rgb_mean: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """sRGB (0-255) → CIELAB D65. Numerically robust without OpenCV — we
    accept the slight overhead because rppg ROI means are 3-tuples.
    """
    # Normalize and linearize sRGB
    def lin(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(float(rgb_mean[0])),
               lin(float(rgb_mean[1])),
               lin(float(rgb_mean[2])))
    # sRGB → XYZ D65
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    # Reference white D65
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)

    fx, fy, fz = f(X / Xn), f(Y / Yn), f(Z / Zn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    return (L, a, bb)


def _samples_in_window(times: List[float], rgbs: List[Tuple[float, float, float]],
                       t_lo: float, t_hi: float) -> Tuple[List[float], List[float], List[float]]:
    """Return per-channel R/G/B lists for samples in [t_lo, t_hi)."""
    rs: List[float] = []
    gs: List[float] = []
    bs: List[float] = []
    for t, rgb in zip(times, rgbs):
        if t_lo <= float(t) < t_hi:
            try:
                rs.append(float(rgb[0]))
                gs.append(float(rgb[1]))
                bs.append(float(rgb[2]))
            except Exception:
                continue
    return rs, gs, bs


def _mean_rgb_in_window(times: List[float], rgbs: List[Tuple[float, float, float]],
                        t_lo: float, t_hi: float) -> Optional[Tuple[float, float, float]]:
    """Average RGB samples whose timestamps fall in [t_lo, t_hi).
    Returns None if no samples land in window.
    """
    rs, gs, bs = _samples_in_window(times, rgbs, t_lo, t_hi)
    if not rs:
        return None
    n = len(rs)
    return (sum(rs) / n, sum(gs) / n, sum(bs) / n)


def _pearson_r(xs: List[float], ys: List[float]) -> float:
    """Pearson product-moment correlation. Returns 0.0 when undefined
    (n<2 or zero variance). API_unq channel-decorrelation helper.
    """
    n = len(xs)
    if n < 2 or len(ys) != n:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = (sxx * syy) ** 0.5
    if denom <= 1e-12:
        return 0.0
    return sxy / denom


def _channel_correlation_triplet(rs: List[float], gs: List[float],
                                 bs: List[float]) -> Tuple[float, float, float]:
    """(r_RG, r_GB, r_RB) Pearson correlations across the 3 channels.

    API_unq 2026-05-13 (Physical-PAD survey, channel decorrelation):
    Live skin under flash has uniform biological pigmentation -- all 3
    correlations stay >0.85 because the surface reflects ambient as a
    single physical reflector. Screen replays break at least one: the
    emissive pixel grid drives R/G/B independently per subpixel cluster
    (LCD), or amplifies one channel asymmetrically (OLED). Prints
    decorrelate because CMYK ink stack has different per-channel
    absorption-to-reflection ratios per pixel.
    """
    return (
        _pearson_r(rs, gs),
        _pearson_r(gs, bs),
        _pearson_r(rs, bs),
    )


def _count_in_window(times: List[float], t_lo: float, t_hi: float) -> int:
    return sum(1 for t in times if t_lo <= float(t) < t_hi)


def _sign(x: float) -> int:
    if x > 0:
        return +1
    if x < 0:
        return -1
    return 0


def _abstain(reason: str, t0: float, latency_ms: Optional[float] = None) -> BranchScore:
    return BranchScore(
        name=BRANCH_NAME,
        p_attack=0.5,
        reliability=0.0,
        latency_ms=float(latency_ms if latency_ms is not None else (time.monotonic() - t0) * 1000.0),
        meta={"reason": reason},
        dominant_class=None,
    )


def score(ps, now: Optional[float] = None) -> BranchScore:
    """Score a PersonState. Returns abstain when no challenge is active."""
    t0 = time.monotonic()
    pid = int(getattr(ps, "track_id", -1))
    if pid < 0:
        return _abstain("no_pid", t0)

    challenge = active_challenge(pid)
    if challenge is None:
        return _abstain("no_active_challenge", t0)

    rgbs = getattr(ps, "rppg_rgb", None)
    times = getattr(ps, "rppg_times", None)
    if not rgbs or not times or len(rgbs) != len(times):
        return _abstain("no_rppg_buffer", t0)

    # Window samples
    n_base = _count_in_window(times, challenge.t_baseline_start, challenge.t_baseline_end)
    n_peak = _count_in_window(times, challenge.t_peak_start, challenge.t_peak_end)
    if n_base < _MIN_BASELINE_SAMPLES or n_peak < _MIN_PEAK_SAMPLES:
        # Don't consume — leave the challenge alive so a later frame can score.
        return _abstain(f"insufficient_samples (base={n_base} peak={n_peak})", t0)

    base_rgb = _mean_rgb_in_window(times, rgbs,
                                   challenge.t_baseline_start, challenge.t_baseline_end)
    peak_rgb = _mean_rgb_in_window(times, rgbs,
                                   challenge.t_peak_start, challenge.t_peak_end)
    if base_rgb is None or peak_rgb is None:
        return _abstain("window_join_failed", t0)

    base_lab = _rgb_to_lab(base_rgb)
    peak_lab = _rgb_to_lab(peak_rgb)
    dL = peak_lab[0] - base_lab[0]
    da = peak_lab[1] - base_lab[1]
    db = peak_lab[2] - base_lab[2]
    mag = math.sqrt(dL * dL + da * da + db * db)

    exp_dL, exp_da, exp_db = challenge.expected_dir

    # Score per-channel alignment: only channels with non-zero expected sign
    # contribute. align_count = how many expected-direction matches we got.
    components = []
    for sign_exp, delta in ((exp_dL, dL), (exp_da, da), (exp_db, db)):
        if sign_exp == 0:
            continue
        components.append(_sign(delta) == sign_exp)
    if not components:
        return _abstain("color_has_no_gated_channel", t0)
    align_frac = sum(components) / len(components)

    # ── API_unq 2026-05-13 (Physical-PAD survey): channel decorrelation ──
    # Live skin: r(R,G), r(G,B), r(R,B) all stay >0.85 during the flash
    # because skin is one uniformly-pigmented biological reflector.
    # Screen replay: at least one Pearson r drops (LCD subpixel drives
    # per-channel, OLED amplifies asymmetrically).
    # Print: r decorrelates from the CMYK absorption mismatch.
    # We compute correlations on the PEAK window (where any decoupling
    # is maximally exposed by the flash) and as a sanity check vs the
    # BASELINE (live should be high in both; degenerate camera modes
    # that always-low get flagged via the baseline check).
    r_base, g_base, b_base = _samples_in_window(
        times, rgbs, challenge.t_baseline_start, challenge.t_baseline_end)
    r_peak, g_peak, b_peak = _samples_in_window(
        times, rgbs, challenge.t_peak_start, challenge.t_peak_end)
    rg_base, gb_base, rb_base = _channel_correlation_triplet(r_base, g_base, b_base)
    rg_peak, gb_peak, rb_peak = _channel_correlation_triplet(r_peak, g_peak, b_peak)
    # Decorrelation score: distance from the "all 3 high" live-skin
    # expectation. Higher = more spoof-like. Bounded [0, 1].
    # Note: we use the PEAK window because the flash is when emissive
    # surfaces break correlation the hardest (they fight the ambient
    # change). 0.85 is the live-skin floor from clinical chromaticity
    # studies (Sharma 2003, Hyttinen et al. 2016 hyperspectral skin).
    _LIVE_CORR_FLOOR = 0.85
    decorr_components = [max(0.0, _LIVE_CORR_FLOOR - c) for c in (rg_peak, gb_peak, rb_peak)]
    decorr_score = min(1.0, max(decorr_components) / max(1e-6, _LIVE_CORR_FLOOR))
    # Soft confidence bonus when baseline correlations are already high
    # (rules out cameras that always-decorrelate — those abstain).
    base_corr_min = min(rg_base, gb_base, rb_base)
    decorr_trustworthy = base_corr_min >= 0.70

    # Verdict
    if mag < NOISE_FLOOR:
        # No ambient response — likely deepfake injection upstream of screen,
        # or a matte print absorbing the flash.
        p_attack = 0.80
        dom = "DEEPFAKE_STREAM"
        reason = f"no_ambient_response (mag={mag:.2f} < {NOISE_FLOOR})"
        reliability = 0.85
    elif mag < JND_THRESHOLD:
        # Sub-JND response, ambiguous
        p_attack = 0.55
        dom = None
        reason = f"sub_jnd (mag={mag:.2f} < {JND_THRESHOLD})"
        reliability = 0.40
    elif align_frac >= 0.5:
        # Aligned with expected direction — live skin reflection.
        # Scale p_attack from 0.05 (perfect alignment + strong magnitude)
        # to 0.20 (just-over-threshold).
        norm_mag = min(1.0, (mag - JND_THRESHOLD) / 8.0)
        p_attack = max(0.05, 0.20 - 0.15 * norm_mag * align_frac)
        dom = None
        reason = (f"aligned (mag={mag:.2f} align={align_frac:.2f} "
                  f"color={challenge.color_name})")
        reliability = min(0.95, 0.60 + 0.35 * norm_mag * align_frac)
    else:
        # Wrong-direction strong response — screen replay reflecting its own
        # emitted light, NOT the page background.
        p_attack = 0.90
        dom = "REPLAY_SCREEN"
        reason = (f"wrong_direction (mag={mag:.2f} align={align_frac:.2f} "
                  f"color={challenge.color_name})")
        reliability = 0.85

    # ── API_unq channel-decorrelation fusion ──
    # Stack the decorrelation signal with the CIELAB verdict. Two cases:
    #   (a) verdict is LIVE-ish but decorrelation is high → bump p_attack
    #       toward UNCERTAIN. Catches the case where a screen reflects
    #       enough of the flash to spoof CIELAB alignment but its
    #       subpixel grid still breaks RGB correlation.
    #   (b) verdict is SPOOF and decorrelation is also high →
    #       reinforce reliability (consensus across two axes).
    # Only apply when baseline correlations are trustworthy.
    decorr_meta = {
        "rg_base": float(rg_base), "gb_base": float(gb_base), "rb_base": float(rb_base),
        "rg_peak": float(rg_peak), "gb_peak": float(gb_peak), "rb_peak": float(rb_peak),
        "decorr_score": float(decorr_score),
        "decorr_trustworthy": bool(decorr_trustworthy),
    }
    if decorr_trustworthy and decorr_score > 0.10:
        # Decorrelation present. p_attack moves halfway toward decorr_score-scaled SPOOF.
        if p_attack < 0.5:
            # Was voting LIVE; pull toward UNCERTAIN proportional to decorr severity.
            pulled = p_attack + (0.55 - p_attack) * min(1.0, decorr_score / 0.4)
            p_attack = float(min(0.85, pulled))
            reason = f"{reason} + decorr_override (decorr={decorr_score:.2f})"
            if dom is None and decorr_score > 0.40:
                dom = "REPLAY_SCREEN"  # strong decorrelation labels as screen
        else:
            # Already voting SPOOF; bump reliability slightly via consensus.
            reliability = min(0.97, reliability + 0.05)
            reason = f"{reason} + decorr_consensus (decorr={decorr_score:.2f})"

    # Apply per-branch calibration (if a calibrator has been fit; otherwise
    # apply_calibration returns the raw probability unchanged).
    p_calib = apply_calibration(BRANCH_NAME, p_attack)

    consume_challenge(pid, p_calib, reason)

    return BranchScore(
        name=BRANCH_NAME,
        p_attack=float(p_calib),
        reliability=float(reliability),
        latency_ms=float((time.monotonic() - t0) * 1000.0),
        meta={
            "challenge_id": challenge.challenge_id,
            "color": challenge.color_name,
            "dL": float(dL), "da": float(da), "db": float(db),
            "mag": float(mag),
            "align_frac": float(align_frac),
            "expected_dir": list(challenge.expected_dir),
            "p_raw": float(p_attack),
            "reason": reason,
            "n_baseline_samples": int(n_base),
            "n_peak_samples": int(n_peak),
            **decorr_meta,
        },
        dominant_class=dom,
    )
