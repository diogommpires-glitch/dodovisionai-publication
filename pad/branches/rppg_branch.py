"""
DODO VISION AI v9.4 PAD Step 2 — rPPG liveness branch.

v9.4.1: pure scorer. All per-person ROI buffers now live on
`PersonState.rppg_rgb` / `rppg_l_cheek` / `rppg_r_cheek`. This branch
only normalizes the three raw outputs of `rppg.extract_liveness(ps)`
into a `BranchScore`. The duplicate `_STATE` dict has been removed.

Three sub-scores are combined:

  a. pulse_snr     — FFT power in [0.75, 3.0] Hz vs total (live > noise)
  b. phase_coherence — Welch coherence at HR ± 0.25 Hz across forehead /
                       left cheek / right cheek ROIs.
  c. refresh_band  — power in 50/60/120 Hz bands vs total. Replay
                     attacks leak display-refresh harmonics.

Output mapping (calibrated via isotonic regression; file written to
`pad/models/rppg_iso.json` once a labelled set is available):

  p_attack = 1 - ( 0.55*norm(pulse_snr) + 0.30*norm(phase_coh) + 0.15*(1-norm(refresh)) )

Citations:
  - Liu, Lagorio, Marasco et al., *Remote Blood Pulse Analysis for
    Face Presentation Attack Detection*, Springer 2018.
  - Lin et al., *Face Liveness Detection by rPPG Features and
    Contextual Patch-Based CNN*, ICBEA 2019.
  - MDPI Sensors 22(8) 2022 — Face Biometric Spoof Detection via rPPG.
  - OR-PAD, IJCV 2025.
"""
from __future__ import annotations
import math
import time
import json
import os
from typing import Optional, List, Tuple, Dict

import numpy as np

from pad.types import BranchScore

# ── Module paths ──
_MOD_DIR = os.path.dirname(os.path.abspath(__file__))
_ISOTONIC_PATH = os.path.join(
    os.path.dirname(_MOD_DIR), "models", "rppg_iso.json")

# ── Calibrated reference ranges (used when no isotonic file exists) ──
_LIVE_SNR_FLOOR = 0.10
_LIVE_SNR_CEIL = 0.40
_LIVE_COH_FLOOR = 0.30
_LIVE_COH_CEIL = 0.75
_REPLAY_REFRESH_TRIGGER = 0.02


def _get_person_state(pid: int):
    """Resolve the PersonState for a pid. Returns None if unavailable.
    Uses the global `person_states` dict exposed by `main.py`. Tests
    monkey-patch via `pad.branches.rppg_branch._person_states_lookup`.
    """
    fn = globals().get("_person_states_lookup")
    if fn is not None:
        try:
            return fn(pid)
        except Exception:
            return None
    try:
        import main as _m
        return _m.person_states.get(pid)
    except Exception:
        return None


def reset_all() -> None:
    """Clear any cached isotonic calibrator. Per-person state lives on
    PersonState so tests construct fresh ones rather than reset here."""
    global _iso_cache
    _iso_cache = None


def feed_rois(pid: int, ts: float,
              forehead_rgb: Optional[Tuple[float, float, float]],
              l_cheek_rgb: Optional[Tuple[float, float, float]],
              r_cheek_rgb: Optional[Tuple[float, float, float]],
              fs_hint: Optional[float] = None) -> None:
    """Thin pass-through to `rppg.feed_liveness_rois` for the
    PersonState associated with `pid`. Tests may also call this directly
    with a monkey-patched `_person_states_lookup`."""
    ps = _get_person_state(pid)
    if ps is None:
        return
    try:
        import rppg as _rppg
    except Exception:
        return
    _rppg.feed_liveness_rois(ps, ts, forehead_rgb, l_cheek_rgb,
                             r_cheek_rgb, fs_hint=fs_hint)


def _coherence_uniformity_from_cheeks(ps) -> float:
    """Pairwise phase-coherence variance across the three ROI pairs is
    computed inside `rppg.extract_liveness` via the mean; we compute a
    second pass here for the "too uniform" anomaly. Returns 0 when we
    can't evaluate. This is cheap — the POS signals are tiny.
    """
    try:
        import rppg as _rppg
    except Exception:
        return 0.0
    sig_fh = _rppg._pos_signal_from_forehead(ps)
    lc_list = list(getattr(ps, "rppg_l_cheek", []))
    rc_list = list(getattr(ps, "rppg_r_cheek", []))
    sig_lc = _rppg._pos_signal_from_tuples(lc_list) if len(lc_list) >= 20 else np.zeros(0)
    sig_rc = _rppg._pos_signal_from_tuples(rc_list) if len(rc_list) >= 20 else np.zeros(0)
    fs = float(getattr(ps, "rppg_fs_est", 20.0) or 20.0)
    coh_vals: List[float] = []
    if sig_fh.size and sig_lc.size:
        coh_vals.append(_rppg._phase_coherence(sig_fh, sig_lc, fs))
    if sig_fh.size and sig_rc.size:
        coh_vals.append(_rppg._phase_coherence(sig_fh, sig_rc, fs))
    if sig_lc.size and sig_rc.size:
        coh_vals.append(_rppg._phase_coherence(sig_lc, sig_rc, fs))
    if len(coh_vals) < 2:
        return 0.0
    var = float(np.var(np.asarray(coh_vals, dtype=float)))
    return max(0.0, 1.0 - min(1.0, var / 0.05))


def _norm(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


# ── Optional isotonic calibrator (loaded once, cached) ──
_iso_cache: Optional[Dict[str, List[Tuple[float, float]]]] = None


def _isotonic_apply(feature: str, x: float) -> float:
    global _iso_cache
    if _iso_cache is None:
        if os.path.exists(_ISOTONIC_PATH):
            try:
                with open(_ISOTONIC_PATH, "r", encoding="utf-8") as fh:
                    _iso_cache = json.load(fh)
            except Exception:
                _iso_cache = {}
        else:
            _iso_cache = {}
    points = _iso_cache.get(feature) if _iso_cache else None
    if not points:
        return x
    pts = sorted(points, key=lambda p: p[0])
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for i in range(1, len(pts)):
        if x <= pts[i][0]:
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            t = (x - x0) / max(x1 - x0, 1e-9)
            return y0 + t * (y1 - y0)
    return x


def score(pid: int, hr_hint_bpm: Optional[float] = None) -> BranchScore:
    """Produce the rPPG liveness BranchScore for one person.

    Pulls the three-output liveness signal from `rppg.extract_liveness`
    using the PersonState rppg_rgb + cheek buffers. Returns rel=0 and
    p_attack=0.5 (non-committal) until enough samples accumulate.
    """
    t0 = time.monotonic()
    ps = _get_person_state(pid)
    if ps is None:
        return BranchScore(
            name="rppg", p_attack=0.5, reliability=0.0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            meta={"n_rois": 0.0},
        )

    n_ok = 0
    if len(getattr(ps, "rppg_rgb", [])) >= 30:
        n_ok += 1
    if len(getattr(ps, "rppg_l_cheek", [])) >= 30:
        n_ok += 1
    if len(getattr(ps, "rppg_r_cheek", [])) >= 30:
        n_ok += 1

    if n_ok == 0:
        return BranchScore(
            name="rppg", p_attack=0.5, reliability=0.0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            meta={"n_rois": 0.0},
        )

    try:
        import rppg as _rppg
    except Exception:
        return BranchScore(
            name="rppg", p_attack=0.5, reliability=0.0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            meta={"n_rois": float(n_ok)},
        )

    pulse_snr, phase_coh, refresh_ratio = _rppg.extract_liveness(
        ps, hr_hint_bpm=hr_hint_bpm)
    coh_uniform = _coherence_uniformity_from_cheeks(ps)
    fs = float(getattr(ps, "rppg_fs_est", 20.0) or 20.0)

    live_snr = _norm(_isotonic_apply("snr", pulse_snr),
                     _LIVE_SNR_FLOOR, _LIVE_SNR_CEIL)
    live_coh = _norm(_isotonic_apply("coh", phase_coh),
                     _LIVE_COH_FLOOR, _LIVE_COH_CEIL)
    refresh_suspicion = (1.0 if refresh_ratio > _REPLAY_REFRESH_TRIGGER
                         else _norm(refresh_ratio, 0.0, _REPLAY_REFRESH_TRIGGER))

    # v9.8.1 (2026-05-12) [9th-pass deep audit]: when fewer than 2 ROIs
    # have data, the branch was still emitting a non-trivial p_attack
    # (because live_snr alone drives the liveness equation while
    # live_coh defaults to 0.5 = neutral). With reliability
    # = (1/3) * min(1, snr/0.10) * 0.667 ≈ 0.22 for n_ok=1 + snr=0.10,
    # the branch exceeds `min_reliability_emit=0.15` and votes into
    # fusion with p_attack ~ 0.55-0.8 on real subjects who happen to
    # be slow to warm the cheek ROIs (glasses occlusion, slight head
    # turn, etc). The audit's verdict: TRUE-ABSTAIN until at least 2
    # ROIs are populated. With reliability=0.0 the branch is filtered
    # out by `usable = [b for b in branches if b.reliability >= 0.15]`
    # in fusion.fuse() -- no false vote into the SPOOF ledger.
    if n_ok < 2:
        return BranchScore(
            name="rppg",
            p_attack=0.5,           # marker only; reliability=0 means
                                    # fusion ignores this entirely
            reliability=0.0,
            dominant_class=None,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            meta={"n_rois": float(n_ok),
                  "abstained_few_rois": True,
                  "pulse_snr_raw": float(pulse_snr)},
        )
    # >= 2 ROIs populated: the standard cross-ROI coherence path. The
    # uniform_penalty catches the "all ROIs report identical coherence"
    # replay-attack signature.
    uniform_penalty = 1.0 if (coh_uniform > 0.8 and pulse_snr < 0.15) else coh_uniform * 0.5

    liveness = 0.55 * live_snr + 0.30 * live_coh + 0.15 * (1.0 - refresh_suspicion)
    p_attack = min(1.0, (1.0 - liveness) + 0.25 * uniform_penalty)

    # API_unq 2026-05-14 (P-anti-rePPG): spatial pulse-propagation check.
    # Real cardiovascular wave shows forehead-leads-cheek by ~50-150 ms
    # (carotid -> facial artery branches). Synthetic injection (rePPG,
    # generative deepfake-with-pulse) applies the signal uniformly --
    # all inter-region lags collapse near 0. When the propagation
    # signature is suspicious, add a paddng penalty to p_attack.
    try:
        prop = _rppg.spatial_propagation_score(ps, hr_hint_bpm=hr_hint_bpm)
    except Exception:
        prop = {"score": 0.5, "reason": "exception",
                "is_plausible_live": False, "is_suspicious": False,
                "forehead_to_lc_lag_s": 0.0,
                "forehead_to_rc_lag_s": 0.0,
                "lc_to_rc_lag_s": 0.0}
    prop_penalty = 0.0
    if prop.get("is_suspicious"):
        # uniform-signal signature: strong push toward SPOOF (deepfake)
        prop_penalty = 0.30
    elif prop.get("is_plausible_live"):
        # plausible propagation: pull toward LIVE
        prop_penalty = -0.15
    p_attack = max(0.0, min(1.0, p_attack + prop_penalty))

    # Reliability — how many ROIs contributed + SNR presence. When the
    # propagation check is informative (live or suspicious), nudge
    # reliability up slightly; "ambiguous" leaves it unchanged.
    reliability = min(1.0,
                      (n_ok / 3.0) *
                      min(1.0, pulse_snr / max(_LIVE_SNR_FLOOR, 1e-6)) *
                      (0.5 + 0.5 * (n_ok / 3.0)))
    if prop.get("is_plausible_live") or prop.get("is_suspicious"):
        reliability = min(1.0, reliability * 1.20)

    dominant = None
    if refresh_suspicion > 0.6:
        dominant = "REPLAY_SCREEN"
    elif pulse_snr < 0.05:
        dominant = "PRINT"
    elif prop.get("is_suspicious"):
        # Synthetic-uniform pulse signature -> classify as deepfake stream
        dominant = "DEEPFAKE_STREAM"

    return BranchScore(
        name="rppg",
        p_attack=round(p_attack, 4),
        reliability=round(reliability, 4),
        latency_ms=(time.monotonic() - t0) * 1000.0,
        meta={
            "pulse_snr": round(pulse_snr, 4),
            "phase_coh": round(phase_coh, 4),
            "refresh_ratio": round(refresh_ratio, 5),
            "coh_uniform": round(coh_uniform, 4),
            "fs": round(fs, 2),
            "n_rois": float(n_ok),
            # API_unq spatial-propagation telemetry (Item 1)
            "prop_score": round(float(prop.get("score", 0.5)), 4),
            "prop_reason": str(prop.get("reason", "")),
            "prop_lag_fh_lc_ms": round(1000.0 * float(prop.get("forehead_to_lc_lag_s", 0.0)), 2),
            "prop_lag_fh_rc_ms": round(1000.0 * float(prop.get("forehead_to_rc_lag_s", 0.0)), 2),
            "prop_lag_lc_rc_ms": round(1000.0 * float(prop.get("lc_to_rc_lag_s", 0.0)), 2),
            "prop_live": bool(prop.get("is_plausible_live", False)),
            "prop_suspicious": bool(prop.get("is_suspicious", False)),
        },
        dominant_class=dominant,
    )
