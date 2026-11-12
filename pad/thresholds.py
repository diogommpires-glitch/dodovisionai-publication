"""
DODO VISION AI v9.4 — PAD decision thresholds.

Single source of truth for verdict thresholds + default per-branch
priors. Never hardcode these inside branches — read from here.

Defaults picked per the canonical PAD literature (Liu et al. CVPR 2018
+ ISO/IEC 30107-3 reportable operating points + Domain Generalization
via Ensemble Stacking, IJCV 2024) so the fused log-odds land in a
sensible regime before per-deployment calibration.

API_unq (2026-05-13): added `radar_veto_logodds_delta` (P0.2). The
radar-veto gate previously hardcoded `p_attack <= 0.25` in fusion.py
and meta_fuse.py. With `DODO_DEPLOY_SPOOF_PRIOR=0.30` (Saerens 2002)
this drifts: radar's LIVE landing point shifts but the gate doesn't
follow. The principled fix is a fixed log-odds margin above the
training prior. logit(0.25)-logit(0.02) ≈ -1.10 - (-3.89) ≈ +2.79
(positive = above-prior). The veto fires when the radar's prior-
relative log-odds delta is BELOW this floor (i.e. radar is voting
strongly LIVE relative to prior).
"""
from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class PADThresholds:
    # Fused decision boundaries on p_attack ∈ [0,1]
    spoof_high: float = 0.75        # p > this → SPOOF
    live_low: float = 0.15          # p < this → LIVE; in-between → UNCERTAIN
    # Prior probability of attack — Beta(2, 98) → 0.02 seems fine for
    # consumer deployments; SOC / high-security bumps to 0.10.
    prior_p_attack: float = 0.02
    # Per-branch reliability floor. A branch claiming reliability=1.0
    # after <3 observations is almost certainly overconfident; clamp.
    max_reliability: float = 0.95
    min_reliability_emit: float = 0.05
    # UI hysteresis: require N of last M verdicts matching to flip the
    # tile badge. Matches Soukupova/Cech blink-hysteresis rationale.
    ui_hysteresis_window: int = 6
    ui_hysteresis_threshold: float = 0.50
    # API_unq (P0.2): per-branch prior-relative log-odds gates.
    # Radar veto fires when radar's `logit(p_attack) - logit(prior)`
    # is at-or-below this floor (i.e. radar voted strongly LIVE
    # relative to the *current* deployment prior). Invariant to
    # DODO_DEPLOY_SPOOF_PRIOR shifts.
    # Default = logit(0.25)-logit(0.02) ≈ +2.79 — matches the pre-
    # API_unq hardcoded gate when prior_p_attack=0.02. Negative
    # values (more conservative) require radar to be more LIVE-
    # confident before vetoing.
    radar_veto_logodds_delta: float = math.log(0.25 / 0.75) - math.log(0.02 / 0.98)
    # Same idea for the "radar STRONG-live → upgrade UNCERTAIN to LIVE"
    # gate (DODO_PAD_TRUST_RADAR=1 path). Default = logit(0.20)-logit(0.02).
    radar_strong_logodds_delta: float = math.log(0.20 / 0.80) - math.log(0.02 / 0.98)


DEFAULT_THRESHOLDS = PADThresholds()
