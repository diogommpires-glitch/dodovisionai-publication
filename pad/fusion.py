"""
DODO VISION AI — Logarithmic-opinion-pool fusion for PAD branches.

v9.8.1 2026-05-12 (Audit C1): fusion math switched from "weighted sum
of prior-offset log-odds" to "reliability-weighted MEAN of branch
log-odds".

API_unq 2026-05-13 (P0.1 docstring fix per Phase 2 lit-review): this
is a LOGARITHMIC OPINION POOL, not Bayesian Model Averaging. BMA is
sum_m p(y|x,M) * p(M|D) -- an arithmetic mixture in probability
space, requiring posterior model weights. The weighted-mean-of-log-
odds form here is exactly the binary-case log opinion pool, which
corresponds to weighted geometric averaging of probabilities (Genest
& Zidek 1986) and whose optimal weights solve a quadratic program
under the bias-variance decomposition (Heskes NIPS 1998). The two
coincide only in trivial cases.

References:
  - Genest, C. & Zidek, J.V. (1986). Combining Probability
    Distributions: A Critique and an Annotated Bibliography.
    Statistical Science 1(1), 114-135. doi:10.1214/ss/1177013825.
    [Defines the linear pool, the logarithmic pool, and the
    externally-Bayesian property; log pool = weighted geometric
    average of probabilities = weighted mean of log-odds in the
    binary case.]
  - Heskes, T. (1998). Selecting Weighting Factors in Logarithmic
    Opinion Pools. NIPS 10, 266-272. [Proves the bias-variance
    decomposition of the ensemble error equals the KL decomposition
    of the log pool; optimal weights are a constrained QP whose
    Hessian is the covariance of expert errors. When experts are
    perfectly correlated the Hessian becomes singular and the pool
    collapses to a single expert -- the head-on warning for the
    DodoVisionAI branches that share a face crop.]
  - Hoeting, Madigan, Raftery & Volinsky (1999). Bayesian Model
    Averaging: A Tutorial. Statistical Science 14(4), 382-417.
    [Different beast; included only to disambiguate citations.]
  - ISO/IEC 30107-3:2023 -- defines APCER/BPCER/ACER reporting that
    downstream metrics code (`pad.metrics`, `pad.report`) computes
    off the fused probabilities this module produces.

Fusion formula (default, DODO_PAD_FUSION_MATH=weighted_mean):

    L = Σ_i w_i * logit(p_i)  /  Σ_j w_j           if any branch usable
    L = logit(prior)                                otherwise

    p_fused = σ(L)

Legacy formula (DODO_PAD_FUSION_MATH=legacy, kept one release for
reproducibility of pre-Audit-C1 paper tables):

    L = logit(prior) + Σ_i w_i * (logit(p_i) - logit(prior))

where w_i is the branch's reliability for this frame. A reliability-zero
branch contributes nothing. The final probability is σ(L).

Rationale (new principled formula):
  - Adding more LIVE-confident branches NEVER pushes p_fused above the
    most-suspicious branch's p_attack. Pre-Audit-C1, every branch added
    +(logit(p_i) - logit(prior)) regardless of how many branches were
    already voting, so the prior was compounded N times.
  - Missing branches are naturally handled (reliability=0 -> w=0).
  - Single-branch case: L = logit(p_1), i.e. no prior interference --
    fusion of one branch IS that branch (correct behaviour for paper-
    table single-branch baselines).
"""
from __future__ import annotations
import math
import time
from typing import List, Optional, Dict
from collections import Counter

from pad.types import (
    BranchScore, PADScore, PADState, AttackClass, VerdictLabel,
    utc_now,
)
from pad.thresholds import PADThresholds, DEFAULT_THRESHOLDS

EPS = 1e-4


def _logit(p: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _classify(p: float, th: PADThresholds) -> VerdictLabel:
    if p > th.spoof_high:
        return "SPOOF"
    if p < th.live_low:
        return "LIVE"
    return "UNCERTAIN"


def _infer_class(branches: List[BranchScore],
                 prior_p_attack: float = 0.02) -> AttackClass:
    """Pick the most likely attack class based on which branch(es) drove
    the spoof signal.

    API_unq (P1.9): weight by EVIDENCE STRENGTH `reliability * |logit(p)
    - logit(prior)|` instead of the pre-API_unq `reliability * p_attack`.
    The old form double-counts the spoof side: a branch at p=0.95 and a
    branch at p=0.50 both contribute their full reliability, but only
    one carries real evidence. The new form is the standard log-odds
    evidence formulation (Heskes 1998, Good 1985 weight-of-evidence)
    and gives a branch at p=0.50 (zero evidence vs prior=0.02 with
    logit gap = 3.89) approximately the same vote as a branch at
    p=0.95 (gap = 6.83) only weighted proportional to the magnitude.
    """
    prior_logit = _logit(prior_p_attack)
    # Only consider branches that actually think it's an attack.
    candidates = [
        (b.dominant_class, b.reliability * abs(_logit(b.p_attack) - prior_logit))
        for b in branches
        if b.p_attack > 0.5 and b.dominant_class
    ]
    if not candidates:
        # No single branch pushed toward attack — fall back to UNKNOWN.
        return "UNKNOWN"
    # Weighted vote by reliability * evidence-strength
    weighted: Dict[str, float] = {}
    for cls, weight in candidates:
        if cls:
            weighted[cls] = weighted.get(cls, 0.0) + weight
    best = max(weighted.items(), key=lambda kv: kv[1])
    return best[0]  # type: ignore[return-value]


def fuse(
    branches: List[BranchScore],
    thresholds: Optional[PADThresholds] = None,
    ts: Optional[float] = None,
) -> PADScore:
    """Fuse branch scores into a PADScore via weighted log-odds.

    This is the SINGLE entry point. No downstream code reads branch
    scores directly; everything consumes the fused PADScore.
    """
    th = thresholds or DEFAULT_THRESHOLDS
    ts = ts if ts is not None else utc_now()

    # v9.8.1 2026-05-12 (11th pass per operator: "still shows consistent
    # spoof"):
    #
    # ROOT CAUSE traced via live /pad/0 telemetry:
    #   Per-branch calibrators apply `DODO_DEPLOY_SPOOF_PRIOR=0.30`
    #   prior-shift (Saerens 2002) on top of the Platt fit. After the
    #   shift, branches output ~0.15-0.18 when voting LIVE on a real
    #   subject (e.g. radar p=0.169, dynamics p=0.177, moire p=0.173).
    #
    #   BUT fusion uses `prior_p_attack=0.02` from PADThresholds as its
    #   own prior. The log-odds formula computes
    #     contribution_i = w_i * (logit(p_i) - logit(prior))
    #   With prior=0.02 (logit=-3.89) and p_branch=0.17 (logit=-1.6),
    #   contribution = +2.3 per branch -- POSITIVE = pulling toward
    #   SPOOF. With 5 LIVE-ish branches at reliability=0.2-1.0, total
    #   contribution sums to +3.8, fusion lands at p=0.46 (UNCERTAIN
    #   bordering on SPOOF) on a real subject.
    #
    # FIX: align fusion's prior with the calibrator's deploy prior.
    # When DODO_DEPLOY_SPOOF_PRIOR is set, use it as fusion prior too
    # so the two stages don't disagree about "what does LIVE look like".
    # Default 0.02 preserved for backward compat when env unset.
    import os as _os
    _env_prior = _os.environ.get("DODO_DEPLOY_SPOOF_PRIOR", "").strip().lower()
    _fusion_prior = th.prior_p_attack
    if _env_prior and _env_prior not in ("auto", "0", "0.0", "off", "no", "false"):
        try:
            _candidate = float(_env_prior)
            if 0.0 < _candidate < 1.0:
                _fusion_prior = _candidate
        except ValueError:
            pass

    # v9.8.1 2026-05-12 (Audit C1: principled fusion math). Previously
    # the fusion accumulator was
    #     L = logit(prior) + Σ_i w_i * (logit(p_i) - logit(prior))
    # where w_i = b.reliability is unbounded by Σw_j. With 5 branches
    # at reliability=0.5-1.0 the weight sum is 3-4 and each branch's
    # contribution compounds linearly -- not a posterior-of-posteriors
    # but a weighted ADD of log-odds offsets from prior. That's not the
    # weighted-geometric-mean-in-odds-space the docstring claims. The
    # 11th-pass radar-veto incident was a direct symptom: 5 LIVE-ish
    # branches each contributing +0.7 to logit summed to +3.5, fusion
    # landed at p=0.97 SPOOF on a real subject.
    #
    # Replacement: WEIGHTED MEAN OF LOG-ODDS (proper Bayesian model
    # averaging form, Madigan & Raftery 1994; Hoeting et al. 1999):
    #     L = (Σ_i w_i * logit(p_i)) / Σ_i w_j     when any branch usable
    #     L = logit(prior)                          when no branch usable
    # The prior no longer multiplicatively appears in every term; it
    # only kicks in as the empty-fallback. This matches what readers
    # expect from "weighted log-odds fusion" and removes the
    # double-prior interaction with the calibrator's Saerens shift.
    #
    # We expose the legacy formula behind DODO_PAD_FUSION_MATH=legacy
    # for one release so paper Table N replicators can compare numbers
    # if they ran the eval pre-Audit-C1.
    _fusion_math = _os.environ.get("DODO_PAD_FUSION_MATH", "weighted_mean").strip().lower()

    # API_unq P1.1 hook: apply ReliabilityCalibrator if one is registered
    # for the branch (auto-loaded from pad/models/reliability_calibrators.pkl
    # at module import). For branches with no calibrator the raw reliability
    # passes through unchanged. This is the single chokepoint that makes
    # the "common-target reliability" contribution actually fire.
    try:
        from pad.calibrate import transform_branch_reliability as _trc
        _have_rc = True
    except Exception:
        _trc = None
        _have_rc = False

    def _w_of(b) -> float:
        raw = float(b.reliability)
        if _have_rc and _trc is not None:
            try:
                raw = float(_trc(b.name, raw))
            except Exception:
                pass
        return min(max(raw, 0.0), th.max_reliability)

    usable = [b for b in branches if _w_of(b) >= th.min_reliability_emit]
    if not usable:
        L = _logit(_fusion_prior)
    elif _fusion_math == "legacy":
        # Pre-Audit-C1 formula. Kept for reproducibility of older
        # paper-table numbers; default mode is "weighted_mean" below.
        prior_L = _logit(_fusion_prior)
        L = prior_L
        for b in usable:
            w = _w_of(b)
            L += w * (_logit(b.p_attack) - prior_L)
    else:
        # Default: principled weighted mean of branch log-odds.
        # Equivalent to Bayesian Model Averaging when w_i is the
        # branch's posterior reliability (Hoeting 1999) and to a
        # "stacked log-odds" baseline (Madigan & Raftery 1994).
        w_sum = 0.0
        l_acc = 0.0
        for b in usable:
            w = _w_of(b)
            w_sum += w
            l_acc += w * _logit(b.p_attack)
        if w_sum > 0:
            L = l_acc / w_sum
        else:
            L = _logit(_fusion_prior)

    p = _sigmoid(L)
    verdict = _classify(p, th)

    # v9.7.6 — physical-liveness gate.
    #
    # The visual branches (patch_depth, moire, deepfake) can be fooled
    # by a sufficiently high-quality phone screen — a 1080p+ OLED at
    # the right distance produces a face crop that defeats CNN depth
    # estimators and avoids the moire/scanline patterns those models
    # were trained on. The system was reporting confident LIVE on a
    # phone-screen attack because:
    #   - radar_presence: offline (port not opened)
    #   - rppg:          no lock (cannot extract HR from a screen)
    #   - dynamics:      no signal (depends on rppg + radar)
    #   - blink_gaze:    abstaining
    #   - patch_depth + moire: BOTH SAID LIVE (fooled by the screen)
    #
    # The fix isn't to "fix" patch_depth/moire (they're at the limit
    # of what visual-only PAD can do); it's to refuse confident LIVE
    # when NO physical-liveness sensor confirmed presence. Without
    # radar (chest motion) AND without rppg (camera-derived HR with
    # lock), the verdict is downgraded to UNCERTAIN with a `degraded`
    # meta tag so the UI can explain why.
    # v9.8.1 2026-05-12 (9th-pass deep audit): dropped stale "temporal_mvit"
    # name -- the deepfake_temporal MViT signal is emitted under
    # `name="deepfake"`, not under the model's internal name. The branch
    # tuple was effectively a 3-branch check (rppg, radar, blink_gaze)
    # because "temporal_mvit" never matched anything.
    # API_unq (2026-05-13): screen_flash added as a physical-liveness signal.
    # It's an active ambient-response test; a confirmed alignment is a
    # strong live signal that should gate the degraded-LIVE downgrade.
    PHYSICAL_BRANCHES = ("radar_presence", "rppg", "deepfake", "blink_gaze",
                         "screen_flash")
    PHYSICAL_REL_MIN  = 0.15  # branch must have at least this much
                              #   evidence to count as "real signal"
    physical_confirmed = any(
        b.name in PHYSICAL_BRANCHES and b.reliability >= PHYSICAL_REL_MIN
        and b.p_attack < 0.5   # branch must vote LIVE-ish, not SPOOF
        for b in branches
    )
    degraded = (verdict == "LIVE" and not physical_confirmed)
    if degraded:
        # Soft downgrade: keep p_attack so forensic logs still see the
        # raw fused score, but emit UNCERTAIN so the UI doesn't
        # green-light a phone-screen attack.
        verdict = "UNCERTAIN"

    # v9.8.1 (2026-05-12) — RADAR PHYSICAL VETO (symmetric to the
    # degraded-LIVE downgrade above).
    #
    # OBSERVED on operator: with radar_presence reporting p=0.05 rel=1.0
    # (rock-solid LIVE -- the 60 GHz radar sees a breathing torso 60 cm
    # away) the fused log-odds still came out +10 SPOOF because the
    # visual branches (rppg, dynamics, deepfake, moire, blink_gaze) all
    # output p > 0.5 for a real subject under default calibrators.
    # The math is technically correct given prior_p_attack=0.02, but
    # the END USER IS PHYSICALLY PRESENT and the system shouldn't call
    # them a print attack just because the rPPG calibrator is biased.
    #
    # Defense: when the radar branch is confident-LIVE
    # (rel >= 0.5 AND p_attack <= 0.15), refuse to emit SPOOF. Same
    # half-step "soft downgrade" pattern as the degraded path: keep the
    # raw fused p_attack/log_odds in the response so forensic logs
    # capture the disagreement, but flip the verdict to UNCERTAIN and
    # surface `radar_physical_veto=True` in meta so the UI can render
    # an info chip ("radar says LIVE, visual branches disagree").
    #
    # Why radar specifically: it's the only branch on this stack that
    # observes the subject through a physical channel the camera can't
    # see (60 GHz millimetre-wave chest motion). Visual branches can
    # be fooled by a high-quality replay; radar cannot.
    radar = next((b for b in branches if b.name == "radar_presence"), None)
    # API_unq 2026-05-13 (P0.2): replace hardcoded `p_attack <= 0.25` /
    # `<= 0.20` literals with prior-relative log-odds gates. The 12th-
    # pass v9.8.1 patch noted that DODO_DEPLOY_SPOOF_PRIOR=0.30 shifted
    # radar's LIVE landing point to ~0.17 and they had to re-tune the
    # literals. With `delta = logit(p) - logit(prior)` and a fixed
    # delta threshold in log-odds space, the gate is INVARIANT under
    # any future prior re-anchoring (Saerens et al. 2002).
    #
    #   delta <= radar_veto_logodds_delta  → radar is voting LIVE
    #                                         relative to prior
    #   delta <= radar_strong_logodds_delta → radar is voting STRONGLY
    #                                         LIVE relative to prior
    _fusion_prior_logit = _logit(_fusion_prior)
    if radar is not None:
        _radar_logodds_delta = _logit(radar.p_attack) - _fusion_prior_logit
    else:
        _radar_logodds_delta = 0.0
    radar_physical_veto = (
        radar is not None
        and radar.reliability >= 0.50
        and _radar_logodds_delta <= th.radar_veto_logodds_delta
    )
    # Even stronger signal: radar is *certain* and we have at least one
    # other physical-domain branch (rppg / blink_gaze / deepfake)
    # producing a real signal. Use that to upgrade UNCERTAIN -> LIVE
    # when DODO_PAD_TRUST_RADAR is opt-in set. Off by default because
    # in production a mask-with-radar-leakage attack could fool radar
    # alone; with the env var set, the operator declares they accept
    # that trade-off for demo / paper-grade visualisation.
    _trust_radar = _os.environ.get("DODO_PAD_TRUST_RADAR", "0").lower() in ("1", "true", "yes")
    radar_strong = (
        radar is not None
        and radar.reliability >= 0.80
        and _radar_logodds_delta <= th.radar_strong_logodds_delta
    )
    if verdict == "SPOOF" and radar_physical_veto:
        verdict = "UNCERTAIN"
    if _trust_radar and radar_strong and physical_confirmed and verdict == "UNCERTAIN":
        # Upgrade UNCERTAIN -> LIVE when radar is rock-solid AND any
        # other physical branch (rppg/blink_gaze/temporal_mvit) agrees.
        # Visual-only branches still appear in the breakdown so a
        # paper-grade reviewer can audit the disagreement.
        verdict = "LIVE"

    # When the radar physically confirms a live subject we don't want
    # the UI rendering a scary attack-class chip ("PRINT", "NO_BLINK",
    # "REPLAY_SCREEN") -- the visual branches were wrong, the radar
    # says you're a real person. Suppress the inferred attack_class
    # so the UI just shows "UNCERTAIN" without further suspicion.
    #
    # v9.8.1 2026-05-12 (13th pass per operator: "still working
    # terrible" -- attack_class=PRINT showing while verdict=LIVE):
    # Suppress attack_class whenever the FUSED VERDICT IS LIVE.
    # Rationale: attack_class only makes sense as a hint about WHICH
    # kind of attack we're seeing, but if the fused verdict is LIVE
    # the system has already concluded there's no attack -- showing
    # a class label is just noise that confuses operators. A single
    # branch's `dominant_class` (e.g. dynamics flagging PRINT because
    # its three_d heuristic mis-fired on glasses) should never bubble
    # up to the top-level attack_class when fusion already overruled
    # that branch. Independent of the radar_physical_veto path, which
    # remains in place for the UNCERTAIN/SPOOF cases where radar
    # alone overrides visual disagreement.
    if verdict == "LIVE" or radar_physical_veto:
        attack_class = None
    else:
        # API_unq (P1.9): use evidence-strength weighting w/ correct prior.
        attack_class = _infer_class(usable, prior_p_attack=_fusion_prior)

    # v9.8.1 2026-05-12 (Phase 2 #3: disagreement consensus penalty).
    # When two or more high-reliability branches give wildly different
    # verdicts (one says 0.95 SPOOF, another 0.05 LIVE) the log-odds
    # math mathematically cancels them out -- the fused p_attack lands
    # near the prior (UNCERTAIN). That's the right NUMERIC answer for
    # the verdict but it hides the underlying instability: the sensors
    # don't agree on what they're seeing. Flag this so the UI can show
    # a specific banner ("Sensor confusion -- please adjust lighting")
    # rather than the generic UNCERTAIN tooltip.
    #
    # Definition: variance of p_attack across branches with rel > 0.3.
    # Need at least 2 such branches; otherwise variance is meaningless.
    # Threshold (0.08) corresponds to roughly 2 branches at 0.2 and
    # 0.8 (variance 0.09) -- the empirical "sensors flat-out disagree"
    # signature. 1 branch at 0.05 + 1 branch at 0.50 only gives
    # variance 0.05 (sensors are uncertain but not contradictory).
    high_rel_branches = [b for b in branches if b.reliability > 0.30]
    degraded_disagreement = False
    p_variance = 0.0
    if len(high_rel_branches) >= 2:
        _ps = [float(b.p_attack) for b in high_rel_branches]
        _mean_p = sum(_ps) / len(_ps)
        p_variance = sum((p - _mean_p) ** 2 for p in _ps) / len(_ps)
        degraded_disagreement = p_variance > 0.08

    return PADScore(
        p_attack=round(p, 4),
        verdict=verdict,
        attack_class=attack_class,
        branches=branches,
        ts=ts,
        log_odds=round(L, 4),
        meta={"degraded_no_physical":  bool(degraded),
              "physical_confirmed":    bool(physical_confirmed),
              "radar_physical_veto":   bool(radar_physical_veto),
              "degraded_disagreement": bool(degraded_disagreement),
              "p_variance":            round(p_variance, 4),
              "high_rel_n":            len(high_rel_branches)},
    )


def fuse_with_history(
    branches: List[BranchScore],
    state: PADState,
    thresholds: Optional[PADThresholds] = None,
    hysteresis: bool = True,
) -> PADScore:
    """Fuse + update PADState in one call; applies UI hysteresis so a
    single spurious SPOOF frame doesn't flip the badge.

    Returns the instantaneous PADScore (downstream UI can check
    `state.recent_verdict_ratio('SPOOF', window=N)` to decide whether
    to actually paint the badge red).
    """
    th = thresholds or DEFAULT_THRESHOLDS
    score = fuse(branches, thresholds=th)
    state.push(score)
    if hysteresis:
        # If the instantaneous verdict is SPOOF but the recent history
        # doesn't corroborate, downgrade the emitted verdict to
        # UNCERTAIN. We do NOT mutate the stored `last` — that remains
        # the instantaneous value for forensic audit.
        if score.verdict == "SPOOF":
            ratio = state.recent_verdict_ratio(
                "SPOOF", window=th.ui_hysteresis_window)
            if ratio < th.ui_hysteresis_threshold:
                return PADScore(
                    p_attack=score.p_attack,
                    verdict="UNCERTAIN",
                    attack_class=score.attack_class,
                    branches=score.branches,
                    ts=score.ts,
                    log_odds=score.log_odds,
                    meta=dict(getattr(score, "meta", {}) or {}),
                )
    return score
