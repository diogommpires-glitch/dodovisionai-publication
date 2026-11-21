"""Tests for v9.8.1 Audit-C1 fusion math.

The new default is `DODO_PAD_FUSION_MATH=weighted_mean`:

    L = Σ_i w_i * logit(p_i) / Σ_j w_j

Properties we want:
  1. Single-branch fusion equals that branch's logit
     (no prior interference).
  2. Adding more LIVE-leaning branches DOES NOT push the fused
     verdict toward SPOOF (which the pre-C1 formula did via the
     compounding prior offsets).
  3. The legacy mode (DODO_PAD_FUSION_MATH=legacy) still reproduces
     the pre-C1 numbers.
"""
from __future__ import annotations
import math
import os

import pytest

from pad.fusion import fuse, _logit, _sigmoid
from pad.types import BranchScore


def _save_env(name: str):
    """Helper for env-var setup that always restores the prior value."""
    prev = os.environ.get(name)
    def _restore():
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev
    return _restore


def test_single_branch_fusion_equals_branch_logit():
    """With one branch under the new math, fused p_attack == branch p_attack."""
    restore = _save_env("DODO_PAD_FUSION_MATH")
    try:
        os.environ["DODO_PAD_FUSION_MATH"] = "weighted_mean"
        # Also clear deploy-prior so the empty-fallback path doesn't kick in.
        prior_restore = _save_env("DODO_DEPLOY_SPOOF_PRIOR")
        os.environ["DODO_DEPLOY_SPOOF_PRIOR"] = ""
        try:
            for p_in in (0.10, 0.50, 0.90):
                br = [BranchScore(name="solo", p_attack=p_in, reliability=0.9)]
                score = fuse(br)
                assert abs(score.p_attack - p_in) < 1e-3, \
                    f"single-branch fusion drifted: input={p_in} fused={score.p_attack}"
        finally:
            prior_restore()
    finally:
        restore()


def test_adding_live_branches_does_not_push_to_spoof():
    """C1 regression: 5 live-leaning branches at p≈0.17 should NOT
    sum to a SPOOF verdict (the pre-fix legacy formula did)."""
    restore = _save_env("DODO_PAD_FUSION_MATH")
    try:
        os.environ["DODO_PAD_FUSION_MATH"] = "weighted_mean"
        br = [
            BranchScore(name="patch_depth",    p_attack=0.025, reliability=0.9),
            BranchScore(name="deepfake",       p_attack=0.034, reliability=0.7),
            BranchScore(name="radar_presence", p_attack=0.169, reliability=1.0),
            BranchScore(name="moire",          p_attack=0.173, reliability=0.4),
            BranchScore(name="dynamics",       p_attack=0.177, reliability=0.4),
        ]
        score = fuse(br)
        # All 5 branches vote well below 0.5; weighted-mean must too.
        assert score.p_attack < 0.30, \
            f"weighted-mean fusion should stay LIVE-ish, got {score.p_attack}"
    finally:
        restore()


def test_legacy_mode_reproduces_compounding_behaviour():
    """The legacy formula (= pre-Audit-C1) preserves the old behaviour
    so users wanting to reproduce older paper-table numbers can opt in."""
    restore_math = _save_env("DODO_PAD_FUSION_MATH")
    restore_prior = _save_env("DODO_DEPLOY_SPOOF_PRIOR")
    try:
        os.environ["DODO_PAD_FUSION_MATH"] = "legacy"
        # Force fusion prior to 0.02 like the pre-fix default so the
        # accumulator drifts toward UNCERTAIN on LIVE-ish branches.
        os.environ["DODO_DEPLOY_SPOOF_PRIOR"] = ""    # use thresholds default
        br = [
            BranchScore(name="patch_depth",    p_attack=0.025, reliability=0.9),
            BranchScore(name="deepfake",       p_attack=0.034, reliability=0.7),
            BranchScore(name="radar_presence", p_attack=0.169, reliability=1.0),
            BranchScore(name="moire",          p_attack=0.173, reliability=0.4),
            BranchScore(name="dynamics",       p_attack=0.177, reliability=0.4),
        ]
        legacy = fuse(br)
        # Same inputs under weighted_mean (new default).
        os.environ["DODO_PAD_FUSION_MATH"] = "weighted_mean"
        new = fuse(br)
        # The new mean fusion should land MORE confident-LIVE than the
        # legacy formula (which compounded the prior).
        assert new.p_attack <= legacy.p_attack + 1e-9, (
            f"weighted_mean should be no MORE attack-leaning than legacy. "
            f"new={new.p_attack} legacy={legacy.p_attack}"
        )
    finally:
        restore_math()
        restore_prior()


def test_zero_branches_falls_back_to_prior():
    """No usable branches -> fused p = prior."""
    restore = _save_env("DODO_PAD_FUSION_MATH")
    restore_prior = _save_env("DODO_DEPLOY_SPOOF_PRIOR")
    try:
        os.environ["DODO_PAD_FUSION_MATH"] = "weighted_mean"
        os.environ["DODO_DEPLOY_SPOOF_PRIOR"] = "0.30"
        # All branches below the emit floor (default min_reliability_emit=0.05)
        br = [BranchScore(name="x", p_attack=0.5, reliability=0.0)]
        score = fuse(br)
        # Prior=0.30 -> sigmoid(logit(0.30)) = 0.30
        assert abs(score.p_attack - 0.30) < 0.01
    finally:
        restore()
        restore_prior()


def test_two_branches_disagreeing_average_their_logits():
    """One branch at 0.05, one at 0.95, equal reliability -> mean
    of their logits = sigmoid(0) = 0.5."""
    restore = _save_env("DODO_PAD_FUSION_MATH")
    try:
        os.environ["DODO_PAD_FUSION_MATH"] = "weighted_mean"
        br = [
            BranchScore(name="A", p_attack=0.05, reliability=0.5),
            BranchScore(name="B", p_attack=0.95, reliability=0.5),
        ]
        score = fuse(br)
        assert abs(score.p_attack - 0.5) < 0.01, \
            f"opposite-vote, equal-weight branches should fuse to 0.5; got {score.p_attack}"
        # And the disagreement flag should fire.
        assert (score.meta or {}).get("degraded_disagreement") is True
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
