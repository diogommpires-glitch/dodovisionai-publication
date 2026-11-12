"""Regression tests for the unified APCER / BPCER helper.

Specifically asserts that the boundary case (a sample whose p_attack
equals the threshold exactly) is treated as ATTACK per ISO/IEC
30107-3:2023 §7. Pre-2026-05-12, four files in this repo each
defined their own helper with subtly different inequalities; these
tests prevent that drift from returning.
"""
from __future__ import annotations
import numpy as np
import pytest

from pad.metrics import (
    apcer_bpcer_acer,
    apcer_per_class,
    best_threshold_min_acer,
    operating_point_at_bpcer,
)


# ─── Core boundary-case test (the whole point of C3) ────────────────


def test_iso_boundary_sample_is_classified_attack():
    """A sample with p_attack == threshold should be classified as
    ATTACK (per ISO/IEC 30107-3:2023 §7). That means:
      - a LIVE sample at threshold contributes to BPCER (misclassified
        as attack);
      - a SPOOF sample at threshold contributes to (1-APCER) -- it is
        correctly classified.
    """
    # Two-sample test set, one of each class, both exactly at threshold.
    apcer, bpcer, acer = apcer_bpcer_acer(
        p_attack=[0.50, 0.50], label=["live", "spoof"], threshold=0.50)
    # APCER = mean(p < t) over spoof: 0.5 < 0.5 is False -> APCER = 0
    # BPCER = mean(p >= t) over live:  0.5 >= 0.5 is True  -> BPCER = 1
    assert apcer == 0.0
    assert bpcer == 1.0
    assert acer == 0.5


def test_below_threshold_spoof_misclassified():
    """A SPOOF sample below threshold should bump APCER."""
    apcer, bpcer, _ = apcer_bpcer_acer(
        p_attack=[0.49, 0.51], label=["spoof", "spoof"], threshold=0.50)
    assert apcer == 0.5   # one of two spoofs misclassified
    assert np.isnan(bpcer)  # no live samples


def test_above_threshold_live_misclassified():
    """A LIVE sample at or above threshold should bump BPCER."""
    apcer, bpcer, _ = apcer_bpcer_acer(
        p_attack=[0.49, 0.50, 0.51], label=["live", "live", "live"], threshold=0.50)
    assert np.isnan(apcer)
    # 2 of 3 live samples >= threshold
    assert bpcer == pytest.approx(2.0 / 3.0)


def test_empty_class_returns_nan():
    """Empty attack or live class -> NaN, never silently 0."""
    a, b, ac = apcer_bpcer_acer([0.1, 0.9], ["live", "live"], 0.5)
    assert np.isnan(a)
    # ACER is bpcer/2 when apcer is NaN
    assert not np.isnan(b)


def test_labels_accept_strings_or_ints():
    """Both 'live'/'spoof' and 0/1 should produce the same result."""
    p = [0.1, 0.4, 0.6, 0.9]
    a_str, b_str, _ = apcer_bpcer_acer(p, ["live", "spoof", "live", "spoof"], 0.5)
    a_int, b_int, _ = apcer_bpcer_acer(p, [0, 1, 0, 1], 0.5)
    assert a_str == a_int
    assert b_str == b_int


# ─── apcer_per_class boundary behaviour ─────────────────────────────


def test_apcer_per_class_boundary():
    recs = [
        {"label": "live",  "p_attack": 0.50},
        {"label": "spoof", "p_attack": 0.50, "attack_class": "PRINT"},
        {"label": "spoof", "p_attack": 0.49, "attack_class": "PRINT"},
    ]
    apcer, bpcer = apcer_per_class(recs, threshold=0.50)
    # PRINT: 1 of 2 spoofs has p < 0.50 (the 0.49 sample) -> APCER = 0.5
    assert apcer["PRINT"] == 0.5
    # LIVE: 1 of 1 lives has p >= 0.50 -> BPCER = 1.0
    assert bpcer == 1.0


def test_apcer_per_class_empty_class_is_nan():
    recs = [
        {"label": "live", "p_attack": 0.0},
        {"label": "spoof", "p_attack": 0.0, "attack_class": "PRINT"},
    ]
    apcer, _ = apcer_per_class(recs, 0.5)
    # PRINT present: the single spoof has p_attack=0.0 < threshold=0.5
    # so APCER (= "misclassified as live") = 1/1 = 1.0 -- the spoof
    # WAS misclassified, that's the whole point of APCER.
    assert apcer["PRINT"] == 1.0
    # REPLAY_SCREEN has no spoof samples -> NaN sentinel.
    assert np.isnan(apcer["REPLAY_SCREEN"])


# ─── best_threshold + operating_point smoke tests ───────────────────


def test_best_threshold_min_acer_returns_finite():
    rng = np.random.default_rng(0)
    live = rng.normal(0.2, 0.1, size=200)
    spoof = rng.normal(0.7, 0.1, size=200)
    scores = np.clip(np.concatenate([live, spoof]), 0, 1)
    labels = np.array([0] * 200 + [1] * 200)
    t, a, b, ac = best_threshold_min_acer(scores, labels)
    assert 0.0 <= t <= 1.0
    assert 0.0 <= a <= 1.0
    assert 0.0 <= b <= 1.0
    assert ac < 0.2   # well-separated classes should ACER very low


def test_operating_point_at_bpcer_targets_correctly():
    # Build a synthetic dataset where we can analytically find the op point.
    rng = np.random.default_rng(0)
    live = rng.normal(0.2, 0.05, size=100)
    spoof = rng.normal(0.8, 0.05, size=100)
    recs = (
        [{"label": "live",  "p_attack": float(v)} for v in live]
        + [{"label": "spoof", "p_attack": float(v), "attack_class": "PRINT"} for v in spoof]
    )
    op = operating_point_at_bpcer(recs, target_bpcer=0.05)
    assert op["bpcer"] is not None
    assert op["bpcer"] <= 0.05 + 1e-9


def test_legacy_caller_compute_apcer_bpcer_still_works():
    """`pad.report.compute_apcer_bpcer` is the legacy public name --
    make sure it still produces correct results after the refactor."""
    from pad.report import compute_apcer_bpcer
    recs = [
        {"label": "live", "p_attack": 0.10},
        {"label": "spoof", "p_attack": 0.90, "attack_class": "PRINT"},
    ]
    apcer, bpcer = compute_apcer_bpcer(recs, 0.5)
    # Easy case: separated -- APCER for PRINT = 0, BPCER = 0
    assert apcer["PRINT"] == 0.0
    assert bpcer == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
