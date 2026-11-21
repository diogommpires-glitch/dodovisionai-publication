"""v9.4 PAD Step 8 — adversarial patch defense tests."""
from __future__ import annotations
import numpy as np

from pad.defense import adv_patch


def test_fourier_cutout_preserves_shape():
    img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    out = adv_patch.fourier_cutout(img, seed=42)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_jpeg_reencode_preserves_shape():
    img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    out = adv_patch.jpeg_reencode(img, quality=85)
    assert out.shape == img.shape


def test_defend_batch_returns_k_samples():
    img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    out = adv_patch.defend_batch(img, k_samples=3)
    # Returns at least jpeg + K-1 cutouts — length >= K
    assert len(out) >= 3


def test_stats_counters_increment():
    before = adv_patch.stats()
    img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    adv_patch.defend_batch(img, k_samples=2)
    after = adv_patch.stats()
    assert after["invocations"] >= before["invocations"] + 1
