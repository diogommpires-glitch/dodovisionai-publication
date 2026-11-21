"""v9.4 PAD Step 8 — Adversarial patch defense module.

Applied ONLY to PAD-bound frames (never to the main tracker input
since that would cost INSTANT-tier latency).
"""
from pad.defense.adv_patch import (
    fourier_cutout, jpeg_reencode, defend_batch, stats,
)
__all__ = ["fourier_cutout", "jpeg_reencode", "defend_batch", "stats"]
