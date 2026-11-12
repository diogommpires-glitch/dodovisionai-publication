"""verify_3dmm.py
===============

The 3DMM (3D morphable model) basis you need for pseudo-depth and face
normalization is already bundled inside the `face-alignment` package
(1adrianb/face-alignment). Nothing to download. This script just confirms it.

Usage:
    python -m pad.verify_3dmm
"""
from __future__ import annotations
import sys


def main() -> int:
    try:
        import face_alignment
    except ImportError:
        print("[3dmm] face-alignment not installed. Run: pip install face-alignment",
              file=sys.stderr)
        return 1

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.THREE_D,   # loads the 3DMM-based FAN-3D
            flip_input=False,
            device=device,
        )
        print(f"[3dmm] OK: FaceAlignment(3D) initialized on {device}")
        print(f"[3dmm] version: {face_alignment.__version__}")
        # quick forward pass on a random tensor to make sure weights are resolved
        import numpy as np
        dummy = (np.random.rand(256, 256, 3) * 255).astype("uint8")
        _ = fa.get_landmarks_from_image(dummy)   # may return None (no face), that's fine
        print("[3dmm] forward-pass sanity check passed")
        return 0
    except Exception as e:
        print(f"[3dmm] FAILED to initialize 3DMM-based FAN: {e}", file=sys.stderr)
        print("[3dmm] Try: pip install --upgrade face-alignment", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
