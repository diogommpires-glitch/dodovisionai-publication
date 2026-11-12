"""fetch_patch_depth.py
====================

There is NO public ONNX for a Patch+Depth CNN face-PAD model at time of writing.
The architecture is from the Atoum et al. 2017 "Face Anti-Spoofing Using
Patch and Depth-Based CNNs" paper (IJCB) and its successors. You have to train it.

This script:
  1. Checks whether models/patch_depth.onnx exists; if so, exits OK.
  2. If not, prints a step-by-step training plan with dataset links and a
     minimum-viable config so you can train a ~15 MB ONNX on CASIA-SURF / OULU-NPU /
     SiW-Mv2 and export it to models/patch_depth.onnx.

Usage:
    python -m pad.fetch_patch_depth
    python -m pad.fetch_patch_depth --export-stub   # writes a random-init ONNX so the
                                                    # rest of the pipeline can load
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from textwrap import dedent

from . import MODEL_DIR

OUT = MODEL_DIR / "patch_depth.onnx"

INSTRUCTIONS = dedent("""
    ================================================================
    Patch+Depth CNN: no public weights. Train it yourself.
    ================================================================

    1. Datasets (request access from the original authors, all free for academic use):
         - CASIA-SURF CeFA  : https://sites.google.com/qq.com/chalearnfacespoofingattac/
         - OULU-NPU         : https://sites.google.com/site/oulunpudatabase/
         - SiW-Mv2          : https://cvlab.cse.msu.edu/project-siw-mv2.html
         - CelebA-Spoof     : https://github.com/ZhangYuanhan-AI/CelebA-Spoof   (no request needed)

       CelebA-Spoof is the fastest path: public, ~625k images, labels in JSON.

    2. Architecture (roughly 2-3 M params, fits ~15 MB fp32):
         patch branch : 96x96 RGB patches -> 5 conv blocks -> global avg pool -> fc
         depth branch : 256x256 RGB image -> U-Net-lite -> 32x32 pseudo-depth map
         fusion       : concat patch logits + depth stats -> fc -> 2-way softmax

       PRNet or 3DDFA_V2 can generate pseudo-depth ground truth for live faces;
       all-zero depth for spoofs. See Liu et al. 2018 "Learning Deep Models for
       Face Anti-Spoofing: Binary or Auxiliary Supervision".

    3. Training skeleton (PyTorch):
         loss = BCE(patch_logits, label) + 1.0 * MSE(depth_pred, depth_gt * label)
         optim = Adam(lr=1e-4), batch=32, 40 epochs, standard color jitter + random crop
         expected EER on OULU protocol 1: 3-5%

    4. Export to ONNX:
         dummy_patch = torch.randn(1, 3, 96, 96)
         dummy_img   = torch.randn(1, 3, 256, 256)
         torch.onnx.export(model, (dummy_patch, dummy_img),
                           "pad/models/patch_depth.onnx",
                           input_names=["patch", "image"],
                           output_names=["live_logit", "depth_map"],
                           dynamic_axes={"patch": {0: "N"}, "image": {0: "N"}},
                           opset_version=17)

    5. Sanity-check:
         python -c "import onnx; m=onnx.load('pad/models/patch_depth.onnx'); onnx.checker.check_model(m); print('OK', m.graph.name)"

    ================================================================
""").strip()


def _export_random_stub() -> None:
    """Write a random-init ONNX with the right I/O signature so the rest of the
    pipeline can load something while you train the real thing. DO NOT ship this
    to production, it returns garbage."""
    import torch
    import torch.nn as nn

    class PatchDepthStub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_head = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1),
            )
            self.depth_head = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 1, 3, padding=1),
                nn.AdaptiveAvgPool2d(32),
            )

        def forward(self, patch, image):  # noqa: D401
            return self.patch_head(patch).squeeze(-1), self.depth_head(image).squeeze(1)

    model = PatchDepthStub().eval()
    dummy_patch = torch.randn(1, 3, 96, 96)
    dummy_img = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model,
        (dummy_patch, dummy_img),
        OUT.as_posix(),
        input_names=["patch", "image"],
        output_names=["live_logit", "depth_map"],
        dynamic_axes={"patch": {0: "N"}, "image": {0: "N"}},
        opset_version=17,
    )
    print(f"[patch_depth] wrote RANDOM-INIT stub to {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
    print("[patch_depth] replace with a real-trained model before evaluating.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-stub", action="store_true",
                        help="write a random-init ONNX with the right I/O shape")
    args = parser.parse_args()

    if OUT.exists():
        print(f"[patch_depth] already present: {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
        return 0

    if args.export_stub:
        _export_random_stub()
        return 0

    print(INSTRUCTIONS)
    return 1


if __name__ == "__main__":
    sys.exit(main())
