"""pad/export_adv_onnx.py -- recovery helper.

If `pad.train_adversarial` finished training but its ONNX export crashed
(historically: Windows cp1252 stdout choking on the dynamo exporter's
Unicode-emoji verbose-print), the `.pth` we now save first is the
source of truth. This script reloads it and emits the matching `.onnx`
without re-training.

Usage:
    python -m pad.export_adv_onnx \\
        --pth pad/models/patch_depth_fas_v8_adv.pth \\
        --onnx pad/models/patch_depth_fas_v8_adv.onnx
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    """CLI: re-export ONNX from a saved .pth without re-running training."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pth", required=True, type=Path,
                    help="Path to the saved state-dict (e.g. patch_depth_fas_v8_adv.pth)")
    ap.add_argument("--onnx", required=True, type=Path,
                    help="Output ONNX path")
    args = ap.parse_args(argv)

    if not args.pth.is_file():
        print(f"[export-adv] PTH not found: {args.pth}", file=sys.stderr)
        return 2

    import torch
    from torchvision.models import mobilenet_v3_small

    model = mobilenet_v3_small(weights=None)
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_feats, 2)
    sd = torch.load(str(args.pth), map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval().cpu()

    # Same legacy-exporter shape contract as pad/train_adversarial.py
    from pad.train_adversarial import export_onnx
    export_onnx(model, args.onnx)
    print(f"[export-adv] OK -- {args.onnx.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
