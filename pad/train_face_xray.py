"""pad/train_face_xray.py -- Face X-ray style deepfake detector.

Reproduces the core contribution of Li et al. 2020 CVPR (Face X-ray): a
model trained on SYNTHETIC BLENDING BOUNDARIES generalises across
deepfake synthesis methods, because every face-swap / face-reenact
operation leaves a blending seam.

Architecture: MobileNetV3-small encoder + auxiliary mask-prediction head
(trained but not exported) + primary binary classifier head. The ONNX
output exports only the classifier so the `deepfake_branch` auto-detector
picks it up as the HF-ViT-224 preprocessing flavour (224x224 input,
ImageNet norm, [N, 2] logits).

Synthetic blending at training time
-----------------------------------
Every batch is doubled: for each real crop `a`, we pick a random other
real crop `b`, sample a random convex-elliptical mask `m`, and construct
a fake via `a * m + b * (1-m)`. Label: fake. The mask `m` is supervision
for the auxiliary mask head (BCE vs predicted mask).

Usage
-----
    python -u -m pad.train_face_xray \\
        --data E:\\pad_datasets\\face_xray \\
        --epochs 30 --batch 32 --num-workers 4 \\
        --out pad/models/face_xray_v2.onnx

Output
------
    pad/models/face_xray_v2.onnx                 (the deepfake branch ONNX)
    pad/reports/face_xray_v2/train_log.txt       (per-epoch metrics)
"""
from __future__ import annotations
import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple


# --- lazy imports so --help works even in minimal envs --------------
def _lazy():
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    from PIL import Image
    return (np, torch, nn, F, Dataset, DataLoader, transforms,
            mobilenet_v3_small, MobileNet_V3_Small_Weights, Image)


def pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda"), f"cuda ({torch.cuda.get_device_name(0)})"
    try:
        import torch_directml as tdm
        if tdm.is_available():
            return tdm.device(), f"directml ({tdm.device_name(0).strip()})"
    except Exception:
        pass
    return __import__("torch").device("cpu"), "cpu"


# ═════════════════════════════════════════════════════════════════════
#  Dataset  -- reads the FF++ live/spoof crops layout, but uses ONLY
#              the live crops at training time; synthetic blending
#              generates the fakes on the fly.
# ═════════════════════════════════════════════════════════════════════

class LiveCropDataset:
    """Emits only REAL face crops. Synthetic fakes are generated in the
    train loop by blending two random items from this dataset."""
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, roots, split: str = "train",
                 val_fraction: float = 0.1, seed: int = 42):
        from torchvision import transforms
        # Accept either a single root or a list of roots (for combined training
        # across FF++ + Celeb-DF-v2 etc.). Roots without a live/ subdir are
        # skipped with a warning rather than failing.
        if isinstance(roots, (str, Path)):
            roots = [Path(roots)]
        else:
            roots = [Path(r) for r in roots]
        self._files = []
        for root in roots:
            live_dir = root / "live"
            if not live_dir.is_dir():
                print(f"[fxr] skip (no live/): {live_dir}", flush=True)
                continue
            count_before = len(self._files)
            for p in sorted(live_dir.iterdir()):
                if p.suffix.lower() in self.EXTS:
                    self._files.append(str(p))
            print(f"[fxr] loaded {len(self._files)-count_before} "
                  f"crops from {live_dir}", flush=True)
        if not self._files:
            raise FileNotFoundError(
                f"No live crops found under any of: "
                f"{[str(r/'live') for r in roots]}. "
                f"Run pad.preprocess_ffpp / pad.preprocess_celeb_df first.")
        random.Random(seed).shuffle(self._files)
        n = len(self._files)
        cut = max(1, int(n * val_fraction))
        self._files = self._files[cut:] if split == "train" else self._files[:cut]

        if split == "train":
            self.tx = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.15, 0.15, 0.15),
                transforms.ToTensor(),
            ])
        else:
            self.tx = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self._files)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self._files[i]).convert("RGB")
        return self.tx(img)


# ═════════════════════════════════════════════════════════════════════
#  Synthetic blending  -- the Face X-ray trick
# ═════════════════════════════════════════════════════════════════════

def _random_elliptical_mask(h: int, w: int, rng: random.Random):
    """Return an HxW mask in [0,1] representing a soft-edged ellipse
    somewhere inside the crop. Emulates the blending boundary typical
    of a face-swap (ellipse around the face region)."""
    import numpy as _np
    cy = rng.uniform(0.35 * h, 0.65 * h)
    cx = rng.uniform(0.35 * w, 0.65 * w)
    ry = rng.uniform(0.25 * h, 0.45 * h)
    rx = rng.uniform(0.25 * w, 0.45 * w)
    theta = rng.uniform(0, 3.14159)
    # Build the ellipse via standard quadric
    yy, xx = _np.meshgrid(_np.arange(h) - cy, _np.arange(w) - cx, indexing="ij")
    ct, st = _np.cos(theta), _np.sin(theta)
    xr = xx * ct + yy * st
    yr = -xx * st + yy * ct
    dist = (xr / rx) ** 2 + (yr / ry) ** 2
    # Soft feather: 0.5 at boundary, drops to 0 / 1 on either side
    mask = 1.0 / (1.0 + _np.exp((dist - 1.0) * 8.0))
    # Random feathering strength for variety
    feather = rng.uniform(2.0, 10.0)
    mask = 1.0 / (1.0 + _np.exp((dist - 1.0) * feather))
    return mask.astype(_np.float32)


def synth_blend_batch(real_batch, device, rng: random.Random):
    """Given a batch of real crops [B,3,H,W] in [0,1], produce a
    same-size batch of synthetic fakes + corresponding blending masks
    [B,1,H,W]."""
    import numpy as _np
    import torch
    B, C, H, W = real_batch.shape
    perm = torch.randperm(B, device=device)
    b = real_batch[perm]
    masks = _np.stack([_random_elliptical_mask(H, W, rng) for _ in range(B)])
    m = torch.from_numpy(masks).to(device).unsqueeze(1)  # [B,1,H,W]
    fake = real_batch * m + b * (1.0 - m)
    return fake, m


# ═════════════════════════════════════════════════════════════════════
#  Model  -- encoder + auxiliary mask head + binary classifier head.
# ═════════════════════════════════════════════════════════════════════

class FaceXrayNet:
    """Factory returns a nn.Module with two forward outputs:
       logits: [B, 2]  (binary real/fake)
       mask  : [B, 1, 224, 224]  (predicted blending mask, train-only)

    Backbone choices:
        mobilenet_v3_small  -- 576-ch @ 7x7, default (lightweight)
        efficientnet_b4     -- 1792-ch @ 7x7, stronger (~19 M params)
    """
    def __new__(cls, pretrained: bool = True,
                backbone_name: str = "mobilenet_v3_small"):
        _, torch, nn, F, _, _, _, mobilenet_v3_small, W, _ = _lazy()
        if backbone_name == "mobilenet_v3_small":
            weights = W.IMAGENET1K_V1 if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
            features = backbone.features          # [B, 576, 7, 7] for 224 input
            avgpool = backbone.avgpool
            feat_dim = backbone.classifier[0].in_features   # 576
        elif backbone_name == "efficientnet_b4":
            from torchvision.models import (
                efficientnet_b4, EfficientNet_B4_Weights,
            )
            w = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_b4(weights=w)
            features = backbone.features          # [B, 1792, 7, 7] for 224 input
            avgpool = backbone.avgpool
            feat_dim = 1792
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        fc = nn.Linear(feat_dim, 2)

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = features
                self.avgpool = avgpool
                self.cls = fc
                # Mask head: upsample feat_dim channels back to 224x224, single channel.
                self.mask_head = nn.Sequential(
                    nn.Conv2d(feat_dim, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 14
                    nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 28
                    nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 56
                    nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),  # 224
                    nn.Conv2d(16, 1, 1),
                )

            def forward(self, x, return_mask=False):
                f = self.features(x)               # [B, 576, 7, 7]
                pooled = self.avgpool(f).flatten(1)
                logits = self.cls(pooled)
                if return_mask:
                    mask = self.mask_head(f)
                    return logits, mask
                return logits

        return _Net()


# ═════════════════════════════════════════════════════════════════════
#  Train / eval loops
# ═════════════════════════════════════════════════════════════════════

def imagenet_normalise(x):
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def run_epoch(model, loader, opt, device, train: bool,
              lambda_mask: float, epoch_idx: int, total_epochs: int,
              log_every: int, rng: random.Random):
    _, torch, nn, F, _, _, _, _, _, _ = _lazy()
    model.train() if train else model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum, mask_loss_sum = 0, 0, 0.0, 0.0
    n_batches = len(loader)
    t_start = time.time()
    with torch.set_grad_enabled(train):
        for i, real in enumerate(loader):
            real = real.to(device, non_blocking=True)
            B = real.size(0)
            # Build synthetic fakes (same batch size; total becomes 2B real+fake)
            fake, mask_gt = synth_blend_batch(real, device, rng)
            x = torch.cat([real, fake], dim=0)
            y = torch.cat([torch.zeros(B, dtype=torch.long, device=device),
                           torch.ones (B, dtype=torch.long, device=device)], dim=0)
            # Mask supervision: real -> zeros, fake -> gt mask
            m = torch.cat([torch.zeros_like(mask_gt), mask_gt], dim=0)
            x_n = imagenet_normalise(x)
            logits, mask_pred = model(x_n, return_mask=True)
            loss_cls = ce(logits, y)
            loss_mask = F.binary_cross_entropy_with_logits(mask_pred, m)
            loss = loss_cls + lambda_mask * loss_mask
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += float(loss) * x.size(0)
            mask_loss_sum += float(loss_mask) * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            if train and log_every > 0 and (i + 1) % log_every == 0:
                dt = time.time() - t_start
                ips = total / max(dt, 1e-6)
                eta_min = (n_batches - i - 1) * (dt / (i + 1)) / 60.0
                print(f"  [fxr ep {epoch_idx}/{total_epochs} batch {i+1:5d}/{n_batches}]  "
                      f"loss={loss_sum/total:.4f}  mask={mask_loss_sum/total:.4f}  "
                      f"acc={correct/total:.3f}  {ips:.0f} img/s  ETA {eta_min:.1f} min",
                      flush=True)
    return (loss_sum / max(total, 1),
            mask_loss_sum / max(total, 1),
            correct / max(total, 1))


def export_onnx(model, out_path: Path):
    """Export the classifier-only forward pass (drop the mask head) so
    the ONNX matches the HF-ViT-224 preprocessing flavour expected by
    deepfake_branch._load_xray() auto-detect."""
    import torch
    model.eval().cpu()
    dummy = torch.randn(1, 3, 224, 224)

    class _CLSWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, x):
            # ImageNet normalisation inlined so the caller can pass raw [0,1]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x = (x - mean) / std
            return self.inner(x, return_mask=False)

    wrapped = _CLSWrapper(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False forces legacy TorchScript exporter (the new dynamo
    # exporter prints unicode ✅ banners that crash on Windows cp1252).
    torch.onnx.export(
        wrapped, dummy, str(out_path),
        input_names=["image"], output_names=["logits"],
        opset_version=17,
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    print(f"[fxr-export] wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, nargs="+",
                    help="one or more dataset roots, each with a `live/` "
                         "subfolder of real face crops")
    ap.add_argument("--out", default="pad/models/face_xray_v2.onnx")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--lambda-mask", type=float, default=0.5,
                    help="weight on mask-BCE loss")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--finetune-from", type=Path, default=None,
                    help="Optional .pth state-dict to warm-start model "
                         "weights from before training. Use with --lr 1e-5 "
                         "to preserve pre-trained signal.")
    ap.add_argument("--backbone", default="mobilenet_v3_small",
                    choices=["mobilenet_v3_small", "efficientnet_b4"],
                    help="feature extractor backbone")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    rng = random.Random(args.seed)

    np, torch, nn, F, _, DataLoader, _, _, _, _ = _lazy()

    data_roots = [Path(p) for p in args.data]
    train_ds = LiveCropDataset(data_roots, split="train", seed=args.seed)
    val_ds   = LiveCropDataset(data_roots, split="val",   seed=args.seed)
    print(f"[fxr] train {len(train_ds)} / val {len(val_ds)} real crops",
          f"(synthetic fakes generated each batch)", flush=True)

    dl_kwargs = {"batch_size": args.batch, "num_workers": args.num_workers}
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = args.prefetch_factor
    device, dev_label = pick_device()
    print(f"[fxr] device: {dev_label}", flush=True)
    if "cuda" in dev_label:
        dl_kwargs["pin_memory"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader   = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    model = FaceXrayNet(pretrained=not args.no_pretrained,
                        backbone_name=args.backbone).to(device)
    if args.finetune_from is not None:
        if not args.finetune_from.is_file():
            print(f"[fxr] WARNING: --finetune-from {args.finetune_from} not "
                  f"found; starting from ImageNet weights instead.",
                  flush=True)
        else:
            state = torch.load(str(args.finetune_from), map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            miss, unexpected = model.load_state_dict(state, strict=False)
            print(f"[fxr] warm-started from {args.finetune_from} "
                  f"(missing={len(miss)}, unexpected={len(unexpected)})",
                  flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Report dir for per-epoch numbers
    report_dir = Path("pad/reports/face_xray_v2")
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / "train_log.txt"
    log_f = open(log_path, "w", encoding="utf-8")

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_mask, tr_acc = run_epoch(
            model, train_loader, opt, device, train=True,
            lambda_mask=args.lambda_mask, epoch_idx=epoch,
            total_epochs=args.epochs, log_every=args.log_every, rng=rng)
        val_loss, val_mask, val_acc = run_epoch(
            model, val_loader, opt, device, train=False,
            lambda_mask=args.lambda_mask, epoch_idx=epoch,
            total_epochs=args.epochs, log_every=0, rng=rng)
        sched.step()
        dt = time.time() - t0
        line = (f"[fxr ep {epoch:03d}/{args.epochs}]  "
                f"train_loss={tr_loss:.4f} mask={tr_mask:.4f} acc={tr_acc:.3f}  "
                f"val_loss={val_loss:.4f} acc={val_acc:.3f}  lr={opt.param_groups[0]['lr']:.1e}  "
                f"({dt:.1f}s)")
        print(line, flush=True)
        log_f.write(line + "\n"); log_f.flush()
        if val_acc > best_val:
            best_val = val_acc
            # Save best .pth alongside the ONNX path so future runs can
            # warm-start via --finetune-from <this file>.
            ckpt_path = Path(args.out).with_suffix(".pth")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(ckpt_path))

    log_f.write(f"[fxr] best val acc: {best_val:.3f}\n")
    log_f.close()
    print(f"[fxr] best val acc: {best_val:.3f}", flush=True)
    export_onnx(model, Path(args.out))
    print("[fxr] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
