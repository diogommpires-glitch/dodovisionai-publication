"""Measure your actual image throughput from the patch_depth/live folder.

This is exactly what the DataLoader does per batch: random-order
image opens + JPEG decode + resize + tensor conversion. Tells us
whether your HDD can feed the GPU or not.

Usage:
    python -m pad.bench_hdd --data D:\\pad_datasets\\patch_depth --n 500
"""
from __future__ import annotations
import argparse
import random
import time
from pathlib import Path


def main() -> int:
    """CLI: micro-benchmark sequential read throughput on the patch_depth mmap. Sanity-checks that disk I/O is not the bottleneck before launching a long training run."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    from PIL import Image
    from torchvision import transforms
    import torch

    root = Path(args.data)
    paths = []
    for sub in ("live", "spoof"):
        d = root / sub
        if d.is_dir():
            for p in d.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    paths.append(p)
    if not paths:
        print(f"[bench] no images found in {root}")
        return 1
    print(f"[bench] total paths: {len(paths)}")

    random.seed(0)
    random.shuffle(paths)
    sample = paths[:args.n]

    tx = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # First pass — purely I/O + decode (no shuffle; closest to DataLoader)
    t0 = time.time()
    for p in sample:
        img = Image.open(p).convert("RGB")
        _ = tx(img)
    dt = time.time() - t0
    ips_cold = args.n / dt
    print(f"[bench] cold run: {args.n} imgs in {dt:.1f}s  →  {ips_cold:.1f} img/s")

    # Second pass — repeat, OS file cache may help
    t0 = time.time()
    for p in sample:
        img = Image.open(p).convert("RGB")
        _ = tx(img)
    dt = time.time() - t0
    ips_warm = args.n / dt
    print(f"[bench] warm run: {args.n} imgs in {dt:.1f}s  →  {ips_warm:.1f} img/s")

    # Implied training time at batch 64
    bps_cold = ips_cold / 64
    bps_warm = ips_warm / 64
    ep_imgs = len(paths) * 0.9       # 90% train split
    ep_cold = ep_imgs / ips_cold / 60
    ep_warm = ep_imgs / ips_warm / 60
    print()
    print(f"[estimate] batches/sec cold={bps_cold:.1f}  warm={bps_warm:.1f}")
    print(f"[estimate] epoch on {int(ep_imgs)} train imgs: "
          f"cold={ep_cold:.1f} min, warm={ep_warm:.1f} min")
    print(f"[estimate] 20-epoch total (cold 1st, warm rest): "
          f"{ep_cold + 19 * ep_warm:.0f} min = {(ep_cold + 19 * ep_warm)/60:.1f} h")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
