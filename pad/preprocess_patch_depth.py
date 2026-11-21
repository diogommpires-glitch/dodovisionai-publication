"""Pre-compute resized uint8 tensors for the patch_depth training set
into a single memory-mapped binary. Eliminates 494k per-image file
opens during training — HDD reads the flat file near-sequentially.

Expected 4-5× training speedup vs opening JPEGs per-sample.

Input:  <DATA>/live/*.jpg and <DATA>/spoof/*.jpg (output of prepare_celeba_spoof)
Output: <OUT>/images.bin  (shape N×224×224×3, uint8, 74 GB for full CelebA-Spoof)
        <OUT>/labels.npy  (int32, length N)
        <OUT>/paths.txt   (one path per line, for debugging)

Usage:
    python -m pad.preprocess_patch_depth \
        --data D:\\pad_datasets\\patch_depth \
        --out  D:\\pad_datasets\\patch_depth_mmap
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
RES = 224


def collect(root: Path):
    items = []
    for label, sub in ((0, "live"), (1, "spoof")):
        d = root / sub
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                items.append((str(p), label))
    return items


def _process_one(args):
    idx, path, arr_path = args
    from PIL import Image
    try:
        img = Image.open(path).convert("RGB").resize((RES, RES), Image.BILINEAR)
        arr = np.memmap(arr_path, dtype=np.uint8, mode="r+",
                        shape=(1, RES, RES, 3), offset=idx * RES * RES * 3)
        arr[0] = np.asarray(img, dtype=np.uint8)
        del arr
        return idx, True, None
    except Exception as e:
        return idx, False, str(e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel image readers (2-8 is sweet spot on HDD)")
    args = ap.parse_args()

    data_root = Path(args.data)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    items = collect(data_root)
    n = len(items)
    if n == 0:
        print(f"[preprocess] no images in {data_root}")
        return 1

    print(f"[preprocess] found {n:,} images")
    print(f"[preprocess] output: {out_root}")
    total_gb = n * RES * RES * 3 / 1e9
    print(f"[preprocess] allocating {total_gb:.1f} GB flat file...")

    arr_path = str(out_root / "images.bin")
    # Pre-allocate the flat binary.
    with open(arr_path, "wb") as fh:
        fh.seek(n * RES * RES * 3 - 1)
        fh.write(b"\0")

    labels = np.asarray([lbl for _, lbl in items], dtype=np.int32)
    np.save(out_root / "labels.npy", labels)
    with open(out_root / "paths.txt", "w", encoding="utf-8") as fh:
        for p, _ in items:
            fh.write(p + "\n")

    # Process in chunks so we don't spawn 494k Pool tasks at once.
    from multiprocessing import Pool
    from tqdm import tqdm

    task_args = [(i, items[i][0], arr_path) for i in range(n)]
    t0 = time.time()
    failed = []
    with Pool(args.workers) as pool:
        for idx, ok, err in tqdm(
            pool.imap_unordered(_process_one, task_args, chunksize=32),
            total=n, desc="preprocess", unit="img", mininterval=5.0,
        ):
            if not ok:
                failed.append((idx, err))

    dt = time.time() - t0
    print(f"[preprocess] wrote {n - len(failed):,} images in {dt/60:.1f} min "
          f"({n/dt:.0f} img/s)")
    if failed:
        print(f"[preprocess] {len(failed)} failures")
        with open(out_root / "failed.txt", "w", encoding="utf-8") as fh:
            for i, e in failed:
                fh.write(f"{i}\t{items[i][0]}\t{e}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
