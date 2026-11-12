"""prepare_celeba_spoof.py
=========================

Flatten the nested CelebA-Spoof directory layout into the simple
`live/` + `spoof/` structure that `pad.train_patch_depth` expects.

CelebA-Spoof ships as:
    CelebA_Spoof/Data/train/<subject_id>/live/*.jpg
    CelebA_Spoof/Data/train/<subject_id>/spoof/*.jpg
    CelebA_Spoof/Data/test/<subject_id>/live/*.jpg
    CelebA_Spoof/Data/test/<subject_id>/spoof/*.jpg

This script creates:
    <OUT>/live/<subject_id>_<filename>.jpg
    <OUT>/spoof/<subject_id>_<filename>.jpg

By default uses NTFS hard links (no disk duplication). Pass --copy
to physically copy the files (slower, uses an extra ~60 GB).

Usage:
    python -m pad.prepare_celeba_spoof \
        --src D:\\pad_datasets\\CelebA-Spoof\\CelebA_Spoof \
        --out D:\\pad_datasets\\patch_depth
    python -m pad.prepare_celeba_spoof --src ... --out ... --copy
    python -m pad.prepare_celeba_spoof --src ... --out ... --limit 10000
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _link_or_copy(src: Path, dst: Path, copy: bool):
    if dst.exists():
        return False
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)
    return True


def walk_subjects(data_root: Path, splits=("train",)):
    """Yield (subject_id, label, src_path) for every image."""
    for split in splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            print(f"[prep] skipping missing split: {split_dir}")
            continue
        for subject_dir in sorted(split_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            sid = subject_dir.name
            for label in ("live", "spoof"):
                cls_dir = subject_dir / label
                if not cls_dir.is_dir():
                    continue
                for f in sorted(cls_dir.iterdir()):
                    if f.suffix.lower() in IMG_EXTS:
                        yield sid, label, f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="path to the extracted CelebA_Spoof root (contains Data/ and metas/)")
    ap.add_argument("--out", required=True,
                    help="output directory for the flat live/ + spoof/ layout")
    ap.add_argument("--splits", default="train",
                    help="comma-separated subset of train,test,val (default: train)")
    ap.add_argument("--copy", action="store_true",
                    help="copy files instead of NTFS hard-linking")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total images per class (useful for quick tests)")
    args = ap.parse_args()

    src = Path(args.src)
    data_root = src / "Data"
    if not data_root.is_dir():
        print(f"[prep] ERROR: expected {data_root} to exist.")
        print(f"       Point --src at the directory that contains Data/ and metas/.")
        return 1

    out = Path(args.out)
    (out / "live").mkdir(parents=True, exist_ok=True)
    (out / "spoof").mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    print(f"[prep] src={src}")
    print(f"[prep] out={out}")
    print(f"[prep] splits={splits}  mode={'COPY' if args.copy else 'HARDLINK'}")

    counts = {"live": 0, "spoof": 0}
    skipped = 0
    processed_any = False

    try:
        from tqdm import tqdm
        bar = tqdm(unit="img", mininterval=1.0)
    except ImportError:
        bar = None

    for sid, label, src_path in walk_subjects(data_root, splits=splits):
        processed_any = True
        if args.limit is not None and counts[label] >= args.limit:
            continue
        dst = out / label / f"{sid}_{src_path.name}"
        if _link_or_copy(src_path, dst, copy=args.copy):
            counts[label] += 1
        else:
            skipped += 1
        if bar is not None:
            bar.update(1)
            bar.set_postfix_str(f"live={counts['live']} spoof={counts['spoof']}")

    if bar is not None:
        bar.close()

    if not processed_any:
        print("[prep] ERROR: no subject folders found under Data/.")
        print(f"       Inspect: {data_root}")
        return 1

    print(f"[prep] done. live={counts['live']} spoof={counts['spoof']} "
          f"skipped(existing)={skipped}")
    print(f"[prep] ready to train:")
    print(f"       python -m pad.train_patch_depth --data {out} --epochs 20 --batch 64")
    return 0


if __name__ == "__main__":
    sys.exit(main())
