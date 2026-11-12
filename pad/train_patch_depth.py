"""train_patch_depth.py
=======================

Train the MobileNetV3-small patch+depth classifier that the
`pad.branches.patch_depth_branch` loads from `pad/models/patch_depth.onnx`.

Device selection (in order of preference):
    CUDA (if torch-CUDA installed) -> DirectML (torch-directml) -> CPU

Dataset layout expected:
    <DATA>/live/*.{jpg,png,jpeg,bmp}
    <DATA>/spoof/*.{jpg,png,jpeg,bmp}

If --bootstrap is passed AND the data dir is empty, the script
synthesizes a tiny starter set using pad.tests.fixtures so you can
end-to-end-test the pipeline (resulting model is TOY; replace before
using on real data).

Usage:
    python -m pad.train_patch_depth --data pad/datasets/my_spoof
    python -m pad.train_patch_depth --bootstrap             # tier A smoke test
    python -m pad.train_patch_depth --data <path> --epochs 20 --batch 64

Exports to pad/models/patch_depth.onnx on completion.
"""
from __future__ import annotations
import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# --- lazy imports so argparse --help works in minimal envs ---
def _lazy():
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    from PIL import Image
    return (torch, nn, Dataset, DataLoader, transforms,
            mobilenet_v3_small, MobileNet_V3_Small_Weights, Image)


def pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    try:
        import torch_directml as tdm
        if tdm.is_available():
            return tdm.device(), f"directml ({tdm.device_name(0).strip()})"
    except Exception:
        pass
    return torch.device("cpu"), "cpu"


def _identity_from_filename(fname: str) -> str:
    """Extract the identity tag from a CelebA-Spoof-style filename like
    '10000_019598.jpg' -> '10000'. Falls back to the bare stem when the
    underscore convention is absent (cross-source datasets).

    v9.7.0 — added to fix the train/val IDENTITY LEAKAGE bug in v9.6.x.
    Pre-fix split was index-based after a global shuffle, which placed
    the same person's patches in BOTH train and val. v_acc trivially
    hit 0.99 because the network learned identity, not liveness.
    """
    import re
    base = Path(fname).stem
    m = re.match(r"^(\d+)_\d+$", base)  # e.g. 10000_019598
    if m:
        return m.group(1)
    # Generic fallback: take the first underscore-separated token
    return base.split("_", 1)[0] if "_" in base else base


def _identity_stratified_split(identity_per_index, val_fraction: float, seed: int = 42):
    """Given an iterable of identity-tags (one per dataset index), return
    (train_indices, val_indices) so that NO identity appears in both
    splits. The split is deterministic for a fixed seed.
    """
    import random as _r
    by_id: dict = {}
    for i, ident in enumerate(identity_per_index):
        by_id.setdefault(ident, []).append(i)
    ids = sorted(by_id.keys())
    _r.Random(seed).shuffle(ids)
    cut = max(1, int(len(ids) * val_fraction))
    val_ids = set(ids[:cut])
    train_idx = []
    val_idx = []
    for ident, idxs in by_id.items():
        if ident in val_ids:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)
    return train_idx, val_idx


class MmapSpoofDataset:
    """Reads from pad.preprocess_patch_depth's flat mmap binary.
    4-5x faster than FolderSpoofDataset on HDDs.

    v9.7.0 — `identity_stratified=True` (default) groups patches by
    identity (parsed from the sibling `paths.txt`) and splits GROUPS,
    not patch indices. Required to make val_acc reflect generalization
    rather than identity overfit. Set False to reproduce v9.6.x.
    """
    def __init__(self, root, split: str = "train", val_fraction: float = 0.1,
                 identity_stratified: bool = True, seed: int = 42):
        import numpy as _np
        from torchvision import transforms
        root = Path(root)
        self._mmap_path = str(root / "images.bin")
        self._labels = _np.load(root / "labels.npy")
        n = len(self._labels)
        paths_file = root / "paths.txt"

        used_stratified = False
        if identity_stratified and paths_file.exists():
            try:
                with open(paths_file, "r", encoding="utf-8") as fh:
                    paths = [ln.strip() for ln in fh if ln.strip()]
                if len(paths) == n:
                    idents = [_identity_from_filename(p) for p in paths]
                    train_idx, val_idx = _identity_stratified_split(
                        idents, val_fraction=val_fraction, seed=seed)
                    self._indices = train_idx if split == "train" else val_idx
                    used_stratified = True
                    n_train_ids = len(set(idents[i] for i in train_idx))
                    n_val_ids = len(set(idents[i] for i in val_idx))
                    overlap = set(idents[i] for i in train_idx) & set(idents[i] for i in val_idx)
                    print(f"[MmapSpoofDataset] identity-stratified split: "
                          f"train={n_train_ids} ids ({len(train_idx)} imgs)  "
                          f"val={n_val_ids} ids ({len(val_idx)} imgs)  "
                          f"overlap={len(overlap)} ids (must be 0)")
                else:
                    print(f"[MmapSpoofDataset] paths.txt size mismatch "
                          f"({len(paths)} vs labels {n}); falling back to index split")
            except Exception as e:
                print(f"[MmapSpoofDataset] identity split failed ({e}); "
                      f"falling back to index split")

        if not used_stratified:
            # Legacy v9.6.x behaviour: index-based shuffle then split.
            # Identity leakage warning is unavoidable here.
            import random as _r
            idx = list(range(n))
            _r.Random(seed).shuffle(idx)
            cut = max(1, int(n * val_fraction))
            self._indices = idx[cut:] if split == "train" else idx[:cut]
            print(f"[MmapSpoofDataset] WARNING: index-based split — "
                  f"identity leakage likely. Pass identity_stratified=True.")
        self._res = 224

        if split == "train":
            self.tx = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
            ])
        else:
            self.tx = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        import numpy as _np
        k = self._indices[i]
        arr = _np.memmap(self._mmap_path, dtype=_np.uint8, mode="r",
                         shape=(self._res, self._res, 3),
                         offset=int(k) * self._res * self._res * 3)
        # Copy out of mmap before yielding (prevents mmap handle leaks in workers)
        img_np = _np.array(arr, copy=True)
        return self.tx(img_np), int(self._labels[k])


class FolderSpoofDataset:
    """Reads live/ and spoof/ subdirectories; returns (img_tensor, label).

    Windows-safe: no module refs or lambdas stored as instance attrs
    (spawn multiprocessing picklability requires this).

    v9.7.0 — `identity_stratified=True` (default) groups files by
    identity (parsed via `_identity_from_filename`) and splits GROUPS.
    """
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: Path, split: str = "train", val_fraction: float = 0.1,
                 identity_stratified: bool = True, seed: int = 42):
        from torchvision import transforms
        self._files = []
        for label, sub in ((0, "live"), (1, "spoof")):
            d = root / sub
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in self.EXTS:
                    self._files.append((str(p), label))
        n = len(self._files)
        used_stratified = False
        if identity_stratified and self._files:
            try:
                idents = [_identity_from_filename(p) for p, _ in self._files]
                train_idx, val_idx = _identity_stratified_split(
                    idents, val_fraction=val_fraction, seed=seed)
                pick = train_idx if split == "train" else val_idx
                self._files = [self._files[i] for i in pick]
                used_stratified = True
                n_train_ids = len(set(idents[i] for i in train_idx))
                n_val_ids = len(set(idents[i] for i in val_idx))
                overlap = set(idents[i] for i in train_idx) & set(idents[i] for i in val_idx)
                print(f"[FolderSpoofDataset] identity-stratified split: "
                      f"train={n_train_ids} ids ({len(train_idx)} imgs)  "
                      f"val={n_val_ids} ids ({len(val_idx)} imgs)  "
                      f"overlap={len(overlap)} ids (must be 0)")
            except Exception as e:
                print(f"[FolderSpoofDataset] identity split failed ({e}); "
                      f"falling back to index split")

        if not used_stratified:
            random.Random(seed).shuffle(self._files)
            cut = max(1, int(n * val_fraction))
            self._files = self._files[cut:] if split == "train" else self._files[:cut]
            print(f"[FolderSpoofDataset] WARNING: index-based split — "
                  f"identity leakage likely. Pass identity_stratified=True.")

        if split == "train":
            self.tx = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
            ])
        else:
            self.tx = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

    def __len__(self): return len(self._files)

    def __getitem__(self, i):
        from PIL import Image
        p, y = self._files[i]
        img = Image.open(p).convert("RGB")
        return self.tx(img), y


def bootstrap_fixtures(out_dir: Path, n_per_class: int = 200) -> None:
    """Use pad.tests.fixtures generators to synthesize a starter
    live+spoof set when the user hasn't downloaded real data yet.
    PRINT fixtures go into spoof/; live fixtures into live/.
    """
    from pad.tests.fixtures import make_live_crop, make_print_crop, make_cutout_crop, make_replay_crop
    import cv2
    (out_dir / "live").mkdir(parents=True, exist_ok=True)
    (out_dir / "spoof").mkdir(parents=True, exist_ok=True)
    for i in range(n_per_class):
        cv2.imwrite(str(out_dir / "live" / f"live_{i:04d}.jpg"),
                    make_live_crop(size=160, seed=i))
        kind = i % 3
        if kind == 0:
            img = make_print_crop(size=160, seed=i)
        elif kind == 1:
            img = make_cutout_crop(size=160, seed=i)
        else:
            img = make_replay_crop(size=160, seed=i)
        cv2.imwrite(str(out_dir / "spoof" / f"spoof_{i:04d}.jpg"), img)
    print(f"[bootstrap] wrote {n_per_class} live + {n_per_class} spoof "
          f"synthetic images to {out_dir}")


def build_model(pretrained: bool = True):
    torch, nn, _, _, _, mobilenet_v3_small, W, _ = _lazy()
    weights = W.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    # Swap the 1000-class head for our 2-class head
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, 2)
    return model


def run_epoch(model, loader, loss_fn, opt, device, train: bool,
              epoch_idx: int = 0, total_epochs: int = 0, log_every: int = 50):
    import torch
    import time as _time
    if train: model.train()
    else: model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    n_batches = len(loader)
    t_start = _time.time()
    with torch.set_grad_enabled(train):
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += float(loss) * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            # Per-batch progress print (train only, to avoid noise on val)
            if train and log_every > 0 and (i + 1) % log_every == 0:
                elapsed = _time.time() - t_start
                ips = total / elapsed
                batches_left = n_batches - (i + 1)
                eta_min = batches_left * (elapsed / (i + 1)) / 60.0
                print(f"  [ep {epoch_idx}/{total_epochs} batch {i+1:5d}/{n_batches}]  "
                      f"loss={loss_sum/total:.4f}  acc={correct/total:.3f}  "
                      f"{ips:.0f} img/s  ETA {eta_min:.1f} min",
                      flush=True)
    return loss_sum / max(total, 1), correct / max(total, 1)


def export_onnx(model, device, out_path: Path):
    import torch, sys, io as _io
    # v9.8 fix: torch.onnx's verbose-print emits a U+2705 (white check mark)
    # checkmark on success, which crashes when stdout is piped through
    # cp1252 (Windows default). Wrap stdout/stderr with a UTF-8 writer
    # that replaces unencodable chars rather than raising. Also save the
    # state_dict FIRST so even if ONNX export blows up, we can recover.
    pth_path = out_path.with_suffix(".pth")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(model.state_dict(), pth_path)
        print(f"[export] state_dict saved -> {pth_path} "
              f"({pth_path.stat().st_size/1e6:.1f} MB)", flush=True)
    except Exception as e:
        print(f"[export] WARN: state_dict save failed: {e}", flush=True)

    # Replace cp1252 stdout/stderr with UTF-8 to survive emoji prints.
    if hasattr(sys.stdout, "buffer"):
        try:
            sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                           errors="replace", line_buffering=True)
            sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                           errors="replace", line_buffering=True)
        except Exception:
            pass

    model.eval().cpu()
    dummy = torch.randn(1, 3, 224, 224)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            input_names=["input"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            dynamo=False,  # legacy TorchScript exporter -- avoids dynamo emoji prints
        )
        print(f"[export] wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)",
              flush=True)
    except Exception as e:
        print(f"[export] ONNX export failed: {type(e).__name__}: {e}\n"
              f"  -> .pth checkpoint at {pth_path} is recoverable via "
              f"scripts/export_pth_to_onnx.py", flush=True)
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="pad/datasets/patch_depth",
                    help="root with live/ and spoof/ subfolders")
    # The branch loader (pad.branches.patch_depth_branch) expects
    # `patch_depth_fas.onnx`. We write both names for maximum compat.
    ap.add_argument("--out", default="pad/models/patch_depth_fas.onnx",
                    help="ONNX export path (branch expects patch_depth_fas.onnx)")
    ap.add_argument("--epochs", type=int, default=20)
    # v9.10: was 64; bumped to 256 to match the max-utilization profile.
    # MobileNetV3-small at AMP bf16 fits comfortably in 5090 24 GB VRAM
    # at this batch (~3 GB peak). Override down for sub-12 GB GPUs.
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    # v9.10: was 0 (Windows multi-proc avoidance circa v9.6). On modern
    # torch + 24-core CPUs, 12 workers is safe and saturates the
    # PIL augmentation pipeline. Set --num-workers 0 to revert.
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--bootstrap", action="store_true",
                    help="synthesize fixtures into --data if empty")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--log-every", type=int, default=50,
                    help="print per-batch progress every N batches (0 = only per-epoch)")
    ap.add_argument("--prefetch-factor", type=int, default=4,
                    help="batches pre-loaded per worker (only when num-workers>0)")
    ap.add_argument("--identity-stratified", dest="identity_stratified",
                    action="store_true", default=True,
                    help="(default ON, v9.7.0+) split train/val by identity, "
                         "preventing the index-shuffle leakage that gave "
                         "v9.6.x its inflated val_acc 0.998")
    ap.add_argument("--no-identity-stratified", dest="identity_stratified",
                    action="store_false",
                    help="reproduce v9.6.x leaky split (NOT RECOMMENDED)")
    ap.add_argument("--split-seed", type=int, default=42,
                    help="seed for the deterministic identity split")
    ap.add_argument("--val-fraction", type=float, default=0.1,
                    help="held-out val identity fraction (default 0.1)")
    args = ap.parse_args()

    torch, nn, _, DataLoader, _, _, _, _ = _lazy()

    data_root = Path(args.data)
    # Accept either the mmap layout (images.bin + labels.npy) OR the
    # classic folder layout (live/ + spoof/). Only fall through to the
    # bootstrap / error path when neither is present.
    has_mmap = ((data_root / "images.bin").exists()
                and (data_root / "labels.npy").exists())
    has_folders = False
    if (data_root / "live").is_dir() and (data_root / "spoof").is_dir():
        has_folders = (any((data_root / "live").iterdir())
                       and any((data_root / "spoof").iterdir()))
    if not (has_mmap or has_folders):
        if args.bootstrap:
            bootstrap_fixtures(data_root, n_per_class=200)
        else:
            print(f"[error] no data in {data_root}: need either "
                  f"images.bin+labels.npy (mmap) or live/+spoof/ (folder)")
            print("        add images yourself OR pass --bootstrap for synthetic fixtures")
            return 1

    device, dev_label = pick_device()
    print(f"[train] device: {dev_label}")

    # Auto-detect mmap preprocessed layout (pad.preprocess_patch_depth).
    # If --data/images.bin + labels.npy exist, use MmapSpoofDataset for
    # the HDD-friendly fast path.
    ds_kwargs = dict(
        val_fraction=args.val_fraction,
        identity_stratified=args.identity_stratified,
        seed=args.split_seed,
    )
    if (data_root / "images.bin").exists() and (data_root / "labels.npy").exists():
        print(f"[train] using MmapSpoofDataset from {data_root / 'images.bin'}")
        train_ds = MmapSpoofDataset(data_root, split="train", **ds_kwargs)
        val_ds = MmapSpoofDataset(data_root, split="val", **ds_kwargs)
    else:
        print(f"[train] using FolderSpoofDataset from {data_root}")
        train_ds = FolderSpoofDataset(data_root, split="train", **ds_kwargs)
        val_ds = FolderSpoofDataset(data_root, split="val", **ds_kwargs)
    print(f"[train] {len(train_ds)} train / {len(val_ds)} val images")

    # HDD-friendly settings: persistent workers + prefetch so HDD
    # seeks amortise across epochs. pin_memory speeds CPU->GPU transfer.
    _dl_kwargs = {"batch_size": args.batch, "num_workers": args.num_workers}
    if args.num_workers > 0:
        _dl_kwargs["persistent_workers"] = True
        _dl_kwargs["prefetch_factor"] = args.prefetch_factor
    if "cuda" in dev_label:
        _dl_kwargs["pin_memory"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **_dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **_dl_kwargs)

    model = build_model(pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, opt, device,
                                    train=True, epoch_idx=epoch,
                                    total_epochs=args.epochs,
                                    log_every=args.log_every)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, opt, device,
                                      train=False, epoch_idx=epoch,
                                      total_epochs=args.epochs, log_every=0)
        dt = time.time() - t0
        print(f"[epoch {epoch:03d}/{args.epochs}]  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
              f"({dt:.1f}s)")
        if val_acc > best_val:
            best_val = val_acc

    print(f"[train] best val acc: {best_val:.3f}")
    out_primary = Path(args.out)
    export_onnx(model, device, out_primary)
    # Also write the alternate filename so both branch variants find it
    alt = out_primary.with_name("patch_depth.onnx" if out_primary.name == "patch_depth_fas.onnx"
                                 else "patch_depth_fas.onnx")
    try:
        import shutil
        shutil.copy(out_primary, alt)
        print(f"[export] copied to {alt}")
    except Exception as e:
        print(f"[export] skip copy to {alt}: {e}")
    print(f"[train] done. Swap FF_PAD_CNN=1 and restart main.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
