"""pad/train_temporal_deepfake.py -- fine-tune torchvision MC3-18 on 16-frame clips.

Inputs:
    <data>/live/*.npy    shape [16, 3, 112, 112] uint8
    <data>/spoof/*.npy   same
    (output of pad.preprocess_video_clips)

Model: torchvision.models.video.mc3_18, Kinetics-400 pretrained.
       Final FC replaced with 2-class head.

Output:
    pad/models/deepfake_temporal.onnx
      input : clip[batch, 3, 16, 112, 112] float32 normalized
      output: logits[batch, 2]
    pad/models/deepfake_temporal.pth (best-val checkpoint)

Usage:
    python -m pad.train_temporal_deepfake \\
        --data E:\\pad_datasets\\video_clips_combined \\
        --epochs 5 --batch 8 --num-workers 2

Runtime on RTX 3060 @ batch 8, ~20 clips/sec -> ~5 epochs on 50k clips = ~3.5 h.
"""
from __future__ import annotations
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np


def _lazy():
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision.models.video import mc3_18, MC3_18_Weights
    return np, torch, nn, F, Dataset, DataLoader, mc3_18, MC3_18_Weights


def pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda"), f"cuda ({torch.cuda.get_device_name(0)})"
    try:
        import torch_directml as tdm
        if tdm.is_available():
            return tdm.device(), f"directml"
    except Exception:
        pass
    return __import__("torch").device("cpu"), "cpu"


# Kinetics-400 mean/std for the pretrained MC3_18 (pytorchvideo convention)
_KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
_KINETICS_STD  = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


# MUST be a MODULE-LEVEL class so torch.multiprocessing DataLoader workers
# can pickle it under Windows spawn semantics. (Defining it inside a
# function makes it a local class, which pickle refuses to serialise
# -> AttributeError: "Can't pickle local object ...", then workers EOF.)
try:
    from torch.utils.data import Dataset as _Dataset
except ImportError:  # torch not installed -> module still importable for --help
    _Dataset = object


class ClipDataset(_Dataset):
    """Reads pre-extracted 16-frame clip tensors saved as .npy uint8
    arrays with shape [T=16, C=3, H=112, W=112]. Returns (tensor, label)."""
    MEAN = _KINETICS_MEAN
    STD  = _KINETICS_STD

    def __init__(self, files_labels, train: bool):
        self.items = list(files_labels)
        self.train = bool(train)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import torch
        p, y = self.items[idx]
        arr = np.load(p)                          # [T, C, H, W] uint8
        arr = arr.astype(np.float32) / 255.0
        mean = self.MEAN[None, :, None, None]
        std  = self.STD[None, :, None, None]
        arr = (arr - mean) / std
        # MC3-18 expects [C, T, H, W]
        arr = np.transpose(arr, (1, 0, 2, 3))
        if self.train and random.random() < 0.5:
            arr = arr[:, :, :, ::-1].copy()        # time-consistent H-flip
        return torch.from_numpy(arr).float(), int(y)


def build_dataset(root: Path):
    """Back-compat shim -- now just returns the module-level ClipDataset."""
    return ClipDataset


def collect_files(root: Path, val_fraction: float = 0.1, seed: int = 42):
    random.seed(seed)
    live = sorted((root / "live").glob("*.npy"))
    spoof = sorted((root / "spoof").glob("*.npy"))
    items = [(p, 0) for p in live] + [(p, 1) for p in spoof]
    random.shuffle(items)
    cut = max(1, int(len(items) * val_fraction))
    return items[cut:], items[:cut]


def build_model(pretrained: bool, num_classes: int = 2):
    _, torch, nn, _, _, _, mc3_18, MC3_18_Weights = _lazy()
    weights = MC3_18_Weights.KINETICS400_V1 if pretrained else None
    m = mc3_18(weights=weights)
    # Replace final FC -- mc3_18 has `fc` as a Linear(512, 400)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def run_epoch(model, loader, opt, device, train: bool, epoch_idx: int,
              total_epochs: int, log_every: int):
    _, torch, nn, _, _, _, _, _ = _lazy()
    model.train() if train else model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    n_batches = len(loader)
    t_start = time.time()
    with torch.set_grad_enabled(train):
        for i, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = ce(logits, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss) * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            if train and log_every > 0 and (i + 1) % log_every == 0:
                dt = time.time() - t_start
                cps = total / max(dt, 1e-6)
                eta = (n_batches - i - 1) * (dt / (i + 1)) / 60.0
                print(f"  [tmp ep {epoch_idx}/{total_epochs} b {i+1:4d}/{n_batches}] "
                      f"loss={loss_sum/total:.4f} acc={correct/total:.3f} "
                      f"{cps:.1f} clip/s ETA {eta:.1f} min", flush=True)
    return loss_sum / max(total, 1), correct / max(total, 1)


def export_onnx(model, out: Path):
    import torch
    model.eval().cpu()
    # MC3 expects [batch, C=3, T=16, H=112, W=112]
    dummy = torch.randn(1, 3, 16, 112, 112)
    out.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False forces the legacy TorchScript exporter; the new
    # torch.export-based exporter prints unicode (✅) banners that
    # crash on Windows cp1252 console encoding.
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["clip"], output_names=["logits"],
        dynamic_axes={"clip": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"[tmp-export] wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="dir with live/*.npy + spoof/*.npy")
    ap.add_argument("--out", type=Path, default=Path("pad/models/deepfake_temporal.onnx"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume-from", type=Path, default=None,
                    help="resume training from this .pth checkpoint. "
                         "The file is expected next to --out as "
                         "{out-stem}_epoch_NN.pth. Sets start_epoch accordingly.")
    ap.add_argument("--save-every-epoch", action="store_true",
                    help="save a checkpoint after each epoch (in addition "
                         "to the best-val one). Use with --resume-from to "
                         "pick up a crashed multi-day run without losing "
                         "progress.")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    np, torch, nn, _, _, DataLoader, _, _ = _lazy()
    torch.manual_seed(args.seed)

    # API_up v9.8: louder error when --data points at the wrong root, since
    # train_temporal_deepfake's expected layout (<data>/live/*.npy +
    # <data>/spoof/*.npy) is non-obvious. If the dir doesn't exist or
    # the live/ + spoof/ subdirs are empty, print a concrete fix-it line.
    if not args.data.is_dir():
        print(f"[tmp] FAIL: --data is not a directory: {args.data}", file=sys.stderr)
        print(f"[tmp]       expected layout: {args.data}/live/*.npy + {args.data}/spoof/*.npy",
              file=sys.stderr)
        print(f"[tmp]       generate via: python -m pad.preprocess_video_clips ...",
              file=sys.stderr)
        return 1
    live_dir  = args.data / "live"
    spoof_dir = args.data / "spoof"
    if not live_dir.is_dir() or not spoof_dir.is_dir():
        print(f"[tmp] FAIL: --data does not have the live/+spoof/ layout",
              file=sys.stderr)
        print(f"[tmp]       --data:  {args.data}", file=sys.stderr)
        print(f"[tmp]       live/:  {live_dir.is_dir()}  spoof/: {spoof_dir.is_dir()}",
              file=sys.stderr)
        print(f"[tmp]       hint: try --data E:\\pad_datasets\\video_clips_celebdf",
              file=sys.stderr)
        return 1
    ClipDataset = build_dataset(args.data)
    train_items, val_items = collect_files(args.data, seed=args.seed)
    if not train_items:
        print(f"[tmp] no clips under {args.data}", file=sys.stderr)
        print(f"[tmp]   live/ has  {sum(1 for _ in live_dir.glob('*.npy'))} *.npy files",
              file=sys.stderr)
        print(f"[tmp]   spoof/ has {sum(1 for _ in spoof_dir.glob('*.npy'))} *.npy files",
              file=sys.stderr)
        return 1
    print(f"[tmp] {len(train_items)} train / {len(val_items)} val clips",
          flush=True)
    train_ds = ClipDataset(train_items, train=True)
    val_ds   = ClipDataset(val_items, train=False)

    device, dev_label = pick_device()
    print(f"[tmp] device: {dev_label}", flush=True)
    dl_kwargs = {"batch_size": args.batch, "num_workers": args.num_workers}
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
    if "cuda" in dev_label:
        dl_kwargs["pin_memory"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader   = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    print("[tmp] building MC3-18 (Kinetics-400 pretrained) ...", flush=True)
    model = build_model(pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Resume from a prior crash -- loads weights + fast-forwards the
    # scheduler so cosine LR picks up at the right point.
    start_epoch = 1
    if args.resume_from is not None and args.resume_from.is_file():
        state = torch.load(str(args.resume_from), map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            # checkpoint dict produced by --save-every-epoch
            model.load_state_dict(state["model"])
            if "optimizer" in state:
                try: opt.load_state_dict(state["optimizer"])
                except Exception as e:
                    print(f"[tmp] warn: optimizer state didn't load: {e}", flush=True)
            start_epoch = int(state.get("epoch", 0)) + 1
            # fast-forward scheduler
            for _ in range(1, start_epoch):
                sched.step()
        else:
            # bare state_dict (best-val checkpoint)
            model.load_state_dict(state)
            print("[tmp] warn: --resume-from is a bare state_dict, epoch "
                  "counter will restart at 1 (cosine LR restarts too).",
                  flush=True)
        print(f"[tmp] resumed from {args.resume_from.name} -> epoch {start_epoch}", flush=True)

    report_dir = Path("pad/reports/deepfake_temporal")
    report_dir.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if start_epoch > 1 else "w"
    log_f = open(report_dir / "train_log.txt", log_mode, encoding="utf-8")

    best_val = 0.0
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, device,
                                    train=True, epoch_idx=epoch,
                                    total_epochs=args.epochs,
                                    log_every=args.log_every)
        val_loss, val_acc = run_epoch(model, val_loader, opt, device,
                                      train=False, epoch_idx=epoch,
                                      total_epochs=args.epochs, log_every=0)
        sched.step()
        dt = time.time() - t0
        line = (f"[tmp ep {epoch:03d}/{args.epochs}]  "
                f"train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
                f"val_loss={val_loss:.4f} acc={val_acc:.3f}  "
                f"lr={opt.param_groups[0]['lr']:.1e}  ({dt:.1f}s)")
        print(line, flush=True); log_f.write(line + "\n"); log_f.flush()
        if val_acc > best_val:
            best_val = val_acc
            pth_path = args.out.with_suffix(".pth")
            pth_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(pth_path))

        # Optional per-epoch checkpoint -- paired with --resume-from lets
        # a multi-day run pick up where it crashed without losing progress.
        if args.save_every_epoch:
            ckpt_path = args.out.with_name(
                f"{args.out.stem}_epoch_{epoch:02d}.pth")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": opt.state_dict(),
                "val_acc":   val_acc,
                "train_acc": tr_acc,
            }, str(ckpt_path))

    log_f.write(f"[tmp] best val acc: {best_val:.3f}\n"); log_f.close()
    print(f"[tmp] best val acc: {best_val:.3f}", flush=True)
    export_onnx(model, args.out)
    print("[tmp] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
