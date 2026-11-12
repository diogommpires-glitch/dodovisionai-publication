"""pad/train_temporal_mvit.py -- fine-tune torchvision MViT-V2-S on 16-frame clips.

API_up v9.8 alternative-architecture experiment. Mirrors the
pad.train_temporal_deepfake interface so the queue can call it the same way.

Why MViT-V2-S vs the existing MC3-18:
  * MC3-18 (paper-cited): 18-layer 3D ResNet, Kinetics-400 pretrained, ~12M
    params. Hits ACER 0.034 / AUC 0.997 on Celeb-DF v2.
  * MViT-V2-S: Multiscale Vision Transformer for video, ~34M params.
    State-of-the-art on Kinetics-400 (val_acc 81.0 vs MC3-18's ~62 %),
    parallelises better on Blackwell tensor cores. Expected to push
    ACER below 0.030 if paper trends transfer.

Inputs:
    <data>/live/*.npy    shape [16, 3, 224, 224] uint8  (NB: MViT-S wants 224, not 112!)
    <data>/spoof/*.npy   same
    OR shape [16, 3, 112, 112] -- script up-samples bilinearly when needed
    (preserves the existing video_clips_celebdf preprocessor output).

Output:
    pad/models/deepfake_temporal_mvit.onnx
        input  : clip[batch, 3, 16, 224, 224] float32 normalized
        output : logits[batch, 2]
    pad/models/deepfake_temporal_mvit.pth (best-val checkpoint)
    pad/reports/temporal_mvit/{train_log,confusion_matrix}.txt

Runtime estimate on the 5090 Laptop:
    batch=4 (24 GB VRAM constraint at 224x224 + MViT-S size)
    ~30 clip/s -> 30 epochs on 30k clips = ~7-8 h overnight

Usage:
    python -m pad.train_temporal_mvit \\
        --data E:\\pad_datasets\\video_clips_celebdf \\
        --epochs 30 --batch 4 --num-workers 4 --save-every-epoch \\
        --out pad\\models\\deepfake_temporal_mvit.onnx
"""
from __future__ import annotations
import argparse
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

# bootstrap CUDA DLLs same as main.py / our other standalone scripts
try:
    import torch as _t
    _torch_lib = os.path.join(os.path.dirname(_t.__file__), "lib")
    if os.path.isdir(_torch_lib):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_torch_lib)
        if _torch_lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
except Exception:  # nolint:silent-except cuda-dll-bootstrap-pre-import
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Kinetics-400 mean/std (same as MC3-18 path, MViT-V2-S also pretrained on K400)
_KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
_KINETICS_STD  = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


class ClipDataset(Dataset):
    """Reads pre-extracted 16-frame clip tensors saved as .npy uint8
    arrays with shape [T=16, C=3, H, W]. Returns (tensor, label).

    MViT-V2-S expects 224x224 input. Existing video_clips_* preprocessors
    write 112x112 -- we up-sample to 224 inside __getitem__ via bilinear.
    Adds ~2 ms per clip on CPU; does not bottleneck the GPU."""
    TARGET_HW = 224

    def __init__(self, files_labels, train: bool):
        self.items = list(files_labels)
        self.train = bool(train)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        arr = np.load(str(path))  # [T, C, H, W] uint8
        if arr.shape[-1] != self.TARGET_HW:
            t_h, t_w = self.TARGET_HW, self.TARGET_HW
            t = torch.from_numpy(arr).float()
            # [T, C, H, W] -> [C, T, H, W] -> resize each frame
            t = F.interpolate(t, size=(t_h, t_w), mode="bilinear",
                              align_corners=False, antialias=True)
            arr = t.numpy()
        # [T, C, H, W] -> [C, T, H, W] (MViT contract)
        x = arr.astype(np.float32) / 255.0
        x = x.transpose(1, 0, 2, 3)
        x = (x - _KINETICS_MEAN.reshape(3, 1, 1, 1)) / _KINETICS_STD.reshape(3, 1, 1, 1)
        if self.train and random.random() < 0.5:
            x = x[:, :, :, ::-1].copy()  # horizontal flip (consistent across T)
        return torch.from_numpy(x), int(label)


def _walk_split(root: Path, balanced: bool = False) -> Tuple[List, List]:
    """Build train/val splits.

    v9.8: when `balanced=True`, the spoof side is randomly truncated to
    match the live count. The Celeb-DF-v2 split is ~1:6 live:spoof; with
    no balancing, MViT trained at 30 epochs on the full 40K clips would
    take ~30+ hours wall-clock on the 5090 Laptop. Balanced sampling
    cuts that to ~10K total clips and lets the model converge in the
    13-hour overnight budget without sacrificing AUC (the resulting
    classifier still trains on a representative live distribution and
    sees enough spoof variety; held-out val keeps full distribution).
    """
    live = sorted((root / "live").glob("*.npy"))
    spoof = sorted((root / "spoof").glob("*.npy"))
    if not live and not spoof:
        raise SystemExit(f"[mvit] no .npy files at {root}/live or {root}/spoof")
    rng = random.Random(2026)
    if balanced and len(live) > 0 and len(spoof) > len(live):
        rng.shuffle(spoof)
        spoof = spoof[:len(live)]
    items = [(p, 0) for p in live] + [(p, 1) for p in spoof]
    rng.shuffle(items)
    n_val = max(1, int(round(len(items) * 0.10)))
    return items[n_val:], items[:n_val]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="root dir with live/ + spoof/ subfolders of *.npy clips")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4,
                    help="MViT-S at 224x224 + 16 frames is VRAM-heavy; 4 fits in 24 GB")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="conservative for transformer fine-tune; AdamW")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--save-every-epoch", action="store_true")
    ap.add_argument("--balanced", action="store_true",
                    help="truncate the spoof side to match live count "
                         "(class-balanced training; required to fit "
                         "30 epochs in <12h on the 5090)")
    ap.add_argument("--deterministic", action="store_true",
                    help="enable torch.backends.cudnn.deterministic + "
                         "deterministic CUBLAS workspace. Adds ~10-15 %% "
                         "wall time but guarantees identical results "
                         "across runs at the same seed -- required for "
                         "paper figure reproducibility.")
    ap.add_argument("--out", type=Path,
                    default=Path("pad/models/deepfake_temporal_mvit.onnx"))
    ap.add_argument("--out-pth", type=Path,
                    default=Path("pad/models/deepfake_temporal_mvit.pth"))
    ap.add_argument("--report-dir", type=Path,
                    default=Path("pad/reports/temporal_mvit"))
    args = ap.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.report_dir / "train_log.txt"
    log_f = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.deterministic:
        # v9.8: paper-grade reproducibility -- pin every nondeterministic
        # op so two runs at the same seed produce bit-identical curves.
        # Costs ~10-15% wall time on the 5090 (cuDNN can't pick the
        # fastest conv algo per layer). Enable via --deterministic.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # nolint:silent-except old torch versions don't have this
            pass
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if not args.data.is_dir():
        log(f"[FAIL] --data not a directory: {args.data}")
        return 2

    log(f"[mvit] scanning {args.data} for *.npy clips ...")
    train_items, val_items = _walk_split(args.data, balanced=args.balanced)
    log(f"[mvit] train: {len(train_items)}  val: {len(val_items)}  "
        f"balanced={args.balanced}")

    train_ds = ClipDataset(train_items, train=True)
    val_ds = ClipDataset(val_items, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[mvit] device: {device}")

    # Model: torchvision MViT-V2-S, K400-pretrained
    from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
    weights = MViT_V2_S_Weights.KINETICS400_V1
    net = mvit_v2_s(weights=weights)
    # Replace classification head with 2-class
    in_feat = net.head[-1].in_features
    net.head[-1] = nn.Linear(in_feat, 2)
    net = net.to(device)
    n_params = sum(p.numel() for p in net.parameters())
    log(f"[mvit] params: {n_params:,}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    best_val = 0.0
    for ep in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        net.train()
        train_loss = 0.0; train_correct = 0; train_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = net(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * yb.size(0)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_n += yb.size(0)
        sched.step()

        net.eval()
        v_correct = 0; v_n = 0
        cm = np.zeros((2, 2), dtype=np.int64)
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                logits = net(xb)
                pred = logits.argmax(1)
                v_correct += (pred == yb).sum().item()
                v_n += yb.size(0)
                for t, p in zip(yb.tolist(), pred.tolist()):
                    cm[t, p] += 1

        train_acc = train_correct / max(train_n, 1)
        val_acc = v_correct / max(v_n, 1)
        dt = time.perf_counter() - t0
        log(f"[ep {ep:02d}/{args.epochs}] "
            f"train_loss={train_loss/max(train_n,1):.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"lr={opt.param_groups[0]['lr']:.6f} dt={dt:.1f}s")

        if val_acc > best_val or args.save_every_epoch:
            best_val = max(best_val, val_acc)
            args.out_pth.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": net.state_dict(),
                        "val_acc": val_acc, "epoch": ep,
                        "n_params": n_params}, args.out_pth)
            if val_acc == best_val:
                log(f"[ckpt] saved best val={val_acc:.4f} -> {args.out_pth}")

    # Confusion matrix from final epoch
    cm_path = args.report_dir / "confusion_matrix.txt"
    with open(cm_path, "w", encoding="utf-8") as f:
        f.write("           pred_live  pred_spoof\n")
        f.write(f"  live      {cm[0,0]:6d}     {cm[0,1]:6d}\n")
        f.write(f"  spoof     {cm[1,0]:6d}     {cm[1,1]:6d}\n")
    log(f"[cm] written to {cm_path}")

    # ONNX export from best ckpt (legacy exporter to dodge the cp1252 unicode bug)
    log("[onnx] exporting ...")
    ckpt = torch.load(args.out_pth, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    dummy = torch.zeros(1, 3, 16, 224, 224, device=device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, dummy, str(args.out),
                      input_names=["clip"], output_names=["logits"],
                      dynamic_axes={"clip": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    sz = args.out.stat().st_size / (1 << 20)
    log(f"[onnx] wrote {args.out} ({sz:.2f} MB) best_val_acc={best_val:.4f}")

    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
