"""pad/train_eye_state.py -- Tiny CNN for OPEN/CLOSED eye classification.

Adds a third independent channel to `blink_detector.py` (alongside
EAR and blendshape `eyeBlinkLeft/Right`). The heuristic EAR + blendshape
combo currently misses ~70 % of glasses-wearer blinks per the v9.7.5
comments in `blink_detector.py`. A trained CNN voting in parallel
breaks the tie when the two heuristic channels disagree.

Architecture
------------
4 conv layers + GAP + 2 FC. ~120 k parameters, 1.5 MB ONNX. Designed for:
  - 64x64 grayscale eye crop (matches MRL Eye Dataset)
  - real-time CPU inference (~1 ms) and NPU eligibility (static shape, INT8 friendly)
  - per-eye independent inference (left/right)

Inputs
------
MRL Eye Dataset extracted directory (see `pad/fetch_mrl_eye.py`). Filename
encoding: `s<sub>_<img>_<gen>_<glasses>_<state>_<refl>_<light>_<sensor>.png`.
The 5th underscore-separated digit (index 4) is the eye state:
  0 = closed, 1 = open.

We split BY SUBJECT (not by image) — train on ~30 subjects, val on ~7.
This stops the model memorising per-subject features and forces it to
learn eye-shape generalisation. Critical for the glasses-wearer case
where val subjects must include glasses.

Usage
-----
    python -m pad.train_eye_state ^
        --data E:\\pad_datasets\\MRL\\mrlEyes_2018_01 ^
        --epochs 15 --batch 256 ^
        --out pad/models/eye_state_v1.onnx ^
        --out-pth pad/models/eye_state_v1.pth

Output
------
    pad/models/eye_state_v1.onnx  -- 1.5 MB INT8-friendly classifier
    pad/models/eye_state_v1.pth   -- training checkpoint (for resume / further FT)
    pad/reports/eye_state_v1/train_log.txt
    pad/reports/eye_state_v1/confusion_matrix.txt
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

# Bootstrap CUDA DLLs same as main.py (stand-alone scripts must, ORT-CUDA fails otherwise).
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


# ---------------- model ----------------
class EyeStateNet(nn.Module):
    """4-conv tiny CNN. Input (N, 1, 64, 64) grayscale; output (N, 2) logits."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.b1 = nn.BatchNorm2d(16)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1); self.b2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1); self.b3 = nn.BatchNorm2d(64)
        self.c4 = nn.Conv2d(64, 64, 3, padding=1); self.b4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.b1(self.c1(x))), 2)  # 32
        x = F.max_pool2d(F.relu(self.b2(self.c2(x))), 2)  # 16
        x = F.max_pool2d(F.relu(self.b3(self.c3(x))), 2)  # 8
        x = F.adaptive_avg_pool2d(F.relu(self.b4(self.c4(x))), 1).flatten(1)  # (N, 64)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------- data ----------------
def _walk_mrl(root: Path) -> List[Tuple[Path, int, str]]:
    """Return [(path, label, subject_id), ...]. Label: 0=closed, 1=open."""
    items: List[Tuple[Path, int, str]] = []
    for p in root.rglob("*.png"):
        parts = p.stem.split("_")
        if len(parts) < 5:
            continue
        try:
            state = int(parts[4])  # 0=closed, 1=open
        except ValueError:
            continue
        sub = parts[0]  # e.g., "s0001"
        items.append((p, state, sub))
    if not items:
        # Some MRL releases use .jpg or nested by subject dir
        for p in root.rglob("*.jpg"):
            parts = p.stem.split("_")
            if len(parts) < 5:
                continue
            try:
                state = int(parts[4])
            except ValueError:
                continue
            sub = parts[0]
            items.append((p, state, sub))
    return items


class EyeDS(Dataset):
    def __init__(self, items: List[Tuple[Path, int, str]], train: bool):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        p, label, _ = self.items[i]
        img = Image.open(p).convert("L").resize((64, 64), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float32) / 255.0
        if self.train:
            if random.random() < 0.5:
                x = x[:, ::-1].copy()  # horizontal flip
            # mild brightness jitter
            x = np.clip(x + random.uniform(-0.1, 0.1), 0.0, 1.0)
        x = (x - 0.5) / 0.5  # normalize to [-1, 1]
        return torch.from_numpy(x).unsqueeze(0), int(label)


def _split_by_subject(items, val_frac: float, seed: int):
    rng = random.Random(seed)
    by_sub: dict = defaultdict(list)
    for p, lbl, sub in items:
        by_sub[sub].append((p, lbl, sub))
    subs = sorted(by_sub.keys())
    rng.shuffle(subs)
    n_val = max(1, int(round(len(subs) * val_frac)))
    val_subs = set(subs[:n_val])
    train_subs = subs[n_val:]
    train = [it for s in train_subs for it in by_sub[s]]
    val = [it for s in val_subs for it in by_sub[s]]
    return train, val, sorted(train_subs), sorted(val_subs)


# ---------------- training ----------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="root of extracted MRL Eye Dataset")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction of SUBJECTS for val (not images)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=Path("pad/models/eye_state_v1.onnx"))
    ap.add_argument("--out-pth", type=Path, default=Path("pad/models/eye_state_v1.pth"))
    ap.add_argument("--report-dir", type=Path,
                    default=Path("pad/reports/eye_state_v1"))
    args = ap.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.report_dir / "train_log.txt"
    log_f = open(log_path, "w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.data.is_dir():
        log(f"[FAIL] data dir not found: {args.data}")
        return 2

    log(f"[ds] scanning {args.data} for *.png/*.jpg with MRL filename pattern")
    items = _walk_mrl(args.data)
    log(f"[ds] found {len(items)} eye images")
    if not items:
        log("[FAIL] no images found; check --data path or MRL filename format")
        return 2

    train, val, ts, vs = _split_by_subject(items, args.val_frac, args.seed)
    log(f"[ds] subjects train={len(ts)} val={len(vs)} (val frac={args.val_frac})")
    log(f"[ds]    train images = {len(train)}  val images = {len(val)}")
    n_closed = sum(1 for _, l, _ in train if l == 0)
    n_open = len(train) - n_closed
    log(f"[ds]    train balance: closed={n_closed}  open={n_open}")

    train_ds = EyeDS(train, train=True)
    val_ds = EyeDS(val, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[run] device={device}  torch={torch.__version__}")
    if device == "cuda":
        log(f"[run] gpu={torch.cuda.get_device_name(0)} "
            f"vram={torch.cuda.get_device_properties(0).total_memory / (1<<30):.1f} GB")

    net = EyeStateNet().to(device)
    n_params = sum(p.numel() for p in net.parameters())
    log(f"[run] params={n_params:,}")

    # Class weighting handles MRL's slight open/closed imbalance.
    pos_weight = torch.tensor([n_open / max(n_closed, 1), 1.0], device=device)
    crit = nn.CrossEntropyLoss(weight=pos_weight)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = 0.0
    for ep in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        net.train()
        train_loss = 0.0
        train_correct = 0
        train_n = 0
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
        v_correct = 0
        v_n = 0
        cm = np.zeros((2, 2), dtype=np.int64)  # rows truth, cols pred
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
            f"train_loss={train_loss / max(train_n,1):.4f}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
            f"lr={opt.param_groups[0]['lr']:.5f}  dt={dt:.1f} s")

        if val_acc > best_val:
            best_val = val_acc
            args.out_pth.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": net.state_dict(),
                        "val_acc": val_acc, "epoch": ep,
                        "n_params": n_params,
                        "subjects_train": ts, "subjects_val": vs},
                       args.out_pth)
            log(f"[ckpt] saved best val={val_acc:.4f} -> {args.out_pth}")

    # Final confusion matrix on best
    log("")
    log("[cm] best validation confusion matrix:")
    log(f"           pred_closed  pred_open")
    log(f"  closed     {cm[0,0]:6d}     {cm[0,1]:6d}")
    log(f"  open       {cm[1,0]:6d}     {cm[1,1]:6d}")
    cm_path = args.report_dir / "confusion_matrix.txt"
    with open(cm_path, "w", encoding="utf-8") as f:
        f.write(f"           pred_closed  pred_open\n")
        f.write(f"  closed     {cm[0,0]:6d}     {cm[0,1]:6d}\n")
        f.write(f"  open       {cm[1,0]:6d}     {cm[1,1]:6d}\n")

    # ONNX export from best ckpt
    log("")
    log("[onnx] exporting...")
    ckpt = torch.load(args.out_pth, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    dummy = torch.zeros(1, 1, 64, 64, device=device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False forces the legacy TorchScript-based exporter. The new
    # torch.export-based exporter (default in torch >= 2.9) prints success
    # banners with Unicode emoji which crash on Windows cp1252 stdout.
    torch.onnx.export(net, dummy, str(args.out),
                      input_names=["eye"], output_names=["logits"],
                      dynamic_axes=None, opset_version=17, dynamo=False)
    sz_mb = args.out.stat().st_size / (1 << 20)
    log(f"[onnx] wrote {args.out}  ({sz_mb:.2f} MB)  best_val_acc={best_val:.4f}")

    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
