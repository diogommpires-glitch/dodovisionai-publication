"""pad/train_adversarial.py — adversarially-trained patch_depth.

Fine-tunes the existing `pad/models/patch_depth_fas.onnx` (converted back
to PyTorch via a fresh MobileNetV3-small with the same head) using
adversarially-perturbed inputs. Optionally trains from scratch. Produces
a hardened classifier whose output stays calibrated under F-Cutout-style
frequency-space perturbations and under FGSM pixel perturbations.

Why do this:
  pad/defense/adv_patch.py is an INFERENCE-time defence. It applies
  F-Cutout + JPEG re-encode to PAD-bound frames, masking some perturbation
  patterns, but the model itself was never trained to be robust. A model
  trained with adversarial augmentation learns representations that
  survive small L_inf perturbations natively.

Attack modes (toggled independently, can stack):
  --fcutout  ...  Frequency-space cutout: spectral band-zero + inverse FFT.
                  Implements RADAP (Pattern Recognition 2024) at training
                  time.
  --fgsm     ...  Fast Gradient Sign Method. One-step L_inf eps perturbation
                  using the current loss gradient. eps configurable via
                  --epsilon (in units of 1/255; default 8).

Training schedule:
  * Each batch is optionally replaced with its perturbed counterpart at a
    rate controlled by --adv-ratio (default 0.5 = 50% perturbed).
  * Perturbed batches use the SAME label as the clean batch.
  * Validation is done on BOTH clean and perturbed holdout, reported as
    two separate accuracy figures.

Produces at the end:
  pad/models/patch_depth_fas_adv.onnx    # hardened model
  pad/models/patch_depth_fas.onnx.prev   # snapshot of the pre-adv model
  Final console line reports:
    clean val_acc / eps=4/255 val_acc / eps=8/255 val_acc

That three-column accuracy table is the headline result for the paper's
adversarial-robustness section.

Example:
    python -u -m pad.train_adversarial \\
        --data E:\\pad_datasets\\patch_depth_mmap \\
        --epochs 10 --batch 64 --num-workers 4 \\
        --fcutout --fgsm --epsilon 8 --adv-ratio 0.5 \\
        > E:\\pad_datasets\\train_adv.log 2>&1

Uses the same `MmapSpoofDataset` layer as `pad.train_patch_depth`, so it
runs at SSD speeds (~600 img/s) when --data points to the mmap directory.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Optional


REPO = Path(__file__).resolve().parent.parent


# ── adversarial perturbation helpers ──
def f_cutout(x, rng):
    """Frequency-space cutout: zero a random spectral band in FFT, inverse.

    Operates on a (C, H, W) float tensor in [0, 1]. Returns perturbed tensor.
    """
    import torch
    with torch.no_grad():
        X = torch.fft.rfft2(x)            # (C, H, W//2+1), complex
        C, H, Wf = X.shape
        # Random band to zero out (keep it cheap: 1-3 horizontal strips)
        mask = torch.ones_like(X, dtype=torch.float32)
        for _ in range(int(rng.randint(1, 4))):
            r0 = int(rng.randint(0, H))
            r1 = min(H, r0 + int(rng.randint(2, 12)))
            mask[:, r0:r1, :] = 0.0
        X = X * mask
        y = torch.fft.irfft2(X, s=(x.shape[-2], x.shape[-1]))
    return y.clamp(0.0, 1.0)


def fgsm_perturb(model, x, y, loss_fn, epsilon_01, amp_dtype=None):
    """One-step FGSM. epsilon_01 is in normalized-pixel units (0..1).
    v9.10: optional bf16/fp16 autocast around the forward+loss; FGSM only
    needs `grad.sign()`, which is dtype-invariant, so reduced precision
    doesn't change the perturbation direction."""
    import torch
    x_adv = x.detach().clone().requires_grad_(True)
    if amp_dtype is not None:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(x_adv)
            loss = loss_fn(logits, y)
    else:
        logits = model(x_adv)
        loss = loss_fn(logits, y)
    grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
    x_adv = (x_adv + epsilon_01 * grad.sign()).clamp(0.0, 1.0).detach()
    return x_adv


def pgd_perturb(model, x, y, loss_fn, epsilon_01, steps=5,
                amp_dtype=None):
    """Multi-step Projected Gradient Descent (Madry et al. 2017).

    Strictly stronger than FGSM: each step takes a smaller gradient-sign
    update of size alpha, then projects back into the L_inf ball of
    radius epsilon. Models trained against PGD are robust to FGSM AND
    to PGD (the reverse is not true -- which is the gap Table 6b
    exposed for v8_adv).

    v9.11: implementation matches the Madry recipe used by the eval
    script `scripts/_v910_pgd_v8_adv_eval.py`, so train-time and
    eval-time attacks are aligned.

    Args:
      epsilon_01: L_inf budget in [0,1] units (typically 4/255 or 8/255).
      steps:      PGD inner iteration count. Madry uses 7-10; we default
                  to 5 to keep training cost ~5x FGSM. See `--pgd-steps`.
      amp_dtype:  optional bf16/fp16 autocast (same logic as fgsm_perturb).
    """
    import torch
    if epsilon_01 <= 0:
        return x.detach().clone()
    # Madry alpha = 2.5 * eps / steps  (slightly overshoots so projection
    # actively re-clips on most steps -- empirically more attack signal)
    alpha = max(epsilon_01 / max(steps, 1) * 2.5, 1.0 / 255.0)
    # Random start inside the L_inf ball (PGD-rand)
    x_orig = x.detach().clone()
    x_adv = x_orig + (torch.rand_like(x_orig) * 2 - 1) * epsilon_01
    x_adv = x_adv.clamp(0.0, 1.0).detach().requires_grad_(True)
    for _ in range(steps):
        if amp_dtype is not None:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(x_adv)
                loss = loss_fn(logits, y)
        else:
            logits = model(x_adv)
            loss = loss_fn(logits, y)
        grad = torch.autograd.grad(loss, x_adv,
                                   retain_graph=False, create_graph=False)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        # Project back to the L_inf ball around x_orig
        x_adv = torch.max(torch.min(x_adv, x_orig + epsilon_01),
                          x_orig - epsilon_01)
        x_adv = x_adv.clamp(0.0, 1.0).detach().requires_grad_(True)
    return x_adv.detach()


# ── training loop ──
def run_epoch(model, loader, loss_fn, opt, device,
              train: bool,
              fcutout: bool, fgsm: bool, epsilon_01: float, adv_ratio: float,
              log_every: int, epoch_idx: int, total_epochs: int,
              amp_dtype=None, channels_last: bool = False,
              pgd: bool = False, pgd_steps: int = 5):
    """v9.10: amp_dtype + channels_last threaded in for the new max-utilization
    profile. bf16 autocast on Blackwell needs no GradScaler, and channels_last
    on a 5090 buys ~10-20% on conv-heavy graphs like MobileNetV3-small."""
    import torch
    import numpy as np
    rng = np.random.RandomState(epoch_idx * 31337 + 7)
    model.train() if train else model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    n_batches = len(loader)
    t_start = time.time()
    ctx = torch.enable_grad() if train else torch.no_grad()
    amp_enabled = amp_dtype is not None
    with ctx:
        for i, (x, y) in enumerate(loader):
            if channels_last:
                x = x.to(device, non_blocking=True,
                         memory_format=torch.channels_last)
            else:
                x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Optionally perturb batch (only when training, stochastic).
            # v9.11: PGD takes precedence over FGSM if both are requested
            # (PGD is strictly stronger and is what the v10 paper actually
            # cares about). Both can coexist with f-cutout.
            if train and (fcutout or fgsm or pgd) and rng.random() < adv_ratio:
                if fcutout:
                    x = torch.stack([f_cutout(xi, rng) for xi in x], dim=0)
                if pgd:
                    x = pgd_perturb(model, x, y, loss_fn, epsilon_01,
                                    steps=pgd_steps, amp_dtype=amp_dtype)
                elif fgsm:
                    x = fgsm_perturb(model, x, y, loss_fn, epsilon_01,
                                     amp_dtype=amp_dtype)

            with torch.amp.autocast("cuda",
                                    dtype=amp_dtype if amp_enabled else torch.float32,
                                    enabled=amp_enabled):
                logits = model(x)
                loss = loss_fn(logits, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += float(loss) * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            if train and log_every > 0 and (i + 1) % log_every == 0:
                elapsed = time.time() - t_start
                ips = total / elapsed
                batches_left = n_batches - (i + 1)
                eta_min = batches_left * (elapsed / (i + 1)) / 60.0
                print(f"  [adv ep {epoch_idx}/{total_epochs} batch {i+1:5d}/{n_batches}]  "
                      f"loss={loss_sum/total:.4f}  acc={correct/total:.3f}  "
                      f"{ips:.0f} img/s  ETA {eta_min:.1f} min",
                      flush=True)
    return loss_sum / max(total, 1), correct / max(total, 1)


def run_adversarial_val(model, loader, loss_fn, device, epsilon_01,
                        amp_dtype=None, channels_last: bool = False):
    """Evaluate clean accuracy AND FGSM-perturbed accuracy on the same loader.
    v9.10: bf16 autocast + channels_last for matching the training-time profile."""
    import torch
    model.eval()
    clean_correct = clean_total = 0
    adv_correct = adv_total = 0
    amp_enabled = amp_dtype is not None
    for x, y in loader:
        if channels_last:
            x = x.to(device, non_blocking=True,
                     memory_format=torch.channels_last)
        else:
            x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        # clean
        with torch.no_grad():
            with torch.amp.autocast("cuda",
                                    dtype=amp_dtype if amp_enabled else torch.float32,
                                    enabled=amp_enabled):
                clean_pred = model(x).argmax(1)
        clean_correct += (clean_pred == y).sum().item()
        clean_total += x.size(0)
        # adversarial (FGSM at the specified epsilon)
        if epsilon_01 > 0:
            x_adv = fgsm_perturb(model, x, y, loss_fn, epsilon_01,
                                 amp_dtype=amp_dtype)
            with torch.no_grad():
                with torch.amp.autocast("cuda",
                                        dtype=amp_dtype if amp_enabled else torch.float32,
                                        enabled=amp_enabled):
                    adv_pred = model(x_adv).argmax(1)
            adv_correct += (adv_pred == y).sum().item()
            adv_total += x.size(0)
    clean_acc = clean_correct / max(1, clean_total)
    adv_acc = (adv_correct / adv_total) if adv_total > 0 else float("nan")
    return clean_acc, adv_acc


# ── main ──
def pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def build_model(pretrained: bool = True,
                load_from: Optional[Path] = None):
    """MobileNetV3-small with a 2-class head. If `load_from` is given (a
    `.pt` or pre-existing checkpoint), load weights; otherwise start from
    ImageNet-pretrained."""
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, 2)
    if load_from is not None and load_from.exists():
        state = torch.load(str(load_from), map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"[adv] loaded weights from {load_from}")
    return model


def save_pth(model, out_pth: Path) -> None:
    """v9.10: persist trained weights to a `.pth` BEFORE the ONNX export.
    This protects the ~2.5 h adversarial training run from being wiped if
    `torch.onnx.export` later raises (it has historically crashed on
    Windows cp1252 stdout when the dynamo path tries to verbose-print
    its Unicode-emoji success line). The `.pth` is what
    `scripts/_v910_pgd_v8_adv_eval.py` actually loads, so it is the
    paper-critical artefact -- the ONNX is just a deployment optimisation."""
    import torch
    out_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_pth))
    print(f"[adv] wrote PTH -> {out_pth} ({out_pth.stat().st_size/1e6:.1f} MB)",
          flush=True)


def export_onnx(model, out_path: Path):
    """v9.10: pinned to the legacy (non-dynamo) torch.onnx exporter via
    `dynamo=False`. The dynamo path emits a verbose-print line containing
    the Unicode check-mark `✅`, which crashes Python's default cp1252
    stdout encoder on Windows -- this happened on the v9.10 v8_adv run and
    nuked the trained weights because no `.pth` had been saved first.
    Same fix as `pad/train_patch_depth.py`. The legacy exporter produces
    the same artefact for this MobileNetV3-small graph."""
    import torch
    model.eval().cpu()
    dummy = torch.randn(1, 3, 224, 224)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    print(f"[adv] wrote ONNX -> {out_path} ({out_path.stat().st_size/1e6:.1f} MB)",
          flush=True)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    # v9.10 max-utilization profile (audited 2026-05-09 against 5090 24 GB):
    #   MobileNetV3-small + FGSM doubles forward passes per step, so peak
    #   activation ~6 MB/sample. batch=256 -> ~1.5 GB. ~10 GB headroom for
    #   the FGSM gradient-w.r.t.-input + bf16 workspace. v9.9 batch=64
    #   left the GPU idle 70% of the wall time (CPU-bound on PIL aug).
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    # v9.10: 4 -> 8. Mirrors the train_patch_depth_v8 audit: 24-core CPU
    # only feeds 4 augmentation pipelines at a time previously, leaving
    # ~20 cores idle. 8 workers fully saturate the augmentation pipeline
    # at batch=256 without RSS pressure (each worker ~600 MB peak).
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    # v9.10: amp_dtype + channels_last, mirroring the train_patch_depth_v8
    # max-utilization profile. bf16 is the production default; fp16 is
    # available for legacy bit-equivalent reproduction; none disables AMP.
    ap.add_argument("--amp-dtype",
                    choices=["bf16", "fp16", "none"], default="bf16",
                    help="AMP precision: bf16 (default), fp16 (legacy), none (FP32)")
    ap.add_argument("--no-channels-last", action="store_true",
                    help="disable channels_last memory format")
    # v9.10: hardware auto-tune (mirror of train_patch_depth_v8). When
    # ON (default), --batch and --num-workers get overridden with values
    # derived from `pad.hw_profile.auto_tune_for`. CLI overrides win.
    ap.add_argument("--auto-tune", dest="auto_tune", action="store_true",
                    default=True,
                    help="(v9.10, default ON) auto-derive --batch and "
                         "--num-workers from detected hardware profile")
    ap.add_argument("--no-auto-tune", dest="auto_tune", action="store_false",
                    help="pin --batch and --num-workers to the CLI defaults")
    ap.add_argument("--fcutout", action="store_true",
                    help="Apply frequency-space cutout during training")
    ap.add_argument("--fgsm", action="store_true",
                    help="Apply FGSM pixel perturbation during training")
    # v9.11: PGD multi-step adversarial training. Stronger than FGSM and
    # produces a model that's robust to ITERATIVE attacks (the gap
    # Table 6b exposed for v8_adv -- FGSM-trained model fails under PGD).
    # Cost: ~`pgd_steps`x slower than FGSM (each adversarial example
    # requires `pgd_steps` extra forward+backward passes).
    ap.add_argument("--pgd", action="store_true",
                    help="Apply multi-step PGD perturbation during training "
                         "(Madry et al. 2017). Strictly stronger than FGSM.")
    ap.add_argument("--pgd-steps", type=int, default=5,
                    help="Number of PGD inner iterations (default 5; "
                         "Madry recipe uses 7-10; trade-off vs training time)")
    ap.add_argument("--epsilon", type=float, default=8.0,
                    help="FGSM perturbation budget in units of 1/255 "
                         "(default 8 = standard adversarial eval)")
    ap.add_argument("--adv-ratio", type=float, default=0.5,
                    help="Fraction of batches that get perturbed (0..1)")
    ap.add_argument("--init-from", type=str, default=None,
                    help="Load starting weights from a .pt checkpoint "
                         "(if empty: ImageNet pretrain)")
    ap.add_argument("--out",
                    default="pad/models/patch_depth_fas_adv.onnx")
    args = ap.parse_args(argv)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    # Reuse the exact mmap dataset from train_patch_depth
    from pad.train_patch_depth import MmapSpoofDataset, FolderSpoofDataset
    # v9.10: hardware probe + opt-out auto-tune
    from pad.hw_profile import (
        hardware_profile, auto_tune_for, format_profile_line,
    )

    device, dev_label = pick_device()
    print(f"[adv] device: {dev_label}")
    _hw_profile = hardware_profile()
    if args.auto_tune:
        _hw_cfg = auto_tune_for(_hw_profile)
        # FGSM doubles the forward pass, so halve the auto-tune batch
        # (the v9.10 audit measured ~6 MB/sample peak under FGSM, vs ~3
        # MB without — so the batch budget is half what plain training
        # supports). Workers heuristic carries over unchanged.
        adv_batch = max(64, _hw_cfg["batch"] // 6)
        _DEFAULT_BATCH = 256
        _DEFAULT_WORKERS = 8
        before_w, before_b = args.num_workers, args.batch
        if args.num_workers == _DEFAULT_WORKERS:
            args.num_workers = _hw_cfg["workers"]
        if args.batch == _DEFAULT_BATCH:
            args.batch = adv_batch
        print(format_profile_line(_hw_profile, {**_hw_cfg, "batch": adv_batch}),
              flush=True)
        if (before_w, before_b) != (args.num_workers, args.batch):
            print(f"[adv] auto-tune adjusted num_workers {before_w}->{args.num_workers} "
                  f"batch {before_b}->{args.batch}",
                  flush=True)
    else:
        print(format_profile_line(_hw_profile), flush=True)
        print("[adv] auto-tune OFF (--no-auto-tune); using CLI defaults",
              flush=True)
    data_root = Path(args.data)
    if (data_root / "images.bin").exists() and (data_root / "labels.npy").exists():
        print(f"[adv] using MmapSpoofDataset at {data_root}")
        train_ds = MmapSpoofDataset(data_root, split="train")
        val_ds = MmapSpoofDataset(data_root, split="val")
    else:
        print(f"[adv] using FolderSpoofDataset at {data_root}")
        train_ds = FolderSpoofDataset(data_root, split="train")
        val_ds = FolderSpoofDataset(data_root, split="val")
    print(f"[adv] {len(train_ds)} train / {len(val_ds)} val images")

    # v9.10.4 defensive harden (matches train_patch_depth_v8.py fix):
    #   * persistent_workers=False -- workers respawn between epochs so
    #     shared file mappings (one per batch IPC on Windows) are fully
    #     released. With persistent_workers=True at workers>=8 the
    #     mappings accumulate until OS commit charge is exhausted ->
    #     Win32 1455 ERROR_COMMITMENT_LIMIT. Cost ~20s spawn per epoch
    #     vs the ~5 min training budget per epoch -- trivial.
    #   * prefetch_factor capped at 2 when num_workers>=8 OR batch>=1024
    #     to keep peak shared-mem commit under ~26 GB (proven safe).
    dl_kw = {"batch_size": args.batch, "num_workers": args.num_workers}
    if args.num_workers > 0:
        dl_kw["persistent_workers"] = False  # v9.10.4: was True; see comment above
        # Cap prefetch when worker count or batch size are large.
        safe_prefetch = (2 if (args.num_workers >= 8 or args.batch >= 1024)
                         else args.prefetch_factor)
        dl_kw["prefetch_factor"] = safe_prefetch
        if safe_prefetch < args.prefetch_factor:
            print(f"[adv] prefetch_factor capped {args.prefetch_factor} -> "
                  f"{safe_prefetch} (defensive: workers={args.num_workers} "
                  f"batch={args.batch} would risk Win32 1455 at higher prefetch)",
                  flush=True)
    if "cuda" in dev_label:
        dl_kw["pin_memory"] = True
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kw)

    # v9.10: cudnn benchmark picks the fastest conv algorithm for the
    # fixed input shape on first batch, ~10-15% on MobileNetV3.
    if "cuda" in dev_label:
        torch.backends.cudnn.benchmark = True
        # v9.10 micro-opt: TF32 on Tensor Cores for the FP32 fallback
        # paths. Implicit on Blackwell with bf16 autocast but explicit
        # is clearer + safe on older torch builds.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass  # nolint:silent-except graceful-degradation -- API absent on torch <2.0

    init_from = Path(args.init_from) if args.init_from else None
    model = build_model(pretrained=init_from is None,
                        load_from=init_from).to(device)
    # v9.10: channels_last (NHWC) on Blackwell + bf16 -> ~10-20% on conv2d
    use_channels_last = (not args.no_channels_last) and ("cuda" in dev_label)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    _AMP_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}
    amp_dtype = _AMP_DTYPE_MAP[args.amp_dtype]
    print(f"[adv] amp_dtype={args.amp_dtype} channels_last={use_channels_last} "
          f"cudnn.benchmark={torch.backends.cudnn.benchmark}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    epsilon_01 = args.epsilon / 255.0
    modes = []
    if args.fcutout: modes.append("F-Cutout")
    if args.pgd:
        # PGD takes precedence over FGSM if both are passed
        modes.append(f"PGD-{args.pgd_steps}step(eps={args.epsilon}/255)")
    elif args.fgsm:
        modes.append(f"FGSM(eps={args.epsilon}/255)")
    if not modes:
        modes = ["NO-PERTURB (plain training)"]
    print(f"[adv] perturbation modes: {', '.join(modes)}  ratio={args.adv_ratio}")

    best_clean = 0.0
    best_adv = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(
            model, train_loader, loss_fn, opt, device,
            train=True,
            fcutout=args.fcutout, fgsm=args.fgsm,
            pgd=args.pgd, pgd_steps=args.pgd_steps,
            epsilon_01=epsilon_01, adv_ratio=args.adv_ratio,
            log_every=args.log_every, epoch_idx=epoch, total_epochs=args.epochs,
            amp_dtype=amp_dtype, channels_last=use_channels_last,
        )
        clean_acc, adv_acc = run_adversarial_val(
            model, val_loader, loss_fn, device, epsilon_01,
            amp_dtype=amp_dtype, channels_last=use_channels_last,
        )
        dt = time.time() - t0
        print(f"[adv ep {epoch:03d}/{args.epochs}]  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f}  "
              f"clean_val_acc={clean_acc:.3f} adv_val_acc={adv_acc:.3f}  "
              f"({dt:.1f}s)",
              flush=True)
        if clean_acc > best_clean: best_clean = clean_acc
        if adv_acc > best_adv:     best_adv = adv_acc

    # Also probe at a second epsilon to produce the paper's three-column table
    print(f"[adv] final evaluation at eps=4/255 and eps=8/255:")
    for eps in (4.0, 8.0):
        _, a = run_adversarial_val(model, val_loader, loss_fn, device, eps/255.0,
                                   amp_dtype=amp_dtype,
                                   channels_last=use_channels_last)
        print(f"       eps={eps:.0f}/255  val_acc={a:.4f}")

    out = Path(args.out)
    # v9.10: ALWAYS write the .pth first so a downstream export crash
    # cannot lose the trained weights (the v9.10 v8_adv run was killed
    # this way when the dynamo ONNX exporter tripped on a cp1252 emoji).
    out_pth = out.with_suffix(".pth")
    save_pth(model, out_pth)

    try:
        export_onnx(model, out)
    except Exception as e:
        # Don't propagate -- weights are safe in .pth, ONNX is recoverable
        # offline via `python -m pad.export_adv_onnx --pth <pth>` if needed.
        print(f"[adv] WARNING: ONNX export raised {e!r}; .pth at {out_pth} "
              f"is the source of truth. Re-export with the dedicated "
              f"helper rather than re-training.", flush=True)
    else:
        # Backup previous clean model if we're overwriting the main path
        canonical = REPO / "pad" / "models" / "patch_depth_fas.onnx"
        if canonical.exists() and out.resolve() == canonical.resolve():
            import shutil
            shutil.copy(canonical, canonical.with_suffix(".onnx.prev"))
            print(f"[adv] backed up old canonical model to {canonical.with_suffix('.onnx.prev')}")
    print(f"[adv] done. best clean val_acc={best_clean:.3f}  "
          f"best adv val_acc={best_adv:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
