"""pad/train_face_xray_adv.py -- adversarially-trained Face X-ray.

Same family as `pad.train_adversarial` (which hardens patch_depth), but
for the deepfake branch's Face X-ray classifier. Builds on the clean
train_face_xray.py (reuses FaceXrayNet + synth_blend_batch + LiveCropDataset).

Recipe:
  * Each batch: real crops + synthetic-blend fakes (as in clean training).
  * A configurable fraction of batches is turned into adversarial
    examples (FGSM or PGD) *targeted at the current loss* before the
    optimizer step.
  * The model is trained on 50% clean + 50% adversarial by default.
  * Validation reports three numbers:
        clean_val_acc   -- untouched images
        adv4_val_acc    -- epsilon = 4/255 adversarial
        adv8_val_acc    -- epsilon = 8/255 adversarial
    That three-column accuracy row is the headline for the paper's
    adversarial-robustness section.

Attack options (v9.6.0):
  --attack fgsm   single-step Fast Gradient Sign Method (Goodfellow 2014).
                  Cheap, sometimes too weak at eps=8 for a model to learn
                  robustness — observed to collapse to 0.50 adv-val in
                  the 2026-04-21 run.
  --attack pgd    multi-step Projected Gradient Descent (Madry 2018).
                  The standard for certifiable-robustness training.
                  Typical hparams at eps=8/255: --pgd-steps 7 --pgd-alpha-ratio 0.25.

Outputs:
    pad/models/face_xray_adv.onnx            # hardened ONNX, drop-in
    pad/models/face_xray_adv.pth             # checkpoint for future FT
    pad/reports/face_xray_adv/train_log.txt  # per-epoch metrics

Usage (FGSM, legacy default):
    python -u -m pad.train_face_xray_adv \\
        --data E:\\pad_datasets\\face_xray E:\\pad_datasets\\Celeb-DF-v2 \\
        --epochs 10 --batch 24 --num-workers 4 \\
        --epsilon 8 --adv-ratio 0.5 \\
        --finetune-from pad/models/face_xray.pth \\
        --out pad/models/face_xray_adv.onnx

Usage (PGD, recommended for robust eps=8/255):
    python -u -m pad.train_face_xray_adv \\
        --data E:\\pad_datasets\\face_xray E:\\pad_datasets\\Celeb-DF-v2 \\
        --epochs 20 --batch 24 --lr 5e-5 \\
        --attack pgd --pgd-steps 7 --pgd-alpha-ratio 0.25 \\
        --epsilon 8 --adv-ratio 0.7 \\
        --finetune-from pad/models/face_xray.pth \\
        --out pad/models/face_xray_adv.onnx

Tips:
    * With --finetune-from pointing at a pre-trained face_xray.pth from
      a clean training run, this script only needs ~10 epochs to converge.
    * Without --finetune-from, it trains from ImageNet + 30 epochs
      recommended (similar cost to clean training + ~20% overhead for adv).
    * PGD is ~N× slower than FGSM in adversarial batches (N = --pgd-steps).
      Factor into epoch-time budget.
"""
from __future__ import annotations
import argparse
import random
import sys
import time
from pathlib import Path
from typing import Optional

# All the heavy dependencies are imported inside functions so --help works
# in minimal environments.

from pad.train_face_xray import (
    LiveCropDataset,
    FaceXrayNet,
    synth_blend_batch,
    imagenet_normalise,
    export_onnx,
    pick_device,
    _lazy,
)


def _loss_at(model, x, y, lambda_mask, mask_gt, nn, F):
    """Shared classification+mask loss used by both FGSM and PGD attacks."""
    x_n = imagenet_normalise(x)
    if mask_gt is not None:
        logits, mask_pred = model(x_n, return_mask=True)
        return (nn.CrossEntropyLoss()(logits, y)
                + lambda_mask * F.binary_cross_entropy_with_logits(mask_pred, mask_gt))
    logits = model(x_n, return_mask=False)
    return nn.CrossEntropyLoss()(logits, y)


def fgsm_perturb(model, x, y, epsilon, lambda_mask, mask_gt=None):
    """Compute an FGSM perturbation of x against the classification loss.
    Returns perturbed x in [0,1]. Model weights are NOT updated here.

    epsilon is in units of 1/255 (so epsilon=8 means 8/255).
    """
    _, torch, nn, F, _, _, _, _, _, _ = _lazy()
    eps = float(epsilon) / 255.0
    x_req = x.clone().detach().requires_grad_(True)
    loss = _loss_at(model, x_req, y, lambda_mask, mask_gt, nn, F)
    grad = torch.autograd.grad(loss, x_req, retain_graph=False,
                               create_graph=False)[0]
    x_adv = (x_req + eps * grad.sign()).clamp(0.0, 1.0).detach()
    return x_adv


def pgd_perturb(model, x, y, epsilon, lambda_mask, mask_gt=None,
                steps: int = 7, alpha_ratio: float = 0.25,
                random_start: bool = True):
    """Projected Gradient Descent (Madry 2018) in the L_inf ball of
    radius epsilon/255 around x. steps=1 with random_start=False
    degenerates to FGSM.

    Args:
        steps: number of gradient ascent iterations. 7 is the Madry
            default; 20 is a stronger attack but is ~3× slower.
        alpha_ratio: per-step size as a fraction of epsilon. 0.25 (so
            alpha = eps / 4) is the Madry default.
        random_start: if True, initialise x_adv at a uniform random
            point inside the ball (recommended — prevents gradient
            masking).
    """
    _, torch, nn, F, _, _, _, _, _, _ = _lazy()
    eps = float(epsilon) / 255.0
    alpha = eps * float(alpha_ratio)
    x_orig = x.clone().detach()
    if random_start and eps > 0.0:
        x_adv = (x_orig + torch.empty_like(x_orig).uniform_(-eps, eps))
        x_adv = x_adv.clamp(0.0, 1.0).detach()
    else:
        x_adv = x_orig.clone().detach()
    for _ in range(max(1, int(steps))):
        x_adv.requires_grad_(True)
        loss = _loss_at(model, x_adv, y, lambda_mask, mask_gt, nn, F)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False,
                                   create_graph=False)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        # Project back into the eps-L_inf ball around x_orig
        delta = (x_adv - x_orig).clamp(-eps, eps)
        x_adv = (x_orig + delta).clamp(0.0, 1.0).detach()
    return x_adv


def adv_perturb(attack: str, model, x, y, epsilon, lambda_mask,
                mask_gt=None, *, pgd_steps: int = 7,
                pgd_alpha_ratio: float = 0.25,
                pgd_random_start: bool = True):
    """Dispatcher that picks between FGSM and PGD. Keeps the caller
    signature small so run_epoch / eval_at_epsilon don't balloon."""
    if attack == "pgd":
        return pgd_perturb(model, x, y, epsilon, lambda_mask, mask_gt,
                           steps=pgd_steps, alpha_ratio=pgd_alpha_ratio,
                           random_start=pgd_random_start)
    # default — fgsm (explicit string or any unknown value)
    return fgsm_perturb(model, x, y, epsilon, lambda_mask, mask_gt)


def run_epoch(model, loader, opt, device, train: bool,
              lambda_mask: float, epoch_idx: int, total_epochs: int,
              adv_ratio: float, epsilon: float,
              log_every: int, rng: random.Random,
              attack: str = "fgsm",
              pgd_steps: int = 7,
              pgd_alpha_ratio: float = 0.25):
    """One epoch with optional adversarial augmentation (FGSM or PGD)."""
    _, torch, nn, F, _, _, _, _, _, _ = _lazy()
    model.train() if train else model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum, mask_loss_sum = 0, 0, 0.0, 0.0
    n_batches = len(loader)
    t_start = time.time()
    ctx = torch.set_grad_enabled(True) if train else torch.no_grad()
    with ctx:
        for i, real in enumerate(loader):
            real = real.to(device, non_blocking=True)
            B = real.size(0)
            fake, mask_gt = synth_blend_batch(real, device, rng)
            x = torch.cat([real, fake], dim=0)
            y = torch.cat([torch.zeros(B, dtype=torch.long, device=device),
                           torch.ones (B, dtype=torch.long, device=device)],
                          dim=0)
            m = torch.cat([torch.zeros_like(mask_gt), mask_gt], dim=0)

            use_adv = train and (rng.random() < adv_ratio)
            if use_adv:
                # Produce adv examples first (model in train mode but this
                # doesn't update weights; just a gradient query).
                x_adv = adv_perturb(attack, model, x, y, epsilon,
                                    lambda_mask, m,
                                    pgd_steps=pgd_steps,
                                    pgd_alpha_ratio=pgd_alpha_ratio)
                x_train = x_adv
            else:
                x_train = x

            x_n = imagenet_normalise(x_train)
            logits, mask_pred = model(x_n, return_mask=True)
            loss_cls = ce(logits, y)
            loss_mask = F.binary_cross_entropy_with_logits(mask_pred, m)
            loss = loss_cls + lambda_mask * loss_mask

            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

            loss_sum += float(loss) * x_train.size(0)
            mask_loss_sum += float(loss_mask) * x_train.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x_train.size(0)
            if train and log_every > 0 and (i + 1) % log_every == 0:
                dt = time.time() - t_start
                ips = total / max(dt, 1e-6)
                eta = (n_batches - i - 1) * (dt / (i + 1)) / 60.0
                tag = attack if use_adv else "clean"
                print(f"  [fxr-adv ep {epoch_idx}/{total_epochs} "
                      f"batch {i+1:5d}/{n_batches}] ({tag}) "
                      f"loss={loss_sum/total:.4f} mask={mask_loss_sum/total:.4f} "
                      f"acc={correct/total:.3f} {ips:.0f} img/s ETA {eta:.1f} min",
                      flush=True)
    return (loss_sum / max(total, 1),
            mask_loss_sum / max(total, 1),
            correct / max(total, 1))


def eval_at_epsilon(model, loader, device, epsilon, lambda_mask,
                    rng: random.Random,
                    attack: str = "fgsm",
                    pgd_steps: int = 7,
                    pgd_alpha_ratio: float = 0.25):
    """Adversarial validation: returns accuracy on adversarially-perturbed
    val set. `attack` chooses the perturbation family — validation is
    always run against the same attack the model is trained with, so a
    PGD-trained model is graded on PGD-perturbed inputs. epsilon == 0
    means clean evaluation (attack argument is ignored)."""
    _, torch, nn, F, _, _, _, _, _, _ = _lazy()
    model.eval()
    total, correct = 0, 0
    for real in loader:
        real = real.to(device, non_blocking=True)
        B = real.size(0)
        fake, mask_gt = synth_blend_batch(real, device, rng)
        x = torch.cat([real, fake], dim=0)
        y = torch.cat([torch.zeros(B, dtype=torch.long, device=device),
                       torch.ones (B, dtype=torch.long, device=device)], dim=0)
        m = torch.cat([torch.zeros_like(mask_gt), mask_gt], dim=0)
        if epsilon > 0:
            # Adversarial perturbation requires gradients temporarily
            with torch.enable_grad():
                x_eval = adv_perturb(attack, model, x, y, epsilon,
                                     lambda_mask, m,
                                     pgd_steps=pgd_steps,
                                     pgd_alpha_ratio=pgd_alpha_ratio)
        else:
            x_eval = x
        with torch.no_grad():
            x_n = imagenet_normalise(x_eval)
            logits = model(x_n, return_mask=False)
        correct += (logits.argmax(1) == y).sum().item()
        total += x_eval.size(0)
    return correct / max(total, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, nargs="+",
                    help="one or more dataset roots, each with live/")
    ap.add_argument("--out", default="pad/models/face_xray_adv.onnx")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--lambda-mask", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--finetune-from", type=Path, default=None,
                    help="Warm-start from a .pth checkpoint (recommended).")
    ap.add_argument("--adv-ratio", type=float, default=0.5,
                    help="fraction of train batches turned into adversarial "
                         "examples (0.0 = pure clean, 1.0 = always adv)")
    ap.add_argument("--epsilon", type=int, default=8,
                    help="adv perturbation in units of 1/255 (default 8)")
    ap.add_argument("--attack", choices=("fgsm", "pgd"), default="fgsm",
                    help="adversarial attack family. fgsm = single-step "
                         "(legacy default, cheap); pgd = multi-step Madry, "
                         "recommended when fgsm collapses to 0.5 adv-val.")
    ap.add_argument("--pgd-steps", type=int, default=7,
                    help="PGD gradient ascent iterations (default 7 per "
                         "Madry 2018; 20 for a stronger attack at ~3× cost).")
    ap.add_argument("--pgd-alpha-ratio", type=float, default=0.25,
                    help="PGD per-step size as fraction of epsilon "
                         "(default 0.25 so alpha = eps/4).")
    ap.add_argument("--backbone", default="mobilenet_v3_small",
                    choices=["mobilenet_v3_small", "efficientnet_b4"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    rng = random.Random(args.seed)

    np, torch, nn, F, _, DataLoader, _, _, _, _ = _lazy()

    data_roots = [Path(p) for p in args.data]
    train_ds = LiveCropDataset(data_roots, split="train", seed=args.seed)
    val_ds   = LiveCropDataset(data_roots, split="val",   seed=args.seed)
    print(f"[fxr-adv] train {len(train_ds)} / val {len(val_ds)} live crops",
          flush=True)

    dl_kwargs = {"batch_size": args.batch, "num_workers": args.num_workers}
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = args.prefetch_factor
    device, dev_label = pick_device()
    print(f"[fxr-adv] device: {dev_label}", flush=True)
    if "cuda" in dev_label:
        dl_kwargs["pin_memory"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader   = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    model = FaceXrayNet(pretrained=not args.no_pretrained,
                        backbone_name=args.backbone).to(device)
    if args.finetune_from is not None and args.finetune_from.is_file():
        state = torch.load(str(args.finetune_from), map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        miss, unexpected = model.load_state_dict(state, strict=False)
        print(f"[fxr-adv] warm-started from {args.finetune_from} "
              f"(missing={len(miss)}, unexpected={len(unexpected)})",
              flush=True)
    elif args.finetune_from is not None:
        print(f"[fxr-adv] WARNING: --finetune-from {args.finetune_from} "
              f"not a file; starting from ImageNet.", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    report_dir = Path("pad/reports/face_xray_adv")
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / "train_log.txt"
    log_f = open(log_path, "w", encoding="utf-8")

    best_clean = 0.0
    best_adv = 0.0
    print(f"[fxr-adv] attack={args.attack} epsilon={args.epsilon}/255 "
          f"adv_ratio={args.adv_ratio}"
          + (f" pgd_steps={args.pgd_steps} "
             f"pgd_alpha_ratio={args.pgd_alpha_ratio}"
             if args.attack == "pgd" else ""),
          flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_mask, tr_acc = run_epoch(
            model, train_loader, opt, device, train=True,
            lambda_mask=args.lambda_mask, epoch_idx=epoch,
            total_epochs=args.epochs, adv_ratio=args.adv_ratio,
            epsilon=args.epsilon, log_every=args.log_every, rng=rng,
            attack=args.attack, pgd_steps=args.pgd_steps,
            pgd_alpha_ratio=args.pgd_alpha_ratio)

        clean_val = eval_at_epsilon(model, val_loader, device,
                                    epsilon=0, lambda_mask=args.lambda_mask,
                                    rng=rng, attack=args.attack,
                                    pgd_steps=args.pgd_steps,
                                    pgd_alpha_ratio=args.pgd_alpha_ratio)
        adv4_val = eval_at_epsilon(model, val_loader, device,
                                   epsilon=4, lambda_mask=args.lambda_mask,
                                   rng=rng, attack=args.attack,
                                   pgd_steps=args.pgd_steps,
                                   pgd_alpha_ratio=args.pgd_alpha_ratio)
        adv8_val = eval_at_epsilon(model, val_loader, device,
                                   epsilon=8, lambda_mask=args.lambda_mask,
                                   rng=rng, attack=args.attack,
                                   pgd_steps=args.pgd_steps,
                                   pgd_alpha_ratio=args.pgd_alpha_ratio)
        sched.step()
        dt = time.time() - t0
        line = (f"[fxr-adv ep {epoch:03d}/{args.epochs}]  "
                f"train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
                f"clean_val={clean_val:.3f} adv4_val={adv4_val:.3f} "
                f"adv8_val={adv8_val:.3f}  lr={opt.param_groups[0]['lr']:.1e}  "
                f"({dt:.1f}s)")
        print(line, flush=True)
        log_f.write(line + "\n"); log_f.flush()
        if clean_val > best_clean:
            best_clean = clean_val
        if adv8_val > best_adv:
            best_adv = adv8_val
            # Save best-adv checkpoint.
            ckpt_path = Path(args.out).with_suffix(".pth")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(ckpt_path))

    log_f.write(f"[fxr-adv] best clean={best_clean:.3f} adv8={best_adv:.3f}\n")
    log_f.close()
    print(f"[fxr-adv] best clean val_acc: {best_clean:.3f}", flush=True)
    print(f"[fxr-adv] best eps=8/255 val_acc: {best_adv:.3f}", flush=True)
    export_onnx(model, Path(args.out))
    print("[fxr-adv] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
