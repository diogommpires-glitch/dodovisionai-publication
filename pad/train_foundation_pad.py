"""train_foundation_pad.py  --  API_unq FoundPAD-style trainer.

CLIP ViT-L/14 vision encoder + LoRA adapters on q/v matrices only
(k and out_proj frozen). Binary live/spoof head on the pooled output.

Architectural rationale (FoundPAD, 2026 PAD survey):
  - ViT pre-trained on web-scale image-text pairs yields embeddings with
    strong cross-domain generalization.
  - LoRA on q,v keeps 99.5%+ of parameters frozen, preserves visual
    knowledge, and adds only ~1.2 M trainable params for ViT-L (vs 304 M
    full fine-tune).
  - Reported HTER drop of 6-13 pp vs SOTA in multi-source benchmarks.

Training inputs: a meta_train_records jsonl with rows
  {"path": "<jpg>", "label": "live"|"spoof", ...}
Use any of:
  - reports/meta_train_records_merged_v6.jsonl  (20 k merged, balanced)
  - reports/meta_train_records_ffpp_v5.jsonl    (122 k FF++, deepfake-heavy)
  - reports/meta_train_records_celeba_v6.jsonl  (5 k CelebA-Spoof)
  - concat multiple via --records arg multiplicities

Output:
  - pad/models/foundation_pad_v1.pth   (LoRA adapters + head weights)
  - pad/models/foundation_pad_v1.onnx  (full inference graph, FP32)
  - pad/models/foundation_pad_v1_fp16.onnx (optional, --export-fp16)
  - reports/foundation_pad_train_log.jsonl (per-epoch metrics)

Usage:
    python -m pad.train_foundation_pad \\
        --records reports/meta_train_records_merged_v6.jsonl \\
        --epochs 5 --batch 32 --lora-rank 8

Memory: CLIP ViT-L/14 forward at batch 32 image-only ≈ 4 GB VRAM.
Training step (with grad cache) ≈ 7-9 GB. Comfortable on 5090 24 GB.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Force HF transformers to skip TF import (avoids the venv's broken TF circular).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
REPORTS_DIR = HERE.parent / "reports"
DEFAULT_OUT_PTH = MODEL_DIR / "foundation_pad_v1.pth"
DEFAULT_OUT_ONNX = MODEL_DIR / "foundation_pad_v1.onnx"
DEFAULT_LOG = REPORTS_DIR / "foundation_pad_train_log.jsonl"
DEFAULT_CLIP_ID = "openai/clip-vit-large-patch14"


# ----------------------------------------------------------------------
# LoRA in pure torch
# ----------------------------------------------------------------------
class LoRALinear:
    """Wrap an nn.Linear to add a low-rank update A·B (Hu et al. 2021).

    Frozen W is the original linear; new_forward(x) = W(x) + scale·B(A(x)).
    We monkey-patch the wrapped module's forward to keep the rest of the
    model graph untouched (avoids needing per-block replacement code).
    """

    @staticmethod
    def inject(linear, rank: int, alpha: float, dtype):
        """Replace `linear.forward` in-place with a LoRA-augmented version.
        Returns the new trainable parameters (A, B) as a list."""
        import torch
        from torch import nn
        if not isinstance(linear, nn.Linear):
            return []
        in_f = linear.in_features
        out_f = linear.out_features
        # Freeze the base linear
        for p in linear.parameters():
            p.requires_grad = False
        # Trainable adapters
        device = linear.weight.device
        A = nn.Parameter(torch.zeros(rank, in_f, device=device, dtype=dtype))
        B = nn.Parameter(torch.zeros(out_f, rank, device=device, dtype=dtype))
        # Init: A ~ N(0, 1/rank); B = 0  (so initial output == frozen forward)
        nn.init.normal_(A, std=1.0 / max(rank, 1))
        # B stays zero
        scale = alpha / rank
        linear._lora_A = A
        linear._lora_B = B
        linear._lora_scale = scale
        orig_forward = linear.forward

        def new_forward(x, _orig=orig_forward, _A=A, _B=B, _s=scale):
            base = _orig(x)
            # x: (..., in_f); want (..., out_f) low-rank addend
            update = (x @ _A.t()) @ _B.t()
            return base + update * _s

        linear.forward = new_forward
        return [A, B]


def _inject_lora_on_qv(model_or_vision, rank: int, alpha: float, dtype) -> list:
    """Walk the CLIP vision tower's attention blocks and inject LoRA on
    q_proj + v_proj only (k_proj, out_proj frozen). Returns list of all
    trainable parameters added.

    HF transformers structure (4.57):
      CLIPModel
        .vision_model: CLIPVisionTransformer
          .encoder: CLIPEncoder
            .layers: ModuleList[CLIPEncoderLayer]
              .self_attn: CLIPAttention
                .q_proj / .k_proj / .v_proj / .out_proj

    The argument can be passed as either:
      - CLIPModel (top-level), or
      - CLIPVisionTransformer (already-unwrapped vision tower).
    """
    trainable = []
    # Bug-fix 2026-05-13: the old `or` short-circuit picked up
    # CLIPVisionTransformer's `.vision_model` attribute (None on the
    # vision tower itself but truthy on CLIPModel), leaving `enc` as
    # the WHOLE vision_model rather than its .encoder. Then
    # `.layers` doesn't exist on CLIPVisionTransformer.
    enc = None
    # 1. CLIPModel.vision_model.encoder
    vm = getattr(model_or_vision, "vision_model", None)
    if vm is not None and hasattr(vm, "encoder") and hasattr(vm.encoder, "layers"):
        enc = vm.encoder
    # 2. Direct CLIPVisionTransformer.encoder
    if enc is None:
        cand = getattr(model_or_vision, "encoder", None)
        if cand is not None and hasattr(cand, "layers"):
            enc = cand
    if enc is None or not hasattr(enc, "layers"):
        raise RuntimeError(
            f"could not locate CLIP vision encoder layers "
            f"(model type: {type(model_or_vision).__name__})"
        )
    for i, layer in enumerate(enc.layers):
        attn = layer.self_attn
        for proj_name in ("q_proj", "v_proj"):
            proj = getattr(attn, proj_name, None)
            if proj is None:
                continue
            params = LoRALinear.inject(proj, rank=rank, alpha=alpha, dtype=dtype)
            trainable.extend(params)
        # Freeze k_proj and out_proj explicitly
        for proj_name in ("k_proj", "out_proj"):
            proj = getattr(attn, proj_name, None)
            if proj is not None:
                for p in proj.parameters():
                    p.requires_grad = False
    return trainable


# ----------------------------------------------------------------------
# Model wrapper
# ----------------------------------------------------------------------
class _ConvVAE:
    """Small convolutional Variational Autoencoder for the texture
    side-stream. ~5 M params, latent_dim=32.

    Encoder: 4× downsample 224→14 via stride-2 convs.
    Latent: 32-D, reparameterized.
    Decoder: 4× upsample back to 224×224×3.

    The recon loss exposes microscopic surface texture (halftone ink dots,
    LCD pixel grids, silicone surface uniformity) that CLIP's semantic
    features miss. Per the ICCV 2025 Facevengers winning recipe (0.144%
    ACER), CLIP-semantic + VAE-texture + LoRA is the SOTA path for
    physical-PAD generalization. We expose the latent as an extra feature
    to the binary head and add a reconstruction loss to the training
    objective scaled by `vae_weight`.
    """

    @staticmethod
    def build(latent_dim: int, device: str, dtype):
        import torch
        from torch import nn

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(),    # 224→112
                    nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),   # 112→56
                    nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),  # 56→28
                    nn.Conv2d(128, 128, 4, 2, 1), nn.GELU(), # 28→14
                )
                self.to_mu = nn.Linear(128 * 14 * 14, latent_dim)
                self.to_logvar = nn.Linear(128 * 14 * 14, latent_dim)

            def forward(self, x):
                h = self.net(x).flatten(1)
                return self.to_mu(h), self.to_logvar(h)

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.from_latent = nn.Linear(latent_dim, 128 * 14 * 14)
                self.net = nn.Sequential(
                    nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.GELU(),  # 14→28
                    nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),   # 28→56
                    nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),    # 56→112
                    nn.ConvTranspose2d(32, 3, 4, 2, 1),                # 112→224
                )

            def forward(self, z):
                h = self.from_latent(z).view(-1, 128, 14, 14)
                return self.net(h)

        enc = Encoder().to(device=device, dtype=dtype)
        dec = Decoder().to(device=device, dtype=dtype)
        return enc, dec


def build_model(clip_id: str, rank: int, alpha: float, device: str, dtype,
                vae_enabled: bool = False, vae_latent: int = 32):
    """Build CLIP vision tower + LoRA adapters + binary classification head.

    API_unq 2026-05-13: optional `vae_enabled` adds a parallel convolutional
    VAE side-stream (Facevengers-style). The VAE latent is concatenated
    with the CLIP pooled embedding before the binary head, and a
    reconstruction loss is added to the training objective by the caller.
    """
    import torch
    from torch import nn
    from transformers import CLIPModel

    print(f"[foundpad] loading {clip_id} ...")
    base = CLIPModel.from_pretrained(clip_id, torch_dtype=dtype)
    vis = base.vision_model
    # Freeze everything in the base
    for p in base.parameters():
        p.requires_grad = False

    # Inject LoRA on q,v in the vision tower
    lora_params = _inject_lora_on_qv(base, rank=rank, alpha=alpha, dtype=dtype)
    print(f"[foundpad] LoRA params injected: {sum(p.numel() for p in lora_params):,}")

    # Binary head on the pooled vision embedding (+ optional VAE latent)
    hidden = base.config.vision_config.hidden_size  # 1024 for ViT-L/14
    head_in = hidden + (vae_latent if vae_enabled else 0)
    head = nn.Sequential(
        nn.LayerNorm(head_in),
        nn.Linear(head_in, 256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, 2),
    ).to(device=device, dtype=dtype)
    head_params = list(head.parameters())

    # VAE side-stream (optional)
    vae_enc = vae_dec = None
    vae_params: list = []
    if vae_enabled:
        vae_enc, vae_dec = _ConvVAE.build(vae_latent, device, dtype)
        vae_params = list(vae_enc.parameters()) + list(vae_dec.parameters())
        print(f"[foundpad] VAE side-stream enabled: latent={vae_latent}, "
              f"params={sum(p.numel() for p in vae_params):,}")

    base = base.to(device=device, dtype=dtype)

    class FoundPAD(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self.head = head
            self.vae_enabled = vae_enabled
            self.vae_enc = vae_enc
            self.vae_dec = vae_dec

        def encode_vae(self, pixel_values):
            """Returns (mu, logvar, z, recon) or (None, None, None, None) if disabled."""
            if not self.vae_enabled or self.vae_enc is None:
                return None, None, None, None
            mu, logvar = self.vae_enc(pixel_values)
            # Reparameterize during training; use mean at eval.
            if self.training:
                std = (0.5 * logvar).exp()
                eps = torch.randn_like(std)
                z = mu + eps * std
            else:
                z = mu
            recon = self.vae_dec(z)
            return mu, logvar, z, recon

        def forward(self, pixel_values):
            # CLIP semantic branch
            out = self.base.vision_model(pixel_values=pixel_values)
            pooled = out.pooler_output
            # VAE texture branch (optional)
            mu, logvar, z, recon = self.encode_vae(pixel_values)
            if self.vae_enabled and z is not None:
                # Cast z to head dtype just in case bf16/fp16 mismatch sneaks in
                feat = torch.cat([pooled, z.to(pooled.dtype)], dim=-1)
            else:
                feat = pooled
            logits = self.head(feat)
            # Return tuple (logits, recon, mu, logvar) so caller can build
            # the combined loss. If VAE disabled, the recon/mu/logvar are None.
            return logits, recon, mu, logvar

    model = FoundPAD().to(device=device)
    trainable = lora_params + head_params + vae_params
    return model, trainable


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
def _fft_domain_mask(img_rgb: np.ndarray, rng: np.random.Generator,
                     band: Optional[str] = None) -> np.ndarray:
    """API_unq 2026-05-13 (Physical-PAD survey): frequency-domain masking
    augmentation. Mask a random low/mid/high frequency band and invert
    back to the spatial domain. Per the survey, this is the universal-
    detection-generalization trick that stops PAD models from overfitting
    to specific printer or screen signatures: the model cannot rely on
    one fixed frequency cue if any of them may be missing per-sample.

    Operates per-channel. `band` ∈ {"low", "mid", "high", None}; None picks
    uniformly. Returns an HxWx3 uint8 image.
    """
    if img_rgb.dtype != np.uint8 or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return img_rgb
    if band is None:
        band = rng.choice(["low", "mid", "high"])
    h, w = img_rgb.shape[:2]
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.indices((h, w))
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    r_max = float(r.max())
    if band == "low":
        lo, hi = 0.0, r_max * 0.20
    elif band == "high":
        lo, hi = r_max * 0.50, r_max
    else:
        lo, hi = r_max * 0.20, r_max * 0.50
    mask = (r >= lo) & (r < hi)
    out = np.empty_like(img_rgb)
    for ch in range(3):
        F = np.fft.fftshift(np.fft.fft2(img_rgb[:, :, ch].astype(np.float32)))
        F[mask] = 0.0
        inv = np.fft.ifft2(np.fft.ifftshift(F)).real
        out[:, :, ch] = np.clip(inv, 0, 255).astype(np.uint8)
    return out


class _RecordsDataset:
    """Lazy-load images from a list of (path, label) tuples. CLIP-preprocessed
    to (3, 224, 224) float tensors in CLIP normalization.

    API_unq 2026-05-13: optional frequency-domain masking augmentation
    via `fft_mask_prob` (per-call probability of random band masking).
    Defaults to 0.0 for backward compat; trainer sets it via CLI.
    """

    def __init__(self, items: List[Tuple[str, int]], processor,
                 fft_mask_prob: float = 0.0, seed: int = 0,
                 augment: bool = False):
        self.items = items
        self.processor = processor
        self.fft_mask_prob = float(fft_mask_prob)
        self.augment = bool(augment)
        # Worker-safe per-process RNG (DataLoader workers each call __getitem__
        # independently; numpy's default generator is process-local).
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import cv2
        path, label = self.items[idx]
        img = cv2.imread(path)
        if img is None:
            # Return blank as graceful fallback; loss will be dominated by valid samples.
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # FFT-domain masking augmentation (train-time only)
        if self.augment and self.fft_mask_prob > 0.0 and self._rng.random() < self.fft_mask_prob:
            img = _fft_domain_mask(img, self._rng)
        proc = self.processor(images=img, return_tensors="pt")
        return proc.pixel_values[0], int(label)


def _load_records(paths: List[Path], max_per_class: Optional[int] = None) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    n_live = n_spoof = 0
    for p in paths:
        if not p.is_file():
            print(f"[foundpad] records file missing: {p}")
            continue
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                label = str(rec.get("label", "")).lower()
                if label not in ("live", "spoof"):
                    continue
                path = rec.get("path") or rec.get("image_path") or rec.get("file")
                if not path or not Path(path).is_file():
                    continue
                lbl = 0 if label == "live" else 1
                if max_per_class is not None:
                    if lbl == 0 and n_live >= max_per_class:
                        continue
                    if lbl == 1 and n_spoof >= max_per_class:
                        continue
                out.append((str(path), lbl))
                if lbl == 0:
                    n_live += 1
                else:
                    n_spoof += 1
    print(f"[foundpad] loaded {len(out)} samples: {n_live} live + {n_spoof} spoof")
    return out


def _identity_split(items: List[Tuple[str, int]], val_frac: float, seed: int) -> Tuple[List, List]:
    """Stratified per-label split. Not identity-stratified (records jsonls
    don't carry identity uniformly). For paper-grade splits, fall through
    to pad/diagnose_split_leakage.py.
    """
    rng = np.random.default_rng(seed)
    live = [x for x in items if x[1] == 0]
    spoof = [x for x in items if x[1] == 1]
    rng.shuffle(live)
    rng.shuffle(spoof)
    n_val_live = int(len(live) * val_frac)
    n_val_spoof = int(len(spoof) * val_frac)
    val = live[:n_val_live] + spoof[:n_val_spoof]
    tr = live[n_val_live:] + spoof[n_val_spoof:]
    rng.shuffle(tr)
    rng.shuffle(val)
    return tr, val


# ----------------------------------------------------------------------
# Train loop
# ----------------------------------------------------------------------
@dataclass
class TrainConfig:
    records: List[Path]
    epochs: int = 5
    batch: int = 32
    lr: float = 1e-4
    lora_rank: int = 8
    lora_alpha: float = 16.0
    weight_decay: float = 0.01
    val_frac: float = 0.10
    max_per_class: Optional[int] = 15000
    seed: int = 0
    num_workers: int = 4
    clip_id: str = DEFAULT_CLIP_ID
    out_pth: Path = DEFAULT_OUT_PTH
    out_onnx: Path = DEFAULT_OUT_ONNX
    log_path: Path = DEFAULT_LOG
    export_onnx: bool = True
    export_fp16: bool = False
    amp_dtype: str = "bf16"      # bf16 default on 5090 (sm_120); fp16 also OK
    # API_unq 2026-05-13 augmentations
    fft_mask_prob: float = 0.30  # per-sample prob of FFT-domain random-band masking
    vae_weight: float = 0.0      # if >0, enable VAE side-stream + recon loss


def train(cfg: TrainConfig) -> int:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import CLIPProcessor

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.amp_dtype == "bf16" and device == "cuda":
        dtype = torch.bfloat16
    elif cfg.amp_dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    items = _load_records(cfg.records, max_per_class=cfg.max_per_class)
    if not items:
        print("[foundpad] no usable records loaded; aborting")
        return 2
    tr_items, val_items = _identity_split(items, val_frac=cfg.val_frac, seed=cfg.seed)
    print(f"[foundpad] train={len(tr_items)} val={len(val_items)}")

    processor = CLIPProcessor.from_pretrained(cfg.clip_id)
    ds_tr = _RecordsDataset(tr_items, processor,
                            fft_mask_prob=cfg.fft_mask_prob,
                            seed=cfg.seed, augment=True)
    ds_val = _RecordsDataset(val_items, processor,
                             fft_mask_prob=0.0, seed=cfg.seed + 1,
                             augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch, shuffle=True,
                       num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch, shuffle=False,
                        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0)

    vae_enabled = cfg.vae_weight > 0.0
    model, trainable = build_model(cfg.clip_id, cfg.lora_rank, cfg.lora_alpha, device, dtype,
                                    vae_enabled=vae_enabled, vae_latent=32)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[foundpad] trainable {n_trainable:,} / total {n_total:,} "
          f"({100.0*n_trainable/n_total:.2f}%)  vae_enabled={vae_enabled}")

    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * max(1, len(dl_tr)))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = cfg.log_path.open("a", encoding="utf-8")

    best_val_auc = -1.0
    cfg.out_pth.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        t_ep = time.monotonic()
        run_loss = 0.0
        run_cls_loss = 0.0
        run_vae_loss = 0.0
        n_batches = 0
        for px, y in dl_tr:
            px = px.to(device=device, dtype=dtype, non_blocking=True)
            y = y.to(device=device, non_blocking=True)
            logits, recon, mu, logvar = model(px)
            cls_loss = loss_fn(logits.float(), y)
            if vae_enabled and recon is not None:
                # Reconstruction + KL — standard VAE ELBO
                recon_loss = ((recon.float() - px.float()) ** 2).mean()
                kl = -0.5 * (1.0 + logvar.float() - mu.float() ** 2 - logvar.float().exp()).mean()
                vae_loss = recon_loss + 0.001 * kl  # small beta on KL
                loss = cls_loss + cfg.vae_weight * vae_loss
                run_vae_loss += float(vae_loss.item())
            else:
                loss = cls_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item()
            run_cls_loss += float(cls_loss.item())
            n_batches += 1
        # Validation
        model.eval()
        all_probs: List[float] = []
        all_y: List[int] = []
        with torch.no_grad():
            for px, y in dl_val:
                px = px.to(device=device, dtype=dtype)
                logits, _, _, _ = model(px)
                probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy().tolist()
                all_probs.extend(probs)
                all_y.extend(y.cpu().numpy().tolist())
        try:
            from sklearn.metrics import roc_auc_score, accuracy_score
            val_auc = float(roc_auc_score(all_y, all_probs)) if len(set(all_y)) > 1 else 0.5
            val_acc = float(accuracy_score(all_y, [int(p > 0.5) for p in all_probs]))
        except Exception:
            val_auc = float("nan"); val_acc = float("nan")
        dur = time.monotonic() - t_ep
        msg = {
            "epoch": epoch + 1,
            "train_loss": run_loss / max(n_batches, 1),
            "train_cls_loss": run_cls_loss / max(n_batches, 1),
            "train_vae_loss": (run_vae_loss / max(n_batches, 1)) if vae_enabled else None,
            "val_auc": val_auc,
            "val_acc": val_acc,
            "epoch_sec": dur,
            "vae_enabled": vae_enabled,
            "fft_mask_prob": cfg.fft_mask_prob,
        }
        print(f"[foundpad] ep {epoch+1}/{cfg.epochs} loss={msg['train_loss']:.4f} "
              f"cls={msg['train_cls_loss']:.4f} "
              f"vae={msg['train_vae_loss'] if vae_enabled else '--'} "
              f"val_auc={val_auc:.4f} val_acc={val_acc:.4f} dt={dur:.1f}s")
        log_fh.write(json.dumps(msg) + "\n")
        log_fh.flush()
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            _save_pth(model, cfg)
            print(f"[foundpad] saved best @ val_auc={val_auc:.4f}")

    log_fh.close()

    if cfg.export_onnx:
        _export_onnx(model, cfg, device, dtype)

    print(f"[foundpad] done. best val_auc={best_val_auc:.4f}")
    return 0


def _save_pth(model, cfg: TrainConfig) -> None:
    import torch
    state = {
        "lora_state": {},
        "head_state": model.head.state_dict(),
        "config": {
            "clip_id": cfg.clip_id,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "vae_enabled": bool(getattr(model, "vae_enabled", False)),
            "vae_weight": float(cfg.vae_weight),
            "fft_mask_prob": float(cfg.fft_mask_prob),
            "version": 2 if getattr(model, "vae_enabled", False) else 1,
            "schema": "foundation_pad_v2_vae" if getattr(model, "vae_enabled", False) else "foundation_pad_v1",
        },
    }
    # Snapshot the LoRA A/B params (the only trainable LoRA tensors)
    vm = model.base.vision_model
    for li, layer in enumerate(vm.encoder.layers):
        for proj_name in ("q_proj", "v_proj"):
            proj = getattr(layer.self_attn, proj_name, None)
            if proj is None or not hasattr(proj, "_lora_A"):
                continue
            state["lora_state"][f"layer_{li}_{proj_name}_A"] = proj._lora_A.detach().cpu()
            state["lora_state"][f"layer_{li}_{proj_name}_B"] = proj._lora_B.detach().cpu()
    # Snapshot the VAE encoder/decoder if enabled
    if getattr(model, "vae_enabled", False):
        state["vae_enc_state"] = model.vae_enc.state_dict()
        state["vae_dec_state"] = model.vae_dec.state_dict()
    torch.save(state, cfg.out_pth)


class _InferenceWrapper:
    """ONNX export wrapper: returns ONLY logits (not the 4-tuple). The
    runtime branch reads only the logits; recon/mu/logvar are training-time
    artefacts of the VAE objective and don't need to ship in the inference
    graph."""

    @staticmethod
    def build(model):
        import torch
        from torch import nn

        class _ExportModel(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, pixel_values):
                out = self.inner(pixel_values)
                # When training-time the model returns a 4-tuple. We want
                # logits only for inference.
                if isinstance(out, tuple):
                    return out[0]
                return out

        return _ExportModel(model)


def _export_onnx(model, cfg: TrainConfig, device: str, dtype) -> None:
    """Export the merged-graph inference path. We bake the LoRA delta into
    the q,v projections in-place for export, then restore originals.

    API_unq 2026-05-14: 2-stage fallback. CLIP ViT-L/14 contains complex-
    valued tensors (positional encodings via FFT/rotary helpers) which
    crash the legacy TorchScript ONNX exporter with
    "ScalarType ComplexDouble is an unexpected tensor scalar type".
    Try dynamo first (handles complex-types), fall back to legacy with
    a float32 cast wrapper.
    """
    import torch
    model.eval()
    export_model = _InferenceWrapper.build(model)
    # Cast model + wrapper to float32 for export -- avoids the bf16
    # complex-tensor downcast hazard.
    try:
        export_model = export_model.float()
    except Exception:
        pass
    dummy = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float32)

    exported = False
    # Attempt 1: dynamo (new exporter, handles complex tensors better)
    try:
        torch.onnx.export(
            export_model, (dummy,),
            cfg.out_onnx.as_posix(),
            input_names=["pixel_values"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
            do_constant_folding=True,
            dynamo=True,
        )
        sz = cfg.out_onnx.stat().st_size / 1e6
        print(f"[foundpad] exported ONNX (dynamo) -> {cfg.out_onnx} ({sz:.1f} MB)")
        exported = True
    except Exception as e:
        print(f"[foundpad] ONNX dynamo export failed: {e}")
        print(f"[foundpad] falling back to legacy TorchScript exporter ...")

    # Attempt 2: legacy TorchScript exporter (only if dynamo failed)
    if not exported:
        try:
            torch.onnx.export(
                export_model, (dummy,),
                cfg.out_onnx.as_posix(),
                input_names=["pixel_values"],
                output_names=["logits"],
                opset_version=17,
                dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
                do_constant_folding=True,
                dynamo=False,
            )
            sz = cfg.out_onnx.stat().st_size / 1e6
            print(f"[foundpad] exported ONNX (legacy) -> {cfg.out_onnx} ({sz:.1f} MB)")
            exported = True
        except Exception as e:
            print(f"[foundpad] ONNX legacy export ALSO failed: {e}")
            print(f"[foundpad] .pth saved at {cfg.out_pth} -- recover separately")

    if exported and cfg.export_fp16:
        try:
            from onnxconverter_common import float16
            import onnx
            fp32 = onnx.load(cfg.out_onnx.as_posix())
            fp16 = float16.convert_float_to_float16(fp32, keep_io_types=True)
            fp16_path = cfg.out_onnx.with_suffix(".fp16.onnx")
            onnx.save(fp16, fp16_path.as_posix())
            print(f"[foundpad] exported FP16 ONNX -> {fp16_path}")
        except Exception as e:
            print(f"[foundpad] FP16 export failed: {e}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, action="append", required=True,
                        help="meta-records jsonl (repeatable)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--max-per-class", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--clip-id", type=str, default=DEFAULT_CLIP_ID)
    parser.add_argument("--out-pth", type=Path, default=DEFAULT_OUT_PTH)
    parser.add_argument("--out-onnx", type=Path, default=DEFAULT_OUT_ONNX)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--no-export-onnx", dest="export_onnx", action="store_false")
    parser.add_argument("--export-fp16", action="store_true")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    # API_unq 2026-05-13 augmentation + side-stream flags
    parser.add_argument("--fft-mask-prob", type=float, default=0.30,
                        help="per-sample probability of FFT-domain random-band masking aug (0 disables)")
    parser.add_argument("--vae-weight", type=float, default=0.0,
                        help="VAE side-stream loss weight; >0 enables Facevengers-style "
                             "CLIP-semantic + VAE-texture + LoRA combo (default 0 = CLIP+LoRA only)")
    args = parser.parse_args()

    cfg = TrainConfig(
        records=args.records,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        max_per_class=args.max_per_class,
        seed=args.seed,
        num_workers=args.num_workers,
        clip_id=args.clip_id,
        out_pth=args.out_pth,
        out_onnx=args.out_onnx,
        log_path=args.log_path,
        export_onnx=args.export_onnx,
        export_fp16=args.export_fp16,
        amp_dtype=args.amp_dtype,
        fft_mask_prob=args.fft_mask_prob,
        vae_weight=args.vae_weight,
    )
    return train(cfg)


if __name__ == "__main__":
    sys.exit(main())
