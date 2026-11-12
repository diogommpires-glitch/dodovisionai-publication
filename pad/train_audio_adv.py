"""pad/train_audio_adv.py  --  adversarially-trained audio PAD (AASIST).

Fine-tunes the existing AASIST-L checkpoint (`pad/models/aasist_l.pth`)
with FGSM adversarial perturbations applied at the waveform level.
Produces a hardened `pad/models/audio_pad_adv.onnx` compatible with the
audio branch's ONNX loader.

Recipe:
    * Reads ASVspoof 2019 LA train protocol + FLACs.
    * Builds a PyTorch DataLoader that serves (waveform [64600], label {0,1}).
    * Each batch: optionally replace with FGSM adversarial counterpart.
        adv waveform = clamp(raw + eps * sign(grad), -1, 1)
    * Loss = CrossEntropy on the 2-class logits.
    * Validates on dev split each epoch; reports clean vs eps=2/100 vs eps=5/100.

Usage:
    python -u -m pad.train_audio_adv \\
        --root E:\\pad_datasets\\ASVspoof2019LA\\LA\\LA \\
        --epochs 5 --batch 16 --adv-ratio 0.5 --epsilon 2

Notes:
    * Epsilon is in units of 1/100 of waveform dynamic range (float32 [-1, 1]).
      eps=2 -> 0.02 amplitude perturbation (imperceptible by a human ear).
    * 5 epochs on 25k utterances at batch 16 -> ~2-3 h on RTX 3060.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np


REPO = Path(__file__).resolve().parent.parent
AASIST_SRC = REPO / "pad" / "models" / "aasist_src"
PTH = REPO / "pad" / "models" / "aasist_l.pth"
CHUNK = 64600

PROTOCOL_FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev":   "ASVspoof2019.LA.cm.dev.trl.txt",
}
FLAC_SUBDIRS = {
    "train": "ASVspoof2019_LA_train/flac",
    "dev":   "ASVspoof2019_LA_dev/flac",
}


def _load_protocol(path: Path) -> List[Tuple[str, int]]:
    """Return list of (utt_id, label_int).  label_int: 0 = bonafide, 1 = spoof."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utt = parts[1]
            lbl = 0 if parts[4].lower() == "bonafide" else 1
            out.append((utt, lbl))
    return out


def _load_audio(flac: Path) -> np.ndarray:
    """Read FLAC -> mono float32 center-chunk of CHUNK samples."""
    import soundfile as sf
    audio, _ = sf.read(str(flac))
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    if len(audio) >= CHUNK:
        start = (len(audio) - CHUNK) // 2
        return audio[start:start + CHUNK]
    padded = np.zeros(CHUNK, dtype=np.float32)
    padded[:len(audio)] = audio
    return padded


def _build_aasist():
    """Instantiate clovaai AASIST from aasist_src repo, return Model(d_args)."""
    import importlib.util, json as _json
    arch_path = AASIST_SRC / "models" / "AASIST.py"
    if not arch_path.is_file():
        raise SystemExit(f"[audio-adv] missing arch file: {arch_path}. "
                         f"git clone https://github.com/clovaai/aasist {AASIST_SRC}")
    spec = importlib.util.spec_from_file_location("aasist_arch", arch_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for conf_name in ("AASIST-L.conf", "AASIST.conf"):
        conf_path = AASIST_SRC / "config" / conf_name
        if conf_path.exists():
            with open(conf_path, "r", encoding="utf-8") as fh:
                conf = _json.load(fh)
            d_args = conf.get("model_config") or conf
            return mod.Model(d_args)
    raise SystemExit("[audio-adv] no AASIST config found")


def _dataset(root: Path, split: str, limit: int, seed: int):
    """Return list of (flac_path, label) pairs."""
    proto = _load_protocol(root / "ASVspoof2019_LA_cm_protocols" / PROTOCOL_FILES[split])
    flac_dir = root / FLAC_SUBDIRS[split]
    rng = random.Random(seed)
    bona = [(flac_dir / f"{u}.flac", y) for (u, y) in proto if y == 0]
    spoof = [(flac_dir / f"{u}.flac", y) for (u, y) in proto if y == 1]
    rng.shuffle(bona); rng.shuffle(spoof)
    half = limit // 2
    return bona[:half] + spoof[:limit - half]


class _AudioDataset:
    """v9.10: replaces the serial Python `_dataloader` with a torch
    `Dataset` so a real multi-process `DataLoader` can fan FLAC decode
    out across all CPU cores. v9.9 path pegged 1 of 24 cores at 100%
    while the GPU sat at <5% util because `soundfile.read` was on the
    critical path. With num_workers=8 the audio_adv phase is ~4x
    faster on the same Legion box.

    Implemented as a duck-typed dataset (not subclassing
    torch.utils.data.Dataset directly) so this module imports without
    requiring torch at parse time -- the train CLI does the import."""
    def __init__(self, items):
        self._items = list(items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        import torch
        p, y = self._items[idx]
        wav = _load_audio(p)        # numpy float32 [CHUNK]
        return torch.from_numpy(wav), int(y)


def _dataloader(items, batch_size: int, num_workers: int = 8,
                shuffle: bool = False):
    """v9.10: returns a real torch DataLoader. The previous (v9.9)
    implementation was a Python `for i in range(...)` generator that
    decoded FLAC serially; we keep the same call signature so existing
    callers don't change, but iterating yields (Tensor[B,CHUNK], Tensor[B])
    rather than (numpy, numpy). Callers must drop their `torch.from_numpy`
    conversions accordingly."""
    import torch
    ds = _AudioDataset(items)
    pin = torch.cuda.is_available()
    dl_kw = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin,
    }
    if num_workers > 0:
        dl_kw["persistent_workers"] = True
        dl_kw["prefetch_factor"] = 4
    return torch.utils.data.DataLoader(ds, **dl_kw)


def pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def _unpack_logits(out):
    """AASIST forward() returns a (logits, embedding) tuple in some
    builds and a bare logits tensor in others. Tolerate both.
    Patched 2026-04-25 after queue_mega stage 17 abort
    (cross_entropy_loss expected Tensor, got tuple).
    """
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def fgsm_wav(model, x, y, eps_amp):
    """One-step waveform perturbation in amplitude units."""
    import torch
    x_req = x.clone().detach().requires_grad_(True)
    logits = _unpack_logits(model(x_req))
    loss = torch.nn.CrossEntropyLoss()(logits, y)
    grad = torch.autograd.grad(loss, x_req)[0]
    return (x_req + eps_amp * grad.sign()).clamp(-1.0, 1.0).detach()


def evaluate(model, loader, device, epsilon_amp=0.0):
    """v9.10: `loader` now yields torch tensors (was numpy arrays in v9.9
    via the serial `_dataloader` generator). The `.to(device, non_blocking=True)`
    is the only conversion needed here."""
    import torch
    model.eval()
    total, correct = 0, 0
    for ws, ys in loader:
        x = ws.to(device, non_blocking=True)
        y = ys.to(device, non_blocking=True)
        if epsilon_amp > 0:
            with torch.enable_grad():
                x = fgsm_wav(model, x, y, epsilon_amp)
        with torch.no_grad():
            logits = _unpack_logits(model(x))
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / max(total, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="ASVspoof 2019 LA root (contains train/dev/ + protocols)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-limit", type=int, default=8000)
    ap.add_argument("--dev-limit", type=int, default=2000)
    ap.add_argument("--adv-ratio", type=float, default=0.5)
    ap.add_argument("--epsilon", type=int, default=2,
                    help="adv perturbation in units of 1/100 of waveform range")
    ap.add_argument("--out", type=Path,
                    default=Path("pad/models/audio_pad_adv.onnx"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    try:
        import torch
        import soundfile as sf  # noqa
    except ImportError as e:
        print(f"[audio-adv] missing dep: {e}")
        return 2

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device, dev_label = pick_device()
    print(f"[audio-adv] device: {dev_label}", flush=True)

    print("[audio-adv] building AASIST model ...", flush=True)
    model = _build_aasist().to(device)
    if PTH.is_file():
        state = torch.load(str(PTH), map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        miss, unexp = model.load_state_dict(state, strict=False)
        print(f"[audio-adv] warm-started from {PTH} "
              f"(missing={len(miss)}, unexpected={len(unexp)})", flush=True)
    else:
        print(f"[audio-adv] no .pth at {PTH}; training from ImageNet-pretrained", flush=True)

    print(f"[audio-adv] loading ASVspoof train ({args.train_limit}) + dev ({args.dev_limit}) ...",
          flush=True)
    train_items = _dataset(args.root, "train", args.train_limit, args.seed)
    dev_items   = _dataset(args.root, "dev",   args.dev_limit,   args.seed + 1)
    print(f"[audio-adv] {len(train_items)} train  /  {len(dev_items)} dev utterances",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()
    eps_amp = float(args.epsilon) / 100.0
    rng = random.Random(args.seed)

    report_dir = REPO / "pad" / "reports" / "audio_adv"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(report_dir / "train_log.txt", "w", encoding="utf-8")

    # v9.10: build DataLoaders ONCE outside the epoch loop so the
    # `persistent_workers=True` setting actually saves the worker
    # spin-up cost. Train loader uses shuffle=True; dev loader doesn't.
    train_loader = _dataloader(train_items, args.batch,
                               num_workers=8, shuffle=True)
    dev_loader = _dataloader(dev_items, args.batch,
                             num_workers=4, shuffle=False)
    best_adv = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total, correct, loss_sum, n_adv = 0, 0, 0.0, 0
        for ws, ys in train_loader:
            x = ws.to(device, non_blocking=True)
            y = ys.to(device, non_blocking=True)
            if rng.random() < args.adv_ratio:
                x = fgsm_wav(model, x, y, eps_amp)
                n_adv += x.size(0)
            logits = _unpack_logits(model(x))
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss) * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        clean_dev = evaluate(model, dev_loader, device, 0.0)
        adv2_dev  = evaluate(model, dev_loader, device, 0.02)
        adv5_dev  = evaluate(model, dev_loader, device, 0.05)
        dt = time.time() - t0
        line = (f"[audio-adv ep {epoch}/{args.epochs}] "
                f"train_loss={loss_sum/total:.4f} acc={correct/total:.3f} "
                f"adv_pct={n_adv/total:.2f}  "
                f"clean_val={clean_dev:.3f} adv2={adv2_dev:.3f} adv5={adv5_dev:.3f}  "
                f"({dt:.1f}s)")
        print(line, flush=True); log_f.write(line + "\n"); log_f.flush()
        if adv5_dev > best_adv:
            best_adv = adv5_dev
            torch.save(model.state_dict(), str(args.out.with_suffix(".pth")))

    log_f.close()

    # Export to ONNX
    print("[audio-adv] exporting ONNX ...", flush=True)
    model.eval().cpu()
    dummy = torch.randn(1, CHUNK)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False forces legacy TorchScript exporter (the new dynamo
    # exporter prints unicode ✅ banners that crash on Windows cp1252).
    torch.onnx.export(
        model, dummy, str(args.out),
        input_names=["waveform"], output_names=["logits"],
        dynamic_axes={"waveform": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    size_mb = args.out.stat().st_size / 1e6
    print(f"[audio-adv] wrote {args.out} ({size_mb:.1f} MB)", flush=True)
    print(f"[audio-adv] best eps=5/100 val_acc: {best_adv:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
