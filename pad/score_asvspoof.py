"""score_asvspoof.py -- score ASVspoof 2019 LA through audio_pad.onnx.

Writes a report-schema jsonl that `pad.calibrate` can fit a calibrator on.

Input assumptions
-----------------
ASVspoof 2019 LA lives at (defaults):
    --root  E:/pad_datasets/ASVspoof2019LA/LA/LA/

Under that, expected layout:
    ASVspoof2019_LA_train/flac/LA_T_*.flac
    ASVspoof2019_LA_dev/flac/LA_D_*.flac
    ASVspoof2019_LA_eval/flac/LA_E_*.flac
    ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.*.trn.txt   <- labels

Protocol format (space-separated):
    SPKR_ID  UTT_ID  -  -  bonafide   or   SPKR_ID  UTT_ID  -  SYSTEM  spoof

Usage
-----
    python -m pad.score_asvspoof \\
        --root E:\\pad_datasets\\ASVspoof2019LA\\LA\\LA \\
        --split train --limit 2000 \\
        --out reports/audio_asvspoof_records.jsonl
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


ONNX = Path(__file__).resolve().parent / "models" / "audio_pad.onnx"
CHUNK = 64600  # AASIST input samples at 16 kHz (~4.04 s)

PROTOCOL_FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev":   "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval":  "ASVspoof2019.LA.cm.eval.trl.txt",
}
FLAC_SUBDIRS = {
    "train": "ASVspoof2019_LA_train/flac",
    "dev":   "ASVspoof2019_LA_dev/flac",
    "eval":  "ASVspoof2019_LA_eval/flac",
}


def load_protocol(path: Path) -> List[Tuple[str, str]]:
    """Return [(utt_id, 'bonafide'|'spoof'), ...]."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utt_id, label = parts[1], parts[4].lower()
            if label in ("bonafide", "spoof"):
                out.append((utt_id, label))
    return out


def load_audio_chunk(flac: Path) -> np.ndarray:
    """Read FLAC -> mono float32 center-chunk of CHUNK samples."""
    try:
        import soundfile as sf
    except ImportError:
        raise SystemExit("[asvspoof] soundfile is required: pip install soundfile")
    audio, _sr = sf.read(str(flac))
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    if len(audio) >= CHUNK:
        start = (len(audio) - CHUNK) // 2
        return audio[start:start + CHUNK]
    padded = np.zeros(CHUNK, dtype=np.float32)
    padded[:len(audio)] = audio
    return padded


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="ASVspoof 2019 LA root (contains "
                         "ASVspoof2019_LA_{train,dev,eval}/ subdirs)")
    ap.add_argument("--split", choices=["train", "dev", "eval"], default="train")
    ap.add_argument("--limit", type=int, default=2000,
                    help="total utterances to score (50/50 stratified)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    proto_path = args.root / "ASVspoof2019_LA_cm_protocols" / PROTOCOL_FILES[args.split]
    flac_dir = args.root / FLAC_SUBDIRS[args.split]
    if not proto_path.is_file():
        print(f"[asvspoof] protocol not found: {proto_path}", file=sys.stderr)
        return 1
    if not flac_dir.is_dir():
        print(f"[asvspoof] flac dir not found: {flac_dir}", file=sys.stderr)
        return 1
    if not ONNX.is_file():
        print(f"[asvspoof] audio_pad.onnx missing at {ONNX}", file=sys.stderr)
        return 2

    proto = load_protocol(proto_path)
    print(f"[asvspoof] protocol has {len(proto):,} utterances")
    bona = [u for u in proto if u[1] == "bonafide"]
    spoof = [u for u in proto if u[1] == "spoof"]
    print(f"[asvspoof] bonafide={len(bona):,}  spoof={len(spoof):,}")

    rng = random.Random(args.seed)
    rng.shuffle(bona); rng.shuffle(spoof)
    half = max(1, args.limit // 2)
    picks = bona[:half] + spoof[:args.limit - half]
    rng.shuffle(picks)
    print(f"[asvspoof] scoring {len(picks):,} utterances")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_ok = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, (utt, label) in enumerate(picks):
            flac = flac_dir / f"{utt}.flac"
            if not flac.is_file():
                continue
            try:
                audio = load_audio_chunk(flac)
            except Exception as e:
                print(f"[asvspoof] skip {utt}: {e}", file=sys.stderr)
                continue
            try:
                out = sess.run(out_names, {input_name: audio[None, :]})
            except Exception as e:
                print(f"[asvspoof] onnx fail on {utt}: {e}", file=sys.stderr)
                continue
            logits = np.asarray(out[0]).reshape(-1)
            # AASIST-L output is [bonafide_logit, spoof_logit] per the
            # ASVspoof convention. Apply softmax and take spoof column.
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            p_attack_raw = float(probs[-1]) if probs.size >= 2 else 0.5
            # Apply per-branch calibration (consistent with image branches).
            try:
                from pad.calibrate import apply_calibration
                p_attack = apply_calibration("audio", p_attack_raw)
            except Exception:
                p_attack = p_attack_raw
            rec = {
                "path": str(flac),
                "label": "live" if label == "bonafide" else "spoof",
                "attack_class": None if label == "bonafide" else "ASV_SPOOF",
                "branches": [{
                    "name": "audio",
                    "p_attack": p_attack,
                    "reliability": 1.0,
                    "dominant_class": None,
                }],
            }
            fh.write(json.dumps(rec) + "\n")
            n_ok += 1
            if n_ok % 200 == 0:
                rate = n_ok / max(time.time() - t0, 0.001)
                print(f"[asvspoof] {n_ok} / {len(picks)}  {rate:.1f} utt/s")
    dt = time.time() - t0
    print(f"[asvspoof] done in {dt/60:.1f} min: {n_ok} records -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
