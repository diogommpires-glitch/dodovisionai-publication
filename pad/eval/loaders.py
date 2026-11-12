"""v9.4.1 step 7 — dataset loaders for the PAD eval harness.

Each loader yields records of the shape:

    {"label": "live" | "spoof",
     "attack_class": one of pad.types.ATTACK_CLASSES (spoof only),
     "path": absolute path to a frame or short clip,
     "dataset": dataset name}

Datasets are EULA-gated so the repo ships NO files. When the expected
directory layout is missing the loader yields nothing and prints one
"skip" line. Never raises. Never crashes.

Supported layouts (paths relative to `pad/datasets/`):

  OULU-NPU/
    Train_files/{1..4}_{1..6}_{01..XX}/*.avi|*.jpg    # filename ⇒ label
  SiW/
    Live/<subj>/*.mov,  Spoof/<subj>/<attack>/*.mov
  Replay-Attack/
    real/, attack/   (Idiap Replay-Attack)
  MSU-MFSD/
    scene01/ ...      (mentioned for completeness; .mov files)
  CASIA-SURF/
    Train/, Val/  (RGB+depth; we only use RGB here)
  OR-PAD/
    bonafide/, spoof/{print, replay}/
  FaceForensics++/
    original_sequences/, manipulated_sequences/<method>/
  ASVspoof2024/
    flac/ + protocols/  (audio; plumbed through `loader_asvspoof`)

The harness calls `discover(pad_datasets_dir)` and iterates whatever
datasets are actually present.
"""
from __future__ import annotations
import os
from typing import Iterator, Dict, List, Optional

# Light re-export so callers don't duplicate the class list
from pad.types import ATTACK_CLASSES


# ── utility ────────────────────────────────────────────────────────
def _walk_ext(root: str, exts):
    if not os.path.isdir(root):
        return
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            low = fn.lower()
            for e in exts:
                if low.endswith(e):
                    yield os.path.join(dp, fn)
                    break


def _print_skip(name: str, path: str) -> None:
    print(f"[pad/eval] dataset not found at expected path — skipping: "
          f"{name} ({path})")


# ── per-dataset loaders ────────────────────────────────────────────
def loader_oulu_npu(root: str) -> Iterator[Dict]:
    path = os.path.join(root, "OULU-NPU")
    if not os.path.isdir(path):
        _print_skip("OULU-NPU", path)
        return
    # OULU filenames encode: P_S_C.avi where S ∈ {1,2,3,4,5,6}.
    # S=1 ⇒ live; S=2,3 ⇒ print; S=4,5,6 ⇒ replay (broadly).
    for fp in _walk_ext(path, (".avi", ".mp4", ".jpg", ".png")):
        stem = os.path.splitext(os.path.basename(fp))[0]
        parts = stem.split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            s = int(parts[1])
            if s == 1:
                yield {"label": "live", "attack_class": None,
                       "path": fp, "dataset": "OULU-NPU"}
            elif s in (2, 3):
                yield {"label": "spoof", "attack_class": "PRINT",
                       "path": fp, "dataset": "OULU-NPU"}
            elif s in (4, 5, 6):
                yield {"label": "spoof", "attack_class": "REPLAY_SCREEN",
                       "path": fp, "dataset": "OULU-NPU"}


def loader_siw(root: str) -> Iterator[Dict]:
    path = os.path.join(root, "SiW")
    live_root = os.path.join(path, "Live")
    spoof_root = os.path.join(path, "Spoof")
    if not os.path.isdir(path):
        _print_skip("SiW", path)
        return
    if os.path.isdir(live_root):
        for fp in _walk_ext(live_root, (".mov", ".mp4", ".avi", ".jpg")):
            yield {"label": "live", "attack_class": None,
                   "path": fp, "dataset": "SiW"}
    if os.path.isdir(spoof_root):
        for fp in _walk_ext(spoof_root, (".mov", ".mp4", ".avi", ".jpg")):
            low = fp.lower()
            if "replay" in low:
                c = "REPLAY_SCREEN"
            elif "print" in low:
                c = "PRINT"
            elif "mask" in low:
                c = "MASK_3D"
            else:
                c = "UNKNOWN"
            yield {"label": "spoof", "attack_class": c,
                   "path": fp, "dataset": "SiW"}


def loader_replay_attack(root: str) -> Iterator[Dict]:
    path = os.path.join(root, "Replay-Attack")
    real = os.path.join(path, "real")
    atk = os.path.join(path, "attack")
    if not os.path.isdir(path):
        _print_skip("Replay-Attack", path)
        return
    for fp in _walk_ext(real, (".mov", ".mp4", ".avi")):
        yield {"label": "live", "attack_class": None,
               "path": fp, "dataset": "Replay-Attack"}
    for fp in _walk_ext(atk, (".mov", ".mp4", ".avi")):
        low = fp.lower()
        c = "PRINT" if "print" in low else (
            "REPLAY_SCREEN" if ("video" in low or "mobile" in low) else "UNKNOWN")
        yield {"label": "spoof", "attack_class": c,
               "path": fp, "dataset": "Replay-Attack"}


def loader_msu_mfsd(root: str) -> Iterator[Dict]:
    path = os.path.join(root, "MSU-MFSD")
    if not os.path.isdir(path):
        _print_skip("MSU-MFSD", path)
        return
    for fp in _walk_ext(path, (".mov", ".mp4", ".avi")):
        low = fp.lower()
        if "real" in low or "genuine" in low:
            yield {"label": "live", "attack_class": None,
                   "path": fp, "dataset": "MSU-MFSD"}
        else:
            c = "REPLAY_SCREEN" if "video" in low else "PRINT"
            yield {"label": "spoof", "attack_class": c,
                   "path": fp, "dataset": "MSU-MFSD"}


def loader_casia_surf(root: str) -> Iterator[Dict]:
    path = os.path.join(root, "CASIA-SURF")
    if not os.path.isdir(path):
        _print_skip("CASIA-SURF", path)
        return
    # CASIA-SURF labels via an associated .txt protocol; we do a
    # best-effort scan based on the standard directory names.
    for fp in _walk_ext(path, (".jpg", ".png", ".bmp")):
        parts = fp.replace("\\", "/").split("/")
        low = "/".join(parts).lower()
        if "real" in low or "live" in low:
            yield {"label": "live", "attack_class": None,
                   "path": fp, "dataset": "CASIA-SURF"}
        elif "fake" in low or "attack" in low or "spoof" in low:
            yield {"label": "spoof", "attack_class": "UNKNOWN",
                   "path": fp, "dataset": "CASIA-SURF"}


def loader_or_pad(root: str) -> Iterator[Dict]:
    path = os.path.join(root, "OR-PAD")
    if not os.path.isdir(path):
        _print_skip("OR-PAD", path)
        return
    bf = os.path.join(path, "bonafide")
    sp = os.path.join(path, "spoof")
    for fp in _walk_ext(bf, (".jpg", ".png", ".mp4")):
        yield {"label": "live", "attack_class": None,
               "path": fp, "dataset": "OR-PAD"}
    for fp in _walk_ext(sp, (".jpg", ".png", ".mp4")):
        low = fp.lower()
        if "print" in low:
            c = "PRINT"
        elif "replay" in low:
            c = "REPLAY_SCREEN"
        elif "mask" in low:
            c = "MASK_3D"
        else:
            c = "UNKNOWN"
        yield {"label": "spoof", "attack_class": c,
               "path": fp, "dataset": "OR-PAD"}


def loader_faceforensics(root: str) -> Iterator[Dict]:
    path = os.path.join(root, "FaceForensics++")
    if not os.path.isdir(path):
        _print_skip("FaceForensics++", path)
        return
    orig = os.path.join(path, "original_sequences")
    manip = os.path.join(path, "manipulated_sequences")
    for fp in _walk_ext(orig, (".mp4", ".jpg", ".png")):
        yield {"label": "live", "attack_class": None,
               "path": fp, "dataset": "FF++"}
    for fp in _walk_ext(manip, (".mp4", ".jpg", ".png")):
        yield {"label": "spoof", "attack_class": "DEEPFAKE_STREAM",
               "path": fp, "dataset": "FF++"}


def loader_asvspoof(root: str) -> Iterator[Dict]:
    path = os.path.join(root, "ASVspoof2024")
    if not os.path.isdir(path):
        _print_skip("ASVspoof2024", path)
        return
    protocols = os.path.join(path, "protocols")
    flacs = os.path.join(path, "flac")
    if not os.path.isdir(protocols) or not os.path.isdir(flacs):
        _print_skip("ASVspoof2024", path)
        return
    protocol_files = [f for f in os.listdir(protocols)
                      if f.endswith(".txt") or f.endswith(".trl")]
    for pf in protocol_files:
        with open(os.path.join(protocols, pf), "r", encoding="utf-8",
                  errors="ignore") as fh:
            for line in fh:
                toks = line.strip().split()
                if len(toks) < 2:
                    continue
                fid, lab = toks[1], toks[-1]
                wav = os.path.join(flacs, fid + ".flac")
                if not os.path.isfile(wav):
                    continue
                label = "live" if lab.lower() in ("bonafide", "live") else "spoof"
                attack = None if label == "live" else "AUDIO_DEEPFAKE"
                yield {"label": label, "attack_class": attack,
                       "path": wav, "dataset": "ASVspoof2024"}


# ── discovery ──────────────────────────────────────────────────────
_LOADERS = [
    loader_oulu_npu, loader_siw, loader_replay_attack, loader_msu_mfsd,
    loader_casia_surf, loader_or_pad, loader_faceforensics,
    loader_asvspoof,
]


def discover(pad_datasets_dir: str) -> Iterator[Dict]:
    """Yield records from every present dataset. Missing ones skip
    quietly. Order is stable across runs for reproducibility."""
    for loader in _LOADERS:
        try:
            yield from loader(pad_datasets_dir)
        except Exception as e:
            print(f"[pad/eval] loader {loader.__name__} error: {e}")
