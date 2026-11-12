"""v9.4 PAD — model fetch helper.

Downloads the public ONNX weights needed by branches 4, 6, 7 into
pad/models/. When an upstream URL is down or gated behind EULA, prints
manual-install instructions.

Usage:
    python -m pad.fetch_models             # fetch everything feasible
    python -m pad.fetch_models --only xray
    python -m pad.fetch_models --list      # print required files + status

Files referenced by the branches:
    pad/models/patch_depth_fas.onnx   (step 4)
    pad/models/face_xray.onnx         (step 6)
    pad/models/deepfake_temporal.onnx (step 6, heavy, gated)
    pad/models/asvspoof_rawnet2.onnx  (step 7)
    pad/models/moire_gbt.txt          (step 3, LightGBM text format)
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Each entry: (filename, branch, default_url, note)
TARGETS = {
    "patch_depth": (
        "patch_depth_fas.onnx", "step 4",
        None,  # No public authoritative release — must train locally
        "Train via pad/TRAINING.md protocol OR convert a community weights checkpoint."
    ),
    "xray": (
        "face_xray.onnx", "step 6",
        None,
        "No official public Face X-ray release. Community ports exist on HF; search for "
        "'face-x-ray onnx'. Save the .onnx to pad/models/face_xray.onnx."
    ),
    "temporal": (
        "deepfake_temporal.onnx", "step 6",
        None,
        "Heavy video-swin / ViT-tiny deepfake head. Gated behind PAD_DEEPFAKE_HEAVY=1. "
        "See pad/TRAINING.md for export protocol from FaceForensics++ training runs."
    ),
    "audio": (
        "asvspoof_rawnet2.onnx", "step 7",
        None,
        "See https://www.asvspoof.org/workshop2024_program for baseline releases. "
        "Export RawNet2 or AASIST to ONNX and save to pad/models/asvspoof_rawnet2.onnx."
    ),
    "moire": (
        "moire_gbt.txt", "step 3",
        None,
        "Train a LightGBM GBT on MSU-MFSD + OULU-NPU replay splits; save with "
        "booster.save_model(path). Heuristic fallback is used when file is missing."
    ),
}


def list_status() -> Dict[str, str]:
    """Return {target_key: 'present'|'MISSING'} for every entry in TARGETS.
    Used by the --list flag and by setup_all.py to gate downstream steps."""
    out = {}
    for key, (fname, step, url, note) in TARGETS.items():
        p = MODELS_DIR / fname
        out[key] = "present" if p.exists() else "MISSING"
    return out


def fetch_one(key: str) -> bool:
    """Download one model from TARGETS into pad/models/. Returns True on
    success or if the file already exists. False on unknown key, missing
    URL (manual-fetch-only entries), or download failure -- the caller
    is expected to print the per-model `note` field for guidance."""
    entry = TARGETS.get(key)
    if entry is None:
        print(f"[fetch] unknown target: {key}")
        return False
    fname, step, url, note = entry
    dest = MODELS_DIR / fname
    if dest.exists():
        print(f"[fetch] {fname} already present ({dest})")
        return True
    if not url:
        print(f"[fetch] {fname} — no automated URL available.")
        print(f"        {note}")
        print(f"        target path: {dest}")
        return False
    import urllib.request
    print(f"[fetch] {url} -> {dest}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DodoVisionAI-v9.4/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out.write(chunk)
        print(f"[fetch] OK: {dest} ({dest.stat().st_size} bytes)")
        return True
    except Exception as e:
        print(f"[fetch] FAILED: {e}")
        print(f"[fetch] {note}")
        return False


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="only fetch this target (key from TARGETS)")
    p.add_argument("--list", action="store_true", help="print status and exit")
    args = p.parse_args(argv)
    if args.list:
        print("v9.4 PAD model inventory")
        print("=" * 50)
        status = list_status()
        for key, (fname, step, url, note) in TARGETS.items():
            print(f"  {key:12s} {step:10s} {status[key]:8s} {fname}")
        return 0
    keys = [args.only] if args.only else list(TARGETS.keys())
    failures = 0
    for k in keys:
        ok = fetch_one(k)
        if not ok:
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
