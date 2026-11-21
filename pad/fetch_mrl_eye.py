"""fetch_mrl_eye.py -- Download + extract the MRL Eye Dataset for blink training.

The MRL Eye Dataset (Vrekas et al. 2018) is the standard public corpus
for eye-state classification: 84,898 grayscale eye crops from 37 subjects,
labelled OPEN/CLOSED in the filename (5th underscore-separated digit).
~1.8 GB zipped; ~3.5 GB extracted.

We use it to train a tiny CNN (`pad/train_eye_state.py`) that adds a
third independent channel to `blink_detector.py` — robust to glasses
where the heuristic EAR + blendshape combo currently misses ~70 % of
blinks per the v9.7.5 comments in `blink_detector.py`.

Usage:
    python -m pad.fetch_mrl_eye                          # default: E:\\pad_datasets\\MRL
    python -m pad.fetch_mrl_eye --root D:\\datasets\\MRL  # custom dir
    python -m pad.fetch_mrl_eye --skip-extract           # download zip only

Filename format (per MRL site):
    s<subject>_<image>_<gender>_<glasses>_<eye_state>_<reflection>_<lighting>_<sensor>.png
    Index 4 (0-based) = eye_state, where 0 = closed, 1 = open.
"""
from __future__ import annotations
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_ROOT = Path(r"E:\pad_datasets\MRL")
ZIP_NAME = "mrlEyes_2018_01.zip"
URL = "http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip"
EXPECTED_BYTES_APPROX = 1_900_000_000  # ~1.8 GB; size check uses >= 0.9 of this


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[mrl_eye] GET {url}")
    print(f"[mrl_eye] -> {dest}")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        read = 0
        chunk = 1 << 20  # 1 MB
        last_pct = -1
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            read += len(buf)
            if total:
                pct = int(100 * read / total)
                if pct != last_pct and pct % 5 == 0:
                    mb = read / (1 << 20)
                    print(f"[mrl_eye] {pct:3d} %  ({mb:.0f} MB)")
                    last_pct = pct
    tmp.replace(dest)
    print(f"[mrl_eye] downloaded {dest.stat().st_size / (1 << 20):.0f} MB")


def _extract(zip_path: Path, root: Path) -> None:
    print(f"[mrl_eye] extracting -> {root}")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    print(f"[mrl_eye] extracted")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"target directory (default: {DEFAULT_ROOT})")
    ap.add_argument("--url", type=str, default=URL,
                    help="dataset download URL (override for mirror)")
    ap.add_argument("--skip-extract", action="store_true",
                    help="download zip only, leave extraction to user")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if zip already present")
    args = ap.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    zip_path = args.root / ZIP_NAME

    if args.force or not zip_path.exists():
        try:
            _download(args.url, zip_path)
        except Exception as e:
            print(f"[mrl_eye] FAIL: {e}", file=sys.stderr)
            print(f"[mrl_eye] You can manually download {args.url}", file=sys.stderr)
            print(f"[mrl_eye] and place at {zip_path}", file=sys.stderr)
            return 2
    else:
        print(f"[mrl_eye] zip already present: {zip_path} "
              f"({zip_path.stat().st_size / (1 << 20):.0f} MB) -- use --force to re-download")

    if zip_path.stat().st_size < int(EXPECTED_BYTES_APPROX * 0.9):
        print(f"[mrl_eye] WARN: zip suspiciously small "
              f"({zip_path.stat().st_size / (1 << 20):.0f} MB, expected ~1800 MB). "
              f"Likely a captive-portal HTML page, not the real dataset.", file=sys.stderr)

    if not args.skip_extract:
        marker = args.root / "mrlEyes_2018_01"
        if marker.exists() and any(marker.iterdir()):
            print(f"[mrl_eye] already extracted: {marker} -- skipping extract")
        else:
            _extract(zip_path, args.root)
    print(f"[mrl_eye] DONE. Train via:")
    print(f"    python -m pad.train_eye_state --data {args.root}\\mrlEyes_2018_01 ^")
    print(f"        --epochs 15 --batch 256 --out pad\\models\\eye_state_v1.onnx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
