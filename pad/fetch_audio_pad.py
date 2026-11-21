"""fetch_audio_pad.py
===================

Downloads AASIST-L (default) or RawNet2 pretrained weights from the official
clovaai/aasist repository and caches them under pad/models/.

AASIST-L is ~10 MB, matches the size budget. Full AASIST is ~35 MB.
Both are released under the clovaai license (research/non-commercial, check LICENSE).

Usage:
    python -m pad.fetch_audio_pad                  # default: AASIST-L
    python -m pad.fetch_audio_pad --model aasist   # full AASIST
    python -m pad.fetch_audio_pad --model rawnet2  # RawNet2 baseline
"""
from __future__ import annotations
import argparse
import sys
import urllib.request
from pathlib import Path

from . import MODEL_DIR

BASE = "https://raw.githubusercontent.com/clovaai/aasist/main/models/weights"

# (remote filename, local filename, approx size MB)
SOURCES: dict[str, tuple[str, str, float]] = {
    "aasist-l": ("AASIST-L.pth",           "aasist_l.pth",  0.9),   # tiny, ~900 KB
    "aasist":   ("AASIST.pth",             "aasist.pth",    1.1),
    # RawNet2 baseline weights are distributed via the ASVspoof2021 baseline repo:
    "rawnet2":  (
        "pre_trained_DF_RawNet2.pth",
        "rawnet2_df.pth",
        25.0,
    ),
}

RAWNET2_URL = (
    "https://raw.githubusercontent.com/asvspoof-challenge/2021/main/"
    "DF/Baseline-RawNet2/pre_trained_DF_RawNet2.pth"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[audio_pad] GET {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        read = 0
        while True:
            chunk = r.read(1 << 15)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if total:
                pct = 100 * read / total
                print(f"\r[audio_pad]   {read/1e6:6.2f} / {total/1e6:6.2f} MB  ({pct:5.1f}%)",
                      end="", flush=True)
    print()
    tmp.rename(dest)


def fetch(model_key: str) -> Path:
    """Download one audio-PAD weight file (key from SOURCES) into
    pad/models/. Returns the local destination Path on success.
    Raises ValueError for an unknown key. If the destination already
    exists, the download is skipped and the existing path returned --
    callers wanting a redownload must remove the file first."""
    if model_key not in SOURCES:
        raise ValueError(f"unknown model {model_key!r}; choose from {list(SOURCES)}")

    remote, local, _ = SOURCES[model_key]
    out = MODEL_DIR / local

    if out.exists():
        print(f"[audio_pad] already present: {out} ({out.stat().st_size/1e6:.1f} MB)")
        return out

    if model_key == "rawnet2":
        url = RAWNET2_URL
    else:
        url = f"{BASE}/{remote}"

    _download(url, out)
    print(f"[audio_pad] OK -> {out} ({out.stat().st_size/1e6:.1f} MB)")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="aasist-l",
                        choices=list(SOURCES.keys()),
                        help="which weight file to download")
    parser.add_argument("--all", action="store_true",
                        help="download every variant")
    args = parser.parse_args()

    keys = list(SOURCES.keys()) if args.all else [args.model]
    for k in keys:
        try:
            fetch(k)
        except Exception as e:
            print(f"[audio_pad] FAILED for {k}: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
