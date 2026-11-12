"""
DODO VISION AI v9.3 — FER model fetcher (plan step 6).

Downloads a small pre-trained Facial Emotion Recognition ONNX model into
`static/fer-mbnv3-int8.onnx` so the client-side ONNX Runtime Web path
(FF_CLIENT_EMO=1) has a model to load.

Usage:
  python fetch_fer_model.py                 # fetch the default model
  python fetch_fer_model.py --url <URL>     # override source
  python fetch_fer_model.py --check         # verify existing file only

The model path is referenced verbatim from ui.py's _initClientEmotion():
    ort.InferenceSession.create('/static/fer-mbnv3-int8.onnx', ...)

So the file MUST live at `<repo>/static/fer-mbnv3-int8.onnx`.

Notes on model choice:
  Any small (<10 MB) 7-class FER ONNX model works. We default to the
  HSEmotion ENet-B0 exports published at
  https://github.com/av-savchenko/face-emotion-recognition which are
  trained on AffectNet and release under Apache-2.0. If the upstream URL
  changes or is not reachable, the script prints install instructions
  so the operator can manually place a model in place.
"""
from __future__ import annotations
import argparse
import hashlib
import sys
from pathlib import Path

# Default source. This is ONE possible FER model; any ONNX with output
# dimension 7 (angry/disgust/fear/happy/neutral/sad/surprise) works.
DEFAULT_URL = "https://huggingface.co/av-savchenko/enet_b0_7_affectnet/resolve/main/enet_b0_7.onnx"
TARGET_DIR = Path(__file__).parent / "static"
TARGET_NAME = "fer-mbnv3-int8.onnx"


def _target_path() -> Path:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    return TARGET_DIR / TARGET_NAME


def check_existing() -> bool:
    p = _target_path()
    if not p.exists():
        print(f"[--] no model at {p}")
        return False
    size_mb = p.stat().st_size / (1024 * 1024)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    print(f"[OK] model found: {p}")
    print(f"     size: {size_mb:.2f} MB")
    print(f"     sha256[:16]: {sha}")
    return True


def fetch(url: str) -> bool:
    import urllib.request
    target = _target_path()
    print(f"[fetch] {url}")
    print(f"[fetch] -> {target}")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "DodoVisionAI-v9.3-model-fetcher/1.0",
        })
        with urllib.request.urlopen(req, timeout=120) as resp, open(target, "wb") as out:
            total = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
                sys.stdout.write(f"\r  {total / (1024*1024):.1f} MB...")
                sys.stdout.flush()
        print()
        return check_existing()
    except Exception as e:
        print(f"\n[FAIL] download failed: {e}")
        print()
        print("Manual install:")
        print("  1. Download any 7-class FER ONNX model (e.g. HSEmotion ENet-B0).")
        print(f"  2. Save it as: {target}")
        print(f"  3. Restart the server — the client will load /static/{TARGET_NAME}")
        print()
        print("Alternative source URLs:")
        print("  - https://github.com/av-savchenko/face-emotion-recognition")
        print("  - Any model exported from HSEmotion/EmotiEffLib with ONNX backend")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch FER ONNX model for client-side emotion (v9.3 step 6)")
    parser.add_argument("--url", default=DEFAULT_URL, help="Source URL for the .onnx model")
    parser.add_argument("--check", action="store_true", help="Only verify the existing model file")
    args = parser.parse_args()
    if args.check:
        sys.exit(0 if check_existing() else 1)
    ok = fetch(args.url)
    sys.exit(0 if ok else 1)
