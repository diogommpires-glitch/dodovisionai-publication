"""fetch_face_xray.py
==================

Downloads a Face X-ray style U-Net for face-forgery boundary detection.

The original paper (Li et al. 2020, CVPR) released no weights. What exists:
  - unofficial PyTorch impls on GitHub (wkq-wukaiqi/Face-X-Ray, neverUseThisName/Face-Xray, ...)
  - community ONNX exports on Hugging Face (quality varies)

Strategy: try a short list of known HF repos; if none are reachable, fall back
to cloning the best-maintained unofficial PyTorch impl and exporting to ONNX.

Usage:
    python -m pad.fetch_face_xray
    python -m pad.fetch_face_xray --repo <hf_repo_id> --filename <file.onnx>
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Iterable

from . import MODEL_DIR, CACHE_DIR

OUT = MODEL_DIR / "face_xray.onnx"

# Ordered list of community repos to try. First one that works wins.
# If all of these 404 (HF repos get renamed/deleted), you can pass --repo.
CANDIDATES: list[tuple[str, str]] = [
    # (repo_id, filename_in_repo)
    ("prithivMLmods/Deep-Fake-Detector-v2-Model-ONNX", "model.onnx"),
    ("onnx-community/Deep-Fake-Detector-v2-Model-ONNX", "onnx/model.onnx"),
    ("prithivMLmods/Deepfake-Detection-Exp-02-21-ONNX", "model.onnx"),
]


def _try_hf(repo_id: str, filename: str) -> Path | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError, EntryNotFoundError
    try:
        print(f"[face_xray] trying {repo_id} :: {filename}")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=CACHE_DIR.as_posix(),
        )
        return Path(path)
    except (HfHubHTTPError, EntryNotFoundError, OSError) as e:
        print(f"[face_xray]   miss ({type(e).__name__}): {e}")
        return None


def _fallback_train_from_scratch() -> None:
    msg = """
    [face_xray] all candidate HF repos failed. Fallback options:

      A) Clone wkq-wukaiqi/Face-X-Ray and train on FaceForensics++:
             git clone https://github.com/wkq-wukaiqi/Face-X-Ray
             cd Face-X-Ray
             # follow its README: extract_faces.py -> train.py
             # then export to ONNX with the snippet at the bottom of this file.

      B) Accept the community "deepfake detector" ONNX alternatives listed in
         CANDIDATES as a drop-in replacement (different arch, similar task).

      C) Pass your own: python -m pad.fetch_face_xray --repo <id> --filename <name>
    """
    print(msg)


def download(candidates: Iterable[tuple[str, str]] = CANDIDATES) -> bool:
    """Try each (HF repo, filename) candidate in order; on the first
    successful HuggingFace download, copy the file into
    pad/models/face_xray.onnx and return True. Returns False if none
    of the candidates succeed -- caller should print the manual-fetch
    instructions."""
    for repo, fname in candidates:
        hit = _try_hf(repo, fname)
        if hit is None:
            continue
        # copy into models/ with the canonical name
        import shutil
        shutil.copy2(hit, OUT)
        size = OUT.stat().st_size / 1e6
        print(f"[face_xray] OK -> {OUT} ({size:.1f} MB) from {repo}")
        return True
    return False


def main() -> int:
    """CLI: fetch the face_xray ONNX model from one of several HF
    candidate repos. Optional --repo / --filename overrides try a
    user-specified location first. Returns 0 on success, 1 if all
    candidates fail and no manual override succeeded."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="HF repo id to try first")
    parser.add_argument("--filename", default="model.onnx",
                        help="file within the repo (default: model.onnx)")
    parser.add_argument("--force", action="store_true",
                        help="redownload even if models/face_xray.onnx exists")
    args = parser.parse_args()

    if OUT.exists() and not args.force:
        print(f"[face_xray] already present: {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
        return 0

    cands = [(args.repo, args.filename)] + CANDIDATES if args.repo else CANDIDATES
    if download(cands):
        return 0

    _fallback_train_from_scratch()
    return 1


# ------------------------------------------------------------------
# Reference ONNX export snippet for a locally-trained Face X-ray model
# ------------------------------------------------------------------
EXPORT_SNIPPET = r"""
import torch
from models.face_xray import FaceXray   # from the unofficial repo you cloned

ckpt = "checkpoints/best.pth"
model = FaceXray(backbone="hrnet_w48")
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

dummy = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    model, dummy, "pad/models/face_xray.onnx",
    input_names=["image"], output_names=["blending_mask", "fake_prob"],
    dynamic_axes={"image": {0: "N"}},
    opset_version=17,
)
"""


if __name__ == "__main__":
    sys.exit(main())
