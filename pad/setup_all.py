"""setup_all.py
=============

One-shot orchestrator. Runs every fetch/train step and prints a final summary.
Non-zero exit if anything critical fails.

Usage:
    python -m pad.setup_all                    # sensible defaults
    python -m pad.setup_all --stub-patch-depth # write random-init patch+depth ONNX
    python -m pad.setup_all --audio-model aasist-l
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

from . import MODEL_DIR


STEPS = [
    "patch_depth",
    "face_xray",
    "audio_pad",
    "moire_gbt",
    "3dmm",
]


def _run(argv: list[str]) -> int:
    print(f"\n>>> {' '.join(argv)}")
    return subprocess.call([sys.executable] + argv)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stub-patch-depth", action="store_true",
                        help="write random-init patch+depth ONNX so pipeline loads")
    parser.add_argument("--audio-model", default="aasist-l",
                        choices=["aasist-l", "aasist", "rawnet2"])
    parser.add_argument("--skip", nargs="*", default=[], choices=STEPS,
                        help="steps to skip")
    args = parser.parse_args()

    results: dict[str, int] = {}

    if "patch_depth" not in args.skip:
        cmd = ["-m", "pad.fetch_patch_depth"]
        if args.stub_patch_depth:
            cmd.append("--export-stub")
        results["patch_depth"] = _run(cmd)

    if "face_xray" not in args.skip:
        results["face_xray"] = _run(["-m", "pad.fetch_face_xray"])

    if "audio_pad" not in args.skip:
        results["audio_pad"] = _run(
            ["-m", "pad.fetch_audio_pad", "--model", args.audio_model]
        )

    if "moire_gbt" not in args.skip:
        results["moire_gbt"] = _run(["-m", "pad.train_moire_gbt"])

    if "3dmm" not in args.skip:
        results["3dmm"] = _run(["-m", "pad.verify_3dmm"])

    # summary
    print("\n" + "=" * 60)
    print(f"{'step':<15} {'status':<10} artifact")
    print("-" * 60)
    artifacts = {
        "patch_depth": MODEL_DIR / "patch_depth.onnx",
        "face_xray":   MODEL_DIR / "face_xray.onnx",
        "audio_pad":   sorted(MODEL_DIR.glob("aasist*.pth"))
                       + sorted(MODEL_DIR.glob("rawnet2*.pth")),
        "moire_gbt":   MODEL_DIR / "moire_gbt.pkl",
        "3dmm":        "bundled with face-alignment",
    }
    for step in STEPS:
        if step in args.skip:
            print(f"{step:<15} {'skipped':<10} -")
            continue
        rc = results.get(step, -1)
        status = "OK" if rc == 0 else f"FAIL({rc})"
        art = artifacts[step]
        if isinstance(art, list):
            art_str = ", ".join(p.name for p in art) or "(none)"
        elif isinstance(art, Path):
            art_str = f"{art.name} ({art.stat().st_size/1e6:.1f} MB)" if art.exists() else "(missing)"
        else:
            art_str = str(art)
        print(f"{step:<15} {status:<10} {art_str}")
    print("=" * 60)

    # patch_depth is expected to fail with rc=1 if no stub + no trained model.
    # That's informational, not fatal. Fail the overall run only on other errors.
    hard_fails = [s for s, rc in results.items() if rc != 0 and s != "patch_depth"]
    return 1 if hard_fails else 0


if __name__ == "__main__":
    sys.exit(main())
