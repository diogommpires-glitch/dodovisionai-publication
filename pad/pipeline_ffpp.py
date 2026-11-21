"""pipeline_ffpp.py
==================

One-shot orchestrator: FaceForensics++ download -> face crops ->
patch/deepfake classifier -> ONNX export. Resume-safe at every stage.

The FF++ author's email (2026-04-19) says:

    "The only available server is EU2. Please execute the script with
     '--server EU2' for now, e.g., python download.py --server EU2 --c c40."

This script honours that. Defaults: EU2 server, c40 compression
(~15 GB total), 30 frames/video, MobileNetV3-small via the existing
`pad.train_patch_depth` trainer.

Files / state:
  FFPP_ROOT/
      faceforensics_download.py            <- you drop this (from the access email)
      original_sequences/...                <- created by the download
      manipulated_sequences/...
  CROPS_ROOT/
      live/ , spoof/ , _manifest.jsonl     <- written by stage 2
  PAD_MODELS/
      face_xray.onnx                        <- overwritten by stage 3

Each stage writes a completion marker to
    <state_dir>/.stage{1,2,3}.done
so re-runs skip finished stages.

Usage (real run):
    python -m pad.pipeline_ffpp \\
        --ffpp-root E:\\pad_datasets\\FaceForensics \\
        --crops-root E:\\pad_datasets\\face_xray \\
        --download-script E:\\pad_datasets\\FaceForensics\\faceforensics_download.py \\
        --server EU2 --compression c40 --epochs 20 \\
        --log E:\\pad_datasets\\pipeline_ffpp.log

Usage (smoke — exercise every stage on synthetic data, no network):
    python -m pad.pipeline_ffpp --smoke

The smoke run completes in ~30 s and verifies the full handoff
(download-script stub -> preprocess -> train 1 epoch -> ONNX).
"""
from __future__ import annotations
import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "pad" / "models"


# ------------------------------------------------------------------
#  shared helpers
# ------------------------------------------------------------------

def _log(logf, msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp}  {msg}"
    print(line, flush=True)
    if logf:
        try:
            logf.write(line + "\n"); logf.flush()
        except Exception:
            pass


def _marker(state_dir: Path, name: str) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f".{name}.done"


def _run(cmd: List[str], cwd: Optional[Path], logf,
         stdin_feed: Optional[str] = None) -> int:
    _log(logf, "RUN: " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None,
                            stdin=subprocess.PIPE if stdin_feed is not None else None,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    if stdin_feed is not None and proc.stdin is not None:
        try:
            proc.stdin.write(stdin_feed)
            proc.stdin.flush()
            proc.stdin.close()
        except Exception as e:
            _log(logf, f"  stdin feed failed: {e}")
    assert proc.stdout is not None
    for line in proc.stdout:
        _log(logf, "  > " + line.rstrip())
    proc.wait()
    _log(logf, f"EXIT: {proc.returncode}")
    return proc.returncode


# ------------------------------------------------------------------
#  stage 1 — download (wraps the official FF++ script)
# ------------------------------------------------------------------

# Dataset keys from the official faceforensics_download_v4.py. "original"
# is the pristine actor videos; the four manipulations are the core FF++
# fakes. We deliberately omit "original_youtube_videos" (the 40 GB raw
# YouTube zip), "DeepFakeDetection*" (Google's actor dataset — extra
# ~20 GB, not needed for Face X-ray), and "FaceShifter" (optional,
# masks are unavailable for it).
FFPP_DATASETS = ("original", "Deepfakes", "Face2Face",
                 "FaceSwap", "NeuralTextures")


def stage1_download(ffpp_root: Path, download_script: Path,
                    server: str, compression: str,
                    datasets: List[str], python_exe: str,
                    state_dir: Path, logf) -> int:
    mark = _marker(state_dir, "stage1")
    if mark.exists():
        _log(logf, "[stage1] already done — skipping")
        return 0
    if not download_script.is_file():
        _log(logf, f"[stage1] FATAL: download script not found at "
             f"{download_script}. Save the script from the FF++ access "
             f"email to that path and re-run.")
        return 1
    ffpp_root.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        cmd = [python_exe, str(download_script), str(ffpp_root),
               "-d", ds, "-c", compression, "--server", server]
        # The official script shows a TOS banner and calls input('') —
        # feed a newline so the subprocess doesn't hang.
        rc = _run(cmd, ffpp_root, logf, stdin_feed="\n")
        if rc != 0:
            _log(logf, f"[stage1] download of {ds} failed (rc={rc})")
            return rc
    mark.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    _log(logf, "[stage1] DONE")
    return 0


# ------------------------------------------------------------------
#  stage 2 — preprocess (video -> face crops)
# ------------------------------------------------------------------

def stage2_preprocess(ffpp_root: Path, crops_root: Path, compression: str,
                      frames_per_video: int, python_exe: str,
                      state_dir: Path, logf) -> int:
    mark = _marker(state_dir, "stage2")
    if mark.exists():
        _log(logf, "[stage2] already done — skipping")
        return 0
    cmd = [python_exe, "-u", "-m", "pad.preprocess_ffpp",
           "--ffpp", str(ffpp_root),
           "--out",  str(crops_root),
           "--frames-per-video", str(frames_per_video),
           "--compression", compression]
    rc = _run(cmd, REPO_ROOT, logf)
    if rc != 0:
        return rc
    # Sanity: there must be both live and spoof crops or the trainer fails.
    n_live = len(list((crops_root / "live").glob("*.jpg"))) if (crops_root / "live").is_dir() else 0
    n_spoof = len(list((crops_root / "spoof").glob("*.jpg"))) if (crops_root / "spoof").is_dir() else 0
    _log(logf, f"[stage2] crops: {n_live} live / {n_spoof} spoof")
    if n_live == 0 or n_spoof == 0:
        _log(logf, "[stage2] FAIL: need both live and spoof crops")
        return 2
    mark.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    _log(logf, "[stage2] DONE")
    return 0


# ------------------------------------------------------------------
#  stage 3 — train (reuses pad.train_patch_depth + auto-copy to face_xray.onnx)
# ------------------------------------------------------------------

def stage3_train(crops_root: Path, out_onnx: Path, epochs: int,
                 batch: int, python_exe: str, state_dir: Path, logf) -> int:
    mark = _marker(state_dir, "stage3")
    if mark.exists():
        _log(logf, "[stage3] already done — skipping")
        return 0
    # The trainer auto-copies its --out to a sibling named patch_depth_fas.onnx
    # (or patch_depth.onnx if --out already IS patch_depth_fas). Route the
    # interim to an isolated subdir so that copy doesn't clobber the live
    # patch_depth_fas.onnx the user already has trained (val_acc 0.999).
    interim = DEFAULT_MODELS_DIR / "_ffpp_interim" / "ffpp_classifier.onnx"
    interim.parent.mkdir(parents=True, exist_ok=True)
    cmd = [python_exe, "-u", "-m", "pad.train_patch_depth",
           "--data", str(crops_root),
           "--out",  str(interim),
           "--epochs", str(epochs),
           "--batch",  str(batch),
           "--log-every", "50"]
    rc = _run(cmd, REPO_ROOT, logf)
    if rc != 0:
        return rc
    if not interim.is_file():
        _log(logf, f"[stage3] FAIL: {interim} not produced by trainer")
        return 3
    # Deploy: copy the trained ONNX over face_xray.onnx. The deepfake
    # branch auto-detects the HF-ViT-224 preprocessing (n_classes==2,
    # 224×224), which is exactly what pad.train_patch_depth emits.
    import shutil
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    if out_onnx.is_file():
        backup = out_onnx.with_suffix(".onnx.prev")
        shutil.copy2(out_onnx, backup)
        _log(logf, f"[stage3] backed up previous ONNX -> {backup}")
    shutil.copy2(interim, out_onnx)
    _log(logf, f"[stage3] deployed trained ONNX -> {out_onnx} "
         f"({out_onnx.stat().st_size/1e6:.1f} MB)")
    mark.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    _log(logf, "[stage3] DONE")
    return 0


# ------------------------------------------------------------------
#  smoke path — synthetic FF++ tree + tiny training budget
# ------------------------------------------------------------------

def _smoke(python_exe: str) -> int:
    import tempfile, cv2, numpy as np
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        ffpp = td_p / "FF"
        crops = td_p / "crops"
        state = td_p / "state"
        models = td_p / "models"
        logp = td_p / "pipeline.log"
        logf = open(logp, "w", encoding="utf-8")
        _log(logf, "[smoke] synthesising a tiny FF++ tree")
        # Build a download-script stub that just materialises fake videos.
        stub = ffpp / "faceforensics_download.py"
        ffpp.mkdir(parents=True, exist_ok=True)
        stub.write_text(_SMOKE_STUB)
        rc = stage1_download(ffpp, stub, server="EU2", compression="c40",
                             datasets=["original", "Deepfakes"],
                             python_exe=python_exe, state_dir=state, logf=logf)
        if rc != 0:
            _log(logf, "[smoke] stage1 failed"); return rc
        rc = stage2_preprocess(ffpp, crops, compression="c40",
                               frames_per_video=4, python_exe=python_exe,
                               state_dir=state, logf=logf)
        if rc != 0:
            _log(logf, "[smoke] stage2 failed"); return rc
        # For training we route the interim ONNX into tmp models dir
        # so the smoke run doesn't overwrite the real face_xray.onnx.
        out_onnx = models / "face_xray.onnx"
        # Override DEFAULT_MODELS_DIR for this call by passing interim via env var.
        os.environ["PAD_PIPELINE_INTERIM_DIR"] = str(models)
        rc = _stage3_train_smoke(crops, out_onnx, epochs=1, batch=8,
                                 python_exe=python_exe, state_dir=state, logf=logf)
        if rc != 0:
            _log(logf, "[smoke] stage3 failed"); return rc
        if not out_onnx.is_file():
            _log(logf, "[smoke] FAIL: ONNX not written"); return 1
        _log(logf, f"[smoke] OK — ONNX at {out_onnx} "
             f"({out_onnx.stat().st_size/1e6:.1f} MB)")
        logf.close()
        return 0


def _stage3_train_smoke(crops_root, out_onnx, epochs, batch,
                        python_exe, state_dir, logf):
    """Like stage3_train but routes the interim ONNX through
    PAD_PIPELINE_INTERIM_DIR (smoke-only — avoids clobbering the real
    pad/models/patch_depth_fas.onnx via the trainer's sibling copy)."""
    interim_dir = Path(os.environ.get("PAD_PIPELINE_INTERIM_DIR",
                                       str(DEFAULT_MODELS_DIR / "_ffpp_interim")))
    interim_dir.mkdir(parents=True, exist_ok=True)
    interim = interim_dir / "ffpp_classifier.onnx"
    cmd = [python_exe, "-u", "-m", "pad.train_patch_depth",
           "--data", str(crops_root),
           "--out",  str(interim),
           "--epochs", str(epochs),
           "--batch",  str(batch),
           "--log-every", "1"]
    rc = _run(cmd, REPO_ROOT, logf)
    if rc != 0:
        return rc
    if not interim.is_file():
        return 3
    import shutil
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(interim, out_onnx)
    return 0


# A minimal download-script stub used by _smoke; writes a couple of
# tiny MP4s into the proper FF++ layout and returns 0.
_SMOKE_STUB = r"""
import sys, os, cv2, numpy as np
from pathlib import Path

def main():
    out = Path(sys.argv[1])
    args = sys.argv[2:]
    dataset = None
    compression = "c40"
    for i, a in enumerate(args):
        if a == "-d" and i+1 < len(args): dataset = args[i+1]
        if a == "-c" and i+1 < len(args): compression = args[i+1]
    if dataset is None:
        print("stub: no -d"); return 1
    if dataset == "original":
        rel = f"original_sequences/youtube/{compression}/videos"
        color = (200, 180, 180)
    else:
        rel = f"manipulated_sequences/{dataset}/{compression}/videos"
        color = (150, 140, 170)
    target = out / rel
    target.mkdir(parents=True, exist_ok=True)
    for j in range(3):
        p = target / f"vid{j:03d}.mp4"
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"),
                             15.0, (320, 240))
        for k in range(8):
            f = np.full((240, 320, 3), color, dtype=np.uint8)
            cv2.ellipse(f, (160, 120), (70, 90), 0, 0, 360, (210, 190, 180), -1)
            cv2.circle(f, (140, 105), 7, (30,30,30), -1)
            cv2.circle(f, (180, 105), 7, (30,30,30), -1)
            vw.write(f)
        vw.release()
    print(f"stub: wrote 3 videos to {target}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""


# ------------------------------------------------------------------
#  entrypoint
# ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ffpp-root",
                    help="where the FF++ download will land / already lives")
    ap.add_argument("--download-script",
                    help="path to the faceforensics_download.py you received "
                         "in the access email")
    ap.add_argument("--server", default="EU2",
                    help="FF++ server; per the 2026-04-19 email only EU2 "
                         "is available")
    ap.add_argument("--compression", default="c40",
                    choices=["c0", "c23", "c40"],
                    help="c40 = smallest (~15 GB); c23 HQ (~40 GB); c0 raw (~500 GB)")
    ap.add_argument("--datasets", nargs="*", default=list(FFPP_DATASETS),
                    help="subset of FFPP_DATASETS to download")
    ap.add_argument("--crops-root",
                    help="where live/ + spoof/ face crops go")
    ap.add_argument("--frames-per-video", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out-onnx", default=str(DEFAULT_MODELS_DIR / "face_xray.onnx"),
                    help="final ONNX path (deepfake branch looks at face_xray.onnx)")
    ap.add_argument("--state-dir",
                    help="dir for stage-complete markers. default: <crops>/.state")
    ap.add_argument("--log", default=None,
                    help="append-only log path (stdout is always used)")
    ap.add_argument("--python", default=sys.executable,
                    help="python interpreter to use for sub-stages")
    ap.add_argument("--skip-download", action="store_true",
                    help="trust an existing FF++ download; start at stage2")
    ap.add_argument("--smoke", action="store_true",
                    help="exercise every stage on synthetic inputs")
    args = ap.parse_args()

    if args.smoke:
        return _smoke(args.python)

    if not args.ffpp_root or not args.crops_root:
        ap.error("--ffpp-root and --crops-root are required unless --smoke")

    ffpp_root = Path(args.ffpp_root)
    crops_root = Path(args.crops_root)
    state_dir = Path(args.state_dir) if args.state_dir else crops_root / ".state"
    out_onnx = Path(args.out_onnx)
    logf = open(args.log, "a", encoding="utf-8") if args.log else None

    try:
        _log(logf, f"[pipeline] ffpp={ffpp_root} crops={crops_root} "
             f"onnx={out_onnx} epochs={args.epochs}")
        if not args.skip_download:
            if not args.download_script:
                _log(logf, "--download-script required (path to the script "
                     "you received in the access email)")
                return 1
            rc = stage1_download(ffpp_root, Path(args.download_script),
                                 args.server, args.compression,
                                 list(args.datasets), args.python,
                                 state_dir, logf)
            if rc != 0: return rc
        else:
            _log(logf, "[stage1] skipped (--skip-download)")
        rc = stage2_preprocess(ffpp_root, crops_root, args.compression,
                               args.frames_per_video, args.python,
                               state_dir, logf)
        if rc != 0: return rc
        rc = stage3_train(crops_root, out_onnx, args.epochs, args.batch,
                          args.python, state_dir, logf)
        if rc != 0: return rc
        _log(logf, "[pipeline] ALL STAGES DONE — restart main.py to pick up "
             "the new face_xray.onnx")
        return 0
    finally:
        if logf:
            # Close the log file. If close() fails we can't really
            # recover (log dir gone, disk full, etc.) but we also can't
            # let it abort cleanup of the rest of the function -- print
            # the error so it's visible in the console at least.
            try:
                logf.close()
            except Exception as e:
                print(f"[pipeline] WARNING: log close failed: {e!r}",
                      file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
