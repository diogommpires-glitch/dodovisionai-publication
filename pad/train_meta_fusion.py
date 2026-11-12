"""pad/train_meta_fusion.py — one-shot meta-fusion + calibration trainer.

Walks a labelled image directory, runs every IMAGE-based PAD branch on each
sample, writes a JSONL in the exact format `pad.report` / `pad.meta_fuse` /
`pad.calibrate` consume, then invokes both trainers to produce:

    pad/models/meta_fuse.pkl     # LightGBM fusion model
    pad/models/calibrators.pkl   # per-branch Platt/isotonic calibrators

Input directories (pick one via --data):

    <data>/live/*.jpg    # real / bona-fide face crops
    <data>/spoof/*.jpg   # attack / manipulated face crops

This matches the layout of `prepare_celeba_spoof.py` and `preprocess_ffpp.py`
so you can reuse `E:\\pad_datasets\\patch_depth` or `E:\\pad_datasets\\face_xray`
directly.

Only three branches emit useful scores from a single still image:
  - moire (static texture analysis)
  - patch_depth (the trained MobileNetV3 classifier)
  - deepfake (the trained face_xray classifier)

The rPPG, dynamics, and audio branches are skipped (they need time series
or audio). Meta-fuser learns to ignore them via reliability=0 imputation —
at runtime on the live webcam feed they will contribute.

Usage — smoke run (30 samples, sanity check):
    python -m pad.train_meta_fusion --data E:\\pad_datasets\\patch_depth --limit 30

Full run (a few thousand samples, 10-20 min):
    python -m pad.train_meta_fusion --data E:\\pad_datasets\\patch_depth --limit 4000

Train-only from an existing records JSONL (skip scoring):
    python -m pad.train_meta_fusion --from-jsonl reports/meta_train.jsonl
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "reports" / "meta_train_records.jsonl"


def _iter_samples(data: Path, limit: Optional[int]) -> List[Tuple[str, Path]]:
    live = sorted((data / "live").iterdir()) if (data / "live").is_dir() else []
    spoof = sorted((data / "spoof").iterdir()) if (data / "spoof").is_dir() else []
    live = [p for p in live if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    spoof = [p for p in spoof if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    items = [("live", p) for p in live] + [("spoof", p) for p in spoof]
    random.Random(42).shuffle(items)
    if limit is not None and limit > 0:
        # Stratified: keep roughly 50/50 live/spoof
        live_q = [x for x in items if x[0] == "live"][: limit // 2]
        spoof_q = [x for x in items if x[0] == "spoof"][: limit - len(live_q)]
        items = live_q + spoof_q
        random.Random(42).shuffle(items)
    return items


def _score_one(path: Path, label: str) -> Optional[Dict]:
    """Run the image-capable PAD branches on a single sample. Returns the
    report-style record or None on decode failure."""
    try:
        import cv2
    except ImportError:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None

    branches_out: List[Dict] = []

    # moire — works on any face-sized BGR crop
    try:
        from pad.branches.moire_branch import score as score_moire
        s = score_moire(img)
        branches_out.append({
            "name": s.name, "p_attack": float(s.p_attack),
            "reliability": float(s.reliability),
            "dominant_class": s.dominant_class,
        })
    except Exception as e:
        branches_out.append({"name": "moire", "p_attack": 0.5,
                             "reliability": 0.0, "dominant_class": None})

    # patch_depth — needs the trained ONNX loaded (should be present)
    try:
        from pad.branches.patch_depth_branch import score as score_pd
        s = score_pd(img)
        branches_out.append({
            "name": s.name, "p_attack": float(s.p_attack),
            "reliability": float(s.reliability),
            "dominant_class": s.dominant_class,
        })
    except Exception:
        branches_out.append({"name": "patch_depth", "p_attack": 0.5,
                             "reliability": 0.0, "dominant_class": None})

    # deepfake — face_xray-style classifier
    try:
        from pad.branches.deepfake_branch import score as score_df
        s = score_df(pid=0, face_crop_bgr=img)
        branches_out.append({
            "name": s.name, "p_attack": float(s.p_attack),
            "reliability": float(s.reliability),
            "dominant_class": s.dominant_class,
        })
    except Exception:
        branches_out.append({"name": "deepfake", "p_attack": 0.5,
                             "reliability": 0.0, "dominant_class": None})

    # rppg / dynamics / audio left implicit — meta-fuser imputes reliability=0

    return {
        "path": str(path),
        "label": label,
        "attack_class": None if label == "live" else "UNKNOWN",
        "branches": branches_out,
    }


def _score_all(data: Path, out_path: Path, limit: Optional[int]) -> int:
    items = _iter_samples(data, limit)
    if not items:
        print(f"[meta] no samples under {data}/live or {data}/spoof")
        return 0
    print(f"[meta] scoring {len(items):,} samples...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    t0 = time.time()
    try:
        from tqdm import tqdm
        bar = tqdm(items, unit="img", mininterval=2.0)
    except ImportError:
        bar = items
    with open(out_path, "w", encoding="utf-8") as fh:
        for label, path in bar:
            rec = _score_one(path, label)
            if rec is None:
                continue
            fh.write(json.dumps(rec) + "\n")
            n_ok += 1
    dt = time.time() - t0
    print(f"[meta] wrote {n_ok:,} records in {dt/60:.1f} min -> {out_path}")
    return n_ok


def _run_meta_fuser_training(jsonl: Path) -> bool:
    print(f"\n[meta] training meta-fuser from {jsonl}")
    try:
        from pad.meta_fuse import main as mf_main
        rc = mf_main(["--from", str(jsonl)])
        return rc == 0
    except SystemExit as e:
        return (e.code or 0) == 0
    except Exception as e:
        print(f"[meta] meta_fuse training failed: {e}")
        return False


def _run_calibrator_training(jsonl: Path) -> bool:
    print(f"\n[meta] fitting per-branch calibrators from {jsonl}")
    try:
        from pad.calibrate import main as cal_main
        rc = cal_main(["--from", str(jsonl)])
        return rc == 0
    except SystemExit as e:
        return (e.code or 0) == 0
    except Exception as e:
        print(f"[meta] calibrator training failed: {e}")
        return False


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=None,
                    help="dir with live/ + spoof/ subfolders "
                         "(skip to re-train from an existing jsonl)")
    ap.add_argument("--from-jsonl", dest="from_jsonl", type=Path, default=None,
                    help="re-train from existing report-format records")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="jsonl path to write (default reports/meta_train_records.jsonl)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total samples (stratified 50/50 live/spoof)")
    ap.add_argument("--skip-meta", action="store_true",
                    help="score only; don't train the meta-fuser")
    ap.add_argument("--skip-calib", action="store_true",
                    help="score only; don't fit calibrators")
    args = ap.parse_args(argv)

    if args.from_jsonl:
        jsonl = args.from_jsonl
        if not jsonl.exists():
            print(f"[meta] {jsonl} not found")
            return 1
    else:
        if args.data is None:
            print("[meta] supply either --data or --from-jsonl")
            return 2
        if not (args.data / "live").is_dir() and not (args.data / "spoof").is_dir():
            print(f"[meta] {args.data} has no live/ or spoof/ subfolders")
            return 2
        n = _score_all(args.data, args.out, args.limit)
        if n == 0:
            return 1
        jsonl = args.out

    ok_meta = True
    ok_cal = True
    if not args.skip_meta:
        ok_meta = _run_meta_fuser_training(jsonl)
    if not args.skip_calib:
        ok_cal = _run_calibrator_training(jsonl)

    print("\n[meta] ====== SUMMARY ======")
    print(f"  records           : {jsonl}")
    print(f"  meta-fuser trained: {'yes' if ok_meta else 'NO'}")
    print(f"  calibrators fitted: {'yes' if ok_cal else 'NO'}")
    if ok_meta and ok_cal:
        print("\n[meta] To activate at runtime:")
        print("    set FF_PAD_META_FUSE=1")
        print("    set FF_PAD_CALIBRATE=1")
        print("  then restart main.py.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
