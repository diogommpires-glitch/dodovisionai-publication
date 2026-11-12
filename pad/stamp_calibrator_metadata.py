"""pad/stamp_calibrator_metadata.py -- force-stamp fit_source +
training_jsonls_sha256 onto calibrators.pkl WITHOUT re-fitting.

Used when pad.refit_single_branch's ECE-improvement gate rejects a
re-fit (because the existing calibrator is already as good as the
candidate) but the metadata is still missing. Per POST_QUEUE_PLAN A.3.
"""
from __future__ import annotations

import hashlib
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKL = REPO / "pad" / "models" / "calibrators.pkl"
REPORTS = REPO / "reports"

TAGS = {
    "moire":             ("replay_v5_yunet",     REPORTS / "meta_train_records_replay_v5.jsonl"),
    "patch_depth":       ("celeba_v5",           REPORTS / "meta_train_records_celeba_v5.jsonl"),
    "deepfake":          ("celebdf_v5",          REPORTS / "meta_train_records_celebdf_v5.jsonl"),
    "deepfake_temporal": ("celebdf_v2_temporal", REPORTS / "deepfake_temporal_v2_records_celebdf.jsonl"),
    "audio":             ("asvspoof_v3",         REPORTS / "audio_asvspoof_records_v3.jsonl"),
}


def _file_sha(p: Path) -> str:
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> int:
    # The codebase serialises calibrators via joblib (not raw pickle) -
    # see pad.calibrate.load_calibrators. Use joblib here too so we can
    # round-trip the CalibratorSet without the gotcha #3 module-path
    # confusion that bare pickle.load runs into.
    sys.path.insert(0, str(REPO))
    import pad.calibrate as _calmod  # noqa: F401  (registers classes)
    import joblib

    if not PKL.exists():
        print(f"[stamp] not found: {PKL}", file=sys.stderr)
        return 1

    backup = PKL.with_suffix(f".pkl.bak_pre_stamp_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(PKL, backup)
    print(f"[stamp] backup -> {backup.name}")

    bundle = joblib.load(PKL)
    print(f"[stamp] loaded calibrators via joblib, type = {type(bundle).__name__}")

    # Locate the metadata dict (handles both attribute and dict-key layouts)
    md = None
    if hasattr(bundle, "metadata") and isinstance(bundle.metadata, dict):
        md = bundle.metadata
        layout = "attr"
    elif isinstance(bundle, dict):
        md = bundle.setdefault("metadata", {})
        layout = "dict"
    else:
        # last resort: inject as attribute
        bundle.metadata = {}
        md = bundle.metadata
        layout = "injected"
    print(f"[stamp] metadata layout: {layout}")
    print(f"[stamp] metadata keys before: {sorted(md.keys())}")

    for branch, (tag, jsonl) in TAGS.items():
        sha = _file_sha(jsonl)
        md.setdefault(branch, {})
        md[branch]["fit_source"] = tag
        md[branch]["training_jsonls_sha256"] = sha
        print(f"[stamp] {branch:18s} tag={tag:22s} sha={sha}")

    joblib.dump(bundle, PKL)
    print(f"[stamp] wrote {PKL} ({PKL.stat().st_size:,} bytes) via joblib")
    print(f"[stamp] regen calib_sprint_summary next:")
    print(f"        python -m pad.write_calib_summary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
