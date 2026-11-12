"""pad/restamp_calibrators_v5.py -- backfill fit_source / training SHA
metadata onto the v5 calibrators.pkl.

Phase A.3 of POST_QUEUE_PLAN.md: stage 06 of queue_mega.cmd produces
calibrators.pkl correctly fit to v5 features but the per-branch
fit_source and training_jsonls_sha256 attributes don't survive the
pickle round-trip (pad/calibrate.py instance-attribute regression).
This wrapper re-runs pad.refit_single_branch for each branch with the
correct --fit-source-tag, then regenerates the summary.

Run AFTER queue_mega.cmd finishes and pad.verify_v5_landings reports
the calibrator fit_source warning.

Usage
-----
    python -m pad.restamp_calibrators_v5
    python -m pad.restamp_calibrators_v5 --dry-run
    python -m pad.restamp_calibrators_v5 --branches moire patch_depth

Branches handled (with their canonical training jsonl):
    moire             -> meta_train_records_replay_v5_full.jsonl
    patch_depth       -> meta_train_records_celeba_v5.jsonl
    deepfake          -> meta_train_records_celebdf_v5_spatial.jsonl
    deepfake_temporal -> deepfake_temporal_v2_records_celebdf.jsonl
    audio             -> audio_asvspoof_records_v3.jsonl

If a v5 jsonl is missing, falls back to v4 with a warning.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS = REPO_ROOT / "reports"

# (branch_name, fit_source_tag, primary_v5_jsonl, fallback_v4_jsonl)
# Updated 2026-04-25: the v5 rescoring stage 18 produces
# meta_train_records_<dataset>_v5.jsonl (no `_full` / `_spatial` suffix
# anymore — those were the v4-era naming scheme). Fallbacks still
# point at the v4-suffixed names for environments where stage 18
# never ran.
BRANCH_RECIPES = [
    ("moire",
        "replay_v5_yunet",
        "meta_train_records_replay_v5.jsonl",
        "meta_train_records_replay_v4_full.jsonl"),
    ("patch_depth",
        "celeba_v5",
        "meta_train_records_celeba_v5.jsonl",
        "meta_train_records_celeba_v4.jsonl"),
    ("deepfake",
        "celebdf_v5",
        "meta_train_records_celebdf_v5.jsonl",
        "meta_train_records_celebdf_v4_spatial.jsonl"),
    ("deepfake_temporal",
        "celebdf_v2_temporal",
        "deepfake_temporal_v2_records_celebdf.jsonl",
        "deepfake_temporal_records_celebdf.jsonl"),
    ("audio",
        "asvspoof_v3",
        "audio_asvspoof_records_v3.jsonl",
        "audio_asvspoof_records.jsonl"),
]


def _resolve_jsonl(primary: str, fallback: str) -> Path:
    pp = REPORTS / primary
    if pp.exists():
        return pp
    pf = REPORTS / fallback
    if pf.exists():
        print(f"[restamp] WARN: {primary} not found, using fallback {fallback}")
        return pf
    raise FileNotFoundError(
        f"neither {primary} nor {fallback} exists under {REPORTS}")


def restamp_one(branch: str, fit_source_tag: str, jsonl: Path,
                dry_run: bool = False) -> int:
    cmd = [
        sys.executable, "-m", "pad.refit_single_branch",
        "--branch", branch,
        "--records", str(jsonl),
        "--fit-source-tag", fit_source_tag,
    ]
    print(f"[restamp] {branch}  tag={fit_source_tag}  source={jsonl.name}")
    if dry_run:
        print(f"[restamp] DRY-RUN: {' '.join(cmd)}")
        return 0
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if proc.returncode != 0:
        print(f"[restamp] FAIL: {branch} returned {proc.returncode}")
    return proc.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branches", nargs="*", default=None,
                    help="restamp only these branches (default: all 5)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands but don't execute")
    ap.add_argument("--skip-summary", action="store_true",
                    help="don't regenerate calib_sprint_summary at the end")
    args = ap.parse_args(argv)

    selected = args.branches if args.branches else [r[0] for r in BRANCH_RECIPES]

    rc_total = 0
    for branch, tag, primary, fallback in BRANCH_RECIPES:
        if branch not in selected:
            continue
        try:
            jsonl = _resolve_jsonl(primary, fallback)
        except FileNotFoundError as e:
            print(f"[restamp] SKIP {branch}: {e}")
            rc_total += 1
            continue
        rc = restamp_one(branch, tag, jsonl, dry_run=args.dry_run)
        rc_total += abs(rc)

    if not args.skip_summary and not args.dry_run:
        print("[restamp] regenerating calib_sprint_summary.md ...")
        proc = subprocess.run(
            [sys.executable, "-m", "pad.write_calib_summary"],
            cwd=REPO_ROOT, check=False)
        if proc.returncode != 0:
            print(f"[restamp] write_calib_summary returned {proc.returncode}")
            rc_total += 1

    if rc_total == 0:
        print("[restamp] all branches re-stamped successfully.")
        print("[restamp] verify with: python -m pad.verify_v5_landings")
    else:
        print(f"[restamp] DONE with {rc_total} non-zero return code(s).")
    return 0 if rc_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
