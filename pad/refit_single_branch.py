"""refit_single_branch.py -- refit one PAD branch calibrator in isolation.

Use case
--------
You just re-scored a dataset through a preprocessor upgrade (YuNet face
detector, new thresholds, etc.) and produced a fresh jsonl. You want to
refit only the affected branch's calibrator without disturbing the
other branches' calibrators in `pad/models/calibrators.pkl`.

This is what this module does:

    1. Load the existing master `calibrators.pkl`.
    2. Fit a fresh CalibratorSet from the provided jsonls.
    3. Compare ECE(after) of the new branch calibrator vs. ECE(after)
       of the existing one. If the new value is better by at least
       `--min-ece-delta` (absolute), swap it in and save.
    4. Otherwise print the comparison and leave the pkl untouched.
    5. `--dry-run` always skips the write.

Safety
------
A timestamped backup of the pkl is made before each write:
    pad/models/calibrators.pkl.bak_pre_<branch>_<stamp>
Restore with a simple `copy /Y <backup> calibrators.pkl`.

Usage
-----
    python -m pad.refit_single_branch --branch moire \\
        --records reports/meta_train_records_replay_v3.jsonl

    python -m pad.refit_single_branch --branch audio \\
        --records reports/audio_asvspoof_records_v3.jsonl \\
        --min-ece-delta 0.001

    python -m pad.refit_single_branch --branch patch_depth \\
        --records reports/meta_train_records_celeba_v3.jsonl \\
                  reports/meta_train_records_replay_v3.jsonl \\
        --dry-run
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List


def _load_jsonl(p: Path) -> List[Dict]:
    out: List[Dict] = []
    if not p.is_file():
        print(f"[refit] warning: input not found: {p}", file=sys.stderr)
        return out
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch", required=True,
                    help="branch name as it appears in records['branches'][*]['name'] "
                         "(e.g. moire, patch_depth, deepfake, audio)")
    ap.add_argument("--records", nargs="+", required=True, type=Path,
                    help="one or more pad.report-schema jsonls")
    ap.add_argument("--min-ece-delta", type=float, default=0.001,
                    help="minimum absolute ECE reduction required to deploy "
                         "the new calibrator (default 0.001). Guards against "
                         "noise-driven swaps.")
    ap.add_argument("--fit-source-tag", default=None,
                    help="string stored in metadata[branch]['fit_source'] "
                         "for audit (e.g. 'replay_v3_yunet_rerun').")
    ap.add_argument("--dry-run", action="store_true",
                    help="compare ECEs and print what would happen; do not "
                         "write the pkl.")
    args = ap.parse_args(argv)

    # Late import so `--help` works even if pad deps are misconfigured
    from pad.calibrate import (
        CALIBRATOR_PATH, load_calibrators, save_calibrators, fit_from_records,
        _reset_calibrators_cache, jsonl_sha256,
    )

    # Collect records
    records: List[Dict] = []
    for p in args.records:
        recs = _load_jsonl(p)
        print(f"[refit] {p.name}: {len(recs)} records")
        records.extend(recs)
    if not records:
        print("[refit] no input records -- nothing to fit", file=sys.stderr)
        return 1

    # Current state
    _reset_calibrators_cache()
    cs_cur = load_calibrators()
    if cs_cur is None:
        print(f"[refit] no master calibrators.pkl at {CALIBRATOR_PATH}; "
              f"run `python -m pad.calibrate_per_branch` first",
              file=sys.stderr)
        return 2
    old_meta = cs_cur.metadata.get(args.branch, {})
    old_ece_after = float(old_meta.get("ece_after", float("inf")))
    old_choice = old_meta.get("choice", "?")
    old_range = old_meta.get("fit_range", 0.0)

    # Candidate fit
    cs_new = fit_from_records(records)
    if args.branch not in cs_new.per_branch:
        print(f"[refit] branch '{args.branch}' does not appear in the "
              f"supplied records (present: {sorted(cs_new.per_branch)})",
              file=sys.stderr)
        return 3
    new_meta = cs_new.metadata.get(args.branch, {})
    new_ece_after = float(new_meta.get("ece_after", float("inf")))
    new_choice = new_meta.get("choice", "?")
    new_range = new_meta.get("fit_range", 0.0)
    new_ece_before = float(new_meta.get("ece_before", float("nan")))

    improvement = old_ece_after - new_ece_after

    print()
    print(f"[refit] branch = {args.branch}")
    print(f"  current : choice={old_choice:<22s} "
          f"ECE(after)={old_ece_after:.4f}  range={old_range}")
    print(f"  candidate: choice={new_choice:<22s} "
          f"ECE {new_ece_before:.4f} -> {new_ece_after:.4f}  "
          f"range={new_range:.3f}  "
          f"(n={len(records)})")
    print(f"  delta   : ECE improvement = {improvement:+.4f}  "
          f"(threshold {args.min_ece_delta:.4f})")

    # API_up v9.8 safeguard: detect ECE-degenerate calibrators that map
    # everything to the marginal class probability. When the model's raw
    # output range is narrow (Platt or isotonic both prone to this), the
    # fit collapses to a near-constant function. ECE looks great on the
    # training distribution because constant=base-rate gives perfect bin
    # alignment, but the calibrator no longer discriminates -- live and
    # spoof get the same probability. We probe the response curve at
    # five anchor points and refuse to deploy when the spread is too small.
    new_cal = cs_new.per_branch[args.branch]
    probe = [float(new_cal.transform(x)) for x in (0.0, 0.25, 0.5, 0.75, 1.0)]
    spread = max(probe) - min(probe)
    monotonic = all(probe[i + 1] >= probe[i] - 1e-3 for i in range(4))
    if spread < 0.05:
        print(f"[refit] REJECT: candidate calibrator is ECE-degenerate "
              f"(response spread = {spread:.4f} across [0,1]; curve = "
              f"{['%.3f'%p for p in probe]}). The Platt/isotonic fit "
              f"collapsed to a near-constant function on a narrow input "
              f"range. Keeping existing calibrator.", file=sys.stderr)
        return 0
    if not monotonic:
        print(f"[refit] REJECT: candidate calibrator response is "
              f"non-monotonic (curve = {['%.3f'%p for p in probe]}). "
              f"Keeping existing calibrator.", file=sys.stderr)
        return 0

    if improvement < args.min_ece_delta:
        print(f"[refit] candidate does not clear the threshold -- "
              f"keeping existing calibrator.")
        return 0

    if args.dry_run:
        print(f"[refit] DRY-RUN: would swap in the new calibrator and save.")
        return 0

    # Timestamped backup
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bak = Path(str(CALIBRATOR_PATH) + f".bak_pre_{args.branch}_{stamp}")
    shutil.copy(CALIBRATOR_PATH, bak)
    print(f"[refit] backup -> {bak.name}")

    # Splice in the new branch without disturbing the rest
    cs_cur.per_branch[args.branch] = cs_new.per_branch[args.branch]
    cs_cur.metadata[args.branch] = dict(new_meta)
    if args.fit_source_tag:
        cs_cur.metadata[args.branch]["fit_source"] = args.fit_source_tag
    cs_cur.metadata[args.branch]["n_fit"] = len(records)
    cs_cur.metadata[args.branch]["training_jsonls_sha256"] = jsonl_sha256(args.records)
    cs_cur.metadata[args.branch]["training_jsonls_names"] = sorted(
        Path(p).name for p in args.records
    )

    save_calibrators(cs_cur)
    # Touch mtime so the hot-reload watcher picks it up immediately
    Path(CALIBRATOR_PATH).touch()

    # Verify via fresh load
    _reset_calibrators_cache()
    cs_verify = load_calibrators()
    meta = cs_verify.metadata[args.branch]
    print(f"[refit] deployed: choice={meta['choice']} "
          f"ECE {meta['ece_before']:.4f} -> {meta['ece_after']:.4f}  "
          f"range={meta['fit_range']:.3f}  "
          f"source={meta.get('fit_source','<unspecified>')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
