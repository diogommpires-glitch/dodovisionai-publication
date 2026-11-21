"""calibrate_per_branch.py -- fit each PAD branch on attack-relevant data only.

Problem
-------
`pad.calibrate` fits all branches on a merged jsonl. But each branch only
discriminates certain attack classes (moire = screens, deepfake = face-swaps,
etc.). Records outside a branch's specialty look random to it, diluting the
fit and triggering the `identity_nosignal` guardrail.

Solution
--------
For each branch, pick ONLY records from datasets where that branch can
discriminate, and fit its calibrator on that subset. Combine the resulting
single-branch calibrators into one master CalibratorSet.

Branch -> relevant datasets:
    moire       -> celeba (screens), replay (screens)
    patch_depth -> celeba (print/mask/screen), replay
    deepfake    -> celebdf (face-swaps)
    audio       -> asvspoof (synthetic speech)

Usage
-----
    python -m pad.calibrate_per_branch \\
        --celeba   reports/meta_train_records_celeba.jsonl \\
        --replay   reports/meta_train_records_replay.jsonl \\
        --celebdf  reports/meta_train_records_celebdf.jsonl \\
        --asvspoof reports/audio_asvspoof_records.jsonl
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from pad.calibrate import (
    fit_from_records, save_calibrators, CalibratorSet,
)


BRANCH_DATASETS = {
    "moire":       ["celeba", "replay"],
    "patch_depth": ["celeba", "replay"],
    "deepfake":    ["celebdf"],
    "audio":       ["asvspoof"],
}


def _load_jsonl(p: Path) -> List[Dict]:
    if p is None or not p.is_file():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main(argv=None) -> int:
    """CLI: fit per-branch calibrators using per-branch training source
    pairings (BRANCH_DATASETS). Each branch can use a different mix of
    JSONLs -- e.g. moire trains on replay+celeba while audio trains on
    asvspoof. The result is then merged into one master CalibratorSet
    with the highest-sample fused calibrator wins.

    Distinct from `pad.calibrate.main`: that one uses ONE jsonl for ALL
    branches; this one allows per-branch source curation. Returns 0 on
    success, 2 if no inputs were provided.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--celeba",   type=Path, default=None)
    ap.add_argument("--replay",   type=Path, default=None)
    ap.add_argument("--celebdf",  type=Path, default=None)
    ap.add_argument("--asvspoof", type=Path, default=None)
    args = ap.parse_args(argv)

    sources = {
        "celeba":   _load_jsonl(args.celeba),
        "replay":   _load_jsonl(args.replay),
        "celebdf":  _load_jsonl(args.celebdf),
        "asvspoof": _load_jsonl(args.asvspoof),
    }
    for k, v in sources.items():
        print(f"[per-branch] source {k}: {len(v)} records")
    print()

    master_per_branch: Dict[str, object] = {}
    master_metadata: Dict[str, Dict] = {}

    for branch, datasets in BRANCH_DATASETS.items():
        combined: List[Dict] = []
        for ds in datasets:
            combined.extend(sources.get(ds, []))
        if not combined:
            print(f"[per-branch] {branch}: no data ({'+'.join(datasets)}); skipping")
            continue
        # Filter to only records that actually contain this branch's score
        relevant = [r for r in combined
                    if any(b.get("name") == branch for b in (r.get("branches") or []))]
        if not relevant:
            print(f"[per-branch] {branch}: no records expose this branch; skipping")
            continue
        print(f"[per-branch] {branch}: fitting on {len(relevant)} records from "
              f"{'+'.join(datasets)}")
        try:
            cs = fit_from_records(relevant)
        except Exception as e:
            print(f"[per-branch] {branch}: fit failed: {e}", file=sys.stderr)
            continue
        cal = cs.per_branch.get(branch)
        meta = cs.metadata.get(branch, {})
        if cal is None:
            print(f"[per-branch] {branch}: no calibrator produced")
            continue
        master_per_branch[branch] = cal
        master_metadata[branch] = meta
        print(f"    choice={meta.get('choice','?'):<22} "
              f"ECE {meta.get('ece_before', float('nan')):.4f} -> "
              f"{meta.get('ece_after', float('nan')):.4f}  "
              f"n={int(meta.get('n', 0))}  range={meta.get('fit_range','-')}")

    if not master_per_branch:
        print("[per-branch] ERROR: no calibrators fit -- nothing to save", file=sys.stderr)
        return 1

    master_cs = CalibratorSet(
        per_branch=master_per_branch,
        fused=None,
        metadata=master_metadata,
        version=1,
    )
    save_calibrators(master_cs)
    print()
    print(f"[per-branch] saved {len(master_per_branch)} per-branch calibrators")
    print(f"  active branches: {sorted(master_per_branch.keys())}")
    print(f"  identity fallbacks: "
          f"{[k for k, m in master_metadata.items() if m.get('choice')=='identity_nosignal']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
