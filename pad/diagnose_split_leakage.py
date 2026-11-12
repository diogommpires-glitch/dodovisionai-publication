"""pad/diagnose_split_leakage.py -- quantify the v9.6.x patch_depth
identity-leakage hypothesis.

Reads the patch_depth_mmap paths.txt, applies the OLD index-based
shuffle-then-cut split (what train_patch_depth.py did before v9.7.0),
counts how many identities ended up in BOTH train and val, and
reports the leak rate.

Backs POST_QUEUE_PLAN.md A.1 with concrete numbers so the regression
narrative isn't speculative.

Usage
-----
    python -m pad.diagnose_split_leakage \\
        --paths E:/pad_datasets/patch_depth_mmap/paths.txt
    # or just:
    make diagnose-split

Output
------
A markdown report and stdout summary like:

    [diagnose] dataset:           494405 patches across 8423 identities
    [diagnose] OLD split:         44496 val / 449909 train
    [diagnose] OLD overlap:       8404 ids in BOTH (99.78% of total)
    [diagnose] NEW split:         49270 val / 445135 train
    [diagnose] NEW overlap:       0 ids in both (LEAK FIXED)
    [diagnose] verdict:           v9.6.x leaked 99.78% of identities into val.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


def old_index_split(n: int, val_fraction: float, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Reproduce the v9.6.x split: global index shuffle, cut at val_fraction.
    val = first `cut`, train = rest. Same as MmapSpoofDataset pre-v9.7.0.
    """
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = max(1, int(n * val_fraction))
    val = idx[:cut]
    train = idx[cut:]
    return train, val


def new_identity_split(idents: List[str], val_fraction: float, seed: int = 42) -> Tuple[List[int], List[int]]:
    """The v9.7.0 fix: split GROUPS, not patches."""
    from pad.train_patch_depth import _identity_stratified_split
    return _identity_stratified_split(idents, val_fraction=val_fraction, seed=seed)


def count_overlap(idents: List[str], train: List[int], val: List[int]) -> int:
    """Count overlapping path/identity samples between two record JSONLs. Used to flag train/test leakage that would inflate paper numbers."""
    train_ids = set(idents[i] for i in train)
    val_ids = set(idents[i] for i in val)
    return len(train_ids & val_ids)


def main(argv=None) -> int:
    """CLI: report train/val identity-overlap on a record pair. Run before publishing a cross-dataset matrix."""
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", type=Path,
                    default=Path("E:/pad_datasets/patch_depth_mmap/paths.txt"),
                    help="paths.txt produced by pad.preprocess_patch_depth")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="markdown output (default: stdout only)")
    args = ap.parse_args(argv)

    if not args.paths.exists():
        print(f"[diagnose] paths.txt not found at {args.paths}", file=sys.stderr)
        return 1

    from pad.train_patch_depth import _identity_from_filename

    paths = [ln.strip() for ln in args.paths.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    n = len(paths)
    idents = [_identity_from_filename(p) for p in paths]
    n_unique = len(set(idents))

    # OLD (leaky) split
    old_train, old_val = old_index_split(n, args.val_fraction, args.seed)
    old_overlap = count_overlap(idents, old_train, old_val)
    old_overlap_pct = (100.0 * old_overlap / n_unique) if n_unique else 0.0

    # NEW (fixed) split
    new_train, new_val = new_identity_split(idents, args.val_fraction, args.seed)
    new_overlap = count_overlap(idents, new_train, new_val)

    summary = []
    summary.append(f"# patch_depth split-leakage diagnostic")
    summary.append("")
    summary.append(f"_Dataset:_ {args.paths}  ")
    summary.append(f"_Total patches:_ **{n:,}**  ")
    summary.append(f"_Unique identities:_ **{n_unique:,}**  ")
    summary.append(f"_val_fraction:_ {args.val_fraction}  ")
    summary.append(f"_seed:_ {args.seed}")
    summary.append("")
    summary.append("## v9.6.x index-based split (the bug)")
    summary.append("")
    summary.append(f"- train: **{len(old_train):,}** patches  ")
    summary.append(f"- val:   **{len(old_val):,}** patches  ")
    summary.append(f"- identities in BOTH train AND val: **{old_overlap:,}** ")
    summary.append(f"  ({old_overlap_pct:.2f}% of all unique identities)")
    summary.append("")
    summary.append("## v9.7.0 identity-stratified split (the fix)")
    summary.append("")
    summary.append(f"- train: **{len(new_train):,}** patches  ")
    summary.append(f"- val:   **{len(new_val):,}** patches  ")
    summary.append(f"- identities in BOTH train AND val: **{new_overlap}** "
                   f"(MUST be 0)")
    summary.append("")
    summary.append("## Verdict")
    summary.append("")
    if old_overlap_pct > 50:
        summary.append(f"**Severe leakage confirmed**: v9.6.x put {old_overlap_pct:.1f}% of "
                       f"identities into both train AND val. The val_acc 0.998 reported "
                       f"during stage 13 is a per-patch identity-recognition score, not "
                       f"a liveness score. v9.7.0's identity-stratified split eliminates "
                       f"this. Expected real val_acc post-fix: 0.80-0.90.")
    else:
        summary.append(f"Leakage at {old_overlap_pct:.1f}% — investigate further.")
    summary.append("")
    summary.append(f"_Generated by pad.diagnose_split_leakage._")

    md = "\n".join(summary)

    print(f"[diagnose] dataset:        {n:,} patches across {n_unique:,} identities")
    print(f"[diagnose] OLD split:      {len(old_val):,} val / {len(old_train):,} train")
    print(f"[diagnose] OLD overlap:    {old_overlap:,} ids in BOTH ({old_overlap_pct:.2f}% of total)")
    print(f"[diagnose] NEW split:      {len(new_val):,} val / {len(new_train):,} train")
    print(f"[diagnose] NEW overlap:    {new_overlap} ids in both " +
          ("(LEAK FIXED)" if new_overlap == 0 else "(SOMETHING IS WRONG)"))
    if old_overlap_pct > 50:
        print(f"[diagnose] VERDICT: v9.6.x leaked {old_overlap_pct:.1f}% of identities into val.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"[diagnose] wrote {args.out}")
    return 0 if new_overlap == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
