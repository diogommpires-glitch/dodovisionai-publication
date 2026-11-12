"""pad/merge_branch_scores.py -- join per-branch scored.jsonls into one
multi-branch meta_train_records jsonl.

Problem
-------
`pad.eval_cross_dataset --branch X --test-data <dir> --name Y` writes
`reports/cross_dataset/Y/scored.jsonl` with ONE branch per record
(the named branch). But `pad.bootstrap_ci`, `pad.run_ablation_matrix`,
`pad.cross_dataset_matrix`, and `pad.meta_fuse` all expect the
multi-branch schema:

    {"path": ..., "label": ..., "attack_class": ...,
     "branches": [{"name": "moire", "p_attack": ..., "reliability": ...},
                  {"name": "patch_depth", ...},
                  {"name": "deepfake", ...},
                  ...]}

This module takes N single-branch jsonls, joins them by their `path`
field, and emits a unified multi-branch jsonl. The "top-level p_attack"
in each input is ignored (it's just the single-branch score); callers
of the multi-branch file should compute a fused p_attack themselves
(bootstrap_ci handles this via its local `_fused_p` helper).

Usage
-----
    python -m pad.merge_branch_scores \\
        --inputs reports/cross_dataset/moire_on_ReplayAttack/scored.jsonl \\
                 reports/cross_dataset/patch_depth_on_ReplayAttack/scored.jsonl \\
                 reports/cross_dataset/face_xray_on_ReplayAttack/scored.jsonl \\
        --out reports/meta_train_records_replay_v4.jsonl
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def _load_jsonl(p: Path) -> List[Dict]:
    out = []
    if not p.is_file():
        print(f"[merge-branches] warning: {p} missing -- skipping",
              file=sys.stderr)
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


def main(argv=None) -> int:
    """CLI: merge multiple per-branch score JSONLs (each one branch's
    p_attack on the same set of paths) into one combined records jsonl
    in the standard pad.report schema (`branches: [{name, p_attack,
    reliability}]`). Records align by `--key` (default `path`).

    Used by the cross-dataset pipeline: `eval_cross_dataset` produces
    one jsonl per branch; this script combines them into something
    `pad.calibrate` and `pad.refit_single_branch` can consume.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, type=Path,
                    help="per-branch scored.jsonl paths (one per branch, "
                         "typically the output of "
                         "pad.eval_cross_dataset --branch X --name Y)")
    ap.add_argument("--out", type=Path, required=True,
                    help="destination merged jsonl")
    ap.add_argument("--key", default="path",
                    help="field used to join records across branches "
                         "(default 'path'; same-path records from "
                         "different branches get unioned)")
    ap.add_argument("--require-all", action="store_true",
                    help="skip records that don't appear in every input "
                         "jsonl (strict multi-branch union). Default: "
                         "emit every record seen in any input.")
    args = ap.parse_args(argv)

    # Read each input and bucket records by key
    per_input_keys: List[set] = []
    merged: Dict[str, Dict] = {}
    for path in args.inputs:
        recs = _load_jsonl(path)
        keys = set()
        for r in recs:
            k = r.get(args.key)
            if k is None:
                continue
            keys.add(k)
            if k not in merged:
                merged[k] = {
                    "path": r.get("path"),
                    "label": r.get("label"),
                    "attack_class": r.get("attack_class"),
                    "branches": [],
                }
            # A scored.jsonl from eval_cross_dataset carries exactly one
            # branch in r["branches"]; just extend our branches list.
            br = r.get("branches") or []
            merged[k]["branches"].extend(br)
        per_input_keys.append(keys)
        print(f"[merge-branches] {path.name}: {len(recs)} records, "
              f"{len(keys)} unique keys")

    if args.require_all and per_input_keys:
        common = per_input_keys[0]
        for s in per_input_keys[1:]:
            common = common & s
        before = len(merged)
        merged = {k: v for k, v in merged.items() if k in common}
        print(f"[merge-branches] --require-all filtered {before}->{len(merged)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in merged.values():
            fh.write(json.dumps(r) + "\n")

    branch_names = set()
    total_branch_entries = 0
    for r in merged.values():
        total_branch_entries += len(r["branches"])
        for b in r["branches"]:
            if isinstance(b, dict) and b.get("name"):
                branch_names.add(b["name"])
    print(f"[merge-branches] wrote {len(merged)} records "
          f"({total_branch_entries} branch entries, "
          f"{len(branch_names)} unique branches: {sorted(branch_names)}) "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
