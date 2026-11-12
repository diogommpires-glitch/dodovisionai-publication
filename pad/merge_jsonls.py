"""Concatenate N .jsonl files into one, skipping any that don't exist.

Usage:
    python -m pad.merge_jsonls <in1.jsonl> [<in2.jsonl> ...] <out.jsonl>

Exits 0 on success, 1 if no input files exist.
"""
from __future__ import annotations
import sys
from pathlib import Path


def main(argv=None) -> int:
    """Concatenate one or more JSONL files into a single output. Last
    positional arg is the output path; everything before is input.
    Missing input files are skipped (warned), not fatal -- handy when
    chaining the script across pipelines that may not have produced
    every artefact. Returns 0 on success, 1 if no inputs exist, 2 on
    bad usage."""
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) < 2:
        print("usage: merge_jsonls <in1> [<in2> ...] <out>", file=sys.stderr)
        return 2
    inputs = [Path(p) for p in args[:-1]]
    out = Path(args[-1])
    existing = [p for p in inputs if p.is_file()]
    if not existing:
        print(f"[merge] no input files exist out of {len(inputs)}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    total_records = 0
    with out.open("w", encoding="utf-8") as fh:
        for p in existing:
            with open(p, "r", encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        fh.write(line if line.endswith("\n") else line + "\n")
                        total_records += 1
    print(f"[merge] {len(existing)} jsonls -> {out} "
          f"({total_records} records total)")
    for p in existing:
        print(f"  - {p.name}")
    missing = [p for p in inputs if not p.is_file()]
    for p in missing:
        print(f"  [skip missing] {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
