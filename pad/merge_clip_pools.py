"""pad/merge_clip_pools.py -- merge video-clip .npy pools into a combined dir.

Replaces the inline Python one-liner in queue_master.cmd stage 3 that was
hitting cmd.exe's nested-quote escaping bug. Extracted to a file so the
shell doesn't have to deal with \" inside f-strings.

Usage:
    python -m pad.merge_clip_pools <src_dir1> [<src_dir2> ...] <dst_dir> \\
        [--min-pct-live 0.9] [--min-pct-spoof 0.9]

Each source dir must have live/ and spoof/ subdirs with *.npy files.
dst_dir/live/ and dst_dir/spoof/ are created and populated via file copy
(skips if a same-named file already exists at the destination).

Sanity-check flags (v9.6.0):
    --min-pct-live   fraction of the live  source-union that must land in
                     dst/live/  before exit code 0  (default 0.0 = no-op)
    --min-pct-spoof  fraction of the spoof source-union that must land in
                     dst/spoof/ before exit code 0  (default 0.0 = no-op)

These flags exist so queue_master.cmd stage 3 can fail loudly instead of
writing the .done marker after a partial merge (as happened on 2026-04-22
when the cmd.exe path-parse error killed the stage mid-copy).
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path


def _count_npy(p: Path) -> int:
    if not p.is_dir():
        return 0
    return sum(1 for _ in p.glob("*.npy"))


def main(argv=None) -> int:
    # Backward-compat contract with callers that pre-date the argparse
    # switch (e.g. tests/test_regression_guards.py::test_merge_clip_pools_errors_on_too_few_args):
    # an empty / too-short argv returns 2 with "usage" in stderr — we
    # cannot rely on argparse's SystemExit because callers use `main([])`
    # and expect a return code, not a raised exception.
    raw = list(argv if argv is not None else sys.argv[1:])
    positional = [a for a in raw if not a.startswith("--")]
    if len(positional) < 2:
        print("usage: merge_clip_pools <src_dir1> [<src_dir2> ...] <dst_dir>",
              file=sys.stderr)
        return 2

    ap = argparse.ArgumentParser(description="merge video-clip .npy pools",
                                 allow_abbrev=False)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--min-pct-live", type=float, default=0.0,
                    help="fraction of the live source-union that must "
                         "land in dst/live/ before returning 0 (default 0.0)")
    ap.add_argument("--min-pct-spoof", type=float, default=0.0,
                    help="fraction of the spoof source-union that must "
                         "land in dst/spoof/ before returning 0 (default 0.0)")
    args = ap.parse_args(raw)

    srcs = [Path(p) for p in args.paths[:-1]]
    dst = Path(args.paths[-1])
    (dst / "live").mkdir(parents=True, exist_ok=True)
    (dst / "spoof").mkdir(parents=True, exist_ok=True)

    src_live_total = sum(_count_npy(s / "live") for s in srcs)
    src_spoof_total = sum(_count_npy(s / "spoof") for s in srcs)

    copied = {"live": 0, "spoof": 0, "skipped": 0}
    for src in srcs:
        if not src.is_dir():
            print(f"[merge-clips] skip (not a dir): {src}", file=sys.stderr)
            continue
        for sub in ("live", "spoof"):
            src_sub = src / sub
            if not src_sub.is_dir():
                continue
            dst_sub = dst / sub
            for p in src_sub.glob("*.npy"):
                dst_file = dst_sub / p.name
                if dst_file.exists():
                    copied["skipped"] += 1
                    continue
                shutil.copy(p, dst_file)
                copied[sub] += 1

    live_n = _count_npy(dst / "live")
    spoof_n = _count_npy(dst / "spoof")
    print(f"[merge-clips] copied live={copied['live']} spoof={copied['spoof']} "
          f"skipped={copied['skipped']}")
    print(f"[merge-clips] destination totals: live={live_n} spoof={spoof_n}")
    print(f"[merge-clips] source totals     : live={src_live_total} "
          f"spoof={src_spoof_total}")
    print(f"[merge-clips] {dst}")

    # Post-merge sanity check
    def _pct(got, need_of):
        return (float(got) / float(need_of)) if need_of > 0 else 1.0
    pct_live = _pct(live_n, src_live_total)
    pct_spoof = _pct(spoof_n, src_spoof_total)
    if args.min_pct_live > 0 or args.min_pct_spoof > 0:
        print(f"[merge-clips] sanity: live={pct_live*100:.1f}% "
              f"(min {args.min_pct_live*100:.0f}%), "
              f"spoof={pct_spoof*100:.1f}% (min {args.min_pct_spoof*100:.0f}%)")
        if pct_live < args.min_pct_live:
            print(f"[merge-clips] FAIL: live fill-rate "
                  f"{pct_live*100:.1f}% < min "
                  f"{args.min_pct_live*100:.0f}%", file=sys.stderr)
            return 3
        if pct_spoof < args.min_pct_spoof:
            print(f"[merge-clips] FAIL: spoof fill-rate "
                  f"{pct_spoof*100:.1f}% < min "
                  f"{args.min_pct_spoof*100:.0f}%", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
