"""One-shot status dashboard. Prints a snapshot of the preprocess,
auto-kicker, and training state in one block.

Usage:
    python -m pad.status
"""
from __future__ import annotations
import os
import re
import subprocess
import time
from pathlib import Path


PREP_LOG = Path(r"D:\pad_datasets\prep.log")
KICK_LOG = Path(r"D:\pad_datasets\auto_kickoff.log")
TRAIN_LOG = Path(r"D:\pad_datasets\train.log")
MMAP_DIR = Path(r"D:\pad_datasets\patch_depth_mmap")
# Total samples in the patch_depth_mmap (224x224x3 RGB uint8 face crops):
#   164,484 live + 329,921 spoof. Verified 2026-04-17 via labels.npy.
# Kept in sync with pad/auto_kickoff_training.py:EXPECTED_N -- if you
# change one, change the other.
EXPECTED_N = 494405
# 224x224x3 = 150,528 bytes per uint8 face crop.
EXPECTED_SIZE = EXPECTED_N * 224 * 224 * 3


def tail_bytes(path: Path, n: int = 4096) -> str:
    """Return the last n bytes of `path` as utf-8 (errors='ignore').
    Returns '' on any IO error -- callers use the empty string as the
    'log not yet readable' signal so a transient lock or a partial
    write doesn't crash the status endpoint."""
    if not path.exists():
        return ""
    try:
        with open(path, "rb") as fh:
            fh.seek(max(0, path.stat().st_size - n))
            return fh.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def prep_status() -> dict:
    out = {"exists": PREP_LOG.exists()}
    if not out["exists"]:
        return out
    out["size_kb"] = PREP_LOG.stat().st_size // 1024
    out["mtime_age_s"] = int(time.time() - PREP_LOG.stat().st_mtime)
    t = tail_bytes(PREP_LOG, 8192)
    # tqdm bar has digits/hashes/spaces; just accept anything between the two |s
    m = re.findall(
        r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([^,\s]+),\s*([\d.]+)\s*img/s",
        t,
    )
    if m:
        pct, done, tot, elapsed, remain, rate = m[-1]
        out["pct"] = int(pct)
        out["done"] = int(done)
        out["total"] = int(tot)
        out["elapsed"] = elapsed
        out["remaining"] = remain
        out["rate"] = float(rate)
    done_re = re.search(
        r"\[preprocess\] wrote (\d+) images in ([\d.]+) min", t)
    if done_re:
        out["finished"] = True
        out["wrote"] = int(done_re.group(1))
    else:
        out["finished"] = False
    return out


def kick_status() -> dict:
    out = {"exists": KICK_LOG.exists()}
    if not out["exists"]:
        out["verdict"] = "not running"
        return out
    t = tail_bytes(KICK_LOG, 4096)
    lines = [l for l in t.splitlines() if l.strip()]
    out["last_line"] = lines[-1] if lines else ""
    if "SUCCESS — training spawned" in t:
        out["verdict"] = "handed off to training"
    elif "preprocess FAILED" in t:
        out["verdict"] = "FAILURE — will NOT launch training"
    elif "state=running" in out["last_line"]:
        out["verdict"] = "polling"
    else:
        out["verdict"] = "unclear"
    return out


def train_status() -> dict:
    out = {"exists": TRAIN_LOG.exists()}
    if not out["exists"]:
        return out
    out["size_kb"] = TRAIN_LOG.stat().st_size // 1024
    out["mtime_age_s"] = int(time.time() - TRAIN_LOG.stat().st_mtime)
    t = tail_bytes(TRAIN_LOG, 8192)
    epochs = re.findall(
        r"\[epoch (\d+)/(\d+)\].*val_acc=([\d.]+)", t)
    batches = re.findall(
        r"\[ep (\d+)/(\d+) batch\s+(\d+)/(\d+)\][^(]*?"
        r"loss=([\d.]+)\s+acc=([\d.]+)\s+(\d+) img/s\s+ETA\s+([\d.]+) min",
        t,
    )
    if epochs:
        e, te, va = epochs[-1]
        out["last_epoch"] = f"{e}/{te} (val_acc={va})"
    if batches:
        e, te, b, tb, loss, acc, ips, eta = batches[-1]
        out["last_batch"] = (
            f"ep {e}/{te}  batch {b}/{tb}  "
            f"loss={loss} acc={acc} {ips} img/s ETA {eta} min"
        )
    if "[export] wrote" in t:
        out["finished"] = True
    else:
        out["finished"] = False
    return out


def python_procs():
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
            text=True, timeout=5,
        )
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        if "python.exe" not in line:
            continue
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) < 5:
            continue
        try:
            mem_kb = int(parts[-1].replace(" K", "").replace(",", "").strip())
        except ValueError:
            mem_kb = 0
        rows.append({"pid": parts[1], "mem_mb": mem_kb // 1024})
    return rows


def gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        parts = out.strip().split(",")
        return float(parts[0]), float(parts[1])
    except Exception:
        return None, None


def disk_free_gb(drive: str = r"D:\\") -> float:
    try:
        import shutil
        return shutil.disk_usage(drive).free / 1e9
    except Exception:
        return -1


def line(c="-", n=70):
    print(c * n)


def _render_once():
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print()
    line("=")
    print(f"  DODO VISION AI — PAD training pipeline status  @ {now}")
    line("=")

    # Preprocess
    p = prep_status()
    print("\n[1/3] PREPROCESS")
    if not p.get("exists"):
        print("  prep.log not found — preprocess never started")
    elif p.get("finished"):
        print(f"  ✅ DONE — wrote {p.get('wrote', '?'):,} images")
    elif "pct" in p:
        print(f"  RUNNING — {p['pct']}% ({p['done']:,} / {p['total']:,})")
        print(f"    elapsed  : {p['elapsed']}")
        print(f"    remaining: {p['remaining']}")
        print(f"    rate     : {p['rate']:.1f} img/s")
        print(f"    log age  : {p['mtime_age_s']}s  "
              f"{'(STALE — check process)' if p['mtime_age_s'] > 120 else '(fresh)'}")
    else:
        print(f"  starting up — log {p.get('size_kb')} KB, no progress line yet")

    # Auto-kicker
    k = kick_status()
    print("\n[2/3] AUTO-KICKER")
    if not k.get("exists"):
        print("  ❌ auto_kickoff.log missing — auto-kicker NOT running")
    else:
        print(f"  verdict: {k.get('verdict')}")
        print(f"  last   : {k.get('last_line')}")

    # Training
    t = train_status()
    print("\n[3/3] TRAINING")
    if not t.get("exists"):
        print("  train.log not created yet — training not launched")
    elif t.get("finished"):
        print(f"  ✅ DONE — ONNX exported. Restart main.py with FF_PAD_CNN=1.")
    else:
        if t.get("last_batch"):
            print(f"  RUNNING — {t['last_batch']}")
        if t.get("last_epoch"):
            print(f"  last completed epoch: {t['last_epoch']}")
        print(f"  log age: {t['mtime_age_s']}s  "
              f"{'(STALE)' if t['mtime_age_s'] > 600 else '(fresh)'}")

    # System
    procs = python_procs()
    util, vram = gpu_stats()
    free = disk_free_gb()
    print("\n[sys]  python.exe processes:")
    for pr in procs:
        print(f"       pid={pr['pid']}  mem={pr['mem_mb']:>4} MB")
    print(f"       GPU util : {util:.0f}%" if util is not None else "       GPU util : ?")
    print(f"       GPU VRAM : {vram:.0f} MB" if vram is not None else "       GPU VRAM : ?")
    print(f"       D: free  : {free:.1f} GB" if free > 0 else "       D: free  : ?")

    line("=")
    print()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch", type=float, default=0,
                    help="auto-refresh every N seconds (default 0 = one-shot)")
    args = ap.parse_args()

    if args.watch <= 0:
        _render_once()
        return 0

    # Live watch mode — clear screen between renders.
    try:
        while True:
            # Windows clear-screen + ANSI reset to reduce flicker
            os.system("cls" if os.name == "nt" else "clear")
            _render_once()
            print(f"(refreshing every {args.watch:.0f}s — Ctrl+C to stop)")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\n[status] watch stopped.")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
