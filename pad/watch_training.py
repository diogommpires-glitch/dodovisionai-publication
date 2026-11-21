"""Training health watchdog. Run in a second cmd window.

Polls the training log + GPU + process status every 60 s, prints one
status line, and alerts loudly if:
  - no log output for > 30 min (training stalled)
  - GPU utilization < 30 % averaged over 5 min (HDD starved)
  - python.exe process died
  - disk free < 5 GB on D: (out of space risk)

Usage:
    python -m pad.watch_training --log D:\\pad_datasets\\train.log
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path


def gpu_stats():
    """Return (util_pct, mem_mb) or (None, None)."""
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


def python_procs():
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/NH"],
            text=True, timeout=5,
        )
        return sum(1 for line in out.splitlines() if "python.exe" in line)
    except Exception:
        return None


def log_tail(path: Path, n: int) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 2048))
            tail = f.read().decode("utf-8", errors="ignore")
        lines = tail.splitlines()[-n:]
        return "\n".join(lines)
    except FileNotFoundError:
        return "(log missing)"


def parse_epoch(log_tail_str: str):
    """Return (current_epoch, total_epochs, last_val_acc) or (None,None,None)."""
    import re
    last = None
    for line in log_tail_str.splitlines():
        m = re.match(r"\[epoch (\d+)/(\d+)\].*val_acc=([\d.]+)", line)
        if m:
            last = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
    return last if last else (None, None, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="D:/pad_datasets/train.log")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--data-disk", default="D:\\")
    args = ap.parse_args()

    log = Path(args.log)
    gpu_hist = []
    last_log_size = -1
    last_log_change = time.time()

    print(f"[watch] polling {log} every {args.interval}s. Ctrl+C to stop.")
    print(f"{'time':<8}  {'GPU%':>5}  {'VRAM':>6}  {'procs':>5}  "
          f"{'log':>6}  {'free':>8}  epoch / status")

    while True:
        now = time.strftime("%H:%M:%S")
        util, mem = gpu_stats()
        procs = python_procs()
        cur_size = log.stat().st_size if log.exists() else 0
        if cur_size != last_log_size:
            last_log_change = time.time()
            last_log_size = cur_size
        silent_for = (time.time() - last_log_change) / 60
        try:
            du = shutil.disk_usage(args.data_disk)
            free_gb = du.free / 1e9
        except Exception:
            free_gb = -1

        tail = log_tail(log, 40)
        ep, total_ep, val_acc = parse_epoch(tail)

        if util is not None:
            gpu_hist.append(util)
            gpu_hist = gpu_hist[-5:]
        avg_gpu = sum(gpu_hist) / len(gpu_hist) if gpu_hist else 0

        status = ""
        if procs is None or procs == 0:
            status = "!! NO PYTHON PROCESS — training died"
        elif silent_for > 30:
            status = f"!! LOG SILENT {silent_for:.0f} MIN — stalled?"
        elif avg_gpu < 30 and len(gpu_hist) == 5:
            status = f"!! GPU starved ({avg_gpu:.0f}% avg, HDD bottleneck)"
        elif free_gb != -1 and free_gb < 5:
            status = f"!! DATA DISK {free_gb:.1f} GB FREE — space low"
        elif ep is not None:
            status = f"epoch {ep}/{total_ep}  val_acc={val_acc:.3f}"
        else:
            status = "warming up (first epoch)"

        util_s = f"{util:4.0f}%" if util is not None else "   ?"
        mem_s = f"{mem:5.0f}M" if mem is not None else "     ?"
        procs_s = f"{procs:3d}" if procs is not None else "  ?"
        free_s = f"{free_gb:5.1f}GB" if free_gb != -1 else "    ?"

        print(f"{now}  {util_s}  {mem_s}  {procs_s}  "
              f"{cur_size:6d}  {free_s}  {status}")
        time.sleep(args.interval)


if __name__ == "__main__":
    import sys
    sys.exit(main())
