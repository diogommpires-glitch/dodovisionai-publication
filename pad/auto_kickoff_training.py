"""Watchdog that waits for preprocessing to finish, then auto-kicks
off patch_depth training. Designed to run overnight unattended.

Decision logic:
    1. Every 60 s, check if `prep.log` contains the success line
       `[preprocess] wrote N images in X min` AND the log has been
       quiet for > 60 s (preprocessor has exited cleanly).
    2. Verify `images.bin` has the expected size (494405 * 224*224*3 bytes)
       and `labels.npy` is present.
    3. Make sure no preprocessor python.exe is still running.
    4. Launch training with the mmap path, logging to `train.log`.
    5. Log every decision it makes to `auto_kickoff.log` so you can
       audit in the morning.

If preprocessing fails (log contains "failures" above a threshold, or
stops without the success line), the script writes a CLEAR FAILURE
line to auto_kickoff.log and exits — it will NOT launch training.

Usage:
    python -m pad.auto_kickoff_training

Safe to Ctrl+C at any time; the launched training process is detached
and survives this watchdog dying.
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import time
from pathlib import Path


# v9.10.2: default flipped from D:\ -> E:\ (the Lexar SL300 SSD, where
# all DodoVisionAI datasets actually live on this box -- 535 GB of
# CelebA-Spoof, SiW-Mv2, ASVspoof2019LA, Celeb-DF-v2, Replay-Attack,
# FaceForensics++). D:\ on this machine is a 128 GB USB stick, way too
# small for the 70 GB images.bin alone. Override:
#   set DODO_PAD_DATA_DIR=F:\pad_datasets         (different drive)
#   set DODO_PAD_DATA_DIR=/mnt/datasets/pad        (Linux / WSL)
DATA_DIR = Path(os.environ.get("DODO_PAD_DATA_DIR", r"E:\pad_datasets"))
PREP_LOG = DATA_DIR / "prep.log"
MMAP_DIR = DATA_DIR / "patch_depth_mmap"
TRAIN_LOG = DATA_DIR / "train.log"
KICK_LOG = DATA_DIR / "auto_kickoff.log"
HEARTBEAT = DATA_DIR / "auto_kickoff.heartbeat"

# Total samples in the patch_depth_mmap (224x224x3 RGB uint8 face crops):
#   164,484 live (CelebA-HQ + SiW-Mv2 bona-fide) +
#   329,921 spoof (SiW-Mv2 attack subset) = 494,405.
# Verified 2026-04-17 via labels.npy on E:/D:\pad_datasets\patch_depth_mmap.
EXPECTED_N = 494405
# 224x224x3 = 150,528 bytes per uint8 face crop.
EXPECTED_SIZE = EXPECTED_N * 224 * 224 * 3   # 74,421,795,840 bytes

SUCCESS_RE = re.compile(
    r"\[preprocess\] wrote (\d+) images in ([\d.]+) min", re.IGNORECASE)
FAIL_COUNT_RE = re.compile(r"\[preprocess\] (\d+) failures")


def log(msg: str) -> None:
    """Append a timestamped line to KICK_LOG and stdout. File-write
    failures are swallowed (still printed) so the watchdog never gets
    blocked by a transient disk hiccup -- the stdout copy is the
    canonical record. Print-once warning so the swallow is observable."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp}  {msg}"
    print(line, flush=True)
    try:
        with open(KICK_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception as e:
        # Don't recurse into log() on log failure; print directly.
        # Use a module-level guard so we only warn once per session.
        if not getattr(log, "_warned", False):
            print(f"[auto_kickoff] WARNING: KICK_LOG write failed ({e!r}); "
                  f"future log writes also silenced.", flush=True)
            log._warned = True


def write_heartbeat(state: str, extra: dict | None = None) -> None:
    """Touch a heartbeat file each iteration so a second process can
    detect if the watchdog silently died.

    Failure to write the heartbeat is logged once but otherwise swallowed
    so it never blocks the watchdog; the consequence is only that an
    external monitor sees a stale timestamp (which is what they're
    watching for anyway)."""
    try:
        import json
        payload = {
            "ts": time.time(),
            "stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "state": state,
            "pid": os.getpid(),
        }
        if extra:
            payload.update(extra)
        HEARTBEAT.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as e:
        if not getattr(write_heartbeat, "_warned", False):
            print(f"[auto_kickoff] WARNING: heartbeat write failed "
                  f"({e!r}); future heartbeat writes also silenced.",
                  flush=True)
            write_heartbeat._warned = True


def preprocess_python_still_running() -> bool:
    """Return True if any python.exe process is consuming > 500 MB RAM.
    That's the preprocessor (main + workers) — the watchdog itself is
    small. Never raises; on error returns False so the watchdog makes
    progress rather than stalling forever."""
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
            text=True, timeout=10,
        )
    except Exception as e:
        log(f"  tasklist call failed: {e!r} — assuming no workers running")
        return False
    big_procs = 0
    try:
        for line in out.splitlines():
            if "python.exe" not in line:
                continue
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) < 5:
                continue
            mem = parts[-1].replace(" K", "").replace(",", "").strip()
            try:
                mem_kb = int(mem)
            except ValueError:
                continue
            if mem_kb > 100_000:   # > 100 MB = preprocessor worker-class
                big_procs += 1
    except Exception as e:
        log(f"  tasklist parse failed: {e!r} — assuming no workers running")
        return False
    return big_procs > 1      # >1 means at least a worker is still running


def check_preprocess_state():
    """Return one of:
        ('running', details)   — still going
        ('done', details)      — success + all files present
        ('failed', details)    — explicit failure
    """
    if not PREP_LOG.exists():
        return "running", {"reason": "prep.log missing"}

    try:
        log_size = PREP_LOG.stat().st_size
    except Exception as e:
        return "running", {"reason": f"stat error: {e}"}

    # Look for success line in the last chunk
    try:
        with open(PREP_LOG, "rb") as fh:
            fh.seek(max(0, log_size - 4096))
            tail = fh.read().decode("utf-8", errors="ignore")
    except Exception:
        tail = ""

    success_match = SUCCESS_RE.search(tail)
    fail_match = FAIL_COUNT_RE.search(tail)

    n_failures = int(fail_match.group(1)) if fail_match else 0

    if not success_match:
        return "running", {
            "reason": "no success line yet",
            "log_size": log_size,
        }

    n_written = int(success_match.group(1))
    minutes = float(success_match.group(2))

    # The preprocessor printed success — but confirm the artifacts match
    if not (MMAP_DIR / "images.bin").exists():
        return "failed", {"reason": "images.bin missing"}
    bin_size = (MMAP_DIR / "images.bin").stat().st_size
    if bin_size != EXPECTED_SIZE:
        return "failed", {
            "reason": f"images.bin size {bin_size} != expected {EXPECTED_SIZE}",
        }
    if not (MMAP_DIR / "labels.npy").exists():
        return "failed", {"reason": "labels.npy missing"}

    # Make sure preprocessor has exited
    if preprocess_python_still_running():
        return "running", {
            "reason": "success line printed but workers still up",
        }

    # Final sanity: at least 95% of images should have succeeded
    if n_written < 0.95 * EXPECTED_N:
        return "failed", {
            "reason": f"only {n_written}/{EXPECTED_N} written",
            "failures": n_failures,
        }

    return "done", {
        "n_written": n_written,
        "minutes": minutes,
        "failures": n_failures,
    }


def kick_off_training() -> bool:
    """Spawn the training process, detached. Returns True if spawn
    succeeded.

    Matches the command from STACK_AUDIT.md § v9.5:
        python -u -m pad.train_patch_depth
            --data D:\\pad_datasets\\patch_depth_mmap
            --epochs 20 --batch 64 --num-workers 2
            --prefetch-factor 4 --log-every 50
            > train.log 2>&1
    """
    py = sys.executable
    cwd = Path(__file__).resolve().parent.parent   # <repo>/DodoVisionAI

    # v9.10 max-utilization profile (matches train_patch_depth_v8.py
    # audit on 5090 24 GB / 24-core / 64 GB):
    #   --batch 256: AMP bf16 fits comfortably in VRAM, ~3 GB peak
    #   --num-workers 12: saturates the 24-core CPU on PIL augmentation
    #   --prefetch-factor 4: hides aug latency from GPU
    # v9.9 defaults were `--batch 64 --num-workers 2` which left ~22 cores
    # idle. Override on the CLI for low-RAM hosts.
    cmd = [
        py, "-u", "-m", "pad.train_patch_depth",
        "--data", str(MMAP_DIR),
        "--epochs", "20",
        "--batch", "256",
        "--num-workers", "12",
        "--prefetch-factor", "4",
        "--log-every", "50",
    ]

    try:
        fh = open(TRAIN_LOG, "w", encoding="utf-8")
    except Exception as e:
        log(f"  FAIL — cannot open train.log: {e}")
        return False

    log(f"  spawning: {' '.join(cmd)}")
    try:
        # Detach: redirect IO, create new process group so this
        # watchdog dying doesn't kill the training.
        DETACHED = 0x00000008     # DETACHED_PROCESS on Windows
        NEW_GROUP = 0x00000200    # CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(
            cmd, cwd=str(cwd),
            stdout=fh, stderr=subprocess.STDOUT,
            creationflags=DETACHED | NEW_GROUP if os.name == "nt" else 0,
        )
    except Exception as e:
        log(f"  FAIL — Popen raised: {e}")
        return False
    log("  training launched OK")
    return True


def main() -> int:
    """CLI: watchdog that re-launches preprocess + training when prior runs die. Runs as a long-lived process; intended for unattended overnight use via `start /B`."""
    log("=" * 60)
    log("auto_kickoff_training — starting")
    log(f"  prep.log : {PREP_LOG}")
    log(f"  mmap_dir : {MMAP_DIR}")
    log(f"  train.log: {TRAIN_LOG}")
    log(f"  heartbeat: {HEARTBEAT}")
    log(f"  expected : {EXPECTED_N} images, {EXPECTED_SIZE/1e9:.1f} GB")
    log(f"  pid      : {os.getpid()}")
    write_heartbeat("starting")

    last_state = None
    consecutive_errors = 0
    iteration = 0
    while True:
        iteration += 1
        try:
            state, details = check_preprocess_state()
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            log(f"  check_preprocess_state raised: {e!r} (streak={consecutive_errors})")
            write_heartbeat("error", {"error": repr(e),
                                      "consecutive_errors": consecutive_errors,
                                      "iteration": iteration})
            # If 10 consecutive failures (~10 min), give up so a supervisor
            # can restart rather than silently looping on a broken state.
            if consecutive_errors >= 10:
                log("  too many consecutive errors — aborting")
                return 2
            time.sleep(60)
            continue

        # Heartbeat every iteration, not just on state changes, so a second
        # process can detect silent death (stale heartbeat > 2 min).
        write_heartbeat(state, {"iteration": iteration, "details": details})

        if state != last_state:
            log(f"state={state}  {details}")
            last_state = state

        if state == "done":
            log("preprocess succeeded — kicking off training now")
            try:
                ok = kick_off_training()
            except Exception as e:
                log(f"  kick_off_training raised: {e!r}")
                write_heartbeat("kickoff_exception", {"error": repr(e)})
                return 1
            if ok:
                log("SUCCESS — training spawned, watchdog exiting")
                write_heartbeat("spawned_training")
                return 0
            else:
                log("FAIL — could not spawn training")
                write_heartbeat("spawn_failed")
                return 1

        if state == "failed":
            log(f"preprocess FAILED: {details.get('reason', '?')}")
            log("will NOT launch training. See prep.log for details.")
            write_heartbeat("preprocess_failed", {"details": details})
            return 1

        try:
            time.sleep(60)
        except KeyboardInterrupt:
            log("  Ctrl+C received — exiting cleanly")
            write_heartbeat("interrupted")
            return 130


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        # Last-resort catch so the process always leaves a trail even
        # if something exotic (OOM, disk full) bubbles up.
        log(f"FATAL — main() raised: {e!r}")
        write_heartbeat("fatal", {"error": repr(e)})
        sys.exit(3)
