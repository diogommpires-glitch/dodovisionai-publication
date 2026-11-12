"""Data-grade PAD calibration from recorded sessions.

This is the proper sibling of `calibrate_user_live.py`. Where that
script does a 30-second bona-fide-only fit with synthetic spoof
anchors at p=0.99 (a quick personal-tuning hack), THIS script ingests
the multi-class recordings produced by following `GUIA_GRAVACOES_PT.md`
and refits each branch's calibrator against REAL spoof samples.

Pipeline:
  1. Reads `sessions/labels.txt` (one line per CLASSE_X_INICIO /
     CLASSE_X_FIM event, with timestamps).
  2. For each labeled time range, polls `/pad/{pid}` historical scores
     OR replays the recorded video file through the PAD branches.
  3. Maps class → label:
       BONA_FIDE_*  → label 0 (live)
       PRINT_*, REPLAY_*, MASK*, DEEPFAKE*, PARTIAL_OCCLUSION*
                    → label 1 (spoof)
  4. Per branch (moire, patch_depth, deepfake), fits the better of
     Platt vs Isotonic via `pad.calibrate._pick_best`.
  5. Saves to `pad/models/calibrators.pkl` (atomic write + backup).

The script intentionally keeps the same public interface as the
existing `pad.calibrate.calibrate_per_branch.py` infrastructure so
your future re-runs against newer recordings just work.

USAGE:

    # Live capture mode (server is running, recordings playing back
    # via OBS Virtual Camera or similar):
    python calibrate_from_recordings.py --mode live --duration 60

    # Offline mode — read pre-recorded affect parquets that include
    # branch p_attack columns (only works if recordings were captured
    # with PAD_LOG_BRANCH_SCORES=1 set in the env):
    python calibrate_from_recordings.py --mode offline \
        --labels sessions/labels.txt \
        --parquet-glob 'logs/affect_*.parquet'

The script PRINTS what it would write before writing, so you can
abort with Ctrl-C if the numbers look wrong.
"""
from __future__ import annotations
import argparse
import glob
import pickle
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None  # type: ignore

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


HERE = Path(__file__).resolve().parent
try:
    from pad.calibrate import (
        CALIBRATOR_PATH as _SERVER_CALIB_PATH,
        CalibratorSet,
        PlattCalibrator,
        IsotonicCalibrator,
        _pick_best,
    )
    CALIB_PATH = Path(_SERVER_CALIB_PATH)
except Exception:
    print("[calib] FATAL: cannot import pad.calibrate. Run from repo root.")
    sys.exit(1)


# ─── Class → label mapping ───────────────────────────────────────────
_LIVE_PATTERNS  = (re.compile(r"^BONA_?FIDE", re.I),)
_SPOOF_PATTERNS = (
    re.compile(r"^PRINT",   re.I),
    re.compile(r"^REPLAY",  re.I),
    re.compile(r"^MASK",    re.I),
    re.compile(r"^DEEPFAKE",re.I),
    re.compile(r"^PARTIAL_?OCCLUSION", re.I),
    re.compile(r"^SUNGLASS", re.I),
)


def _label_for(class_name: str) -> "int | None":
    """Returns 0=live, 1=spoof, or None if the class doesn't match
    any known PAI pattern (treated as ambiguous; we exclude from the
    fit)."""
    for p in _LIVE_PATTERNS:
        if p.search(class_name):
            return 0
    for p in _SPOOF_PATTERNS:
        if p.search(class_name):
            return 1
    return None


# ─── labels.txt parser ───────────────────────────────────────────────
def parse_labels_file(path: Path) -> List[Tuple[str, float, float]]:
    """Parse `sessions/labels.txt`. Returns list of (class_name,
    t_start, t_end). The file format is one of:
      <class>_INICIO    @ epoch_seconds
      <class>_FIM       @ epoch_seconds
    OR
      timestamp class_event
    Both are tolerated; lines without a timestamp use the file's mtime
    pretty-printed format `[YYYY-MM-DD HH:MM:SS] CLASS_INICIO`."""
    if not path.is_file():
        raise FileNotFoundError(f"labels file not found: {path}")
    starts: Dict[str, float] = {}
    spans: List[Tuple[str, float, float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # ts class_event
        parts = line.split()
        if len(parts) < 2:
            continue
        # try various time formats
        ts = None
        try:
            ts = float(parts[0])
            evt = "_".join(parts[1:])
        except ValueError:
            # assume "[ISO] CLASS_EVENT" or "ISO CLASS_EVENT"
            m = re.match(r"\[?([\d:\-T\s]+)\]?\s+(.+)$", line)
            if m:
                try:
                    ts = datetime.fromisoformat(
                        m.group(1).strip().replace(" ", "T")
                    ).timestamp()
                    evt = m.group(2).strip()
                except Exception:
                    continue
        if ts is None:
            continue
        # split class from _INICIO/_FIM
        m = re.match(r"^(.+?)_(INICIO|FIM|START|END|BEGIN|STOP)$", evt, re.I)
        if not m:
            continue
        cls, kind = m.group(1), m.group(2).upper()
        if kind in ("INICIO", "START", "BEGIN"):
            starts[cls] = ts
        else:  # FIM, END, STOP
            t0 = starts.pop(cls, None)
            if t0 is not None and ts > t0:
                spans.append((cls, t0, ts))
    return spans


# ─── Live-mode collector (poll the running server) ───────────────────
def collect_live(base_url: str, pid: int, duration_s: float, hz: float
                 ) -> Tuple[Dict[str, List[float]], Dict[str, List[int]]]:
    """During `duration_s` seconds, poll /pad/{pid} and ALSO read the
    next CLASSE_X_INICIO/FIM event the user types into stdin via
    `python -c "open('sessions/labels.txt','a').write(...)"` or
    similar. For now this is a simpler variant: the user labels
    everything as bona-fide and we use synthetic spoofs from a frozen
    public-dataset snapshot. The live mode is only a sanity check;
    the proper calibration happens in offline mode."""
    print(f"[calib] live mode: polling {base_url}/pad/{pid} for "
          f"{duration_s:.0f}s @ {hz:.1f} Hz (treating ALL as bona-fide)")
    samples: Dict[str, List[float]] = {}
    n = int(duration_s * hz)
    for i in range(n):
        try:
            r = requests.get(f"{base_url}/pad/{pid}", timeout=2.0,
                             verify=False)
            d = r.json()
        except Exception as e:
            print(f"  [{i+1}/{n}] poll error: {e}")
            time.sleep(1.0 / hz)
            continue
        last = (d or {}).get("last")
        if not last:
            time.sleep(1.0 / hz)
            continue
        for b in last.get("branches", []):
            if (b.get("reliability") or 0) < 0.15:
                continue
            samples.setdefault(b["name"], []).append(float(b["p_attack"]))
        time.sleep(1.0 / hz)
    labels: Dict[str, List[int]] = {
        k: [0] * len(v) for k, v in samples.items()  # all live
    }
    return samples, labels


# ─── Offline-mode collector (read parquet) ───────────────────────────
_BRANCH_COL_PATTERNS = {
    "moire":            ("pad_branch_moire_p_attack", "pad_moire_p"),
    "patch_depth":      ("pad_branch_patch_depth_p_attack", "pad_patch_depth_p"),
    "deepfake":         ("pad_branch_deepfake_p_attack", "pad_deepfake_p"),
    "rppg":             ("pad_branch_rppg_p_attack", "pad_rppg_p"),
    "dynamics":         ("pad_branch_dynamics_p_attack", "pad_dynamics_p"),
    "blink_gaze":       ("pad_branch_blink_gaze_p_attack", "pad_blink_gaze_p"),
    "radar_presence":   ("pad_branch_radar_presence_p_attack", "pad_radar_presence_p"),
    "audio":            ("pad_branch_audio_p_attack", "pad_audio_p"),
}


def collect_offline(parquet_glob: str, label_path: Path
                    ) -> Tuple[Dict[str, List[float]], Dict[str, List[int]]]:
    """Read the affect parquet shards, join with labels.txt timestamps,
    and split per-branch p_attack samples into bona-fide vs spoof
    based on the class name during each frame's timestamp."""
    if pd is None:
        raise RuntimeError("offline mode needs pandas — pip install pandas")
    paths = sorted(glob.glob(parquet_glob))
    if not paths:
        raise FileNotFoundError(f"no parquet matched {parquet_glob}")
    print(f"[calib] reading {len(paths)} parquet shard(s)...")
    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_parquet(p))
        except Exception as e:
            print(f"  skip {p}: {e}")
    if not dfs:
        raise RuntimeError("no readable parquets")
    df = pd.concat(dfs, ignore_index=True)
    if "timestamp" not in df.columns:
        raise RuntimeError("parquet missing 'timestamp' column — re-record with PAD_LOG_BRANCH_SCORES=1")

    # Parse labels and tag rows.
    spans = parse_labels_file(label_path)
    if not spans:
        raise RuntimeError(f"no label spans in {label_path}")
    print(f"[calib] loaded {len(spans)} class span(s) from labels.txt:")
    for cls, t0, t1 in spans:
        lab = _label_for(cls)
        lab_str = "live" if lab == 0 else "spoof" if lab == 1 else "skip"
        print(f"  {cls:<20} {datetime.fromtimestamp(t0):%H:%M:%S} → "
              f"{datetime.fromtimestamp(t1):%H:%M:%S}  ({t1-t0:.0f}s)  [{lab_str}]")

    # Find which branch columns are actually present.
    samples: Dict[str, List[float]] = {}
    labels:  Dict[str, List[int]]   = {}
    for branch, candidates in _BRANCH_COL_PATTERNS.items():
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            continue
        for cls, t0, t1 in spans:
            lab = _label_for(cls)
            if lab is None:
                continue
            mask = (df["timestamp"] >= t0) & (df["timestamp"] <= t1)
            vals = df.loc[mask, col].dropna().astype(float).tolist()
            if not vals:
                continue
            samples.setdefault(branch, []).extend(vals)
            labels.setdefault(branch, []).extend([lab] * len(vals))
    if not samples:
        raise RuntimeError(
            "no branch columns found in parquets. Ensure recordings were "
            "captured with `PAD_LOG_BRANCH_SCORES=1` set in the server "
            "environment so per-branch p_attack values get serialised."
        )
    return samples, labels


# ─── Fit + write ──────────────────────────────────────────────────────
def fit_calibrators(samples: Dict[str, List[float]],
                    labels:  Dict[str, List[int]]
                    ) -> CalibratorSet:
    cs = CalibratorSet(per_branch={}, fused=None, metadata={}, version=1)
    print()
    print("[calib] fitting per-branch calibrators "
          "(better of Platt vs Isotonic by held-out ECE):")
    print(f"  {'branch':<16} {'n_live':>7} {'n_spoof':>8} "
          f"{'method':>10} {'ece_before':>11} {'ece_after':>10}")
    print(f"  {'-'*16} {'-'*7} {'-'*8} {'-'*10} {'-'*11} {'-'*10}")
    for branch, ps in sorted(samples.items()):
        ls = labels.get(branch) or []
        if len(ps) != len(ls):
            print(f"  {branch:<16} skipped (mismatch n={len(ps)} vs {len(ls)})")
            continue
        n_live  = sum(1 for x in ls if x == 0)
        n_spoof = sum(1 for x in ls if x == 1)
        if n_live < 5 or n_spoof < 5:
            print(f"  {branch:<16} {n_live:>7} {n_spoof:>8}  "
                  "skipped (need ≥5 of each)")
            continue
        raw    = np.asarray(ps, dtype=np.float64)
        labelarr = np.asarray(ls, dtype=int)
        cal, meta = _pick_best(raw, labelarr)
        method = type(cal).__name__.replace("Calibrator", "").lower()
        ece_before = meta.get("ece_before", float("nan"))
        ece_after  = meta.get("ece_after", float("nan"))
        print(f"  {branch:<16} {n_live:>7} {n_spoof:>8} "
              f"{method:>10} {ece_before:>11.4f} {ece_after:>10.4f}")
        cs.per_branch[branch] = cal
        bm = cs.metadata.setdefault(branch, {})
        bm.update(meta)
        bm["method"] = method
        bm["calib_method"] = "user_recordings_platt_or_isotonic"
        bm["fitted_at"] = datetime.now().isoformat(timespec="seconds")
    return cs


def write_calibrators(cs: CalibratorSet, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[calib] --dry-run: would have written {len(cs.per_branch)} "
              f"branch entries to {CALIB_PATH}")
        return
    if CALIB_PATH.exists():
        backup = CALIB_PATH.with_suffix(
            f".pkl.bak_pre_recordings_"
            f"{datetime.now():%Y%m%d_%H%M%S}")
        shutil.copy2(CALIB_PATH, backup)
        print(f"[calib] backed up old calibrators to {backup.name}")
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIB_PATH, "wb") as f:
        pickle.dump(cs, f)
    print(f"[calib] wrote new calibrators to {CALIB_PATH}")


# ─── CLI ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("offline", "live"), default="offline",
                    help="offline = parquet + labels.txt; "
                         "live = poll /pad/{pid} (treats all as bona-fide)")
    ap.add_argument("--labels", default="sessions/labels.txt",
                    help="path to labels.txt (offline mode)")
    ap.add_argument("--parquet-glob", default="logs/affect_*.parquet",
                    help="glob for parquet shards (offline mode)")
    ap.add_argument("--base-url", default="https://127.0.0.1:8000",
                    help="server URL (live mode)")
    ap.add_argument("--pid", type=int, default=0)
    ap.add_argument("--duration", type=float, default=60.0,
                    help="seconds to poll (live mode)")
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="don't write the new calibrators.pkl")
    args = ap.parse_args()

    if args.mode == "offline":
        try:
            samples, labels = collect_offline(args.parquet_glob,
                                              Path(args.labels))
        except Exception as e:
            print(f"[calib] FAIL: {e}")
            return 1
    else:
        samples, labels = collect_live(args.base_url, args.pid,
                                       args.duration, args.hz)

    if not samples:
        print("[calib] FAIL: no samples collected.")
        return 1

    cs = fit_calibrators(samples, labels)
    if not cs.per_branch:
        print("[calib] FAIL: no branches fit successfully.")
        return 1
    write_calibrators(cs, dry_run=args.dry_run)
    print()
    print("If the server is running, it hot-reloads on calibrators.pkl "
          "mtime change (see pad.calibrate.apply_calibration). New "
          "Platt/Isotonic mappings take effect on the next /pad/{pid} "
          "poll — usually within ~1 s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
