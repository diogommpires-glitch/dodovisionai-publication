"""Quick user-bona-fide PAD calibration.

Use case: the public-dataset Platt calibrators in `pad/models/calibrators.pkl`
were fit against ASVspoof / CelebA-Spoof / Replay-Attack distributions, which
don't include a specific operator's face + lighting + glasses combination. The
result is a strong false-positive on a real face — patch_depth fires
p=0.99 [PRINT], moire p=0.72 [REPLAY_SCREEN], driving the fused verdict to
SPOOF p=0.91 even though the subject is alive.

This script does NOT replace the dataset-trained calibrators (those are still
needed to detect real attacks). It ADDS personal anchor points: while
recording the user's actual face for ~30 s, it observes the per-branch
output distribution, then refits the Platt scaler with synthetic spoof
anchors at p=1.0 mixed with the observed live samples. The result:

  * Branches that consistently say "spoof" against this real face get
    pulled down toward p=0.1.
  * Branches that already say "live" stay where they are.
  * Real spoofs (which would push the branches even higher than the
    observed live mean) still cross the SPOOF threshold — the scaler
    doesn't flatten the curve, it shifts it.

Usage (with the server running, looking at a real live face):
    python -m pad.calibrate_user_live --duration 30 --pid 0
or:
    python pad/calibrate_user_live.py --duration 30 --pid 0

The new calibrators are written to `pad/models/calibrators.pkl`. The server's
`apply_calibration` hot-reloads via mtime so no restart is needed.
"""
from __future__ import annotations
import argparse
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests
import urllib3
# v9.7.4: `start_inst.cmd` runs the server with a self-signed TLS cert
# on port 8000. Calling http://... gets RemoteDisconnected (server
# rejects non-TLS handshake), and calling https://... triggers a
# certificate-verify warning on every poll. Suppress that warning so
# the polling output stays readable; we're hitting localhost so MITM
# concerns don't apply.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


HERE = Path(__file__).resolve().parent
# v9.7.4 BUGFIX: when the script lived in `pad/`, this used to evaluate
# to `pad/models/calibrators.pkl`. After moving the script to the repo
# root (to avoid `pad/types.py` shadowing stdlib `types`), the relative
# path silently resolved to `<repo>/models/calibrators.pkl` — a phantom
# file the server never reads. The pickle was written successfully and
# 3 branches were "personalised" but the server kept using the
# unchanged `pad/models/calibrators.pkl`, so the PAD panel never
# updated. Now: import the canonical path from `pad.calibrate` so the
# script and the server are guaranteed to agree on file location, and
# we get a `from pad.calibrate import CALIBRATOR_PATH as _CP`-style
# single source of truth that survives any future refactor.
try:
    from pad.calibrate import CALIBRATOR_PATH as _SERVER_CALIB_PATH
    CALIB_PATH = Path(_SERVER_CALIB_PATH)
except Exception:
    # Last-resort fallback if pad.calibrate isn't importable for some
    # reason — point at the path the server actually uses on this
    # machine. The path is repo-relative so this still works after
    # `git clone` to a different folder.
    CALIB_PATH = HERE / "pad" / "models" / "calibrators.pkl"


def _backup_calibrators() -> Path:
    """Make a timestamped copy of calibrators.pkl before overwriting,
    so we can roll back if the personal calibration overshoots."""
    if not CALIB_PATH.exists():
        return CALIB_PATH
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = CALIB_PATH.with_suffix(f".pkl.bak_pre_user_live_{stamp}")
    shutil.copy2(CALIB_PATH, backup)
    return backup


def _resolve_base_url(base_url: str) -> str:
    """Detect whether the server is running with TLS (start_inst.cmd) or
    plain HTTP (start.bat) and return the working URL. The user's
    launcher of choice differs by host setup — we shouldn't make them
    edit the script every time."""
    candidates = [base_url]
    if base_url.startswith("http://"):
        candidates.append("https://" + base_url[len("http://"):])
    elif base_url.startswith("https://"):
        candidates.append("http://" + base_url[len("https://"):])
    for url in candidates:
        try:
            r = requests.get(f"{url}/pad/stats", timeout=1.5, verify=False)
            if r.status_code < 500:
                if url != base_url:
                    print(f"[calib] note: {base_url} unreachable, "
                          f"using {url} instead "
                          f"(server is running with "
                          f"{'TLS' if url.startswith('https') else 'plain HTTP'})")
                return url
        except Exception:
            continue
    # No candidate worked — return the original; the poll loop will
    # surface the connection error per-poll for the user to see.
    return base_url


def _poll_pad(base_url: str, pid: int, duration_s: float, hz: float = 1.0
              ) -> Dict[str, List[float]]:
    """Hit /pad/{pid} for `duration_s` seconds, collect per-branch
    p_attack values. Returns {branch_name: [p, p, p, ...]}."""
    base_url = _resolve_base_url(base_url)
    samples: Dict[str, List[float]] = {}
    n_polls = int(duration_s * hz)
    interval = 1.0 / hz
    print(f"[calib] polling {base_url}/pad/{pid} every {interval:.1f}s "
          f"for {duration_s:.0f}s ({n_polls} polls)...")
    skipped_warmup = 0
    for i in range(n_polls):
        try:
            r = requests.get(f"{base_url}/pad/{pid}", timeout=2.0,
                             verify=False)
            data = r.json()
        except Exception as e:
            print(f"  [{i+1:2d}/{n_polls}] poll failed: {e}")
            time.sleep(interval)
            continue
        last = (data or {}).get("last")
        if not last:
            print(f"  [{i+1:2d}/{n_polls}] no PAD state yet")
            time.sleep(interval)
            continue
        for b in last.get("branches", []):
            name = b.get("name")
            p = b.get("p_attack")
            rel = b.get("reliability") or 0.0
            if name is None or p is None:
                continue
            # Skip warmup branches — they sit at p=0.5 with rel<0.1
            # and would pollute the personal-live distribution with
            # uninformative samples.
            if rel < 0.15:
                skipped_warmup += 1
                continue
            samples.setdefault(name, []).append(float(p))
        live_count = sum(len(v) for v in samples.values())
        print(f"  [{i+1:2d}/{n_polls}] verdict={last.get('verdict','?')} "
              f"p_fused={last.get('p_attack',0):.3f} "
              f"branches_collected={live_count} "
              f"skipped_warmup={skipped_warmup}")
        time.sleep(interval)
    print(f"[calib] collected per branch:")
    for name, ps in sorted(samples.items()):
        if ps:
            arr = np.array(ps)
            print(f"  {name:<16} n={len(ps):3d} "
                  f"mean={arr.mean():.3f} median={np.median(arr):.3f} "
                  f"std={arr.std():.3f}")
    return samples


def _fit_personal_platt(live_ps: np.ndarray,
                        n_synth_spoof: int = 30
                        ) -> "object":
    """Fit a Platt calibrator with the user's observed live samples
    (label=0) plus synthetic spoof anchors (label=1) at p=0.99.

    We re-use the existing `PlattCalibrator` from pad.calibrate so the
    pickle format stays identical and the server's `transform_branch`
    interface keeps working unchanged.

    """
    from pad.calibrate import PlattCalibrator
    if len(live_ps) < 5:
        # Too few samples — return identity to avoid harmful overshoot
        return PlattCalibrator(a=1.0, b=0.0, n=0)
    # Synthetic spoof anchors at p=0.99 (just below 1.0 to avoid
    # logit blowup). 30 anchors against ~30 live samples gives a 50/50
    # class balance, which Platt expects.
    spoof_ps = np.full(n_synth_spoof, 0.99, dtype=np.float64)
    raw = np.concatenate([live_ps.astype(np.float64), spoof_ps])
    labels = np.concatenate([
        np.zeros(len(live_ps), dtype=int),
        np.ones(n_synth_spoof,  dtype=int),
    ])
    return PlattCalibrator.fit(raw, labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000",
                    help="PAD server URL (default: http://127.0.0.1:8000)")
    ap.add_argument("--pid", type=int, default=0,
                    help="Person/face id to poll (default: 0)")
    ap.add_argument("--duration", type=float, default=30.0,
                    help="Seconds of live face to record (default: 30)")
    ap.add_argument("--hz", type=float, default=2.0,
                    help="Polls per second (default: 2)")
    ap.add_argument("--branches", default="moire,patch_depth,deepfake",
                    help="Comma-list of branches to recalibrate. "
                         "Cheap branches (rppg, dynamics, blink_gaze, "
                         "radar_presence) are skipped because they're "
                         "already calibrated against intra-modality "
                         "physics and don't suffer the dataset-mismatch "
                         "false-positive problem the heavy ML branches do.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would change but don't write the pkl.")
    args = ap.parse_args()

    target_branches = [b.strip() for b in args.branches.split(",") if b.strip()]
    print(f"[calib] target branches: {target_branches}")

    # 1. Collect live samples.
    samples = _poll_pad(args.base_url, args.pid, args.duration, args.hz)
    if not samples:
        print("[calib] FAIL: no samples collected. Is the server running "
              "and showing a face?")
        return 1

    # 2. Load existing calibrator set so we don't lose other branches.
    from pad.calibrate import load_calibrators, CalibratorSet, PlattCalibrator
    cs = load_calibrators()
    if cs is None:
        # Build a fresh empty set if nothing exists yet
        cs = CalibratorSet(per_branch={}, fused=None, metadata={}, version=1)
        print("[calib] no existing calibrators.pkl; starting fresh")
    else:
        print(f"[calib] loaded existing calibrators with "
              f"{len(cs.per_branch)} branch entries")

    # 3. Fit personal Platt for each target branch.
    print()
    print("[calib] fitting per-branch personal calibrators:")
    print(f"  {'branch':<16} {'n':>4} {'mean_raw':>9} "
          f"{'mean_after':>11} {'shift':>8}")
    print(f"  {'-'*16} {'-'*4} {'-'*9} {'-'*11} {'-'*8}")
    n_changed = 0
    for name in target_branches:
        ps = samples.get(name)
        if not ps or len(ps) < 5:
            print(f"  {name:<16} skipped (n={len(ps or [])} < 5)")
            continue
        arr = np.array(ps)
        new_cal = _fit_personal_platt(arr)
        # Sanity-check: apply the new calibrator to the same samples
        # and report what they map to.
        after = np.array([new_cal.transform(p) for p in ps])
        shift = arr.mean() - after.mean()
        print(f"  {name:<16} {len(ps):>4} "
              f"{arr.mean():>9.3f} {after.mean():>11.3f} {shift:>+8.3f}")
        cs.per_branch[name] = new_cal
        meta = cs.metadata.setdefault(name, {})
        meta["calib_method"] = "user_live_platt"
        meta["live_samples_n"] = float(len(ps))
        meta["live_mean_before"] = float(arr.mean())
        meta["live_mean_after"] = float(after.mean())
        meta["fitted_at"] = datetime.now().isoformat(timespec="seconds")
        n_changed += 1

    if n_changed == 0:
        print("[calib] no branches changed. Aborting save.")
        return 0

    # 4. Save (with backup).
    if args.dry_run:
        print(f"[calib] --dry-run: would have updated {n_changed} branches "
              f"in {CALIB_PATH}")
        return 0

    backup = _backup_calibrators()
    print(f"[calib] backed up old calibrators to {backup.name}")
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIB_PATH, "wb") as f:
        pickle.dump(cs, f)
    print(f"[calib] wrote new calibrators to {CALIB_PATH}")
    print(f"[calib] {n_changed} branch(es) personalised.")
    print()
    print("The running server hot-reloads calibrators on file mtime change "
          "(see pad.calibrate.apply_calibration), so the new values take "
          "effect on the next /pad/{pid} poll. Watch the PAD DEBUG panel "
          "— moire/patch_depth/deepfake p_attack should drop within ~1 s.")
    print()
    print("If the personalisation overshoots and you want to revert:")
    print(f"  copy {backup.name} -> calibrators.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
