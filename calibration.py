"""
DODO VISION AI — Calibration & Training
Train/calibrate rPPG HR, SpO2, and BPM against oximeter ground truth.
Collects paired samples and computes correction factors.
Also trains the local LLM by logging affect+response pairs for future fine-tuning.
"""
import os, json, time
import numpy as np
import config as cfg

CALIBRATION_FILE = os.path.join(cfg.CALIBRATION_DIR, "calibration.json")
RPPG_PAIRS_FILE = os.path.join(cfg.CALIBRATION_DIR, "rppg_pairs.json")
LLM_TRAINING_FILE = os.path.join(cfg.CALIBRATION_DIR, "llm_training.jsonl")
# v9.7.6 — separate radar pair store. Schema:
#   [{"radar_a_hr": ..., "radar_b_hr": ..., "oxi_hr": ..., "timestamp": ...}]
# Either radar field may be null when only one radar is connected. We
# previously co-mingled radar_hr inside rppg_pairs.json which conflated
# camera-rPPG with radar; that was fine when there was one radar but
# fails the moment radar A and B have different bias profiles (e.g.
# A reads 64 BPM and B reads 118 BPM for the same subject).
RADAR_PAIRS_FILE = os.path.join(cfg.CALIBRATION_DIR, "radar_pairs.json")

# ── Current calibration state ──
_calibration = {
    "rppg_hr_offset": 0.0,      # additive correction: corrected = raw + offset
    "rppg_hr_scale": 1.0,       # multiplicative: corrected = raw * scale + offset
    "spo2_offset": 0.0,
    "spo2_scale": 1.0,
    "samples_collected": 0,
    "last_calibrated": None,
}

# ── Paired samples buffer ──
_rppg_pairs = []   # [{"camera_hr": x, "oxi_hr": y, "radar_hr": z, "timestamp": t}, ...]
_radar_pairs = []  # [{"radar_hr": x, "oxi_hr": y, "timestamp": t}, ...]
# v9.7.6 — automatic collection on by default. The legacy CAL rPPG /
# STOP CAL UI buttons turned this on/off, but the user kept ending
# sessions with _collecting=False (which then persisted across UI
# restarts even though the value itself doesn't persist — the issue
# was nobody clicking CAL rPPG fresh each session). Default-on means
# pairs accumulate continuously whenever rPPG and oximeter both have
# valid readings. Set DODO_AUTOCALIBRATE=0 in env to disable.
_collecting = (os.environ.get("DODO_AUTOCALIBRATE", "1") not in ("0", "false", "False"))
_radar_calibration = {"scale": 1.0, "offset": 0.0, "samples": 0}

def _load():
    global _calibration
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, "r") as f:
                _calibration = json.load(f)
            print(f"[OK] Calibration loaded ({_calibration['samples_collected']} samples)")
        except Exception:
            pass

def _save():
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(_calibration, f, indent=2)

def _save_pairs():
    with open(RPPG_PAIRS_FILE, "w") as f:
        json.dump(_rppg_pairs, f, indent=2)

# v9.7.6 — persist radar pairs to disk so they survive restarts.
# Without this, every server boot started with `_radar_pairs = []` and
# the user's accumulated radar/oxi pairs were lost.
def _save_radar_pairs():
    try:
        with open(RADAR_PAIRS_FILE, "w") as f:
            json.dump(_radar_pairs, f, indent=2)
    except Exception as e:
        print(f"[CAL] Failed to save radar pairs: {e}")

def _load_radar_pairs():
    """Load accumulated radar pairs from disk on module import."""
    global _radar_pairs
    if not os.path.exists(RADAR_PAIRS_FILE):
        return
    try:
        with open(RADAR_PAIRS_FILE, "r") as f:
            loaded = json.load(f)
        if isinstance(loaded, list):
            _radar_pairs = loaded
            print(f"[CAL] Loaded {len(_radar_pairs)} radar pairs from disk")
    except Exception as e:
        print(f"[CAL] Failed to load radar pairs: {e}")

def start_collection():
    """Legacy alias — auto-collect is now default-on. Kept for callers
    that still POST /calibrate/start. Just makes sure the flag is on
    and returns a status."""
    global _collecting
    _collecting = True
    return {"status": "collecting",
            "msg": f"Auto-collection ON (n={len(_rppg_pairs)} pairs).",
            "collecting": True}

def stop_collection():
    """Legacy. Now ALSO compatible with the new pause flow — flips
    _collecting off but does NOT auto-train. The user explicitly hits
    TRAIN rPPG / TRAIN Radar A / B from the UI to fit. We keep the
    return shape of the legacy fit-on-stop behavior so old callers
    still see {status, msg}."""
    global _collecting
    _collecting = False
    return {"status": "ok",
            "msg": f"Collection paused (n={len(_rppg_pairs)} pairs accumulated). "
                   "Click TRAIN rPPG / Radar A / Radar B to fit.",
            "collecting": False}

# v9.7.6 — explicit toggle for the new auto-collect flow.
def pause_collection():
    """Pause auto-collection (e.g. before deliberately moving / talking
    so noisy samples don't get into the training set). Idempotent."""
    global _collecting
    _collecting = False
    return {"status": "paused", "collecting": False,
            "n_rppg": len(_rppg_pairs), "n_radar": len(_radar_pairs)}

def resume_collection():
    """Resume auto-collection."""
    global _collecting
    _collecting = True
    return {"status": "collecting", "collecting": True,
            "n_rppg": len(_rppg_pairs), "n_radar": len(_radar_pairs)}

def is_collecting():
    return _collecting

def add_sample(camera_hr, oxi_hr, radar_hr=None, camera_spo2=None, oxi_spo2=None,
               radar_a_hr=None, radar_b_hr=None):
    """Add a paired sample during collection.

    v9.7.6: split single `radar_hr` into `radar_a_hr` + `radar_b_hr`
    for dual-radar setups. The legacy `radar_hr` arg is treated as
    radar A so existing call sites keep working without changes.
    Either radar field may be None when only one radar is connected.
    """
    if not _collecting:
        return
    if camera_hr is None or oxi_hr is None:
        return
    # Legacy single-radar arg → radar A
    if radar_a_hr is None and radar_hr is not None:
        radar_a_hr = radar_hr
    sample = {
        "camera_hr": float(camera_hr),
        "oxi_hr": float(oxi_hr),
        "radar_hr": float(radar_a_hr) if radar_a_hr else None,  # legacy field
        "camera_spo2": float(camera_spo2) if camera_spo2 else None,
        "oxi_spo2": float(oxi_spo2) if oxi_spo2 else None,
        "timestamp": time.time(),
    }
    _rppg_pairs.append(sample)
    # v9.7.6: store radar pairs in their own list with both radars'
    # readings so we can train per-radar corrections later.
    if (radar_a_hr or radar_b_hr) and oxi_hr:
        _radar_pairs.append({
            "radar_a_hr": float(radar_a_hr) if radar_a_hr else None,
            "radar_b_hr": float(radar_b_hr) if radar_b_hr else None,
            "oxi_hr": float(oxi_hr),
            "timestamp": time.time(),
        })
        _save_radar_pairs()
    _save_pairs()

def _compute_calibration():
    """Compute linear correction: oxi = scale * camera + offset (least squares)."""
    cam = np.array([s["camera_hr"] for s in _rppg_pairs if s["camera_hr"] and s["oxi_hr"]])
    oxi = np.array([s["oxi_hr"] for s in _rppg_pairs if s["camera_hr"] and s["oxi_hr"]])
    if len(cam) < 5:
        return {"status": "error", "msg": "Not enough valid HR pairs"}
    # Linear regression: oxi = scale * cam + offset
    A = np.vstack([cam, np.ones(len(cam))]).T
    result = np.linalg.lstsq(A, oxi, rcond=None)
    scale, offset = result[0]
    residuals = oxi - (scale * cam + offset)
    mae = round(float(np.mean(np.abs(residuals))), 2)
    _calibration["rppg_hr_scale"] = round(float(scale), 4)
    _calibration["rppg_hr_offset"] = round(float(offset), 2)
    _calibration["samples_collected"] = len(cam)
    _calibration["last_calibrated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _calibration["mae_bpm"] = mae
    # Step 5: Compute R-squared
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((oxi - np.mean(oxi)) ** 2))
    r_squared = round(1 - ss_res / max(ss_tot, 1e-9), 4)
    _calibration["r_squared"] = r_squared
    # Step 5: Save regression file for rppg.py auto-correction
    reg_file = getattr(cfg, 'CALIBRATION_REGRESSION_FILE', os.path.join(cfg.CALIBRATION_DIR, "rppg_correction.json"))
    try:
        with open(reg_file, "w") as rf:
            json.dump({"slope": float(scale), "intercept": float(offset),
                        "r_squared": r_squared, "n_pairs": len(cam),
                        "mae_bpm": mae, "timestamp": time.time()}, rf, indent=2)
        print(f"[CAL] Regression saved: y={scale:.4f}*x+{offset:.2f} R²={r_squared}")
    except Exception as e:
        print(f"[CAL] Failed to save regression: {e}")
    # SpO2 calibration if available
    cam_spo2 = [s["camera_spo2"] for s in _rppg_pairs if s.get("camera_spo2") and s.get("oxi_spo2")]
    oxi_spo2 = [s["oxi_spo2"] for s in _rppg_pairs if s.get("camera_spo2") and s.get("oxi_spo2")]
    if len(cam_spo2) >= 5:
        A2 = np.vstack([np.array(cam_spo2), np.ones(len(cam_spo2))]).T
        r2 = np.linalg.lstsq(A2, np.array(oxi_spo2), rcond=None)
        _calibration["spo2_scale"] = round(float(r2[0][0]), 4)
        _calibration["spo2_offset"] = round(float(r2[0][1]), 2)
    _save()
    return {
        "status": "ok",
        "samples": len(cam),
        "hr_scale": _calibration["rppg_hr_scale"],
        "hr_offset": _calibration["rppg_hr_offset"],
        "mae_bpm": mae,
        "msg": f"Calibrated on {len(cam)} samples. MAE: {mae} BPM. Correction: HR = {scale:.3f} * camera + {offset:.1f}",
    }

def correct_hr(raw_hr):
    """Apply calibration correction to camera HR."""
    if raw_hr is None:
        return None
    return round(raw_hr * _calibration["rppg_hr_scale"] + _calibration["rppg_hr_offset"], 1)

def correct_radar_hr(raw_hr):
    """Apply calibration correction to radar HR."""
    if raw_hr is None or _radar_calibration["samples"] < 5:
        return raw_hr
    return round(raw_hr * _radar_calibration["scale"] + _radar_calibration["offset"], 1)

def calibrate_radar():
    """Compute radar HR correction from collected radar-oxi pairs."""
    if len(_radar_pairs) < 5:
        return {"status": "error", "msg": f"Need 5+ radar/oxi pairs, got {len(_radar_pairs)}"}
    radar = np.array([p["radar_hr"] for p in _radar_pairs])
    oxi = np.array([p["oxi_hr"] for p in _radar_pairs])
    A = np.vstack([radar, np.ones(len(radar))]).T
    result = np.linalg.lstsq(A, oxi, rcond=None)
    scale, offset = result[0]
    _radar_calibration["scale"] = round(float(scale), 4)
    _radar_calibration["offset"] = round(float(offset), 2)
    _radar_calibration["samples"] = len(radar)
    mae = round(float(np.mean(np.abs(oxi - (scale * radar + offset)))), 2)
    print(f"[CAL] Radar: y={scale:.4f}*x+{offset:.2f} MAE={mae} ({len(radar)} pairs)")
    return {"status": "ok", "scale": _radar_calibration["scale"], "offset": _radar_calibration["offset"], "mae": mae, "pairs": len(radar)}

def correct_spo2(raw_spo2):
    """Apply calibration correction to camera SpO2."""
    if raw_spo2 is None:
        return None
    return round(raw_spo2 * _calibration["spo2_scale"] + _calibration["spo2_offset"], 1)

def get_status():
    return {
        "collecting": _collecting,
        "pairs_collected": len(_rppg_pairs),
        "n_rppg":  len(_rppg_pairs),
        "n_radar": len(_radar_pairs),
        "calibration": _calibration,
        "auto_collect_default": True,   # v9.7.6
    }

def recompute_from_disk():
    """v9.7.6 — Train rPPG correction from the accumulated pairs file.

    The live pipeline appends to `_rppg_pairs` only while `_collecting`
    is True (CAL rPPG button held down). Pairs persist to
    `calibration/rppg_pairs.json` via _save_pairs(), but on a fresh
    server boot the in-memory list starts empty — so calling
    `_compute_calibration()` directly fits on 0 pairs and refuses.

    This loads everything from disk first, then runs the same linear
    regression as `stop_collection()`. Idempotent — safe to call any
    time. Returns the same {status, samples, hr_scale, hr_offset,
    mae_bpm, msg} payload so the existing UI handler works unchanged.
    """
    global _rppg_pairs
    if not os.path.exists(RPPG_PAIRS_FILE):
        return {"status": "error",
                "msg": f"No pairs file at {RPPG_PAIRS_FILE} — collect some first via CAL rPPG"}
    try:
        with open(RPPG_PAIRS_FILE, "r") as f:
            loaded = json.load(f)
        if not isinstance(loaded, list):
            return {"status": "error", "msg": "rppg_pairs.json is not a list"}
        # Replace in-memory buffer with what's on disk. We do this
        # rather than `extend` so re-running the trainer doesn't
        # double-count samples that are already in memory + on disk.
        _rppg_pairs = loaded
    except Exception as e:
        return {"status": "error",
                "msg": f"Could not parse {RPPG_PAIRS_FILE}: {type(e).__name__}: {e}"}
    if len(_rppg_pairs) < 5:
        return {"status": "error",
                "msg": f"Only {len(_rppg_pairs)} pairs on disk; need 5+"}

    # v9.7.6 — quality-gated training. The naive _compute_calibration
    # silently saves whatever LSQ produces, which is dangerous: if your
    # pairs were collected with the subject at rest (oximeter HR
    # roughly constant), the regression has no variation to learn from
    # and produces a near-zero slope + huge intercept — effectively
    # clamping all camera-HR readings to a constant. Better to refuse,
    # explain why, and either fall back to a mean-offset correction or
    # leave the identity transform in place.
    cam = np.array([s["camera_hr"] for s in _rppg_pairs
                    if s.get("camera_hr") and s.get("oxi_hr")])
    oxi = np.array([s["oxi_hr"] for s in _rppg_pairs
                    if s.get("camera_hr") and s.get("oxi_hr")])
    if len(cam) < 30:
        return {"status": "error",
                "msg": f"Only {len(cam)} valid HR pairs; need 30+ for stable fit"}

    # Drop wild outliers — camera_hr that differs from oxi by > 15 BPM
    # is almost certainly an rPPG lock-loss, not real bias to learn.
    diff = cam - oxi
    mask = np.abs(diff) < 15
    cam_f, oxi_f = cam[mask], oxi[mask]
    if len(cam_f) < 30:
        return {"status": "error",
                "msg": (f"After dropping outliers (|cam-oxi|>15) only "
                        f"{len(cam_f)} pairs remain; need 30+. Your rPPG "
                        f"lock-quality during collection was poor.")}

    # Linear regression on the cleaned set
    A = np.vstack([cam_f, np.ones(len(cam_f))]).T
    slope, intercept = np.linalg.lstsq(A, oxi_f, rcond=None)[0]
    residuals = oxi_f - (slope * cam_f + intercept)
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((oxi_f - np.mean(oxi_f)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)

    oxi_sd  = float(np.std(oxi_f))
    bias    = float(np.mean(diff[mask]))   # camera - oxi mean offset

    # Decision tree:
    #   1. R² >= 0.30  → save full slope+intercept (good fit)
    #   2. R² < 0.30 but oxi has variation (sd > 3) → fall back to
    #      mean-offset only: slope=1, intercept=-bias. Same as
    #      "subtract the average bias", which is the most you can
    #      honestly extract from poorly-correlated data.
    #   3. R² < 0.30 AND oxi sd < 3 → ground truth is too static for
    #      ANY rppg_hr→oxi_hr mapping to be meaningful. Save identity
    #      (slope=1, intercept=0) and warn.
    out = {"n_total": int(len(cam)), "n_used": int(len(cam_f)),
           "oxi_mean": round(float(np.mean(oxi_f)), 2),
           "oxi_sd":   round(oxi_sd, 2),
           "bias_bpm": round(bias, 2),
           "mae_bpm":  round(mae, 2),
           "r_squared": round(r2, 4)}

    reg_file = getattr(cfg, "CALIBRATION_REGRESSION_FILE",
                       os.path.join(cfg.CALIBRATION_DIR, "rppg_correction.json"))
    if r2 >= 0.30:
        kind = "full"
        save_slope, save_intercept = float(slope), float(intercept)
        msg = (f"GOOD FIT — saved slope={slope:.4f} intercept={intercept:.2f} "
               f"R²={r2:.3f} on {len(cam_f)} pairs.")
    elif oxi_sd > 3.0:
        kind = "mean_offset"
        save_slope, save_intercept = 1.0, -bias
        msg = (f"POOR FIT (R²={r2:.3f}) — fell back to MEAN-OFFSET correction: "
               f"camera_hr - {bias:.1f}. To learn a slope, collect pairs while "
               f"your HR varies (e.g. some jumping jacks then rest, so oxi_hr "
               f"spans 60-130 BPM).")
    else:
        kind = "identity"
        save_slope, save_intercept = 1.0, 0.0
        msg = (f"REFUSING TO SAVE — oxi_hr was nearly constant (sd={oxi_sd:.1f} "
               f"BPM, mean={float(np.mean(oxi_f)):.1f}). Your collected "
               f"pairs have no HR variation in the ground truth, so no "
               f"meaningful correction can be learned. Identity transform "
               f"kept. Re-collect during HR variation.")
    out["kind"] = kind
    out["slope"] = save_slope
    out["intercept"] = save_intercept
    try:
        with open(reg_file, "w") as rf:
            json.dump({
                "slope": save_slope, "intercept": save_intercept,
                "r_squared": round(r2, 4),
                "n_pairs": len(cam_f), "n_total": len(cam),
                "mae_bpm": round(mae, 2),
                "bias_bpm": round(bias, 2),
                "oxi_sd": round(oxi_sd, 2),
                "kind": kind,
                "timestamp": time.time(),
            }, rf, indent=2)
        # ASCII-only — Windows cmd defaults to cp1252 and chokes on
        # arrow / superscript chars even though the file write succeeds.
        print(f"[CAL] {kind} -> y={save_slope:.4f}*x+{save_intercept:.2f}  "
              f"R^2={r2:.3f}  MAE={mae:.2f}  ({len(cam_f)}/{len(cam)} pairs)")
    except Exception as e:
        return {"status": "error", "msg": f"Could not write {reg_file}: {e}"}

    # Also update the legacy _calibration dict so /calibrate/status
    # reflects the new fit.
    _calibration["rppg_hr_scale"]  = round(save_slope, 4)
    _calibration["rppg_hr_offset"] = round(save_intercept, 2)
    _calibration["samples_collected"] = len(cam_f)
    _calibration["mae_bpm"] = round(mae, 2)
    _calibration["r_squared"] = round(r2, 4)
    _calibration["last_calibrated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save()

    out["status"] = "ok"
    out["msg"] = msg
    return out


def recompute_radar_from_disk(which: str = "a"):
    """v9.7.6 — Train per-radar correction from accumulated radar pairs.

    Mirror of `recompute_from_disk` but for one of the two radars.
    `which` is 'a' or 'b'. Reads `radar_pairs.json`, filters to entries
    that have a value for the requested radar, runs the same quality-
    gated regression as rPPG (drop |radar-oxi|>15 outliers, refuse to
    save if R² < 0.30 unless oxi has variation, etc.), then writes the
    correction to `calibration/radar_<which>_correction.json`.

    The radar.py module auto-loads these files at boot via its own
    `_load_calibration()` mirror of the rppg.py pattern.
    """
    global _radar_pairs
    which = which.lower()
    if which not in ("a", "b"):
        return {"status": "error", "msg": f"which must be 'a' or 'b', got {which!r}"}
    if not os.path.exists(RADAR_PAIRS_FILE):
        return {"status": "error",
                "msg": f"No radar pairs file at {RADAR_PAIRS_FILE} — collect via CAL rPPG first"}
    try:
        with open(RADAR_PAIRS_FILE, "r") as f:
            loaded = json.load(f)
        if not isinstance(loaded, list):
            return {"status": "error", "msg": "radar_pairs.json is not a list"}
        _radar_pairs = loaded
    except Exception as e:
        return {"status": "error",
                "msg": f"Could not parse {RADAR_PAIRS_FILE}: {type(e).__name__}: {e}"}

    field = f"radar_{which}_hr"
    rad = np.array([p[field] for p in _radar_pairs
                    if p.get(field) is not None and p.get("oxi_hr") is not None])
    oxi = np.array([p["oxi_hr"] for p in _radar_pairs
                    if p.get(field) is not None and p.get("oxi_hr") is not None])
    if len(rad) < 30:
        return {"status": "error",
                "msg": (f"Only {len(rad)} valid pairs for radar {which.upper()}; "
                        f"need 30+. Make sure radar {which.upper()} was connected "
                        f"during CAL rPPG collection.")}

    # Outlier filter — radars can briefly latch onto a wall reflection
    # or off-target body part, producing momentary readings 60-80 BPM
    # away from truth. We drop |radar-oxi|>20 (looser than rPPG's 15
    # because radar has higher legitimate range and the wild reading
    # we want to filter out is more like 50+ BPM off, not 16-20).
    diff = rad - oxi
    mask = np.abs(diff) < 20
    rad_f, oxi_f = rad[mask], oxi[mask]
    if len(rad_f) < 30:
        return {"status": "error",
                "msg": (f"After dropping outliers (|radar-oxi|>20) only "
                        f"{len(rad_f)} pairs remain for radar {which.upper()}; "
                        f"need 30+. Radar {which.upper()} lock quality was poor "
                        f"during collection — check sensor mounting / aim.")}

    A = np.vstack([rad_f, np.ones(len(rad_f))]).T
    slope, intercept = np.linalg.lstsq(A, oxi_f, rcond=None)[0]
    residuals = oxi_f - (slope * rad_f + intercept)
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((oxi_f - np.mean(oxi_f)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    oxi_sd  = float(np.std(oxi_f))
    bias    = float(np.mean(diff[mask]))   # radar - oxi mean offset

    out = {"which": which, "n_total": int(len(rad)), "n_used": int(len(rad_f)),
           "oxi_mean": round(float(np.mean(oxi_f)), 2),
           "oxi_sd":   round(oxi_sd, 2),
           "bias_bpm": round(bias, 2),
           "mae_bpm":  round(mae, 2),
           "r_squared": round(r2, 4)}

    reg_file = os.path.join(cfg.CALIBRATION_DIR, f"radar_{which}_correction.json")
    if r2 >= 0.30:
        kind = "full"
        save_slope, save_intercept = float(slope), float(intercept)
        msg = (f"Radar {which.upper()}: GOOD FIT — saved slope={slope:.4f} "
               f"intercept={intercept:.2f} R^2={r2:.3f} on {len(rad_f)} pairs.")
    elif oxi_sd > 3.0:
        kind = "mean_offset"
        save_slope, save_intercept = 1.0, -bias
        msg = (f"Radar {which.upper()}: POOR FIT (R^2={r2:.3f}) — fell back to "
               f"MEAN-OFFSET: radar_hr - {bias:.1f}. Collect pairs across HR "
               f"variation (rest+exercise) to learn a real slope.")
    else:
        kind = "identity"
        save_slope, save_intercept = 1.0, 0.0
        msg = (f"Radar {which.upper()}: REFUSING TO SAVE — oxi_hr nearly constant "
               f"(sd={oxi_sd:.1f}). No correction can be learned without HR "
               f"variation. Identity transform kept.")
    out["kind"] = kind
    out["slope"] = save_slope
    out["intercept"] = save_intercept
    try:
        with open(reg_file, "w") as rf:
            json.dump({
                "slope": save_slope, "intercept": save_intercept,
                "r_squared": round(r2, 4),
                "n_pairs": len(rad_f), "n_total": len(rad),
                "mae_bpm": round(mae, 2),
                "bias_bpm": round(bias, 2),
                "oxi_sd": round(oxi_sd, 2),
                "kind": kind, "which": which,
                "timestamp": time.time(),
            }, rf, indent=2)
        print(f"[CAL] Radar {which.upper()} {kind} -> y={save_slope:.4f}*x"
              f"+{save_intercept:.2f}  R^2={r2:.3f}  MAE={mae:.2f}  "
              f"({len(rad_f)}/{len(rad)} pairs)")
    except Exception as e:
        return {"status": "error", "msg": f"Could not write {reg_file}: {e}"}

    out["status"] = "ok"
    out["msg"] = msg
    return out


def reset():
    """Clear all calibration data."""
    global _rppg_pairs, _calibration
    _rppg_pairs = []
    _calibration = {
        "rppg_hr_offset": 0.0, "rppg_hr_scale": 1.0,
        "spo2_offset": 0.0, "spo2_scale": 1.0,
        "samples_collected": 0, "last_calibrated": None,
    }
    _save()
    return {"status": "reset"}

# ── LLM Training Data Collection ──
def log_llm_pair(affect_vector, llm_response):
    """Log affect vector + LLM response pair for future fine-tuning."""
    if not llm_response or len(llm_response) < 10:
        return
    try:
        pair = {
            "timestamp": time.time(),
            "affect_vector": affect_vector,
            "response": llm_response,
        }
        with open(LLM_TRAINING_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(pair) + "\n")
    except Exception:
        pass

def get_llm_training_count():
    """Count how many LLM training pairs we have."""
    if not os.path.exists(LLM_TRAINING_FILE):
        return 0
    try:
        with open(LLM_TRAINING_FILE, "r") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

# Load on import
_load()
# v9.7.6 — load accumulated radar pairs from disk so trainings can run
# on full history rather than just this-session.
_load_radar_pairs()
