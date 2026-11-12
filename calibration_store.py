"""calibration_store.py — v9.5.x per-person baseline persistence.

Problem: every session warm-start costs ~60 s before HR / emotion / rPPG
baselines stabilise. If we tracked you yesterday at 72 BPM resting, we
ought to recall that the next time we see you.

Solution: on graceful shutdown (and every 60 s) we serialise each
PersonState's baseline slot to `calibration/baselines/<identity>.json`.
On session start, when face_identity attaches an identity to a freshly-
registered PersonState, we replay the cached values.

Schema (pretty-printed JSON, one file per identity):

    {
        "identity": "operator",
        "updated_at": 1776640000.0,
        "hr_mean": 71.3, "hr_std": 4.1,
        "hrv_rmssd_mean": 38.2, "hrv_rmssd_std": 6.4,
        "valence_mean": 0.12, "arousal_mean": -0.05,
        "blink_rate_pm_mean": 16.8,
        "perclos_mean": 0.08,
        "pupil_mean": 0.22,
        "n_samples": 1843,
        "version": 1
    }

The file is tolerant of missing keys — newer baselines extend the
schema; older ones just lack the new fields.
"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile
import time
from typing import Any, Mapping, Optional

from log_setup import get_logger

log = get_logger(__name__)


_ROOT = pathlib.Path(__file__).resolve().parent / "calibration" / "baselines"
_SCHEMA_VERSION = 1
_PERSIST_KEYS = (
    "hr_mean", "hr_std",
    "hrv_rmssd_mean", "hrv_rmssd_std",
    "valence_mean", "arousal_mean",
    "blink_rate_pm_mean", "perclos_mean",
    "pupil_mean",
    # v9.7.4: per-person rest-state facial asymmetry. Without this
    # the deception "facial_asymmetry" indicator falls back to a 0.35
    # population prior on every cold start, which over-fires for
    # subjects with naturally-asymmetric rest expressions. Persisting
    # it lets the calibrated baseline survive across sessions.
    "facial_asymmetry_mean",
    "n_samples",
)


def _path_for(identity: str) -> pathlib.Path:
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in identity)
    if not safe:
        safe = "_anon"
    return _ROOT / f"{safe}.json"


def save(identity: str, baseline: Mapping[str, Any]) -> bool:
    """Persist a baseline dict. Returns True on success."""
    if not identity:
        return False
    _ROOT.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "identity": identity,
        "updated_at": round(time.time(), 3),
        "version": _SCHEMA_VERSION,
    }
    for k in _PERSIST_KEYS:
        if k in baseline and baseline[k] is not None:
            payload[k] = baseline[k]
    path = _path_for(identity)
    try:
        # atomic write: tmpfile + rename
        fd, tmp = tempfile.mkstemp(dir=str(_ROOT), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
        return True
    except Exception as e:
        log.warning("save baseline %r failed: %s", identity, e)
        return False


def load(identity: str) -> Optional[dict[str, Any]]:
    """Return the cached baseline dict for `identity`, or None."""
    if not identity:
        return None
    path = _path_for(identity)
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version", 0) > _SCHEMA_VERSION:
            log.warning("baseline %s has newer schema v%d, ignoring",
                        identity, data.get("version"))
            return None
        return data
    except Exception as e:
        log.warning("load baseline %r failed: %s", identity, e)
        return None


def list_identities() -> list[str]:
    """Enumerate all persisted identities."""
    if not _ROOT.is_dir():
        return []
    return sorted(p.stem for p in _ROOT.glob("*.json"))


def apply_to_person_state(ps: Any, baseline: Mapping[str, Any]) -> int:
    """Hydrate `ps` with the baseline values that match its attributes.
    Returns the number of fields applied. Never raises."""
    if not baseline:
        return 0
    n = 0
    for key in _PERSIST_KEYS:
        if key not in baseline:
            continue
        val = baseline[key]
        if val is None:
            continue
        if hasattr(ps, key):
            try:
                setattr(ps, key, val)
                n += 1
            except Exception:
                pass
        # Backward-compat alias: "hr_mean" might map to "baseline_hr"
        alias = {
            "hr_mean": "baseline_hr",
            "hrv_rmssd_mean": "baseline_hrv",
            "valence_mean": "baseline_valence",
            "arousal_mean": "baseline_arousal",
            "blink_rate_pm_mean": "baseline_blink_rate",
            "pupil_mean": "baseline_pupil",
            # v9.7.4: persisted asymmetry hydrates `baseline_facial_asymmetry`
            "facial_asymmetry_mean": "baseline_facial_asymmetry",
        }.get(key)
        if alias and hasattr(ps, alias):
            try:
                setattr(ps, alias, val)
                n += 1
            except Exception:
                pass
    return n


def extract_from_person_state(ps: Any) -> dict[str, Any]:
    """Pull the baseline-able attributes off `ps` into a dict for save()."""
    out: dict[str, Any] = {}
    for key in _PERSIST_KEYS:
        if hasattr(ps, key):
            val = getattr(ps, key)
            if val is not None:
                out[key] = val
    # Also read legacy attribute names
    fallbacks = {
        "hr_mean": ("baseline_hr", "heart_rate"),
        "valence_mean": ("baseline_valence",),
        "arousal_mean": ("baseline_arousal",),
        # v9.7.4: same alias rules apply on save — pull from the
        # canonical PersonState attribute when the *_mean key is empty.
        "pupil_mean": ("baseline_pupil",),
        "facial_asymmetry_mean": ("baseline_facial_asymmetry",),
        "blink_rate_pm_mean": ("baseline_blink_rate",),
        "hrv_rmssd_mean": ("baseline_hrv",),
    }
    for pref_key, alt_keys in fallbacks.items():
        if out.get(pref_key) is None:
            for alt in alt_keys:
                v = getattr(ps, alt, None)
                if v is not None:
                    out[pref_key] = v
                    break
    return out


__all__ = [
    "save", "load", "list_identities",
    "apply_to_person_state", "extract_from_person_state",
]
