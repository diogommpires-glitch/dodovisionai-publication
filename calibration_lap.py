"""
DODO VISION AI v8.8 — 90-second Calibration Lap

3-stage emotional baseline lap that drives a subject through:
  1. 30s neutral rest          -> baseline vital signs
  2. 30s positive recall       -> "happy" affective anchor
  3. 30s stress recall         -> "stress" affective anchor

Output: per-person calibration profile saved as JSON, with:
  * per-stage median / std / n / min / max for every captured metric
  * affective contrast (positive vs stress, both vs neutral)
  * integration with PersonState.cold_baseline / warm_baseline

The lap is non-blocking. The orchestrator drops every persons_data list
into feed(); samples are 1Hz-throttled internally and stages auto-advance.
The active lap surfaces the current prompt (which the UI displays to the
subject) and the live remaining time per stage.
"""
from __future__ import annotations
import os
import json
import time
from typing import Any, Dict, List, Optional
from statistics import median, stdev

import config as cfg

LAP_DIR = os.path.join(cfg.CALIBRATION_DIR, "laps")
os.makedirs(LAP_DIR, exist_ok=True)


# ─── Stages ───────────────────────────────────────────────
# Each stage: name, duration in seconds, on-screen prompt for the subject.
STAGES: List[Dict[str, Any]] = [
    {
        "name": "neutral",
        "duration_s": 30.0,
        "prompt": "Rest. Look at the camera and breathe normally for 30 seconds.",
    },
    {
        "name": "positive",
        "duration_s": 30.0,
        "prompt": "Vividly recall a happy memory — a moment that made you laugh out loud.",
    },
    {
        "name": "stress",
        "duration_s": 30.0,
        "prompt": "Vividly recall a stressful situation — a deadline, a confrontation, a loss.",
    },
]
TOTAL_DURATION_S: float = sum(s["duration_s"] for s in STAGES)

# Metrics we sample at ≈1 Hz from each persons_data dict.
SAMPLE_KEYS = (
    "heart_rate_bpm", "hrv_rmssd", "stress_index", "valence", "arousal",
    "blink_rate", "pupil_dilation", "gsr_conductance", "resp_rate",
    "facial_asymmetry", "perclos",
)


class CalibrationLap:
    """One running 90-second lap. Targets a single track_id (or all
    detected faces if `target_track` is None)."""

    def __init__(self, target_track: Optional[int] = None,
                 subject: str = "", examiner: str = ""):
        self.target_track = target_track
        self.subject = subject
        self.examiner = examiner
        self.start_time = time.time()
        self.stage_index = 0
        self.stage_started_at = self.start_time
        self.completed = False
        self.aborted = False
        self.last_save_path: Optional[str] = None
        self.profile: Dict[str, Any] = {}
        # buffers[stage_name][metric_key] = list[float]
        self._buffers: Dict[str, Dict[str, List[float]]] = {
            s["name"]: {k: [] for k in SAMPLE_KEYS} for s in STAGES
        }
        self._last_sample_t = 0.0

    # ─── Public API ──────────────────────────────────────
    def _tick_auto_advance(self, now: float) -> None:
        """v9.8.1 2026-05-12 (14th pass per operator: "LAP is not
        working stays stuck on phase 1/3"). The auto-advance logic
        previously lived ONLY inside `feed()`, so the lap only
        progressed when main.py's fast-path passed in a persons
        list. If the WS hiccuped, the face detector lost the
        subject, or main.py's frame loop wasn't ticking for any
        reason, the stage timer froze. Hoist the advance logic
        into a helper so `status()` (which the UI polls every ~1s
        regardless of frame flow) also drives stage transitions.
        """
        if self.completed or self.aborted:
            return
        # Advance through as many stages as the elapsed time covers.
        # The while-loop catches the edge case where status() hasn't
        # been polled for >duration_s (e.g. the UI was hidden for
        # 60+ s while LAP is in phase 1) -- we don't want to merely
        # bump from stage 0 -> stage 1; we want to fast-forward to
        # the correct stage based on wall-clock time.
        while self.stage_index < len(STAGES):
            cur = STAGES[self.stage_index]
            if (now - self.stage_started_at) < cur["duration_s"]:
                break
            self._lock_stage(cur["name"])
            self.stage_index += 1
            if self.stage_index >= len(STAGES):
                self._finalize()
                return
            self.stage_started_at = now

    def status(self) -> Dict[str, Any]:
        now = time.time()
        # v9.8.1 2026-05-12 (14th pass): always pulse the auto-
        # advance first so a long status-poll-only loop still drives
        # the lap forward.
        self._tick_auto_advance(now)
        if self.completed:
            return {
                "active": False, "completed": True, "aborted": False,
                "stage": "complete", "stage_index": len(STAGES),
                "stage_total": len(STAGES), "stage_remaining_s": 0.0,
                "stage_prompt": "Calibration complete.",
                "total_elapsed_s": round(now - self.start_time, 1),
                "total_duration_s": TOTAL_DURATION_S,
                "samples_per_stage": self._samples_per_stage(),
                "target_track": self.target_track,
                "subject": self.subject,
                "profile": self.profile,
                "saved_path": self.last_save_path,
            }
        if self.aborted:
            return {
                "active": False, "completed": False, "aborted": True,
                "stage": "aborted", "stage_index": self.stage_index,
                "stage_total": len(STAGES), "stage_remaining_s": 0.0,
                "stage_prompt": "Calibration aborted.",
                "total_elapsed_s": round(now - self.start_time, 1),
                "total_duration_s": TOTAL_DURATION_S,
                "samples_per_stage": self._samples_per_stage(),
                "target_track": self.target_track,
                "subject": self.subject,
                "profile": self.profile,
            }
        stage = STAGES[self.stage_index]
        in_stage = now - self.stage_started_at
        return {
            "active": True, "completed": False, "aborted": False,
            "stage": stage["name"],
            "stage_index": self.stage_index,
            "stage_total": len(STAGES),
            "stage_remaining_s": round(max(0.0, stage["duration_s"] - in_stage), 1),
            "stage_prompt": stage["prompt"],
            "total_elapsed_s": round(now - self.start_time, 1),
            "total_duration_s": TOTAL_DURATION_S,
            "samples_per_stage": self._samples_per_stage(),
            "target_track": self.target_track,
            "subject": self.subject,
        }

    def feed(self, persons: List[Dict[str, Any]]) -> bool:
        """Push the current frame's persons list. 1Hz throttled internally,
        auto-advances stages, finalizes on completion."""
        if self.completed or self.aborted:
            return False
        now = time.time()
        # advance stage if elapsed (always check, regardless of throttle)
        # v9.8.1 2026-05-12 (14th pass): factored into _tick_auto_advance
        # so status() and feed() share the same logic. The status()
        # path now also drives the lap forward, so even when the WS
        # is briefly silent the stage counter keeps advancing on
        # wall-clock time.
        self._tick_auto_advance(now)
        if self.completed:
            return True
        # 1Hz sampling throttle
        if now - self._last_sample_t < 1.0:
            return False
        self._last_sample_t = now
        if not persons:
            return False
        targets = persons if self.target_track is None else [
            p for p in persons if p.get("track_id") == self.target_track
        ]
        if not targets:
            return False
        cur_stage = STAGES[self.stage_index]["name"]
        for p in targets:
            for k in SAMPLE_KEYS:
                v = p.get(k)
                if isinstance(v, (int, float)):
                    self._buffers[cur_stage][k].append(float(v))
        return True

    def abort(self) -> Dict[str, Any]:
        # Lock whatever we have so the partial profile isn't lost
        for stage in STAGES[: self.stage_index + 1]:
            if stage["name"] not in self.profile:
                self._lock_stage(stage["name"])
        self.aborted = True
        # v8.7 (Step 9): forensic event log
        try:
            import forensic_log as _fl
            _fl.log_event("calibration_lap_aborted",
                person_id=self.target_track,
                severity=_fl.SEVERITY_WARN,
                payload={"subject": self.subject,
                         "stage_at_abort": STAGES[self.stage_index]["name"]
                                            if self.stage_index < len(STAGES) else "complete",
                         "samples_per_stage": self._samples_per_stage()})
        except Exception:
            pass
        return self.status()

    # ─── Internals ───────────────────────────────────────
    def _samples_per_stage(self) -> Dict[str, int]:
        out = {}
        for s in STAGES:
            n = sum(len(v) for v in self._buffers[s["name"]].values())
            out[s["name"]] = n
        return out

    def _lock_stage(self, stage_name: str) -> None:
        if stage_name in self.profile:
            return
        stats: Dict[str, Dict[str, float]] = {}
        for k, vals in self._buffers[stage_name].items():
            if not vals:
                continue
            stats[k] = {
                "median": round(median(vals), 3),
                "std":    round(stdev(vals), 3) if len(vals) >= 2 else 0.0,
                "n":      len(vals),
                "min":    round(min(vals), 3),
                "max":    round(max(vals), 3),
            }
        self.profile[stage_name] = stats

    def _finalize(self) -> None:
        # Compute affective contrast across stages
        contrast: Dict[str, Dict[str, Optional[float]]] = {}
        pos = self.profile.get("positive", {})
        stress = self.profile.get("stress", {})
        neutral = self.profile.get("neutral", {})
        for k in SAMPLE_KEYS:
            pv = (pos.get(k) or {}).get("median")
            sv = (stress.get(k) or {}).get("median")
            nv = (neutral.get(k) or {}).get("median")
            if pv is None and sv is None:
                continue
            contrast[k] = {
                "positive_minus_neutral": (round(pv - nv, 3)
                                           if (pv is not None and nv is not None) else None),
                "stress_minus_neutral":   (round(sv - nv, 3)
                                           if (sv is not None and nv is not None) else None),
                "stress_minus_positive":  (round(sv - pv, 3)
                                           if (pv is not None and sv is not None) else None),
            }
        self.profile["contrast"] = contrast
        self.profile["meta"] = {
            "subject": self.subject,
            "examiner": self.examiner,
            "target_track": self.target_track,
            "started_at": self.start_time,
            "ended_at": time.time(),
            "duration_s": round(time.time() - self.start_time, 1),
            "stages": [s["name"] for s in STAGES],
        }
        self.completed = True
        self.last_save_path = self._save_profile()
        print(f"[CAL-LAP] Completed (subject={self.subject!r}, track={self.target_track}, "
              f"saved={self.last_save_path})")
        # v8.7 (Step 9): forensic event log
        try:
            import forensic_log as _fl
            _fl.log_event("calibration_lap_completed",
                person_id=self.target_track,
                payload={"subject": self.subject,
                         "saved_path": self.last_save_path,
                         "samples_per_stage": self._samples_per_stage(),
                         "duration_s": round(time.time() - self.start_time, 1)})
        except Exception:
            pass

    def _save_profile(self) -> Optional[str]:
        try:
            tag = self.subject.strip() or (
                f"track_{self.target_track}" if self.target_track is not None else "all"
            )
            # Sanitize filename a bit
            tag = "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)[:48]
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"lap_{tag}_{ts}.json"
            path = os.path.join(LAP_DIR, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.profile, f, indent=2, default=str)
            return path
        except Exception as e:
            print(f"[CAL-LAP] Save failed: {e}")
            return None


# ─── Module-level singleton: only one lap at a time ─────────
_active: Optional[CalibrationLap] = None


def start(target_track: Optional[int] = None,
          subject: str = "", examiner: str = "") -> Dict[str, Any]:
    global _active
    if _active and not (_active.completed or _active.aborted):
        st = _active.status()
        st["error"] = "lap_already_active"
        return st
    _active = CalibrationLap(target_track=target_track, subject=subject, examiner=examiner)
    print(f"[CAL-LAP] Started (subject={subject!r}, target_track={target_track})")
    # v8.7 (Step 9): forensic event log
    try:
        import forensic_log as _fl
        _fl.log_event("calibration_lap_started",
            person_id=target_track,
            payload={"subject": subject, "examiner": examiner,
                     "duration_s": TOTAL_DURATION_S,
                     "stages": [s["name"] for s in STAGES]})
    except Exception:
        pass
    return _active.status()


def feed(persons_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if _active is None:
        return {"active": False, "error": "no_active_lap"}
    _active.feed(persons_list or [])
    return _active.status()


def status() -> Dict[str, Any]:
    if _active is None:
        return {"active": False, "completed": False, "aborted": False}
    return _active.status()


def abort() -> Dict[str, Any]:
    if _active is None:
        return {"active": False, "error": "no_active_lap"}
    return _active.abort()


# Auto-feed hook used by the per-frame orchestrator (main.py).
# Frequent calls are safe — feed() is 1Hz-throttled.
def maybe_feed(persons_list: List[Dict[str, Any]]) -> None:
    if _active is None or _active.completed or _active.aborted:
        return
    try:
        _active.feed(persons_list or [])
    except Exception:
        pass


# ─── Profile management ─────────────────────────────────────
def list_profiles() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(LAP_DIR):
        return out
    for fn in sorted(os.listdir(LAP_DIR), reverse=True):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(LAP_DIR, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("meta") or {}
            out.append({
                "filename": fn,
                "subject": meta.get("subject"),
                "duration_s": meta.get("duration_s"),
                "started_at": meta.get("started_at"),
                "size_kb": round(os.path.getsize(path) / 1024, 1),
            })
        except Exception:
            pass
    return out


def load_profile(filename: str) -> Optional[Dict[str, Any]]:
    if not filename:
        return None
    # Strip any path components for safety
    fname = os.path.basename(filename)
    path = os.path.join(LAP_DIR, fname)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def apply_profile_to_person(person_state, profile: Dict[str, Any]) -> bool:
    """Hot-load a saved lap into a PersonState as the cold baseline.
    The neutral-stage medians become the baseline; positive/stress stages
    are kept as recall anchors for downstream truth/deception scoring."""
    if not profile or "neutral" not in profile or person_state is None:
        return False
    neutral = profile["neutral"]

    def _md(key):
        return (neutral.get(key) or {}).get("median")

    person_state.cold_baseline = {
        "hr":     _md("heart_rate_bpm"),
        "stress": _md("stress_index"),
        "blink":  _md("blink_rate"),
        "pupil":  _md("pupil_dilation"),
        "hrv":    _md("hrv_rmssd"),
        "gsr":    _md("gsr_conductance"),
        "f0":     None,
    }
    person_state._cold_locked = True
    # Mirror to legacy single-baseline fields
    person_state.baseline_hr = person_state.cold_baseline["hr"]
    person_state.baseline_stress = person_state.cold_baseline["stress"]
    person_state.baseline_blink_rate = person_state.cold_baseline["blink"]
    person_state.baseline_pupil = person_state.cold_baseline["pupil"]
    person_state.baseline_hrv = person_state.cold_baseline["hrv"]
    person_state.baseline_gsr = person_state.cold_baseline["gsr"]
    useful = sum(1 for v in person_state.cold_baseline.values() if v)
    person_state.is_baselined = useful >= 3
    # Anchor recall stages on the PersonState for truth-engine consumers
    person_state.recall_positive = profile.get("positive", {})
    person_state.recall_stress   = profile.get("stress", {})
    person_state.calibration_contrast = profile.get("contrast", {})
    return True
