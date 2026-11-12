"""
DODO VISION AI — Forensic Module
Session recording, audit trail, data integrity, and confidence intervals.
Moves the system toward forensic-grade evidence collection.
"""
import os, json, time, hashlib
import config as cfg
from safe_paths import safe_child

SESSIONS_DIR = os.path.join(cfg.CALIBRATION_DIR, "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

_session = None

class ForensicSession:
    """Records all data from a session for later review and audit."""
    def __init__(self, examiner="", subject="", notes=""):
        self.session_id = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        self.start_time = time.time()
        self.examiner = examiner
        self.subject = subject
        self.notes = notes
        self.events = []       # timestamped event log
        self.affect_log = []   # sampled affect vectors (every 1s)
        self.voice_log = []    # voice metrics
        self.truth_tests = []  # truth test results
        self._last_affect_log = 0
        self.session_dir = os.path.join(SESSIONS_DIR, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        self._log_event("session_start", {
            "examiner": examiner, "subject": subject, "notes": notes,
            "system_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(f"[FORENSIC] Session started: {self.session_id}")

    def _log_event(self, event_type, data=None):
        entry = {
            "timestamp": time.time(),
            "elapsed_s": round(time.time() - self.start_time, 3),
            "type": event_type,
            "data": data or {},
        }
        self.events.append(entry)

    def log_affect(self, av):
        """Log affect vector (throttled to 1 per second to save space)."""
        now = time.time()
        if now - self._last_affect_log < 1.0:
            return
        self._last_affect_log = now
        self.affect_log.append({
            "t": round(now - self.start_time, 2),
            "hr": av.get("heart_rate_bpm"),
            "stress": av.get("stress_index"),
            "valence": av.get("valence"),
            "arousal": av.get("arousal"),
            "emotion": av.get("dominant_emotion"),
            "blink": av.get("blink_rate"),
            "pupil": av.get("pupil_dilation"),
            "perclos": av.get("perclos"),
            "gsr": av.get("gsr_conductance"),
        })

    def log_voice(self, metrics):
        """Log voice metrics."""
        if metrics and metrics.get("speech_active"):
            self.voice_log.append({
                "t": round(time.time() - self.start_time, 2),
                "f0": metrics.get("f0_hz"),
                "jitter": metrics.get("jitter"),
                "energy": metrics.get("energy_db"),
                "stress": metrics.get("stress_level"),
            })

    def log_truth_test(self, result):
        """Log a truth test result."""
        self.truth_tests.append({
            "t": round(time.time() - self.start_time, 2),
            "result": result,
        })
        self._log_event("truth_test", {"verdict": result.get("deception_indicators", {}).get("verdict")})

    def add_note(self, note):
        """Add an examiner note."""
        self._log_event("examiner_note", {"note": note})

    def end(self):
        """End session and save all data with integrity hash."""
        self._log_event("session_end", {
            "duration_s": round(time.time() - self.start_time, 1),
            "affect_samples": len(self.affect_log),
            "voice_samples": len(self.voice_log),
            "truth_tests": len(self.truth_tests),
            "events": len(self.events),
        })
        # Build session data
        session_data = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": time.time(),
            "duration_s": round(time.time() - self.start_time, 1),
            "examiner": self.examiner,
            "subject": self.subject,
            "notes": self.notes,
            "events": self.events,
            "affect_log": self.affect_log,
            "voice_log": self.voice_log,
            "truth_tests": self.truth_tests,
        }
        # Save as JSON
        data_path = os.path.join(self.session_dir, "session_data.json")
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, default=str)
        # Compute SHA-256 integrity hash
        with open(data_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        hash_path = os.path.join(self.session_dir, "integrity.sha256")
        with open(hash_path, "w") as f:
            f.write(f"{file_hash}  session_data.json\n")
        print(f"[FORENSIC] Session saved: {self.session_dir}")
        print(f"[FORENSIC] SHA-256: {file_hash}")
        return {
            "session_id": self.session_id,
            "path": self.session_dir,
            "hash": file_hash,
            "duration_s": session_data["duration_s"],
            "affect_samples": len(self.affect_log),
            "voice_samples": len(self.voice_log),
            "truth_tests": len(self.truth_tests),
        }


def compute_confidence_interval(values, confidence=0.95):
    """Compute confidence interval for a list of values.
    Returns: {"mean": x, "ci_low": x, "ci_high": x, "n": n}
    """
    import numpy as np
    if not values or len(values) < 3:
        return None
    arr = np.array([v for v in values if v is not None])
    if len(arr) < 3:
        return None
    n = len(arr)
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    # t-value approximation for 95% CI
    t = 1.96 if n > 30 else 2.045 if n > 20 else 2.228 if n > 10 else 2.776
    ci_low = mean - t * se
    ci_high = mean + t * se
    return {"mean": round(mean, 2), "ci_low": round(ci_low, 2),
            "ci_high": round(ci_high, 2), "n": n, "se": round(se, 3)}


# ── Session management ──
def start_session(examiner="", subject="", notes=""):
    global _session
    if _session:
        _session.end()
    _session = ForensicSession(examiner, subject, notes)
    return {"session_id": _session.session_id}

def get_session():
    return _session

def end_session():
    global _session
    if _session:
        result = _session.end()
        _session = None
        return result
    return None

def add_note(note):
    if _session:
        _session.add_note(note)
        return {"ok": True}
    return {"error": "No active session"}


# ══════════════════════════════════════════════
#  Step 7: Session Replay
# ══════════════════════════════════════════════
def list_sessions():
    """List all recorded sessions with metadata."""
    sessions = []
    for sid in os.listdir(SESSIONS_DIR):
        sess_dir = os.path.join(SESSIONS_DIR, sid)
        if not os.path.isdir(sess_dir):
            continue
        data_file = os.path.join(sess_dir, "session_data.json")
        info = {"session_id": sid, "path": sess_dir}
        if os.path.exists(data_file):
            try:
                with open(data_file, "r") as f:
                    data = json.load(f)
                info["duration_s"] = data.get("duration_s")
                info["affect_samples"] = len(data.get("affect_log", []))
                info["voice_samples"] = len(data.get("voice_log", []))
                info["truth_tests"] = len(data.get("truth_tests", []))
                info["subject"] = data.get("subject", "")
                info["examiner"] = data.get("examiner", "")
            except Exception:
                pass
        # Check integrity
        hash_file = os.path.join(sess_dir, "integrity.sha256")
        info["has_integrity"] = os.path.exists(hash_file)
        sessions.append(info)
    sessions.sort(key=lambda s: s["session_id"], reverse=True)
    return sessions


class SessionPlayer:
    """Load and step through recorded sessions."""
    def __init__(self):
        self.loaded = False
        self.session_id = ""
        self._data = {}
        self._affect_log = []
        self._cursor = 0

    def load(self, session_id):
        try:
            data_file = safe_child(safe_child(SESSIONS_DIR, session_id), "session_data.json")
        except ValueError as e:
            return {"error": str(e)}
        if not os.path.exists(data_file):
            return {"error": f"Session '{session_id}' not found"}
        try:
            with open(data_file, "r") as f:
                self._data = json.load(f)
            self._affect_log = self._data.get("affect_log", [])
            self._cursor = 0
            self.session_id = session_id
            self.loaded = True
            return {
                "session_id": session_id,
                "frames": len(self._affect_log),
                "duration_s": self._data.get("duration_s"),
                "subject": self._data.get("subject"),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_frame(self, index=None):
        if not self.loaded or not self._affect_log:
            return None
        idx = index if index is not None else self._cursor
        if 0 <= idx < len(self._affect_log):
            self._cursor = idx + 1
            frame = self._affect_log[idx]
            frame["_replay_index"] = idx
            frame["_replay_total"] = len(self._affect_log)
            return frame
        return None

    def seek(self, index):
        self._cursor = max(0, min(index, len(self._affect_log) - 1))

    def get_events_at(self, elapsed_s, window=1.0):
        """Get events near a specific timestamp."""
        events = self._data.get("events", [])
        return [e for e in events if abs(e.get("elapsed_s", 0) - elapsed_s) < window]

    def get_voice_at(self, elapsed_s, window=1.0):
        """Get voice data near a specific timestamp."""
        voice = self._data.get("voice_log", [])
        return [v for v in voice if abs(v.get("t", 0) - elapsed_s) < window]

    def get_state(self):
        return {
            "loaded": self.loaded,
            "session_id": self.session_id,
            "cursor": self._cursor,
            "total": len(self._affect_log),
        }


# Global replay player
session_player = SessionPlayer()

