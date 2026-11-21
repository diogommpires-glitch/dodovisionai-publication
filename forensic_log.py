"""
DODO VISION AI v8.8 — SQLite Forensic Event Log

A persistent, queryable, append-only event store that runs alongside the
in-memory JSON forensic.py recorder. Where forensic.py snapshots session
state, this module is a *time-series chain of evidence*: every notable
moment (baseline lock, micro-expression, deception spike, calibration
lap completion, oximeter disconnect, etc.) lands here as a row.

Designed to be:
  * Thread-safe (single connection guarded by a lock).
  * Cheap to write (single INSERT per event, payload as JSON blob).
  * Easy to query (indexes on ts, session_id, person_id, event_type).
  * Durable (commit-per-write so a crash never loses an event).

Schema:
  events(
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL,                 -- epoch seconds
    iso_ts      TEXT,                 -- human-readable UTC ISO-8601
    session_id  TEXT,                 -- forensic.ForensicSession id, or 'global'
    person_id   TEXT,                 -- track_id-as-string, or NULL
    identity    TEXT,                 -- recognised face name, or NULL
    event_type  TEXT,                 -- e.g. 'baseline_locked'
    severity    TEXT,                 -- 'info' | 'warn' | 'critical'
    payload     TEXT                  -- JSON-encoded extra fields
  )

This module is intentionally minimal: no migrations, no ORM, no async.
SQLite handles the durability and concurrency we need.
"""
from __future__ import annotations
import os
import json
import time
import sqlite3
import threading
from typing import Any, Dict, List, Optional

import config as cfg

LOG_DIR = os.path.join(cfg.CALIBRATION_DIR, "forensic_db")
os.makedirs(LOG_DIR, exist_ok=True)
DB_PATH = os.path.join(LOG_DIR, "events.sqlite")

SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"
SEVERITY_CRITICAL = "critical"

# Whitelisted event types — keeps the schema disciplined. Unknown types are
# still accepted but a warning is printed; this catches typos in callers.
KNOWN_EVENT_TYPES = {
    "session_start", "session_end",
    "person_first_seen", "person_pruned",
    "baseline_cold_locked", "baseline_warm_drift",
    "calibration_lap_started", "calibration_lap_completed", "calibration_lap_aborted",
    "calibration_profile_loaded",
    "rppg_confirmed", "rppg_quality_drop",
    "oximeter_connected", "oximeter_disconnected",
    "radar_connected", "radar_disconnected",
    "deception_spike", "truth_test_completed",
    "micro_expression",
    "llm_analysis_emitted", "llm_governor_throttled",
    "identity_recognised", "identity_lost",
    "system_warning", "system_error",
    # v8.9: ICS (Intrinsic Cognitive Security) violation event types
    "ics_rsvp_attack", "ics_photic_driving", "ics_info_flood",
    # v8.9: WCAG 2.3.1 spatial/contrast/chromaticity violations
    "ics_wcag_spatial_exceed", "ics_wcag_contrast_exceed", "ics_wcag_red_saturation",
}


class ForensicEventLog:
    """Append-only SQLite event store. One process-wide singleton."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.RLock()
        # SQLite connections are per-thread by default; allow cross-thread use
        # because our writers come from FastAPI worker threads + WS thread.
        self._conn = sqlite3.connect(db_path, check_same_thread=False,
                                     isolation_level=None)  # autocommit
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._ensure_schema()
        print(f"[FORENSIC-DB] event log ready at {db_path}")

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          REAL    NOT NULL,
                    iso_ts      TEXT    NOT NULL,
                    session_id  TEXT,
                    person_id   TEXT,
                    identity    TEXT,
                    event_type  TEXT    NOT NULL,
                    severity    TEXT    NOT NULL,
                    payload     TEXT
                )
            """)
            for stmt in (
                "CREATE INDEX IF NOT EXISTS ix_events_ts ON events(ts)",
                "CREATE INDEX IF NOT EXISTS ix_events_session ON events(session_id)",
                "CREATE INDEX IF NOT EXISTS ix_events_person ON events(person_id)",
                "CREATE INDEX IF NOT EXISTS ix_events_type ON events(event_type)",
                "CREATE INDEX IF NOT EXISTS ix_events_severity ON events(severity)",
            ):
                self._conn.execute(stmt)

    # ─── write API ──────────────────────────────────────────────
    def log(self,
            event_type: str,
            *,
            session_id: Optional[str] = None,
            person_id: Optional[Any] = None,
            identity: Optional[str] = None,
            severity: str = SEVERITY_INFO,
            payload: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """Insert one event. Returns the new row id, or None on failure."""
        if event_type not in KNOWN_EVENT_TYPES:
            print(f"[FORENSIC-DB] WARN: unknown event_type {event_type!r}")
        if severity not in (SEVERITY_INFO, SEVERITY_WARN, SEVERITY_CRITICAL):
            severity = SEVERITY_INFO
        ts = time.time()
        iso_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
        sid = (session_id or "global")[:64]
        pid = None if person_id is None else str(person_id)[:64]
        ident = (identity or None)
        if ident is not None:
            ident = str(ident)[:128]
        try:
            payload_json = json.dumps(payload or {}, default=str)[:8192]
        except Exception:
            payload_json = "{}"
        try:
            with self._lock:
                cur = self._conn.execute(
                    "INSERT INTO events (ts, iso_ts, session_id, person_id, identity, event_type, severity, payload) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (ts, iso_ts, sid, pid, ident, event_type, severity, payload_json),
                )
                return cur.lastrowid
        except Exception as e:
            print(f"[FORENSIC-DB] insert failed for {event_type}: {e}")
            return None

    # ─── query API ──────────────────────────────────────────────
    def recent(self, limit: int = 100,
               session_id: Optional[str] = None,
               person_id: Optional[Any] = None,
               event_type: Optional[str] = None,
               severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Most recent matching events (newest first)."""
        clauses, params = [], []
        if session_id:
            clauses.append("session_id = ?"); params.append(session_id)
        if person_id is not None:
            clauses.append("person_id = ?"); params.append(str(person_id))
        if event_type:
            clauses.append("event_type = ?"); params.append(event_type)
        if severity:
            clauses.append("severity = ?"); params.append(severity)
        sql = "SELECT id, ts, iso_ts, session_id, person_id, identity, event_type, severity, payload FROM events"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(max(1, min(2000, limit))))
        out = []
        try:
            with self._lock:
                rows = self._conn.execute(sql, tuple(params)).fetchall()
            for r in rows:
                try:
                    payload = json.loads(r[8]) if r[8] else {}
                except Exception:
                    payload = {"_raw": r[8]}
                out.append({
                    "id": r[0], "ts": r[1], "iso_ts": r[2],
                    "session_id": r[3], "person_id": r[4], "identity": r[5],
                    "event_type": r[6], "severity": r[7], "payload": payload,
                })
        except Exception as e:
            print(f"[FORENSIC-DB] query failed: {e}")
        return out

    def stats(self) -> Dict[str, Any]:
        """Counts per type / severity for a quick overview."""
        try:
            with self._lock:
                total = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                by_type = self._conn.execute(
                    "SELECT event_type, COUNT(*) FROM events GROUP BY event_type ORDER BY 2 DESC"
                ).fetchall()
                by_sev = self._conn.execute(
                    "SELECT severity, COUNT(*) FROM events GROUP BY severity"
                ).fetchall()
                first = self._conn.execute("SELECT MIN(ts) FROM events").fetchone()[0]
                last = self._conn.execute("SELECT MAX(ts) FROM events").fetchone()[0]
                sessions = self._conn.execute(
                    "SELECT COUNT(DISTINCT session_id) FROM events"
                ).fetchone()[0]
            return {
                "db_path": self.db_path,
                "total_events": int(total or 0),
                "distinct_sessions": int(sessions or 0),
                "first_ts": first, "last_ts": last,
                "by_event_type": [{"event_type": k, "count": v} for k, v in by_type],
                "by_severity": [{"severity": k, "count": v} for k, v in by_sev],
            }
        except Exception as e:
            return {"error": str(e)}

    def purge_older_than(self, days: int) -> int:
        """Delete events older than `days` days. Returns rows removed."""
        cutoff = time.time() - max(0, int(days)) * 86400
        try:
            with self._lock:
                cur = self._conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
                self._conn.execute("VACUUM")
                return cur.rowcount or 0
        except Exception as e:
            print(f"[FORENSIC-DB] purge failed: {e}")
            return 0

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.close()
        except Exception:
            pass


# ─── Module-level singleton ─────────────────────────────────────
_log: Optional[ForensicEventLog] = None
_init_lock = threading.Lock()


def get_log() -> ForensicEventLog:
    global _log
    if _log is None:
        with _init_lock:
            if _log is None:
                _log = ForensicEventLog()
    return _log


# ─── Convenience wrappers used throughout the codebase ──────────
def log_event(event_type: str, **kwargs) -> Optional[int]:
    """Top-level helper. Most callers should use this rather than .log()."""
    try:
        return get_log().log(event_type, **kwargs)
    except Exception as e:
        # Logging must never crash the caller. Print and move on.
        print(f"[FORENSIC-DB] log_event({event_type!r}) failed: {e}")
        return None


def recent(limit: int = 100, **filters) -> List[Dict[str, Any]]:
    return get_log().recent(limit=limit, **filters)


def stats() -> Dict[str, Any]:
    return get_log().stats()


def purge_older_than(days: int) -> int:
    return get_log().purge_older_than(days)
