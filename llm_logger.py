"""
DODO VISION AI v7.1 — LLM Training-Pair Logger (Next Step #7)

Append-only sink that llm.py calls after each Groq/Together analysis.
Produces the JSONL that train_qlora.py already consumes:

    {"ts": 1712900000.1,
     "kind": "per_face" | "room",
     "affect": { ... the exact affect vector sent to the LLM ... },
     "prompt": "<system prompt hash or filename>",
     "response": "<assistant output>",
     "backend": "groq-llama-3.3-70b",
     "latency_ms": 812}

File location: calibration/llm_training_pairs.jsonl  (matches train_qlora.py default)
"""
from __future__ import annotations
import json, os, threading, time
from typing import Any, Optional

import config as cfg

_PATH = os.path.join(cfg.CALIBRATION_DIR, "llm_training_pairs.jsonl")
_LOCK = threading.Lock()
_ENABLED = True
# v9.7.5: session-id stamping. The forensic_export QLoRA pipeline
# filters pairs by session_id; without a stamp on each row, it
# matches zero pairs and refuses to export. The stamp is set at
# server startup (`set_session_id()` from main.py) so every pair
# logged within one server lifetime shares the same id.
_SESSION_ID: str = ""
# v9.7.5: in-memory counter of pairs logged in THIS session, for the
# UI badge. Decoupled from disk so the badge updates instantly.
_SESSION_PAIRS_LOGGED: int = 0


def set_session_id(session_id: str) -> None:
    """Set the session id stamped onto every subsequent pair. Call
    this once at server boot."""
    global _SESSION_ID
    _SESSION_ID = str(session_id or "")


def get_session_id() -> str:
    return _SESSION_ID


def get_session_pair_count() -> int:
    """In-memory count of pairs logged so far this session. Cheap;
    suitable for a 1-Hz UI poll."""
    return _SESSION_PAIRS_LOGGED


def get_total_pair_count() -> int:
    """Disk-backed count: total pairs ever logged (current session +
    every prior session, including the legacy migrated ones). Reads
    the pairs file once per call — fine at the 1-Hz cadence the UI
    badge polls."""
    if not os.path.exists(_PATH):
        return 0
    try:
        with open(_PATH, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def set_enabled(on: bool) -> None:
    global _ENABLED
    _ENABLED = bool(on)


def log_pair(
    *,
    kind: str,                     # "per_face" or "room"
    affect: Any,
    response: str,
    prompt_hint: str = "",
    backend: str = "unknown",
    latency_ms: Optional[int] = None,
) -> None:
    """Append one training-example line. Never raises."""
    if not _ENABLED or not response:
        return
    row = {
        "ts": time.time(),
        "kind": kind,
        "affect": _sanitize(affect),
        "prompt": prompt_hint,
        "response": response.strip(),
        "backend": backend,
        # v9.7.5: stamp the session id so forensic_export's
        # filter-by-session can actually match. Without this, every
        # pair had session_id=None and the export was empty.
        "session_id": _SESSION_ID,
    }
    if latency_ms is not None:
        row["latency_ms"] = int(latency_ms)
    try:
        with _LOCK:
            with open(_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
            global _SESSION_PAIRS_LOGGED
            _SESSION_PAIRS_LOGGED += 1
    except Exception as e:
        # Never break the live pipeline because of a log failure.
        print(f"[llm_logger] write failed: {e}")


def _sanitize(obj: Any, depth: int = 0) -> Any:
    """Strip numpy types, clamp deep nesting, drop bulky arrays."""
    if depth > 6:
        return "..."
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize(v, depth + 1) for k, v in list(obj.items())[:64]}
    if isinstance(obj, (list, tuple)):
        if len(obj) > 128:
            return [_sanitize(v, depth + 1) for v in obj[:128]] + ["..."]
        return [_sanitize(v, depth + 1) for v in obj]
    try:
        return float(obj)
    except Exception:
        return str(obj)


def stats() -> dict:
    if not os.path.exists(_PATH):
        return {"rows": 0, "path": _PATH}
    rows = 0
    with open(_PATH, "r", encoding="utf-8") as f:
        for _ in f:
            rows += 1
    return {"rows": rows, "path": _PATH,
            "size_bytes": os.path.getsize(_PATH)}


if __name__ == "__main__":
    print(json.dumps(stats(), indent=2))
