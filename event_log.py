"""Rotating structured-JSON event log for DodoVisionAI.

Why this exists
---------------
* The existing ``logger.py`` is a Parquet *data* logger — affect vectors,
  vitals, PAD verdicts. Strict schema, no narrative.
* The existing ``perf.enable_log_mirror()`` mirrors raw stdout/stderr
  into ``logs/dodo_console.log`` — useful but unstructured; grepping for
  a specific subsystem still drowns in radar JSON noise.

This module fills the gap with a tiny, dependency-free structured event
sink:

* One JSON object per line, one file per UTC date
  (``logs/events_2026-04-25.jsonl``).
* ``{ts, ts_iso, level, tag, msg, **ctx}`` — predictable schema you can
  ``jq`` over.
* Auto-rotates on date change; background flush thread coalesces writes
  so log volume from a 30 Hz pipeline doesn't bottleneck on fsync.
* ``tagged(tag)`` returns helpers bound to that subsystem so call sites
  read like ``log_radar.warn("filter rejected outlier", value=212)``.

Designed to be safe to import from anywhere (no circular config deps —
file location is read lazily on first write).
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Public levels — Prometheus-ish, not RFC-5424.
LEVELS = ("debug", "info", "warn", "error", "critical")

_LOG_DIR_ENV = "DODO_EVENT_LOG_DIR"
_LOG_LEVEL_ENV = "DODO_EVENT_LOG_LEVEL"

_DEFAULT_DIR = Path(__file__).resolve().parent / "logs"

_q: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=10_000)
_writer_thread: Optional[threading.Thread] = None
_started = False
_drop_counter = 0
_drop_lock = threading.Lock()


def _level_index(level: str) -> int:
    try:
        return LEVELS.index(level.lower())
    except ValueError:
        return 1  # treat unknown as info


def _min_level() -> int:
    raw = os.environ.get(_LOG_LEVEL_ENV, "info").lower()
    return _level_index(raw)


def _log_dir() -> Path:
    raw = os.environ.get(_LOG_DIR_ENV)
    p = Path(raw).expanduser() if raw else _DEFAULT_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def _log_path_for(ts_seconds: float) -> Path:
    dt = datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
    return _log_dir() / f"events_{dt.strftime('%Y-%m-%d')}.jsonl"


def _writer_loop() -> None:
    """Drains the queue and writes lines, rotating files on date change."""
    current_path: Optional[Path] = None
    fh = None
    try:
        while True:
            item = _q.get()
            if item is None:  # sentinel
                break
            target = _log_path_for(time.time())
            if target != current_path:
                if fh is not None:
                    try:
                        fh.close()
                    except OSError:
                        pass  # nolint:silent-except rotating, drop quietly
                try:
                    fh = target.open("a", encoding="utf-8")
                    current_path = target
                except OSError as e:
                    print(f"[event_log] cannot open {target}: {e}", file=sys.stderr)
                    fh = None
                    current_path = None
                    continue
            try:
                fh.write(item)  # already includes trailing newline
                fh.flush()
            except OSError as e:
                print(f"[event_log] write failed: {e}", file=sys.stderr)
                try:
                    fh.close()
                except OSError:
                    pass  # nolint:silent-except rotating, drop quietly
                fh = None
                current_path = None
    finally:
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass  # nolint:silent-except shutdown best-effort


def _ensure_started() -> None:
    global _writer_thread, _started
    if _started:
        return
    _writer_thread = threading.Thread(
        target=_writer_loop, name="event-log-writer", daemon=True
    )
    _writer_thread.start()
    _started = True


def stop() -> None:
    """Flush + stop the writer thread. Safe to call from atexit."""
    global _started
    if not _started:
        return
    try:
        _q.put_nowait(None)
    except queue.Full:
        pass  # nolint:silent-except sentinel optional
    if _writer_thread is not None:
        _writer_thread.join(timeout=1.0)
    _started = False


def log(level: str, tag: str, msg: str, **ctx: Any) -> None:
    """Emit one structured line. Non-blocking; drops on queue overflow."""
    if _level_index(level) < _min_level():
        return
    _ensure_started()
    now = time.time()
    record: dict[str, Any] = {
        "ts": round(now, 6),
        "ts_iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(timespec="milliseconds"),
        "level": level.lower(),
        "tag": tag,
        "msg": msg,
    }
    if ctx:
        # Only keep JSON-safe values; everything else gets repr()'d.
        safe_ctx: dict[str, Any] = {}
        for k, v in ctx.items():
            try:
                json.dumps(v)
                safe_ctx[k] = v
            except (TypeError, ValueError):
                safe_ctx[k] = repr(v)
        record.update(safe_ctx)
    try:
        line = json.dumps(record, separators=(",", ":")) + "\n"
    except Exception as e:
        line = json.dumps({
            "ts": now, "level": "error", "tag": "event_log",
            "msg": f"serialization failed: {type(e).__name__}: {e}",
            "_orig_tag": tag, "_orig_msg": msg,
        }) + "\n"
    try:
        _q.put_nowait(line)
    except queue.Full:
        global _drop_counter
        with _drop_lock:
            _drop_counter += 1


def dropped_count() -> int:
    with _drop_lock:
        return _drop_counter


class _Tagged:
    __slots__ = ("_tag",)

    def __init__(self, tag: str) -> None:
        self._tag = tag

    def debug(self, msg: str, **ctx: Any) -> None:
        log("debug", self._tag, msg, **ctx)

    def info(self, msg: str, **ctx: Any) -> None:
        log("info", self._tag, msg, **ctx)

    def warn(self, msg: str, **ctx: Any) -> None:
        log("warn", self._tag, msg, **ctx)

    def error(self, msg: str, **ctx: Any) -> None:
        log("error", self._tag, msg, **ctx)

    def critical(self, msg: str, **ctx: Any) -> None:
        log("critical", self._tag, msg, **ctx)

    def exc(self, msg: str, exc: BaseException, **ctx: Any) -> None:
        """Convenience: log an exception with type + repr."""
        ctx.setdefault("exc_type", type(exc).__name__)
        ctx.setdefault("exc_repr", repr(exc))
        log("error", self._tag, msg, **ctx)


def tagged(tag: str) -> _Tagged:
    """Return a tag-bound helper. Cheap; cache at module scope."""
    return _Tagged(tag)


# Register an atexit cleanup so the queue drains on graceful shutdown.
import atexit as _atexit
_atexit.register(stop)
