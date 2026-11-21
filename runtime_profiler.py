"""Per-frame timing breakdown CSV writer.

Why this exists
---------------
Paper §5.7 Table 5 is empty (handoff §6.2). The end-to-end ``ms`` field
exists on every frame envelope, but per-stage breakdown
(MediaPipe / DeepFace / rPPG / PAD) is not recorded anywhere persistent.

This module accepts ``record(mode, total_ms, **stages)`` calls from the
WS hot path and flushes batches to ``reports/runtime_profile.csv``. Two
flush triggers:

* Buffered row count >= ``FLUSH_EVERY_ROWS`` (default 200), OR
* Wall-clock time since last flush >= ``FLUSH_EVERY_SECONDS`` (default 30).

Whichever fires first. Buffer is locked, flush happens off the hot path
on the next ``record()`` call (no background thread — keeps it cheap).

CSV schema is dynamic: any keyword passed to ``record()`` becomes a
column. Header is rewritten on first flush. Subsequent flushes append.
"""
from __future__ import annotations

import csv
import os
import threading
import time
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path(__file__).resolve().parent / "reports" / "runtime_profile.csv"

FLUSH_EVERY_ROWS = int(os.environ.get("DODO_RUNTIME_FLUSH_ROWS", "200"))
FLUSH_EVERY_SECONDS = float(os.environ.get("DODO_RUNTIME_FLUSH_SECS", "30"))

_buffer: list[dict[str, Any]] = []
_buffer_lock = threading.Lock()
_columns: list[str] = []
_last_flush = time.time()
_path = _DEFAULT_PATH
_header_written = False


def configure(path: Path | None = None,
              flush_rows: int | None = None,
              flush_seconds: float | None = None) -> None:
    """Optional reconfigure hook. Call before record() if needed."""
    global _path, FLUSH_EVERY_ROWS, FLUSH_EVERY_SECONDS, _header_written
    if path is not None:
        _path = path
        _header_written = False  # new file -> rewrite header on next flush
    if flush_rows is not None:
        FLUSH_EVERY_ROWS = flush_rows
    if flush_seconds is not None:
        FLUSH_EVERY_SECONDS = flush_seconds


def record(mode: str, total_ms: float, **stages: float) -> None:
    """Append one row. Triggers flush if thresholds passed."""
    global _last_flush
    row = {
        "ts": round(time.time(), 3),
        "mode": mode,
        "total_ms": round(float(total_ms), 2),
    }
    for k, v in stages.items():
        if v is None:
            continue
        try:
            row[k] = round(float(v), 2)
        except (TypeError, ValueError):
            row[k] = v
    with _buffer_lock:
        _buffer.append(row)
        for k in row.keys():
            if k not in _columns:
                _columns.append(k)
        should_flush = (
            len(_buffer) >= FLUSH_EVERY_ROWS
            or (time.time() - _last_flush) >= FLUSH_EVERY_SECONDS
        )
    if should_flush:
        flush()


def flush() -> int:
    """Write buffered rows to CSV. Returns count flushed."""
    global _last_flush, _header_written
    with _buffer_lock:
        if not _buffer:
            _last_flush = time.time()
            return 0
        rows = list(_buffer)
        cols = list(_columns)
        _buffer.clear()
    try:
        _path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not _path.exists() or _header_written is False
        if write_header and _path.exists():
            # New columns may have appeared since the header was last
            # written — rewrite the file with the unified header. Cheap
            # because the file size at this granularity is bounded.
            try:
                existing_rows: list[dict[str, str]] = []
                with _path.open("r", newline="", encoding="utf-8") as f:
                    rdr = csv.DictReader(f)
                    existing_rows = list(rdr)
                    if rdr.fieldnames:
                        for fn in rdr.fieldnames:
                            if fn not in cols:
                                cols.append(fn)
                with _path.open("w", newline="", encoding="utf-8") as f:
                    wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                    wr.writeheader()
                    for r in existing_rows:
                        wr.writerow(r)
                    for r in rows:
                        wr.writerow(r)
            except Exception as e:
                print(f"[runtime_profiler] header rewrite failed: {type(e).__name__}: {e}")
                # Fall back to plain append.
                with _path.open("a", newline="", encoding="utf-8") as f:
                    wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                    if write_header:
                        wr.writeheader()
                    for r in rows:
                        wr.writerow(r)
        else:
            with _path.open("a", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                if write_header:
                    wr.writeheader()
                for r in rows:
                    wr.writerow(r)
        _header_written = True
        _last_flush = time.time()
        return len(rows)
    except Exception as e:
        print(f"[runtime_profiler] flush failed: {type(e).__name__}: {e}")
        # Don't lose the rows — push them back so the next flush retries.
        with _buffer_lock:
            _buffer[:0] = rows
        return 0


def buffered_count() -> int:
    with _buffer_lock:
        return len(_buffer)
