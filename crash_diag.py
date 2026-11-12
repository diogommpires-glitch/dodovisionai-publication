"""crash_diag.py — v9.5.x live-process crash diagnostics.

Extends the bare `faulthandler.enable()` that main.py already installs
with:

  1. A 30-second heartbeat thread that dumps the Python stack of every
     live thread to `logs/heartbeat.log` — if the process freezes we
     still get a post-mortem.
  2. A USR1-like trigger via `POST /ops/dump_stacks` that writes the
     same snapshot on demand.
  3. Ring-buffer of the last 100 WS frames (metadata only, no JPEGs),
     flushed to disk on SIGTERM / crash so we can see what the last
     few frames looked like.

Consumed by main.py at startup:

    import crash_diag
    crash_diag.install(app)
"""
from __future__ import annotations

import collections
import faulthandler
import os
import pathlib
import signal
import sys
import threading
import time
import traceback
from typing import Any, Optional

from log_setup import get_logger

log = get_logger(__name__)

_LOG_DIR = pathlib.Path("logs")
_LOG_DIR.mkdir(exist_ok=True)

_HEARTBEAT_FILE = _LOG_DIR / "heartbeat.log"
_CRASH_LOG     = _LOG_DIR / "crash.log"
_FRAME_RING    = _LOG_DIR / "last_frames.jsonl"

_FRAME_BUFFER: collections.deque = collections.deque(maxlen=100)
_FRAME_LOCK = threading.Lock()

_HEARTBEAT_INTERVAL_S = float(os.environ.get("DODO_HEARTBEAT_S", "30"))


def _dump_all_stacks(label: str) -> str:
    """Return a human-readable dump of every live thread's Python stack."""
    lines = [f"═══ {label} @ {time.strftime('%Y-%m-%d %H:%M:%S')} ═══"]
    current_frames = sys._current_frames()
    for tid, frame in current_frames.items():
        thread_name = "?"
        for t in threading.enumerate():
            if t.ident == tid:
                thread_name = t.name
                break
        lines.append(f"\n[thread {tid} ({thread_name})]")
        lines.extend(traceback.format_stack(frame))
    return "".join(lines)


def _heartbeat_loop() -> None:
    while True:
        try:
            time.sleep(_HEARTBEAT_INTERVAL_S)
            snapshot = _dump_all_stacks("heartbeat")
            # Atomic write so readers never see a partial file
            tmp = _HEARTBEAT_FILE.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(snapshot)
            try:
                os.replace(tmp, _HEARTBEAT_FILE)
            except Exception:
                pass
        except Exception as e:
            log.warning("heartbeat loop error: %s", e)


def record_frame_meta(meta: dict[str, Any]) -> None:
    """Ring-buffer hook — call from the WS handler with the outgoing
    frame's summary (NOT the full envelope: no JPEGs or landmarks)."""
    with _FRAME_LOCK:
        _FRAME_BUFFER.append({**meta, "_ts": time.time()})


def _flush_frame_ring() -> None:
    with _FRAME_LOCK:
        rows = list(_FRAME_BUFFER)
    if not rows:
        return
    import json
    try:
        with open(_FRAME_RING, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
    except Exception as e:
        log.warning("frame ring flush failed: %s", e)


def _term_handler(signum: int, frame) -> None:
    """On SIGTERM / SIGINT, flush both ring + stack dump before exit."""
    snapshot = _dump_all_stacks(f"signal {signum}")
    try:
        with open(_CRASH_LOG, "ab") as f:
            f.write(snapshot.encode("utf-8"))
            f.write(b"\n")
    except Exception:
        pass
    _flush_frame_ring()
    # Chain the default handler so uvicorn's graceful-shutdown path fires.
    signal.default_int_handler(signum, frame)  # type: ignore[misc]


def install(app=None) -> None:
    """Wire up heartbeat, signals, and (if `app` passed) an HTTP route."""
    # Faulthandler — crash-trigger only. The periodic stack dump comes
    # from our own Python heartbeat thread below; mixing the two (native
    # faulthandler.dump_traceback_later + our thread, both writing the
    # same heartbeat file) has been observed to corrupt the file and
    # occasionally co-occur with a segfault during a /selftest probe.
    # Pick ONE source of truth — the Python thread, which we control.
    try:
        faulthandler.enable(file=open(_CRASH_LOG, "ab", buffering=0),
                            all_threads=True)
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception as e:
        log.warning("faulthandler setup failed: %s", e)

    # Python heartbeat thread — dumps to a separate file, atomically.
    threading.Thread(target=_heartbeat_loop, daemon=True,
                     name="heartbeat").start()

    # Signal handlers. SIGBREAK is Windows-only.
    for sig in ("SIGINT", "SIGTERM", "SIGBREAK"):
        s = getattr(signal, sig, None)
        if s is None:
            continue
        try:
            signal.signal(s, _term_handler)
        except Exception:
            pass

    if app is not None:
        @app.post("/ops/dump_stacks", tags=["ops"])
        async def _dump_stacks():
            snap = _dump_all_stacks("on-demand")
            _flush_frame_ring()
            try:
                with open(_CRASH_LOG, "ab") as f:
                    f.write(snap.encode("utf-8"))
            except Exception:
                pass
            return {
                "ok": True,
                "threads": len(threading.enumerate()),
                "last_frame_count": len(_FRAME_BUFFER),
                "crash_log": str(_CRASH_LOG),
            }
    log.info("crash diagnostics installed (heartbeat every %ss)",
             _HEARTBEAT_INTERVAL_S)


__all__ = ["install", "record_frame_meta"]
