"""session_replay.py — v9.5.x session recording + replay.

Records every outbound `t="frame"` envelope to a compact msgpack stream
(one blob per frame), indexed by session id. Replays the stream back
through a mock WebSocket so you can reproduce a glitchy session without
the original hardware.

Recording turns on when `DODO_REPLAY_RECORD=1`. Files land in
`sessions/replay/<session_id>.mpk`.

Replay is exposed via `GET /replay/list` (session ids) and
`GET /replay/<id>/stream` (server-sent events, one JSON frame per chunk).
Consumers can render directly or re-post to `/ws/detect` for a
true-to-life reproduction.
"""
from __future__ import annotations

import os
import pathlib
import time
import threading
from typing import Any, Iterator, Optional

try:
    import msgpack
    _HAVE_MSGPACK = True
except Exception:
    _HAVE_MSGPACK = False

from log_setup import get_logger
from safe_paths import safe_child

log = get_logger(__name__)


_ROOT = pathlib.Path(__file__).resolve().parent / "sessions" / "replay"


def enabled() -> bool:
    return os.environ.get("DODO_REPLAY_RECORD") == "1" and _HAVE_MSGPACK


def session_path(session_id: str) -> pathlib.Path:
    """Validate the session before opening files or starting an HTTP stream."""
    return safe_child(_ROOT, session_id, suffix=".mpk")


class _Recorder:
    """One recorder per connected WebSocket; stays alive for the session."""

    def __init__(self, session_id: str) -> None:
        _ROOT.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.path = session_path(session_id)
        self._fh = open(self.path, "ab")
        self._lock = threading.Lock()
        self.n_frames = 0
        self.started_at = time.time()

    def record(self, envelope: dict) -> None:
        if not _HAVE_MSGPACK:
            return
        try:
            blob = msgpack.packb(envelope, use_bin_type=True, default=str)
            # Prefix with 4-byte big-endian length so replayer can chunk.
            header = len(blob).to_bytes(4, "big")
            with self._lock:
                self._fh.write(header)
                self._fh.write(blob)
                self.n_frames += 1
        except Exception as e:
            log.warning("replay record failed: %s", e)

    def close(self) -> None:
        try:
            with self._lock:
                self._fh.close()
            log.info("replay session %s closed: %d frames, %.1f s, %.1f KB",
                     self.session_id, self.n_frames,
                     time.time() - self.started_at,
                     self.path.stat().st_size / 1024)
        except Exception:
            pass


def new_recorder(session_id: Optional[str] = None) -> Optional[_Recorder]:
    """Create a recorder if replay is enabled, else None."""
    if not enabled():
        return None
    session_id = session_id or time.strftime("%Y%m%d-%H%M%S")
    try:
        return _Recorder(session_id)
    except Exception as e:
        log.warning("could not open recorder: %s", e)
        return None


def list_sessions() -> list[dict[str, Any]]:
    """Enumerate recorded sessions with size + frame estimate."""
    if not _ROOT.is_dir():
        return []
    out = []
    for p in sorted(_ROOT.glob("*.mpk")):
        try:
            size = p.stat().st_size
            out.append({
                "session_id": p.stem,
                "path": str(p),
                "size_bytes": size,
                "mtime": p.stat().st_mtime,
            })
        except Exception:
            pass
    return out


def stream(session_id: str) -> Iterator[dict]:
    """Yield each envelope from a recorded session."""
    path = session_path(session_id)
    if not path.is_file() or not _HAVE_MSGPACK:
        return
    with open(path, "rb") as f:
        while True:
            header = f.read(4)
            if len(header) < 4:
                return
            n = int.from_bytes(header, "big")
            blob = f.read(n)
            if len(blob) < n:
                return
            try:
                yield msgpack.unpackb(blob, raw=False)
            except Exception as e:
                log.warning("replay stream parse error: %s", e)
                return


__all__ = ["enabled", "new_recorder", "list_sessions", "stream", "session_path"]

