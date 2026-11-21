"""
DODO VISION AI v7.2 — Perf counters + log rotation

Lightweight, thread-safe timing/event counter plus a rotating stdout
mirror for startup banners, radar prints, and other spammy lines.

Usage
-----
    import perf
    with perf.timer("detect"): run_detector()
    perf.inc("frames_processed")
    perf.snapshot()          # dict of {counter: value, timer: {p50,p95,n}}

FastAPI:
    @app.get("/perf") -> perf.snapshot()
"""
from __future__ import annotations
import os, time, threading, contextlib
from collections import deque
from typing import Dict, Deque, Any

_lock = threading.Lock()
_counters: Dict[str, int] = {}
_timers: Dict[str, Deque[float]] = {}
_WINDOW = 500            # keep last N samples per timer


def inc(name: str, by: int = 1) -> None:
    with _lock:
        _counters[name] = _counters.get(name, 0) + by


def observe(name: str, ms: float) -> None:
    with _lock:
        dq = _timers.get(name)
        if dq is None:
            dq = _timers[name] = deque(maxlen=_WINDOW)
        dq.append(float(ms))


@contextlib.contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        observe(name, (time.perf_counter() - t0) * 1000.0)


def _percentile(sorted_vals, p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return float(sorted_vals[k])


def snapshot() -> Dict[str, Any]:
    with _lock:
        counters = dict(_counters)
        timers = {}
        for name, dq in _timers.items():
            if not dq:
                continue
            s = sorted(dq)
            timers[name] = {
                "n": len(s),
                "p50_ms": round(_percentile(s, 50), 3),
                "p95_ms": round(_percentile(s, 95), 3),
                "p99_ms": round(_percentile(s, 99), 3),
                "max_ms": round(s[-1], 3),
            }
    return {"counters": counters, "timers": timers}


def reset() -> None:
    with _lock:
        _counters.clear()
        _timers.clear()


# ── Rotating stdout mirror ──
class RotatingStdoutMirror:
    """Tee stdout to a rolling file (keeps last ~5 MB total across 3 files)."""
    def __init__(self, path: str = "logs/dodo_console.log",
                 max_bytes: int = 2_000_000, backups: int = 3):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.max_bytes = max_bytes
        self.backups = backups
        self._orig = None
        self._fh = None

    def _open(self):
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)

    def _rotate_if_needed(self):
        try:
            if os.path.exists(self.path) and os.path.getsize(self.path) >= self.max_bytes:
                for i in range(self.backups, 0, -1):
                    src = f"{self.path}.{i - 1}" if i > 1 else self.path
                    dst = f"{self.path}.{i}"
                    if os.path.exists(src):
                        if os.path.exists(dst):
                            os.remove(dst)
                        os.rename(src, dst)
                if self._fh:
                    self._fh.close()
                self._open()
        except Exception:
            pass

    def write(self, msg):
        if self._orig:
            try:
                self._orig.write(msg)
            except UnicodeEncodeError:
                # v8.3: Windows cp1252 terminal can't encode Unicode chars
                # like \u2192 (->). Replace with ASCII equivalents.
                safe = msg.encode(self._orig.encoding or "cp1252", errors="replace").decode(self._orig.encoding or "cp1252", errors="replace")
                self._orig.write(safe)
        if self._fh:
            self._rotate_if_needed()
            try:
                self._fh.write(msg)
            except Exception:
                pass

    def flush(self):
        if self._orig:
            self._orig.flush()
        if self._fh:
            try:
                self._fh.flush()
            except Exception:
                pass

    # ── Stream-protocol shims ──
    # Libraries like uvicorn.logging.ColourizedFormatter and rich call
    # sys.stdout.isatty() / .fileno() / .encoding when configuring logging.
    # Delegate to the wrapped original so color detection and file-descriptor
    # lookups keep working after install().
    def isatty(self) -> bool:
        try:
            return bool(self._orig and self._orig.isatty())
        except Exception:
            return False

    def fileno(self) -> int:
        if self._orig is not None:
            return self._orig.fileno()
        raise OSError("RotatingStdoutMirror: no underlying stdout")

    @property
    def encoding(self) -> str:
        return getattr(self._orig, "encoding", "utf-8") if self._orig else "utf-8"

    @property
    def errors(self):
        return getattr(self._orig, "errors", None) if self._orig else None

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def __getattr__(self, name):
        # Last-resort passthrough for any other attribute a logging
        # formatter might probe (e.g. `buffer`, `closed`, `mode`).
        # Only reached when the attribute is NOT set on the instance,
        # so it won't shadow write/flush/isatty/fileno defined above.
        orig = self.__dict__.get("_orig")
        if orig is not None and hasattr(orig, name):
            return getattr(orig, name)
        raise AttributeError(name)

    def install(self):
        import sys
        if self._orig is not None:
            return
        self._open()
        self._orig = sys.stdout
        sys.stdout = self  # type: ignore

    def uninstall(self):
        import sys
        if self._orig is None:
            return
        sys.stdout = self._orig
        self._orig = None
        if self._fh:
            self._fh.close()
            self._fh = None


_mirror: RotatingStdoutMirror | None = None


def enable_log_mirror(path: str = "logs/dodo_console.log") -> None:
    global _mirror
    if _mirror is not None:
        return
    _mirror = RotatingStdoutMirror(path=path)
    _mirror.install()


def attach(app) -> None:
    """Mount GET /perf on a FastAPI app."""
    from fastapi.responses import JSONResponse

    @app.get("/perf")
    def _perf():
        return JSONResponse(snapshot())
