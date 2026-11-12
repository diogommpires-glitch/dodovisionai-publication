"""COM-port lockfile mutex for serial peripherals.

Why this exists
---------------
Orphan Python processes routinely hold COM ports after a hard exit. The
next launch fails with ``PermissionError: [WinError 5] Access is
denied`` and the user spends 5 minutes hunting for the stale process.
Worse, when ``radar.start_dual()`` and ``oximeter.start()`` race for the
same port, neither emits a useful error.

This module wraps a tiny lockfile protocol around port acquisition:

* ``~/.dodo_vision/locks/<safe_port>.lock`` carries
  ``{pid, started_at, owner, port}`` as JSON.
* ``acquire(port, owner)`` returns ``True`` if no live process holds the
  port, taking ownership; ``False`` otherwise. Stale lockfiles whose
  PID is dead are swept automatically.
* ``release(port)`` deletes the lockfile (only if we still own it).

The implementation is **deliberately advisory** — opening the actual
``serial.Serial`` is the real mutex. This file only makes the failure
mode legible: instead of "PermissionError: WinError 5", you get
"port COM5 is held by pid=12345 (radar_a) since 2026-04-25T18:20:00".

Cross-platform: relies only on stdlib (``os.kill(pid, 0)`` for liveness
on POSIX; ``OpenProcess`` via ``ctypes`` on Windows). No external deps.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

_LOCK_DIR = Path.home() / ".dodo_vision" / "locks"


def _safe_filename(port: str) -> str:
    """Map ``COM5`` / ``/dev/ttyUSB0`` into a filename-safe stem."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", port)


def _lock_path(port: str) -> Path:
    return _LOCK_DIR / f"{_safe_filename(port)}.lock"


def _pid_alive(pid: int) -> bool:
    """Return True iff a process with this PID currently exists.

    Best-effort: false-negatives are safe (we'd just skip an over-eager
    sweep). False-positives mean we won't take over a dead lockfile,
    which still degrades to the underlying serial PermissionError that
    the user already sees today.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259  # documented Windows constant

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
            )
            if not handle:
                return False
            try:
                exit_code = wintypes.DWORD()
                ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                if not ok:
                    return False
                return exit_code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception as e:
            print(f"[port_lock] PID liveness probe failed: {type(e).__name__}: {e}")
            return True  # err on the side of NOT yanking the lock
    # POSIX
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # PID exists but we can't signal it
    except Exception as e:
        print(f"[port_lock] POSIX liveness probe failed: {type(e).__name__}: {e}")
        return True


def _read_lock(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        print(f"[port_lock] read {path.name} corrupt: {type(e).__name__}: {e}")
        return None


def _atomic_write(path: Path, payload: dict) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
        return True
    except OSError as e:
        print(f"[port_lock] write {path.name} failed: {type(e).__name__}: {e}")
        try:
            tmp.unlink()
        except OSError:
            pass  # nolint:silent-except tmp cleanup
        return False


def acquire(port: str, owner: str) -> bool:
    """Try to claim ``port`` for ``owner``. Return True on success.

    ``owner`` is a free-form string (e.g. ``"radar_a"``, ``"radar_b"``,
    ``"oximeter"``) used purely for diagnostics in error messages.
    """
    if not port:
        return False
    path = _lock_path(port)
    existing = _read_lock(path)
    if existing:
        pid = int(existing.get("pid") or 0)
        if pid and pid != os.getpid() and _pid_alive(pid):
            print(
                f"[port_lock] {port} held by pid={pid} "
                f"({existing.get('owner') or '?'}) since "
                f"{existing.get('started_at') or '?'} — refusing to take over"
            )
            return False
        if pid and pid != os.getpid():
            print(f"[port_lock] {port} stale lockfile (pid={pid} dead) — sweeping")
    payload = {
        "pid": os.getpid(),
        "started_at": time.time(),
        "owner": owner,
        "port": port,
    }
    return _atomic_write(path, payload)


def release(port: str) -> None:
    """Drop our lock on ``port`` (no-op if we don't own it)."""
    if not port:
        return
    path = _lock_path(port)
    existing = _read_lock(path)
    if not existing:
        return
    if int(existing.get("pid") or 0) != os.getpid():
        # Not ours — leave alone.
        return
    try:
        path.unlink()
    except OSError:
        pass  # nolint:silent-except best-effort cleanup


def describe(port: str) -> Optional[dict]:
    """Return the current lockfile contents, or None if absent."""
    return _read_lock(_lock_path(port))
