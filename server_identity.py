"""Server-identity stamp for DodoVisionAI.

Why this exists
---------------
"Am I running new code?" recurs every session. Without git, file mtimes
flicker as OneDrive syncs, and the banner version (`v9.7.x`) is
hand-edited so it lies if the user forgets to bump.

This module computes a **content hash** of the critical Python files at
boot and records the PID + wall-clock start time. The result is exposed
to:

* the boot banner (one line),
* the new ``/health/full`` endpoint (machine-readable),
* the WebSocket envelope (so the UI can show it in the top-bar pill).

If the user reloads the page and the hash changed, the UI knows to clear
caches and resume from a clean slate.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

# Files whose content changes are user-meaningful. Adding a file here
# means a stale-server warning fires when it changes between sessions.
# Keep this list short — every entry is hashed at every boot.
_CRITICAL_FILES = (
    "main.py",
    "config.py",
    "ui.py",
    "radar.py",
    "rppg.py",
    "analysis.py",
    "detectors.py",
)

_REPO_ROOT = Path(__file__).resolve().parent

_BOOT_PID: int = 0
_BOOT_TIME: float = 0.0
_CONTENT_HASH: str = "?"
_PER_FILE: dict[str, str] = {}


def _hash_file(path: Path) -> str:
    """Return SHA256 hex of the file content, or '<missing>' if not found."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "<missing>"


def init() -> dict:
    """Initialise identity. Idempotent — call once at boot."""
    global _BOOT_PID, _BOOT_TIME, _CONTENT_HASH, _PER_FILE
    if _BOOT_PID:  # already initialised
        return summary()
    _BOOT_PID = os.getpid()
    _BOOT_TIME = time.time()
    digests: list[str] = []
    per_file: dict[str, str] = {}
    for name in _CRITICAL_FILES:
        d = _hash_file(_REPO_ROOT / name)
        per_file[name] = d[:12] if d != "<missing>" else d
        digests.append(f"{name}:{d}")
    rolling = hashlib.sha256("\n".join(digests).encode("utf-8")).hexdigest()
    _CONTENT_HASH = rolling[:12]
    _PER_FILE = per_file
    return summary()


def summary() -> dict:
    """Return the current identity as a JSON-safe dict."""
    return {
        "pid": _BOOT_PID,
        "started_at": _BOOT_TIME,
        "uptime_s": round(time.time() - _BOOT_TIME, 1) if _BOOT_TIME else 0.0,
        "content_hash": _CONTENT_HASH,
        "per_file": dict(_PER_FILE),
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }


def banner_line(app_version: str) -> str:
    """One-liner suitable for the boot banner."""
    s = summary()
    return (
        f"  Identity:   pid={s['pid']} hash={s['content_hash']} "
        f"py={s['python']} ver={app_version}"
    )


def is_stale(client_hash: str | None) -> bool:
    """Return True iff the client's last-seen hash differs from current."""
    if not client_hash:
        return False
    return client_hash != _CONTENT_HASH
