"""log_setup.py — v9.5.x structured logging.

Central configuration for Python's `logging` module. Replaces the ad-hoc
`print(...)` calls scattered throughout the codebase with a proper
leveled, structured logger.

Two output modes, chosen by env var `DODO_LOG_FORMAT`:

    text  (default) — human-readable timestamped stderr output
    json           — one JSON object per line, for log aggregation

Usage:
    from log_setup import get_logger
    log = get_logger(__name__)
    log.info("hello world")
    log.warning("hr=%d looks high", hr)
    log.error("branch failed", exc_info=True)

Rules of thumb:
    log.debug    — noisy, one-per-frame stuff (off by default)
    log.info     — subsystem init, once-per-session milestones
    log.warning  — recoverable anomaly, degraded mode
    log.error    — something broke, but the server kept running
    log.critical — server is about to exit

Level chosen by env var `DODO_LOG_LEVEL` (default INFO).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Mapping


_LEVEL_NAME = os.environ.get("DODO_LOG_LEVEL", "INFO").upper()
_FORMAT = os.environ.get("DODO_LOG_FORMAT", "text").lower()


class _JsonFormatter(logging.Formatter):
    """One-line JSON per record, safe for Vector/Loki/CloudWatch ingestion."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Any extra={...} kwargs show up as top-level keys
        for key, value in record.__dict__.items():
            if key in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "levelname", "levelno", "lineno",
                "message", "module", "msecs", "msg", "name", "pathname",
                "process", "processName", "relativeCreated", "stack_info",
                "thread", "threadName", "taskName",
            ):
                continue
            try:
                json.dumps(value)  # skip unserialisable
                payload[key] = value
            except Exception:
                payload[key] = repr(value)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _TextFormatter(logging.Formatter):
    """Coloured single-line format for human eyes, ISO-ish timestamps."""

    _COLORS = {
        "DEBUG":   "\033[90m",
        "INFO":    "\033[36m",
        "WARNING": "\033[33m",
        "ERROR":   "\033[31m",
        "CRITICAL": "\033[41m\033[97m",
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        t = time.strftime("%H:%M:%S", time.localtime(record.created))
        colour = self._COLORS.get(record.levelname, "")
        reset = self._RESET if colour else ""
        head = f"{t}.{int(record.msecs):03d} {colour}{record.levelname:<7}{reset} {record.name}:"
        body = record.getMessage()
        if record.exc_info:
            body += "\n" + self.formatException(record.exc_info)
        return f"{head} {body}"


_CONFIGURED = False


def configure(force: bool = False) -> None:
    """Install our handler(s) on the root logger. Idempotent.

    In addition to the stderr StreamHandler, if ``DODO_LOG_FILE`` is set a
    ``RotatingFileHandler`` is installed so ``logs/dodo_console.log`` can no
    longer grow unbounded. Rotation defaults: 10 MB per file x 5 backups,
    overridable via ``DODO_LOG_FILE_BYTES`` and ``DODO_LOG_FILE_BACKUPS``.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    fmt = os.environ.get("DODO_LOG_FORMAT", _FORMAT).lower()
    level_name = os.environ.get("DODO_LOG_LEVEL", _LEVEL_NAME).upper()
    log_file = os.environ.get("DODO_LOG_FILE", "").strip()
    root = logging.getLogger()
    root.handlers.clear()

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    if fmt == "json":
        stream_handler.setFormatter(_JsonFormatter())
    else:
        stream_handler.setFormatter(_TextFormatter())
    root.addHandler(stream_handler)

    if log_file:
        try:
            import logging.handlers as _lh
            max_bytes = int(os.environ.get("DODO_LOG_FILE_BYTES", str(10 * 1024 * 1024)))
            backups = int(os.environ.get("DODO_LOG_FILE_BACKUPS", "5"))
            os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".",
                        exist_ok=True)
            fh = _lh.RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backups,
                encoding="utf-8", delay=True,
            )
            fh.setFormatter(_JsonFormatter() if fmt == "json" else _TextFormatter())
            root.addHandler(fh)
        except Exception as exc:
            sys.stderr.write(f"[log_setup] file handler failed: {exc!r}\n")

    level = getattr(logging, level_name, logging.INFO)
    root.setLevel(level)
    for noisy in ("urllib3", "httpx", "websockets", "uvicorn.access",
                  "onnxruntime", "matplotlib"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Canonical entry point: `log = get_logger(__name__)`."""
    configure()
    return logging.getLogger(name)


# Auto-configure on import so modules that forget to call get_logger() still
# get a consistent format. Caller can re-configure() later if they want json.
configure()
