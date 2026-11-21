"""pad/notify.py
==================

Single source of truth for the project's ntfy push helper. Eight
scripts in scripts/ used to copy-paste the same `def ntfy(...)`
implementation -- this module replaces them so a future topic /
schema change happens in one place.

Use
---
    from pad.notify import ntfy
    ntfy("DODO event title", "body text", tags="rocket,sparkles")

Notifications are disabled unless `DODO_NTFY_TOKEN` and a 32–64 character
topic are configured. Use a reserved topic with restricted read/write access.
Select it via `topic=` or `DODO_NTFY_TOPIC`; keep both values out of Git.

Failure mode is intentional: ntfy push errors are *swallowed* and
optionally logged via the `log_fn` callback. We never want a network
blip to crash a 6-hour training run for the sake of a notification.

Tag conventions in this codebase
--------------------------------
- `rocket`         long job started
- `sparkles`       milestone reached
- `white_check_mark`  job completed cleanly
- `trophy`         everything in a multi-phase pipeline finished
- `warning`        job failed but pipeline continued
- `wrench`         hotfix / config change deployed
- `shield`         safety mechanism activated (watchdog, collapse-guard)
- `zap`            performance win
- `arrows_counterclockwise`  auto-resume / recovery in progress
"""
from __future__ import annotations
import os
import re
import urllib.request
from typing import Callable, Optional

def ntfy(title: str,
         body: str = "",
         tags: str = "wrench",
         priority: str = "default",
         topic: Optional[str] = None,
         timeout_s: float = 10.0,
         log_fn: Optional[Callable[[str], None]] = None) -> bool:
    """Push one notification to ntfy.sh and return True on HTTP 200.

    Failures (network down, ntfy.sh outage, DNS hiccup) are caught and
    optionally logged via `log_fn`; the function never raises. The
    caller's normal control flow continues regardless.

    `topic` precedence: kwarg > `DODO_NTFY_TOPIC` env var.
    """
    target_topic = topic if topic is not None else os.environ.get("DODO_NTFY_TOPIC", "")
    token = os.environ.get("DODO_NTFY_TOKEN", "")
    if not token or not re.fullmatch(r"[A-Za-z0-9_-]{32,64}", target_topic):
        return False
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{target_topic}",
            data=body.encode("utf-8"),
            method="POST",
            headers={
                "Title": title,
                "Tags": tags,
                "Priority": priority,
                "Authorization": f"Bearer {token}",
            },
        )
        urllib.request.urlopen(req, timeout=timeout_s).read()
        return True
    except Exception as e:
        if log_fn is not None:
            try:
                log_fn(f"[ntfy] push failed: {type(e).__name__}")
            except Exception:
                # Logger itself is broken? Drop silently rather than
                # propagating into the caller's hot path.
                pass  # nolint:silent-except logger-broken-fallback
        return False


def is_available() -> bool:
    """Quick HEAD-style probe -- is ntfy.sh reachable right now?
    Useful for boot-time diagnostics. Returns False on any exception."""
    try:
        req = urllib.request.Request(
            "https://ntfy.sh/", method="HEAD",
        )
        with urllib.request.urlopen(req, timeout=5.0) as r:
            return r.status == 200
    except Exception:
        return False

