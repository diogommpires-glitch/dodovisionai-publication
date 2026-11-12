"""auth.py — v9.5.x bearer-token authentication.

Default-OFF on localhost so nothing breaks for single-user dev. Turn on
by setting `DODO_AUTH_TOKEN=<secret>` in the env. When set:

    * Every ``/ws/detect`` connection must present the token as a
      ``token`` query param OR a ``Sec-WebSocket-Protocol: Bearer.<token>``
      subprotocol.
    * **HTTP:** ``main.py`` registers middleware that requires
      ``Authorization: Bearer <token>`` on all routes except a small
      public set: ``/``, ``/health``, ``/ready``, ``/metrics``, ``/webcam``,
      ``/flags``, ``/version``, ``/favicon.ico``, ``/static/*``, and OpenAPI
      docs paths (``/docs``, ``/redoc``, ``/openapi.json``). This covers
      ``/pad/*``, ``/detect/*``, ``/faces/*``, etc.
    * The public ``/webcam`` shell prompts the operator for their token.
      Server credentials are never embedded in the response.
    * ``/health``, ``/ready``, ``/metrics``, ``/favicon.ico``, ``/static/*``,
      and ``/webcam`` stay usable without a manual header for probes and UI shell.

Multiple tokens can be configured: `DODO_AUTH_TOKEN=secret1,secret2`.
The server accepts any of them (useful for rolling rotation).
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit


def browser_request_allowed(connection) -> bool:
    """Allow local native clients and same-origin browsers, including WebSockets."""
    if connection.headers.get("sec-fetch-site") == "cross-site":
        return False
    try:
        host = urlsplit("//" + connection.headers.get("host", ""))
        if (
            host.hostname not in ("localhost", "127.0.0.1")
            or host.username is not None
            or host.password is not None
            or host.path
            or host.query
            or host.fragment
        ):
            return False
        scheme = "https" if connection.url.scheme in ("https", "wss") else "http"
        port = host.port or (443 if scheme == "https" else 80)
        origin_header = connection.headers.get("origin")
        if origin_header is None:
            return True
        origin = urlsplit(origin_header)
        return (
            origin.scheme == scheme
            and origin.hostname == host.hostname
            and (origin.port or (443 if scheme == "https" else 80)) == port
            and origin.username is None
            and origin.password is None
            and not origin.path
            and not origin.query
            and not origin.fragment
        )
    except ValueError:
        return False


def _load_tokens() -> frozenset[str]:
    raw = os.environ.get("DODO_AUTH_TOKEN", "").strip()
    if not raw:
        return frozenset()
    return frozenset(t.strip() for t in raw.split(",") if t.strip())


_TOKENS = _load_tokens()


def enabled() -> bool:
    """True iff the operator configured at least one token."""
    return len(_TOKENS) > 0


def check(token: str | None) -> bool:
    """True iff `token` is accepted (or auth is disabled)."""
    if not _TOKENS:
        return True
    return token in _TOKENS


def extract_from_ws(websocket) -> str | None:
    """Pull the token out of a FastAPI / starlette WebSocket.

    Order tried: `?token=...` query param → `Sec-WebSocket-Protocol`
    subprotocol (shape `Bearer.<token>`). Returns None when neither is
    present.
    """
    qp = websocket.query_params.get("token")
    if qp:
        return qp
    proto = websocket.headers.get("sec-websocket-protocol", "")
    for chunk in proto.split(","):
        chunk = chunk.strip()
        if chunk.lower().startswith("bearer."):
            return chunk.split(".", 1)[1]
    return None


def extract_from_http(request) -> str | None:
    """Pull the token out of an HTTP request (Authorization header)."""
    hdr = request.headers.get("authorization", "")
    if hdr.lower().startswith("bearer "):
        return hdr.split(" ", 1)[1].strip() or None
    return None


def reload() -> None:
    """Re-read DODO_AUTH_TOKEN from the env (for tests)."""
    global _TOKENS
    _TOKENS = _load_tokens()


__all__ = [
    "browser_request_allowed",
    "check",
    "enabled",
    "extract_from_http",
    "extract_from_ws",
    "reload",
]
