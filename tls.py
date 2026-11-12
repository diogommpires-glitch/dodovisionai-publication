"""tls.py — v9.5.x TLS / `wss://` support.

Two modes, triggered by env vars:

    DODO_TLS_CERT + DODO_TLS_KEY   → use these PEM files directly
    DODO_TLS_SELFSIGN=1            → auto-generate a dev cert on startup
                                      (written to certs/localhost.{crt,key})

When neither is set the server runs plain HTTP/WS (current behaviour).

The helper returns kwargs that get merged into `uvicorn.Config(...)`. If
nothing is configured the helper returns an empty dict and uvicorn stays
on HTTP.

The self-signed path requires `cryptography` (pure-Python, already in
the dependency tree via `websockets`). No OpenSSL binary needed.
"""
from __future__ import annotations

import datetime as _dt
import os
import pathlib
from typing import Any


CERT_DIR = pathlib.Path(__file__).resolve().parent / "certs"


def _self_signed(host: str = "localhost") -> tuple[str, str]:
    """Generate a throwaway cert valid for the given hostname. Returns
    (cert_path, key_path). Regenerates iff the existing cert is expired."""
    CERT_DIR.mkdir(exist_ok=True)
    crt = CERT_DIR / "localhost.crt"
    key = CERT_DIR / "localhost.key"
    if crt.exists() and key.exists():
        try:
            mtime = _dt.datetime.fromtimestamp(crt.stat().st_mtime, tz=_dt.timezone.utc)
            if _dt.datetime.now(_dt.timezone.utc) - mtime < _dt.timedelta(days=300):
                return str(crt), str(key)
        except Exception:
            pass
    # Build a fresh cert.
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    pk = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, host),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "DodoVisionAI dev"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(pk.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_dt.datetime.utcnow() - _dt.timedelta(days=1))
        .not_valid_after(_dt.datetime.utcnow() + _dt.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(host),
                x509.DNSName("127.0.0.1"),
                x509.IPAddress(__import__("ipaddress").ip_address("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(pk, hashes.SHA256())
    )
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key.write_bytes(pk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    return str(crt), str(key)


def uvicorn_kwargs() -> dict[str, Any]:
    """Return TLS-related kwargs for uvicorn, or {} if no TLS configured."""
    cert = os.environ.get("DODO_TLS_CERT")
    key = os.environ.get("DODO_TLS_KEY")
    if cert and key:
        return {"ssl_certfile": cert, "ssl_keyfile": key}
    if os.environ.get("DODO_TLS_SELFSIGN") == "1":
        c, k = _self_signed()
        return {"ssl_certfile": c, "ssl_keyfile": k}
    return {}


def enabled() -> bool:
    return bool(uvicorn_kwargs())


__all__ = ["uvicorn_kwargs", "enabled"]
