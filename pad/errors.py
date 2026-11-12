"""pad/errors.py — structured error taxonomy for PAD + fusion subsystems.

Centralises error codes so the WebSocket client, HTTP clients, and the paper
reproducibility section all see the same strings. Consumers return dicts:

    {"error_code": "CALIBRATOR_MISSING",
     "message": "pad/models/calibrators.pkl not on disk"}

Usage:
    from pad.errors import ErrorCode, error_payload
    raise PADError(ErrorCode.BRANCH_TIMEOUT, "moire took 850 ms")
    response = error_payload(ErrorCode.CALIBRATOR_MISSING, path=str(p))
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class ErrorCode(str, Enum):
    """Canonical PAD / fusion error codes. Values are stable strings
    (do not renumber -- clients may match on them)."""

    # Model / artefact issues
    CALIBRATOR_MISSING       = "CALIBRATOR_MISSING"
    CALIBRATOR_CORRUPT       = "CALIBRATOR_CORRUPT"
    CALIBRATOR_DEGENERATE    = "CALIBRATOR_DEGENERATE"
    ONNX_MISSING             = "ONNX_MISSING"
    ONNX_LOAD_FAILED         = "ONNX_LOAD_FAILED"
    META_FUSER_MISSING       = "META_FUSER_MISSING"

    # Branch-level runtime
    BRANCH_TIMEOUT           = "BRANCH_TIMEOUT"
    BRANCH_CRASHED           = "BRANCH_CRASHED"
    BRANCH_LOW_RELIABILITY   = "BRANCH_LOW_RELIABILITY"
    BRANCH_NOT_AVAILABLE     = "BRANCH_NOT_AVAILABLE"

    # Fusion / policy
    FUSION_NO_USABLE_BRANCH  = "FUSION_NO_USABLE_BRANCH"
    FUSION_THRESHOLD_UNSET   = "FUSION_THRESHOLD_UNSET"
    ADV_DEFENSE_SKIPPED      = "ADV_DEFENSE_SKIPPED"

    # Auth / rate / protocol
    AUTH_MISSING             = "AUTH_MISSING"
    AUTH_INVALID             = "AUTH_INVALID"
    RATE_LIMITED             = "RATE_LIMITED"
    PROTOCOL_VIOLATION       = "PROTOCOL_VIOLATION"

    # Data / sensor
    FRAME_DECODE_FAILED      = "FRAME_DECODE_FAILED"
    NO_FACE_DETECTED         = "NO_FACE_DETECTED"
    SENSOR_STALE             = "SENSOR_STALE"
    SENSOR_DISCONNECTED      = "SENSOR_DISCONNECTED"

    # Generic
    INTERNAL_ERROR           = "INTERNAL_ERROR"


@dataclass
class PADError(Exception):
    code: ErrorCode
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # noqa: D401
        msg = self.message or self.code.value
        if self.details:
            return f"[{self.code.value}] {msg} ({self.details})"
        return f"[{self.code.value}] {msg}"

    def as_dict(self) -> Dict[str, Any]:
        out = {"error_code": self.code.value, "message": self.message or self.code.value}
        if self.details:
            out["details"] = self.details
        return out


def error_payload(code: ErrorCode, message: str = "", **details) -> Dict[str, Any]:
    """Convenience builder returning the JSON-serialisable error body."""
    body: Dict[str, Any] = {
        "error_code": code.value,
        "message": message or code.value,
    }
    if details:
        body["details"] = details
    return body


__all__ = ["ErrorCode", "PADError", "error_payload"]
