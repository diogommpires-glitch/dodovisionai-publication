"""envelope_schema.py — v9.5.x typed WebSocket envelope schema.

Machine-readable contract for every envelope the server emits on
`/ws/detect`. Both static type-checkers (mypy / pyright) and FastAPI's
OpenAPI generator read from these definitions.

The multiplex protocol ships SEVEN envelope shapes, discriminated by the
`t` (type) field:

    t="frame"    — primary per-frame payload (faces, av, persons, …)
    t="hr"       — per-person heart-rate update
    t="spo2"     — per-person SpO2 update
    t="decep"    — per-person deception score update
    t="llm"      — LLM summary / insight
    t="vitals"   — radar / dual-cam / GSR bundle
    t="corr"     — tracker correction signal

Legacy frames without a `t` tag are treated as t="frame" by the client.

These definitions are descriptive, not enforcing. The server still emits
dicts via `sanitize(...)` so runtime shape drift is caught by tests, not
by the type system.
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict, Literal, Union


# ── Sub-structures ────────────────────────────────────────────────────

class BlinkInfo(TypedDict, total=False):
    blinked: bool
    blink_rate_pm: int
    eye_openness: float
    perclos: float
    drowsiness: Literal["alert", "fatigued", "drowsy"]


class GazeInfo(TypedDict, total=False):
    horizontal: float
    vertical: float


class PoseInfo(TypedDict, total=False):
    pitch: float
    yaw: float
    roll: float


class IrisPoint(TypedDict):
    x: float
    y: float


class IrisInfo(TypedDict, total=False):
    l_iris: IrisPoint
    r_iris: IrisPoint
    l_ring: list[IrisPoint]
    r_ring: list[IrisPoint]
    l_eye: list[IrisPoint]
    r_eye: list[IrisPoint]
    l_dilation: float
    r_dilation: float
    avg_dilation: float


class FaceFast(TypedDict, total=False):
    i: int
    lm: list[list[float]]            # [[x, y], ...] length 478
    aus: dict[str, float]
    pose: PoseInfo
    gaze: GazeInfo
    blink: BlinkInfo
    micro: dict[str, Any]
    bs_emotion: dict[str, float]
    fused_emotion: dict[str, float]
    iris: IrisInfo


class AffectVector(TypedDict, total=False):
    valence: float
    arousal: float
    dominant_emotion: str
    emotion_scores: dict[str, float]
    heart_rate_bpm: Optional[float]
    confirmed_bpm: Optional[float]
    hrv_rmssd: Optional[float]
    resp_rate: Optional[float]
    stress_index: Optional[float]
    blink_rate: Optional[float]
    perclos: float
    drowsiness: str
    radar_hr: Optional[float]
    radar_breath: Optional[float]
    radar_presence: Optional[bool]
    radar_distance: Optional[float]
    radar_people_count: Optional[int]
    radar_b_hr: Optional[float]
    radar_b_breath: Optional[float]
    radar_b_distance: Optional[float]
    radar_b_people_count: Optional[int]
    radar_people_total: Optional[int]
    radar_distance_source: Optional[Literal["radar", "camera"]]
    radar_people_source: Optional[Literal["radar", "camera"]]
    pupil_dilation: Optional[float]


class PADBranchScore(TypedDict, total=False):
    name: str
    p_attack: float
    reliability: float
    latency_ms: float
    meta: dict[str, Any]
    dominant_class: Optional[str]


class PADVerdict(TypedDict, total=False):
    verdict: Literal["LIVE", "SPOOF", "UNCERTAIN"]
    p_attack: float
    attack_class: Optional[str]
    log_odds: float
    ts: float
    branches: list[PADBranchScore]


class PersonAffect(TypedDict, total=False):
    track_id: int
    identity: Optional[str]
    heart_rate: Optional[float]
    confirmed_bpm: Optional[float]
    hrv_rmssd: Optional[float]
    valence: float
    arousal: float
    dominant_emotion: str
    emotion_scores: dict[str, float]
    pad: Optional[PADVerdict]


# ── Envelope types ────────────────────────────────────────────────────

class FrameEnvelope(TypedDict, total=False):
    """Primary per-frame payload. Fires at the server's frame cadence
    (typically 25-30 Hz for fast frames, 2 Hz for full frames)."""
    t: Literal["frame"]
    faces: list[FaceFast]
    av: AffectVector
    persons: list[PersonAffect]
    hr: Optional[float]
    wave: Optional[list[float]]
    confirmed_bpm: Optional[float]
    elapsed: float
    buf: int
    samples: int
    mode: Literal["fast", "full", "fast_lmk"]
    ms: float
    ts_server_mono: float


class HREnvelope(TypedDict):
    t: Literal["hr"]
    pid: int
    bpm: float
    ts_server_mono: float


class SpO2Envelope(TypedDict):
    t: Literal["spo2"]
    pid: int
    spo2: float
    ts_server_mono: float


class DeceptionEnvelope(TypedDict):
    t: Literal["decep"]
    pid: int
    score: float
    conf: Literal["none", "low", "medium", "high"]
    ts_server_mono: float


class LLMEnvelope(TypedDict, total=False):
    t: Literal["llm"]
    pid: int
    summary: str
    tokens: int
    ts: float


class VitalsEnvelope(TypedDict, total=False):
    t: Literal["vitals"]
    radar: dict[str, Optional[float]]
    gsr: Optional[dict[str, Any]]
    voice: Optional[dict[str, Any]]
    ts: float


class CorrEnvelope(TypedDict):
    t: Literal["corr"]
    pid: int
    lm: list[list[float]]
    conf: float
    ts_server_mono: float


Envelope = Union[
    FrameEnvelope, HREnvelope, SpO2Envelope, DeceptionEnvelope,
    LLMEnvelope, VitalsEnvelope, CorrEnvelope,
]

__all__ = [
    "BlinkInfo", "GazeInfo", "PoseInfo", "IrisInfo",
    "FaceFast", "AffectVector", "PADBranchScore", "PADVerdict",
    "PersonAffect",
    "FrameEnvelope", "HREnvelope", "SpO2Envelope", "DeceptionEnvelope",
    "LLMEnvelope", "VitalsEnvelope", "CorrEnvelope",
    "Envelope",
]
