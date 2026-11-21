from pydantic import BaseModel, Field
from typing import Dict, Optional, Any

class HeadPose(BaseModel):
    pitch: float
    yaw: float
    roll: float

class Gaze(BaseModel):
    horizontal: float
    vertical: float

class MicroExpression(BaseModel):
    detected: bool
    muscle: Optional[str] = None
    intensity: Optional[float] = None
    source: Optional[str] = None

class DeceptionState(BaseModel):
    score: float
    baselined: bool
    indicators: Dict[str, float]
    cold_baseline: Dict[str, float]
    warm_baseline: Dict[str, float]
    baseline_drift: Dict[str, float]
    baseline_drift_score: float
    cold_locked: bool

class VocalState(BaseModel):
    f0: Optional[float]
    jitter: Optional[float]
    stress: float

class SignalQuality(BaseModel):
    face: float
    rppg: float

class TrustGauge(BaseModel):
    trust: float
    level: str
    explain: str

class PersonAffectState(BaseModel):
    track_id: int
    identity: Optional[str] = None
    valence: float
    arousal: float
    dominant_emotion: str
    emotion_scores: Dict[str, float] = Field(default_factory=dict)
    
    heart_rate_bpm: Optional[float] = None
    confirmed_bpm: Optional[float] = None
    calibrated_hr: Optional[float] = None
    hrv_rmssd: Optional[float] = None
    stress_index: Optional[float] = None
    resp_rate: Optional[float] = None
    
    blink_rate: Optional[float] = None
    perclos: float = 0.0
    drowsiness: str = "alert"
    
    head_pose: Optional[HeadPose] = None
    gaze: Optional[Gaze] = None
    micro_expression: Optional[MicroExpression] = None
    pupil_dilation: Optional[float] = None
    
    age: Optional[int] = None
    gender: Optional[str] = None
    race: Optional[str] = None
    
    oxi_hr: Optional[float] = None
    oxi_spo2: Optional[float] = None
    radar_hr: Optional[float] = None
    radar_breath: Optional[float] = None
    radar_distance: Optional[float] = None
    radar_presence: Optional[bool] = None
    gsr_conductance: Optional[float] = None
    gsr_scr: Optional[float] = None
    
    duchenne_smile: bool = False
    fake_smile: bool = False
    facial_asymmetry: float = 0.0
    
    deception: Optional[DeceptionState] = None
    vocal: Optional[VocalState] = None
    
    llm_analysis: str = ""
    rppg_quality: float = 0.0
    hr_source: Optional[str] = None
    signal_quality: Optional[SignalQuality] = None
    timestamp: float
    trust: Optional[TrustGauge] = None
    
    _access_level: Optional[str] = None
    alertness_state: Optional[str] = None

class WebSocketEnvelope(BaseModel):
    t: str = "frame"
    ts_server_mono: Optional[float] = None
    ms: float = 0.0
    mode: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None
    hr: Optional[float] = None
    confirmed_bpm: Optional[float] = None
    faces: list[Dict[str, Any]] = Field(default_factory=list)
    persons: list[PersonAffectState] = Field(default_factory=list)
    av: Optional[Dict[str, Any]] = None
    gsr: Optional[Dict[str, Any]] = None
    voice: Optional[Dict[str, Any]] = None
    dual_cam: Optional[Dict[str, Any]] = None
    rppg_q: Optional[Dict[str, Any]] = None
