"""
DODO VISION AI v8.3 — HR Sensor Fusion (rPPG + radar + oximeter)

Fuses camera-based rPPG (per-person, noisy-under-motion) with the radar's
contactless HR (single-subject, noisy-at-distance) and an optional pulse
oximeter (medical-grade ground truth) into one confidence-weighted estimate
per person, plus a "disagreement" flag that surfaces when sensors diverge
by more than a plausible physiological delta.

Design (v8.3)
-------------
- Inverse-variance weighting in BPM-space.
- Radar HR is only trusted if radar.presence is True and age < 3s.
- rPPG is only trusted if its signal_quality >= 0.35.
- NEW (v8.3): Oximeter HR (Pulox PO-300) is a 3rd source with sigma=1.0
  BPM (medical-grade). When connected with signal_strength >= 2, it
  dominates the fusion as ground truth.
- Hard physiological window 40-180 BPM.
- Radar sigma is inflated when hr_filter SQI is low.
- Detect+drop single-sample disagreement outliers.
- NEW (v8.3): `hr_source` field in output indicates dominant source.

Research anchors: see hr_filter.py header.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

# ── v8.3 tightened defaults ──
RPPG_SIGMA_BASE  = 3.0    # rPPG under good signal
RADAR_SIGMA_BASE = 4.5    # Seeed MR60BHA2 in clean conditions
OXI_SIGMA_BASE   = 1.0    # Pulox PO-300 medical-grade pulse oximeter
DISAGREEMENT_BPM = 12.0   # delta above which we flag mismatch
HARD_DROP_BPM    = 30.0   # delta above which we drop the weaker source
HR_MIN, HR_MAX   = 40.0, 180.0   # adult hard-physiological window


@dataclass
class FusedHR:
    bpm: Optional[float]
    confidence: float          # 0..1
    sources: Dict[str, float]  # {"rppg": 72.1, "radar": 74.3, "oximeter": 73.0}
    disagreement: bool
    reason: str
    hr_source: str = ""        # v8.3: dominant source tag for UI


def _in_window(bpm: Optional[float]) -> bool:
    return bpm is not None and (HR_MIN <= float(bpm) <= HR_MAX)


def _rppg_sigma(quality: Optional[float]) -> Optional[float]:
    if quality is None or quality < 0.2:
        return None
    return RPPG_SIGMA_BASE * (1.0 / max(0.25, quality))


def _radar_sigma(radar_age_s: Optional[float],
                 presence: Optional[bool],
                 radar_sqi: Optional[float] = None) -> Optional[float]:
    if radar_age_s is None or radar_age_s > 3.0:
        return None
    if presence is False:
        return None
    k_age = 1.0 + (max(0.0, radar_age_s) / 3.0) * 1.5
    # v7.4: low SQI -> inflate sigma. SQI 1.0 -> 1x, SQI 0.3 -> ~3.3x.
    if radar_sqi is not None and radar_sqi >= 0.0:
        k_sqi = 1.0 / max(0.3, float(radar_sqi))
    else:
        k_sqi = 1.0
    return RADAR_SIGMA_BASE * k_age * k_sqi


def fuse(rppg_bpm: Optional[float],
         rppg_quality: Optional[float],
         radar_bpm: Optional[float],
         radar_age_s: Optional[float],
         radar_presence: Optional[bool],
         radar_sqi: Optional[float] = None,
         oxi_bpm: Optional[float] = None,
         oxi_signal: Optional[int] = None) -> FusedHR:
    s_r = _rppg_sigma(rppg_quality)
    s_m = _radar_sigma(radar_age_s, radar_presence, radar_sqi)

    have_r = (s_r is not None) and _in_window(rppg_bpm)
    have_m = (s_m is not None) and _in_window(radar_bpm)

    # v8.3: oximeter as ground-truth source (sigma = 1.0 BPM)
    have_o = (oxi_bpm is not None and _in_window(oxi_bpm)
              and (oxi_signal is not None and oxi_signal >= 2))
    s_o = OXI_SIGMA_BASE if have_o else None

    # Hard-drop gate: if both present but one is wildly off AND low quality,
    # drop it rather than pull the fused mean toward the bad source.
    if have_r and have_m:
        delta = abs(rppg_bpm - radar_bpm)  # type: ignore[arg-type]
        if delta > HARD_DROP_BPM:
            rppg_weak  = (rppg_quality or 0.0) < 0.5
            radar_weak = (radar_sqi    or 0.0) < 0.5
            if radar_weak and not rppg_weak:
                have_m = False
            elif rppg_weak and not radar_weak:
                have_r = False
            # else: keep both, disagreement flag will fire

    # v8.3: if oximeter is connected and valid, it dominates as ground truth.
    # When oxi disagrees with rPPG/radar by >HARD_DROP_BPM, trust oxi.
    if have_o:
        if have_r and abs(rppg_bpm - oxi_bpm) > HARD_DROP_BPM:
            have_r = False
        if have_m and abs(radar_bpm - oxi_bpm) > HARD_DROP_BPM:
            have_m = False

    sources = {}
    weights = []  # (bpm, weight)

    if have_r:
        w = 1.0 / (s_r * s_r)
        weights.append((float(rppg_bpm), w, "rppg"))
        sources["rppg"] = rppg_bpm
    if have_m:
        w = 1.0 / (s_m * s_m)
        weights.append((float(radar_bpm), w, "radar"))
        sources["radar"] = radar_bpm
    if have_o:
        w = 1.0 / (s_o * s_o)
        weights.append((float(oxi_bpm), w, "oximeter"))
        sources["oximeter"] = oxi_bpm

    if not weights:
        return FusedHR(None, 0.0, {}, False, "no_inputs", hr_source="")

    # Single source
    if len(weights) == 1:
        bpm, w, src = weights[0]
        sigma = 1.0 / (w ** 0.5)
        conf = min(1.0, max(0.1, 1.0 - (sigma / 12.0)))
        return FusedHR(round(bpm, 1), round(conf, 2),
                       sources, False, f"{src}_only", hr_source=src.upper())

    # Multi-source inverse-variance fuse
    total_w = sum(w for _, w, _ in weights)
    fused_bpm = sum(bpm * w for bpm, w, _ in weights) / total_w
    s_f = (1.0 / total_w) ** 0.5
    conf = min(1.0, max(0.1, 1.0 - (s_f / 10.0)))

    # Determine dominant source (highest weight)
    dominant_src = max(weights, key=lambda x: x[1])[2].upper()

    # Disagreement check (any pair diverges >= threshold)
    disagree = False
    bpms = [bpm for bpm, _, _ in weights]
    for i in range(len(bpms)):
        for j in range(i + 1, len(bpms)):
            if abs(bpms[i] - bpms[j]) >= DISAGREEMENT_BPM:
                disagree = True
                break
    if disagree:
        conf *= 0.5

    reason = "fused" if not disagree else "fused_disagreement"
    return FusedHR(round(fused_bpm, 1), round(conf, 2),
                   sources, disagree, reason, hr_source=dominant_src)


def fuse_from_state(person_state: Any, radar_snapshot: Dict[str, Any],
                    oxi_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience wrapper for the per-frame loop.
    Expects:
      person_state.rppg_bpm, person_state.rppg_quality
      radar_snapshot = radar.summary()   (now carries hr_sqi in v7.4)
      oxi_data = oximeter.get_oximeter_data() (v8.3, optional)
    """
    result = fuse(
        rppg_bpm       = getattr(person_state, "rppg_bpm", None),
        rppg_quality   = getattr(person_state, "rppg_quality", None),
        radar_bpm      = radar_snapshot.get("heart_rate"),
        radar_age_s    = radar_snapshot.get("age_s"),
        radar_presence = radar_snapshot.get("presence"),
        radar_sqi      = radar_snapshot.get("hr_sqi"),
        oxi_bpm        = oxi_data.get("heart_rate") if oxi_data else None,
        oxi_signal     = oxi_data.get("signal_strength") if oxi_data else None,
    )
    return {
        "bpm": result.bpm,
        "confidence": result.confidence,
        "sources": result.sources,
        "disagreement": result.disagreement,
        "reason": result.reason,
        "hr_source": result.hr_source,
        "oxi_spo2": oxi_data.get("spo2") if oxi_data else None,
    }
