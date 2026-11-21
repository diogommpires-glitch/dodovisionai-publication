"""
DODO VISION AI v8.8 — Cross-modal TRUST Gauge

Cross-checks the three independent heart-rate sources we have access to:
  - rPPG  : camera (POS algorithm, hr_confidence in [0,1])
  - Radar : 60 GHz mmWave (Seeed MR60BHA2, presence flag is signal_ok)
  - Oxi   : Pulox PO-300 over HID/serial (ground truth, low noise)

When two or more agree within ~5 BPM the trust score is high; growing
divergence drops it. The gauge also surfaces *which* sensors are live so
the LLM and UI can call out e.g. "rPPG drift, oxi disconnected" rather
than silently averaging conflicting numbers.

Public API:
    score_trust(person_dict)  -> dict
    score_trust_state(person_state) -> dict   # convenience wrapper

Returned shape:
    {
      "trust":        0.0..1.0,
      "level":        "high"|"medium"|"low"|"none",
      "sources":      ["rppg","radar","oxi"],
      "agreement_bpm": float | None,    # max pairwise delta among active
      "consensus_bpm": float | None,    # weighted median of active sources
      "divergence_flag": bool,
      "explain":      "string"
    }
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple


# Tunables (BPM)
_TIGHT_AGREEMENT_BPM = 5.0   # all sources within this -> trust ~1.0
_SOFT_AGREEMENT_BPM  = 12.0  # within this -> partial trust
_HARD_DIVERGENCE_BPM = 20.0  # beyond this -> divergence flag

# Per-source baseline weights for the consensus median.
# Oximeter is closest to ground truth, then radar, then rPPG.
_SOURCE_WEIGHT = {"oxi": 1.0, "radar": 0.7, "rppg": 0.5}


def _collect_sources(p: Dict[str, Any]) -> List[Tuple[str, float, float]]:
    """Return [(name, hr, weight)] for each source that has a usable value."""
    out: List[Tuple[str, float, float]] = []
    rppg_hr = p.get("heart_rate_bpm") or p.get("calibrated_hr") or p.get("confirmed_bpm")
    rppg_q  = (p.get("signal_quality") or {}).get("rppg") if isinstance(p.get("signal_quality"), dict) else None
    if rppg_hr and 30 <= rppg_hr <= 220:
        w = _SOURCE_WEIGHT["rppg"] * (0.5 + 0.5 * (rppg_q or 0.5))
        out.append(("rppg", float(rppg_hr), w))
    radar_hr = p.get("radar_hr")
    if radar_hr and 30 <= radar_hr <= 220:
        out.append(("radar", float(radar_hr), _SOURCE_WEIGHT["radar"]))
    oxi_hr = p.get("oxi_hr")
    if oxi_hr and 30 <= oxi_hr <= 220:
        out.append(("oxi", float(oxi_hr), _SOURCE_WEIGHT["oxi"]))
    return out


def _weighted_median(values: List[Tuple[float, float]]) -> Optional[float]:
    """Weighted median of [(value, weight)] pairs."""
    if not values:
        return None
    sv = sorted(values, key=lambda t: t[0])
    total = sum(w for _, w in sv)
    if total <= 0:
        return None
    cum = 0.0
    half = total / 2.0
    for v, w in sv:
        cum += w
        if cum >= half:
            return v
    return sv[-1][0]


def score_trust(p: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a cross-modal trust score from a person/affect dict."""
    sources = _collect_sources(p)
    names = [s[0] for s in sources]

    if not sources:
        return {
            "trust": 0.0, "level": "none", "sources": [],
            "agreement_bpm": None, "consensus_bpm": None,
            "divergence_flag": False,
            "explain": "no_hr_sources_available",
        }

    if len(sources) == 1:
        name, hr, _ = sources[0]
        # Single source: trust = source's nominal weight (oxi > radar > rppg)
        # rPPG quality further modulates it
        base = _SOURCE_WEIGHT[name]
        if name == "rppg":
            q = (p.get("signal_quality") or {}).get("rppg", 0.5)
            base *= 0.5 + 0.5 * float(q or 0.0)
        return {
            "trust": round(min(1.0, base), 3),
            "level": "medium" if base >= 0.6 else "low",
            "sources": names,
            "agreement_bpm": 0.0, "consensus_bpm": round(hr, 1),
            "divergence_flag": False,
            "explain": f"single_source:{name}",
        }

    # 2+ sources: measure pairwise spread
    hrs = [hr for _, hr, _ in sources]
    lo, hi = min(hrs), max(hrs)
    spread = hi - lo
    consensus = _weighted_median([(hr, w) for _, hr, w in sources])

    if spread <= _TIGHT_AGREEMENT_BPM:
        trust = 0.9 + 0.1 * (1.0 - spread / max(1.0, _TIGHT_AGREEMENT_BPM))
        level = "high"
        explain = f"all_within_{_TIGHT_AGREEMENT_BPM:.0f}bpm"
    elif spread <= _SOFT_AGREEMENT_BPM:
        # Linear interpolate between high and low confidence
        frac = (spread - _TIGHT_AGREEMENT_BPM) / (_SOFT_AGREEMENT_BPM - _TIGHT_AGREEMENT_BPM)
        trust = 0.7 * (1 - frac) + 0.45 * frac
        level = "medium"
        explain = f"partial_agreement_spread={spread:.1f}bpm"
    else:
        # Hard divergence -> low trust
        # but if oxi is present, it's still authoritative on its own number
        if "oxi" in names:
            trust = 0.4
            level = "low"
            explain = f"hr_divergence_spread={spread:.1f}bpm_oxi_authoritative"
        else:
            trust = 0.25
            level = "low"
            explain = f"hr_divergence_spread={spread:.1f}bpm"

    return {
        "trust": round(max(0.0, min(1.0, trust)), 3),
        "level": level,
        "sources": names,
        "agreement_bpm": round(spread, 1),
        "consensus_bpm": round(consensus, 1) if consensus is not None else None,
        "divergence_flag": spread >= _HARD_DIVERGENCE_BPM,
        "explain": explain,
    }


def score_trust_state(person_state) -> Dict[str, Any]:
    """Convenience: score from a PersonState object directly."""
    if person_state is None:
        return score_trust({})
    try:
        return score_trust(person_state.build_affect_dict())
    except Exception:
        return score_trust({})
