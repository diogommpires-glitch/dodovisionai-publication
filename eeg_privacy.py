"""
DODO VISION AI v7.4 — EEG privacy guardrail

Reference
---------
Frank, Hwu, Jain, Song (USENIX Security 2014, sec12-final56-frank) showed
that consumer EEG headsets can leak PII (PIN digits, face recognition,
month of birth, location) via P300 responses to crafted visual stimuli.

DodoVisionAI does NOT present stimuli (it's passive affect sensing), but
we should still:
 1. Refuse to run the EEG pipeline if the current session's UI is driving
    a rapid flashing stimulus pattern (future feature; stub here).
 2. Scrub any feature vector that looks like it might encode a P300 peak
    from the truth-engine prompt and the forensic bundle.
 3. Log a clear warning the first time EEG is enabled so the user knows
    the risk.

This module provides the guardrail hooks. Call `check_stimulus_safe()`
before enabling EEG, and `scrub_p300_window()` on any raw window you plan
to ship off-device.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

# P300 occurs ~250–500 ms after a salient stimulus. If we ever add stimulus
# presentation, we MUST tag the samples in that window as privacy-sensitive.
P300_WINDOW_MS = (250, 500)

_warned = False


def warn_once() -> None:
    global _warned
    if _warned:
        return
    _warned = True
    log.warning(
        "EEG pipeline enabled. Passive affect sensing is in use; do NOT "
        "present rapid visual/auditory stimuli to the subject while the "
        "headset is worn — see Frank et al. USENIX Security 2014 for the "
        "P300 side-channel risk."
    )


def check_stimulus_safe(ui_state: Optional[Dict] = None) -> bool:
    """Return True if the UI is NOT currently driving a stimulus pattern.

    `ui_state` may carry keys like {"flashing": bool, "flash_hz": float}.
    Conservative: any flash > 2 Hz is flagged unsafe.
    """
    if not ui_state:
        return True
    if ui_state.get("flashing") and float(ui_state.get("flash_hz", 0)) > 2.0:
        return False
    return True


def scrub_p300_window(samples_ms: Sequence[float],
                      values: Sequence[float],
                      stimulus_events_ms: Sequence[float]) -> List[float]:
    """Zero-out any sample within P300_WINDOW_MS of a stimulus event.

    Operates on a (ms, value) paired stream. Returns the scrubbed values.
    If `stimulus_events_ms` is empty, values are returned unchanged.
    """
    if not stimulus_events_ms:
        return list(values)
    out: List[float] = []
    evs = sorted(stimulus_events_ms)
    for t, v in zip(samples_ms, values):
        risky = False
        for ev in evs:
            dt = t - ev
            if P300_WINDOW_MS[0] <= dt <= P300_WINDOW_MS[1]:
                risky = True
                break
            if ev > t + P300_WINDOW_MS[1]:
                break  # events are sorted; no further overlap possible
        out.append(0.0 if risky else float(v))
    return out


def scrub(affect_dict: Dict) -> Dict:
    """Drop EEG-related keys from affect_vector for privacy.

    Parameters
    ----------
    affect_dict : dict
        Affect vector from forensic logging.

    Returns
    -------
    scrubbed : dict
        Copy with keys starting with eeg_, p300_, and proxy_ removed.
    """
    return {k: v for k, v in affect_dict.items()
            if not (k.startswith("eeg_") or k.startswith("p300_")
                    or k.startswith("proxy_"))}
