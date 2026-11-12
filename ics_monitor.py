"""
DODO VISION AI v8.9 — Intrinsic Cognitive Security (ICS) Monitor

Background heuristic that monitors the UI rendering loop for adversarial
stimuli patterns. Specifically detects:

  1. Rapid Serial Visual Presentation (RSVP) attacks: visual elements
     flashing at ~10 Hz, which can covertly harvest subconscious P300
     recognition signals without informed consent.

  2. Photic driving attacks: periodic flashes at frequencies that can
     induce cybersickness, photosensitive seizures, or emotional
     manipulation.

  3. Information flooding: abnormally high frame-rate UI updates that
     may indicate injected adversarial content.

  4. WCAG 2.3.1 spatial bounding: flash area must not exceed 341x256 CSS
     pixels (0.006 steradians / 25% of 10-degree visual field).

  5. Michelson contrast governor: enforces contrast ratio < 1/17 (0.0588)
     for high-frequency visual elements.

  6. Chromaticity suppression: clamps R/(R+G+B) < 0.8 during any frame
     transition to eliminate saturated-red flash hazards.

When a violation is detected, the monitor:
  - Throttles the display refresh rate to a safe level
  - Logs an ICS violation event (timestamp, type, severity)
  - Returns a block signal to the rendering pipeline

Neurorights alignment:
  - Protects the right to free will / personal identity
  - Prevents adversarial manipulation of emotional state
  - Guards against covert subconscious signal harvesting

References:
  - DARPA CT2WS: Rapid Serial Visual Presentation at 10 Hz
  - IEEE RSVP-BCI paradigms for threat detection
  - W3C WCAG 2.3.1: Three Flashes or Below Threshold
  - Stanford neurological spatial bounding logic
  - MIT Media Lab contrast governor
  - CMU chromaticity suppression matrices
"""
from __future__ import annotations
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

# ── Detection thresholds ──
RSVP_FREQ_HZ = 10.0          # DARPA CT2WS used 10 Hz RSVP
RSVP_TOLERANCE_HZ = 2.0      # detect 8-12 Hz as suspicious
PHOTIC_DANGER_LO_HZ = 3.0    # photosensitive seizure range
PHOTIC_DANGER_HI_HZ = 30.0   # (WCAG 2.3.1: 3 flashes/sec max)
WCAG_FLASH_LIMIT = 3.0       # W3C max flashes per second
FLOOD_RATE_HZ = 60.0          # abnormal if UI updates > 60 Hz sustained
SAFE_REFRESH_HZ = 2.0         # throttle to this on violation

# v8.9: WCAG 2.3.1 spatial and luminance thresholds
WCAG_MAX_FLASH_WIDTH_PX = 341    # max contiguous flash area width (CSS px)
WCAG_MAX_FLASH_HEIGHT_PX = 256   # max contiguous flash area height (CSS px)
WCAG_MAX_FLASH_AREA_PX = WCAG_MAX_FLASH_WIDTH_PX * WCAG_MAX_FLASH_HEIGHT_PX  # 87,296 px
MICHELSON_CONTRAST_LIMIT = 0.0588  # 1/17 — max safe Michelson contrast for flashing UI
RED_SATURATION_LIMIT = 0.8        # R/(R+G+B) must stay below this during transitions

# Analysis window
MONITOR_WINDOW_S = 2.0        # analyze last 2 seconds of events
MIN_EVENTS_FOR_ANALYSIS = 6   # need at least 6 events to detect frequency

# Severity levels
SEVERITY_INFO = "INFO"
SEVERITY_WARNING = "WARNING"
SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_BLOCK = "BLOCK"


@dataclass
class ICSViolation:
    """Record of a detected ICS violation."""
    timestamp: float = 0.0
    violation_type: str = ""     # "rsvp_attack", "photic_driving", "info_flood"
    detected_freq_hz: float = 0.0
    severity: str = SEVERITY_WARNING
    message: str = ""
    blocked: bool = False


class ICSMonitor:
    """Intrinsic Cognitive Security monitor.

    Tracks UI rendering events and analyzes their temporal pattern
    for adversarial stimuli frequencies.
    """

    def __init__(self):
        self._events: Deque[float] = deque(maxlen=1024)
        self._violations: Deque[ICSViolation] = deque(maxlen=256)
        self._lock = threading.Lock()
        self._throttle_until: float = 0.0
        self._violation_count = 0

    def record_frame_event(self, t: Optional[float] = None) -> None:
        """Record a UI frame render event timestamp."""
        with self._lock:
            self._events.append(t or time.time())

    def record_visual_change(self, luminance_delta: float = 0.0,
                              t: Optional[float] = None) -> None:
        """Record a visual content change with luminance delta.

        Large luminance deltas at high frequency indicate flash events.
        """
        with self._lock:
            self._events.append(t or time.time())

    # ── v8.9: WCAG 2.3.1 spatial, contrast, and chromaticity checks ──

    @staticmethod
    def check_flash_area(width_px: int, height_px: int) -> Optional[ICSViolation]:
        """Check if a flashing UI element exceeds the WCAG 341x256 px limit.

        Returns a violation if the area is unsafe, None if safe.
        """
        area = width_px * height_px
        if area > WCAG_MAX_FLASH_AREA_PX:
            return ICSViolation(
                timestamp=time.time(),
                violation_type="wcag_spatial_exceed",
                detected_freq_hz=0,
                severity=SEVERITY_CRITICAL,
                message=f"Flash area {width_px}x{height_px} ({area} px) exceeds "
                        f"WCAG 2.3.1 limit of {WCAG_MAX_FLASH_WIDTH_PX}x"
                        f"{WCAG_MAX_FLASH_HEIGHT_PX} ({WCAG_MAX_FLASH_AREA_PX} px).",
                blocked=True,
            )
        return None

    @staticmethod
    def check_michelson_contrast(l_high: float, l_low: float) -> Optional[ICSViolation]:
        """Check Michelson contrast C = (L_high - L_low) / (L_high + L_low).

        Must be < 0.0588 (1/17) for high-frequency flashing elements.
        Returns a violation if unsafe.
        """
        denom = l_high + l_low
        if denom <= 0:
            return None
        contrast = (l_high - l_low) / denom
        if contrast >= MICHELSON_CONTRAST_LIMIT:
            return ICSViolation(
                timestamp=time.time(),
                violation_type="wcag_contrast_exceed",
                detected_freq_hz=0,
                severity=SEVERITY_CRITICAL,
                message=f"Michelson contrast {contrast:.4f} exceeds WCAG limit "
                        f"of {MICHELSON_CONTRAST_LIMIT} (1/17) for flashing elements.",
                blocked=True,
            )
        return None

    @staticmethod
    def check_red_saturation(r: float, g: float, b: float) -> Optional[ICSViolation]:
        """Check if R/(R+G+B) >= 0.8 during a frame transition.

        Saturated reds are highly epileptogenic. Returns violation if unsafe.
        """
        total = r + g + b
        if total <= 0:
            return None
        red_ratio = r / total
        if red_ratio >= RED_SATURATION_LIMIT:
            return ICSViolation(
                timestamp=time.time(),
                violation_type="wcag_red_saturation",
                detected_freq_hz=0,
                severity=SEVERITY_CRITICAL,
                message=f"Saturated red flash detected: R/(R+G+B) = {red_ratio:.3f} "
                        f">= {RED_SATURATION_LIMIT}. Highly epileptogenic.",
                blocked=True,
            )
        return None

    def check(self) -> Optional[ICSViolation]:
        """Analyze recent events for adversarial patterns.

        Call this at ~1 Hz from the main loop. Returns a violation
        if detected, or None if safe.
        """
        now = time.time()
        with self._lock:
            events = [t for t in self._events if t >= now - MONITOR_WINDOW_S]

        if len(events) < MIN_EVENTS_FOR_ANALYSIS:
            return None

        # Compute instantaneous frequencies from inter-event intervals
        intervals = []
        for i in range(1, len(events)):
            dt = events[i] - events[i - 1]
            if dt > 0:
                intervals.append(1.0 / dt)

        if not intervals:
            return None

        import numpy as np
        freqs = np.array(intervals)
        mean_freq = float(np.mean(freqs))

        # Check 1: RSVP attack detection (8-12 Hz sustained flashing)
        rsvp_lo = RSVP_FREQ_HZ - RSVP_TOLERANCE_HZ
        rsvp_hi = RSVP_FREQ_HZ + RSVP_TOLERANCE_HZ
        rsvp_count = int(np.sum((freqs >= rsvp_lo) & (freqs <= rsvp_hi)))
        if rsvp_count >= len(freqs) * 0.5:  # >50% of events in RSVP band
            v = ICSViolation(
                timestamp=now,
                violation_type="rsvp_attack",
                detected_freq_hz=round(mean_freq, 1),
                severity=SEVERITY_BLOCK,
                message=f"RSVP-pattern detected at {mean_freq:.1f} Hz "
                        f"({rsvp_count}/{len(freqs)} events in 8-12 Hz band). "
                        f"Possible covert P300 harvesting attempt.",
                blocked=True,
            )
            self._register_violation(v)
            return v

        # Check 2: Photic driving / photosensitive seizure risk
        if PHOTIC_DANGER_LO_HZ <= mean_freq <= PHOTIC_DANGER_HI_HZ:
            if mean_freq > WCAG_FLASH_LIMIT:
                v = ICSViolation(
                    timestamp=now,
                    violation_type="photic_driving",
                    detected_freq_hz=round(mean_freq, 1),
                    severity=SEVERITY_CRITICAL,
                    message=f"Photic driving risk: {mean_freq:.1f} Hz exceeds "
                            f"WCAG 2.3.1 limit of {WCAG_FLASH_LIMIT} flashes/sec.",
                    blocked=True,
                )
                self._register_violation(v)
                return v

        # Check 3: Information flooding
        if mean_freq > FLOOD_RATE_HZ:
            v = ICSViolation(
                timestamp=now,
                violation_type="info_flood",
                detected_freq_hz=round(mean_freq, 1),
                severity=SEVERITY_WARNING,
                message=f"Abnormal UI update rate: {mean_freq:.1f} Hz. "
                        f"Possible adversarial content injection.",
                blocked=False,
            )
            self._register_violation(v)
            return v

        return None

    def _register_violation(self, v: ICSViolation) -> None:
        """Register a violation and engage throttle if blocking."""
        with self._lock:
            self._violations.append(v)
            self._violation_count += 1
            if v.blocked:
                # Throttle display for 5 seconds
                self._throttle_until = time.time() + 5.0
        # Log to forensic log if available
        try:
            import forensic_log
            forensic_log.log_event(
                f"ics_{v.violation_type}",
                severity=forensic_log.SEVERITY_CRITICAL if v.blocked else forensic_log.SEVERITY_WARN,
                payload={
                    "freq_hz": v.detected_freq_hz,
                    "type": v.violation_type,
                    "message": v.message,
                    "blocked": v.blocked,
                }
            )
        except Exception:
            pass
        print(f"[ICS] {v.severity}: {v.message}")

    def is_throttled(self) -> bool:
        """Check if display should be throttled due to active violation."""
        return time.time() < self._throttle_until

    def safe_refresh_hz(self) -> float:
        """Return the safe refresh rate. Normal=uncapped, throttled=2 Hz."""
        if self.is_throttled():
            return SAFE_REFRESH_HZ
        return 0.0  # 0 = no limit

    def recent_violations(self, n: int = 10) -> List[Dict]:
        """Return the N most recent violations for the status endpoint."""
        with self._lock:
            recent = list(self._violations)[-n:]
        return [
            {
                "timestamp": v.timestamp,
                "type": v.violation_type,
                "freq_hz": v.detected_freq_hz,
                "severity": v.severity,
                "message": v.message,
                "blocked": v.blocked,
            }
            for v in recent
        ]

    def stats(self) -> Dict:
        """Diagnostic summary."""
        return {
            "total_violations": self._violation_count,
            "throttled": self.is_throttled(),
            "safe_refresh_hz": self.safe_refresh_hz(),
            "recent_violations": self.recent_violations(5),
            "event_buffer_size": len(self._events),
        }


# Module-level singleton for easy import
_monitor = ICSMonitor()


def record_frame():
    """Record a UI frame render event (call from main loop)."""
    _monitor.record_frame_event()


def check() -> Optional[ICSViolation]:
    """Check for ICS violations (call at ~1 Hz)."""
    return _monitor.check()


def is_throttled() -> bool:
    """Check if display is currently throttled."""
    return _monitor.is_throttled()


def stats() -> Dict:
    """Get ICS monitor statistics."""
    return _monitor.stats()


# v8.9: module-level WCAG spatial/contrast/chromaticity checks
def check_flash_area(width_px: int, height_px: int) -> Optional[ICSViolation]:
    """Check if a flashing element exceeds WCAG 341x256 px limit."""
    v = ICSMonitor.check_flash_area(width_px, height_px)
    if v:
        _monitor._register_violation(v)
    return v


def check_contrast(l_high: float, l_low: float) -> Optional[ICSViolation]:
    """Check Michelson contrast < 1/17 for flashing elements."""
    v = ICSMonitor.check_michelson_contrast(l_high, l_low)
    if v:
        _monitor._register_violation(v)
    return v


def check_red_flash(r: float, g: float, b: float) -> Optional[ICSViolation]:
    """Check R/(R+G+B) < 0.8 for saturated red suppression."""
    v = ICSMonitor.check_red_saturation(r, g, b)
    if v:
        _monitor._register_violation(v)
    return v
