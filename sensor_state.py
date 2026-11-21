"""
DODO VISION AI v8.9 -- Isolated Sensor State Buffer (Steps 1-2)

Decoupled, thread-safe state buffer for all external sensors. Each sensor
writes to its own isolated state machine via background threads; the frame
processor reads a unified snapshot at fixed frequency.

Architecture (CMU context-sensing + MIT Dempster-Shafer):
  - Each sensor has an independent state slot with timestamp + confidence
  - A single-sensor latency spike cannot corrupt other slots
  - The WS handler reads a frozen snapshot, never awaits sensor I/O directly
  - Dempster-Shafer confidence weights allow partial-state broadcast:
    timed-out sensors get weight=0 instead of raising UnboundLocalError
  - Zero-Order Hold: stale data is held with decaying confidence

References:
  - CMU context-sensing architecture (Dey & Abowd)
  - MIT Media Lab Dempster-Shafer evidence theory for sensor fusion
  - Stanford robust georeferenced fallback architecture
"""
from __future__ import annotations
import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional


# ── Staleness thresholds (seconds) ──
RADAR_STALE_S = 2.0
OXIMETER_STALE_S = 5.0
GSR_STALE_S = 3.0
DEFAULT_STALE_S = 5.0

# ── Dempster-Shafer confidence decay ──
# After data goes stale, confidence decays exponentially:
#   w(t) = w_0 * exp(-decay_rate * (t - t_stale))
DS_DECAY_RATE = 0.5  # halves every ~1.4 seconds after going stale


@dataclass
class SensorSlot:
    """Isolated state machine for a single sensor modality."""
    name: str = ""
    data: Optional[Dict] = None
    timestamp: float = 0.0
    confidence: float = 0.0       # Dempster-Shafer mass / belief weight
    stale_threshold_s: float = DEFAULT_STALE_S
    update_count: int = 0
    error: Optional[str] = None

    @property
    def age_s(self) -> float:
        """Seconds since last update."""
        if self.timestamp <= 0:
            return float("inf")
        return time.time() - self.timestamp

    @property
    def is_stale(self) -> bool:
        return self.age_s > self.stale_threshold_s

    @property
    def effective_confidence(self) -> float:
        """Confidence with age-decay applied (Dempster-Shafer weight).

        Fresh data keeps its original confidence. Once stale, confidence
        decays exponentially so downstream fusion naturally down-weights it.
        """
        if not self.is_stale:
            return self.confidence
        overage = self.age_s - self.stale_threshold_s
        return self.confidence * math.exp(-DS_DECAY_RATE * overage)

    def snapshot(self) -> Dict:
        """Return a frozen copy of this slot for the frame processor."""
        return {
            "data": dict(self.data) if self.data else None,
            "confidence": round(self.effective_confidence, 3),
            "age_s": round(self.age_s, 2),
            "stale": self.is_stale,
            "updates": self.update_count,
            "error": self.error,
        }


class SensorStateBuffer:
    """Thread-safe, decoupled sensor state buffer.

    Background sensor threads call push() to update their slot.
    The frame processor calls snapshot() to get a consistent, frozen
    view of all sensor states without ever blocking on sensor I/O.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._slots: Dict[str, SensorSlot] = {
            "radar_a": SensorSlot(name="radar_a", stale_threshold_s=RADAR_STALE_S),
            "radar_b": SensorSlot(name="radar_b", stale_threshold_s=RADAR_STALE_S),
            "oximeter": SensorSlot(name="oximeter", stale_threshold_s=OXIMETER_STALE_S),
            "gsr": SensorSlot(name="gsr", stale_threshold_s=GSR_STALE_S),
        }

    def push(self, sensor_name: str, data: Optional[Dict],
             confidence: float = 1.0, error: Optional[str] = None) -> None:
        """Push new data from a sensor thread. Thread-safe.

        Args:
            sensor_name: one of "radar_a", "radar_b", "oximeter", "gsr"
            data: the sensor reading dict, or None if disconnected
            confidence: belief mass [0, 1] for Dempster-Shafer weighting
            error: optional error string if sensor is faulted
        """
        with self._lock:
            if sensor_name not in self._slots:
                self._slots[sensor_name] = SensorSlot(
                    name=sensor_name,
                    stale_threshold_s=DEFAULT_STALE_S,
                )
            slot = self._slots[sensor_name]
            slot.data = data
            slot.timestamp = time.time()
            slot.confidence = max(0.0, min(1.0, confidence))
            slot.update_count += 1
            slot.error = error

    def get(self, sensor_name: str) -> Optional[Dict]:
        """Get data from a sensor slot (None if no data or stale).

        This is the safe replacement for direct sensor.get_*_data() calls.
        Returns the raw data dict or None, matching the old API contract.
        """
        with self._lock:
            slot = self._slots.get(sensor_name)
            if slot is None or slot.data is None:
                return None
            if slot.is_stale:
                return None  # Zero-Order Hold expired
            return dict(slot.data)

    def get_with_confidence(self, sensor_name: str) -> tuple:
        """Get (data, confidence) tuple for Dempster-Shafer fusion.

        Always returns a tuple — never raises. If sensor is missing,
        returns (None, 0.0).
        """
        with self._lock:
            slot = self._slots.get(sensor_name)
            if slot is None or slot.data is None:
                return (None, 0.0)
            return (dict(slot.data), slot.effective_confidence)

    def snapshot(self) -> Dict[str, Dict]:
        """Frozen snapshot of ALL sensor states for the frame processor.

        Returns a dict of {sensor_name: slot_snapshot}.
        """
        with self._lock:
            return {name: slot.snapshot() for name, slot in self._slots.items()}

    def stats(self) -> Dict:
        """Diagnostic summary for /status endpoint."""
        with self._lock:
            return {
                name: {
                    "connected": slot.data is not None and not slot.is_stale,
                    "confidence": round(slot.effective_confidence, 3),
                    "age_s": round(slot.age_s, 2),
                    "updates": slot.update_count,
                    "error": slot.error,
                }
                for name, slot in self._slots.items()
            }


# ══════════════════════════════════════════════
#  Dempster-Shafer Fusion Helpers
# ══════════════════════════════════════════════

def ds_combine_hr(*sources: tuple) -> Optional[float]:
    """Combine heart rate estimates using Dempster-Shafer belief fusion.

    Each source is (hr_value, confidence). Returns the weighted
    fusion or None if no valid sources.

    This is a simplified Dempster's rule of combination for scalar
    estimates: weighted average by belief mass, with conflict
    normalization.
    """
    valid = [(hr, conf) for hr, conf in sources
             if hr is not None and conf > 0.01]
    if not valid:
        return None
    total_conf = sum(c for _, c in valid)
    if total_conf <= 0:
        return None
    fused = sum(hr * c for hr, c in valid) / total_conf
    return round(fused, 1)


def ds_select_best(*sources: tuple) -> tuple:
    """Select the highest-confidence source. Returns (value, confidence, index)."""
    best_val, best_conf, best_idx = None, 0.0, -1
    for i, (val, conf) in enumerate(sources):
        if val is not None and conf > best_conf:
            best_val, best_conf, best_idx = val, conf, i
    return (best_val, best_conf, best_idx)


# ══════════════════════════════════════════════
#  v9.0 Step 2: Zero-Order Hold Mediator with Age-Decay Interpolation (MIT)
# ══════════════════════════════════════════════

class ZOHMediator:
    """Continuous five-dimensional affect sensing mediator on a rigid tick.

    Operates at a fixed frequency (default 50 Hz = 20 ms window). If any
    sensor fails to return data within a tick, the mediator executes a
    Zero-Order Hold — it extrapolates the last known valid state while
    mathematically decaying its confidence over time using an age-decay
    metric.  This guarantees continuous downstream data flow without
    blocking the event loop.

    MIT Framework reference: robust autonomous sensor fusion with
    guaranteed delivery at fixed cadence.
    """

    def __init__(self, buffer: SensorStateBuffer, tick_hz: int = 50,
                 decay_rate: float = DS_DECAY_RATE):
        self._buffer = buffer
        self._tick_hz = tick_hz
        self._tick_interval = 1.0 / tick_hz  # 20 ms at 50 Hz
        self._decay_rate = decay_rate
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: "threading.Thread | None" = None

        # ZOH state: last valid reading per sensor with hold timestamp
        self._zoh_state: Dict[str, Dict] = {}
        self._tick_count = 0
        self._stale_broadcasts = 0
        self._fresh_broadcasts = 0

    def start(self) -> None:
        """Start the 50 Hz mediator tick in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._tick_loop, daemon=True, name="zoh_mediator")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _tick_loop(self) -> None:
        """Rigid tick at self._tick_hz.  Each tick snapshots all sensors.

        If a sensor is stale, the ZOH holds the last known value with
        exponentially decaying confidence:
          confidence(t) = c0 * exp(-decay_rate * (t - t_stale))
        """
        while not self._stop.is_set():
            t_start = time.time()
            snap = self._buffer.snapshot()
            with self._lock:
                for name, slot_data in snap.items():
                    if slot_data["data"] is not None and not slot_data["stale"]:
                        # Fresh data — update ZOH state
                        self._zoh_state[name] = {
                            "data": slot_data["data"],
                            "original_confidence": slot_data["confidence"],
                            "confidence": slot_data["confidence"],
                            "captured_at": time.time(),
                            "hold_active": False,
                        }
                        self._fresh_broadcasts += 1
                    else:
                        # Missing or stale — execute Zero-Order Hold
                        held = self._zoh_state.get(name)
                        if held and held["data"] is not None:
                            age = time.time() - held["captured_at"]
                            # BUG FIX: always decay from the ORIGINAL captured
                            # confidence, not the already-decayed value. The old
                            # code applied exp(-rate * total_age) to the running
                            # held["confidence"], causing double-exponential
                            # collapse (confidence → 0 in seconds instead of
                            # the intended gradual decay).
                            c0 = held.get("original_confidence", held["confidence"])
                            decayed_conf = c0 * math.exp(
                                -self._decay_rate * age)
                            held["confidence"] = max(0.0, decayed_conf)
                            held["hold_active"] = True
                            self._stale_broadcasts += 1
                        # If no prior data exists, ZOH emits None with conf=0
                self._tick_count += 1

            # Sleep to maintain rigid tick rate
            elapsed = time.time() - t_start
            remaining = self._tick_interval - elapsed
            if remaining > 0:
                self._stop.wait(remaining)

    def get_held(self, sensor_name: str) -> tuple:
        """Get ZOH-held (data, confidence, hold_active) for a sensor.

        Returns the last valid reading with age-decayed confidence.
        Always returns a tuple — never raises.
        """
        with self._lock:
            held = self._zoh_state.get(sensor_name)
            if held is None:
                return (None, 0.0, False)
            return (held["data"], held["confidence"], held["hold_active"])

    def get_fused_snapshot(self) -> Dict[str, Dict]:
        """Snapshot of ALL sensors via ZOH — guaranteed non-None for known sensors."""
        with self._lock:
            result = {}
            for name, held in self._zoh_state.items():
                result[name] = {
                    "data": held["data"],
                    "confidence": round(held["confidence"], 4),
                    "hold_active": held["hold_active"],
                    "age_s": round(time.time() - held["captured_at"], 3)
                        if held.get("captured_at") else None,
                }
            return result

    def stats(self) -> Dict:
        """Diagnostic summary."""
        with self._lock:
            return {
                "tick_hz": self._tick_hz,
                "ticks": self._tick_count,
                "fresh_broadcasts": self._fresh_broadcasts,
                "stale_zoh_broadcasts": self._stale_broadcasts,
                "held_sensors": list(self._zoh_state.keys()),
                "decay_rate": self._decay_rate,
            }


# ══════════════════════════════════════════════
#  Module-level singletons
# ══════════════════════════════════════════════
_buffer = SensorStateBuffer()

# v9.0 Step 2: ZOH mediator running at 50 Hz alongside the buffer
_mediator: "ZOHMediator | None" = None


def get_buffer() -> SensorStateBuffer:
    """Get the process-wide sensor state buffer."""
    return _buffer


def get_mediator() -> ZOHMediator:
    """Get the process-wide ZOH mediator (lazy start)."""
    global _mediator
    if _mediator is None:
        _mediator = ZOHMediator(_buffer)
        _mediator.start()
    return _mediator
