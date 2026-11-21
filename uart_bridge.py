"""
DODO VISION AI v8.9 — USB UART Bridge for Microcontroller-Buffered Oximetry (Steps 10-11)

Isolates the MAX30102 I2C polling from the Python runtime by communicating
with a dedicated microcontroller (Arduino Nano / ESP32-C3) that:
  1. Reads the MAX30102 FIFO via hardware I2C at 400kHz (Step 10)
  2. Aggregates samples into timestamped packets
  3. Streams packets to this module over USB UART at 460,800 bps (Step 11)

This eliminates the I2C buffer overrun that crashes the Python backend when
the Global Interpreter Lock (GIL) blocks the polling thread for >320ms (the
time it takes the 32-sample FIFO at 100 SPS to fill).

Packet protocol (from MCU):
  STX (0x02) | LEN (1 byte) | TYPE (1 byte) | PAYLOAD (LEN bytes) | CRC8 | ETX (0x03)

  TYPE 0x01: SpO2 + HR sample
    PAYLOAD: hr_bpm(uint16) | spo2(uint8) | signal_quality(uint8) | ir_ac(uint16) | timestamp_ms(uint32)

  TYPE 0x02: FIFO batch (raw IR+RED samples for custom processing)
    PAYLOAD: n_samples(uint8) | [ir(uint32) | red(uint32)] * n_samples

  TYPE 0xFF: MCU status/heartbeat
    PAYLOAD: uptime_ms(uint32) | fifo_overruns(uint16) | i2c_errors(uint16)

References:
  - MAX30102 datasheet, Maxim Integrated (FIFO architecture, I2C Fast Mode)
  - USB CDC ACM class for virtual COM port
"""
from __future__ import annotations
import struct
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

import config as cfg

# Packet framing constants
STX = 0x02
ETX = 0x03
TYPE_SPO2_HR = 0x01
TYPE_FIFO_BATCH = 0x02
TYPE_MCU_STATUS = 0xFF

# CRC-8 (polynomial 0x07, init 0x00) — lightweight for MCU
def _crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


class UartBridge:
    """Asynchronous USB UART bridge reader for microcontroller-buffered oximetry.

    Runs a background thread that reads framed packets from the MCU,
    parses them, and stores the latest values in a thread-safe buffer.
    The main oximeter module reads from this buffer instead of doing
    direct I2C/HID communication.
    """

    def __init__(self, port: str = "auto",
                 baud: int = None,
                 timeout: float = 0.5):
        self._port = port
        self._baud = baud or getattr(cfg, "OXIMETER_UART_BRIDGE_BAUD", 460800)
        self._timeout = timeout
        self._serial = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

        # Latest parsed data
        self._hr = None
        self._spo2 = None
        self._signal_quality = 0
        self._timestamp = 0.0
        self._connected = False
        self._rx_count = 0
        self._errors = 0

        # MCU status
        self._mcu_uptime_ms = 0
        self._mcu_fifo_overruns = 0
        self._mcu_i2c_errors = 0

        # Raw FIFO buffer for advanced processing
        self._fifo_buffer: deque = deque(maxlen=256)

    def start(self) -> bool:
        """Open the serial port and start the reader thread."""
        port = self._resolve_port()
        if port is None:
            print("[UART-BRIDGE] No MCU bridge port found")
            return False
        try:
            import serial
            self._serial = serial.Serial(
                port=port, baudrate=self._baud,
                timeout=self._timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            self._connected = True
            self._stop.clear()
            self._thread = threading.Thread(target=self._reader_loop, daemon=True,
                                           name="uart_bridge")
            self._thread.start()
            print(f"[UART-BRIDGE] Connected to {port} @ {self._baud} bps")
            return True
        except Exception as e:
            print(f"[UART-BRIDGE] Failed to open {port}: {e}")
            return False

    def stop(self) -> None:
        """Stop the reader thread and close the port."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial:
            try:
                self._serial.close()
            except Exception:
                pass
        self._connected = False

    def _resolve_port(self) -> Optional[str]:
        """Auto-detect the MCU bridge port, or use configured port."""
        if self._port != "auto":
            return self._port
        try:
            import serial.tools.list_ports
            for p in serial.tools.list_ports.comports():
                desc = (p.description or "").lower()
                # Common MCU bridge descriptions
                if any(kw in desc for kw in ["ch340", "cp210", "ftdi", "arduino", "esp32"]):
                    return p.device
        except Exception:
            pass
        return None

    def _reader_loop(self) -> None:
        """Background thread: read framed packets from serial."""
        buf = bytearray()
        while not self._stop.is_set():
            try:
                if not self._serial or not self._serial.is_open:
                    self._stop.wait(1.0)
                    continue
                chunk = self._serial.read(256)
                if not chunk:
                    continue
                buf.extend(chunk)
                # Parse all complete packets in buffer
                while len(buf) >= 5:  # minimum: STX + LEN + TYPE + CRC + ETX
                    # Find STX
                    stx_idx = buf.find(bytes([STX]))
                    if stx_idx < 0:
                        buf.clear()
                        break
                    if stx_idx > 0:
                        buf = buf[stx_idx:]  # discard garbage before STX
                    if len(buf) < 4:
                        break
                    pkt_len = buf[1]
                    total = 2 + 1 + pkt_len + 1 + 1  # STX+LEN + TYPE + PAYLOAD + CRC + ETX
                    if len(buf) < total:
                        break  # incomplete packet
                    if buf[total - 1] != ETX:
                        buf = buf[1:]  # bad frame, skip STX
                        self._errors += 1
                        continue
                    pkt_type = buf[2]
                    payload = bytes(buf[3:3 + pkt_len])
                    crc_recv = buf[3 + pkt_len]
                    crc_calc = _crc8(bytes([pkt_len, pkt_type]) + payload)
                    if crc_recv != crc_calc:
                        buf = buf[total:]
                        self._errors += 1
                        continue
                    # Valid packet
                    self._parse_packet(pkt_type, payload)
                    self._rx_count += 1
                    buf = buf[total:]
            except Exception:
                self._errors += 1
                self._stop.wait(0.1)

    def _parse_packet(self, pkt_type: int, payload: bytes) -> None:
        """Parse a validated packet payload.

        v9.0 Step 6 (MIT Hardware Timestamping): The MCU stamps each sample
        with millisecond-precision timestamps BEFORE UART transmission.  This
        eliminates OS-level network jitter from the time-series arrays used
        for HRV analysis.  The host-side arrival time is recorded separately
        for drift monitoring.
        """
        with self._lock:
            if pkt_type == TYPE_SPO2_HR and len(payload) >= 10:
                hr = struct.unpack("<H", payload[0:2])[0]
                spo2 = payload[2]
                sig_q = payload[3]
                mcu_ts_ms = struct.unpack("<I", payload[6:10])[0]
                self._hr = hr if hr > 0 else None
                self._spo2 = spo2 if spo2 > 0 else None
                self._signal_quality = sig_q
                host_arrival = time.time()
                self._timestamp = host_arrival

                # v9.0 Step 6: hardware timestamp alignment
                # Store MCU timestamp for jitter-free HRV analysis.
                # The MCU clock is monotonic ms since boot; we align it to
                # host wall-clock on the first sample, then track drift.
                self._last_mcu_ts_ms = mcu_ts_ms
                if not hasattr(self, "_mcu_epoch_offset"):
                    # First sample: establish epoch mapping
                    self._mcu_epoch_offset = host_arrival - (mcu_ts_ms / 1000.0)
                    self._hw_timestamps = []
                # Convert MCU ms to host-aligned float seconds
                hw_ts = self._mcu_epoch_offset + (mcu_ts_ms / 1000.0)
                if not hasattr(self, "_hw_timestamps"):
                    self._hw_timestamps = []
                self._hw_timestamps.append(hw_ts)
                # Keep only last 300 timestamps for HRV
                if len(self._hw_timestamps) > 300:
                    self._hw_timestamps = self._hw_timestamps[-300:]

                # Drift detection: warn if MCU-vs-host diverges
                drift_ms = abs(hw_ts - host_arrival) * 1000
                drift_warn = getattr(cfg, "HW_TIMESTAMP_DRIFT_WARN_MS", 50)
                if drift_ms > drift_warn and self._rx_count > 10:
                    if self._rx_count % 100 == 0:  # rate-limit warnings
                        print(f"[UART-BRIDGE] Clock drift: {drift_ms:.1f}ms "
                              f"(MCU={mcu_ts_ms}ms, host={host_arrival:.3f}s)")

                # v9.0 Step 5: ToF seat validation — parse proximity if present
                if len(payload) >= 12:
                    tof_mm = struct.unpack("<H", payload[10:12])[0]
                    self._tof_distance_mm = tof_mm / 10.0  # 0.1mm resolution
                    try:
                        from oximeter import update_tof_proximity
                        update_tof_proximity(self._tof_distance_mm)
                    except ImportError:
                        pass

            elif pkt_type == TYPE_FIFO_BATCH and len(payload) >= 1:
                n = payload[0]
                offset = 1
                for _ in range(n):
                    if offset + 8 > len(payload):
                        break
                    ir = struct.unpack("<I", payload[offset:offset+4])[0]
                    red = struct.unpack("<I", payload[offset+4:offset+8])[0]
                    self._fifo_buffer.append((ir, red, time.time()))
                    offset += 8

            elif pkt_type == TYPE_MCU_STATUS and len(payload) >= 8:
                self._mcu_uptime_ms = struct.unpack("<I", payload[0:4])[0]
                self._mcu_fifo_overruns = struct.unpack("<H", payload[4:6])[0]
                self._mcu_i2c_errors = struct.unpack("<H", payload[6:8])[0]

    def get_data(self) -> Optional[Dict[str, Any]]:
        """Get the latest oximeter reading from the MCU bridge.

        v9.0 Step 6: includes hardware timestamp for jitter-free HRV.

        Returns a dict compatible with oximeter.get_oximeter_data() API,
        or None if no valid data.
        """
        with self._lock:
            if self._hr is None and self._spo2 is None:
                return None
            age = time.time() - self._timestamp if self._timestamp > 0 else float("inf")
            if age > 5.0:
                return None  # stale
            result = {
                "heart_rate": self._hr,
                "spo2": self._spo2,
                "signal_strength": self._signal_quality,
                "source": "uart_bridge",
                "age_s": round(age, 2),
            }
            # v9.0 Step 6: attach hardware timestamp and ToF distance
            if hasattr(self, "_last_mcu_ts_ms"):
                result["hw_timestamp_ms"] = self._last_mcu_ts_ms
            if hasattr(self, "_tof_distance_mm"):
                result["tof_distance_mm"] = self._tof_distance_mm
            return result

    def get_hw_timestamps(self, n: int = 60) -> list:
        """Get last N hardware-aligned timestamps for HRV analysis.

        v9.0 Step 6: These timestamps are derived from the MCU's
        millisecond-precision clock, aligned to host wall-clock on first
        sample.  They are free of OS-level network jitter that would
        destroy HRV RMSSD accuracy.
        """
        with self._lock:
            ts_list = getattr(self, "_hw_timestamps", [])
            return list(ts_list[-n:])

    def get_raw_fifo(self, n: int = 32) -> list:
        """Get the last N raw IR/RED samples from the FIFO buffer."""
        with self._lock:
            return list(self._fifo_buffer)[-n:]

    def status(self) -> Dict[str, Any]:
        """Diagnostic status for /status endpoint."""
        with self._lock:
            return {
                "connected": self._connected,
                "port": self._port,
                "baud": self._baud,
                "rx_packets": self._rx_count,
                "parse_errors": self._errors,
                "mcu_uptime_s": round(self._mcu_uptime_ms / 1000, 1),
                "mcu_fifo_overruns": self._mcu_fifo_overruns,
                "mcu_i2c_errors": self._mcu_i2c_errors,
                "last_hr": self._hr,
                "last_spo2": self._spo2,
            }


# ══════════════════════════════════════════════
#  Module-level singleton
# ══════════════════════════════════════════════
_bridge: Optional[UartBridge] = None


def get_bridge() -> Optional[UartBridge]:
    """Get the UART bridge singleton. Returns None if not enabled."""
    global _bridge
    if not getattr(cfg, "OXIMETER_UART_BRIDGE_ENABLED", False):
        return None
    if _bridge is None:
        _bridge = UartBridge()
        if not _bridge.start():
            _bridge = None
    return _bridge


def get_bridge_data() -> Optional[Dict]:
    """Convenience: get oximeter data from the UART bridge, or None."""
    b = get_bridge()
    return b.get_data() if b else None
