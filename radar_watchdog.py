"""
DODO VISION AI v7.2 — Radar watchdog + auto-reconnect

Monitors radar.summary() and transparently restarts the reader if:
  - readings have gone stale (> STALE_SECS)
  - the COM port disappeared and reappeared
  - an unhandled error was recorded in radar_data["error"]

Usage
-----
    import radar_watchdog
    radar_watchdog.start()     # call once after radar.start() in main.py
    radar_watchdog.stop()      # on app shutdown
"""
from __future__ import annotations
import threading, time

STALE_SECS = 8.0
BACKOFF_SECS = 3.0
MAX_BACKOFF = 30.0

_stop = threading.Event()
_thread = None
_last_port = None


def _loop():
    global _last_port
    import radar
    backoff = BACKOFF_SECS
    _last_perm_err_logged = 0  # v8.3: throttle PermissionError log spam
    while not _stop.is_set():
        try:
            snap = radar.summary()
            age = snap.get("age_s")
            connected = snap.get("connected")
            port = snap.get("port")
            err = snap.get("error") or ""
            if port:
                _last_port = port

            needs_reconnect = (
                (not connected) or
                (age is not None and age > STALE_SECS) or
                bool(err)
            )

            if needs_reconnect:
                # v8.3: exponential backoff on PermissionError (errno 13)
                # to avoid flooding the console every 2 seconds when another
                # process holds the COM port.
                is_perm_err = "PermissionError" in str(err) or "Access is denied" in str(err)
                if is_perm_err:
                    now = time.time()
                    if now - _last_perm_err_logged > backoff:
                        print(f"[radar-watchdog] COM port held by another process (backoff {backoff:.0f}s)")
                        _last_perm_err_logged = now
                    backoff = min(MAX_BACKOFF, backoff * 2)
                    _stop.wait(timeout=backoff)
                    continue

                print(f"[radar-watchdog] reconnecting (age={age}s, err={err})")
                try:
                    radar.stop()
                except Exception:
                    pass
                time.sleep(0.5)
                ok = radar.start()
                if not ok:
                    backoff = min(MAX_BACKOFF, backoff * 1.5)
                    _stop.wait(timeout=backoff)
                    continue
                backoff = BACKOFF_SECS
                _last_perm_err_logged = 0
        except Exception as e:
            print(f"[radar-watchdog] exception: {e}")
        _stop.wait(timeout=2.0)


def start():
    global _thread
    if _thread and _thread.is_alive():
        return False
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="radar-watchdog", daemon=True)
    _thread.start()
    print("[OK] Radar watchdog started")
    return True


def stop():
    _stop.set()
