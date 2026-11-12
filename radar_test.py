"""
DODO VISION AI — Radar Bridge Diagnostic
Usage:
    python radar_test.py              # auto-detect port, JSON mode
    python radar_test.py COM5         # force port
    python radar_test.py COM5 raw     # raw hex passthrough

Prints the first 60 seconds of output from the radar bridge so you can
see exactly what (if anything) the ESP32C6 is emitting.
"""
import sys, time, json

def main():
    try:
        import serial, serial.tools.list_ports
    except ImportError:
        print("pip install pyserial --break-system-packages")
        sys.exit(1)

    port = sys.argv[1] if len(sys.argv) > 1 else None
    mode = sys.argv[2] if len(sys.argv) > 2 else "json"

    if not port:
        kws = ("CH340", "CP210", "ESP32", "XIAO", "Seeed", "USB Serial", "USB-SERIAL", "CH9102")
        cands = []
        for p in serial.tools.list_ports.comports():
            desc = (p.description or "") + " " + (p.manufacturer or "")
            if any(k.lower() in desc.lower() for k in kws):
                cands.append(p.device)
        if not cands:
            cands = [p.device for p in serial.tools.list_ports.comports()]
        if not cands:
            print("No serial ports found. Plug in the ESP32C6.")
            sys.exit(2)
        port = cands[0]
        print(f"[i] Auto-selected port: {port}")

    print(f"[i] Opening {port} @115200 ({mode} mode)")
    ser = serial.Serial(port, 115200, timeout=1)
    time.sleep(0.2)

    t_end = time.time() + 60
    lines, valid_json, metrics = 0, 0, 0
    last_hr = last_br = None
    buf = b""
    while time.time() < t_end:
        buf += ser.read(ser.in_waiting or 1)
        if mode == "raw":
            if buf:
                print(" ".join(f"{b:02X}" for b in buf), end=" ", flush=True)
                buf = b""
            continue
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            lines += 1
            try:
                d = json.loads(line.decode("utf-8", errors="ignore"))
                valid_json += 1
                if "hr" in d or "br" in d:
                    metrics += 1
                    last_hr = d.get("hr", last_hr)
                    last_br = d.get("br", last_br)
                    print(d)
                else:
                    print("[status]", d)
            except Exception:
                print("[non-json]", line[:120])
    print("\n─── SUMMARY ───")
    print(f"Total lines: {lines}")
    print(f"Valid JSON : {valid_json}")
    print(f"With hr/br : {metrics}")
    print(f"Last HR    : {last_hr}")
    print(f"Last BR    : {last_br}")
    if lines == 0:
        print("→ Bridge silent. Check: USB CDC On Boot enabled? Correct COM port (the ESP32C6 side, not the radar side)? Sketch actually uploaded?")
    elif valid_json == 0:
        print("→ Bridge talking but not JSON. Probably the old sketch or the hex-passthrough fallback. Reflash radar_sketch.ino v7.1 with the Seeed_Arduino_mmWave library installed.")
    elif metrics == 0:
        print("→ Bridge emitting JSON but no hr/br. Library present but radar UART not wired — verify D6/D7 and that the radar module has power (red LED).")
    else:
        print("→ Bridge OK. If the UI still shows nothing, check RADAR_PORT in config.py and that nothing else (Arduino Serial Monitor) is holding the port.")

if __name__ == "__main__":
    main()
