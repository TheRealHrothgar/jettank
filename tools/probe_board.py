#!/usr/bin/env python3
"""Listen to the Yahboom expansion board. READ ONLY - never transmits.

Why this exists, and why it only listens:

The last time we sent frames to this board the treads ran away. Two causes were
identified. First, instantiating Rosmaster_Lib is itself unsafe: its __init__
sends set_uart_servo_torque(1) at device id 0xFC before any override can apply.
Second, the board streams FUNC 0x08, which that library has no case for - so
its function numbering may not match this firmware at all, which means the
"emergency stop" frames we were sending (0x10/0x11/0x12) were themselves
guesses.

You cannot safely guess a motor command. You can safely listen. The board
free-runs a telemetry stream, so everything needed to confirm framing - header
bytes, device id, checksum rule, which functions this firmware actually uses -
arrives without transmitting a single byte.

    python3 tools/probe_board.py --port /dev/ttyTHS1 --seconds 20

This module opens the serial port read-only and has no write path at all. That
is deliberate: there is nothing here to accidentally call.
"""
from __future__ import annotations

import argparse
import collections
import sys
import time

# What the vendor library believes, for comparison against what we observe.
VENDOR_DEVICE_ID = 0xFC
VENDOR_FUNCS = {
    0x01: "AUTO_REPORT", 0x02: "BEEP", 0x03: "PWM_SERVO", 0x04: "PWM_SERVO_ALL",
    0x05: "RGB", 0x06: "RGB_EFFECT", 0x0A: "REPORT_SPEED", 0x0B: "REPORT_MPU_RAW",
    0x0C: "REPORT_IMU_ATT", 0x0D: "REPORT_ENCODER", 0x0E: "REPORT_ICM_RAW",
    0x0F: "RESET_STATE", 0x10: "MOTOR", 0x11: "CAR_RUN", 0x12: "MOTION",
    0x13: "SET_MOTOR_PID", 0x14: "SET_YAW_PID", 0x15: "SET_CAR_TYPE",
    0x20: "UART_SERVO", 0x21: "UART_SERVO_ID", 0x22: "UART_SERVO_TORQUE",
    0x23: "ARM_CTRL", 0x51: "VERSION",
}


def checksum_ok(device_id: int, body: bytes, given: int) -> bool:
    """Yahboom checksum: sum(body) + (257 - device_id), low byte."""
    return (sum(body) + (257 - device_id)) & 0xFF == given


def parse(buf: bytearray) -> tuple[list[dict], bytearray]:
    """Pull complete frames out of buf. Returns (frames, remainder).

    Frame shape, as far as we know it:
        0xFF <device_id> <len> <func> <payload...> <checksum>
    where <len> counts from <len> itself through the checksum.
    """
    frames: list[dict] = []
    i = 0
    while i < len(buf):
        if buf[i] != 0xFF:
            i += 1
            continue
        if len(buf) - i < 4:
            break                                   # need more bytes
        device_id, length = buf[i + 1], buf[i + 2]
        if not 3 <= length <= 64:
            i += 1                                  # not a plausible frame
            continue
        end = i + 2 + length
        if end > len(buf):
            break
        body = bytes(buf[i + 2:end - 1])            # len .. last payload byte
        frame = {
            "raw": bytes(buf[i:end]),
            "device_id": device_id,
            "func": buf[i + 3],
            "payload": bytes(buf[i + 4:end - 1]),
            "checksum": buf[end - 1],
        }
        frame["checksum_ok"] = checksum_ok(device_id, body, frame["checksum"])
        frames.append(frame)
        i = end
    return frames, buf[i:]


def describe_payload(func: int, p: bytes) -> str:
    """Best-effort reading of a payload, clearly labelled as inference."""
    if not p:
        return ""
    bits = [f"{len(p)}B"]
    if len(p) >= 2:
        # Signed 16-bit big-endian pairs, the usual shape for this family.
        vals = [int.from_bytes(p[i:i + 2], "big", signed=True)
                for i in range(0, len(p) - 1, 2)]
        bits.append("i16: " + " ".join(str(v) for v in vals[:9]))
        # An accelerometer at rest reads about 1g on one axis. In this family
        # that lands near 0x4000-0x4200, which is a useful fingerprint.
        if any(0x3E00 <= abs(v) <= 0x4400 for v in vals):
            bits.append("<- one axis near 1g, looks like an IMU")
    return "  ".join(bits)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Passively observe the Yahboom board")
    ap.add_argument("--port", default="/dev/ttyTHS1")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--show", type=int, default=8, help="example frames per function")
    args = ap.parse_args(argv)

    try:
        import serial
    except ImportError:
        print("pyserial is not installed:  pip install pyserial", file=sys.stderr)
        return 2

    print(f"listening on {args.port} @ {args.baud} for {args.seconds:.0f}s "
          f"- READ ONLY, nothing is transmitted\n")
    try:
        # No writing anywhere in this program; the port is only ever read.
        ser = serial.Serial(args.port, args.baud, timeout=0.2)
    except Exception as exc:                        # noqa: BLE001
        print(f"could not open {args.port}: {exc}", file=sys.stderr)
        print("\nIs the expansion board powered on? It runs off the robot "
              "battery, not the Jetson.", file=sys.stderr)
        return 1

    buf = bytearray()
    counts: collections.Counter = collections.Counter()
    bad = collections.Counter()
    devices: collections.Counter = collections.Counter()
    samples: dict[int, list[bytes]] = collections.defaultdict(list)
    total = 0
    deadline = time.monotonic() + args.seconds

    try:
        while time.monotonic() < deadline:
            chunk = ser.read(256)
            if chunk:
                buf += chunk
                frames, buf = parse(buf)
                for f in frames:
                    total += 1
                    counts[f["func"]] += 1
                    devices[f["device_id"]] += 1
                    if not f["checksum_ok"]:
                        bad[f["func"]] += 1
                    if len(samples[f["func"]]) < args.show:
                        samples[f["func"]].append(f["payload"])
            elif not buf and total == 0 and time.monotonic() > deadline - args.seconds + 3:
                pass
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

    print(f"frames: {total}   leftover bytes: {len(buf)}\n")
    if not total:
        print("Nothing received. Either the board is unpowered, the UART is not "
              "wired to this port, or the baud rate is wrong.")
        print("Try --baud 9600 / 38400 / 57600, and check the 40-pin ribbon.")
        return 1

    print("device ids seen:")
    for d, n in devices.most_common():
        note = "  (vendor library assumes 0xFC)" if d != VENDOR_DEVICE_ID else ""
        print(f"  0x{d:02X}  x{n}{note}")

    print("\nfunctions seen:")
    for func, n in counts.most_common():
        name = VENDOR_FUNCS.get(func, "UNKNOWN TO THE VENDOR LIBRARY")
        flag = f"  BAD CHECKSUM x{bad[func]}" if bad[func] else ""
        print(f"  0x{func:02X}  x{n:<5} {name}{flag}")
        for p in samples[func][:3]:
            print(f"          {p.hex(' ')}   {describe_payload(func, p)}")

    print("\n--- what this tells us ---")
    if bad:
        print("* Some checksums failed. The checksum rule or the frame length "
              "convention is not what we assume - resolve that before trusting "
              "any command encoding.")
    else:
        print("* All checksums pass, so header, length and checksum rule are "
              "confirmed. Command encoding can be built on this.")
    unknown = [f for f in counts if f not in VENDOR_FUNCS]
    if unknown:
        print("* Functions this firmware uses that the vendor library does not "
              "know: " + ", ".join(f"0x{f:02X}" for f in unknown))
        print("  Function numbering may differ from the library. Do NOT assume "
              "0x10/0x11/0x12 mean motor commands on this firmware.")
    if devices and max(devices, key=devices.get) != VENDOR_DEVICE_ID:
        print(f"* The board identifies as 0x{max(devices, key=devices.get):02X}, "
              f"not the 0x{VENDOR_DEVICE_ID:02X} the library assumes.")
    print("\nNext step is a FUNC_VERSION (0x51) query - a read request, no "
          "motion - to pin the firmware. Only after that should any servo "
          "command be tried, camera pan/tilt first, with the treads off the "
          "ground.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
