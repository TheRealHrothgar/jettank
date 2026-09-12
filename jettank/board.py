"""Yahboom expansion board protocol - VERIFIED AGAINST THE ACTUAL HARDWARE.

Everything here was derived from the board's own telemetry stream, not from the
vendor library. That distinction matters: the vendor library disagrees with
this firmware in three ways, and following it is what caused the tread runaway.

  Frame     FF FD <len> <func> <payload...> <checksum>      (21 bytes for 0x08)
  Device    0xFD                     (the vendor library assumes 0xFC)
  Length    counts from <len> through <checksum> inclusive
  Checksum  sum(frame[2:-1]) & 0xFF  (a PLAIN sum - the vendor library adds a
            257-device_id complement, which does not validate here)

Confirmed empirically: 375/375 frames parse and checksum-match with these
rules, at 25 Hz, over repeated captures.

  FUNC 0x08 - combined telemetry, 16-byte payload. Not present in the vendor
              library at all, which is the clearest sign its function numbering
              cannot be trusted for this firmware.
                p[0:3]   zero in every frame observed
                p[3:15]  six little-endian int16: accel XYZ then gyro XYZ,
                         accel at 16384 LSB/g (verified: az reads ~16584 with
                         the robot sitting level, gyro ~0 while stationary)
                p[15]    battery in decivolts (0x7A = 12.2 V)

WHY THERE ARE NO MOTION COMMANDS HERE
Framing is verified; the meaning of command function numbers is NOT. The board
appears not to reject bad checksums - the vendor library's frames, which carry
the wrong device id AND fail this checksum rule, still triggered the beeper and
still moved the treads. A board that acts on frames it cannot validate will act
on a wrong guess too. So this module reads, and encodes frames, and stops
there. Motion stays with MotionGuard and the dry-run driver until each command
function is confirmed individually with the treads off the ground.
"""
from __future__ import annotations

import logging
import struct
import threading
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

HEADER = 0xFF
DEVICE_ID = 0xFD
FUNC_TELEMETRY = 0x08

ACCEL_LSB_PER_G = 16384.0
TELEMETRY_LEN = 21


def checksum(frame_without_cs: bytes) -> int:
    """Plain sum from the length byte through the last payload byte."""
    return sum(frame_without_cs[2:]) & 0xFF


def encode(func: int, payload: bytes = b"", device_id: int = DEVICE_ID) -> bytes:
    """Build a frame. Encoding only - nothing here transmits."""
    length = len(payload) + 3          # len + func + payload + checksum
    head = bytes((HEADER, device_id, length, func)) + payload
    return head + bytes((checksum(head),))


@dataclass(frozen=True)
class Telemetry:
    accel_g: tuple[float, float, float]
    gyro: tuple[int, int, int]
    battery_v: float
    at: float

    @property
    def tilt_g(self) -> float:
        """How far off level, as a fraction of g in the horizontal plane."""
        ax, ay, _ = self.accel_g
        return (ax * ax + ay * ay) ** 0.5

    @property
    def moving(self) -> bool:
        return any(abs(g) > 200 for g in self.gyro)


def decode_telemetry(payload: bytes) -> Telemetry | None:
    if len(payload) < 16:
        return None
    ax, ay, az, gx, gy, gz = struct.unpack("<6h", payload[3:15])
    return Telemetry(
        accel_g=(ax / ACCEL_LSB_PER_G, ay / ACCEL_LSB_PER_G, az / ACCEL_LSB_PER_G),
        gyro=(gx, gy, gz),
        battery_v=payload[15] / 10.0,
        at=time.monotonic(),
    )


def parse(buf: bytearray) -> tuple[list[tuple[int, bytes]], bytearray]:
    """Extract (func, payload) frames. Silently drops frames that fail checksum."""
    out: list[tuple[int, bytes]] = []
    i = 0
    while i < len(buf):
        if buf[i] != HEADER or (i + 1 < len(buf) and buf[i + 1] != DEVICE_ID):
            i += 1
            continue
        if len(buf) - i < 4:
            break
        length = buf[i + 2]
        if not 3 <= length <= 64:
            i += 1
            continue
        end = i + 2 + length
        if end > len(buf):
            break
        frame = bytes(buf[i:end])
        if checksum(frame[:-1]) == frame[-1]:
            out.append((frame[3], frame[4:-1]))
            i = end
        else:
            i += 1                      # resynchronise on the next header
    return out, buf[i:]


class BoardReader:
    """Read-only telemetry reader. Has no write path, by design.

    Runs on a thread because the port is a blocking read, and keeps only the
    most recent sample - this is a status source, not a log.
    """

    def __init__(self, port: str = "/dev/ttyTHS1", baud: int = 115200) -> None:
        self.port = port
        self.baud = baud
        self._ser = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Telemetry | None = None
        self.frames = 0
        self.bad_frames = 0

    @property
    def available(self) -> bool:
        return self._latest is not None and (time.monotonic() - self._latest.at) < 3.0

    def latest(self) -> Telemetry | None:
        with self._lock:
            return self._latest

    def start(self) -> bool:
        try:
            import serial
        except ImportError:
            log.warning("pyserial not installed - no board telemetry")
            return False
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.3)
        except Exception as exc:  # noqa: BLE001 - an absent board is normal
            log.info("expansion board not readable on %s (%s)", self.port, exc)
            return False
        self._thread = threading.Thread(target=self._run, name="board", daemon=True)
        self._thread.start()
        log.info("reading expansion board telemetry on %s", self.port)
        return True

    def _run(self) -> None:
        buf = bytearray()
        while not self._stop.is_set():
            try:
                chunk = self._ser.read(128)
            except Exception as exc:  # noqa: BLE001
                log.warning("board read failed (%s)", exc)
                return
            if not chunk:
                continue
            buf += chunk
            frames, buf = parse(buf)
            for func, payload in frames:
                self.frames += 1
                if func != FUNC_TELEMETRY:
                    continue
                t = decode_telemetry(payload)
                if t is not None:
                    with self._lock:
                        self._latest = t
            if len(buf) > 4096:
                buf = buf[-256:]

    def status(self) -> dict:
        t = self.latest()
        if t is None:
            return {"board": "not detected"}
        return {
            "board": "connected",
            "battery_v": round(t.battery_v, 1),
            "upright": round(t.accel_g[2], 2),
            "tilt": round(t.tilt_g, 2),
            "physically_moving": t.moving,
            "telemetry_age_s": round(time.monotonic() - t.at, 1),
        }

    def stop(self) -> None:
        self._stop.set()
        if self._ser is not None:
            with __import__("contextlib").suppress(Exception):
                self._ser.close()
