"""Driver for the Yahboom expansion board.

Yahboom's Jetson expansion boards put an STM32 microcontroller behind a USB
serial link (they also expose CAN and SBUS). So the transport is a serial port;
what we do not yet know for certain is the exact frame format, which varies
across their product lines.

This module therefore separates the two concerns:

  * `find_port()` and `YahboomSerial` - transport and discovery. These are
    correct regardless of protocol and can be verified on the bench.
  * `Protocol` - the frame encoding. The layout below follows the common
    Yahboom scheme (0xFF 0xFE header, length, command, payload, checksum) but
    MUST be confirmed against the board's own SDK before trusting it with the
    arm, which can damage itself against its end stops.

Until confirmed, `YahboomRobot` starts in dry-run mode: it encodes and logs
frames without writing them. Call `arm_live()` to actually transmit.
"""
from __future__ import annotations

import glob
import logging
import os

log = logging.getLogger(__name__)

# USB vendor IDs seen on Yahboom expansion boards / their USB-serial bridges.
LIKELY_VIDS = {
    "1a86",  # QinHeng CH340/CH341
    "10c4",  # Silicon Labs CP210x
    "0403",  # FTDI
    "0483",  # STMicroelectronics (STM32 virtual COM port)
}


def candidate_ports() -> list[str]:
    """Serial ports that could plausibly be the expansion board."""
    return sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))


def port_info(dev: str) -> dict[str, str]:
    """Read udev properties for a serial device without shelling out."""
    info: dict[str, str] = {}
    name = os.path.basename(dev)
    for base in (f"/sys/class/tty/{name}/device", f"/sys/class/tty/{name}"):
        real = os.path.realpath(base)
        # walk up to the USB device node that carries idVendor/idProduct
        for _ in range(6):
            vid = os.path.join(real, "idVendor")
            pid = os.path.join(real, "idProduct")
            if os.path.exists(vid) and os.path.exists(pid):
                try:
                    info["vid"] = open(vid).read().strip()
                    info["pid"] = open(pid).read().strip()
                    for k, f in (("manufacturer", "manufacturer"), ("product", "product")):
                        p = os.path.join(real, f)
                        if os.path.exists(p):
                            info[k] = open(p).read().strip()
                except OSError:
                    pass
                return info
            real = os.path.dirname(real)
    return info


def find_port() -> str | None:
    """Best guess at the expansion board's serial port, or None."""
    ports = candidate_ports()
    if not ports:
        return None
    scored: list[tuple[int, str]] = []
    for p in ports:
        info = port_info(p)
        score = 0
        if info.get("vid", "").lower() in LIKELY_VIDS:
            score += 10
        blob = f"{info.get('manufacturer','')} {info.get('product','')}".lower()
        if "yahboom" in blob or "stm32" in blob:
            score += 20
        scored.append((score, p))
        log.info("serial candidate %s vid=%s pid=%s %s (score %d)",
                 p, info.get("vid", "?"), info.get("pid", "?"),
                 info.get("product", ""), score)
    scored.sort(reverse=True)
    best_score, best = scored[0]
    if best_score == 0:
        log.warning("no strong match for the expansion board; guessing %s", best)
    return best


class Protocol:
    """Frame encoder. UNVERIFIED - confirm against the board's SDK."""

    HEAD = bytes((0xFF, 0xFE))

    CMD_MOTOR = 0x01
    CMD_SERVO = 0x02
    CMD_GRIPPER = 0x03
    CMD_BUZZER = 0x04

    @classmethod
    def frame(cls, cmd: int, payload: bytes) -> bytes:
        body = bytes((len(payload) + 2, cmd)) + payload
        checksum = sum(body) & 0xFF
        return cls.HEAD + body + bytes((checksum,))

    @classmethod
    def motor(cls, left: float, right: float) -> bytes:
        def enc(v: float) -> int:
            v = max(-1.0, min(1.0, v))
            return int(round(v * 100)) & 0xFF  # two's complement in one byte
        return cls.frame(cls.CMD_MOTOR, bytes((enc(left), enc(right))))

    @classmethod
    def servo(cls, joint_id: int, angle_deg: float) -> bytes:
        a = int(round(max(0.0, min(180.0, angle_deg))))
        return cls.frame(cls.CMD_SERVO, bytes((joint_id & 0xFF, a & 0xFF)))


class YahboomRobot:
    """Serial driver. Starts in dry-run so a wrong protocol cannot move the arm."""

    JOINTS = {"base": 1, "shoulder": 2, "elbow": 3, "wrist": 4, "roll": 5, "grip": 6}

    def __init__(self, port: str | None = None, baud: int = 115200, live: bool = False) -> None:
        self._port = port or find_port()
        self._baud = baud
        self._live = live
        self._ser = None
        if self._port is None:
            raise RuntimeError("no candidate serial port for the Yahboom board")

    def open(self) -> None:
        import serial  # pyserial; only needed when actually talking to hardware

        self._ser = serial.Serial(self._port, self._baud, timeout=0.5)
        log.info("opened %s at %d baud (live=%s)", self._port, self._baud, self._live)

    def arm_live(self) -> None:
        """Enable real transmission. Only after the protocol is confirmed."""
        self._live = True
        log.warning("Yahboom driver is now LIVE - frames will be transmitted")

    def _send(self, frame: bytes, what: str) -> None:
        if not self._live or self._ser is None:
            log.info("[dry-run] %s -> %s", what, frame.hex(" "))
            return
        self._ser.write(frame)

    # ---- RobotDriver interface ----
    def drive(self, left: float, right: float) -> None:
        self._send(Protocol.motor(left, right), f"drive l={left:.2f} r={right:.2f}")

    def stop(self) -> None:
        self._send(Protocol.motor(0.0, 0.0), "stop")

    def arm(self, joint: str, angle: float) -> None:
        jid = self.JOINTS.get(joint)
        if jid is None:
            log.error("unknown joint %r; known: %s", joint, ", ".join(self.JOINTS))
            return
        self._send(Protocol.servo(jid, angle), f"arm {joint}={angle:.1f}")

    def gripper(self, closed: bool) -> None:
        self.arm("grip", 20.0 if closed else 120.0)

    # Pan/tilt IDs on the camera gimbal. Unverified against firmware 3.2, so
    # these stay dry-run along with everything else motion-capable.
    PAN_ID, TILT_ID = 1, 2

    def look(self, pan: float, tilt: float) -> None:
        self._send(Protocol.servo(self.PAN_ID, pan), f"look pan={pan:.1f}")
        self._send(Protocol.servo(self.TILT_ID, tilt), f"look tilt={tilt:.1f}")

    def say(self, text: str) -> None:
        # The speaker is an ALSA device, not the expansion board.
        log.info("[speak] %s", text)

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
