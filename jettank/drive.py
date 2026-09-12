"""Motion commands for the Yahboom expansion board.

Separate from board.py on purpose. That module reads and is asserted by test to
contain no write call at all; this one is the only place that transmits, so
"can this code move the robot?" is answered by which file you are looking at.

Encodings are the vendor library's, re-framed onto the wire format actually
verified against this firmware:

    vendor                          here
    device id 0xFC                  0xFD          (observed in every frame)
    checksum  sum + (257 - id)      plain sum     (validates; the vendor's does not)
    length    len(cmd) - 1          identical     (this part the vendor got right)

So the payload layouts below are inherited, but the envelope is ours. That
distinction matters: the envelope is confirmed, the payload meanings are not.

WHAT IS AND IS NOT KNOWN
Verified from telemetry: header, device id, length rule, checksum rule.
NOT verified: that this firmware's function numbers mean what the vendor
library says. FUNC 0x08 streams constantly and the vendor library has no case
for it at all, which is direct evidence its numbering is at least incomplete.

Hence MOTOR_FUNC and friends are treated as hypotheses until an operator
confirms each one with the treads off the ground, using tools/verify_motion.py.
Until that file records a confirmation, `build()` refuses to hand out a live
driver. This is not caution for its own sake: the last time these numbers were
guessed, the treads ran away.
"""
from __future__ import annotations

import json
import logging
import os
import struct
import threading
import time
from pathlib import Path

from .board import DEVICE_ID, HEADER, checksum

log = logging.getLogger(__name__)

# Where operator confirmations are recorded. Written only by
# tools/verify_motion.py, after a human watched the robot do the right thing.
VERIFY_FILE = Path(os.environ.get(
    "JETTANK_MOTION_VERIFIED", Path.home() / ".jettank_motion.json"))

# Hypotheses, from the vendor library. Confirmed individually before use.
FUNC_BEEP = 0x02
FUNC_PWM_SERVO = 0x03
FUNC_RGB = 0x05
FUNC_MOTOR = 0x10          # payload: 4 signed bytes, -100..100, motors 1-4
FUNC_CAR_RUN = 0x11
FUNC_MOTION = 0x12

# Hard ceiling in the driver, below whatever MotionGuard allows. Two independent
# limits, because this one survives even if the guard is misconfigured.
MAX_MOTOR = int(os.environ.get("JETTANK_MAX_MOTOR", "30"))      # of 100

# The camera head is a different risk class from the treads and is gated
# separately. Pan/tilt cannot drive the robot anywhere: worst case it points
# the camera somewhere useless, which is visible and instantly reversible.
# Treads can leave the table. So servos default to on and motors do not.
SERVOS_DEFAULT_ON = os.environ.get("JETTANK_SERVOS", "1") not in ("0", "false", "no")

# Angle limits, applied here as well as upstream. The gimbal has end stops and
# driving a servo into one stalls it, which draws current and cooks it.
PAN_LIMIT = int(os.environ.get("JETTANK_PAN_LIMIT", "80"))
TILT_LIMIT = int(os.environ.get("JETTANK_TILT_LIMIT", "40"))


def encode(func: int, payload: bytes) -> bytes:
    """Frame a command using the verified envelope."""
    length = len(payload) + 3
    head = bytes((HEADER, DEVICE_ID, length, func)) + payload
    return head + bytes((checksum(head),))


def motor_frame(m1: int, m2: int, m3: int, m4: int) -> bytes:
    """Raw four-motor command. Values are clamped here as well as upstream."""
    vals = [max(-MAX_MOTOR, min(MAX_MOTOR, int(v))) for v in (m1, m2, m3, m4)]
    return encode(FUNC_MOTOR, struct.pack("4b", *vals))


def beep_frame(ms: int) -> bytes:
    return encode(FUNC_BEEP, struct.pack("<h", max(0, min(int(ms), 5000))))


def servo_frame(servo_id: int, angle: int) -> bytes:
    return encode(FUNC_PWM_SERVO,
                  bytes((max(1, min(int(servo_id), 4)), max(0, min(int(angle), 180)))))


def rgb_frame(index: int, r: int, g: int, b: int) -> bytes:
    return encode(FUNC_RGB, bytes((index & 0xFF, r & 0xFF, g & 0xFF, b & 0xFF)))


def load_verification() -> dict:
    try:
        return json.loads(VERIFY_FILE.read_text())
    except Exception:  # noqa: BLE001 - absent or unreadable both mean unverified
        return {}


def save_verification(data: dict) -> None:
    VERIFY_FILE.write_text(json.dumps(data, indent=2))
    log.info("recorded motion verification in %s", VERIFY_FILE)


class BoardLink:
    """The only object in the codebase that writes to the board."""

    def __init__(self, port: str = "/dev/ttyTHS1", baud: int = 115200) -> None:
        self.port, self.baud = port, baud
        self._ser = None
        self._lock = threading.Lock()

    def open(self) -> bool:
        import serial

        self._ser = serial.Serial(self.port, self.baud, timeout=0.3)
        log.info("board link open on %s (writes enabled)", self.port)
        return True

    def send(self, frame: bytes) -> None:
        if self._ser is None:
            raise RuntimeError("board link is not open")
        with self._lock:
            self._ser.write(frame)

    def close(self) -> None:
        if self._ser is not None:
            # Always leave the motors commanded to zero, whatever happened.
            with __import__("contextlib").suppress(Exception):
                self.send(motor_frame(0, 0, 0, 0))
            with __import__("contextlib").suppress(Exception):
                self._ser.close()
            self._ser = None


class RosmasterDriver:
    """RobotDriver over the verified envelope.

    Every method checks that its underlying function was confirmed by an
    operator. An unconfirmed function logs and does nothing rather than
    transmitting a guess.
    """

    # Which of the four motor channels drive which tread. Determined
    # empirically by verify_motion.py - a tank has two treads and the board
    # takes four values, and which maps to which is not documented.
    def __init__(self, link: BoardLink, verification: dict | None = None) -> None:
        self._link = link
        self._v = verification if verification is not None else load_verification()
        self.left_channels = tuple(self._v.get("left_channels", (1, 2)))
        self.right_channels = tuple(self._v.get("right_channels", (3, 4)))
        self._invert_left = bool(self._v.get("invert_left", False))
        self._invert_right = bool(self._v.get("invert_right", False))
        self._last_cmd = 0.0
        self.pan = 0.0
        self.tilt = 0.0

    def confirmed(self, name: str) -> bool:
        # Servos are allowed without a recorded confirmation because the
        # consequence of being wrong is bounded - see SERVOS_DEFAULT_ON.
        if name == "servo" and SERVOS_DEFAULT_ON and "servo" not in self._v:
            return True
        return bool(self._v.get(name))

    # ---- motion ----
    def drive(self, left: float, right: float) -> None:
        """left/right in -1..1. Scaled to the driver's own ceiling."""
        if not self.confirmed("motor"):
            log.warning("[drive] motor function unconfirmed - not transmitting "
                        "(run tools/verify_motion.py)")
            return
        l = int(max(-1.0, min(1.0, left)) * MAX_MOTOR) * (-1 if self._invert_left else 1)
        r = int(max(-1.0, min(1.0, right)) * MAX_MOTOR) * (-1 if self._invert_right else 1)
        m = [0, 0, 0, 0]
        for ch in self.left_channels:
            m[ch - 1] = l
        for ch in self.right_channels:
            m[ch - 1] = r
        self._link.send(motor_frame(*m))
        self._last_cmd = time.monotonic()

    def stop(self) -> None:
        """Always transmitted, confirmed or not.

        A stop that refuses to send because a function is unverified is worse
        than useless. If the motor function number is wrong this is a no-op on
        a board that was never moving; if it is right, it stops.
        """
        with __import__("contextlib").suppress(Exception):
            self._link.send(motor_frame(0, 0, 0, 0))

    # ---- camera head ----
    def look(self, pan: float, tilt: float) -> None:
        """Aim the camera head. Angles are degrees from centre."""
        if not self.confirmed("servo"):
            log.warning("[look] servos disabled - not transmitting")
            return
        pan_id = int(self._v.get("pan_servo", 1))
        tilt_id = int(self._v.get("tilt_servo", 2))
        p = max(-PAN_LIMIT, min(PAN_LIMIT, float(pan)))
        t = max(-TILT_LIMIT, min(TILT_LIMIT, float(tilt)))
        self._link.send(servo_frame(pan_id, int(90 + p)))
        time.sleep(0.02)                     # the board drops back-to-back frames
        self._link.send(servo_frame(tilt_id, int(90 - t)))
        self.pan, self.tilt = p, t

    def arm(self, joint: str, angle: float) -> None:
        log.warning("[arm] the arm's protocol is not verified - not transmitting")

    def gripper(self, closed: bool) -> None:
        log.warning("[gripper] not verified - not transmitting")

    def beep(self, ms: int = 100) -> None:
        if not self.confirmed("beep"):
            log.warning("[beep] unconfirmed - not transmitting")
            return
        self._link.send(beep_frame(ms))

    def say(self, text: str) -> None:
        log.info("[speak] %s", text)      # the speaker is ALSA, not the board


def build(port: str = "/dev/ttyTHS1"):
    """Return a live driver, or None if motion has not been verified.

    Deliberately returns None rather than a driver that silently does nothing:
    the caller then falls back to NullRobot and says so, instead of appearing
    to work.
    """
    v = load_verification()
    if not v.get("motor") and not SERVOS_DEFAULT_ON:
        log.info("nothing verified to drive - see tools/verify_motion.py")
        return None
    link = BoardLink(port)
    try:
        link.open()
    except Exception as exc:  # noqa: BLE001
        log.warning("could not open board for writing (%s)", exc)
        return None
    driver = RosmasterDriver(link, v)
    driver.stop()                     # known state before anything else happens
    log.info("board driver active: servos=%s motors=%s",
             driver.confirmed("servo"),
             "verified" if v.get("motor") else "NOT verified (dry)")
    return driver
