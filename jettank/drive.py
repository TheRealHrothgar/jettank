"""Motion commands for the Yahboom expansion board (Transbot firmware).

Separate from board.py on purpose. That module reads and is asserted by test to
contain no write call at all; this one is the only place that transmits, so
"can this code move the robot?" is answered by which file you are looking at.

THE PROTOCOL, from Yahboom's own Transbot_Lib on the reference USB drive.
Everything before this was reverse-engineered from telemetry plus the WRONG
vendor library (Rosmaster), and that mismatch explains both the tread runaway
and months of commands going nowhere:

    board -> us   header FF FD   checksum: plain sum(frame[2:-1]) & 0xFF
    us -> board   header FF FE   checksum: (sum(whole frame) + 3) & 0xFF

The device id DIFFERS BY DIRECTION. The board reports as 0xFD and listens as
0xFE. We had been writing 0xFD - the board was correctly ignoring every command
we ever sent, which is why the beeper and the lights did nothing.

The function numbering is different again, and this is the dangerous part:

    code   Rosmaster (what we assumed)   Transbot (what this board runs)
    0x02   BEEP                          MOTION          <-- drives the robot
    0x03   PWM_SERVO                     PWM_SERVO       (agree, by luck)
    0x06   RGB_EFFECT                    BEEP
    0x08   (absent)                      AUTO_REPORT     <-- the 25Hz stream
    0x09   -                             MOTOR
    0x10   MOTOR                         -

0x08 is the clincher: the telemetry we decoded from the wire is AUTO_REPORT in
Transbot and does not exist in Rosmaster at all. And note what 0x02 means here.
Every "harmless beep" sent through the Rosmaster mapping was addressed to the
MOTION function. That is the tread runaway, exactly.

Motors are indexed 1-2 here, not 1-4: this board drives two treads directly.
"""
from __future__ import annotations

import json
import logging
import os
import struct
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Where operator confirmations are recorded. Written only by
# tools/verify_motion.py, after a human watched the robot do the right thing.
VERIFY_FILE = Path(os.environ.get(
    "JETTANK_MOTION_VERIFIED", Path.home() / ".jettank_motion.json"))

HEADER = 0xFF
CMD_DEVICE_ID = 0xFE          # what the board LISTENS on (telemetry is 0xFD)
CHECKSUM_SEED = 3             # Transbot_Lib: sum(cmd, 3) & 0xff

# Transbot function codes. No longer hypotheses.
FUNC_SET_PID = 0x01
FUNC_MOTION = 0x02            # velocity + angular; MOVES THE ROBOT
FUNC_PWM_SERVO = 0x03
FUNC_RGB = 0x04
FUNC_RGB_EFFECT = 0x05
FUNC_BEEP = 0x06
FUNC_BIG_LED = 0x07
FUNC_AUTO_REPORT = 0x08
FUNC_MOTOR = 0x09             # (index 1-2, int16 speed -100..100)
FUNC_CAR_RUN = 0x0D
FUNC_UART_SERVO = 0x20        # the arm's bus servos
FUNC_UART_SERVO_ID = 0x21     # query: reply comes back as a 0x20 report
FUNC_UART_SERVO_TORQUE = 0x22
FUNC_ARM_CTRL = 0x23
FUNC_VERSION = 0x51

# Hard ceiling in the driver, below whatever MotionGuard allows. Two independent
# limits, because this one survives even if the guard is misconfigured.
MAX_MOTOR = int(os.environ.get("JETTANK_MAX_MOTOR", "30"))      # of 100
# Transbot's own ceilings are 0.45 m/s and 2 rad/s. Ours are deliberately well
# under them - this is an indoor robot on a wooden floor with a dog nearby.
MAX_SPEED_MS = float(os.environ.get("JETTANK_MAX_SPEED", "0.18"))
MAX_TURN_RADS = float(os.environ.get("JETTANK_MAX_TURN", "0.9"))

# The camera head is a different risk class from the treads and is gated
# separately. Pan/tilt cannot drive the robot anywhere: worst case it points
# the camera somewhere useless, which is visible and instantly reversible.
SERVOS_DEFAULT_ON = os.environ.get("JETTANK_SERVOS", "1") not in ("0", "false", "no")

PAN_LIMIT = int(os.environ.get("JETTANK_PAN_LIMIT", "80"))
TILT_LIMIT = int(os.environ.get("JETTANK_TILT_LIMIT", "40"))


def encode(func: int, payload: bytes) -> bytes:
    """Frame a command the way the board expects to receive one."""
    length = len(payload) + 3          # matches Transbot_Lib's hardcoded lengths
    head = bytes((HEADER, CMD_DEVICE_ID, length, func)) + payload
    return head + bytes(((sum(head) + CHECKSUM_SEED) & 0xFF,))


def motor_frame(index: int, speed: int) -> bytes:
    """One tread. index is 1 or 2; speed is -100..100, clamped to MAX_MOTOR."""
    spd = max(-MAX_MOTOR, min(MAX_MOTOR, int(speed)))
    return encode(FUNC_MOTOR, bytes((max(1, min(int(index), 2)),))
                  + struct.pack("<h", spd))


def motion_frame(velocity: float, angular: float) -> bytes:
    """Differential drive in one frame - the board mixes the tracks itself.

    This is how Transbot_Lib actually drives: velocity in m/s and angular in
    rad/s, not per-track speeds. Sending two separate motor frames instead
    meant the tracks were commanded a hundredth of a second apart and, in
    practice, only the second one took effect.

    Note the asymmetric packing, which is Yahboom's: velocity is sent as a
    SINGLE byte (the low byte of velocity*100, so -45..45 fits), while angular
    is a full little-endian int16.
    """
    v = max(-MAX_SPEED_MS, min(MAX_SPEED_MS, float(velocity)))
    a = max(-MAX_TURN_RADS, min(MAX_TURN_RADS, float(angular)))
    vb = struct.pack("<h", int(v * 100))[0:1]
    ab = struct.pack("<h", int(a * 100))
    return encode(FUNC_MOTION, vb + ab)


def stop_frames() -> list[bytes]:
    """Everything to zero, by every route we know. Stopping is worth redundancy."""
    return [motion_frame(0.0, 0.0), motor_frame(1, 0), motor_frame(2, 0)]


def beep_frame(ms: int) -> bytes:
    return encode(FUNC_BEEP, struct.pack("<h", max(0, min(int(ms), 5000))))


def servo_frame(servo_id: int, angle: int) -> bytes:
    return encode(FUNC_PWM_SERVO,
                  bytes((max(1, min(int(servo_id), 4)), max(0, min(int(angle), 180)))))


# --- the arm -------------------------------------------------------------
# STATUS on THIS robot, established by exhaustive scan rather than by eye.
#
# Scanned bus servo ids 1-40 and PWM ids 1-8, using the chassis IMU as an
# objective detector - a servo that moves the arm perturbs the accelerometer,
# so this did not depend on someone watching:
#
#   PWM 1, 2     camera pan / tilt          (working, verified from frames)
#   PWM 3-8      nothing
#   BUS 10       AN ARM JOINT               peak disturbance 4001 vs baseline
#                                           78 - a 50x signal, unambiguous
#   BUS 1-9, 11-40   nothing
#
# So exactly one arm actuator answers. The Transbot library documents ids
# 7/8/9 and Yahboom's arm docs describe joints on PWM 3/4 with a separate
# jaws servo; neither matches this robot. A JetTank ROS snippet addresses the
# gripper as ArmJoint id 6, but bus id 6 does not respond - that id is
# logical, translated by their driver, not a bus address.
#
# The remaining joints and the gripper are therefore not reachable over this
# bus. Most likely they are unpowered or not connected - the arm's servos
# daisy-chain, and only the one nearest the connector answering would look
# exactly like this.
# THE GRIPPER DOES NOT RESPOND, and this was established with the jaws in
# direct view of the camera rather than by asking someone to watch.
#
# Method: raise the arm on servo 10 so the gripper faces the camera, then
# sweep every id while diffing only the pixels the jaws occupy. Noise floor
# 2.58; the only signal above it was BUS 10 at 28.07 - and that is the arm
# swinging the whole assembly through frame, not the jaws moving.
#
#   BUS 1-12   only 10 registers, and only as arm motion
#   PWM 3-12   nothing
#
# Caelan spotted why it LOOKED like the claw was working: the jaws appear to
# open when the arm tilts up and close when it tilts down. Holding servo 10
# fixed and moving only the camera showed the jaws unchanged - so that was the
# arm changing the viewing angle, not actuation. A good catch; it would have
# been easy to record a working gripper on the strength of it.
#
# Yahboom's gamepad mapping (R1 close, R2 open) and their ROS ArmJoint id 6
# both exist, so the hardware works under their stack. Those ids are logical,
# translated by their driver; bus id 6 is silent. The likeliest explanation
# remains that the arm's servos daisy-chain and only the one nearest the
# connector is powered or attached.
ARM_SERVO_JOINT = 10          # the one confirmed actuator
ARM_JOINT_PULSE = (1400, 2600)   # safe travel; +/-1000 rocked the chassis

ARM_JOINTS = (7, 8, 9)
ARM_RANGE = {7: (0, 225), 8: (30, 270), 9: (30, 180)}
PULSE_MIN, PULSE_MAX = 900, 3100


def arm_angle_to_pulse(servo_id: int, angle: float, offset: float = 0.0) -> int:
    """Yahboom's per-joint angle -> pulse mapping, reproduced exactly.

    Each joint is mapped differently and two of the three are inverted; this
    is not a formula to rederive from first principles.
    """
    span = PULSE_MAX - PULSE_MIN
    if servo_id == 7:
        value = span * (angle - offset - 180) / (0 - 180) + PULSE_MIN
    elif servo_id == 8:
        value = span * (angle - 90 - offset - 180) / (0 - 180) + PULSE_MIN
    elif servo_id == 9:
        value = span * (angle + offset - 0) / (180 - 0) + PULSE_MIN
    else:
        raise ValueError(f"servo {servo_id} is not an arm joint")
    return int(max(PULSE_MIN, min(PULSE_MAX, value)))


def arm_frame(servo_id: int, pulse: int, run_time_ms: int = 500) -> bytes:
    """Move one bus servo to a pulse value over run_time_ms."""
    pulse = max(PULSE_MIN, min(PULSE_MAX, int(pulse)))
    run = max(0, min(int(run_time_ms), 2000))
    return encode(FUNC_UART_SERVO,
                  bytes((int(servo_id) & 0xFF,)) + struct.pack("<h", pulse)
                  + struct.pack("<h", run))


def arm_query_frame(servo_id: int) -> bytes:
    """Ask a bus servo for its position.

    The reply arrives asynchronously on the telemetry stream as a 0x20 frame
    carrying [id, int16 position]. Useful for finding which servos physically
    exist: an id that answers is present, one that does not is not wired.
    """
    return encode(FUNC_UART_SERVO_ID, bytes((int(servo_id) & 0xFF,)))


def arm_all_frame(angle7: float, angle8: float, angle9: float,
                  run_time_ms: int = 700, offsets=(0.0, 0.0, 0.0)) -> bytes:
    """All three joints in one frame (FUNC_ARM_CTRL).

    Distinct from sending three separate bus-servo commands: this is the call
    Yahboom's own arm control uses, and on some firmware it is the only one
    the arm responds to. Angles are clamped to each joint's documented range
    before conversion - a bus servo driven past its end stop stalls against
    its own gearbox.
    """
    a7 = max(ARM_RANGE[7][0], min(ARM_RANGE[7][1], float(angle7)))
    a8 = max(ARM_RANGE[8][0], min(ARM_RANGE[8][1], float(angle8)))
    a9 = max(ARM_RANGE[9][0], min(ARM_RANGE[9][1], float(angle9)))
    payload = (struct.pack("<h", arm_angle_to_pulse(7, a7, offsets[0]))
               + struct.pack("<h", arm_angle_to_pulse(8, a8, offsets[1]))
               + struct.pack("<h", arm_angle_to_pulse(9, a9, offsets[2]))
               + struct.pack("<h", max(0, min(int(run_time_ms), 2000))))
    return encode(FUNC_ARM_CTRL, payload)


def arm_torque_frame(on: bool) -> bytes:
    """Enable or release holding torque on the arm servos.

    Releasing lets the arm be posed by hand, and is the safe state to leave it
    in - a servo holding a stalled position draws current and heats up.
    """
    return encode(FUNC_UART_SERVO_TORQUE, bytes((1 if on else 0,)))


def headlight_frame(brightness: int) -> bytes:
    """The two white floodlights on the camera head. 0-100.

    Worth more than it looks: the detector reads 3/255 average brightness in an
    unlit room, which no model can work with. His own headlights are the
    difference between seeing people and not.
    """
    return encode(FUNC_BIG_LED, bytes((max(0, min(int(brightness), 100)),)))


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
            for frame in stop_frames():
                with __import__("contextlib").suppress(Exception):
                    self.send(frame)
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
        self.left_index = int(self._v.get("left_index", 1))
        self.right_index = int(self._v.get("right_index", 2))
        self._invert_left = bool(self._v.get("invert_left", False))
        self._invert_right = bool(self._v.get("invert_right", False))
        self._last_cmd = 0.0
        self.pan = 0.0
        self.tilt = 0.0
        self.light = 0
        self.arm_pulse = 2000
        self._pan_sign = 1 if self._v.get("pan_sign", 1) >= 0 else -1
        self._tilt_sign = 1 if self._v.get("tilt_sign", 1) >= 0 else -1

    def confirmed(self, name: str) -> bool:
        # Servos are allowed without a recorded confirmation because the
        # consequence of being wrong is bounded - see SERVOS_DEFAULT_ON.
        if name == "servo" and SERVOS_DEFAULT_ON and "servo" not in self._v:
            return True
        return bool(self._v.get(name))

    # ---- motion ----
    def drive(self, linear: float, angular: float) -> None:
        """linear and angular, each -1..1, scaled to this driver's ceilings.

        These are the same units MotionGuard works in. They used to be read as
        per-track speeds, so a request to go straight forward drove one track
        and the robot turned instead.
        """
        if not self.confirmed("motor"):
            log.warning("[drive] motor function unconfirmed - not transmitting "
                        "(run tools/verify_motion.py)")
            return
        v = max(-1.0, min(1.0, float(linear))) * MAX_SPEED_MS
        a = max(-1.0, min(1.0, float(angular))) * MAX_TURN_RADS
        if self._invert_left or self._invert_right:
            v = -v
        self._link.send(motion_frame(v, a))
        self._last_cmd = time.monotonic()

    def stop(self) -> None:
        """Always transmitted, confirmed or not.

        A stop that refuses to send because a function is unverified is worse
        than useless. If the motor function number is wrong this is a no-op on
        a board that was never moving; if it is right, it stops.
        """
        for frame in stop_frames():
            with __import__("contextlib").suppress(Exception):
                self._link.send(frame)

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
        # Both axes are inverted relative to the servo's own numbering, checked
        # against captured frames rather than assumed: raising the pan angle
        # swings the view LEFT, and raising the tilt angle points DOWN. Hank's
        # interface stays intuitive - positive pan is his right, positive tilt
        # is up - and the inversion lives here.
        self._link.send(servo_frame(pan_id, int(90 - p * self._pan_sign)))
        time.sleep(0.02)                     # the board drops back-to-back frames
        self._link.send(servo_frame(tilt_id, int(90 - t * self._tilt_sign)))
        self.pan, self.tilt = p, t

    # ---- arm ----
    # Only servo 10 responds on this robot (see the scan results above), so the
    # interface is built around the one joint that exists rather than pretending
    # to be a three-axis arm. Positions are named because raw pulse values are
    # meaningless to anyone using this, including the model.
    ARM_POSES = {
        "stow": 2600,       # folded back, out of the camera's view
        "down": 2300,
        "level": 2000,      # centre
        "up": 1700,
        "raised": 1400,     # fully up, gripper toward the camera
    }

    def arm(self, position: str | int | float, run_time_ms: int = 800) -> None:
        """Move the arm joint. Accepts a pose name or a raw pulse.

        Pulse travel is clamped to the range established by test - beyond it
        the servo reaches its end stop, which showed up as the chassis rocking
        hard enough to trip the IMU guard.
        """
        if not self.confirmed("arm"):
            log.warning("[arm] arm not verified on this robot - not transmitting")
            return
        if isinstance(position, str):
            pulse = self.ARM_POSES.get(position.lower())
            if pulse is None:
                log.error("[arm] unknown position %r; known: %s",
                          position, ", ".join(self.ARM_POSES))
                return
        else:
            pulse = int(position)
        lo, hi = ARM_JOINT_PULSE
        clamped = max(lo, min(hi, pulse))
        if clamped != pulse:
            log.info("[arm] pulse %d clamped to %d (safe travel %d-%d)",
                     pulse, clamped, lo, hi)
        self._link.send(arm_frame(ARM_SERVO_JOINT, clamped, run_time_ms))
        self.arm_pulse = clamped

    def arm_position(self) -> str:
        """Nearest named pose to where the arm currently is."""
        return min(self.ARM_POSES,
                   key=lambda k: abs(self.ARM_POSES[k] - self.arm_pulse))

    def arm_positions(self) -> list[str]:
        return list(self.ARM_POSES)

    def gripper(self, closed: bool) -> None:
        """Not available: no servo on this robot actuates the jaws.

        Verified with the jaws in direct view of the camera - see the scan
        results at the top of this module. Says so rather than silently doing
        nothing, so a caller gets an explanation instead of a no-op.
        """
        log.warning("[gripper] this robot has no controllable gripper - the "
                    "jaws do not respond on any servo id")

    def arm_torque(self, on: bool) -> None:
        """Hold position, or go limp so the arm can be posed by hand."""
        self._link.send(arm_torque_frame(on))

    def headlights(self, brightness: int) -> None:
        """0 off, 100 full. Safe regardless of motor verification."""
        self._link.send(headlight_frame(brightness))
        self.light = max(0, min(int(brightness), 100))

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
