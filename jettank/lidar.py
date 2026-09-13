"""RPLidar A1 - 360 degree range, which is the sensor obstacle avoidance wanted.

The camera detector can only see things it has names for. It knows people,
chairs and dogs; it does not know walls, doors, table legs or steps. On the
first live drive Hank was nose-first into a plain wooden board and correctly
reported a clear path, because nothing in frame had a name. This sensor has no
such gap: it measures geometry, not objects, and does it in the dark.

    camera   ~57 ms   WHAT is there, if it has a name
    lidar    ~780 points/sec, 360 degrees   WHERE the surfaces are

WHAT IT COST TO GET HERE
Three sessions of enumeration failures - `error -32`, then `-71`, and once the
whole xHCI controller dying and taking every USB device with it. It was never
the cable, the port, or the Jetson's power budget. The CP2102 adapter is
powered from its own lead, not from USB, and with that lead disconnected the
chip cannot answer a single setup packet. Restore the lead and it enumerates
first time.

MOTOR CONTROL, AND WHY IT MATTERS ON BATTERY
The motor runs off DTR on the adapter and draws the bulk of the ~700mA. Held
spinning it is most of the idle draw of the whole robot. So it is spun up on
demand and stopped again - Caelan's idea, and the right one: a full sweep takes
200ms at 5Hz, so "look around now" costs a second of motor time rather than
running it continuously to describe a room nobody is in.
"""
from __future__ import annotations

import logging
import math
import os
import statistics
import threading
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

PORT = os.environ.get("JETTANK_LIDAR_PORT", "/dev/rplidar")
BAUD = int(os.environ.get("JETTANK_LIDAR_BAUD", "115200"))

# Protocol
SYNC = 0xA5
CMD_STOP = 0x25
CMD_SCAN = 0x20
CMD_INFO = 0x50
CMD_HEALTH = 0x52

SPIN_UP_S = float(os.environ.get("JETTANK_LIDAR_SPINUP", "1.6"))

# Readings closer than this are the robot seeing its own chassis - the mast
# sits above the deck and clips the arm and antennas.
MIN_VALID_MM = 120.0


@dataclass(frozen=True)
class Scan:
    """One sweep, reduced to what a control loop needs."""

    points: list[tuple[float, float]]      # (angle_deg, distance_mm)
    at: float

    def sector_min(self, centre_deg: float, width_deg: float) -> float | None:
        """Closest return within a sector, or None if nothing was seen there.

        Uses the 10th percentile rather than the single minimum: one spurious
        short return - dust, a reflection off the chassis - would otherwise
        stop the robot dead.
        """
        half = width_deg / 2.0
        vals = []
        for angle, dist in self.points:
            delta = abs((angle - centre_deg + 180) % 360 - 180)
            if delta <= half and dist >= MIN_VALID_MM:
                vals.append(dist)
        if len(vals) < 3:
            return None
        vals.sort()
        return vals[max(0, int(len(vals) * 0.10))]

    @property
    def forward_mm(self) -> float | None:
        """Clearance straight ahead, across the width he actually occupies."""
        return self.sector_min(0.0, 40.0)

    def clearest_heading(self) -> tuple[float, float] | None:
        """(bearing, distance) of the most open direction. For getting unstuck."""
        best = None
        for centre in range(-180, 180, 15):
            d = self.sector_min(float(centre), 30.0)
            if d is not None and (best is None or d > best[1]):
                best = (float(centre), d)
        return best


class Lidar:
    """Background reader with on-demand motor control."""

    def __init__(self, port: str = PORT, baud: int = BAUD) -> None:
        self.port, self.baud = port, baud
        self._ser = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._points: list[tuple[float, float]] = []
        self._last_full: Scan | None = None
        self.spinning = False
        self.points_seen = 0

    @property
    def available(self) -> bool:
        return os.path.exists(self.port)

    def open(self) -> bool:
        if self._ser is not None:
            return True
        if not self.available:
            log.info("no lidar at %s", self.port)
            return False
        try:
            import serial

            self._ser = serial.Serial(self.port, self.baud, timeout=1.0)
            self._ser.dtr = True            # DTR high = motor OFF
            log.info("lidar open on %s", self.port)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("could not open lidar (%s)", exc)
            return False

    def info(self) -> dict:
        """Model and firmware. Cheap, and does not need the motor."""
        if not self.open():
            return {}
        try:
            self._ser.reset_input_buffer()
            self._ser.write(bytes((SYNC, CMD_INFO)))
            self._ser.flush()
            resp = self._ser.read(27)
            if len(resp) < 27 or resp[0] != SYNC:
                return {}
            p = resp[7:]
            return {"model": p[0], "firmware": f"{p[2]}.{p[1]}", "hardware": p[3]}
        except Exception:  # noqa: BLE001
            return {}

    # ---- motor ----
    def spin_up(self) -> None:
        """Start the motor. Takes ~1.5s to reach a usable rate."""
        if not self.open() or self.spinning:
            return
        self._ser.dtr = False
        self.spinning = True
        time.sleep(SPIN_UP_S)

    def spin_down(self) -> None:
        """Stop the motor. This is most of the sensor's power draw."""
        if self._ser is None:
            return
        with __import__("contextlib").suppress(Exception):
            self._ser.write(bytes((SYNC, CMD_STOP)))
            self._ser.flush()
            self._ser.dtr = True
        self.spinning = False

    # ---- scanning ----
    def start(self) -> bool:
        """Spin up and begin reading in the background."""
        if not self.open():
            return False
        self.spin_up()
        try:
            self._ser.reset_input_buffer()
            self._ser.write(bytes((SYNC, CMD_SCAN)))
            self._ser.flush()
            if len(self._ser.read(7)) < 7:      # response descriptor
                log.warning("lidar did not acknowledge scan start")
                return False
        except Exception as exc:  # noqa: BLE001
            log.warning("lidar scan start failed (%s)", exc)
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._read, name="lidar", daemon=True)
        self._thread.start()
        return True

    def _read(self) -> None:
        pending: list[tuple[float, float]] = []
        while not self._stop.is_set() and self._ser is not None:
            try:
                b = self._ser.read(5)
            except Exception:  # noqa: BLE001
                return
            if len(b) < 5:
                continue
            # Two redundant check bits; both must agree or the byte stream has
            # slipped and everything after it is nonsense.
            start, inverse = b[0] & 0x01, (b[0] >> 1) & 0x01
            if start == inverse or not (b[1] & 0x01):
                continue
            # The revolution boundary MUST be handled before any quality
            # filtering. The first sample of a sweep frequently has no return
            # at all, so dropping zero-quality samples first discarded every
            # start flag - the reader saw thousands of points and never
            # assembled a single scan.
            if start:
                if len(pending) > 40:
                    with self._lock:
                        self._last_full = Scan(points=pending, at=time.monotonic())
                pending = []

            quality = b[0] >> 2
            angle = ((b[1] >> 1) | (b[2] << 7)) / 64.0
            dist = (b[3] | (b[4] << 8)) / 4.0
            if quality == 0 or dist <= 0:
                continue
            self.points_seen += 1
            pending.append((angle, dist))

    def latest(self, max_age_s: float = 1.5) -> Scan | None:
        with self._lock:
            scan = self._last_full
        if scan is None or time.monotonic() - scan.at > max_age_s:
            return None
        return scan

    def sweep(self, timeout_s: float = 4.0) -> Scan | None:
        """Spin up, take one good sweep, spin down. The burst pattern.

        This is the mode that suits a battery-powered robot: a full revolution
        costs about 200ms, so a look around is a second of motor time rather
        than running it continuously.
        """
        already = self.spinning
        if not already and not self.start():
            return None
        deadline = time.monotonic() + timeout_s
        scan = None
        while time.monotonic() < deadline:
            scan = self.latest(max_age_s=timeout_s)
            if scan is not None and len(scan.points) > 100:
                break
            time.sleep(0.1)
        if not already:
            self.stop()
        return scan

    def usb_suspend(self, on: bool = True) -> bool:
        """Let the USB device autosuspend, on top of stopping the motor.

        Stopping the motor removes the bulk of the draw - roughly 600mA of the
        ~700mA. This takes the remainder: the CP2102 bridge and the ranging
        core idle at around 100mA, which is small but not nothing when the
        whole robot is on a 12V pack.

        Returns False if the sysfs node is not writable, which is normal
        without root and is not worth failing over.
        """
        node = self._usb_power_node()
        if node is None:
            return False
        try:
            node.write_text("auto" if on else "on")
            log.info("lidar usb autosuspend %s", "enabled" if on else "disabled")
            return True
        except OSError as exc:
            log.debug("could not set usb power control (%s)", exc)
            return False

    def _usb_power_node(self):
        """Find the USB device backing our tty, via sysfs."""
        from pathlib import Path

        try:
            tty = os.path.realpath(self.port).rsplit("/", 1)[-1]
            base = Path(f"/sys/class/tty/{tty}/device")
            # walk up to the usb_device that owns this interface
            for _ in range(4):
                base = base.resolve().parent
                candidate = base / "power" / "control"
                if candidate.exists():
                    return candidate
        except Exception:  # noqa: BLE001
            pass
        return None

    def status(self) -> dict:
        scan = self.latest()
        if scan is None:
            return {"lidar": "idle" if self.available else "not detected",
                    "spinning": self.spinning}
        fwd = scan.forward_mm
        return {
            "lidar": "scanning",
            "spinning": self.spinning,
            "points": len(scan.points),
            "forward_clearance_mm": round(fwd) if fwd else None,
            "age_s": round(time.monotonic() - scan.at, 2),
        }

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        self.spin_down()

    def close(self) -> None:
        self.stop()
        if self._ser is not None:
            with __import__("contextlib").suppress(Exception):
                self._ser.close()
            self._ser = None
