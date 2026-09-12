"""Actuator interface for the Yahboom Jettank.

The concrete transport is unknown until we can inspect the board (Yahboom ships
a serial protocol to the expansion MCU on some models, a Python SDK on others),
so this is a narrow interface with a safe no-op implementation. Swapping in the
real driver should not require touching the loop.
"""
from __future__ import annotations

import logging
from typing import Protocol

log = logging.getLogger(__name__)


class RobotDriver(Protocol):
    def drive(self, left: float, right: float) -> None: ...
    def stop(self) -> None: ...
    def arm(self, joint: str, angle: float) -> None: ...
    def gripper(self, closed: bool) -> None: ...
    def look(self, pan: float, tilt: float) -> None: ...
    def say(self, text: str) -> None: ...


class NullRobot:
    """Logs intent without moving anything. Used until the real SDK is wired up."""

    def drive(self, left: float, right: float) -> None:
        log.info("[robot] drive l=%.2f r=%.2f", left, right)

    def stop(self) -> None:
        log.info("[robot] stop")

    def arm(self, joint: str, angle: float) -> None:
        log.info("[robot] arm %s -> %.1f deg", joint, angle)

    def gripper(self, closed: bool) -> None:
        log.info("[robot] gripper %s", "close" if closed else "open")

    def look(self, pan: float, tilt: float) -> None:
        log.info("[robot] look pan=%.1f tilt=%.1f", pan, tilt)

    def say(self, text: str) -> None:
        log.info("[robot] say %r", text)


def build() -> RobotDriver:
    """Return the best available driver, falling back to NullRobot.

    Prefers the verified-envelope driver in drive.py. That driver gates its own
    capabilities: the camera servos work by default because a mis-aimed camera
    is bounded and reversible, while the treads stay inert until an operator
    confirms the motor function with tools/verify_motion.py. The old yahboom.py
    path is not used - its framing does not validate against this firmware.
    """
    try:
        from .drive import build as build_board

        driver = build_board()
        if driver is not None:
            return driver
    except Exception as exc:  # noqa: BLE001 - never let this stop perception
        log.warning("board driver unavailable (%s)", exc)
    log.info("no board driver - using NullRobot (logs intent, moves nothing)")
    return NullRobot()
