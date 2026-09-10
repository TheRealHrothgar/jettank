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

    def say(self, text: str) -> None:
        log.info("[robot] say %r", text)


def build() -> RobotDriver:
    """Return the best available driver. Falls back to NullRobot."""
    try:
        # Placeholder for the real Yahboom SDK once identified on the board.
        raise ImportError
    except ImportError:
        log.warning("no Yahboom SDK found - using NullRobot (no motion)")
        return NullRobot()
