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
    """Return the best available driver. Falls back to NullRobot.

    The Yahboom driver starts in dry-run: it encodes and logs frames but does
    not transmit, because the frame format is not yet confirmed against the
    board's own SDK. Sending a wrong servo command can drive the arm into its
    end stops. Call `arm_live()` on the driver once verified.
    """
    try:
        from .yahboom import YahboomRobot

        robot = YahboomRobot()
        robot.open()
        log.info("Yahboom driver active (dry-run; call arm_live() to transmit)")
        return robot
    except Exception as exc:  # noqa: BLE001 - any failure must not stop perception
        log.warning("no Yahboom board usable (%s) - using NullRobot (no motion)", exc)
        return NullRobot()
