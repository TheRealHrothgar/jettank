"""Motion safety.

Written after an unverified command set made the treads run away. The rules:

  * Motion is OFF unless explicitly enabled. A remote model cannot enable it.
  * Every motion command is clamped to a configured ceiling.
  * A watchdog stops the robot if no command arrives within `timeout_s`, so a
    dropped network link or a hung planner cannot leave it driving.
  * `estop()` latches. Nothing moves again until `clear_estop()` is called
    locally.

The cloud model proposes; this module disposes.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotionLimits:
    max_linear: float = 0.30     # m/s-ish, in driver units
    max_angular: float = 0.60
    timeout_s: float = 1.0       # no command within this -> stop
    max_run_s: float = 3.0       # a single command may not drive longer than this


class MotionGuard:
    """Gate between any planner and the actuators."""

    def __init__(self, driver, limits: MotionLimits | None = None, enabled: bool = False) -> None:
        self._driver = driver
        self._limits = limits or MotionLimits()
        self._enabled = enabled
        self._estopped = False
        self._lock = threading.RLock()
        self._last_cmd = 0.0
        self._moving = False
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._watchdog, name="motion-watchdog", daemon=True)
        self._thread.start()
        log.info("MotionGuard active (motion %s)", "ENABLED" if enabled else "DISABLED")

    # ---- state ----
    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled and not self._estopped

    def enable(self, on: bool) -> None:
        """Local-only. Deliberately not exposed as a model-callable tool."""
        with self._lock:
            self._enabled = on
            if not on:
                self._halt("motion disabled")
        log.warning("motion %s", "ENABLED" if on else "DISABLED")

    def estop(self, reason: str = "manual") -> None:
        with self._lock:
            self._estopped = True
            self._halt(f"E-STOP ({reason})")
        log.error("E-STOP latched: %s", reason)

    def clear_estop(self) -> None:
        with self._lock:
            self._estopped = False
        log.warning("E-STOP cleared")

    @property
    def estopped(self) -> bool:
        with self._lock:
            return self._estopped

    # ---- motion ----
    def drive(self, linear: float, angular: float, reason: str = "") -> tuple[bool, str]:
        """Request motion. Returns (accepted, explanation)."""
        with self._lock:
            if self._estopped:
                return False, "E-STOP is latched; motion refused"
            if not self._enabled:
                return False, "motion is disabled; refused (enable locally to permit movement)"
            lin = _clamp(linear, self._limits.max_linear)
            ang = _clamp(angular, self._limits.max_angular)
            clamped = (lin != linear) or (ang != angular)
            try:
                self._driver.drive(lin, ang)
            except Exception as exc:  # noqa: BLE001 - never let a driver fault propagate
                log.error("driver.drive failed: %s", exc)
                return False, f"driver error: {exc}"
            self._last_cmd = time.monotonic()
            self._moving = (lin != 0.0) or (ang != 0.0)
            note = f"linear={lin:.2f} angular={ang:.2f}"
            if clamped:
                note += " (clamped to limits)"
            if reason:
                note += f" [{reason}]"
            return True, note

    def stop(self) -> None:
        with self._lock:
            self._halt("requested")

    def _halt(self, why: str) -> None:
        try:
            self._driver.stop()
        except Exception as exc:  # noqa: BLE001
            log.error("driver.stop failed: %s", exc)
        if self._moving:
            log.info("motion halted: %s", why)
        self._moving = False

    def _watchdog(self) -> None:
        while not self._stop_evt.wait(0.1):
            with self._lock:
                if not self._moving:
                    continue
                idle = time.monotonic() - self._last_cmd
                if idle > self._limits.timeout_s:
                    self._halt(f"watchdog: no command for {idle:.1f}s")
                elif idle > self._limits.max_run_s:
                    self._halt("watchdog: max run time")

    def close(self) -> None:
        self._stop_evt.set()
        self.stop()

    def status(self) -> dict:
        with self._lock:
            return {
                "motion_enabled": self._enabled,
                "estopped": self._estopped,
                "moving": self._moving,
                "max_linear": self._limits.max_linear,
                "max_angular": self._limits.max_angular,
                "watchdog_timeout_s": self._limits.timeout_s,
            }


def _clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, float(v)))
