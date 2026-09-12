"""Battery supervision for untethered operation.

On wall power a flat battery is a nuisance. On battery it is a filesystem
corruption: the Jetson browns out mid-write, and the NVMe it was writing to is
what everything boots from. So this watches the pack voltage the expansion
board already reports and acts before that happens, in three stages.

    WARN      say so, out loud, and keep working
    DISARM    turn the motors off - they are the largest and spikiest load,
              and a stall under a sagging pack is what drags it under
    CRITICAL  announce, then shut the machine down cleanly

Voltage is a poor fuel gauge, which shapes the design more than the numbers do:

  * It sags under load and recovers at rest, so a single low reading means very
    little. Everything here works on a median over a rolling window.
  * A pack near empty falls off a cliff rather than declining linearly, which
    is why DISARM sits well above CRITICAL rather than just before it.
  * Thresholds are for a 3S lithium pack (12.6V full, ~9.9V empty). A different
    pack needs different numbers, hence the environment overrides.

Defaults are deliberately early. Stopping a robot that had another ten minutes
in it costs a recharge; getting this wrong costs a reflash.
"""
from __future__ import annotations

import logging
import os
import statistics
import subprocess
import time
from collections import deque

log = logging.getLogger(__name__)


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


# 3S lithium: 12.6V full, 11.1V nominal, 9.9V empty.
WARN_V = _f("JETTANK_BATT_WARN", 11.1)
DISARM_V = _f("JETTANK_BATT_DISARM", 10.6)
CRITICAL_V = _f("JETTANK_BATT_CRITICAL", 10.1)

WINDOW = 15                 # samples in the rolling median (~15s at 1Hz)
WARN_REPEAT_S = _f("JETTANK_BATT_WARN_REPEAT", 120.0)
SHUTDOWN = os.environ.get("JETTANK_BATT_SHUTDOWN", "1") not in ("0", "false", "no")


class BatteryMonitor:
    """Tracks pack voltage and escalates as it falls."""

    def __init__(self, loop) -> None:
        self._loop = loop
        self._samples: deque[float] = deque(maxlen=WINDOW)
        self._last_warn = 0.0
        self.state = "ok"              # ok | warn | disarm | critical
        self.disarmed_for_battery = False

    @property
    def voltage(self) -> float | None:
        """Median of the window, or None until there is enough to trust.

        A median rather than the latest reading: voltage sags hard under a
        motor start and recovers a second later, and acting on that transient
        would disarm a healthy robot every time it set off.
        """
        if len(self._samples) < 5:
            return None
        return round(statistics.median(self._samples), 2)

    def sample(self, volts: float) -> None:
        if volts > 5.0:                # ignore obviously bogus reads
            self._samples.append(volts)

    def assess(self) -> str | None:
        """Update state; return a message to say aloud, if any."""
        v = self.voltage
        if v is None:
            return None

        if v <= CRITICAL_V:
            if self.state != "critical":
                self.state = "critical"
                return (f"Battery critical at {v:.1f} volts. Shutting down now "
                        f"to avoid corrupting my disk.")
            return None

        if v <= DISARM_V:
            if self.state != "disarm":
                self.state = "disarm"
                self.disarmed_for_battery = True
                return (f"Battery is low, {v:.1f} volts. I've turned my motors "
                        f"off. I can still see and talk.")
            return None

        if v <= WARN_V:
            now = time.monotonic()
            if self.state != "warn" or now - self._last_warn > WARN_REPEAT_S:
                self.state = "warn"
                self._last_warn = now
                return f"Battery is getting low, {v:.1f} volts."
            return None

        # Recovered - a pack rests upward after a load comes off, so require
        # clear daylight above the threshold before calling it well again.
        if self.state != "ok" and v > WARN_V + 0.3:
            self.state = "ok"
            return f"Battery recovered to {v:.1f} volts."
        return None

    def shutdown(self) -> None:
        if not SHUTDOWN:
            log.warning("battery critical, but automatic shutdown is disabled")
            return
        log.warning("battery critical - shutting down")
        subprocess.run(["sudo", "-n", "shutdown", "-h", "now"],
                       capture_output=True, timeout=10)
