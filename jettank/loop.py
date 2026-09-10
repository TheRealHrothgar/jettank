"""The perception/reasoning loop.

Two rates, deliberately decoupled:

  fast  - camera -> local VLM -> observation. Runs on-device, never waits on
          the network, and is what keeps the robot responsive.
  slow  - observations -> cloud LLM -> high-level plan. Runs as a background
          task; if it is slow or offline the fast loop is unaffected.

The fast loop owns the robot. The cloud only ever *suggests*.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import time
from collections import deque

from . import config as cfg_mod
from .camera import Camera
from .cloud import CloudAgent
from .robot import build as build_robot
from .vlm import LocalVLM

log = logging.getLogger("jettank")


class Loop:
    def __init__(self, cfg, static_image_b64: str | None = None) -> None:
        self.cfg = cfg
        self.static_image_b64 = static_image_b64
        self.camera = Camera(cfg.camera.device, cfg.camera.width, cfg.camera.height, cfg.camera.fps)
        self.vlm = LocalVLM(cfg.vlm.base_url, cfg.vlm.model, cfg.vlm.timeout_s, cfg.vlm.max_tokens)
        self.cloud = CloudAgent(
            cfg.cloud.provider, cfg.cloud.base_url, cfg.cloud.model,
            cfg.cloud.api_key, cfg.cloud.timeout_s, cfg.cloud.max_tokens,
        )
        self.robot = build_robot()
        self.observations: deque[str] = deque(maxlen=32)
        self.plan: dict | None = None
        self._last_cloud = 0.0
        self._cloud_task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.frames_seen = 0
        self.vlm_calls = 0
        self.cloud_calls = 0

    # ---------------- fast loop ----------------

    async def perceive_forever(self) -> None:
        interval = self.cfg.vlm.interval_s
        while not self._stop.is_set():
            started = time.monotonic()
            if self.static_image_b64:
                seq, img = self.frames_seen + 1, self.static_image_b64
            else:
                seq, img = self.camera.latest_jpeg_b64()
            if img:
                self.frames_seen = seq
                text = await self.vlm.describe(img)
                if text:
                    self.vlm_calls += 1
                    self.observations.append(text)
                    log.info("[see] %s", text)
                    self._maybe_escalate()
            elapsed = time.monotonic() - started
            await asyncio.sleep(max(0.0, interval - elapsed))

    # ---------------- slow loop ----------------

    def _maybe_escalate(self) -> None:
        """Kick off a cloud call if enough time has passed and none is in flight."""
        if not self.cloud.enabled:
            return
        if self._cloud_task and not self._cloud_task.done():
            return
        if time.monotonic() - self._last_cloud < self.cfg.cloud.min_interval_s:
            return
        self._last_cloud = time.monotonic()
        self._cloud_task = asyncio.create_task(self._think())

    async def _think(self) -> None:
        plan = await self.cloud.think(list(self.observations))
        if not plan:
            return
        self.cloud_calls += 1
        self.plan = plan
        log.info(
            "[plan] %s -> action=%s target=%s conf=%s",
            plan.get("assessment", "?"), plan.get("action", "?"),
            plan.get("target", ""), plan.get("confidence", "?"),
        )
        self._apply(plan)

    def _apply(self, plan: dict) -> None:
        """Translate a cloud plan into actuator calls. Conservative by default."""
        action = str(plan.get("action", "idle")).lower()
        say = str(plan.get("say", "")).strip()
        if say:
            self.robot.say(say)
        if action == "explore":
            self.robot.drive(0.25, 0.25)
        elif action == "approach":
            self.robot.drive(0.2, 0.2)
        elif action == "retreat":
            self.robot.drive(-0.25, -0.25)
        elif action == "grasp":
            self.robot.stop()
            self.robot.gripper(closed=True)
        elif action == "release":
            self.robot.gripper(closed=False)
        else:
            self.robot.stop()

    # ---------------- lifecycle ----------------

    async def status_forever(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(10.0)
            log.info(
                "[stat] frames=%d vlm=%d cloud=%d obs=%d",
                self.frames_seen, self.vlm_calls, self.cloud_calls, len(self.observations),
            )

    async def run(self) -> None:
        if not self.static_image_b64:
            self.camera.start()
        ok = await self.vlm.available()
        if not ok:
            log.error(
                "local VLM %r not available at %s - pull it first "
                "(e.g. `ollama pull %s`)",
                self.cfg.vlm.model, self.cfg.vlm.base_url, self.cfg.vlm.model,
            )
        log.info("cloud provider=%s enabled=%s", self.cfg.cloud.provider, self.cloud.enabled)
        tasks = [
            asyncio.create_task(self.perceive_forever()),
            asyncio.create_task(self.status_forever()),
        ]
        try:
            await self._stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            if self._cloud_task:
                self._cloud_task.cancel()
            for t in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await t
            self.robot.stop()
            if not self.static_image_b64:
                self.camera.stop()
            await self.vlm.aclose()
            await self.cloud.aclose()

    def request_stop(self) -> None:
        self._stop.set()


async def _amain(argv=None) -> int:
    p = argparse.ArgumentParser(description="Jettank local-VLM / cloud-LLM loop")
    p.add_argument("--once", action="store_true", help="one perception pass, then exit")
    p.add_argument("--image", help="use a still image file instead of a camera")
    p.add_argument("--duration", type=float, default=0.0,
                   help="run the continuous loop for N seconds, then exit (0 = forever)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = cfg_mod.load()

    static = None
    if args.image:
        import base64
        from pathlib import Path as _P
        raw = _P(args.image).read_bytes()
        static = base64.b64encode(raw).decode('ascii')
        log.info('using still image %s (%d bytes)', args.image, len(raw))

    loop = Loop(cfg, static_image_b64=static)

    if args.once:
        await loop.vlm.available()
        if static:
            img = static
        else:
            loop.camera.start()
            await asyncio.sleep(1.0)  # let a frame arrive
            _, img = loop.camera.latest_jpeg_b64()
        if not img:
            log.error("no frame from camera")
            return 1
        text = await loop.vlm.describe(img)
        log.info("[see] %s", text)
        if text and loop.cloud.enabled:
            log.info("[plan] %s", await loop.cloud.think([text]))
        if not static:
            loop.camera.stop()
        await loop.vlm.aclose()
        await loop.cloud.aclose()
        return 0

    import signal

    running = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            running.add_signal_handler(sig, loop.request_stop)
    if args.duration > 0:
        running.call_later(args.duration, loop.request_stop)
    await loop.run()
    return 0


def main() -> int:
    try:
        return asyncio.run(_amain())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
