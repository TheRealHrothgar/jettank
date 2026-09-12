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
import threading
import time
from collections import deque

from . import config as cfg_mod
from .audio import Transcriber, VoiceListener, match_wake_word
from .camera import build_camera
from .codegen import BehaviorRunner, BehaviorStore, BehaviorWriter
from .console import Console
from .sandbox import Sandbox, docker_available
from .skills import SkillStore
from .sysinfo import collect as collect_sysinfo, describe as describe_sysinfo
from .cloud import AgentSession, CloudAgent, register_tools
from .faces import FaceEngine
from .narrator import Narrator
from .robot import build as build_robot
from .safety import MotionGuard
from .tools import TOOL_SCHEMAS, Speaker, ToolBox
from .vlm import LocalVLM

log = logging.getLogger("jettank")


class Loop:
    def __init__(self, cfg, static_image_b64: str | None = None) -> None:
        self.cfg = cfg
        self.static_image_b64 = static_image_b64
        self.camera = build_camera(cfg.camera.device, cfg.camera.width, cfg.camera.height, cfg.camera.fps)
        self.vlm = LocalVLM(cfg.vlm.base_url, cfg.vlm.model, cfg.vlm.timeout_s, cfg.vlm.max_tokens)
        self.cloud = CloudAgent(
            cfg.cloud.provider, cfg.cloud.base_url, cfg.cloud.model,
            cfg.cloud.api_key, cfg.cloud.timeout_s, cfg.cloud.max_tokens,
        )
        self.robot = build_robot()

        # Everything the cloud agent is allowed to touch goes through here.
        # Motion starts disabled: MotionGuard.enable() is a local-operator
        # action and is deliberately not reachable as a tool.
        self.guard = MotionGuard(self.robot, enabled=False)
        self.faces = FaceEngine()
        self.narrator = Narrator(cfg.vlm.base_url, cfg.vlm.narrator_model,
                                 enabled=cfg.vlm.narrator)
        self.speaker = Speaker(narrator=self.narrator)
        # Off the critical path, same reasoning as the TTS voice preload.
        threading.Thread(target=self.narrator.warm, name="narrator-warm",
                         daemon=True).start()
        self.skills = SkillStore()

        # Code generation happens in the cloud (Anthropic Messages API); the
        # resulting code runs in a Docker container that has no network, no
        # devices, and reaches the robot only through a guard-gated socket.
        # If Docker is missing we keep the writer but refuse to run, rather
        # than quietly executing generated code in this process.
        self.behavior_store = BehaviorStore()
        self.writer = BehaviorWriter(self.cloud, self.behavior_store,
                                     max_tokens=cfg.cloud.codegen_max_tokens)
        self.sandbox = None
        if docker_available():
            self.sandbox = Sandbox(
                lambda t, a: self.toolbox.dispatch(t, a),
                describe=self._describe_now,
            )
        else:
            log.warning("docker not available - generated behaviours cannot be run")
        self.behaviors = BehaviorRunner(self.behavior_store, self.sandbox,
                                        narrator=self.narrator)

        self.toolbox = ToolBox(self, self.guard, self.camera, self.faces,
                               self.speaker, skills=self.skills,
                               writer=self.writer, behaviors=self.behaviors)
        register_tools(TOOL_SCHEMAS)

        # Config objects are frozen - they record how we booted. Settings the
        # agent may retune at runtime live here instead.
        self.vlm_interval = cfg.vlm.interval_s
        self.cloud_min_interval = cfg.cloud.min_interval_s
        self.cam_width = cfg.camera.width
        self.cam_height = cfg.camera.height

        self.voice: VoiceListener | None = None
        if cfg.voice.enabled:
            self.voice = VoiceListener(
                device=cfg.voice.mic,
                transcriber=Transcriber(cfg.voice.stt_backend, cfg.voice.stt_model),
                threshold=cfg.voice.threshold,
                silence_ms=cfg.voice.silence_ms,
            )

        self.console: Console | None = None
        if cfg.console.enabled:
            self.console = Console(self, cfg.console.host, cfg.console.port)

        # Built last, on purpose: the system facts describe voice, the console
        # and the sandbox, so they must all exist before the snapshot is taken.
        self.system_facts = describe_sysinfo(collect_sysinfo(self))
        self.agent = AgentSession(self.cloud, self.toolbox,
                                  max_tokens=cfg.cloud.agent_max_tokens,
                                  system_facts=self.system_facts)

        self.transcript: deque[dict] = deque(maxlen=32)
        self.observations: deque[str] = deque(maxlen=32)
        self.last_image_b64: str | None = None
        self.plan: dict | None = None
        self._last_cloud = 0.0
        self._cloud_task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.frames_seen = 0
        self.vlm_calls = 0
        self.cloud_calls = 0

    # ---------------- fast loop ----------------

    async def perceive_forever(self) -> None:
        while not self._stop.is_set():
            interval = self.vlm_interval
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
                    self.last_image_b64 = img
                    log.info("[see] %s", text)
                    if self.console:
                        self.console.event("see", text)
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
        if time.monotonic() - self._last_cloud < self.cloud_min_interval:
            return
        self._last_cloud = time.monotonic()
        self._cloud_task = asyncio.create_task(self._think())

    async def _think(self) -> None:
        frame = self.last_image_b64 if self.cfg.cloud.send_frames else None
        plan = await self.cloud.think(list(self.observations), image_b64=frame)
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

    # Cloud plan -> (linear, angular). Everything motion-capable goes through
    # MotionGuard, same as the agent's tools: the autonomous path must not be
    # a way around the interlock the interactive path respects.
    _MOVES = {
        "explore": (0.25, 0.0),
        "approach": (0.20, 0.0),
        "retreat": (-0.25, 0.0),
    }

    def _apply(self, plan: dict) -> None:
        """Translate a cloud plan into actuator calls. Conservative by default."""
        action = str(plan.get("action", "idle")).lower()
        say = str(plan.get("say", "")).strip()
        if say:
            self.say(say)
        if action in self._MOVES:
            linear, angular = self._MOVES[action]
            accepted, why = self.guard.drive(linear, angular, f"plan:{action}")
            if not accepted:
                log.info("[plan] %s not performed: %s", action, why)
        elif action == "grasp":
            self.guard.stop()
            self.robot.gripper(closed=True)
        elif action == "release":
            self.robot.gripper(closed=False)
        else:
            self.guard.stop()

    async def _describe_now(self) -> str:
        """What the local vision model sees right now. Used by behaviours."""
        img = self.last_image_b64
        if img is None and not self.static_image_b64:
            _, img = self.camera.latest_jpeg_b64()
        img = img or self.static_image_b64
        if not img:
            return ""
        return await self.vlm.describe(img) or ""

    # ---------------- runtime reconfiguration ----------------

    # Keys the agent may change, mapped to (coercion, applier). Anything not
    # listed here is rejected by tools.SETTABLE before it ever gets this far.
    def apply_setting(self, key: str, value: str) -> dict:
        try:
            if key == "vlm_interval":
                v = max(0.2, float(value))
                self.vlm_interval = v
            elif key == "cloud_min_interval":
                v = max(1.0, float(value))
                self.cloud_min_interval = v
            elif key in ("cam_width", "cam_height"):
                v = int(value)
                if not 64 <= v <= 4096:
                    return {"ok": False, "error": "must be between 64 and 4096"}
                setattr(self, key, v)
                self._restart_camera()
            else:
                return {"ok": False, "error": f"{key!r} is not settable"}
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": f"bad value for {key}: {exc}"}
        return {"ok": True, "key": key, "value": v}

    def _restart_camera(self) -> None:
        """Camera geometry is fixed at open time, so a resize means a reopen."""
        if self.static_image_b64:
            return
        with contextlib.suppress(Exception):
            self.camera.stop()
        self.camera = build_camera(
            self.cfg.camera.device, self.cam_width,
            self.cam_height, self.cfg.camera.fps,
        )
        self.toolbox._camera = self.camera
        self.camera.start()

    # ---------------- conversational agent ----------------

    async def instruct(self, text: str, with_frame: bool = True) -> dict:
        """Run one instruction through the cloud agent with tools available."""
        frame = None
        if with_frame and self.cfg.cloud.send_frames:
            frame = self.last_image_b64
            if frame is None and not self.static_image_b64:
                _, frame = self.camera.latest_jpeg_b64()
            frame = frame or self.static_image_b64
        result = await self.agent.run(text, image_b64=frame)
        if not result.get("ok"):
            log.warning("[agent] %s", result.get("error", "failed"))
            if self.console:
                self.console.event("err", f"agent: {result.get('error', 'failed')}")
        reply = result.get("reply", "")
        if reply:
            log.info("[agent] %s", reply)
            self.say(reply)
        return result

    # ---------------- voice command and control ----------------

    async def listen_forever(self) -> None:
        """Mic -> transcript -> wake word -> agent -> spoken reply.

        next_utterance() blocks on the arecord pipe, so it runs in a worker
        thread; the perception loop keeps its cadence regardless.
        """
        v = self.voice
        if v is None:
            return
        if not v.available:
            log.warning("voice enabled but no mic/STT backend - voice control disabled")
            return
        v.start()
        await asyncio.to_thread(v.calibrate)
        wake = self.cfg.voice.wake_word
        log.info("voice control ready (wake word: %r)", wake or "<always on>")
        # People say "hey Hank" and then *pause* before the actual request, so
        # VAD correctly ends the utterance on the wake word alone. Rather than
        # fight that, a wake word opens a window during which the next thing
        # said is taken as the command. The window also stays open briefly
        # after a reply, so a follow-up does not need the wake word again.
        open_until = 0.0
        try:
            while not self._stop.is_set():
                text = await asyncio.to_thread(v.next_utterance)
                if not text:
                    continue
                log.info("[hear] %s", text)
                if self.console:
                    self.console.event("hear", f"heard: {text}")

                command = match_wake_word(text, wake)
                listening = time.monotonic() < open_until
                if command is None:
                    if not listening:
                        self.transcript.append({"heard": text, "acted": False})
                        continue
                    command = text.strip()  # inside the window: no wake word needed
                if not command:
                    open_until = time.monotonic() + self.cfg.voice.follow_up_s
                    self.say("I'm listening.")
                    continue

                self.transcript.append({"heard": text, "acted": True})
                open_until = 0.0
                await self.instruct(command)
                open_until = time.monotonic() + self.cfg.voice.follow_up_s
        finally:
            v.stop()

    def say(self, text: str) -> None:
        """Speak, with the mic gated so the robot does not transcribe itself."""
        if self.voice:
            self.voice.mute(True)
        try:
            self.speaker.say(text)
        finally:
            if self.voice:
                self.voice.mute(False)

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
        log.info(
            "cloud provider=%s enabled=%s send_frames=%s",
            self.cfg.cloud.provider, self.cloud.enabled, self.cfg.cloud.send_frames,
        )
        if self.console is not None:
            self.console.start()
        tasks = [
            asyncio.create_task(self.perceive_forever()),
            asyncio.create_task(self.status_forever()),
        ]
        if self.voice is not None:
            tasks.append(asyncio.create_task(self.listen_forever()))
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
            if self.console is not None:
                self.console.stop()
            self.guard.close()
            self.robot.stop()
            if not self.static_image_b64:
                self.camera.stop()
            self.narrator.close()
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
    p.add_argument("--say", metavar="TEXT",
                   help="run one instruction through the cloud agent, then exit")
    p.add_argument("--voice", action="store_true", help="enable spoken command and control")
    p.add_argument("--console", action="store_true", help="serve the browser console")
    p.add_argument("--console-port", type=int, help="console port (default 8080)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    # CLI flags are just a friendlier way to set the environment the frozen
    # config reads, so there is exactly one place a setting comes from.
    import os
    if args.voice:
        os.environ["JETTANK_VOICE"] = "1"
    if args.console or args.console_port:
        os.environ["JETTANK_CONSOLE"] = "1"
    if args.console_port:
        os.environ["JETTANK_CONSOLE_PORT"] = str(args.console_port)

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
            frame = img if cfg.cloud.send_frames else None
            log.info("[plan] %s", await loop.cloud.think([text], image_b64=frame))
        if not static:
            loop.camera.stop()
        await loop.vlm.aclose()
        await loop.cloud.aclose()
        return 0

    if args.say:
        if not loop.cloud.enabled:
            log.error("no cloud provider configured; set JETTANK_CLOUD_PROVIDER")
            return 1
        if static is None:
            loop.camera.start()
            await asyncio.sleep(1.0)
        result = await loop.instruct(args.say)
        if result.get("tools_used"):
            log.info("[tool] %s", ", ".join(result["tools_used"]))
        print(result.get("reply") or f"(no reply: {result.get('error', 'unknown')})")
        if static is None:
            loop.camera.stop()
        loop.guard.close()
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
