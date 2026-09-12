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
import random
import threading
import time
from collections import deque

from . import config as cfg_mod
from .audio import Transcriber, VoiceListener, match_wake_word
from .board import BoardReader
from .camera import build_camera
from .codegen import BehaviorRunner, BehaviorStore, BehaviorWriter
from .console import Console
from .control import KILL_EXIT_CODE, classify
from .sandbox import Sandbox, docker_available
from .skills import SkillStore
from .sysinfo import collect as collect_sysinfo, describe as describe_sysinfo
from .cloud import AgentSession, CloudAgent, register_tools
from .faces import FaceEngine
from .live import Reloader
from .narrator import Narrator
from .power import BatteryMonitor
from .robot import build as build_robot
from .safety import MotionGuard
from .tools import TOOL_SCHEMAS, Speaker, ToolBox
from .vlm import LocalVLM

log = logging.getLogger("jettank")

# Varied so it does not become wallpaper - the same sentence every time stops
# being heard after a day.
REST_PHRASES = [
    "I'll rest here. Say hey Hank when you need me.",
    "Going quiet. Just say hey Hank.",
    "I'll stop listening now. Hey Hank brings me back.",
    "Resting. Say hey Hank whenever.",
]


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
        # Read-only telemetry from the expansion board: battery, tilt, whether
        # the chassis is actually moving. Independent of the motion path, which
        # stays dry-run - this only listens.
        self.board = BoardReader()
        self.board.start()

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
        self.vlm_idle_interval = cfg.vlm.idle_interval_s
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

        # Conversation state. Hank is either resting (wake word required) or
        # awake (everything he hears is addressed to him, and he remembers the
        # exchange). He announces the transition back so it is never ambiguous
        # which mode he is in - a robot that has silently stopped listening is
        # indistinguishable from one that is broken.
        self._awake_until = 0.0
        self._awake = False
        self._work: asyncio.Task | None = None
        self._killed = False
        self.reloader = Reloader(self)
        self.battery = BatteryMonitor(self)
        self._arm_requested = 0.0
        self.reloader = Reloader(self)
        self.battery = BatteryMonitor(self)

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
            # Awake means someone is engaged and wants quick answers; resting
            # means the room is empty and every inference is battery burned to
            # describe nothing.
            interval = self.vlm_interval if self._awake else self.vlm_idle_interval
            started = time.monotonic()
            if self.static_image_b64:
                seq, img = self.frames_seen + 1, self.static_image_b64
            else:
                seq, img = self.camera.latest_jpeg_b64()
            # A vanished camera can still hand back a short or stale buffer,
            # and the VLM answers that with a 500. Cheapest possible guard: a
            # real JPEG starts with SOI and is never this small.
            if img and len(img) > 1024:
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
        # Do not pay for cloud reasoning about an empty room that nobody asked
        # about. Perception keeps running locally regardless.
        if self.cfg.cloud.autonomy_when_awake_only and not self._awake:
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
            # The planner runs on a timer, not in response to anyone. Speaking
            # here talks over replies and narrates to an empty room.
            if self.cfg.cloud.autonomy_speaks:
                self.say(say)
            else:
                log.info("[plan] (unspoken) %s", say)
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

    async def instruct(self, text: str, with_frame: bool = True,
                       remember: bool = False) -> dict:
        """Run one instruction through the cloud agent with tools available."""
        frame = None
        if with_frame and self.cfg.cloud.send_frames:
            frame = self.last_image_b64
            if frame is None and not self.static_image_b64:
                _, frame = self.camera.latest_jpeg_b64()
            frame = frame or self.static_image_b64
        result = await self.agent.run(text, image_b64=frame, remember=remember)
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
        """Mic -> control words -> wake word -> agent, without ever going deaf.

        The instruction runs as a separate task rather than being awaited here.
        That matters: awaiting it meant Hank stopped listening for the whole of
        inference, so nothing could interrupt him - which makes a stop word
        impossible by construction.
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
        log.info("voice control ready (wake word: %r; stop/kill always active)", wake)
        try:
            while not self._stop.is_set():
                text = await asyncio.to_thread(v.next_utterance)
                if not text:
                    continue
                log.info("[hear] %s", text)
                if self.console:
                    self.console.event("hear", f"heard: {text}")

                # Control words are checked first and work whether Hank is
                # awake, resting, thinking or speaking. An emergency control
                # that only works in one mode is not one.
                control = classify(text)
                if control == "kill":
                    await self._kill("voice")
                    return
                if control == "stop":
                    self._interrupt("voice")
                    continue
                if control == "disarm":
                    self._set_motion(False, "voice")
                    continue
                if control == "arm":
                    self._set_motion(True, "voice")
                    continue
                if control == "reload":
                    self.reload_now("voice")
                    continue

                command = match_wake_word(text, wake)
                if command is None:
                    if not self._awake:
                        self.transcript.append({"heard": text, "acted": False})
                        continue
                    command = text.strip()   # awake: take it as addressed to him
                elif not self._awake:
                    self._wake()

                if not command:
                    self._awake_until = time.monotonic() + self.cfg.voice.follow_up_s
                    self.say("I'm listening.")
                    continue

                self.transcript.append({"heard": text, "acted": True})
                self._start_work(command)
        finally:
            v.stop()

    # ---------------- work in flight ----------------

    def _start_work(self, command: str) -> None:
        """Run an instruction in the background so listening continues.

        A new instruction while one is running replaces it - if you interrupt
        Hank with a different question, you meant the new one.
        """
        if self._work and not self._work.done():
            log.info("[voice] superseding the previous request")
            self._cancel_work()
        self._awake_until = time.monotonic() + self.cfg.voice.conversation_timeout_s
        self._work = asyncio.create_task(self._run_work(command))

    async def _run_work(self, command: str) -> None:
        try:
            await self.instruct(command, remember=True)
        except asyncio.CancelledError:
            log.info("[voice] request cancelled")
            raise
        finally:
            # Extend the window from when the answer *finished*, not when it
            # was asked, so a long task does not eat the follow-up window.
            self._awake_until = max(
                self._awake_until,
                time.monotonic() + self.cfg.voice.conversation_timeout_s)

    def _cancel_work(self) -> None:
        if self._work and not self._work.done():
            self._work.cancel()
        self._work = None

    def _interrupt(self, source: str) -> None:
        """STOP: abandon what is in flight, keep listening.

        Deliberately does not change the wake state. Saying stop is engagement,
        not dismissal - going deaf immediately afterwards would be wrong.
        """
        log.info("[control] STOP from %s", source)
        if self.console:
            self.console.event("err", "STOP - cancelled in-flight work")
        self.speaker.abort()          # cut speech mid-sentence
        self.guard.stop()             # and halt motion
        self._cancel_work()
        self._awake_until = max(self._awake_until,
                                time.monotonic() + self.cfg.voice.follow_up_s)

    async def _kill(self, source: str) -> None:
        """KILL: stop the process in a way systemd will not undo.

        Exits with KILL_EXIT_CODE, which the unit lists in
        RestartPreventExitStatus. Coming back is then a deliberate shell
        action - `sudo systemctl start hank` - which is the point: a kill
        switch that the machine can reverse on its own is not a kill switch.
        """
        log.warning("[control] KILL from %s - shutting down", source)
        if self.console:
            self.console.event("err", "KILL - shutting down; restart from a shell")
        self.guard.estop("kill word")
        self._cancel_work()
        self.speaker.abort()
        # Said synchronously, before teardown, so it is actually heard.
        with contextlib.suppress(Exception):
            self.speaker.say("Shutting down. You will have to restart me from a terminal.")
        self._killed = True
        self.request_stop()

    def _set_motion(self, enable: bool, source: str) -> None:
        """Arm or disarm the motors. Arming reaches here only from a human.

        The model can request arming (see the set_motion tool) but cannot
        perform it: a request just makes Hank ask out loud, and the arming
        itself happens when a person says the control phrase. Disarming has no
        such restriction - anything may stop the robot.
        """
        if enable:
            self.guard.clear_estop()
            self.guard.enable(True)
            self._arm_requested = 0.0
            log.warning("[control] MOTION ARMED by %s", source)
            if self.console:
                self.console.event("err", f"motion ARMED ({source})")
            self.say("Motion armed. Say Hank disarm to stop me moving.")
        else:
            self.guard.enable(False)
            self.guard.stop()
            log.info("[control] motion disarmed by %s", source)
            if self.console:
                self.console.event("agent", f"motion disarmed ({source})")
            self.say("Motion disarmed.")

    def request_arm(self, reason: str = "") -> dict:
        """Called by the set_motion tool when Hank wants to move.

        Does not arm anything. It asks, and records that an ask is outstanding
        so the reply can say so honestly.
        """
        if self.guard.status().get("motion_enabled"):
            return {"ok": True, "detail": "motion is already armed"}
        self._arm_requested = time.monotonic()
        return {"ok": False, "awaiting_human": True,
                "detail": "I cannot arm my own motors. A person has to say "
                          "'Hank arm motion' out loud. Tell them that, and why "
                          "you want to move."}

    def reload_now(self, source: str = "manual") -> dict:
        """Re-apply prompts, settings and the hardware map in place."""
        try:
            result = self.reloader.reload()
        except Exception as exc:  # noqa: BLE001 - a bad edit must not kill Hank
            log.exception("reload failed")
            if self.console:
                self.console.event("err", f"reload failed: {exc}")
            self.say("I could not reload. The old settings are still running.")
            return {"ok": False, "error": str(exc)}
        bits = [k for k in ("modules", "prompts", "settings", "hardware")
                if result.get(k)]
        log.info("[live] reload from %s: %s", source, bits or "no changes")
        if self.console:
            self.console.event("agent", f"reloaded ({', '.join(bits) or 'no changes'})")
        if source == "voice":
            self.say("Reloaded." if bits else "Nothing had changed.")
        return {"ok": True, **result}

    async def peripherals_forever(self) -> None:
        """Reconnect the camera and mic when they reappear.

        This robot gets its USB replugged constantly - swapping a LIDAR for a
        flash drive, moving devices between ports to chase power. Requiring a
        service restart every time turns a ten second physical change into a
        twenty second reboot plus a lost conversation, and it is avoidable:
        neither device needs anything but reopening.
        """
        import os

        while not self._stop.is_set():
            await asyncio.sleep(5.0)

            # Camera: present on disk but handing back nothing usable.
            if not self.static_image_b64:
                _, img = self.camera.latest_jpeg_b64()
                healthy = bool(img) and len(img) > 1024
                if not healthy and os.path.exists(self.cfg.camera.device):
                    log.info("[peripherals] camera looks dead but %s exists - reopening",
                             self.cfg.camera.device)
                    try:
                        self._restart_camera()
                    except Exception as exc:  # noqa: BLE001
                        log.debug("camera reopen failed: %s", exc)

            # Microphone: arecord exits when its device disappears.
            v = self.voice
            if v is not None and v.available:
                proc = getattr(v, "_proc", None)
                if proc is not None and proc.poll() is not None:
                    log.info("[peripherals] microphone went away - reopening")
                    try:
                        v.stop()
                        v.start()
                        await asyncio.to_thread(v.calibrate)
                        log.info("[peripherals] microphone back")
                    except Exception as exc:  # noqa: BLE001
                        log.debug("mic reopen failed: %s", exc)

    async def battery_forever(self) -> None:
        """Watch the pack and act before a brown-out corrupts the disk.

        Runs regardless of whether anyone is talking to him: the risk is
        highest exactly when he has been left running unattended.
        """
        while not self._stop.is_set():
            await asyncio.sleep(1.0)
            t = self.board.latest()
            if t is None:
                continue
            self.battery.sample(t.battery_v)
            message = self.battery.assess()
            if not message:
                continue
            log.warning("[power] %s", message)
            if self.console:
                self.console.event("err", message)

            if self.battery.state == "disarm" and self.guard.status().get("motion_enabled"):
                self.guard.enable(False)
                self.guard.stop()
            self.say(message)

            if self.battery.state == "critical":
                self.guard.enable(False)
                self.guard.stop()
                await asyncio.sleep(4.0)      # let him finish the sentence
                self.battery.shutdown()
                return

    async def watch_forever(self) -> None:
        """Reload automatically when a watched file changes on disk.

        Polling rather than inotify: the interval is seconds, the file list is
        a dozen entries, and it works the same over a network mount.
        """
        if not self.cfg.live.watch:
            return
        self.reloader.changed_files()          # prime, do not fire on startup
        log.info("watching for live edits every %.0fs", self.cfg.live.interval_s)
        while not self._stop.is_set():
            await asyncio.sleep(self.cfg.live.interval_s)
            try:
                changed = await asyncio.to_thread(self.reloader.changed_files)
            except Exception as exc:  # noqa: BLE001
                log.debug("watch failed: %s", exc)
                continue
            if changed:
                log.info("[live] changed on disk: %s", ", ".join(changed))
                self.reload_now("watch")

    def _wake(self) -> None:
        self._awake = True
        log.info("[voice] awake - conversation open, no wake word needed")
        if self.console:
            self.console.event("agent", "awake, listening")

    async def rest_watcher(self) -> None:
        """Put Hank back to sleep after a quiet spell, out loud.

        Announced rather than silent: the failure mode of a wake word is a
        person talking to a robot that stopped listening some time ago, with
        nothing to indicate when. One short line removes that ambiguity.
        """
        while not self._stop.is_set():
            await asyncio.sleep(1.0)
            if not self._awake or time.monotonic() < self._awake_until:
                continue
            self._awake = False
            self._awake_until = 0.0
            # Resting changes how Hank LISTENS, nothing else. Anything already
            # running - a behaviour, an agent turn, a long capture - continues
            # and still reports when it finishes. Only STOP cancels work.
            self.agent.reset()          # forget the conversation, not the task
            log.info("[voice] resting - wake word required again")
            if self.console:
                self.console.event("agent", "resting; say the wake word to start again")
            self.say(random.choice(REST_PHRASES))

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
            asyncio.create_task(self.watch_forever()),
            asyncio.create_task(self.battery_forever()),
            asyncio.create_task(self.peripherals_forever()),
        ]
        if self.voice is not None:
            tasks.append(asyncio.create_task(self.listen_forever()))
            tasks.append(asyncio.create_task(self.rest_watcher()))
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
            self.board.stop()
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
    # A kill word exits with a code the unit lists in RestartPreventExitStatus,
    # so systemd leaves it down until a human starts it from a shell.
    return KILL_EXIT_CODE if loop._killed else 0


def main() -> int:
    try:
        return asyncio.run(_amain())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
