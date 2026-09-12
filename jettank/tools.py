"""Tools the cloud model may call.

Design rules:
  * Every tool returns a plain dict; failures are reported, never raised, so a
    bad call degrades into a message the model can reason about.
  * Motion goes through MotionGuard and can be refused. The model is told why.
  * There is deliberately NO tool to enable motion or clear an E-STOP. Those
    are local-operator actions.
"""
from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)


def _piper_ready() -> bool:
    try:
        import piper  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True

# ---- schemas (Anthropic tool-use format) ----

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "get_status",
        "description": (
            "Read the robot's current state: what hardware is present, whether motion is "
            "permitted, battery, and recent observations. Call this first if unsure."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "capture_image",
        "description": "Capture a fresh frame from the robot's camera and return it for you to look at.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "look",
        "description": (
            "Aim the camera using its pan/tilt servos. pan: -90 (left) to 90 (right). "
            "tilt: -45 (down) to 45 (up). Both relative to centre."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pan": {"type": "number", "description": "degrees, -90..90"},
                "tilt": {"type": "number", "description": "degrees, -45..45"},
            },
            "required": ["pan", "tilt"],
        },
    },
    {
        "name": "drive",
        "description": (
            "Drive the treads. linear: forward/back, angular: turn rate. Values are clamped "
            "to safe limits and a watchdog stops motion if you do not send another command "
            "within about a second. May be refused if motion is disabled - that is normal and "
            "not an error you should retry endlessly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "linear": {"type": "number", "description": "-1..1, forward positive"},
                "angular": {"type": "number", "description": "-1..1, left positive"},
                "reason": {"type": "string", "description": "why you are moving"},
            },
            "required": ["linear", "angular"],
        },
    },
    {
        "name": "stop",
        "description": "Stop all movement immediately.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "speak",
        "description": "Say something aloud through the robot's speaker.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "enroll_face",
        "description": (
            "Enrol a person's face under a name. Captures several frames and stores face "
            "embeddings locally on the robot. Ask the person to look at the camera and hold "
            "still. Faces never leave the device."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "the person's name"},
                "samples": {"type": "integer", "description": "frames to capture, default 5"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "identify_face",
        "description": "Look at the current camera view and report which enrolled people are visible.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_faces",
        "description": "List everyone currently enrolled.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_config",
        "description": (
            "Change a runtime setting on the robot. Allowed keys: vlm_interval (seconds between "
            "local vision passes), cam_width, cam_height, cloud_min_interval. Takes effect "
            "immediately; not persisted across restarts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
        },
    },
]

SETTABLE = {"vlm_interval", "cam_width", "cam_height", "cloud_min_interval"}


class ToolBox:
    """Dispatches model tool calls to the robot."""

    def __init__(self, loop, guard, camera, faces, speaker) -> None:
        self._loop = loop
        self._guard = guard
        self._camera = camera
        self._faces = faces
        self._speaker = speaker

    def dispatch(self, name: str, args: dict) -> dict:
        fn: Callable[..., dict] | None = getattr(self, f"_t_{name}", None)
        if fn is None:
            return {"ok": False, "error": f"unknown tool {name!r}"}
        try:
            return fn(**(args or {}))
        except TypeError as exc:
            return {"ok": False, "error": f"bad arguments for {name}: {exc}"}
        except Exception as exc:  # noqa: BLE001 - a tool fault must not kill the loop
            log.exception("tool %s failed", name)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # ---- implementations ----
    def _t_get_status(self) -> dict:
        st = {"ok": True, **self._guard.status()}
        st["recent_observations"] = list(self._loop.observations)[-5:]
        st["camera"] = getattr(self._camera, "_device", "unknown")
        st["enrolled_faces"] = self._faces.list_names() if self._faces else []
        return st

    def _t_capture_image(self) -> dict:
        seq, b64 = self._camera.latest_jpeg_b64()
        if not b64:
            return {"ok": False, "error": "no frame available from the camera"}
        return {"ok": True, "seq": seq, "image_b64": b64}

    def _t_look(self, pan: float, tilt: float) -> dict:
        drv = self._loop.robot
        if not hasattr(drv, "look"):
            return {"ok": False, "error": "this robot has no pan/tilt servos wired up yet"}
        drv.look(float(pan), float(tilt))
        return {"ok": True, "pan": float(pan), "tilt": float(tilt)}

    def _t_drive(self, linear: float, angular: float, reason: str = "") -> dict:
        accepted, note = self._guard.drive(float(linear), float(angular), reason)
        return {"ok": accepted, "detail": note}

    def _t_stop(self) -> dict:
        self._guard.stop()
        return {"ok": True, "detail": "stopped"}

    def _t_speak(self, text: str) -> dict:
        if not text.strip():
            return {"ok": False, "error": "nothing to say"}
        return self._speaker.say(text)

    def _t_enroll_face(self, name: str, samples: int = 5) -> dict:
        if not self._faces:
            return {"ok": False, "error": "face recognition is not available on this robot"}
        return self._faces.enroll(name, int(samples), self._camera)

    def _t_identify_face(self) -> dict:
        if not self._faces:
            return {"ok": False, "error": "face recognition is not available on this robot"}
        return self._faces.identify(self._camera)

    def _t_list_faces(self) -> dict:
        if not self._faces:
            return {"ok": False, "error": "face recognition is not available on this robot"}
        return {"ok": True, "names": self._faces.list_names()}

    def _t_set_config(self, key: str, value: str) -> dict:
        if key not in SETTABLE:
            return {"ok": False, "error": f"{key!r} is not settable; allowed: {sorted(SETTABLE)}"}
        return self._loop.apply_setting(key, value)


class Speaker:
    """Text to speech, best available engine first.

    piper     - neural, offline, runs fine on the Orin's CPU. Sounds human.
    espeak-ng - formant synth from the 1980s. Always available, sounds it.
                Kept only so a missing voice model degrades to *something*.

    Audio is rendered to a WAV on stdout and piped into `aplay -D`, rather
    than letting the engine choose an output: on JetPack the ALSA default is a
    Tegra APE virtual card, and relying on it is how a robot talks into a void.
    """

    def __init__(self, device: str = "auto", voice: str | None = None) -> None:
        from .audio import pick_speaker

        self._device = pick_speaker(device)
        self._engine: str | None = None
        self._piper_voice = self._find_voice(voice)

        self._voice = None          # loaded lazily; see _load_piper
        self._voice_lock = threading.Lock()
        if self._piper_voice and _piper_ready():
            self._engine = "piper"
        else:
            for cand in ("espeak-ng", "espeak"):
                if shutil.which(cand):
                    self._engine = cand
                    break
        log.info("speech: %s%s", self._engine or "none",
                 f" ({Path(self._piper_voice).stem})" if self._engine == "piper" else "")
        if self._engine == "piper":
            # Load the voice off the critical path so the first thing Hank
            # says is not two seconds late.
            threading.Thread(target=self._warm, name="tts-warm", daemon=True).start()

    def _warm(self) -> None:
        try:
            self._load_piper()
        except Exception as exc:  # noqa: BLE001 - fall back at speak time
            log.warning("could not preload voice (%s)", exc)

    @staticmethod
    def _find_voice(voice: str | None) -> str | None:
        """Locate a Piper .onnx voice: explicit path, env, then the voices dir."""
        cand = voice or os.environ.get("JETTANK_TTS_VOICE", "")
        if cand and Path(cand).exists():
            return cand
        vdir = Path(__file__).resolve().parent.parent / "voices"
        if cand:
            named = vdir / f"{cand}.onnx"
            if named.exists():
                return str(named)
            log.warning("voice %r not found in %s", cand, vdir)
        found = sorted(vdir.glob("*.onnx")) if vdir.is_dir() else []
        return str(found[0]) if found else None

    def _load_piper(self):
        """Load the voice once and keep it.

        Shelling out to `python -m piper` costs ~2.5s per utterance, almost all
        of it importing onnxruntime and re-reading the 61MB model - far more
        than the synthesis itself (RTF ~0.1). Held in-process, the same call
        starts speaking in well under a second.
        """
        with self._voice_lock:
            if self._voice is None:
                from piper import PiperVoice

                t = time.monotonic()
                self._voice = PiperVoice.load(self._piper_voice)
                log.info("loaded voice %s in %.1fs",
                         Path(self._piper_voice).stem, time.monotonic() - t)
            return self._voice

    def _speak_piper(self, text: str) -> dict:
        """Synthesise and play as chunks arrive, so speech starts immediately."""
        voice = self._load_piper()
        proc = None
        played = 0
        try:
            for chunk in voice.synthesize(text):
                pcm = getattr(chunk, "audio_int16_bytes", None)
                if pcm is None:  # older piper returned raw arrays
                    pcm = bytes(chunk.audio_int16_array)
                if proc is None:
                    proc = subprocess.Popen(
                        ["aplay", "-q", "-D", self._device, "-f", "S16_LE",
                         "-r", str(getattr(chunk, "sample_rate", 22050)),
                         "-c", str(getattr(chunk, "sample_channels", 1)), "-t", "raw", "-"],
                        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                proc.stdin.write(pcm)
                played += len(pcm)
        finally:
            if proc is not None:
                with contextlib.suppress(BrokenPipeError, OSError):
                    proc.stdin.close()
                proc.wait(timeout=120)
        if not played:
            return {"ok": False, "error": "TTS produced no audio", "spoken_text": text}
        return {"ok": True, "spoken_text": text}

    def _render(self, text: str) -> bytes:
        """WAV bytes from the fallback engine."""
        return subprocess.run([self._engine, "--stdout", text],
                              capture_output=True, timeout=30).stdout

    def say(self, text: str) -> dict:
        if self._engine is None:
            log.info("[speak] %s", text)
            return {"ok": False, "error": "no TTS engine available (pip install piper-tts, or apt install espeak-ng)",
                    "spoken_text": text}
        try:
            if self._engine == "piper":
                return self._speak_piper(text)
            wav = self._render(text)
            if not wav:
                return {"ok": False, "error": "TTS produced no audio",
                        "spoken_text": text}
            subprocess.run(["aplay", "-q", "-D", self._device], input=wav,
                           timeout=60, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {"ok": True, "spoken_text": text}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "TTS timed out", "spoken_text": text}
        except FileNotFoundError as exc:
            return {"ok": False, "error": f"audio playback unavailable: {exc}",
                    "spoken_text": text}
