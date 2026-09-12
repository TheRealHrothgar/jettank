"""Configuration for the Jettank perception/agent loop.

Values come from environment variables so the same image runs on the robot and
on a laptop for testing. Nothing here is secret-bearing at import time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_i(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_b(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_f(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class CameraConfig:
    device: str = field(default_factory=lambda: _env("JETTANK_CAMERA", "0"))
    # Resolution dominates VLM cost: prompt tokens scale with pixel count, and
    # on an Orin Nano the prompt eval is the bottleneck, not the decode. 640x480
    # is ample for obstacle-level scene description.
    width: int = field(default_factory=lambda: _env_i("JETTANK_CAM_WIDTH", 640))
    height: int = field(default_factory=lambda: _env_i("JETTANK_CAM_HEIGHT", 480))
    fps: int = field(default_factory=lambda: _env_i("JETTANK_CAM_FPS", 30))


@dataclass(frozen=True)
class VLMConfig:
    """Local, on-device vision model. Must stay fast enough for the tight loop."""

    base_url: str = field(default_factory=lambda: _env("JETTANK_VLM_URL", "http://127.0.0.1:11434"))
    model: str = field(default_factory=lambda: _env("JETTANK_VLM_MODEL", "qwen2.5vl:3b"))
    # How often we ask the local model to describe the scene.
    interval_s: float = field(default_factory=lambda: _env_f("JETTANK_VLM_INTERVAL", 1.5))
    timeout_s: float = field(default_factory=lambda: _env_f("JETTANK_VLM_TIMEOUT", 20.0))
    max_tokens: int = 128
    # A small text-only model that turns structured findings and machine text
    # into spoken English. Local and cheap, so it can sit in front of every
    # utterance without cloud latency or spend.
    narrator_model: str = field(
        default_factory=lambda: _env("JETTANK_NARRATOR_MODEL", "qwen2.5:1.5b"))
    narrator: bool = field(default_factory=lambda: _env_b("JETTANK_NARRATOR", True))


@dataclass(frozen=True)
class CloudConfig:
    """Provider-agnostic cloud reasoning model.

    provider: "anthropic" | "openai" | "none"
    "openai" covers any OpenAI-compatible endpoint, which is how most internal
    gateways (including NVIDIA's) expose models.
    """

    provider: str = field(default_factory=lambda: _env("JETTANK_CLOUD_PROVIDER", "none"))
    base_url: str = field(default_factory=lambda: _env("JETTANK_CLOUD_URL", ""))
    model: str = field(default_factory=lambda: _env("JETTANK_CLOUD_MODEL", "claude-opus-5"))
    api_key_env: str = field(default_factory=lambda: _env("JETTANK_CLOUD_KEY_ENV", "ANTHROPIC_API_KEY"))
    timeout_s: float = field(default_factory=lambda: _env_f("JETTANK_CLOUD_TIMEOUT", 45.0))
    # Never escalate to the cloud more often than this, regardless of events.
    min_interval_s: float = field(default_factory=lambda: _env_f("JETTANK_CLOUD_MIN_INTERVAL", 6.0))
    # The planner only emits a small JSON object, so it stays cheap.
    max_tokens: int = 512
    # The agent path is different: a single tool call can carry a long
    # natural-language code-generation request, and truncation there arrives
    # as a *valid-looking* tool_use block with fields silently missing.
    agent_max_tokens: int = field(
        default_factory=lambda: _env_i("JETTANK_AGENT_MAX_TOKENS", 8192))
    # Generated code needs room for a whole module.
    codegen_max_tokens: int = field(
        default_factory=lambda: _env_i("JETTANK_CODEGEN_MAX_TOKENS", 8192))
    # The autonomous planner runs on a timer whether or not anyone is talking
    # to Hank. Letting it speak means he narrates the room to an empty room
    # every few seconds, which is both irritating and a good way to miss the
    # replies you actually asked for. Off by default: he speaks when spoken to.
    autonomy_speaks: bool = field(
        default_factory=lambda: _env_b("JETTANK_AUTONOMY_SPEAKS", False))
    # Only escalate to the cloud while someone is actually in conversation.
    # Idling at one Opus call with a frame every few seconds costs real money
    # to describe an empty room to nobody.
    autonomy_when_awake_only: bool = field(
        default_factory=lambda: _env_b("JETTANK_AUTONOMY_AWAKE_ONLY", True))
    # Send the camera frame itself, not just the local VLM's text description.
    # Better fine-detail reasoning, at the cost of imagery leaving the device.
    send_frames: bool = field(default_factory=lambda: _env_b("JETTANK_CLOUD_SEND_FRAMES", True))

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


@dataclass(frozen=True)
class VoiceConfig:
    """Spoken command and control. Audio is transcribed on-device; only the
    resulting text reaches the cloud agent."""

    enabled: bool = field(default_factory=lambda: _env_b("JETTANK_VOICE", False))
    # "auto" finds the USB capture card. Do NOT use "default": on JetPack that
    # is a Tegra APE virtual card that returns silence forever.
    mic: str = field(default_factory=lambda: _env("JETTANK_MIC", "auto"))
    stt_backend: str = field(default_factory=lambda: _env("JETTANK_STT_BACKEND", "auto"))
    stt_model: str = field(default_factory=lambda: _env("JETTANK_STT_MODEL", "base.en"))
    # Empty means always-on: every utterance becomes a command. A wake word is
    # strongly preferred on a robot that is also listening to a room.
    wake_word: str = field(default_factory=lambda: _env("JETTANK_WAKE_WORD", "hey hank"))
    # Energy gate, 0..1. Raise it if the robot's own fans keep triggering it.
    threshold: float = field(default_factory=lambda: _env_f("JETTANK_MIC_THRESHOLD", 0.02))
    # Silero needs a real pause to call end-of-utterance. Too short and it
    # splits mid-sentence; too long and the robot feels sluggish.
    silence_ms: int = field(default_factory=lambda: _env_i("JETTANK_MIC_SILENCE_MS", 900))
    # Once woken, Hank stays in conversation and does not need the wake word
    # again. The window restarts after every exchange, so a real back-and-forth
    # never lapses mid-flow; this is the silence after which he rests.
    conversation_timeout_s: float = field(
        default_factory=lambda: _env_f("JETTANK_CONVERSATION_TIMEOUT", 90.0))
    # Kept for the brief window right after a bare wake word, before anything
    # has actually been asked.
    follow_up_s: float = field(default_factory=lambda: _env_f("JETTANK_FOLLOW_UP", 20.0))


@dataclass(frozen=True)
class ConsoleConfig:
    """Browser console: live camera view plus a command box."""

    enabled: bool = field(default_factory=lambda: _env_b("JETTANK_CONSOLE", False))
    # Binds to all interfaces by default because the point is to reach it from
    # a laptop on the same LAN. There is no auth - keep it off untrusted nets.
    host: str = field(default_factory=lambda: _env("JETTANK_CONSOLE_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_i("JETTANK_CONSOLE_PORT", 8080))


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    console: ConsoleConfig = field(default_factory=ConsoleConfig)
    # Safety: the local loop must keep running even if everything else stalls.
    watchdog_s: float = field(default_factory=lambda: _env_f("JETTANK_WATCHDOG", 5.0))


def load() -> Config:
    return Config()
