"""Configuration for the Jettank perception/agent loop.

Values come from environment variables so the same image runs on the robot and
on a laptop for testing. Nothing here is secret-bearing at import time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


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
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass(frozen=True)
class VLMConfig:
    """Local, on-device vision model. Must stay fast enough for the tight loop."""

    base_url: str = field(default_factory=lambda: _env("JETTANK_VLM_URL", "http://127.0.0.1:11434"))
    model: str = field(default_factory=lambda: _env("JETTANK_VLM_MODEL", "qwen2.5vl:3b"))
    # How often we ask the local model to describe the scene.
    interval_s: float = field(default_factory=lambda: _env_f("JETTANK_VLM_INTERVAL", 1.5))
    timeout_s: float = field(default_factory=lambda: _env_f("JETTANK_VLM_TIMEOUT", 20.0))
    max_tokens: int = 128


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
    max_tokens: int = 512
    # Send the camera frame itself, not just the local VLM's text description.
    # Better fine-detail reasoning, at the cost of imagery leaving the device.
    send_frames: bool = field(default_factory=lambda: _env_b("JETTANK_CLOUD_SEND_FRAMES", True))

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    # Safety: the local loop must keep running even if everything else stalls.
    watchdog_s: float = field(default_factory=lambda: _env_f("JETTANK_WATCHDOG", 5.0))


def load() -> Config:
    return Config()
