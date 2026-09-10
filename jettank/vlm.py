"""Local vision-language model client.

Targets an Ollama-compatible HTTP endpoint, which is how jetson-containers and
Ollama both expose vision models on Jetson. Kept deliberately small: this runs
in the tight loop and must never block it for long.
"""
from __future__ import annotations

import logging
import httpx

log = logging.getLogger(__name__)

SCENE_PROMPT = (
    "You are the eyes of a small tracked robot with a camera, LIDAR and a gripper arm. "
    "Describe what you see in one or two short sentences. "
    "Name objects, their rough position (left/centre/right), and anything that blocks "
    "forward motion. Be concrete and terse. Do not speculate."
)


class LocalVLM:
    def __init__(self, base_url: str, model: str, timeout_s: float, max_tokens: int) -> None:
        self._url = base_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def available(self) -> bool:
        try:
            r = await self._client.get(f"{self._url}/api/tags")
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            log.info("local models present: %s", ", ".join(names) or "(none)")
            return any(n.split(":")[0] == self._model.split(":")[0] for n in names)
        except Exception as exc:  # noqa: BLE001 - availability probe, never fatal
            log.warning("local VLM not reachable at %s: %s", self._url, exc)
            return False

    async def describe(self, image_b64: str, prompt: str = SCENE_PROMPT) -> str | None:
        """One-shot scene description. Returns None on any failure."""
        payload = {
            "model": self._model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {"num_predict": self._max_tokens, "temperature": 0.2},
        }
        try:
            r = await self._client.post(f"{self._url}/api/generate", json=payload)
            r.raise_for_status()
            return (r.json().get("response") or "").strip() or None
        except Exception as exc:  # noqa: BLE001 - a dropped frame must not kill the loop
            log.warning("local VLM inference failed: %s", exc)
            return None

    async def aclose(self) -> None:
        await self._client.aclose()
