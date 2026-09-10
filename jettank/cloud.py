"""Provider-agnostic cloud LLM client.

Deliberately thin: one `think()` call taking the robot's recent observations and
returning a high-level plan. Swapping providers is a config change, so no
decision about Anthropic vs. an internal gateway is baked into the loop.
"""
from __future__ import annotations

import base64
import json
import logging
import httpx

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the high-level planner for a small tracked robot (Yahboom Jettank on a "
    "Jetson Orin Nano). A local vision model reports what the camera sees, and you may "
    "also be given the current camera frame. Trust the image over the text where they "
    "disagree. You decide what the robot should do next.\n\n"
    "You do not control the robot directly and you are NOT a safety system - an "
    "on-board loop handles obstacle stops and arm limits and may override you.\n\n"
    "Reply with strict JSON only:\n"
    '{"assessment": "<one sentence>", "action": "<one of: idle, explore, approach, '
    'retreat, grasp, release, speak>", "target": "<object or empty>", '
    '"say": "<short phrase to speak, or empty>", "confidence": <0.0-1.0>}'
)


class CloudAgent:
    """Wraps whichever cloud model is configured. `provider == "none"` disables it."""

    def __init__(
        self,
        provider: str,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_s: float,
        max_tokens: int,
    ) -> None:
        self._provider = provider.lower()
        self._model = model
        self._key = api_key
        self._max_tokens = max_tokens
        self._base = base_url.rstrip("/") if base_url else self._default_base()
        self._client = httpx.AsyncClient(timeout=timeout_s)

    def _default_base(self) -> str:
        if self._provider == "anthropic":
            return "https://api.anthropic.com"
        return ""

    @property
    def enabled(self) -> bool:
        if self._provider == "none":
            return False
        if not self._key:
            log.warning("cloud provider %r configured but no API key present", self._provider)
            return False
        return True

    @staticmethod
    def _media_type(image_b64: str) -> str:
        """Sniff the format from the decoded header; Anthropic requires the real type."""
        try:
            head = base64.b64decode(image_b64[:16], validate=False)[:4]
        except Exception:  # noqa: BLE001 - fall back to the camera's format
            return "image/jpeg"
        if head.startswith(b"\x89PNG"):
            return "image/png"
        if head.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if head.startswith(b"GIF8"):
            return "image/gif"
        if head[:4] == b"RIFF":
            return "image/webp"
        return "image/jpeg"

    async def think(self, observations: list[str], image_b64: str | None = None) -> dict | None:
        """Send recent observations (and optionally the frame), get a plan back."""
        if not self.enabled:
            return None
        recent = "\n".join(f"- {o}" for o in observations[-8:])
        user = f"Recent observations from the robot's camera:\n{recent}\n\nWhat should it do next?"
        try:
            if self._provider == "anthropic":
                raw = await self._anthropic(user, image_b64)
            else:
                raw = await self._openai_compatible(user, image_b64)
        except Exception as exc:  # noqa: BLE001 - cloud is best-effort by design
            log.warning("cloud call failed: %s", exc)
            return None
        return self._parse(raw)

    async def _anthropic(self, user: str, image_b64: str | None = None) -> str:
        content: list[dict] = []
        if image_b64:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": self._media_type(image_b64),
                    "data": image_b64,
                },
            })
        content.append({"type": "text", "text": user})
        r = await self._client.post(
            f"{self._base}/v1/messages",
            headers={
                "x-api-key": self._key or "",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self._model,
                "max_tokens": self._max_tokens,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": content}],
            },
        )
        r.raise_for_status()
        parts = r.json().get("content", [])
        return "".join(p.get("text", "") for p in parts if p.get("type") == "text")

    async def _openai_compatible(self, user: str, image_b64: str | None = None) -> str:
        if image_b64:
            url = f"data:{self._media_type(image_b64)};base64,{image_b64}"
            user_content: object = [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": user},
            ]
        else:
            user_content = user
        r = await self._client.post(
            f"{self._base}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._key}", "content-type": "application/json"},
            json={
                "model": self._model,
                "max_tokens": self._max_tokens,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _parse(raw: str) -> dict | None:
        if not raw:
            return None
        text = raw.strip()
        # Models often wrap JSON in a fenced block; tolerate that.
        if text.startswith("```"):
            text = text.split("```")[1] if "```" in text[3:] else text.strip("`")
            text = text.removeprefix("json").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if 0 <= start < end:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            log.warning("could not parse cloud reply as JSON: %.120s", raw)
            return None

    async def aclose(self) -> None:
        await self._client.aclose()
