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

# Imported lazily via a holder so cloud.py has no hard dependency on the robot
# side of the codebase (keeps the stubbed transport tests working).
TOOL_SCHEMAS_REF: list = [[]]


def register_tools(schemas: list) -> None:
    TOOL_SCHEMAS_REF[0] = schemas

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

    async def _messages(self, messages: list[dict], tools: list[dict] | None = None,
                        system: str | None = None) -> dict | None:
        """Raw Messages API call returning the parsed response, or None on failure."""
        body: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system or SYSTEM_PROMPT,
            "messages": messages,
        }
        if tools:
            body["tools"] = tools
        try:
            r = await self._client.post(
                f"{self._base}/v1/messages",
                headers={"x-api-key": self._key or "",
                         "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json=body,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001 - cloud is best-effort
            log.warning("cloud messages call failed: %s", exc)
            return None

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


AGENT_SYSTEM_PROMPT = (
    "You are the operator of a small tracked robot (a Yahboom Jettank on a Jetson Orin Nano). "
    "You can see through its camera, speak through its speaker, aim its pan/tilt camera, drive "
    "its treads, enrol and recognise faces, and adjust a few runtime settings.\n\n"
    "You are NOT the safety system. An on-board guard clamps speeds, stops the robot if you go "
    "quiet, and can refuse motion outright. If a drive call is refused because motion is "
    "disabled, accept it and say so - do not retry in a loop.\n\n"
    "Be useful and concrete. Prefer looking before moving. When enrolling a face, tell the person "
    "what to do, capture, then confirm. Keep spoken output short and natural - it is read aloud.\n\n"
    "Call get_status first if you are unsure of the robot's state."
)


class AgentSession:
    """Runs a tool-use conversation between the cloud model and the robot.

    Kept separate from `CloudAgent.think` so the fast perception loop stays a
    simple one-shot call; this is the slower, interactive path.
    """

    def __init__(self, agent: "CloudAgent", toolbox, max_turns: int = 8) -> None:
        self._agent = agent
        self._tools = toolbox
        self._max_turns = max_turns
        self.transcript: list[dict] = []

    async def run(self, instruction: str, image_b64: str | None = None) -> dict:
        """Give the model an instruction; let it drive until it is done."""
        if not self._agent.enabled:
            return {"ok": False, "error": "cloud agent is not configured"}

        content: list[dict] = []
        if image_b64:
            content.append({
                "type": "image",
                "source": {"type": "base64",
                           "media_type": self._agent._media_type(image_b64),
                           "data": image_b64},
            })
        content.append({"type": "text", "text": instruction})
        messages: list[dict] = [{"role": "user", "content": content}]

        used: list[str] = []
        for turn in range(self._max_turns):
            reply = await self._agent._messages(messages, tools=TOOL_SCHEMAS_REF[0],
                                                system=AGENT_SYSTEM_PROMPT)
            if reply is None:
                return {"ok": False, "error": "cloud call failed", "tools_used": used}

            blocks = reply.get("content", [])
            messages.append({"role": "assistant", "content": blocks})
            calls = [b for b in blocks if b.get("type") == "tool_use"]
            said = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()

            if not calls:
                return {"ok": True, "reply": said, "tools_used": used, "turns": turn + 1}

            results = []
            for call in calls:
                name, args = call.get("name", ""), call.get("input", {}) or {}
                log.info("[agent] tool %s(%s)", name, args)
                out = self._tools.dispatch(name, args)
                used.append(name)
                # Images come back as a real image block so the model can look at them.
                if name == "capture_image" and out.get("ok") and out.get("image_b64"):
                    payload: object = [
                        {"type": "image",
                         "source": {"type": "base64",
                                    "media_type": self._agent._media_type(out["image_b64"]),
                                    "data": out["image_b64"]}},
                        {"type": "text", "text": "current camera view"},
                    ]
                else:
                    payload = json.dumps(out)
                results.append({"type": "tool_result", "tool_use_id": call.get("id"),
                                "content": payload})
            messages.append({"role": "user", "content": results})

        return {"ok": False, "error": f"gave up after {self._max_turns} turns",
                "tools_used": used}
