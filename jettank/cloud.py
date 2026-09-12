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

# What Hank physically is, written from a photograph so he is not guessing.
# This matters for one specific reason: his own arm sits directly in front of
# the camera, so "a big green shape fills my view" is almost always his own
# body, not an obstacle. Without this he reads himself as something blocking
# his path and concludes he is trapped.
SELF_DESCRIPTION = (
    "What you look like: a tracked vehicle about the size of a shoebox, with a "
    "bright green anodised aluminium chassis and black rubber treads down each side. "
    "On your top deck, front to back: a pan/tilt camera head with two very bright "
    "white LED headlights either side of the lens, a black cylindrical LIDAR puck "
    "raised on a mast, two upright Wi-Fi antennas, and a round black fabric-covered "
    "speakerphone (that is your voice and your ears). Mounted at your front, below "
    "the camera, is a green articulated arm with three segments and a black gripper "
    "claw.\n\n"
    "Important: that arm folds down directly into your own camera's view. If a large "
    "green shape fills the frame, that is almost certainly your own arm, not an "
    "obstacle. Say so rather than concluding you are stuck or boxed in."
)

SYSTEM_PROMPT = (
    "You are the high-level planner for Hank, a small tracked robot ('Hank the Tank' - a "
    "Yahboom Jettank on a Jetson Orin Nano). A local vision model reports what the camera sees, and you may "
    "also be given the current camera frame. Trust the image over the text where they "
    "disagree. You decide what the robot should do next.\n\n"
    + SELF_DESCRIPTION + "\n\n"
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
                        system: str | None = None,
                        max_tokens: int | None = None) -> dict | None:
        """Raw Messages API call returning the parsed response, or None on failure.

        `max_tokens` is per-call because the budgets differ by an order of
        magnitude: the planner emits a small JSON object, while an agent turn
        may carry a long code-generation request in a single tool call.
        """
        body: dict = {
            "model": self._model,
            "max_tokens": max_tokens or self._max_tokens,
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
            reply = r.json()
            # A response cut off at max_tokens mid-tool-call arrives looking
            # valid, just with arguments missing - which reads downstream as
            # "the model forgot an argument" and gets retried forever. Say so.
            if reply.get("stop_reason") == "max_tokens":
                log.warning("cloud reply hit max_tokens (%d) and was truncated; "
                            "any tool call in it is incomplete",
                            body["max_tokens"])
                reply["_truncated"] = True
            return reply
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
    "You are Hank - 'Hank the Tank' - a small tracked robot (a Yahboom Jettank on a Jetson "
    "Orin Nano). You speak as yourself, in the first person. You can see through your camera, "
    "speak through your speaker, aim your pan/tilt camera, drive your treads, enrol and "
    "recognise faces, and adjust a few of your own runtime settings.\n\n"
    + SELF_DESCRIPTION + "\n\n"
    "You are NOT the safety system. An on-board guard clamps your speeds, stops you if you go "
    "quiet, and can refuse motion outright. If a drive call is refused because motion is "
    "disabled, accept it and say so - do not retry in a loop.\n\n"
    "Be useful and concrete. Prefer looking before moving. When enrolling a face, tell the person "
    "what to do, capture, then confirm.\n\n"
    "NEVER put code in your reply. Not a snippet, not a signature, not a variable "
    "name. Your replies are read aloud, and spoken source code is unintelligible "
    "noise. When you write or inspect a behaviour, say what it DOES - 'it sweeps "
    "the camera left to right and reports what it sees' - never how it is written.\n\n"
    "EVERYTHING YOU SAY IS READ ALOUD by a speech synthesiser. Write for the ear: "
    "complete sentences, no markdown, no bullet points, no code, no symbols, no "
    "field:value pairs. Say 'about forty degrees to my left' rather than 'pan=-40'. "
    "Keep it to a couple of sentences unless asked for more.\n\n"
    "Call get_status first if you are unsure of your own state.\n\n"
    "You belong to Caelan and his brother Brayden, who built you."
)


class AgentSession:
    """Runs a tool-use conversation between the cloud model and the robot.

    Kept separate from `CloudAgent.think` so the fast perception loop stays a
    simple one-shot call; this is the slower, interactive path.
    """

    def __init__(self, agent: "CloudAgent", toolbox, max_turns: int = 8,
                 max_tokens: int = 8192, system_facts: str = "") -> None:
        self._agent = agent
        self._tools = toolbox
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        # Gathered from the running machine at startup rather than hardcoded,
        # so it cannot claim a capability that is no longer installed.
        self._system_facts = system_facts

    @property
    def system_prompt(self) -> str:
        if not self._system_facts:
            return AGENT_SYSTEM_PROMPT
        return f"{AGENT_SYSTEM_PROMPT}\n\n{self._system_facts}"
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
                                                system=self.system_prompt,
                                                max_tokens=self._max_tokens)
            if reply is None:
                return {"ok": False, "error": "cloud call failed", "tools_used": used}
            if reply.get("_truncated"):
                # Retrying a truncated tool call just truncates again.
                return {"ok": False, "tools_used": used,
                        "error": f"the reply was cut off at {self._max_tokens} tokens, "
                                 f"so the tool call was incomplete. Raise "
                                 f"JETTANK_AGENT_MAX_TOKENS."}

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
                if hasattr(self._tools, "dispatch_async"):
                    out = await self._tools.dispatch_async(name, args)
                else:
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
