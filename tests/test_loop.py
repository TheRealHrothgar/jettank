"""Exercises the real parsing, inference and escalation logic with a fake transport."""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tests.fake_httpx as fake  # noqa: E402

# Install the stub before jettank imports httpx.
stub = types.ModuleType("httpx")
stub.AsyncClient = fake.AsyncClient
stub.HTTPStatusError = fake.HTTPStatusError
sys.modules["httpx"] = stub

from jettank.cloud import CloudAgent  # noqa: E402
from jettank.vlm import LocalVLM  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail and not cond else ''}")


# ---------- JSON parsing ----------
print("\nCloudAgent._parse")
p = CloudAgent._parse
check("plain json", p('{"action":"explore","confidence":0.8}')["action"] == "explore")
check("fenced json", (p('```json\n{"action":"grasp"}\n```') or {}).get("action") == "grasp")
check(
    "prose-wrapped json",
    (p('Sure! Here you go:\n{"action":"retreat","say":"backing up"}\nHope that helps.') or {}).get("action")
    == "retreat",
)
check("garbage returns None", p("not json at all") is None)
check("empty returns None", p("") is None)


# ---------- local VLM ----------
print("\nLocalVLM")


def vlm_handler(method, url, payload):
    if method == "GET" and url.endswith("/api/tags"):
        return fake.Response(200, {"models": [{"name": "qwen2.5vl:3b"}]})
    if method == "POST" and url.endswith("/api/generate"):
        assert payload["images"] == ["BASE64"], "image not forwarded"
        assert payload["stream"] is False, "must not stream"
        return fake.Response(200, {"response": "  A red ball centre-left, clear path ahead.  "})
    return fake.Response(404, {})


fake.AsyncClient.handler = vlm_handler
vlm = LocalVLM("http://127.0.0.1:11434", "qwen2.5vl:3b", 5.0, 128)
check("model detected as available", asyncio.run(vlm.available()) is True)
desc = asyncio.run(vlm.describe("BASE64"))
check("describe returns trimmed text", desc == "A red ball centre-left, clear path ahead.", repr(desc))


def vlm_broken(method, url, payload):
    return fake.Response(500, {})


fake.AsyncClient.handler = vlm_broken
check("inference failure returns None, no raise", asyncio.run(vlm.describe("BASE64")) is None)


# ---------- cloud agent ----------
print("\nCloudAgent")
agent_none = CloudAgent("none", "", "m", None, 5.0, 256)
check("provider=none is disabled", agent_none.enabled is False)
check("disabled think() returns None", asyncio.run(agent_none.think(["x"])) is None)

check("missing api key disables", CloudAgent("anthropic", "", "m", None, 5.0, 256).enabled is False)


def anthropic_handler(method, url, payload):
    assert url.endswith("/v1/messages"), url
    assert payload["model"] == "claude-opus-5"
    assert "system" in payload
    return fake.Response(200, {"content": [{"type": "text", "text": '{"assessment":"ball ahead","action":"approach","target":"ball","say":"","confidence":0.7}'}]})


fake.AsyncClient.handler = anthropic_handler
a = CloudAgent("anthropic", "", "claude-opus-5", "key-123", 5.0, 256)
check("anthropic enabled with key", a.enabled is True)
plan = asyncio.run(a.think(["a red ball is ahead"]))
check("anthropic plan parsed", (plan or {}).get("action") == "approach", repr(plan))


def openai_handler(method, url, payload):
    assert url.endswith("/v1/chat/completions"), url
    assert payload["messages"][0]["role"] == "system"
    return fake.Response(200, {"choices": [{"message": {"content": '{"action":"idle","confidence":0.1}'}}]})


fake.AsyncClient.handler = openai_handler
o = CloudAgent("openai", "https://gateway.internal", "some-model", "key", 5.0, 256)
plan = asyncio.run(o.think(["nothing interesting"]))
check("openai-compatible plan parsed", (plan or {}).get("action") == "idle", repr(plan))


def cloud_broken(method, url, payload):
    raise RuntimeError("network down")


fake.AsyncClient.handler = cloud_broken
check("cloud outage returns None, no raise", asyncio.run(o.think(["x"])) is None)

# only the last 8 observations should be sent
sent = {}


def capture_handler(method, url, payload):
    sent["payload"] = payload
    return fake.Response(200, {"choices": [{"message": {"content": '{"action":"idle"}'}}]})


fake.AsyncClient.handler = capture_handler
asyncio.run(o.think([f"obs{i}" for i in range(20)]))
body = sent["payload"]["messages"][1]["content"]
check("only recent observations sent", body.count("- obs") == 8, f"count={body.count('- obs')}")
check("oldest observation dropped", "obs0\n" not in body)
check("newest observation kept", "obs19" in body)

# ---------- multimodal ----------
print("\nMultimodal (frames sent to the cloud)")
import base64 as _b64

PNG_B64 = _b64.b64encode(open("testdata/scene.png", "rb").read()).decode()
JPEG_B64 = _b64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 32).decode()

check("png media type sniffed", CloudAgent._media_type(PNG_B64) == "image/png")
check("jpeg media type sniffed", CloudAgent._media_type(JPEG_B64) == "image/jpeg")
check("garbage falls back to jpeg", CloudAgent._media_type("!!!!") == "image/jpeg")

cap = {}


def mm_anthropic(method, url, payload):
    cap["p"] = payload
    return fake.Response(200, {"content": [{"type": "text", "text": '{"action":"approach"}'}]})


fake.AsyncClient.handler = mm_anthropic
a2 = CloudAgent("anthropic", "", "claude-opus-5", "k", 5.0, 256)
asyncio.run(a2.think(["a red ball"], image_b64=PNG_B64))
blocks = cap["p"]["messages"][0]["content"]
check("anthropic sends a list of content blocks", isinstance(blocks, list))
check("image block first", blocks[0]["type"] == "image")
check("image is base64 source", blocks[0]["source"]["type"] == "base64")
check("correct media_type forwarded", blocks[0]["source"]["media_type"] == "image/png")
check("image data matches frame", blocks[0]["source"]["data"] == PNG_B64)
check("text block still present", blocks[-1]["type"] == "text" and "red ball" in blocks[-1]["text"])

# text-only: a single text block is the canonical Anthropic shape and is valid
asyncio.run(a2.think(["a red ball"], image_b64=None))
tb = cap["p"]["messages"][0]["content"]
check("anthropic text-only sends one text block",
      isinstance(tb, list) and len(tb) == 1 and tb[0]["type"] == "text")
check("anthropic text-only carries no image", not any(b.get("type") == "image" for b in tb))


def mm_openai(method, url, payload):
    cap["p"] = payload
    return fake.Response(200, {"choices": [{"message": {"content": '{"action":"idle"}'}}]})


fake.AsyncClient.handler = mm_openai
o2 = CloudAgent("openai", "https://gw", "m", "k", 5.0, 256)
asyncio.run(o2.think(["x"], image_b64=JPEG_B64))
uc = cap["p"]["messages"][1]["content"]
check("openai sends content list", isinstance(uc, list))
check("openai uses image_url data URI", uc[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
asyncio.run(o2.think(["x"], image_b64=None))
check("openai text-only stays a plain string", isinstance(cap["p"]["messages"][1]["content"], str))

# the toggle must actually gate it
import os as _os
from jettank import config as _cfg
_os.environ["JETTANK_CLOUD_SEND_FRAMES"] = "false"
check("send_frames honours 'false'", _cfg.load().cloud.send_frames is False)
_os.environ["JETTANK_CLOUD_SEND_FRAMES"] = "true"
check("send_frames honours 'true'", _cfg.load().cloud.send_frames is True)
del _os.environ["JETTANK_CLOUD_SEND_FRAMES"]
check("send_frames defaults on", _cfg.load().cloud.send_frames is True)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
