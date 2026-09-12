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


# ---------- safety guard ----------
print("\nMotionGuard")
from jettank.safety import MotionGuard, MotionLimits  # noqa: E402


class RecordingRobot:
    def __init__(self):
        self.calls = []

    def drive(self, left, right):
        self.calls.append((round(left, 3), round(right, 3)))

    def stop(self):
        self.calls.append("stop")

    def look(self, pan, tilt):
        self.calls.append(("look", pan, tilt))

    def say(self, text):
        self.calls.append(("say", text))


rb = RecordingRobot()
g = MotionGuard(rb, MotionLimits(timeout_s=99, max_run_s=99), enabled=False)
ok, why = g.drive(1.0, 0.0)
check("motion refused when disabled", ok is False)
check("refusal says why", "disabl" in why.lower() or "not enabled" in why.lower(), why)
check("nothing reached the robot", rb.calls == [], str(rb.calls))

g.enable(True)
ok, _ = g.drive(1.0, 0.0)
check("motion accepted once armed", ok is True)
check("command reached the robot", any(c != "stop" for c in rb.calls))
check("speed clamped to limits", all(
    c == "stop" or max(abs(c[0]), abs(c[1])) <= 0.31 for c in rb.calls), str(rb.calls))

g.estop("test")
ok, why = g.drive(1.0, 0.0)
check("estop refuses motion", ok is False)
check("estop is reported", "stop" in why.lower(), why)
g.clear_estop()
check("clear_estop releases the latch", g.drive(0.1, 0.0)[0] is True)
g.close()

# no tool may arm motion or clear an estop - that is a local-operator act
from jettank.tools import TOOL_SCHEMAS, ToolBox  # noqa: E402
names = {t["name"] for t in TOOL_SCHEMAS}
check("no enable-motion tool exposed", not (names & {"enable_motion", "arm", "clear_estop", "set_enabled"}),
      str(sorted(names)))
from jettank.tools import SETTABLE  # noqa: E402
check("settable keys are tuning only, never safety",
      SETTABLE == {"vlm_interval", "cam_width", "cam_height", "cloud_min_interval"}, str(SETTABLE))

# ---------- tool dispatch ----------
print("\nToolBox dispatch")


class FakeCam:
    _device = "/dev/video0"

    def __init__(self, b64=JPEG_B64):
        self._b64 = b64

    def latest_jpeg_b64(self):
        return (7, self._b64)


class FakeLoop:
    def __init__(self, robot):
        self.robot = robot
        self.observations = ["a wall", "a chair"]
        self.applied = []

    def apply_setting(self, key, value):
        self.applied.append((key, value))
        return {"ok": True, "key": key, "value": value}


class FakeSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)
        return {"ok": True, "spoken_text": text}


rb2 = RecordingRobot()
fl = FakeLoop(rb2)
guard2 = MotionGuard(rb2, MotionLimits(timeout_s=99, max_run_s=99), enabled=False)
spk = FakeSpeaker()
tb = ToolBox(fl, guard2, FakeCam(), None, spk)

check("unknown tool is reported, not raised", tb.dispatch("nope", {})["ok"] is False)
check("bad args are reported, not raised", tb.dispatch("drive", {"linear": "x"})["ok"] is False)
check("get_status works", tb.dispatch("get_status", {})["ok"] is True)
check("status carries observations",
      tb.dispatch("get_status", {})["recent_observations"] == ["a wall", "a chair"])
check("capture_image returns a frame", tb.dispatch("capture_image", {})["image_b64"] == JPEG_B64)
check("capture_image handles a dead camera",
      ToolBox(fl, guard2, FakeCam(None), None, spk).dispatch("capture_image", {})["ok"] is False)
check("drive is refused while disarmed", tb.dispatch("drive", {"linear": 1, "angular": 0})["ok"] is False)
check("drive refusal is not an exception", "detail" in tb.dispatch("drive", {"linear": 1, "angular": 0}))
check("stop always works", tb.dispatch("stop", {})["ok"] is True)
check("speak reaches the speaker", tb.dispatch("speak", {"text": "hello"})["ok"] is True and spk.said == ["hello"])
check("speak rejects empty text", tb.dispatch("speak", {"text": "   "})["ok"] is False)
check("look drives the servos", tb.dispatch("look", {"pan": 10, "tilt": -5})["ok"] is True)
check("look reached the robot", ("look", 10.0, -5.0) in rb2.calls, str(rb2.calls))
check("faces degrade gracefully when absent", tb.dispatch("list_faces", {})["ok"] is False)
check("set_config rejects unknown keys", tb.dispatch("set_config", {"key": "root_pw", "value": "x"})["ok"] is False)
check("set_config accepts allowed keys", tb.dispatch("set_config", {"key": "vlm_interval", "value": "3"})["ok"] is True)
check("set_config reached the loop", fl.applied == [("vlm_interval", "3")])
guard2.close()

# ---------- the autonomous path is gated too ----------
print("\nAutonomous plan gating")
from jettank.loop import Loop  # noqa: E402


class PlanLoop:
    """Just enough Loop to exercise _apply without touching hardware."""
    _MOVES = Loop._MOVES
    _apply = Loop._apply

    def __init__(self, guard, robot):
        self.guard, self.robot, self.spoken = guard, robot, []

    def say(self, text):
        self.spoken.append(text)


rb4 = RecordingRobot()
guard4 = MotionGuard(rb4, MotionLimits(timeout_s=99, max_run_s=99), enabled=False)
pl = PlanLoop(guard4, rb4)
pl._apply({"action": "explore", "say": "off I go"})
check("cloud plan cannot move a disarmed robot",
      all(c == "stop" or c[0] == "look" for c in rb4.calls), str(rb4.calls))
check("plan speech still happens", pl.spoken == ["off I go"])
guard4.enable(True)
rb4.calls.clear()
pl._apply({"action": "retreat"})
check("cloud plan moves once armed", any(c != "stop" for c in rb4.calls), str(rb4.calls))
check("plan speed is clamped", all(
    c == "stop" or max(abs(c[0]), abs(c[1])) <= 0.31 for c in rb4.calls), str(rb4.calls))
rb4.calls.clear()
pl._apply({"action": "idle"})
check("idle stops", "stop" in rb4.calls)
guard4.close()

# ---------- voice ----------
print("\nVoice command and control")
from jettank.audio import _rms, match_wake_word  # noqa: E402

check("wake word strips punctuation", match_wake_word("Hey, Tank! drive forward", "hey tank") == "drive forward")
check("wake word is case-insensitive", match_wake_word("HEY TANK stop", "hey tank") == "stop")
check("wake word mid-sentence still matches", match_wake_word("um hey tank look left", "hey tank") == "look left")
check("no wake word means no command", match_wake_word("just chatting", "hey tank") is None)
check("bare wake word yields empty command", match_wake_word("hey tank", "hey tank") == "")
check("empty wake word is always-on", match_wake_word("  do the thing ", "") == "do the thing")
check("silence reads as near-zero rms", _rms(b"\x00\x00" * 100) < 0.001)
check("loud audio reads high", _rms(b"\x00\x40" * 100) > 0.4)
check("empty buffer is safe", _rms(b"") == 0.0)

# ---------- console ----------
print("\nConsole")
from jettank.console import Console  # noqa: E402


class ConsoleLoop:
    def __init__(self, guard):
        self.guard = guard
        self.camera = FakeCam()
        self.static_image_b64 = None
        self.frames_seen = self.vlm_calls = self.cloud_calls = 0


rb3 = RecordingRobot()
guard3 = MotionGuard(rb3, MotionLimits(timeout_s=99, max_run_s=99), enabled=False)
con = Console(ConsoleLoop(guard3))
check("console reports disarmed state", con._snapshot()["motion"] == "disabled")
con.control("arm")
check("console can arm motion", con._snapshot()["motion"] == "enabled")
check("armed guard now accepts drive", guard3.drive(0.1, 0.0)[0] is True)
con.control("estop")
check("console estop latches", con._snapshot()["estop"] == "LATCHED")
check("estop blocks drive", guard3.drive(0.1, 0.0)[0] is False)
con.control("disarm")
check("console can disarm", con._snapshot()["motion"] == "disabled")
check("unknown console action is reported", "unknown" in con.control("nuke"))
check("console serves the frame as bytes", con.frame_jpeg()[:3] == b"\xff\xd8\xff")
check("console event feed is capped", (
    [con.event("see", str(i)) for i in range(250)], len(con._snapshot()["events"]) == 200)[1])
guard3.close()

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
