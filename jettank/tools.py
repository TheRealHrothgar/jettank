"""Tools the cloud model may call.

Design rules:
  * Every tool returns a plain dict; failures are reported, never raised, so a
    bad call degrades into a message the model can reason about.
  * Motion goes through MotionGuard and can be refused. The model is told why.
  * There is deliberately NO tool to enable motion or clear an E-STOP. Those
    are local-operator actions.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)


# Characters and constructs that a speech synthesiser reads out literally, or
# stumbles over. Text reaching TTS comes from three places - an LLM reply, a
# VLM description, and generated code - and none of them are writing for the
# ear. "Room scan - -60 deg: I see..." is not a sentence, it is a log line.
_MD = re.compile(r"[*_`#>~|]+")
_BRACKETS = re.compile(r"[\[\](){}<>]")
_URL = re.compile(r"https?://\S+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF]")
_MULTI_PUNCT = re.compile(r"([.,!?])\1+")
_SPACE = re.compile(r"\s+")
_NEG_NUM = re.compile(r"(?<![\w])-(\d)")
_DEGREES = re.compile(r"\b(-?\d+)\s*(?:deg|degs|degrees)\b", re.I)
_ABBREV = [
    (re.compile(r"\bpan\b", re.I), "pan"),
    (re.compile(r"\be\.g\.", re.I), "for example"),
    (re.compile(r"\bi\.e\.", re.I), "that is"),
    (re.compile(r"\betc\.?", re.I), "and so on"),
    (re.compile(r"\bvs\.?\b", re.I), "versus"),
    (re.compile(r"\bapprox\.?\b", re.I), "about"),
]


# Source code must never reach the synthesiser. Stripping its punctuation does
# not help - what is left is a stream of identifiers ("async def run robot
# params") which is exactly the gibberish this was chasing. Code is removed
# outright and replaced with a mention, because the only sane spoken form of a
# function body is a description of it.
_FENCED = re.compile(r"```[\w+-]*\n?.*?(?:```|$)", re.S)
_INDENTED_BLOCK = re.compile(r"(?:^[ \t]{4,}\S.*(?:\n|$)){2,}", re.M)
_CODE_LINE = re.compile(
    r"^\s*(?:from|import|def|async\s+def|class|return|await|for|while|if|elif|"
    r"else|try|except|finally|with|yield|raise|assert|lambda|@\w+)\b.*$|"
    r"^\s*[\w.\[\]\'\"]+\s*=\s*.+$|"      # assignment
    r"^\s*[\w.]+\(.*\)\s*:?\s*$|"            # bare call
    r"^\s*[)\]}>][,;:]?\s*$",                  # closing bracket line
    re.M)


def strip_code(text: str) -> tuple[str, bool]:
    """Remove code from text destined for speech. Returns (text, had_code)."""
    if not text:
        return "", False
    original = text
    t = _FENCED.sub(" ", text)
    t = _INDENTED_BLOCK.sub(" ", t)
    t = _CODE_LINE.sub(" ", t)
    had = len(t) < len(original) - 8
    # A line that is mostly punctuation and identifiers is code we missed.
    kept = []
    for line in t.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) > 8:
            symbols = sum(c in "(){}[]<>=+*/\\|&^~;:_" for c in stripped)
            if symbols / len(stripped) > 0.18:
                had = True
                continue
        kept.append(line)
    return " ".join(" ".join(kept).split()), had


def for_speech(text: str, max_chars: int = 600) -> str:
    """Turn machine text into something worth hearing.

    Not cosmetic: Piper reads stray punctuation aloud, so an unfiltered log
    line becomes audible gibberish. Symbols are removed or spoken properly,
    list separators become sentence breaks, and the result is trimmed at a
    sentence boundary rather than mid-word.
    """
    if not text:
        return ""
    t, had_code = strip_code(str(text))
    if had_code and not t.strip():
        return "I have written the code."
    if had_code:
        t = t.rstrip(" .,:;") + ". The code itself is saved, I will not read it out."
    t = _URL.sub("a link", t)
    # Arrows first: "=>" must not become "equals >" and then "equals equals".
    for arrow in ("=>", "->", "-->", "<-", "→"):
        t = t.replace(arrow, " then ")
    t = _EMOJI.sub(" ", t)
    t = _MD.sub(" ", t)
    t = _BRACKETS.sub(" ", t)
    t = _DEGREES.sub(r"\1 degrees", t)
    t = _NEG_NUM.sub(r"minus \1", t)         # "-60" -> "minus 60", not "dash 60"
    for pat, repl in _ABBREV:
        t = pat.sub(repl, t)
    # Separators that are punctuation on a page but silence in the ear.
    t = t.replace(";", ".").replace(" - ", ", ").replace(" -- ", ", ")
    t = t.replace(":", ",").replace(" / ", " or ")
    t = t.replace("=", " is ").replace("&", " and ").replace("%", " percent")
    t = re.sub(r"(\d)\s*/\s*(\d)", r"\1 of \2", t)   # "3/5" -> "3 of 5"
    t = t.replace("_", " ")
    t = _MULTI_PUNCT.sub(r"\1", t)
    t = _SPACE.sub(" ", t).strip(" ,.-")

    t = t.strip(" ,")
    if t and t[-1] not in ".!?":
        t += "."
    t = _pace(t)
    # Cap last, so pacing cannot push the result back over the limit.
    if len(t) > max_chars:
        cut = t[:max_chars]
        stop = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        t = cut[:stop + 1] if stop > max_chars // 3 else cut.rsplit(" ", 1)[0]
        if t and t[-1] not in ".!?":
            t += "."
    return t


# Only genuinely unpunctuated runs get broken up. An earlier version split at
# commas and conjunctions to "add pacing", which wrecked well-formed prose:
# "...a lamp on the right. casting light onto a window." - a fragment starting
# mid-clause on a lowercase word. Spoken, that is far worse than a long
# sentence. Piper already paces on commas and full stops; the job here is only
# to rescue text that has no punctuation for it to work with.
_RUNAWAY_WORDS = 34


def _pace(text: str) -> str:
    """Break up only sentences that give the synthesiser nothing to work with.

    A sentence with internal commas is left completely alone - Piper handles
    it. A long sentence with no internal punctuation at all is split, and the
    new sentence is capitalised so it does not sound like a fragment.
    """
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        words = sentence.split()
        if len(words) <= _RUNAWAY_WORDS or "," in sentence:
            out.append(sentence)
            continue
        # No commas and very long: insert breaks before a conjunction, and
        # capitalise so each piece reads as its own sentence.
        piece: list[str] = []
        for w in words:
            if len(piece) >= 12 and w.lower() in ("and", "but", "then", "so", "while"):
                out.append(_as_sentence(piece))
                piece = [w]
                continue
            piece.append(w)
        if piece:
            if len(piece) <= 3 and out:
                out[-1] = out[-1].rstrip(".") + " " + " ".join(piece)
            else:
                out.append(_as_sentence(piece))
    paced = " ".join(p for p in out if p.strip(" ."))
    if paced and paced[-1] not in ".!?":
        paced += "."
    return paced


def _as_sentence(words: list[str]) -> str:
    s = " ".join(words).strip(" ,")
    if not s:
        return ""
    s = s[0].upper() + s[1:]
    return s if s[-1] in ".!?" else s + "."


def _run_coro(coro):
    """Run a coroutine to completion from any thread.

    dispatch() is synchronous and gets called both from the event loop (an
    agent turn) and from worker threads (a skill, the sandbox bridge), so it
    cannot assume whether a loop is already running here. A fresh thread with
    its own loop is correct in both cases and costs nothing at this rate.
    """
    box: dict = {}

    def runner() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller
            box["error"] = exc

    t = threading.Thread(target=runner, name="tool-async", daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("value", {"ok": False, "error": "no result"})


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
    {
        "name": "define_skill",
        "description": (
            "Teach yourself a new named routine so you can do it again later on request. "
            "A skill is a sequence of your OWN tool calls - you cannot write code. Use this "
            "when someone says 'whenever I ask you to X, do Y' or 'learn this'. Read the "
            "steps back to them in plain words afterwards so they can confirm. "
            "Parameters let a skill be reused: declare them in 'params' and refer to them "
            "in step arguments as {name}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "short lowercase name, e.g. 'patrol'"},
                "description": {"type": "string", "description": "what it does, one line"},
                "params": {
                    "type": "array", "items": {"type": "string"},
                    "description": "names of values supplied when the skill is run",
                },
                "steps": {
                    "type": "array",
                    "description": "tool calls, in order",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string", "description": "name of one of your tools"},
                            "args": {"type": "object", "description": "arguments; may use {param}"},
                            "repeat": {"type": "integer", "description": "run this step N times (1-10)"},
                            "when": {
                                "type": "object",
                                "description": (
                                    "run only if an earlier step matched, e.g. "
                                    '{"after": 0, "key": "ok", "equals": true}'
                                ),
                            },
                        },
                        "required": ["tool"],
                    },
                },
            },
            "required": ["name", "steps"],
        },
    },
    {
        "name": "run_skill",
        "description": "Run a routine you were taught earlier. Check list_skills if unsure of the name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "values": {"type": "object", "description": "values for the skill's parameters"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "list_skills",
        "description": "List the routines you have been taught, with what each one does.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "forget_skill",
        "description": "Delete a routine you were taught.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "move_arm",
        "description": (
            "Move your arm to a named position: stow (folded back, jaws closed), down, "
            "level, up, or raised (fully up toward your camera, jaws open). "
            "Your arm has ONE joint and your JAWS ARE LINKED TO IT - they close as the "
            "arm folds down and open as it folds up. Grip and arm angle are the same "
            "control, so you cannot grab something and then lift it: lifting is what "
            "opens the jaws. Say so if asked to pick something up."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "position": {
                    "type": "string",
                    "description": "stow, down, level, up, or raised",
                },
            },
            "required": ["position"],
        },
    },
    {
        "name": "set_motion",
        "description": (
            "Turn your own motors OFF, or ASK to have them turned on. "
            "enabled=false disarms immediately and always works - stopping is always "
            "allowed. enabled=true does NOT arm you: you cannot arm your own motors. "
            "It records that you want to move, and you must then tell the person out "
            "loud that they need to say 'Hank arm motion', and why you want to move. "
            "Do not pretend you have been armed, and do not ask repeatedly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "reason": {"type": "string", "description": "why you want to move"},
            },
            "required": ["enabled"],
        },
    },
    {
        "name": "write_behavior",
        "description": (
            "Write NEW PYTHON CODE for yourself, to do something your existing tools cannot "
            "already do, and save it under a name. Use this when someone says 'program "
            "yourself to...', 'write code to...', or asks for a capability you lack. "
            "The code is generated in the cloud and runs in a locked-down container. "
            "Put the FULL request in 'request' - the code generator cannot see this "
            "conversation. Afterwards, describe in plain words what it does."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "lowercase identifier, e.g. 'patrol'"},
                "request": {
                    "type": "string",
                    "description": "complete description of the behaviour, in plain English",
                },
            },
            "required": ["name", "request"],
        },
    },
    {
        "name": "run_behavior",
        "description": (
            "Run a coded behaviour you wrote earlier, in the sandbox. "
            "Check list_behaviors if unsure of the name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "params": {"type": "object", "description": "values passed to the behaviour"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "list_behaviors",
        "description": "List the coded behaviours you have written.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_behavior",
        "description": "Read back a behaviour's source, so you can explain or check it.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
]

SETTABLE = {"vlm_interval", "cam_width", "cam_height", "cloud_min_interval"}


class ToolBox:
    """Dispatches model tool calls to the robot."""

    # Handlers that are coroutines. dispatch() is called from worker threads
    # as well as the event loop, so these get their own loop rather than
    # assuming one is running on the calling thread.
    ASYNC_TOOLS = {"write_behavior", "run_behavior"}

    def __init__(self, loop, guard, camera, faces, speaker, skills=None,
                 writer=None, behaviors=None) -> None:
        self._loop = loop
        self._guard = guard
        self._camera = camera
        self._faces = faces
        self._speaker = speaker
        self._skills = skills
        self._writer = writer
        self._behaviors = behaviors
        self._runner = None
        if skills is not None:
            from .skills import SkillRunner

            # The runner dispatches back through this same ToolBox, so every
            # step inside a skill gets the identical gating a direct call does.
            self._runner = SkillRunner(skills, self.dispatch)
        self._depth = 0

    async def dispatch_async(self, name: str, args: dict) -> dict:
        """Dispatch from the event loop, awaiting coroutine handlers natively.

        The async tools reach the cloud over the shared httpx client, which is
        bound to the loop that created it. Running them on a private loop
        raises "bound to a different event loop", so callers that *have* a loop
        must use this rather than dispatch().
        """
        if name not in self.ASYNC_TOOLS:
            return self.dispatch(name, args)
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return {"ok": False, "error": f"unknown tool {name!r}"}
        try:
            return await fn(**(args or {}))
        except TypeError as exc:
            return {"ok": False, "error": f"bad arguments for {name}: {exc}"}
        except Exception as exc:  # noqa: BLE001
            log.exception("tool %s failed", name)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def dispatch(self, name: str, args: dict) -> dict:
        fn: Callable[..., dict] | None = getattr(self, f"_t_{name}", None)
        if fn is None:
            return {"ok": False, "error": f"unknown tool {name!r}"}
        try:
            if name in self.ASYNC_TOOLS:
                # Reached from a worker thread (a skill, the sandbox bridge).
                # These tools need the loop's httpx client, so refuse clearly
                # instead of failing deep inside with a cross-loop error.
                return {"ok": False,
                        "error": f"{name} can only be called directly by the agent, "
                                 f"not from inside a skill or a behaviour"}
            return fn(**(args or {}))
        except TypeError as exc:
            return {"ok": False, "error": f"bad arguments for {name}: {exc}"}
        except Exception as exc:  # noqa: BLE001 - a tool fault must not kill the loop
            log.exception("tool %s failed", name)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # ---- implementations ----
    def _t_get_status(self) -> dict:
        st = {"ok": True, **self._guard.status()}
        board = getattr(self._loop, "board", None)
        if board is not None:
            st.update(board.status())
        batt = getattr(self._loop, "battery", None)
        if batt is not None and batt.voltage is not None:
            st["battery_state"] = batt.state
            st["battery_v"] = batt.voltage
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

    def _t_move_arm(self, position: str) -> dict:
        drv = self._loop.robot
        if not hasattr(drv, "arm_positions"):
            return {"ok": False, "error": "this robot has no controllable arm"}
        known = drv.arm_positions()
        if str(position).lower() not in known:
            return {"ok": False, "error": f"unknown position {position!r}",
                    "known_positions": known}
        drv.arm(str(position).lower())
        return {"ok": True, "position": drv.arm_position(),
                "jaws": drv.grip_state() if hasattr(drv, "grip_state") else "unknown",
                "note": "jaws are linked to the arm angle, not independent"}

    def _t_set_motion(self, enabled: bool, reason: str = "") -> dict:
        # Disabling is unconditional; enabling is a request a human must grant.
        if not enabled:
            self._loop._set_motion(False, "hank")
            return {"ok": True, "motion_enabled": False, "detail": "motors disarmed"}
        return self._loop.request_arm(reason)

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

    # ---- skills ----
    def _t_define_skill(self, name: str, steps: list, description: str = "",
                        params: list | None = None) -> dict:
        if self._skills is None:
            return {"ok": False, "error": "skills are not available on this robot"}
        from .skills import validate

        known = {t["name"] for t in TOOL_SCHEMAS}
        try:
            skill = validate(name, description, steps, params or [], known)
            self._skills.add(skill)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "name": skill.name, "steps": len(skill.steps),
                "params": skill.params,
                "detail": f"learned {skill.name!r}; say the name to run it"}

    def _t_run_skill(self, name: str, values: dict | None = None) -> dict:
        if self._runner is None:
            return {"ok": False, "error": "skills are not available on this robot"}
        # A skill may not call itself, directly or through another skill.
        if self._depth >= 2:
            return {"ok": False, "error": "skills cannot nest this deeply"}
        self._depth += 1
        try:
            return self._runner.run(name, values or {})
        finally:
            self._depth -= 1

    def _t_list_skills(self) -> dict:
        if self._skills is None:
            return {"ok": False, "error": "skills are not available on this robot"}
        return {"ok": True, "skills": [
            {"name": s.name, "description": s.description, "params": s.params,
             "steps": len(s.steps)} for s in self._skills.all()]}

    def _t_forget_skill(self, name: str) -> dict:
        if self._skills is None:
            return {"ok": False, "error": "skills are not available on this robot"}
        return ({"ok": True, "detail": f"forgot {name!r}"} if self._skills.forget(name)
                else {"ok": False, "error": f"no skill called {name!r}"})

    # ---- generated behaviours ----
    async def _t_write_behavior(self, name: str, request: str) -> dict:
        if self._writer is None or not self._writer.available:
            return {"ok": False, "error": "code generation needs a cloud provider configured"}
        out = await self._writer.write(name, request)
        # Deliberately NOT returning the source. Anything in a tool result
        # tends to come back in the spoken reply, and source read aloud is
        # gibberish. read_behavior exists for when it is actually wanted.
        out.pop("source", None)
        return out

    async def _t_run_behavior(self, name: str, params: dict | None = None) -> dict:
        if self._behaviors is None:
            return {"ok": False, "error": "behaviours are not available on this robot"}
        return await self._behaviors.run(name, params or {})

    def _t_list_behaviors(self) -> dict:
        if self._behaviors is None:
            return {"ok": False, "error": "behaviours are not available on this robot"}
        return {"ok": True, "behaviors": self._behaviors.store.names()}

    def _t_read_behavior(self, name: str) -> dict:
        if self._behaviors is None:
            return {"ok": False, "error": "behaviours are not available on this robot"}
        src = self._behaviors.store.read(name)
        return ({"ok": True, "name": name, "source": src,
                 "note": "This is source code. Do not read it aloud - describe "
                         "what it does in plain words instead."}
                if src is not None
                else {"ok": False, "error": f"no behaviour called {name!r}"})

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

    def __init__(self, device: str = "auto", voice: str | None = None,
                 narrator=None) -> None:
        from .audio import pick_speaker

        # Machine text is rephrased before it is spoken; for_speech() is the
        # mechanical fallback when the narrator is off or unreachable.
        self._narrator = narrator
        self._last_spoke = 0.0
        # The aplay process currently playing, so speech can be cut off
        # mid-sentence. Without this, "stop" would have to wait politely for
        # Hank to finish saying whatever prompted it.
        self._playing: subprocess.Popen | None = None
        self._abort = threading.Event()
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

    # Piper's own defaults, deliberately. An earlier version slowed the rate
    # and injected silence between sentence chunks to "improve pacing"; a
    # side-by-side listening test on the robot showed plain default Piper was
    # clearly the most intelligible, and both adjustments made it worse. The
    # model's own prosody already handles commas and full stops. Left as knobs
    # for tuning, but do not change the defaults without listening first.
    RATE = float(os.environ.get("JETTANK_TTS_RATE", "1.0"))          # >1 slower
    SENTENCE_GAP_MS = int(os.environ.get("JETTANK_TTS_GAP_MS", "0"))

    def _syn_config(self):
        """Piper's own gain stage.

        The USB speakerphone tops out at -2.5 dB even with the ALSA PCM control
        at 100%, so if that is still not loud enough the remaining headroom has
        to come from synthesis. `volume` is a linear multiplier; above ~1.5 it
        starts to clip, hence the cap. `normalize_audio` evens out quiet
        phrases, which matters more than peak level for intelligibility.
        """
        gain = float(os.environ.get("JETTANK_TTS_GAIN", "1.0"))
        try:
            from piper import SynthesisConfig

            return SynthesisConfig(volume=max(0.1, min(gain, 2.0)),
                                   length_scale=max(0.8, min(self.RATE, 1.6)),
                                   normalize_audio=True)
        except Exception as exc:  # noqa: BLE001 - older piper lacks it
            log.debug("SynthesisConfig unavailable (%s)", exc)
            return None

    def _lead_in_ms(self) -> int:
        """Full lead-in only when the amplifier has had time to sleep."""
        warm = (time.monotonic() - self._last_spoke) < self.WARM_WINDOW_S
        return self.LEAD_IN_WARM_MS if warm else self.LEAD_IN_MS

    @staticmethod
    def resample(pcm: bytes, src_rate: int, dst_rate: int, channels: int = 1) -> bytes:
        """Convert sample rate offline, so ALSA never has to do it live."""
        if src_rate == dst_rate or not pcm:
            return pcm
        try:
            import audioop

            converted, _ = audioop.ratecv(pcm, 2, channels, src_rate, dst_rate, None)
            return converted
        except Exception as exc:  # noqa: BLE001 - fall back to letting ALSA cope
            log.debug("offline resample unavailable (%s)", exc)
            return pcm

    @staticmethod
    def _silence(rate: int, ms: int) -> bytes:
        return b"\x00" * (2 * int(rate * ms / 1000))

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

    # Playback buffering. Raw PCM into `aplay -t raw -` garbles speech at
    # aplay's defaults: it sizes its ALSA buffer from the stream parameters and
    # begins consuming immediately, so the first burst is playing before much
    # has been synthesised, and it underruns. An underrun on this USB device
    # does not sound like a click - it sounds like slurred, mispaced speech,
    # which is why this read as a text problem for so long.
    #
    # The fix is not to buffer the whole utterance (that costs latency and
    # memory for nothing); it is to give ALSA a real buffer and not start
    # playing until there is enough audio to stay ahead of it. After that,
    # writes to a blocking pipe self-pace: aplay consumes at exactly realtime,
    # so the write blocks whenever we get ahead. No sleeping, no rate maths.
    PREBUFFER_MS = int(os.environ.get("JETTANK_TTS_PREBUFFER_MS", "400"))
    # The USB speakerphone powers its amplifier down when idle and takes a
    # moment to come back, so the opening syllable is lost - clipped in the
    # hardware, not in the audio, which is why the PCM measures correct and
    # still sounds wrong. Silence in front gives the DAC time to settle.
    #
    # Two values, because the cost only applies when it is actually cold. In a
    # back-and-forth conversation the amplifier is still up from the previous
    # reply, and paying half a second before every turn would make Hank feel
    # sluggish for no benefit.
    # The speakerphone supports exactly one rate - 48000 Hz, both directions
    # (see /proc/asound/card0/stream0). Anything else makes ALSA's plug layer
    # resample in realtime, and with the mic also open it is doing two
    # conversions at once on a full-speed USB device. That is what the skipping
    # and the garbled long utterances were: not the text, not the buffering,
    # and not aplay - two realtime resamplers fighting over one clock.
    #
    # So we resample once, offline, and hand the device its native rate.
    DEVICE_RATE = int(os.environ.get("JETTANK_AUDIO_RATE", "48000"))
    LEAD_IN_MS = int(os.environ.get("JETTANK_TTS_LEAD_IN_MS", "600"))
    LEAD_IN_WARM_MS = int(os.environ.get("JETTANK_TTS_LEAD_IN_WARM_MS", "120"))
    WARM_WINDOW_S = float(os.environ.get("JETTANK_TTS_WARM_WINDOW", "8.0"))
    ALSA_BUFFER_US = int(os.environ.get("JETTANK_TTS_BUFFER_US", "1000000"))
    ALSA_PERIOD_US = int(os.environ.get("JETTANK_TTS_PERIOD_US", "100000"))
    WRITE_BLOCK_MS = 100

    # Set JETTANK_TTS_DUMP_DIR to capture exactly what is sent to aplay. Every
    # reconstruction of a speech fault has sounded clean; the only way to stop
    # guessing is to keep the real bytes from the run that actually failed.
    DUMP_DIR = os.environ.get("JETTANK_TTS_DUMP_DIR", "")

    def _dump(self, pcm: bytes, rate: int, channels: int, text: str) -> None:
        if not self.DUMP_DIR or not pcm:
            return
        try:
            d = Path(self.DUMP_DIR)
            d.mkdir(parents=True, exist_ok=True)
            stem = d / f"say-{int(time.time())}-{abs(hash(text)) % 10000:04d}"
            with wave.open(str(stem) + ".wav", "wb") as w:
                w.setnchannels(channels)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(pcm)
            (stem.with_suffix(".txt")).write_text(text)
            log.info("dumped speech to %s.wav (%.1fs)", stem, len(pcm) / 2 / channels / rate)
        except Exception as exc:  # noqa: BLE001 - diagnostics must never break speech
            log.debug("speech dump failed: %s", exc)

    def _speak_piper(self, text: str) -> dict:
        """Synthesise to a temp file, then hand the file to aplay.

        Streaming raw PCM into aplay was rewritten three times and kept
        failing on the robot in ways that never reproduced on the bench:
        garbled long utterances, then skipping. The reason is that streaming
        makes playback depend on *our* process being scheduled promptly for
        the entire duration of the audio - and on this board, speech competes
        with VLM inference on the GPU, whisper on the CPU, and the perception
        loop. Miss a deadline and the sound card runs dry.

        Handing aplay a file removes that dependency entirely: it reads from
        disk at its own pace and cannot be starved by anything we do. Every
        listening test of file playback has been clean, including under load
        and with the mic held open, which is the evidence this is built on.

        The cost is synthesis latency before the first sound - roughly 0.15x
        the audio length, so about 0.7s for a five-second reply. That is a
        real cost and it buys reliability, which for speech is worth more.
        """
        voice = self._load_piper()
        syn = self._syn_config()
        chunks = list(voice.synthesize(text, syn) if syn else voice.synthesize(text))
        if not chunks:
            return {"ok": False, "error": "TTS produced no audio", "spoken_text": text}
        if self._abort.is_set():
            return {"ok": False, "spoken_text": text, "aborted": True,
                    "error": "speech interrupted"}

        rate = getattr(chunks[0], "sample_rate", 22050)
        channels = getattr(chunks[0], "sample_channels", 1)
        gap = (self._silence(rate, self.SENTENCE_GAP_MS)
               if self.SENTENCE_GAP_MS > 0 else b"")
        # Lead-in silence still matters: the speakerphone's amplifier sleeps
        # and clips the opening syllable without it.
        pcm = gap.join(self._pcm_of(c) for c in chunks)
        # Resample to the device's native rate before adding the lead-in, so
        # the silence is generated at the final rate and stays exact.
        if self.DEVICE_RATE and self.DEVICE_RATE != rate:
            pcm = self.resample(pcm, rate, self.DEVICE_RATE, channels)
            rate = self.DEVICE_RATE
        pcm = self._silence(rate, self._lead_in_ms()) + pcm

        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False,
                                             prefix="hank-say-") as fh:
                path = fh.name
            with wave.open(path, "wb") as w:
                w.setnchannels(channels)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(pcm)
            self._dump(pcm, rate, channels, text)

            proc = subprocess.Popen(["aplay", "-q", "-D", self._device, path],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE)
            self._playing = proc
            duration = len(pcm) / 2 / channels / rate
            try:
                proc.wait(timeout=duration + 20)
            except subprocess.TimeoutExpired:
                proc.kill()
                return {"ok": False, "error": "playback timed out",
                        "spoken_text": text}
        except FileNotFoundError:
            return {"ok": False, "error": "aplay not installed", "spoken_text": text}
        finally:
            self._playing = None
            if path:
                with contextlib.suppress(OSError):
                    Path(path).unlink()

        if self._abort.is_set():
            return {"ok": False, "spoken_text": text, "aborted": True,
                    "error": "speech interrupted"}
        self._last_spoke = time.monotonic()
        return {"ok": True, "spoken_text": text,
                "duration_s": round(len(pcm) / 2 / channels / rate, 2)}

    @staticmethod
    def _pcm_of(chunk) -> bytes:
        pcm = getattr(chunk, "audio_int16_bytes", None)
        if pcm is None:
            pcm = bytes(chunk.audio_int16_array)
        return pcm

    def _render(self, text: str) -> bytes:
        """WAV bytes from the fallback engine."""
        return subprocess.run([self._engine, "--stdout", text],
                              capture_output=True, timeout=30).stdout

    def abort(self) -> bool:
        """Cut off speech immediately. Safe to call from any thread."""
        self._abort.set()
        proc = self._playing
        if proc is None or proc.poll() is not None:
            return False
        with contextlib.suppress(Exception):
            proc.kill()
        return True

    def say(self, text: str) -> dict:
        self._abort.clear()
        if self._narrator is not None:
            try:
                text = self._narrator.narrate(str(text))
            except Exception as exc:  # noqa: BLE001 - never block speech
                log.warning("narration failed (%s)", exc)
        spoken = for_speech(text)
        if not spoken:
            return {"ok": False, "error": "nothing speakable in that text",
                    "spoken_text": ""}
        text = spoken
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
