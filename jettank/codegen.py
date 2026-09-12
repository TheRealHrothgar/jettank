"""Behaviours: Python that the cloud model writes, and Hank then runs.

The generation happens server-side, through the Anthropic Messages API - the
same CloudAgent transport the rest of the robot uses. The local VLM is not
involved; a 3B vision model has no business writing code that drives hardware.

Flow:
    voice/console -> agent tool `write_behavior`
                  -> Anthropic API generates a module
                  -> AST validation
                  -> written to behaviors/<name>.py
                  -> imported and callable as `run_behavior`

ON THE SECURITY BOUNDARY, stated plainly: `validate_source()` is a static AST
check. It stops accidents and lazy escapes. It is NOT a sandbox - a determined
code generator can get out of any in-process Python restriction, and pretending
otherwise would be worse than not having it. The controls that actually hold
are elsewhere and are physical:

  * the Yahboom driver is dry-run, so no frame reaches the motors at all;
  * MotionGuard starts disabled and is armed only from the console by a human;
  * generated code is handed a `robot` facade, not the guard, the driver or
    the loop - so the ordinary path to motion is the gated one.

Behaviours are written to disk and reviewable. Read them before arming motion.
"""
from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)

BEHAVIOR_DIR = Path(os.environ.get(
    "JETTANK_BEHAVIOR_DIR", Path(__file__).resolve().parent.parent / "behaviors"))

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")

# Modules generated code may import. Deliberately tiny and side-effect free.
ALLOWED_IMPORTS = {"math", "time", "random", "asyncio", "statistics", "json"}

# Names that are either escape hatches or a route to the safety interlock.
BANNED_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "memoryview", "exit", "quit",
}
BANNED_ATTRS = {
    "enable", "clear_estop", "estop", "_guard", "_loop", "_driver", "_robot",
    "arm_live", "_ser", "__class__", "__bases__", "__subclasses__",
    "__globals__", "__code__", "__dict__", "__mro__", "__builtins__",
}

CODEGEN_SYSTEM = """You write short Python behaviour modules for a small tracked robot called Hank.

Emit ONLY a Python module. No prose, no markdown fences, no explanation.

The module MUST define exactly this coroutine:

    async def run(robot, **params):
        ...
        return <findings>

YOUR CODE DOES NOT SPEAK. There is no say() and no printing to the user. Return
your findings as plain data - a dict or a list of dicts is ideal - and keep them
complete and structured. Something else turns them into spoken English, so do
NOT pre-format for speech, do NOT abbreviate, and do NOT truncate. Full
descriptions, real numbers, explicit keys. Machine-readable is exactly right.

Good:   {"stops": [{"pan": -60, "saw": "a lamp beside a window"}, ...],
         "obstacles": [], "recentred": true}
Bad:    "Room scan - -60 deg: lamp; ahead: TV"

`robot` is the ONLY way to affect the world. Every method is a coroutine - always await it:

    await robot.look(pan, tilt)                aim the camera head, degrees
    await robot.drive(linear, angular, reason) move; MAY BE REFUSED - see below
    await robot.stop()                         stop moving
    await robot.capture_image()                -> {"ok":bool, "image_b64":str}
    await robot.describe()                     -> str, what the local vision model sees now
    await robot.identify_face()                -> {"ok":bool, "names":[...]}
    await robot.status()                       -> dict incl. motion_enabled, estopped
    await robot.sleep(seconds)                 pause (use this, not time.sleep)
    robot.log(message)                         write to the robot's log

Rules, all of which matter:
* drive() returns {"ok": False, ...} when motion is disabled. That is NORMAL and
  expected - the operator arms motion physically. Check the result, tell the user
  plainly, and carry on. NEVER loop retrying it.
* You may import only: math, time, random, asyncio, statistics, json.
* No file access, no network, no subprocess, no eval/exec, no dunder attributes.
* Keep it under ~60 lines. Bound every loop - no `while True`.
* Prefer look/capture/describe over driving. Driving is a last resort.
* Handle failure: every robot call can return ok=False.
* `params` carries whatever the caller passes; give parameters sensible defaults.
* Return data, never prose meant for a listener. Someone else does the talking.

Write the module now."""


class ValidationError(Exception):
    pass


def validate_source(src: str) -> ast.Module:
    """Static checks on generated code. A speed bump, not a sandbox."""
    if len(src) > 20_000:
        raise ValidationError("module too long")
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        raise ValidationError(f"generated code does not parse: {exc}") from exc

    has_run = any(isinstance(n, ast.AsyncFunctionDef) and n.name == "run"
                  for n in tree.body)
    if not has_run:
        raise ValidationError("module must define `async def run(robot, **params)`")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise ValidationError(f"import of {a.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise ValidationError(f"import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise ValidationError(f"use of {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            raise ValidationError(f"access to {node.attr!r} is not allowed")
        elif isinstance(node, ast.While):
            # `while True` with an await inside is the classic runaway.
            if isinstance(node.test, ast.Constant) and node.test.value:
                raise ValidationError("unbounded `while True` is not allowed")
    return tree


class RobotFacade:
    """What generated code is allowed to touch.

    Every call routes through ToolBox.dispatch, so a behaviour gets exactly the
    gating, clamping and watchdog a direct tool call gets. The guard, driver and
    Loop are deliberately not reachable from here.
    """

    def __init__(self, dispatch, describe=None, log_fn=None) -> None:
        self._dispatch = dispatch
        self._describe = describe
        self._log = log_fn or (lambda m: log.info("[behaviour] %s", m))

    async def _call(self, tool: str, **args) -> dict:
        return await asyncio.to_thread(self._dispatch, tool, args)

    async def say(self, text: str) -> dict:
        return await self._call("speak", text=str(text))

    async def look(self, pan: float, tilt: float) -> dict:
        return await self._call("look", pan=float(pan), tilt=float(tilt))

    async def drive(self, linear: float, angular: float, reason: str = "") -> dict:
        return await self._call("drive", linear=float(linear),
                                angular=float(angular), reason=str(reason))

    async def stop(self) -> dict:
        return await self._call("stop")

    async def capture_image(self) -> dict:
        return await self._call("capture_image")

    async def identify_face(self) -> dict:
        return await self._call("identify_face")

    async def status(self) -> dict:
        return await self._call("get_status")

    async def describe(self) -> str:
        if self._describe is None:
            return ""
        return await self._describe()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, min(float(seconds), 10.0)))

    def log(self, message: str) -> None:
        self._log(str(message))


class BehaviorStore:
    def __init__(self, directory: Path = BEHAVIOR_DIR) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.py"

    def names(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.py")
                      if not p.stem.startswith("_"))

    def read(self, name: str) -> str | None:
        p = self.path(name)
        return p.read_text() if p.exists() else None

    def write(self, name: str, source: str, request: str) -> Path:
        header = (f'"""Generated behaviour: {name}\n\n'
                  f'Written by the cloud model from this request:\n'
                  f'    {request}\n\n'
                  f'Generated code - review before arming motion.\n"""\n')
        p = self.path(name)
        p.write_text(header + source.rstrip() + "\n")
        return p

    def forget(self, name: str) -> bool:
        p = self.path(name)
        if not p.exists():
            return False
        p.unlink()
        return True

    def load(self, name: str):
        """Import the module fresh so an edit takes effect without a restart."""
        p = self.path(name)
        if not p.exists():
            return None
        validate_source(p.read_text())          # re-check on every load
        spec = importlib.util.spec_from_file_location(f"jettank_behavior_{name}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def _strip_fences(text: str) -> str:
    """Models wrap code in ```python fences however firmly you ask them not to."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


class BehaviorWriter:
    """Generates behaviour modules through the Anthropic Messages API."""

    def __init__(self, agent, store: BehaviorStore | None = None,
                 max_attempts: int = 2, max_tokens: int = 8192) -> None:
        self._agent = agent
        self.store = store or BehaviorStore()
        self._max_attempts = max_attempts
        self._max_tokens = max_tokens

    @property
    def available(self) -> bool:
        return bool(self._agent and self._agent.enabled)

    async def write(self, name: str, request: str, tools_note: str = "") -> dict:
        if not self.available:
            return {"ok": False, "error": "no cloud provider configured, so nothing can generate code"}
        name = (name or "").strip().lower()
        if not NAME_RE.match(name):
            return {"ok": False, "error": "name must be lowercase letters, digits or _"}

        ask = request if not tools_note else f"{request}\n\nContext: {tools_note}"
        messages = [{"role": "user", "content": [{"type": "text", "text": ask}]}]
        last_error = ""

        for attempt in range(self._max_attempts):
            reply = await self._agent._messages(messages, system=CODEGEN_SYSTEM,
                                                max_tokens=self._max_tokens)
            if reply is None:
                return {"ok": False, "error": "the code-generation call failed"}
            text = " ".join(b.get("text", "") for b in reply.get("content", [])
                            if b.get("type") == "text")
            src = _strip_fences(text)
            try:
                validate_source(src)
            except ValidationError as exc:
                last_error = str(exc)
                log.warning("generated behaviour rejected (attempt %d): %s",
                            attempt + 1, last_error)
                # Hand the failure back so the model can correct itself.
                messages += [
                    {"role": "assistant", "content": [{"type": "text", "text": text}]},
                    {"role": "user", "content": [{"type": "text", "text":
                        f"That was rejected by the validator: {last_error}. "
                        f"Return a corrected module. Code only."}]},
                ]
                continue

            path = self.store.write(name, src, request)
            log.info("wrote behaviour %s (%d lines)", path, src.count("\n") + 1)
            return {"ok": True, "name": name, "path": str(path),
                    "lines": src.count("\n") + 1, "source": src,
                    "detail": f"wrote {name}; review it before arming motion"}

        return {"ok": False, "error": f"could not produce valid code: {last_error}"}


class BehaviorRunner:
    """Runs a generated behaviour inside the Docker sandbox.

    Generated code is never imported into this process. `validate_source` still
    runs first, but only as a cheap way to catch obvious junk before paying for
    a container - the actual boundary is the sandbox (see jettank/sandbox.py).
    If Docker is unavailable the behaviour is refused rather than run in-process:
    silently downgrading a security boundary is worse than not running at all.
    """

    def __init__(self, store: BehaviorStore, sandbox, timeout_s: float = 120.0,
                 narrator=None) -> None:
        self.store = store
        self._sandbox = sandbox
        self._timeout = timeout_s
        self._narrator = narrator

    async def run(self, name: str, params: dict | None = None) -> dict:
        src = self.store.read(name)
        if src is None:
            return {"ok": False, "error": f"no behaviour called {name!r}",
                    "known": self.store.names()}
        try:
            validate_source(src)
        except ValidationError as exc:
            return {"ok": False, "error": f"behaviour failed validation: {exc}"}
        if self._sandbox is None:
            return {"ok": False,
                    "error": "the sandbox is unavailable, so generated code will not "
                             "be run (docker missing, or the user is not in the "
                             "docker group)"}
        out = await self._sandbox.run(src, params or {})
        out["name"] = name
        # Behaviours are silent, so composing the spoken version happens here.
        if out.get("ok") and self._narrator is not None and out.get("result") is not None:
            spoken = await asyncio.to_thread(
                self._narrator.compose, out["result"], out.get("context", ""))
            if spoken:
                out["spoken"] = spoken
        return out
