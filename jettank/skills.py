"""Skills: named routines Hank can be taught by voice and run later.

A skill is a *composition of existing tools*, never arbitrary code. That is a
safety decision, not a limitation of effort. Hank's motion interlock lives in
MotionGuard, and `enable()` is deliberately not reachable as a tool - if a
skill could execute Python it could simply call it, and the whole interlock
that exists because of the tread runaway would be worth nothing. Composing
tools means every step inherits the same gating, clamping and watchdog that a
direct call gets, with no second code path to audit.

Within that, skills are still real programs: parameters, sequential steps,
conditional steps, and repetition.

Stored as JSON in ~/.jettank_skills.json so they survive a reboot.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_DB = Path(os.environ.get("JETTANK_SKILLS_DB",
                                 Path.home() / ".jettank_skills.json"))

MAX_STEPS = 24          # a skill is a routine, not a program that runs forever
MAX_REPEAT = 10
MAX_SKILLS = 100
NAME_RE = re.compile(r"^[a-z][a-z0-9 _-]{0,40}$")

# Tools a skill may not call, because they would let a taught routine rewrite
# the robot or recurse without bound.
FORBIDDEN = {"define_skill", "forget_skill", "set_config",
             "write_behavior", "run_behavior"}


@dataclass
class Step:
    """One tool call. `args` values may contain {param} placeholders."""

    tool: str
    args: dict = field(default_factory=dict)
    # Optional guard: run this step only if a previous step's result matched.
    # e.g. {"after": 0, "key": "ok", "equals": True}
    when: dict | None = None
    repeat: int = 1

    def to_json(self) -> dict:
        d = {"tool": self.tool, "args": self.args}
        if self.when:
            d["when"] = self.when
        if self.repeat != 1:
            d["repeat"] = self.repeat
        return d


@dataclass
class Skill:
    name: str
    description: str
    steps: list[Step]
    params: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"name": self.name, "description": self.description,
                "params": self.params, "steps": [s.to_json() for s in self.steps]}

    @staticmethod
    def from_json(d: dict) -> "Skill":
        return Skill(
            name=d["name"],
            description=d.get("description", ""),
            params=list(d.get("params", [])),
            steps=[Step(tool=s["tool"], args=s.get("args", {}),
                        when=s.get("when"), repeat=int(s.get("repeat", 1)))
                   for s in d.get("steps", [])],
        )


class SkillStore:
    def __init__(self, path: Path = DEFAULT_DB) -> None:
        self.path = Path(path)
        self._skills: dict[str, Skill] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            for d in raw.get("skills", []):
                s = Skill.from_json(d)
                self._skills[s.name] = s
            log.info("loaded %d skill(s) from %s", len(self._skills), self.path)
        except Exception as exc:  # noqa: BLE001 - a corrupt file must not stop boot
            log.warning("could not read skills from %s (%s)", self.path, exc)

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"skills": [s.to_json() for s in self._skills.values()]}, indent=2))
        tmp.replace(self.path)
        os.chmod(self.path, 0o600)

    def add(self, skill: Skill) -> None:
        if len(self._skills) >= MAX_SKILLS and skill.name not in self._skills:
            raise ValueError(f"too many skills (limit {MAX_SKILLS})")
        self._skills[skill.name] = skill
        self._save()

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name.strip().lower())

    def names(self) -> list[str]:
        return sorted(self._skills)

    def all(self) -> list[Skill]:
        return [self._skills[n] for n in self.names()]

    def forget(self, name: str) -> bool:
        if self._skills.pop(name.strip().lower(), None) is None:
            return False
        self._save()
        return True


def validate(name: str, description: str, steps: list[dict],
             params: list[str], known_tools: set[str]) -> Skill:
    """Turn a model-supplied definition into a Skill, or raise ValueError.

    Validation is strict and happens at *definition* time so a malformed skill
    fails when Hank is taught it - while the person is still talking to him and
    can correct it - rather than halfway through executing it later.
    """
    name = (name or "").strip().lower()
    if not NAME_RE.match(name):
        raise ValueError("name must be lowercase letters, digits, spaces, _ or -")
    if not steps:
        raise ValueError("a skill needs at least one step")
    if len(steps) > MAX_STEPS:
        raise ValueError(f"too many steps (limit {MAX_STEPS})")

    params = [p.strip() for p in (params or []) if p.strip()]
    parsed: list[Step] = []
    for i, raw in enumerate(steps):
        tool = str(raw.get("tool", "")).strip()
        if tool in FORBIDDEN:
            raise ValueError(f"step {i}: {tool!r} cannot be used inside a skill")
        if tool not in known_tools:
            raise ValueError(f"step {i}: unknown tool {tool!r}")
        repeat = int(raw.get("repeat", 1))
        if not 1 <= repeat <= MAX_REPEAT:
            raise ValueError(f"step {i}: repeat must be 1..{MAX_REPEAT}")

        args = raw.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError(f"step {i}: args must be an object")
        for v in args.values():
            for ph in re.findall(r"\{(\w+)\}", str(v)):
                if ph not in params:
                    raise ValueError(
                        f"step {i}: uses {{{ph}}} but that is not a parameter")

        when = raw.get("when")
        if when is not None:
            if not isinstance(when, dict) or "after" not in when:
                raise ValueError(f"step {i}: 'when' needs an 'after' step index")
            if not 0 <= int(when["after"]) < i:
                raise ValueError(f"step {i}: 'when.after' must be an earlier step")
        parsed.append(Step(tool=tool, args=args, when=when, repeat=repeat))

    return Skill(name=name, description=(description or "").strip(),
                 steps=parsed, params=params)


def _substitute(args: dict, values: dict) -> dict:
    """Fill {param} placeholders, keeping numbers as numbers where possible."""
    out = {}
    for k, v in args.items():
        if isinstance(v, str):
            m = re.fullmatch(r"\{(\w+)\}", v)
            if m:                       # whole value is one placeholder
                out[k] = values.get(m.group(1), v)
                continue
            v = re.sub(r"\{(\w+)\}", lambda g: str(values.get(g.group(1), g.group(0))), v)
        out[k] = v
    return out


def _passes(when: dict | None, results: list[dict]) -> bool:
    if not when:
        return True
    idx = int(when.get("after", -1))
    if not 0 <= idx < len(results):
        return False
    actual = results[idx].get(when.get("key", "ok"))
    if "equals" in when:
        return actual == when["equals"]
    if "contains" in when:
        return str(when["contains"]).lower() in str(actual).lower()
    return bool(actual)


class SkillRunner:
    """Executes a skill by dispatching each step through the normal toolbox."""

    def __init__(self, store: SkillStore, dispatch, budget_s: float = 90.0) -> None:
        self._store = store
        self._dispatch = dispatch
        self._budget_s = budget_s

    def run(self, name: str, values: dict | None = None) -> dict:
        skill = self._store.get(name)
        if skill is None:
            return {"ok": False, "error": f"no skill called {name!r}",
                    "known": self._store.names()}
        values = values or {}
        missing = [p for p in skill.params if p not in values]
        if missing:
            return {"ok": False, "error": f"missing parameter(s): {', '.join(missing)}"}

        deadline = time.monotonic() + self._budget_s
        results: list[dict] = []
        trace: list[dict] = []
        for i, step in enumerate(skill.steps):
            if not _passes(step.when, results):
                results.append({"ok": True, "skipped": True})
                trace.append({"step": i, "tool": step.tool, "skipped": "condition not met"})
                continue
            args = _substitute(step.args, values)
            for _ in range(step.repeat):
                if time.monotonic() > deadline:
                    trace.append({"step": i, "error": "skill exceeded its time budget"})
                    return {"ok": False, "error": "skill timed out", "ran": trace}
                out = self._dispatch(step.tool, args)
                results.append(out)
                # Images are huge; note that one arrived rather than echoing it.
                summary = {k: v for k, v in out.items() if k != "image_b64"}
                if "image_b64" in out:
                    summary["image"] = "captured"
                trace.append({"step": i, "tool": step.tool, "result": summary})
        return {"ok": True, "skill": skill.name, "ran": trace}
