"""Apply changes without restarting Hank.

Restarting costs about twenty seconds - Piper's voice loads, whisper loads, the
mic recalibrates, Ollama reloads the vision model - and it drops whatever
conversation was in progress. When you are iterating on a prompt or a speed
limit, that is the whole feedback loop.

WHAT RELOADS, AND WHY THE LINE IS WHERE IT IS

  reloadable        prompts, runtime settings, the motion/servo map,
                    behaviours, skills
  NOT reloadable    anything holding a device: the mic's arecord pipe, the
                    serial link to the board, the loaded Piper voice, the
                    camera

The rule is ownership of hardware, not code purity. Re-importing a module whose
instance holds an open file descriptor does not give you new behaviour - it
gives you two objects that both think they own the device, and a class identity
mismatch that surfaces later as an inexplicable isinstance failure. So this
reloads *data and text*, plus a small allow-list of modules that hold nothing.

Anything not on that list needs `systemctl restart hank`, and reload() says so
rather than pretending it worked.
"""
from __future__ import annotations

import importlib
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Modules safe to re-import: pure functions and constants, no open handles, no
# long-lived instances anyone else holds a reference to.
RELOADABLE_MODULES = (
    "jettank.control",      # wake/stop/kill/arm phrases
    "jettank.sysinfo",      # how the machine is described
    "jettank.narrator",     # narration prompts
    "jettank.skills",       # skill validation rules
)

# Deliberately absent, and why:
#   jettank.audio    VoiceListener owns the arecord pipe
#   jettank.tools    Speaker holds the loaded Piper voice
#   jettank.drive    BoardLink owns the serial port
#   jettank.board    BoardReader owns a reader thread
#   jettank.loop     reloading the running object's own class is not meaningful
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


class Reloader:
    """Re-applies editable state onto a running Loop."""

    def __init__(self, loop) -> None:
        self._loop = loop
        self._seen: dict[str, float] = {}

    # ---- prompts -------------------------------------------------------
    def _prompt_files(self) -> dict[str, Path]:
        if not PROMPT_DIR.is_dir():
            return {}
        return {p.stem: p for p in PROMPT_DIR.glob("*.md")}

    def load_prompts(self) -> list[str]:
        """Override the built-in prompts from prompts/*.md if present.

        Files win over the constants in cloud.py, so a prompt can be edited and
        applied while Hank is mid-conversation. An absent file just means the
        built-in stays.
        """
        from . import cloud

        applied = []
        for stem, path in self._prompt_files().items():
            try:
                text = path.read_text().strip()
            except OSError as exc:
                log.warning("could not read %s (%s)", path, exc)
                continue
            if not text:
                continue
            if stem == "agent":
                cloud.AGENT_SYSTEM_PROMPT = text
                self._loop.agent._system_facts = self._loop.system_facts
                applied.append("agent")
            elif stem == "planner":
                cloud.SYSTEM_PROMPT = text
                applied.append("planner")
            elif stem == "codegen":
                from . import codegen

                codegen.CODEGEN_SYSTEM = text
                applied.append("codegen")
            elif stem == "self":
                # Extra self-description appended to the live system facts.
                self._loop.system_facts = f"{self._loop.system_facts}\n\n{text}"
                self._loop.agent._system_facts = self._loop.system_facts
                applied.append("self")
        return applied

    # ---- settings ------------------------------------------------------
    def reload_settings(self) -> list[str]:
        """Re-read the environment file and re-apply what can change live."""
        from . import config as cfg_mod

        changed = []
        env_path = Path.home() / ".jettank.env"
        if env_path.exists():
            import os

            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ[k.strip()] = v.strip().strip("'\"")
        fresh = cfg_mod.load()
        # Only values the running loop actually consults each pass. Things
        # fixed at construction (camera device, mic) are not touched.
        for attr, new in (("vlm_interval", fresh.vlm.interval_s),
                          ("cloud_min_interval", fresh.cloud.min_interval_s)):
            if getattr(self._loop, attr) != new:
                setattr(self._loop, attr, new)
                changed.append(f"{attr}={new}")
        self._loop.cfg = fresh
        return changed

    # ---- hardware map --------------------------------------------------
    def reload_motion_map(self) -> list[str]:
        """Pick up tools/verify_motion.py results without a restart."""
        from .drive import load_verification

        driver = getattr(self._loop, "robot", None)
        if driver is None or not hasattr(driver, "_v"):
            return []
        before = dict(driver._v)
        v = load_verification()
        if v == before:
            return []
        driver._v = v
        driver.left_index = int(v.get("left_index", 1))
        driver.right_index = int(v.get("right_index", 2))
        driver._invert_left = bool(v.get("invert_left", False))
        driver._invert_right = bool(v.get("invert_right", False))
        driver._pan_sign = 1 if v.get("pan_sign", 1) >= 0 else -1
        driver._tilt_sign = 1 if v.get("tilt_sign", 1) >= 0 else -1
        return [f"motion map: {sorted(v)}"]

    # ---- modules -------------------------------------------------------
    def reload_modules(self) -> list[str]:
        done = []
        for name in RELOADABLE_MODULES:
            mod = sys.modules.get(name)
            if mod is None:
                continue
            try:
                importlib.reload(mod)
                done.append(name.split(".")[-1])
            except Exception as exc:  # noqa: BLE001 - a bad edit must not kill Hank
                log.error("could not reload %s: %s", name, exc)
        # Re-register tool schemas in case they changed shape.
        from .cloud import register_tools
        from .tools import TOOL_SCHEMAS

        register_tools(TOOL_SCHEMAS)
        return done

    # ---- everything ----------------------------------------------------
    def reload(self) -> dict:
        started = time.monotonic()
        result = {
            "modules": self.reload_modules(),
            "prompts": self.load_prompts(),
            "settings": self.reload_settings(),
            "hardware": self.reload_motion_map(),
        }
        # Skills and behaviours need no action: skills are read from JSON on
        # demand, and a behaviour is re-read and re-validated on every run.
        self._loop.skills._load()
        result["skills"] = self._loop.skills.names()
        result["elapsed_s"] = round(time.monotonic() - started, 2)
        result["needs_restart_for"] = [
            "audio devices", "the serial link", "the camera", "the voice model",
        ]
        log.info("[live] reloaded %s", {k: v for k, v in result.items() if v})
        return result

    # ---- watching ------------------------------------------------------
    def changed_files(self) -> list[str]:
        """Which watched files have a newer mtime than last time we looked."""
        watched: list[Path] = list(self._prompt_files().values())
        pkg = Path(__file__).resolve().parent
        watched += [pkg / f"{m.split('.')[-1]}.py" for m in RELOADABLE_MODULES]
        watched.append(Path.home() / ".jettank.env")
        from .drive import VERIFY_FILE

        watched.append(VERIFY_FILE)

        out = []
        for path in watched:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            key = str(path)
            if self._seen.get(key) is not None and mtime > self._seen[key]:
                out.append(path.name)
            self._seen[key] = mtime
        return out
