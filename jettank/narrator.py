"""Translate machine text into something worth hearing.

The earlier attempt pushed this upstream: the code generator was told to write
speech-friendly strings. That was the wrong place. It cripples the data - a
behaviour that sweeps the camera *should* return per-angle observations, not a
pre-chewed sentence - and it only fixes the one producer that got the
instruction. Agent replies, VLM descriptions and tool results all end up spoken
too, and none of them are written for the ear either.

So narration is its own stage, at the point of speaking:

    structured/machine text  ->  Narrator (local 1.5B LLM)  ->  for_speech()  ->  TTS

The small local model is the right tool. Rephrasing is easy, the model is
already on the robot, it runs on the GPU in well under a second, and it costs
nothing per call - so this can sit in front of every utterance without adding
cloud latency or spend.

Two deliberate properties:
  * Narration never invents. The prompt forbids adding facts, and the result
    is discarded if it comes back implausibly long - a small model that starts
    rambling is caught rather than spoken.
  * It always degrades to `for_speech()`. If Ollama is down, slow, or the
    model is missing, Hank still talks; he just sounds more mechanical.

Synchronous on purpose: it is called from Speaker.say(), which runs on worker
threads and inside the sandbox bridge as well as on the event loop, and a sync
client sidesteps the cross-loop problems an async one causes there.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

SYSTEM = (
    "You turn a robot's internal notes into one or two sentences it can say out loud.\n"
    "Rules:\n"
    "- Keep every fact. Add nothing. Invent nothing.\n"
    "- Drop field names, angles as raw numbers, punctuation used as separators, "
    "and repetition.\n"
    "- Use natural spatial words: 'on my left', 'straight ahead', 'to my right'.\n"
    "- Speak as the robot, first person.\n"
    "- Two sentences maximum. No lists, no symbols, no markdown.\n"
    "- Output ONLY the sentence. No preamble, no quotes."
)

# Text that already reads as prose does not need a model round-trip. These are
# the markers of text written for a screen rather than a listener.
_MACHINE = re.compile(
    r"[{}\[\]<>|`*_=]"           # code/markup punctuation
    r"|\w+\s*[:=]\s*-?\d"        # field: value / field=value
    r"|;\s"                      # semicolon as a separator
    r"|\b-?\d+\s*(?:deg|degrees|px|ms)\b"
    r"|\b(?:pan|tilt|ok|true|false|null|none)\b\s*[:=]"
    r"|\s-\s-?\d"                # " - -60"
)


def needs_narration(text: str) -> bool:
    """Is this machine-shaped enough to be worth a rephrase?"""
    if not text or len(text) < 12:
        return False
    if _MACHINE.search(text):
        return True
    # Several short colon/semicolon-joined clauses is a log line, not a sentence.
    return text.count(";") >= 2 or text.count(":") >= 2


COMPOSE_SYSTEM = (
    "You are the voice of a small robot called Hank. You are given structured "
    "findings the robot collected, and you say what it found, out loud, as Hank.\n"
    "Rules:\n"
    "- ALWAYS first person: 'I saw', 'in front of me', 'on my left'. NEVER say "
    "'Hank' or 'the robot' or 'it' - you ARE Hank, talking about yourself.\n"
    "- Two or three sentences. Natural spoken English.\n"
    "- Translate data into words: angles become 'on my left' / 'straight ahead' / "
    "'on my right'; booleans and keys become plain statements.\n"
    "- Group and summarise. Do not read every entry if they repeat.\n"
    "- Keep every fact, add none, invent none. If something failed, say so.\n"
    "- No symbols, no lists, no markdown, no field names.\n"
    "- Output ONLY what Hank says."
)


def speakable_gist(findings, limit: int = 5) -> str:
    """Salvage the human-readable prose out of structured findings.

    Used only when the local model cannot be reached. The rule is narrow on
    purpose: keys, numbers, booleans and short tokens are NOT spoken, because
    reading a data structure aloud is the exact failure this whole stage
    exists to prevent. Free-text values are already English, so those are
    lifted out and joined. If there is no prose in there at all, this returns
    an empty string and Hank says nothing rather than saying nonsense.
    """
    found: list[str] = []

    def walk(node) -> None:
        if len(found) >= limit:
            return
        if isinstance(node, str):
            # Free text, not an identifier or a code-ish token.
            if len(node.split()) >= 3 and not _MACHINE.search(node):
                text = node.strip()
                if text not in found:
                    found.append(text)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)

    walk(findings)
    if not found:
        return ""
    out = ". ".join(f.rstrip(".") for f in found)
    return out + "."


class Narrator:
    def __init__(self, base_url: str, model: str, timeout_s: float = 12.0,
                 compose_timeout_s: float = 45.0, enabled: bool = True) -> None:
        # Composition gets a far longer budget than narration: it happens once
        # at the end of a behaviour, not per utterance, and the first call may
        # have to load the model into VRAM.
        self._compose_timeout = compose_timeout_s
        self._base = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_s
        self._enabled = enabled
        self._client = None
        self._broken = False        # stop retrying a backend that is not there

    @property
    def available(self) -> bool:
        return self._enabled and not self._broken

    def _http(self, timeout: float | None = None):
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self._compose_timeout)
        return self._client

    def warm(self) -> None:
        """Load the model into VRAM ahead of first use.

        Without this the first composition pays the cold-load cost and times
        out, which silently falls back to speaking raw JSON.
        """
        if not self.available:
            return
        try:
            self._http().post(f"{self._base}/api/generate",
                              json={"model": self._model, "prompt": "hi",
                                    "stream": False,
                                    "options": {"num_predict": 1}})
            log.info("narrator ready (%s)", self._model)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not warm narrator %s (%s)", self._model, exc)

    def narrate(self, text: str) -> str:
        """Return a spoken-English version of `text`, or `text` unchanged."""
        if not self.available or not needs_narration(text):
            return text
        try:
            r = self._http().post(
                f"{self._base}/api/generate",
                json={
                    "model": self._model,
                    "system": SYSTEM,
                    "prompt": text,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 120},
                },
            )
            r.raise_for_status()
            out = (r.json().get("response") or "").strip().strip('"')
        except Exception as exc:  # noqa: BLE001 - speech must never depend on this
            log.warning("narrator unavailable (%s); speaking the raw text", exc)
            self._broken = True
            return text

        if not out:
            return text
        # A small model that starts rambling gets discarded rather than spoken.
        if len(out) > max(240, len(text) * 3):
            log.debug("narrator output too long (%d chars); using raw text", len(out))
            return text
        # First line only - some models tack on an explanation.
        return out.splitlines()[0].strip() or text

    def compose(self, findings, context: str = "") -> str:
        """Turn a behaviour's structured findings into something to say.

        This is the counterpart to behaviours being silent. They return data;
        this composes it. Unlike narrate(), it always runs - structured
        findings are never speakable as-is - and it is given the originating
        request so the reply answers the question that was asked.
        """
        import json as _json

        if isinstance(findings, (dict, list)):
            payload = _json.dumps(findings, default=str)[:4000]
        else:
            payload = str(findings)[:4000]
        if not payload.strip():
            return ""
        prompt = (f"The robot was asked to: {context}\n\n" if context else "")
        prompt += f"It gathered these findings:\n{payload}\n\nSay what it found."
        if not self.available:
            return speakable_gist(findings)
        try:
            r = self._http().post(
                f"{self._base}/api/generate",
                json={"model": self._model, "system": COMPOSE_SYSTEM,
                      "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.4, "num_predict": 160}},
            )
            r.raise_for_status()
            out = (r.json().get("response") or "").strip().strip('"')
        except Exception as exc:  # noqa: BLE001
            log.warning("composer unavailable (%s); falling back to a plain summary", exc)
            return speakable_gist(findings)
        return out.splitlines()[0].strip() if out else flatten(findings)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
