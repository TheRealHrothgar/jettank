"""Voice control words that never reach the cloud agent.

STOP and KILL are handled in the listen loop, before any reasoning happens.
That is the whole point: they exist to interrupt inference, so routing them
through inference would defeat them. They are also the only spoken input that
works while Hank is talking or thinking.

    STOP  abandon whatever is in flight - the agent turn, the speech, a running
          behaviour - and halt motion. Hank stays awake and listening.
    KILL  stop the process outright, in a way that systemd will not restart.
          Recovery is deliberately a shell action: someone has to be at a
          terminal, which is the point of a kill switch.

MATCHING IS WHOLE-UTTERANCE, NOT SUBSTRING. "stop" stops; "don't stop looking
at that" does not. A kill switch that fires on a word buried in a sentence is
worse than no kill switch, because people stop trusting it and start avoiding
the word. The cost of exact matching is that you must say it cleanly and alone,
which is also how people instinctively use an emergency control.
"""
from __future__ import annotations

import os
import re

# Exit code that tells systemd not to restart us. Paired with
# RestartPreventExitStatus in deploy/hank.service - without that line the
# service comes straight back up and KILL does nothing.
KILL_EXIT_CODE = 42


def _phrases(env: str, default: str) -> tuple[str, ...]:
    raw = os.environ.get(env, default)
    return tuple(p.strip().lower() for p in raw.split(",") if p.strip())


# Both are prefixed with his name, so neither can fire on ordinary speech, and
# they are phonetically far apart - "halt" and "override" share nothing. That
# distance matters asymmetrically: a STOP misheard as a KILL takes Hank offline
# until someone walks to a terminal, while the reverse merely fails to cancel.
#
# Hence the two lists are tuned differently on purpose:
#   STOP is permissive - it includes the ways speech-to-text actually mangles
#   "halt" (hold, alt, hall). A false stop costs one cancelled request.
#   KILL is strict - exact phrasings only, no near-misses. Better that it
#   occasionally fails to fire and you say it again.
STOP_PHRASES = _phrases(
    "JETTANK_STOP_WORDS",
    "hank halt,halt hank,hank hold,hank alt,hank hall,hanks halt,hank halts,"
    "hank haul,hank holt",
)
KILL_PHRASES = _phrases(
    "JETTANK_KILL_WORDS",
    "hank override,override hank",
)

# Arming is a control word, not a tool, for the same reason halt and override
# are: it is handled here, before any reasoning, so the decision to let the
# robot move is always a human's and never a model's. Hank can ASK to be armed
# (see the set_motion tool) but the arming itself happens only when a person
# says one of these out loud.
#
# Disarming is not symmetric with arming and is not meant to be. Anything may
# disarm - a tool call, the console, a control word - because stopping is
# always the safe direction. Only starting needs a human.
ARM_PHRASES = _phrases(
    "JETTANK_ARM_WORDS",
    "hank arm motion,hank enable motion,arm motion hank,enable motion hank,"
    "hank you may move,hank permission to move,hank motion on",
)
DISARM_PHRASES = _phrases(
    "JETTANK_DISARM_WORDS",
    "hank disarm,hank disarm motion,hank disable motion,disarm hank,"
    "hank motion off,hank do not move,hank dont move,hank stay still",
)

# Re-apply edited prompts, settings and the hardware map without restarting.
# A control word because it is most useful exactly when you are iterating and
# do not want to stop what he is doing.
RELOAD_PHRASES = _phrases(
    "JETTANK_RELOAD_WORDS",
    "hank reload,reload hank,hank refresh,hank reload yourself",
)

# Speech-to-text adds trailing punctuation and the occasional filler.
_FILLER = re.compile(r"^(?:uh|um|er|ah|ok|okay|hey|please|now|just)\s+", re.I)
_PUNCT = re.compile(r"[^\w\s]")


def normalise(text: str) -> str:
    """Reduce an utterance to bare words for exact comparison."""
    t = _PUNCT.sub(" ", (text or "").lower())
    t = " ".join(t.split())
    while True:
        stripped = _FILLER.sub("", t)
        if stripped == t:
            return t
        t = stripped


def classify(text: str) -> str | None:
    """Return 'kill', 'stop', 'disarm', 'arm', or None.

    Checked against the whole utterance only. Order is by safety, not by
    likelihood: kill, then stop, then disarm, then arm. If an utterance somehow
    matches more than one, the more restrictive reading wins - the only
    ambiguity that matters is one that could start the robot moving.
    """
    t = normalise(text)
    if not t:
        return None
    if t in KILL_PHRASES:
        return "kill"
    if t in STOP_PHRASES:
        return "stop"
    if t in DISARM_PHRASES:
        return "disarm"
    if t in ARM_PHRASES:
        return "arm"
    if t in RELOAD_PHRASES:
        return "reload"
    return None
