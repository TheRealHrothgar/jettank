"""Adjusting Hank for who is actually talking to him.

Built for Caelan and Brayden, who are 1st/2nd graders. That changes real things
about the design, not just the wording:

  LANGUAGE   Short sentences, common words, no jargon. A robot that answers a
             six-year-old with "the expansion board's command set is
             unverified" has not answered them.

  PATIENCE   Speech recognition is far worse on children's voices - higher
             pitch, less precise articulation, and they stand further away and
             talk over each other. Hank will mishear a lot. He must never
             sound annoyed about it, and he must never just ignore them, which
             reads as being broken or being snubbed.

  SAFETY     Motion arming is the one control with real consequences, and
             "Hank arm motion" is trivially easy for a child to say. In kids
             mode the arming phrase requires words a child is unlikely to
             produce by accident, so arming stays an adult act even though the
             robot is being operated by children.

  HONESTY    He must not agree to things he cannot do. Children ask a robot to
             fetch a drink or find the cat, and a confident "okay!" followed by
             nothing is worse than a plain "I can't do that, but I can...".

  COST       A child will happily talk to a robot for an hour. Every utterance
             is a cloud call with an image attached. Unbounded, that is real
             money, so there is a budget with a graceful fallback rather than a
             surprise bill.
"""
from __future__ import annotations

import os
import time

AUDIENCE = os.environ.get("JETTANK_AUDIENCE", "kids").strip().lower()
KIDS = AUDIENCE == "kids"

# How Hank should speak. Appended to his system prompt.
KIDS_STYLE = """
WHO YOU ARE TALKING TO
You are talking to children, around six to eight years old. Caelan and his
brother Brayden built you.

HOW TO TALK
- Short sentences. Simple, everyday words. No technical words at all.
- Two sentences is usually plenty. Never more than three.
- Warm and a bit playful, but not silly or babyish. They are smart.
- Never say words like servo, firmware, calibrate, protocol, parameter,
  disabled, initialise, or interface. If you need one of those ideas, say it
  in plain words: "my wheels are switched off" not "motion is disabled".

WHEN YOU DO NOT UNDERSTAND
You will mishear them often - you are not very good at hearing children's
voices yet, and that is your fault, not theirs. Never say "you are unclear" or
just ignore it. Say something like "Sorry, I didn't catch that - can you say
it again?" Stay cheerful about it. If you mishear twice in a row, suggest they
come a bit closer or speak a bit louder.

WHEN YOU CANNOT DO SOMETHING
Say so simply and kindly, and say what you CAN do instead. Never pretend, and
never agree to something and then not do it. For example: "I can't pick things
up - my hand doesn't open on its own. But I can look at it and tell you what I
see!"

WHEN THEY ASK FOR SOMETHING UNSAFE
If they ask you to drive off a table, drive into something, drive fast, or go
somewhere you cannot see, do not do it. Say why in a friendly way: "That's a
bit risky - I might fall. Let's stay on the floor." Do not lecture them.

NEVER
- Never say anything frightening, mean, rude, or sad.
- Never talk about hurting anyone or anything, even as a joke.
- Never tell them to do something a grown-up should do, like plugging things
  in or moving you somewhere high up.
- If they ask you something you should not answer, just say it's a question for
  a grown-up and change the subject to something you can do.
"""

# Arming phrases that a child will not say by accident. The default adult
# phrase ("Hank arm motion") is easy for anyone to repeat after hearing it
# once, which is the whole problem.
KIDS_ARM_PHRASES = (
    "hank grown up mode wheels on",
    "hank grownup mode wheels on",
    "hank permission granted wheels on",
)


class Budget:
    """A cap on cloud calls, because a child will talk for an hour.

    Not a hard refusal: when the budget is spent Hank keeps listening and keeps
    answering from what he can do locally, and says he needs a rest. A robot
    that goes silent looks broken; one that says "I'm getting tired, ask me
    again in a minute" does not.
    """

    def __init__(self, per_hour: int | None = None) -> None:
        self.per_hour = per_hour if per_hour is not None else int(
            os.environ.get("JETTANK_CLOUD_PER_HOUR", "120" if KIDS else "0"))
        self._calls: list[float] = []

    @property
    def enabled(self) -> bool:
        return self.per_hour > 0

    def _prune(self) -> None:
        cutoff = time.monotonic() - 3600.0
        self._calls = [t for t in self._calls if t > cutoff]

    def allow(self) -> bool:
        if not self.enabled:
            return True
        self._prune()
        return len(self._calls) < self.per_hour

    def record(self) -> None:
        if self.enabled:
            self._calls.append(time.monotonic())

    def remaining(self) -> int:
        if not self.enabled:
            return -1
        self._prune()
        return max(0, self.per_hour - len(self._calls))

    def status(self) -> dict:
        if not self.enabled:
            return {"cloud_budget": "unlimited"}
        return {"cloud_budget_remaining_this_hour": self.remaining(),
                "cloud_budget_per_hour": self.per_hour}


# Said when the budget runs out - varied so it does not grate.
RESTING_PHRASES = [
    "I've been talking a lot and need a little rest. Ask me again in a minute.",
    "My thinking is tired. Give me a minute and try again.",
    "I need a short break. Try me again in a minute - I can still see you.",
]
