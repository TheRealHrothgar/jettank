"""Driving forward while actually looking where he is going.

The loop is the point. Hank's vision-language model takes ~10s a frame, which
is fine for "what do you see?" and useless for not hitting things - at 0.18 m/s
he covers nearly two metres in one inference. The local detector runs in ~43ms,
so the control loop closes at ~8Hz and he stops within a few centimetres of
deciding to.

    capture -> detect -> is my path blocked? -> drive or stop -> repeat

WHAT "BLOCKED" MEANS FROM ONE CAMERA
A single camera cannot measure distance. What it can measure is how much of the
frame something occupies and where its base sits, and for obstacle avoidance
that is enough, because both grow sharply as you approach. Two signals are
combined:

  * lateral    is it in the central strip - the part he would actually hit -
               rather than off to one side he will pass
  * proximity  the bottom edge of the box. On a floor, nearer things have a
               lower base. This is far more reliable than box area, which
               confuses a distant sofa with a near chair leg.

Both are deliberately conservative. The failure modes are not symmetric:
stopping for a shadow costs a second, and not stopping costs a robot.

WHAT THIS CANNOT SEE - READ THIS BEFORE TRUSTING IT
An object detector only detects its classes. This one knows COCO: people,
chairs, dogs, sofas, bottles. It does NOT know about walls, doors, table legs,
steps, or a plain wooden board - and on the first live test Hank was nose-first
into exactly that and reported a clear path, because there was genuinely
nothing in the frame it had a name for.

So this is obstacle avoidance for THINGS, not for GEOMETRY. It will stop for a
person or a dog and drive straight into a wall. The mitigations are that the
run is time-boxed and the speed is low, which bounds the damage rather than
preventing it.

Measuring geometry needs a different sensor. The LIDAR is the right answer and
is currently blocked on power; monocular depth estimation would also work and
costs GPU we do not have spare. Until one of those exists, this belongs on a
robot someone is watching.

SAFETY
Every motion command goes through MotionGuard, so this inherits arming, speed
clamping and the watchdog. It cannot move a disarmed robot. It stops on its own
time limit, on a blocked path, on losing the camera, and when interrupted. The
stop is issued before the report, always - a robot that describes what it found
while still moving toward it has its priorities wrong.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# COCO ids this model emits that are worth stopping for. Deliberately broad:
# an unknown obstacle is still an obstacle, so anything detected in the path
# counts and this list only names them for the report.
COCO_NAMES = {
    1: "a person", 16: "a bird", 17: "a cat", 18: "a dog", 44: "a bottle",
    62: "a chair", 63: "a sofa", 64: "a potted plant", 65: "a bed",
    67: "a dining table", 70: "a toilet", 72: "a television", 73: "a laptop",
    77: "a mobile phone", 84: "a book", 85: "a clock", 86: "a vase",
    88: "a teddy bear", 90: "a toothbrush",
}

# Central strip of the frame he would actually drive into. Narrower than the
# full width on purpose: things at the edges get passed, not hit.
PATH_LEFT, PATH_RIGHT = 0.30, 0.70

# A box whose base sits below this is close enough to matter. Tuned for a
# camera about 12cm off the floor looking level.
NEAR_BASE = 0.72
# ...or which simply fills the view, for something tall and close.
NEAR_HEIGHT = 0.55

MIN_SCORE = 0.40


@dataclass
class Obstacle:
    name: str
    score: float
    box: tuple[float, float, float, float]      # x0, y0, x1, y1 normalised

    @property
    def base(self) -> float:
        return self.box[3]

    @property
    def centre_x(self) -> float:
        return (self.box[0] + self.box[2]) / 2.0

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def in_path(self) -> bool:
        # Overlapping the central strip at all, not just centred in it - a
        # chair leg clipping the edge of his track still stops him.
        return self.box[2] > PATH_LEFT and self.box[0] < PATH_RIGHT

    @property
    def near(self) -> bool:
        return self.base >= NEAR_BASE or self.height >= NEAR_HEIGHT

    @property
    def side(self) -> str:
        return "left" if self.centre_x < 0.5 else "right"


@dataclass
class DriveResult:
    travelled_s: float = 0.0
    stopped_because: str = ""
    obstacles: list[str] = field(default_factory=list)
    frames: int = 0
    detections: int = 0
    mean_detect_ms: float = 0.0

    def spoken(self) -> str:
        """A sentence, not a log line - this goes to the speech pipeline."""
        if self.stopped_because == "blocked" and self.obstacles:
            what = self.obstacles[0]
            return (f"I stopped after {self.travelled_s:.0f} seconds because "
                    f"{what} was in my way.")
        if self.stopped_because == "blocked":
            return f"I stopped after {self.travelled_s:.0f} seconds, something was in my way."
        if self.stopped_because == "distance":
            return f"I drove forward for {self.travelled_s:.0f} seconds and the way stayed clear."
        if self.stopped_because == "refused":
            return "I could not move. My motors are not armed."
        if self.stopped_because == "interrupted":
            return "I stopped because you told me to."
        if self.stopped_because == "no camera":
            return "I stopped because I could not see."
        return f"I stopped after {self.travelled_s:.0f} seconds."


def assess(detections, min_score: float = MIN_SCORE) -> tuple[bool, list[Obstacle]]:
    """Turn raw detections into (blocked, obstacles-worth-reporting).

    `detections` is (boxes, classes, scores) as the SSD model returns them.
    """
    boxes, classes, scores = detections
    found: list[Obstacle] = []
    for box, cls, score in zip(boxes, classes, scores):
        if float(score) < min_score:
            continue
        # This model emits ymin, xmin, ymax, xmax, normalised.
        y0, x0, y1, x1 = (float(v) for v in box)
        found.append(Obstacle(name=COCO_NAMES.get(int(cls), "something"),
                              score=round(float(score), 2),
                              box=(x0, y0, x1, y1)))
    blocking = [o for o in found if o.in_path and o.near]
    # Nearest first, by how low its base sits.
    blocking.sort(key=lambda o: -o.base)
    return bool(blocking), blocking or found


class ForwardDrive:
    """Drive forward until something is in the way, or time runs out."""

    def __init__(self, guard, camera, detector, speaker=None,
                 rate_hz: float = 8.0) -> None:
        self._guard = guard
        self._camera = camera
        self._detector = detector
        self._speaker = speaker
        self._period = 1.0 / max(1.0, rate_hz)
        self._abort = False

    def stop(self) -> None:
        """Interrupt from another thread."""
        self._abort = True

    def run(self, seconds: float = 5.0, speed: float = 0.5) -> DriveResult:
        """Drive forward for up to `seconds`, checking the path every frame.

        Time-boxed to 20s maximum regardless of what is asked, because the
        detector cannot see walls and a bounded run is the actual safety
        property here.
        """
        result = DriveResult()
        self._abort = False
        if not self._detector.load():
            self._guard.stop()
            result.stopped_because = "no detector"
            return result

        deadline = time.monotonic() + max(0.5, min(seconds, 20.0))
        started = time.monotonic()
        detect_ms: list[float] = []
        try:
            while time.monotonic() < deadline:
                if self._abort:
                    result.stopped_because = "interrupted"
                    break

                seq, frame = self._camera.latest_jpeg_b64()
                if not frame or len(frame) < 1024:
                    result.stopped_because = "no camera"
                    break
                result.frames += 1

                raw = self._detector.raw(frame)
                if raw is None:
                    result.stopped_because = "no detector"
                    break
                detect_ms.append(self._detector.last_ms)
                blocked, obstacles = assess(raw)
                result.detections += len(obstacles)

                if blocked:
                    # Stop FIRST, then describe. Always this order.
                    self._guard.stop()
                    result.stopped_because = "blocked"
                    result.obstacles = [f"{o.name} on my {o.side}" if not
                                        (0.4 < o.centre_x < 0.6) else
                                        f"{o.name} right in front of me"
                                        for o in obstacles[:2]]
                    break

                accepted, note = self._guard.drive(speed, 0.0, "forward drive")
                if not accepted:
                    result.stopped_because = "refused"
                    log.info("[drive] refused: %s", note)
                    break
                time.sleep(self._period)
            else:
                result.stopped_because = "distance"
        finally:
            # Unconditionally, on every path including an exception.
            self._guard.stop()

        result.travelled_s = round(time.monotonic() - started, 1)
        if detect_ms:
            result.mean_detect_ms = round(sum(detect_ms) / len(detect_ms), 1)
        log.info("[drive] %s after %.1fs (%d frames, %.0fms/detect)",
                 result.stopped_because, result.travelled_s, result.frames,
                 result.mean_detect_ms)
        return result
