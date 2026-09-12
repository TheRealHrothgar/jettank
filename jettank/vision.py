"""Fast local person detection - the perception a control loop can actually use.

Hank's vision-language model takes about ten seconds a frame. That is fine for
"what do you see?" and useless for "follow me": by the time it answers, the
person has left the room. Following needs bearing updated several times a
second, which is a different kind of perception, not a faster version of the
same one.

    VLM (qwen2.5vl:3b)   ~10 s/frame   what things ARE
    this (ssd-mobilenet) ~43 ms/frame  where a person IS

Deliberately on the CPU. The GPU is held by the vision model at 99% whenever
it runs, on a board where CPU and GPU share 8GB, so a detector competing for it
would slow both. At 43ms on six idle ARM cores there is no reason to.

Bearing is trustworthy; range is not. A bounding box gives a good horizontal
angle and only a crude distance, because apparent height depends on the
person's actual height, their posture, and whether their legs are in frame. So
the distance here is labelled an estimate and is the part the LIDAR should
replace when it is wired up - camera for bearing and identity, LIDAR for range.
"""
from __future__ import annotations

import io
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_PATH = Path(os.environ.get(
    "JETTANK_DETECT_MODEL",
    Path(__file__).resolve().parent.parent / "models" / "ssd.onnx"))

# COCO class 1 is person in this model's label map.
PERSON_CLASS = 1
MIN_SCORE = float(os.environ.get("JETTANK_DETECT_MIN_SCORE", "0.45"))

# Horizontal field of view of the USB camera, degrees. Bearing is only as good
# as this number; measure it if precision ever matters.
CAMERA_HFOV = float(os.environ.get("JETTANK_CAMERA_HFOV", "60.0"))

# Rough scale for the distance estimate: an average standing adult is ~1.7m,
# and a box spanning the full frame height at this FOV is about a metre away.
TYPICAL_PERSON_M = 1.7


@dataclass(frozen=True)
class Person:
    """One detection, in units a control loop can use directly."""

    bearing_deg: float      # negative left, positive right, from centre
    score: float
    box: tuple[float, float, float, float]   # normalised x0,y0,x1,y1
    distance_m: float | None                 # ESTIMATE - see module docstring

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def centred(self) -> bool:
        return abs(self.bearing_deg) < 8.0


class PersonDetector:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.path = Path(model_path)
        self._session = None
        self._input = None
        self.available = False
        self.last_ms = 0.0

    def load(self) -> bool:
        if self._session is not None:
            return True
        if not self.path.exists():
            log.info("no detection model at %s - person tracking unavailable",
                     self.path)
            return False
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            # Leave a core free: whisper and Piper also want CPU, and a
            # detector that starves them makes Hank deaf while he looks.
            opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 2)
            self._session = ort.InferenceSession(
                str(self.path), opts, providers=["CPUExecutionProvider"])
            self._input = self._session.get_inputs()[0]
            self.available = True
            log.info("person detector ready (%s)", self.path.name)
            return True
        except Exception as exc:  # noqa: BLE001 - absence must not break Hank
            log.warning("could not load detector (%s)", exc)
            return False

    def raw(self, jpeg_b64: str):
        """Run the model and return (boxes, classes, scores) for ALL classes.

        detect() filters to people; obstacle avoidance wants everything, since
        an unrecognised object is still something to not drive into.
        Returns None on failure rather than raising - the caller is a control
        loop and needs to stop, not crash.
        """
        if not self.load():
            return None
        try:
            import base64

            import numpy as np
            from PIL import Image

            img = Image.open(io.BytesIO(base64.b64decode(jpeg_b64))).convert("RGB")
            x = np.asarray(img.resize((300, 300)), dtype=np.uint8)[None, ...]
            started = time.monotonic()
            boxes, classes, scores, _ = self._session.run(
                None, {self._input.name: x})
            self.last_ms = (time.monotonic() - started) * 1000
            return boxes[0], classes[0], scores[0]
        except Exception as exc:  # noqa: BLE001
            log.warning("detection failed: %s", exc)
            return None

    def detect(self, jpeg_b64: str, max_people: int = 4) -> list[Person]:
        """Find people in a base64 JPEG. Returns [] on any failure."""
        if not self.load():
            return []
        try:
            import base64

            import numpy as np
            from PIL import Image

            img = Image.open(io.BytesIO(base64.b64decode(jpeg_b64))).convert("RGB")
            # This model takes uint8 NHWC at whatever size it is given; 300px
            # is what it was trained at and is plenty for a person at 3 metres.
            small = img.resize((300, 300))
            x = np.asarray(small, dtype=np.uint8)[None, ...]

            started = time.monotonic()
            boxes, classes, scores, _ = self._session.run(
                None, {self._input.name: x})
            self.last_ms = (time.monotonic() - started) * 1000
        except Exception as exc:  # noqa: BLE001
            log.warning("detection failed: %s", exc)
            return []

        found: list[Person] = []
        for box, cls, score in zip(boxes[0], classes[0], scores[0]):
            if int(cls) != PERSON_CLASS or float(score) < MIN_SCORE:
                continue
            # This model emits ymin, xmin, ymax, xmax - normalised.
            y0, x0, y1, x1 = (float(v) for v in box)
            centre_x = (x0 + x1) / 2.0
            bearing = (centre_x - 0.5) * CAMERA_HFOV
            height = max(1e-3, y1 - y0)
            # Crude similar-triangles estimate; see the module docstring for
            # why this is not to be trusted for anything but "nearer/further".
            distance = TYPICAL_PERSON_M / (2 * height * math.tan(
                math.radians(CAMERA_HFOV / 2)))
            found.append(Person(bearing_deg=round(bearing, 1),
                                score=round(float(score), 2),
                                box=(x0, y0, x1, y1),
                                distance_m=round(min(distance, 15.0), 1)))
            if len(found) >= max_people:
                break
        # Nearest first: the one to follow is the one filling most of the frame.
        found.sort(key=lambda p: -p.height)
        return found

    def nearest(self, jpeg_b64: str) -> Person | None:
        people = self.detect(jpeg_b64)
        return people[0] if people else None


def fuse(person: Person | None, lidar_range_m: float | None) -> dict:
    """Combine camera bearing with a LIDAR range, when one is available.

    Kept separate and tolerant of a missing LIDAR on purpose: the camera is
    what identifies a person, the LIDAR is what measures the gap, and either
    can be absent. With both, bearing comes from vision and range from the
    laser, which is the combination worth having - vision range is a guess and
    laser bearing does not tell you what it hit.
    """
    if person is None:
        return {"seen": False}
    out = {
        "seen": True,
        "bearing_deg": person.bearing_deg,
        "confidence": person.score,
        "centred": person.centred,
        "distance_m": person.distance_m,
        "range_source": "camera estimate",
    }
    if lidar_range_m is not None:
        out["distance_m"] = round(lidar_range_m, 2)
        out["range_source"] = "lidar"
    return out
