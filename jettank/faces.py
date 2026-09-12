"""Local face enrolment and identification.

Embeddings are computed and stored on the robot. Nothing about a face is sent
to the cloud model - it only ever sees names and confidences. That keeps
biometric data on-device even though the planner is remote.

Backends are tried in order of preference; if none is available the tools say
so honestly rather than pretending.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_DB = Path(os.environ.get("JETTANK_FACE_DB", str(Path.home() / ".jettank_faces.json")))
MATCH_THRESHOLD = 0.6   # face_recognition distance; lower is stricter


class FaceStore:
    def __init__(self, path: Path = DEFAULT_DB) -> None:
        self.path = path
        self._data: dict[str, list[list[float]]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
                log.info("loaded %d enrolled faces from %s", len(self._data), self.path)
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("could not read face db (%s); starting empty", exc)
                self._data = {}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data))
        tmp.replace(self.path)
        self.path.chmod(0o600)

    def add(self, name: str, encodings: list) -> int:
        self._data.setdefault(name, []).extend([list(map(float, e)) for e in encodings])
        self._save()
        return len(self._data[name])

    def names(self) -> list[str]:
        return sorted(self._data)

    def all(self) -> list[tuple[str, list[float]]]:
        return [(n, e) for n, encs in self._data.items() for e in encs]

    def forget(self, name: str) -> bool:
        if name in self._data:
            del self._data[name]
            self._save()
            return True
        return False


class FaceEngine:
    """Wraps whichever face backend is installed."""

    def __init__(self, store: FaceStore | None = None) -> None:
        self.store = store or FaceStore()
        self._fr = None
        self._np = None
        try:
            import face_recognition  # type: ignore
            import numpy  # type: ignore

            self._fr = face_recognition
            self._np = numpy
            log.info("face backend: face_recognition")
        except ImportError:
            log.warning(
                "no face backend available - install with: "
                "sudo apt-get install -y python3-numpy cmake && pip3 install face_recognition"
            )

    @property
    def available(self) -> bool:
        return self._fr is not None

    def list_names(self) -> list[str]:
        return self.store.names()

    def _decode(self, b64: str):
        import io

        raw = base64.b64decode(b64)
        return self._fr.load_image_file(io.BytesIO(raw))

    def enroll(self, name: str, samples: int, camera) -> dict:
        if not self.available:
            return {"ok": False, "error": "face recognition backend not installed on the robot"}
        name = name.strip()
        if not name:
            return {"ok": False, "error": "a name is required"}
        samples = max(1, min(int(samples), 10))
        got, skipped = [], 0
        for i in range(samples):
            _, b64 = camera.latest_jpeg_b64()
            if not b64:
                skipped += 1
                time.sleep(0.4)
                continue
            img = self._decode(b64)
            encs = self._fr.face_encodings(img)
            if len(encs) == 1:
                got.append(encs[0])
            else:
                skipped += 1   # zero or several faces: ambiguous, don't enrol
            time.sleep(0.6)
        if not got:
            return {
                "ok": False,
                "error": "no usable frames - need exactly one clearly visible face",
                "frames_tried": samples,
                "frames_skipped": skipped,
            }
        total = self.store.add(name, got)
        return {
            "ok": True, "name": name, "captured": len(got),
            "skipped": skipped, "total_for_person": total,
            "detail": f"enrolled {len(got)} sample(s) for {name}",
        }

    def identify(self, camera) -> dict:
        if not self.available:
            return {"ok": False, "error": "face recognition backend not installed on the robot"}
        known = self.store.all()
        if not known:
            return {"ok": True, "faces": [], "detail": "nobody is enrolled yet"}
        _, b64 = camera.latest_jpeg_b64()
        if not b64:
            return {"ok": False, "error": "no frame from the camera"}
        img = self._decode(b64)
        encs = self._fr.face_encodings(img)
        if not encs:
            return {"ok": True, "faces": [], "detail": "no faces visible"}
        names = [n for n, _ in known]
        vecs = self._np.array([e for _, e in known])
        out = []
        for enc in encs:
            dists = self._fr.face_distance(vecs, enc)
            idx = int(dists.argmin())
            best = float(dists[idx])
            out.append({
                "name": names[idx] if best <= MATCH_THRESHOLD else "unknown",
                "confidence": round(max(0.0, 1.0 - best), 3),
            })
        return {"ok": True, "faces": out, "detail": f"{len(out)} face(s) in view"}
