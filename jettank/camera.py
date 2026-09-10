"""Camera capture that always yields the most recent frame.

A robot loop must never consume stale frames, so the grabber thread keeps only
the latest one and drops everything else.
"""
from __future__ import annotations

import base64
import logging
import threading
import time

log = logging.getLogger(__name__)


class Camera:
    def __init__(self, device: str, width: int, height: int, fps: int) -> None:
        self._device = int(device) if device.isdigit() else device
        self._width, self._height, self._fps = width, height, fps
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap = None

    def start(self) -> None:
        import cv2

        self._cap = cv2.VideoCapture(self._device)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        # Keep the driver buffer tiny so we read fresh frames, not queued ones.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open camera {self._device!r}")
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        log.info("camera %s opened at %dx%d", self._device, self._width, self._height)

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1

    def latest(self):
        """Return (seq, frame) or (0, None) if nothing has arrived yet."""
        with self._lock:
            if self._frame is None:
                return 0, None
            return self._seq, self._frame.copy()

    def latest_jpeg_b64(self, quality: int = 80) -> tuple[int, str | None]:
        import cv2

        seq, frame = self.latest()
        if frame is None:
            return 0, None
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return seq, None
        return seq, base64.b64encode(buf.tobytes()).decode("ascii")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
