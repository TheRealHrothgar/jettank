"""Camera capture that always yields the most recent frame.

A robot loop must never consume stale frames, so the grabber thread keeps only
the latest one and drops everything else.
"""
from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time

log = logging.getLogger(__name__)


class GStreamerCamera:
    """Frame grabber using gst-launch-1.0.

    JetPack ships GStreamer but not OpenCV, and a link-local Jetson has no
    internet to install one, so this is the practical capture path on the
    robot. One subprocess per frame is coarse, but capture (~0.3 s) is an
    order of magnitude cheaper than the VLM inference it feeds, so it is not
    the bottleneck.
    """

    def __init__(self, device: str, width: int, height: int) -> None:
        self._device = device if device.startswith("/dev/") else f"/dev/video{device}"
        self._width, self._height = width, height
        self._seq = 0
        self._tmp = tempfile.mkdtemp(prefix="jettank-cam-")

    def start(self) -> None:
        if not shutil.which("gst-launch-1.0"):
            raise RuntimeError("gst-launch-1.0 not found")
        if not os.path.exists(self._device):
            raise RuntimeError(f"{self._device} does not exist")
        # prove it can actually produce a frame before we claim success
        if self.grab_jpeg() is None:
            raise RuntimeError(f"no frame from {self._device}")
        log.info("gstreamer camera ready on %s", self._device)

    def grab_jpeg(self) -> bytes | None:
        out = os.path.join(self._tmp, "frame.jpg")
        try:
            os.unlink(out)
        except FileNotFoundError:
            pass
        # Prefer the camera's native MJPEG (no conversion); fall back to encoding.
        pipelines = [
            ["gst-launch-1.0", "-q", "v4l2src", f"device={self._device}", "num-buffers=1",
             "!", f"image/jpeg,width={self._width},height={self._height}",
             "!", "filesink", f"location={out}"],
            ["gst-launch-1.0", "-q", "v4l2src", f"device={self._device}", "num-buffers=1",
             "!", "videoconvert", "!", "jpegenc", "!", "filesink", f"location={out}"],
        ]
        for pipe in pipelines:
            try:
                subprocess.run(pipe, timeout=20, check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                continue
            if os.path.exists(out) and os.path.getsize(out) > 1024:
                with open(out, "rb") as fh:
                    data = fh.read()
                if data[:3] == b"\xff\xd8\xff":   # valid JPEG SOI
                    self._seq += 1
                    return data
        return None

    def latest_jpeg_b64(self, quality: int = 80):
        data = self.grab_jpeg()
        if data is None:
            return 0, None
        return self._seq, base64.b64encode(data).decode("ascii")

    def stop(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)


def build_camera(device: str, width: int, height: int, fps: int):
    """Return the best available camera backend.

    OpenCV gives a proper threaded grabber; GStreamer is the fallback for the
    robot, where OpenCV is not installed.
    """
    try:
        import cv2  # noqa: F401
    except ImportError:
        log.info("OpenCV not available - using the GStreamer backend")
        return GStreamerCamera(device, width, height)
    return Camera(device, width, height, fps)


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
