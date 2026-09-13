"""Camera capture that always yields the most recent frame.

A robot loop must never consume stale frames, so the grabber thread keeps only
the latest one and drops everything else.
"""
from __future__ import annotations

import base64
import contextlib
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


def auto_expose(device: str, target: int = 115, tries: int = 8, camera=None) -> dict:
    """Set gain and exposure so frames are actually usable.

    This matters more than it sounds. The camera ships with gain at 0 and
    aperture-priority auto-exposure, and in the room this robot lives in that
    produced frames averaging 4/255 - effectively black. The object detector
    found nothing for several rounds of testing and the obvious suspects
    (model, threshold, lighting) were all wrong: the camera simply was not
    exposing.

    Switches to manual exposure and searches for a mid-grey average. Manual
    rather than auto because auto-exposure hunts as the robot drives between
    light and shade, and a detector fed a hunting exposure sees objects appear
    and vanish.
    """
    import base64
    import io
    import subprocess

    def apply(**kw):
        args: list[str] = []
        for k, v in kw.items():
            args += ["-c", f"{k}={v}"]
        subprocess.run(["v4l2-ctl", "-d", device] + args,
                       capture_output=True, timeout=5)

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        log.debug("numpy/PIL unavailable - leaving camera exposure alone")
        return {}

    # Use the caller's camera when there is one. A v4l2 device cannot be
    # opened twice, so creating our own here stole the loop's camera and left
    # it falling back to the slow per-frame backend - which is exactly the
    # bottleneck the streaming backend existed to remove.
    owned = camera is None
    cam = camera or build_camera(device, 640, 480, 30)
    if owned:
        cam.start()
    time.sleep(1.2)
    exposure, gain, measured = 2500, 100, 0.0
    try:
        for _ in range(tries):
            apply(auto_exposure=1, exposure_time_absolute=exposure,
                  gain=gain, brightness=0)
            # The sensor keeps delivering frames captured under the OLD
            # settings for a moment. Measuring one of those latched onto a
            # reading of 123 while the camera actually settled at 251 - a
            # blown-out frame the detector could do nothing with. Discard a
            # few, then average, so the number is what the camera is really
            # producing rather than what it was producing a moment ago.
            samples: list[float] = []
            for n in range(6):
                time.sleep(0.35)
                _, b64 = cam.latest_jpeg_b64()
                if not b64:
                    continue
                value = float(np.asarray(
                    Image.open(io.BytesIO(base64.b64decode(b64)))).mean())
                if n >= 3:                 # first three are stale
                    samples.append(value)
            if not samples:
                continue
            measured = sum(samples) / len(samples)
            if abs(measured - target) < 18:
                break
            if measured > target:
                # Drop gain before exposure: gain adds noise, and a detector
                # on a noisy frame invents boxes.
                if gain > 10:
                    gain = max(0, int(gain * target / max(measured, 1)))
                else:
                    exposure = max(20, int(exposure * target / max(measured, 1)))
            elif exposure < 4000:
                exposure = min(5000, int(exposure * target / max(measured, 1)))
            else:
                gain = min(100, gain + 20)
    except Exception as exc:  # noqa: BLE001 - never block startup on this
        log.warning("auto-exposure failed (%s)", exc)
    finally:
        if owned:
            cam.stop()
    log.info("camera exposure set: exposure=%d gain=%d (mean %.0f/255)",
             exposure, gain, measured)
    return {"exposure": exposure, "gain": gain, "mean": round(measured)}


class StreamingCamera:
    """One long-lived GStreamer pipeline, frames parsed off its stdout.

    The per-frame subprocess version costs ~800ms a frame, which was fine when
    the only consumer was a vision model taking ten seconds anyway. It is not
    fine for a control loop: measured on the robot, the obstacle-avoidance
    drive ran at 1.1Hz with an 80ms detector, so the camera was 90% of the
    cycle and Hank travelled ~16cm between looks.

    Keeping the pipeline open and reading MJPEG frames as they arrive removes
    that entirely. JPEG framing is self-delimiting - SOI ffd8ff to EOI ffd9 -
    so no container parsing is needed.
    """

    SOI = b"\xff\xd8\xff"
    EOI = b"\xff\xd9"

    def __init__(self, device: str, width: int, height: int, fps: int = 30) -> None:
        self._device = device if device.startswith("/dev/") else f"/dev/video{device}"
        self._width, self._height, self._fps = width, height, fps
        self._proc = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: bytes | None = None
        self._seq = 0

    def start(self) -> None:
        # Idempotent: build_camera() starts the pipeline to prove the device
        # works, and callers start it again. Spawning a second gst-launch on
        # the same device fails in a confusing way.
        if self._proc is not None and self._proc.poll() is None:
            return
        if not shutil.which("gst-launch-1.0"):
            raise RuntimeError("gst-launch-1.0 not found")
        if not os.path.exists(self._device):
            raise RuntimeError(f"{self._device} does not exist")
        self._proc = subprocess.Popen(
            ["gst-launch-1.0", "-q", "v4l2src", f"device={self._device}",
             "!", f"image/jpeg,width={self._width},height={self._height}",
             "!", "fdsink", "fd=1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self._stop.clear()
        self._thread = threading.Thread(target=self._read, name="camera", daemon=True)
        self._thread.start()
        for _ in range(50):                      # up to 5s for the first frame
            if self._latest is not None:
                log.info("streaming camera ready on %s", self._device)
                return
            time.sleep(0.1)
        self.stop()
        raise RuntimeError(f"no frame from {self._device}")

    def _read(self) -> None:
        buf = bytearray()
        while not self._stop.is_set() and self._proc and self._proc.stdout:
            chunk = self._proc.stdout.read(65536)
            if not chunk:
                return
            buf += chunk
            # Keep only the most recent complete frame; a control loop wants
            # the freshest image, never a backlog.
            end = buf.rfind(self.EOI)
            if end == -1:
                if len(buf) > 4_000_000:         # runaway guard
                    del buf[:-65536]
                continue
            start = buf.rfind(self.SOI, 0, end)
            if start == -1:
                del buf[:end + 2]
                continue
            with self._lock:
                self._latest = bytes(buf[start:end + 2])
                self._seq += 1
            del buf[:end + 2]

    def latest_jpeg_b64(self, quality: int = 80):
        with self._lock:
            data, seq = self._latest, self._seq
        if not data:
            return 0, None
        return seq, base64.b64encode(data).decode("ascii")

    def stop(self) -> None:
        """Shut the pipeline down gently.

        SIGTERM lets GStreamer close the v4l2 device and release its USB
        endpoints in order. Killing it instead leaves transfers in flight, and
        the resulting stop-endpoint command is what wedged the xHCI controller
        and took every USB device down with it. Worth waiting a few seconds
        for; only escalate if it genuinely will not exit.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._proc is not None:
            with contextlib.suppress(Exception):
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    log.warning("camera pipeline would not exit - killing it")
                    self._proc.kill()
                    self._proc.wait(timeout=2)
            self._proc = None
            # Give the device a moment to settle before anyone reopens it.
            time.sleep(0.4)


def build_camera(device: str, width: int, height: int, fps: int):
    """Return the best available camera backend.

    Streaming GStreamer first: it keeps one pipeline open and is ~10x faster
    per frame than respawning gst-launch, which is the difference between a
    control loop at 1Hz and one at 8Hz.
    """
    try:
        cam = StreamingCamera(device, width, height, fps)
        cam.start()
        return cam
    except Exception as exc:  # noqa: BLE001
        log.info("streaming camera unavailable (%s) - falling back", exc)

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
