"""Voice input: microphone -> utterance segmentation -> local speech-to-text.

Speech stays on the device. The mic feeds a simple energy gate that cuts the
stream into utterances, and only the resulting *text* is ever sent to the cloud
agent. That keeps the always-on channel from becoming an always-on upload.

Capture shells out to `arecord` rather than binding a Python audio library:
the USB mic is an ALSA device, arecord is already installed on JetPack, and a
subprocess that dies cannot take the perception loop down with it.

Backends are selected by JETTANK_STT_BACKEND:
  auto           - try faster-whisper, then whisper.cpp, then give up
  faster-whisper - the Python package (CTranslate2; uses the GPU if built for it)
  whisper-cpp    - a whisper.cpp binary named by JETTANK_WHISPER_BIN
  none           - disabled
"""
from __future__ import annotations

import array
import logging
import math
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

log = logging.getLogger(__name__)

RATE = 16000          # what every whisper build expects
CHANNELS = 1
SAMPLE_BYTES = 2      # s16le
CHUNK_MS = 30
CHUNK_FRAMES = RATE * CHUNK_MS // 1000
CHUNK_BYTES = CHUNK_FRAMES * CHANNELS * SAMPLE_BYTES


def _rms(buf: bytes) -> float:
    """Root-mean-square of a signed 16-bit little-endian block, normalised 0..1."""
    if not buf:
        return 0.0
    samples = array.array("h")
    samples.frombytes(buf[: len(buf) - (len(buf) % 2)])
    if not samples:
        return 0.0
    total = sum(float(s) * float(s) for s in samples)
    return math.sqrt(total / len(samples)) / 32768.0


# ---------------------------------------------------------------- transcription

class Transcriber:
    """Speech-to-text over a WAV file. Returns '' rather than raising."""

    def __init__(self, backend: str = "auto", model: str = "base.en",
                 binary: str | None = None) -> None:
        self.backend = "none"
        self._model_name = model
        self._model = None
        self._binary = binary or os.environ.get("JETTANK_WHISPER_BIN", "whisper-cli")
        self._ggml = os.environ.get("JETTANK_WHISPER_MODEL", "")

        want = (backend or "auto").lower()
        if want == "none":
            return
        if want in ("auto", "faster-whisper") and self._try_faster_whisper():
            return
        if want in ("auto", "whisper-cpp") and self._try_whisper_cpp():
            return
        log.warning(
            "no speech-to-text backend available; voice control is off. "
            "Install faster-whisper (pip install faster-whisper) or set "
            "JETTANK_WHISPER_BIN/JETTANK_WHISPER_MODEL for whisper.cpp."
        )

    def _try_faster_whisper(self) -> bool:
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # noqa: BLE001
            log.debug("faster-whisper unavailable: %s", exc)
            return False
        for device, compute in (("cuda", "float16"), ("cpu", "int8")):
            try:
                self._model = WhisperModel(self._model_name, device=device,
                                           compute_type=compute)
                self.backend = f"faster-whisper:{device}"
                log.info("STT backend %s (%s)", self.backend, self._model_name)
                return True
            except Exception as exc:  # noqa: BLE001 - no CUDA build is normal
                log.debug("faster-whisper on %s failed: %s", device, exc)
        return False

    def _try_whisper_cpp(self) -> bool:
        if not shutil.which(self._binary):
            return False
        if not self._ggml or not Path(self._ggml).exists():
            log.debug("whisper.cpp binary found but JETTANK_WHISPER_MODEL is unset")
            return False
        self.backend = "whisper-cpp"
        log.info("STT backend whisper-cpp (%s)", self._ggml)
        return True

    @property
    def available(self) -> bool:
        return self.backend != "none"

    def transcribe(self, wav_path: str) -> str:
        if not self.available:
            return ""
        try:
            if self.backend.startswith("faster-whisper"):
                segments, _ = self._model.transcribe(wav_path, language="en",
                                                     vad_filter=True)
                return " ".join(s.text.strip() for s in segments).strip()
            out = subprocess.run(
                [self._binary, "-m", self._ggml, "-f", wav_path, "-nt", "-np", "-l", "en"],
                capture_output=True, text=True, timeout=120,
            )
            return " ".join(out.stdout.split()).strip()
        except Exception as exc:  # noqa: BLE001 - a bad utterance must not kill the loop
            log.warning("transcription failed: %s", exc)
            return ""


# ---------------------------------------------------------------- capture

class VoiceListener:
    """Yields transcribed utterances from the microphone.

    Segmentation is deliberately crude - an energy gate with hysteresis. It
    costs nothing, works on a noisy robot at arm's length, and whisper's own
    VAD cleans up whatever slips through.
    """

    def __init__(self, device: str = "default", transcriber: Transcriber | None = None,
                 threshold: float = 0.02, silence_ms: int = 700,
                 min_speech_ms: int = 400, max_utterance_s: float = 15.0) -> None:
        self.device = device
        self.stt = transcriber or Transcriber()
        self.threshold = threshold
        self._silence_chunks = max(1, silence_ms // CHUNK_MS)
        self._min_chunks = max(1, min_speech_ms // CHUNK_MS)
        self._max_chunks = int(max_utterance_s * 1000) // CHUNK_MS
        self._proc: subprocess.Popen | None = None
        self._muted = False

    @property
    def available(self) -> bool:
        return self.stt.available and shutil.which("arecord") is not None

    def mute(self, on: bool) -> None:
        """Gate capture while the robot is speaking, so it cannot hear itself."""
        self._muted = on

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            ["arecord", "-D", self.device, "-f", "S16_LE", "-r", str(RATE),
             "-c", str(CHANNELS), "-t", "raw", "-q"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        log.info("listening on %s (backend %s)", self.device, self.stt.backend)

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None

    def next_utterance(self) -> str:
        """Block until one utterance has been captured and transcribed.

        Returns '' on silence, a dropped mic, or an empty transcript - callers
        loop on it and should treat '' as 'nothing to do'.
        """
        if self._proc is None or self._proc.stdout is None:
            return ""
        speech: list[bytes] = []
        quiet = 0
        while True:
            chunk = self._proc.stdout.read(CHUNK_BYTES)
            if not chunk or len(chunk) < CHUNK_BYTES:
                return ""  # mic went away; caller decides whether to restart
            if self._muted:
                speech.clear()
                quiet = 0
                continue
            loud = _rms(chunk) >= self.threshold
            if loud:
                speech.append(chunk)
                quiet = 0
            elif speech:
                speech.append(chunk)
                quiet += 1
                if quiet >= self._silence_chunks:
                    break
            if len(speech) >= self._max_chunks:
                break
        if len(speech) < self._min_chunks + self._silence_chunks:
            return ""
        return self._transcribe(b"".join(speech))

    def _transcribe(self, pcm: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            path = fh.name
        try:
            with wave.open(path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPLE_BYTES)
                wf.setframerate(RATE)
                wf.writeframes(pcm)
            return self.stt.transcribe(path)
        finally:
            Path(path).unlink(missing_ok=True)


def match_wake_word(text: str, wake: str) -> str | None:
    """Return the command after the wake word, or None if it is absent.

    An empty wake word means always-on. Matching is loose on purpose: STT
    routinely renders 'hey tank' as 'hey, tank.' or 'Hey Tank!'.
    """
    if not wake:
        return text.strip()
    norm = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text)
    norm = " ".join(norm.split())
    key = " ".join(wake.lower().split())
    idx = norm.find(key)
    if idx < 0:
        return None
    return norm[idx + len(key):].strip()
