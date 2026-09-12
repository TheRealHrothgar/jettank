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
import collections
import math
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

log = logging.getLogger(__name__)

RATE = 16000          # what every whisper build expects
# The speakerphone runs at exactly one rate, 48000 Hz, in both directions. Ask
# arecord for 16000 and ALSA's plug layer resamples in realtime - and with
# playback also running it does that alongside a second live conversion, on a
# full-speed USB device. Capture at the device's own rate and convert offline;
# whisper still gets its 16 kHz, ALSA is left with nothing to do.
DEVICE_RATE = int(os.environ.get("JETTANK_AUDIO_RATE", "48000"))
CHANNELS = 1
SAMPLE_BYTES = 2      # s16le
CHUNK_MS = 30
CHUNK_FRAMES = RATE * CHUNK_MS // 1000
CHUNK_BYTES = CHUNK_FRAMES * CHANNELS * SAMPLE_BYTES
# What we actually pull off the device per chunk, before converting to 16 kHz.
DEVICE_CHUNK_BYTES = DEVICE_RATE * CHUNK_MS // 1000 * CHANNELS * SAMPLE_BYTES


def downsample(pcm: bytes, src_rate: int = DEVICE_RATE, dst_rate: int = RATE) -> bytes:
    """Device rate -> whisper's rate, done here rather than by ALSA."""
    if src_rate == dst_rate or not pcm:
        return pcm
    try:
        import audioop

        out, _ = audioop.ratecv(pcm, SAMPLE_BYTES, CHANNELS, src_rate, dst_rate, None)
        return out
    except Exception as exc:  # noqa: BLE001
        log.debug("offline downsample unavailable (%s)", exc)
        return pcm
WARMUP_MS = 1200      # arecord + AGC settling; the levels here are garbage


def pick_mic(preferred: str = "auto") -> str:
    """Find the USB microphone's ALSA name.

    Do not trust `default`. On JetPack the default capture device is one of the
    Tegra APE virtual cards, which opens happily and returns *digital silence*
    forever - no error, no warning, just a mic that never hears anything. The
    USB capture card is the one we want, addressed as plughw:<card>,0 so ALSA
    handles any rate conversion.
    """
    if preferred and preferred != "auto":
        return preferred
    return _pick("arecord", "microphone")


def _pick(tool: str, what: str) -> str:
    try:
        out = subprocess.run([tool, "-l"], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception as exc:  # noqa: BLE001
        log.warning("could not enumerate %s devices (%s)", what, exc)
        return "default"
    for line in out.splitlines():
        if not line.startswith("card "):
            continue
        # "card 0: Phone [USB Speaker Phone], device 0: USB Audio [USB Audio]"
        if "APE" in line or "ADMAIF" in line or "HDA" in line:
            continue
        try:
            card = line.split()[1].rstrip(":")
            dev = line.split("device ")[1].split(":")[0].strip()
        except (IndexError, ValueError):
            continue
        name = f"plughw:{card},{dev}"
        log.info("%s: %s (%s)", what, name, line.split("[")[1].split("]")[0])
        return name
    log.warning("no USB %s found; falling back to 'default', which on this "
                "board may be silent", what)
    return "default"


def pick_speaker(preferred: str = "auto") -> str:
    """Find the USB speaker's ALSA name, same reasoning as pick_mic()."""
    if preferred and preferred != "auto":
        return preferred
    return _pick("aplay", "speaker")


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


class SpeechDetector:
    """Neural voice-activity detection (Silero, shipped with faster-whisper).

    An energy gate cannot segment this robot's audio. The USB speakerphone has
    hardware AGC and noise suppression, so it normalises levels: measured on
    the bench, speech sat only 2.2x above silence and the silence p90 was
    *above* the speech median. Any fixed threshold either swallows quiet speech
    or trips constantly on fan noise. Silero looks at spectral shape instead of
    loudness, which AGC does not flatten.
    """

    def __init__(self) -> None:
        self._vad = None
        self._np = None
        try:
            import numpy as np

            from faster_whisper.vad import VadOptions, get_speech_timestamps

            self._np = np
            self._get = get_speech_timestamps
            self._opts = VadOptions
            self._vad = True
        except Exception as exc:  # noqa: BLE001
            log.debug("Silero VAD unavailable (%s); falling back to energy gate", exc)

    @property
    def available(self) -> bool:
        return bool(self._vad)

    def speech_regions(self, pcm: bytes, min_silence_ms: int) -> list[dict]:
        """Return [{'start': sample, 'end': sample}] for speech in `pcm`."""
        audio = self._np.frombuffer(pcm, dtype=self._np.int16).astype("float32") / 32768.0
        return self._get(audio, self._opts(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=200,
        ))


# ---------------------------------------------------------------- capture

class VoiceListener:
    """Yields transcribed utterances from the microphone.

    Segmentation is deliberately crude - an energy gate with hysteresis. It
    costs nothing, works on a noisy robot at arm's length, and whisper's own
    VAD cleans up whatever slips through.
    """

    def __init__(self, device: str = "auto", transcriber: Transcriber | None = None,
                 threshold: float = 0.02, silence_ms: int = 700,
                 min_speech_ms: int = 400, max_utterance_s: float = 15.0,
                 preroll_ms: int = 400) -> None:
        self.device = pick_mic(device)
        self.stt = transcriber or Transcriber()
        self.threshold = threshold
        self._silence_ms = silence_ms
        self._silence_chunks = max(1, silence_ms // CHUNK_MS)
        self._min_chunks = max(1, min_speech_ms // CHUNK_MS)
        self._max_chunks = int(max_utterance_s * 1000) // CHUNK_MS
        # Audio from *before* the gate opened. An energy gate necessarily
        # triggers partway into the first syllable, and the first word is the
        # wake word - the one word we cannot afford to clip. So keep a rolling
        # buffer and prepend it when speech starts.
        self._preroll: collections.deque[bytes] = collections.deque(
            maxlen=max(1, preroll_ms // CHUNK_MS))
        self._proc: subprocess.Popen | None = None
        self._muted = False
        self._detector = SpeechDetector()
        self._max_bytes = int(max_utterance_s * 1000) // CHUNK_MS * CHUNK_BYTES

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
            ["arecord", "-D", self.device, "-f", "S16_LE", "-r", str(DEVICE_RATE),
             "-c", str(CHANNELS), "-t", "raw", "-q"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        # arecord and the speakerphone's AGC both settle over the first second
        # and emit a transient that calibration and VAD would both misread.
        self._discard(WARMUP_MS)
        log.info("listening on %s (backend %s)", self.device, self.stt.backend)

    def _read_chunk(self) -> bytes | None:
        """One chunk at the device rate, returned at whisper's rate."""
        if self._proc is None or self._proc.stdout is None:
            return None
        raw = self._proc.stdout.read(DEVICE_CHUNK_BYTES)
        if not raw or len(raw) < DEVICE_CHUNK_BYTES:
            return None
        return downsample(raw)

    def _discard(self, ms: int) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        for _ in range(max(0, ms // CHUNK_MS)):
            if self._read_chunk() is None:
                return

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None

    def calibrate(self, seconds: float = 1.5) -> float:
        """Measure the room + fan noise floor and lift the gate above it.

        The Jetson's fan and the USB speakerphone's own preamp put the floor
        around 0.005-0.02 depending on power mode, which is right on top of a
        fixed threshold. Measuring beats guessing, but never let calibration
        *lower* the gate below the configured value - a loud room should make
        the robot harder to trigger, not easier.
        """
        if self._proc is None or self._proc.stdout is None:
            return self.threshold
        peak = 0.0
        for _ in range(max(1, int(seconds * 1000) // CHUNK_MS)):
            chunk = self._read_chunk()
            if chunk is None:
                break
            peak = max(peak, _rms(chunk))
        # 3x an already-high floor puts the gate above conversational speech.
        # 1.6x with a ceiling keeps quiet talkers audible.
        floor = round(min(peak * 1.6, 0.12), 4)
        if floor > self.threshold:
            log.info("noise floor %.4f - raising speech gate %.4f -> %.4f",
                     peak, self.threshold, floor)
            self.threshold = floor
        else:
            log.info("noise floor %.4f - keeping speech gate at %.4f",
                     peak, self.threshold)
        return self.threshold

    def next_utterance(self) -> str:
        """Block until one utterance has been captured and transcribed.

        Returns '' on silence, a dropped mic, or an empty transcript - callers
        loop on it and should treat '' as 'nothing to do'.
        """
        if self._proc is None or self._proc.stdout is None:
            return ""
        if self._detector.available:
            return self._next_vad()
        return self._next_energy()

    def _next_vad(self) -> str:
        """Accumulate audio and cut when Silero says the speaker has finished."""
        buf = bytearray()
        while True:
            chunk = self._read_chunk()
            if chunk is None:
                return ""
            if self._muted:
                buf.clear()
                continue
            buf.extend(chunk)
            # Only re-run VAD every ~300ms; it is cheap but not free.
            if len(buf) % (CHUNK_BYTES * 10) or len(buf) < CHUNK_BYTES * 20:
                if len(buf) < self._max_bytes:
                    continue
            try:
                regions = self._detector.speech_regions(bytes(buf), self._silence_ms)
            except Exception as exc:  # noqa: BLE001 - never let VAD kill the loop
                log.warning("VAD failed (%s); using energy gate this round", exc)
                return self._next_energy()

            if not regions:
                # Nothing but noise. Keep a tail in case speech just started.
                if len(buf) > CHUNK_BYTES * 100:
                    del buf[:-CHUNK_BYTES * 20]
                continue

            end = regions[-1]["end"] * SAMPLE_BYTES
            trailing_ms = (len(buf) - end) // (CHUNK_BYTES // CHUNK_MS)
            if trailing_ms >= self._silence_ms or len(buf) >= self._max_bytes:
                start = regions[0]["start"] * SAMPLE_BYTES
                return self._transcribe(bytes(buf[start:end]))

    def _next_energy(self) -> str:
        """Fallback segmenter for when Silero is not installed."""
        speech: list[bytes] = []
        quiet = 0
        while True:
            chunk = self._read_chunk()
            if chunk is None:
                return ""  # mic went away; caller decides whether to restart
            if self._muted:
                speech.clear()
                self._preroll.clear()
                quiet = 0
                continue
            loud = _rms(chunk) >= self.threshold
            if loud:
                if not speech:
                    speech.extend(self._preroll)  # recover the clipped onset
                    self._preroll.clear()
                speech.append(chunk)
                quiet = 0
            elif not speech:
                self._preroll.append(chunk)
            else:
                speech.append(chunk)
                quiet += 1
                if quiet >= self._silence_chunks:
                    break
            if len(speech) >= self._max_chunks:
                break
        if len(speech) < self._min_chunks + self._silence_chunks + len(self._preroll):
            return ""
        self._preroll.clear()
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


_FILLERS = {"um", "uh", "er", "ah", "so", "well", "okay", "ok", "hey", "hi",
            "hello", "please", "yeah", "and"}

# Far-field speech in a noisy room defeats exact matching. Observed in a live
# session, base.en rendered attempts to address him as "I think team", "hey
# hey", and similar - he was listening the whole time and simply never saw the
# name. So the name itself is matched by edit distance, not equality.
#
# The blocklist is the whole reason this is safe. "thank" is one edit from
# "hank" and "thank you" is extremely common, so it must never match; same for
# ordinary words that happen to land nearby.
_NAME_BLOCK = {"thank", "thanks", "think", "thinks", "than", "that", "hand",
               "hands", "happy", "have", "hard", "back", "black", "bank",
               "rank", "ranks", "hankering"}


def _close_enough(word: str, name: str, max_edits: int = 1) -> bool:
    """Is `word` within `max_edits` of `name`? Short words are matched exactly.

    Levenshtein, bounded and written out rather than pulled in, because one
    function is cheaper than a dependency on the robot.
    """
    if word in _NAME_BLOCK:
        return False
    if abs(len(word) - len(name)) > max_edits:
        return False
    if len(name) <= 3:
        return word == name
    prev = list(range(len(name) + 1))
    for i, wc in enumerate(word, 1):
        cur = [i]
        for j, nc in enumerate(name, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wc != nc)))
        if min(cur) > max_edits:
            return False
        prev = cur
    return prev[-1] <= max_edits


def match_wake_word(text: str, wake: str) -> str | None:
    """Return what was said after the wake phrase, or None if absent.

    `wake` is a comma-separated list, because people do not use one fixed
    phrase. Watching a live session, Hank was addressed as "Hi Hank", "How are
    you doing Hank?" and just "Hank." within a minute, and ignored all of it
    while waiting for exactly "hey hank" - which reads, to the person talking,
    as a robot that is broken.

    Empty means always-on. Matching is deliberately loose about punctuation and
    case: speech-to-text renders the same phrase as "Hey, Hank!", "hey hank"
    and "Hi Hank." in consecutive utterances.
    """
    if not wake:
        return text.strip()
    norm = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text)
    norm = " ".join(norm.split())
    if not norm:
        return None

    # Longest phrases first, so "hey hank" wins over a bare "hank" and the
    # remainder is not left with a stray word.
    phrases = sorted((" ".join(w.lower().split()) for w in wake.split(",") if w.strip()),
                     key=len, reverse=True)
    best: tuple[int, int] | None = None
    for key in phrases:
        idx = norm.find(key)
        if idx < 0:
            continue
        # Must fall on word boundaries: "hank" should not match inside "thank".
        before_ok = idx == 0 or norm[idx - 1] == " "
        after = idx + len(key)
        after_ok = after == len(norm) or norm[after] == " "
        if not (before_ok and after_ok):
            continue
        if best is None or idx < best[0]:
            best = (idx, after)
    if best is None:
        # No exact phrase matched. Fall back to finding the name itself,
        # allowing for mis-transcription - this is the difference between a
        # robot that answers across a room and one that appears deaf.
        names = {p.split()[-1] for p in phrases if p.split()}
        words = norm.split()
        for i, w in enumerate(words):
            if any(_close_enough(w, n) for n in names):
                best = (sum(len(x) + 1 for x in words[:i]),
                        sum(len(x) + 1 for x in words[:i + 1]) - 1)
                break
    if best is None:
        return None
    # Keep what was said on BOTH sides of the name, not just after it. People
    # put it at either end - "Hank, what do you see" and "what do you see,
    # Hank" are the same request, and dropping the leading half turned the
    # second into a bare wake word with the question thrown away.
    command = " ".join((norm[:best[0]] + " " + norm[best[1]:]).split())
    # Keeping the leading half means keeping its filler too.
    while True:
        head, _, rest = command.partition(" ")
        if head in _FILLERS and rest:
            command = rest
            continue
        return command
