from __future__ import annotations

import argparse
import os
import select
import signal
import subprocess
import sys
import threading
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import DictationError
from .telemetry import TELEMETRY_OPT_OUTS
from .user_config import DictationSettings

SAMPLE_RATE = 16_000
FRAME_MS = 80
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2

SILERO_REPOSITORY = "istupakov/silero-vad-onnx"
SILERO_REVISION = "b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6"
SILERO_FILENAME = "silero_vad.onnx"
# At 16 kHz the model takes 512 new samples per call, preceded by the last 64
# samples of the previous call.
SILERO_CHUNK_SAMPLES = 512
SILERO_CONTEXT_SAMPLES = 64
START_PROBABILITY = 0.5
CONTINUE_PROBABILITY = 0.35


@dataclass(frozen=True)
class VadCaptureSettings:
    end_silence_ms: float
    max_seconds: float
    minimum_rms: float
    start_speech_frames: int

    @classmethod
    def from_dictation(cls, settings: DictationSettings) -> VadCaptureSettings:
        return cls(
            end_silence_ms=settings.vad_end_silence_ms,
            max_seconds=settings.vad_max_seconds,
            minimum_rms=settings.vad_min_speech_rms,
            start_speech_frames=settings.vad_start_speech_frames,
        )


def load_silero_session() -> Any:
    """Load the pinned Silero VAD model on one CPU thread, downloading it once."""

    os.environ.update(TELEMETRY_OPT_OUTS)
    try:
        import onnxruntime
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise DictationError(
            "VAD needs ONNX Runtime and huggingface_hub; install local-dictation "
            "with the cpu, cuda, or whisper extra"
        ) from error
    try:
        model = hf_hub_download(
            SILERO_REPOSITORY, SILERO_FILENAME, revision=SILERO_REVISION
        )
    except Exception as error:
        raise DictationError(
            f"couldn't download the Silero VAD model: {error}"
        ) from error
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    try:
        return onnxruntime.InferenceSession(
            model, sess_options=options, providers=["CPUExecutionProvider"]
        )
    except Exception as error:
        raise DictationError(
            f"couldn't load the Silero VAD model {model}: {error}"
        ) from error


class SpeechDetector:
    """Silero voice activity detector for 16-bit mono PCM frames.

    Frames quieter than ``minimum_rms`` never count as speech. Once a frame is
    speech, later frames stay speech until the model's confidence drops below
    ``CONTINUE_PROBABILITY``.
    """

    def __init__(self, *, minimum_rms: float, session: Any | None = None) -> None:
        import numpy as np

        self._np: Any = np
        self._session: Any = load_silero_session() if session is None else session
        self._sample_rate = np.array(SAMPLE_RATE, dtype=np.int64)
        self.minimum_rms = minimum_rms
        self.reset()

    def reset(self) -> None:
        """Forget earlier audio so the next frame starts a new stream."""

        np = self._np
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(SILERO_CONTEXT_SAMPLES, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._speaking = False

    def is_speech(self, frame: bytes) -> bool:
        if len(frame) != FRAME_BYTES:
            return False
        np = self._np
        samples = np.frombuffer(frame, dtype="<i2")
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        audio = np.concatenate((self._pending, samples.astype(np.float32) / 32768))
        probability = 0.0
        while len(audio) >= SILERO_CHUNK_SAMPLES:
            chunk = audio[:SILERO_CHUNK_SAMPLES]
            audio = audio[SILERO_CHUNK_SAMPLES:]
            probability = max(probability, self._probability(chunk))
        self._pending = audio
        threshold = CONTINUE_PROBABILITY if self._speaking else START_PROBABILITY
        self._speaking = probability >= threshold
        return self._speaking and rms >= self.minimum_rms

    def _probability(self, chunk: Any) -> float:
        np = self._np
        window = np.concatenate((self._context, chunk))[np.newaxis, :]
        output, self._state = self._session.run(
            ["output", "stateN"],
            {"input": window, "state": self._state, "sr": self._sample_rate},
        )
        self._context = chunk[-SILERO_CONTEXT_SAMPLES:]
        return float(output[0, 0])


def _read_frame(
    process: subprocess.Popen[bytes],
    stop_requested: threading.Event,
    *,
    poll_timeout: float = 0.5,
) -> bytes:
    if process.stdout is None:
        raise DictationError("microphone stream is unavailable")
    data = bytearray()
    while len(data) < FRAME_BYTES:
        if stop_requested.is_set():
            raise DictationError("VAD dictation was cancelled")
        stdout = process.stdout
        pollable = False
        try:
            pollable = select.select([stdout], [], [], poll_timeout) == (
                [stdout],
                [],
                [],
            )
        except (OSError, ValueError):
            pollable = True
        if not pollable:
            continue
        chunk = stdout.read(FRAME_BYTES - len(data))
        if not chunk:
            detail = (
                process.stderr.read().decode(errors="replace").strip()
                if process.stderr is not None
                else ""
            )
            raise DictationError(
                f"microphone stream ended: {detail or process.returncode}"
            )
        data.extend(chunk)
    return bytes(data)


def _stop_microphone(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def capture_vad_audio(
    path: Path,
    *,
    source: str,
    settings: VadCaptureSettings,
    detector: SpeechDetector,
    stop_requested: threading.Event,
) -> str:
    """Wait for speech and capture a WAV until silence or the duration limit."""

    detector.reset()
    command = ["pw-record", "--raw"]
    if source:
        command.extend(("--target", source))
    command.extend(("--channels=1", "--rate=16000", "--format=s16", "-"))
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    has_speech = False
    candidate_frames: list[bytes] = []
    silence_ms = 0.0
    captured_frame_count = 0
    outcome = "maximum duration"
    try:
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            while True:
                if stop_requested.is_set():
                    raise DictationError("VAD dictation was cancelled")
                frame = _read_frame(process, stop_requested)
                speech_detected = detector.is_speech(frame)
                if not has_speech:
                    if speech_detected:
                        candidate_frames.append(frame)
                    else:
                        candidate_frames.clear()
                    if len(candidate_frames) < settings.start_speech_frames:
                        continue
                    has_speech = True
                    for candidate in candidate_frames:
                        output.writeframesraw(candidate)
                    captured_frame_count += len(candidate_frames)
                    candidate_frames.clear()
                    silence_ms = 0
                elif speech_detected:
                    output.writeframesraw(frame)
                    captured_frame_count += 1
                    silence_ms = 0
                else:
                    output.writeframesraw(frame)
                    captured_frame_count += 1
                    silence_ms += FRAME_MS

                duration = captured_frame_count * FRAME_MS / 1000
                if has_speech and silence_ms >= settings.end_silence_ms:
                    outcome = "silence"
                    break
                if has_speech and duration >= settings.max_seconds:
                    break
    finally:
        _stop_microphone(process)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    return outcome


def check_noise_rejection() -> int:
    """Return how many frames of loud white noise the model heard as silence."""

    import numpy as np

    detector = SpeechDetector(minimum_rms=0)
    noise = np.random.default_rng(0).normal(0, 3000, FRAME_SAMPLES * 40)
    pcm = noise.astype("<i2").tobytes()
    frames = [
        pcm[offset : offset + FRAME_BYTES] for offset in range(0, len(pcm), FRAME_BYTES)
    ]
    heard = sum(detector.is_speech(frame) for frame in frames)
    if heard:
        raise DictationError(
            f"Silero VAD heard speech in {heard} of {len(frames)} frames of white noise"
        )
    return len(frames)


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m local_dictation.vad",
        description="Check that the Silero VAD model loads and ignores white noise",
    )
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(arguments)
    try:
        frames = check_noise_rejection()
    except DictationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Silero VAD heard no speech in {frames} frames of white noise")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
