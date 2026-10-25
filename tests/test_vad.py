from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import threading
import time
import unittest
import wave
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from local_dictation import vad
from local_dictation.errors import DictationError
from local_dictation.user_config import default_user_config


def _settings(
    *,
    silence_ms: float = 160,
    max_seconds: float = 2,
    start_speech_frames: int = 3,
) -> vad.VadCaptureSettings:
    return vad.VadCaptureSettings(
        end_silence_ms=silence_ms,
        max_seconds=max_seconds,
        minimum_rms=0,
        start_speech_frames=start_speech_frames,
    )


def _process(frames: list[bytes]) -> mock.Mock:
    process = mock.Mock()
    process.stdout = io.BytesIO(b"".join(frames))
    process.stderr = io.BytesIO()
    process.poll.return_value = None
    process.wait.return_value = 0
    return process


class VadSettingsTests(unittest.TestCase):
    def test_defaults_are_suitable_for_dictation(self) -> None:
        settings = vad.VadCaptureSettings.from_dictation(
            default_user_config().dictation
        )

        self.assertEqual(settings.end_silence_ms, 900)
        self.assertEqual(settings.max_seconds, 120)
        self.assertEqual(settings.minimum_rms, 1100)
        self.assertEqual(settings.start_speech_frames, 3)


class _FakeSession:
    """Stands in for the Silero model.

    Without scripted probabilities it hears speech wherever a chunk has a
    positive offset, which the speech frames below have and noise doesn't.
    """

    def __init__(self, probabilities: list[float] | None = None) -> None:
        self.probabilities = probabilities
        self.inputs: list[dict[str, Any]] = []

    def run(self, outputs: list[str], inputs: dict[str, Any]) -> list[Any]:
        assert outputs == ["output", "stateN"]
        self.inputs.append(inputs)
        if self.probabilities is not None:
            probability = self.probabilities.pop(0)
        else:
            offset = float(np.mean(inputs["input"][0, vad.SILERO_CONTEXT_SAMPLES :]))
            probability = 0.9 if offset > 0.1 else 0.02
        return [np.array([[probability]], dtype=np.float32), inputs["state"] + 1]


def _frame(level: int, *, alternating: bool = False) -> bytes:
    samples = np.full(vad.FRAME_SAMPLES, level, dtype="<i2")
    if alternating:
        samples[1::2] = -level
    return samples.tobytes()


SPEECH = _frame(8000)
NOISE = _frame(1600, alternating=True)
QUIET = _frame(200, alternating=True)


class SpeechDetectorTests(unittest.TestCase):
    def test_model_reads_512_samples_after_64_samples_of_context(self) -> None:
        session = _FakeSession([0.0] * 5)
        detector = vad.SpeechDetector(minimum_rms=0, session=session)
        pcm = np.arange(2 * vad.FRAME_SAMPLES, dtype="<i2")

        detector.is_speech(pcm[: vad.FRAME_SAMPLES].tobytes())
        detector.is_speech(pcm[vad.FRAME_SAMPLES :].tobytes())

        audio = pcm.astype(np.float32) / 32768
        self.assertEqual(len(session.inputs), 5)
        for call, inputs in enumerate(session.inputs):
            with self.subTest(call=call):
                start = call * 512
                context = audio[start - 64 : start] if call else np.zeros(64)
                np.testing.assert_array_equal(
                    inputs["input"],
                    np.concatenate((context, audio[start : start + 512]))[None, :],
                )
                np.testing.assert_array_equal(
                    inputs["state"], np.full((2, 1, 128), call, dtype=np.float32)
                )
                self.assertEqual(inputs["sr"].dtype, np.int64)
                self.assertEqual(inputs["sr"].shape, ())
                self.assertEqual(int(inputs["sr"]), vad.SAMPLE_RATE)

    def test_speech_starts_at_half_and_continues_down_to_035(self) -> None:
        # Frames alternate between two and three complete 512-sample chunks.
        session = _FakeSession(
            [0.1, 0.45]
            + [0.2, 0.55, 0.1]
            + [0.36, 0.2]
            + [0.3, 0.34, 0.1]
            + [0.45, 0.1]
        )
        detector = vad.SpeechDetector(minimum_rms=0, session=session)

        decisions = [detector.is_speech(SPEECH) for _ in range(5)]

        self.assertEqual(decisions, [False, True, True, False, False])

    def test_quiet_frames_are_not_speech_but_keep_the_model_in_step(self) -> None:
        session = _FakeSession([0.9] * 5)
        detector = vad.SpeechDetector(minimum_rms=1100, session=session)

        self.assertFalse(detector.is_speech(QUIET))
        self.assertEqual(len(session.inputs), 2)
        self.assertTrue(detector.is_speech(SPEECH))
        self.assertEqual(len(session.inputs), 5)

    def test_reset_starts_a_new_stream(self) -> None:
        session = _FakeSession([0.9, 0.9, 0.4, 0.4])
        detector = vad.SpeechDetector(minimum_rms=0, session=session)
        self.assertTrue(detector.is_speech(SPEECH))

        detector.reset()

        self.assertFalse(detector.is_speech(SPEECH))
        self.assertEqual(len(session.inputs), 4)
        first = session.inputs[2]
        np.testing.assert_array_equal(first["input"][0, :64], np.zeros(64))
        np.testing.assert_array_equal(
            first["input"][0, 64:], np.full(512, 8000 / 32768)
        )
        np.testing.assert_array_equal(first["state"], np.zeros((2, 1, 128)))

    def test_incomplete_frame_is_not_speech(self) -> None:
        session = _FakeSession([0.9] * 4)
        detector = vad.SpeechDetector(minimum_rms=0, session=session)

        self.assertFalse(detector.is_speech(b"\0"))
        self.assertEqual(session.inputs, [])


class SileroModelTests(unittest.TestCase):
    def _modules(
        self, *, download: Any = None, session: Any = None
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        hub = SimpleNamespace(
            hf_hub_download=download or mock.Mock(return_value="/models/silero.onnx")
        )
        runtime = SimpleNamespace(
            SessionOptions=SimpleNamespace,
            InferenceSession=session or mock.Mock(return_value="session"),
        )
        return hub, runtime

    @contextlib.contextmanager
    def _installed(self, hub: object, runtime: object) -> Iterator[None]:
        with mock.patch.dict(
            sys.modules, {"huggingface_hub": hub, "onnxruntime": runtime}
        ):
            yield

    def test_pinned_model_runs_on_one_cpu_thread_with_telemetry_off(self) -> None:
        hub, runtime = self._modules()
        with (
            self._installed(hub, runtime),
            mock.patch.dict(os.environ, {"ORT_DISABLE_TELEMETRY": "0"}),
        ):
            session = vad.load_silero_session()
            self.assertEqual(os.environ["ORT_DISABLE_TELEMETRY"], "1")

        self.assertEqual(session, "session")
        hub.hf_hub_download.assert_called_once_with(
            "istupakov/silero-vad-onnx",
            "silero_vad.onnx",
            revision="b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6",
        )
        call = runtime.InferenceSession.call_args
        self.assertEqual(call.args, ("/models/silero.onnx",))
        self.assertEqual(call.kwargs["providers"], ["CPUExecutionProvider"])
        self.assertEqual(call.kwargs["sess_options"].intra_op_num_threads, 1)
        self.assertEqual(call.kwargs["sess_options"].inter_op_num_threads, 1)

    def test_missing_engine_names_the_extras_that_provide_it(self) -> None:
        with (
            mock.patch.dict(sys.modules, {"onnxruntime": None}),
            mock.patch.dict(os.environ),
            self.assertRaisesRegex(DictationError, "cpu, cuda, or whisper extra"),
        ):
            vad.load_silero_session()

    def test_download_failure_is_a_dictation_error(self) -> None:
        hub, runtime = self._modules(download=mock.Mock(side_effect=OSError("offline")))
        with (
            self._installed(hub, runtime),
            mock.patch.dict(os.environ),
            self.assertRaisesRegex(DictationError, "couldn't download .*: offline"),
        ):
            vad.load_silero_session()
        runtime.InferenceSession.assert_not_called()

    def test_unloadable_model_is_a_dictation_error(self) -> None:
        hub, runtime = self._modules(session=mock.Mock(side_effect=RuntimeError("bad")))
        with (
            self._installed(hub, runtime),
            mock.patch.dict(os.environ),
            self.assertRaisesRegex(DictationError, "couldn't load .*: bad"),
        ):
            vad.load_silero_session()


class NoiseCheckTests(unittest.TestCase):
    def test_check_passes_when_white_noise_is_silence(self) -> None:
        output = io.StringIO()
        with (
            mock.patch.object(vad, "load_silero_session", return_value=_FakeSession()),
            contextlib.redirect_stdout(output),
        ):
            result = vad.main(["--check"])

        self.assertEqual(result, 0)
        self.assertIn("no speech in 40 frames", output.getvalue())

    def test_check_fails_when_white_noise_sounds_like_speech(self) -> None:
        errors = io.StringIO()
        with (
            mock.patch.object(
                vad, "load_silero_session", return_value=_FakeSession([0.9] * 100)
            ),
            contextlib.redirect_stderr(errors),
        ):
            result = vad.main(["--check"])

        self.assertEqual(result, 1)
        self.assertIn("heard speech in 40 of 40 frames", errors.getvalue())


class VadCaptureTests(unittest.TestCase):
    def _capture(
        self,
        frames: list[bytes],
        speech: list[bool] | None = None,
        *,
        settings: vad.VadCaptureSettings | None = None,
        detector: Any = None,
    ) -> tuple[str, bytes, mock.Mock]:
        process = _process(frames)
        if detector is None:
            detector = mock.Mock()
            detector.is_speech.side_effect = speech
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.wav"
            with mock.patch.object(vad.subprocess, "Popen", return_value=process):
                outcome = vad.capture_vad_audio(
                    path,
                    source="microphone",
                    settings=settings or _settings(),
                    detector=detector,
                    stop_requested=threading.Event(),
                )
            audio = path.read_bytes()
        return outcome, audio, process

    def test_sustained_silence_finishes_valid_wav(self) -> None:
        frame = b"\x01\x00" * vad.FRAME_SAMPLES
        outcome, audio, process = self._capture(
            [frame] * 5, [True, True, True, False, False]
        )

        self.assertEqual(outcome, "silence")
        process.send_signal.assert_called_once()
        with wave.open(io.BytesIO(audio), "rb") as recorded:
            self.assertEqual(recorded.getnframes(), vad.FRAME_SAMPLES * 5)
            self.assertEqual(recorded.getframerate(), vad.SAMPLE_RATE)

    def test_each_capture_starts_a_new_detector_stream(self) -> None:
        detector = mock.Mock()
        detector.is_speech.side_effect = [True, True, True, False, False]

        self._capture(
            [b"\x01\x00" * vad.FRAME_SAMPLES] * 5,
            settings=_settings(),
            detector=detector,
        )

        detector.reset.assert_called_once_with()

    def test_noise_above_the_loudness_floor_still_ends_the_utterance(self) -> None:
        settings = vad.VadCaptureSettings.from_dictation(
            default_user_config().dictation
        )
        detector = vad.SpeechDetector(
            minimum_rms=settings.minimum_rms, session=_FakeSession()
        )
        trailing = int(settings.end_silence_ms // vad.FRAME_MS) + 1

        outcome, audio, _process = self._capture(
            [NOISE] * 5 + [SPEECH] * 5 + [NOISE] * (trailing + 5),
            settings=settings,
            detector=detector,
        )

        self.assertEqual(outcome, "silence")
        with wave.open(io.BytesIO(audio), "rb") as recorded:
            self.assertEqual(recorded.getnframes(), vad.FRAME_SAMPLES * (5 + trailing))

    def test_quiet_background_ends_the_utterance(self) -> None:
        settings = vad.VadCaptureSettings.from_dictation(
            default_user_config().dictation
        )
        detector = vad.SpeechDetector(
            minimum_rms=settings.minimum_rms, session=_FakeSession()
        )
        trailing = int(settings.end_silence_ms // vad.FRAME_MS) + 1

        outcome, audio, _process = self._capture(
            [QUIET] * 5 + [SPEECH] * 5 + [QUIET] * (trailing + 5),
            settings=settings,
            detector=detector,
        )

        self.assertEqual(outcome, "silence")
        with wave.open(io.BytesIO(audio), "rb") as recorded:
            self.assertEqual(recorded.getnframes(), vad.FRAME_SAMPLES * (5 + trailing))

    def test_short_pause_does_not_finish(self) -> None:
        frame = b"\x01\x00" * vad.FRAME_SAMPLES
        outcome, _audio, _process = self._capture(
            [frame] * 8,
            [True, True, True, False, True, True, False, False],
        )

        self.assertEqual(outcome, "silence")

    def test_pre_speech_silence_is_not_written_to_wav(self) -> None:
        frame = b"\x01\x00" * vad.FRAME_SAMPLES
        outcome, audio, _process = self._capture(
            [frame] * 7,
            [False, False, True, True, True, False, False],
        )

        self.assertEqual(outcome, "silence")
        with wave.open(io.BytesIO(audio), "rb") as recorded:
            self.assertEqual(recorded.getnframes(), vad.FRAME_SAMPLES * 5)

    def test_isolated_speech_frame_does_not_activate_capture(self) -> None:
        frame = b"\x01\x00" * vad.FRAME_SAMPLES
        outcome, audio, _process = self._capture(
            [frame] * 8,
            [True, False, False, True, True, True, False, False],
        )

        self.assertEqual(outcome, "silence")
        with wave.open(io.BytesIO(audio), "rb") as recorded:
            self.assertEqual(recorded.getnframes(), vad.FRAME_SAMPLES * 5)

    def test_stop_while_waiting_for_speech_cleans_up_microphone(self) -> None:
        frame = b"\0" * vad.FRAME_BYTES
        process = _process([frame])
        stop_requested = threading.Event()
        stop_requested.set()
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(vad.subprocess, "Popen", return_value=process),
            self.assertRaisesRegex(DictationError, "cancelled"),
        ):
            vad.capture_vad_audio(
                Path(temporary) / "capture.wav",
                source="",
                settings=_settings(),
                detector=mock.Mock(is_speech=mock.Mock(return_value=False)),
                stop_requested=stop_requested,
            )

        process.send_signal.assert_called_once()

    def test_stop_during_stalled_pipe_read_cleans_up_microphone(self) -> None:
        read_fd, write_fd = os.pipe()
        process = mock.Mock()
        process.stdout = os.fdopen(read_fd, "rb", buffering=0)
        process.stderr = io.BytesIO()
        process.poll.return_value = None
        stop_requested = threading.Event()

        def request_stop_after_delay() -> None:
            time.sleep(0.05)
            stop_requested.set()

        try:
            with (
                tempfile.TemporaryDirectory() as temporary,
                mock.patch.object(vad.subprocess, "Popen", return_value=process),
                self.assertRaisesRegex(DictationError, "cancelled"),
            ):
                threading.Thread(target=request_stop_after_delay, daemon=True).start()
                vad.capture_vad_audio(
                    Path(temporary) / "capture.wav",
                    source="",
                    settings=_settings(),
                    detector=mock.Mock(is_speech=mock.Mock(return_value=False)),
                    stop_requested=stop_requested,
                )
        finally:
            os.close(write_fd)

        process.send_signal.assert_called_once()

    def test_maximum_duration_finishes_active_speech(self) -> None:
        frame = b"\x01\x00" * vad.FRAME_SAMPLES
        outcome, _audio, _process = self._capture(
            [frame] * 3,
            [True, True, True],
            settings=_settings(max_seconds=0.24),
        )

        self.assertEqual(outcome, "maximum duration")


if __name__ == "__main__":
    unittest.main()
