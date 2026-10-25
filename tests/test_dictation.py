from __future__ import annotations

import io
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

from local_dictation import dictation
from local_dictation.desktop import Window
from local_dictation.errors import DictationError, NoSpeechError
from local_dictation.user_config import default_user_config


class DictationTests(unittest.TestCase):
    def test_begin_starts_dictation_recording(self) -> None:
        with mock.patch.object(dictation, "start_recording") as start:
            dictation.run("begin")
        start.assert_called_once_with()

    def test_start_records_from_configured_source(self) -> None:
        settings = replace(default_user_config().dictation, source="usb-microphone")
        with (
            mock.patch.object(dictation, "_ensure_dictation_allowed"),
            mock.patch.object(dictation.recorder, "start_recording") as start,
            mock.patch.object(dictation, "notify"),
            redirect_stdout(io.StringIO()),
        ):
            dictation.start_recording(settings)

        start.assert_called_once_with(
            dictation.PATHS, source="usb-microphone", ready=mock.ANY
        )

    def test_retry_transcribes_pending_generation_with_configured_injection(
        self,
    ) -> None:
        generation = Path(
            "/runtime/local-dictation/recordings/"
            "recording-0123456789abcdef0123456789abcdef.wav"
        )
        user_config = default_user_config()
        with (
            mock.patch.object(
                dictation.recorder, "retry_generation", return_value=generation
            ) as retry,
            mock.patch.object(dictation, "_ensure_dictation_allowed") as allowed,
            mock.patch.object(dictation, "transcribe_and_type") as transcribe,
        ):
            dictation.retry(generation, user_config)

        retry.assert_called_once_with(
            dictation.PATHS,
            generation,
            active_message="cannot retry while dictation is recording",
        )
        allowed.assert_called_once_with()
        transcribe.assert_called_once_with(generation, user_config.dictation)

    def test_retry_is_rejected_before_transcription_while_recording(self) -> None:
        with (
            mock.patch.object(
                dictation.recorder,
                "retry_generation",
                side_effect=DictationError("cannot retry while dictation is recording"),
            ),
            mock.patch.object(dictation, "transcribe_and_type") as transcribe,
            self.assertRaisesRegex(DictationError, "cannot retry"),
        ):
            dictation.retry(Path("/runtime/recording.wav"))

        transcribe.assert_not_called()

    def test_toggle_starts_when_idle(self) -> None:
        with (
            mock.patch.object(dictation, "recording_active", return_value=False),
            mock.patch.object(dictation, "start_recording") as start,
        ):
            dictation.run("toggle")
        start.assert_called_once_with()

    def test_toggle_stops_transcribes_and_types_when_recording(self) -> None:
        generation = Path("/runtime/dictation/recordings/recording-generation.wav")
        with (
            mock.patch.object(dictation, "recording_active", return_value=True),
            mock.patch.object(
                dictation, "stop_recording", return_value=generation
            ) as stop,
            mock.patch.object(dictation, "_ensure_dictation_allowed"),
            mock.patch.object(dictation, "transcribe_and_type") as transcribe,
        ):
            dictation.run("toggle")
        stop.assert_called_once_with()
        transcribe.assert_called_once_with(generation)

    def test_vad_second_press_stops_always_on_listener(self) -> None:
        with (
            mock.patch.object(
                dictation.recorder, "request_vad_stop", return_value=True
            ) as stop,
            mock.patch.object(dictation, "_ensure_dictation_allowed") as allowed,
            mock.patch.object(dictation, "notify"),
        ):
            dictation.run("vad")

        stop.assert_called_once_with(dictation.PATHS)
        allowed.assert_not_called()

    def test_vad_listener_rearms_after_transcribing(self) -> None:
        generation = Path("/runtime/dictation/recordings/recording-generation.wav")
        settings = mock.Mock(minimum_rms=1100)

        def capture_once(*_args: object, **kwargs: object) -> None:
            stop_requested = kwargs["stop_requested"]
            assert isinstance(stop_requested, threading.Event)
            stop_requested.set()

        with (
            mock.patch.object(
                dictation.recorder, "request_vad_stop", return_value=False
            ),
            mock.patch.object(dictation, "_ensure_dictation_allowed") as allowed,
            mock.patch.object(dictation, "socket_ready", return_value=True),
            mock.patch.object(
                dictation.VadCaptureSettings,
                "from_dictation",
                return_value=settings,
            ),
            mock.patch.object(dictation, "SpeechDetector") as detector,
            mock.patch.object(dictation.signal, "getsignal", return_value=None),
            mock.patch.object(dictation.signal, "signal"),
            mock.patch.object(dictation.recorder, "claim_current_process") as claim,
            mock.patch.object(
                dictation, "capture_vad_audio", side_effect=capture_once
            ) as capture,
            mock.patch.object(
                dictation.recorder,
                "complete_current_recording",
                return_value=generation,
            ) as complete,
            mock.patch.object(
                dictation.recorder, "abandon_current_recording"
            ) as abandon,
            mock.patch.object(dictation, "transcribe_and_type") as transcribe,
            mock.patch.object(dictation, "notify"),
        ):
            dictation.run("vad")

        claim.assert_called_once_with(dictation.PATHS, mode="vad")
        self.assertEqual(capture.call_args.kwargs["detector"], detector.return_value)
        complete.assert_called_once_with(
            dictation.PATHS,
            mode="vad",
            retain_owner=True,
        )
        transcribe.assert_called_once_with(generation)
        abandon.assert_called_once_with(dictation.PATHS, mode="vad")
        self.assertEqual(allowed.call_count, 2)

    def test_vad_listener_rearms_when_stt_finds_no_speech(self) -> None:
        generation = Path("/runtime/dictation/recordings/recording-generation.wav")

        def capture_false_activation(*_args: object, **kwargs: object) -> None:
            stop_requested = kwargs["stop_requested"]
            assert isinstance(stop_requested, threading.Event)
            stop_requested.set()

        with (
            mock.patch.object(
                dictation.recorder, "request_vad_stop", return_value=False
            ),
            mock.patch.object(dictation, "_ensure_dictation_allowed"),
            mock.patch.object(dictation, "socket_ready", return_value=True),
            mock.patch.object(
                dictation.VadCaptureSettings,
                "from_dictation",
                return_value=mock.Mock(minimum_rms=1100),
            ),
            mock.patch.object(dictation, "SpeechDetector"),
            mock.patch.object(dictation.signal, "getsignal", return_value=None),
            mock.patch.object(dictation.signal, "signal"),
            mock.patch.object(dictation.recorder, "claim_current_process"),
            mock.patch.object(
                dictation,
                "capture_vad_audio",
                side_effect=capture_false_activation,
            ),
            mock.patch.object(
                dictation.recorder,
                "complete_current_recording",
                return_value=generation,
            ),
            mock.patch.object(dictation.recorder, "abandon_current_recording"),
            mock.patch.object(
                dictation,
                "transcribe_and_type",
                side_effect=NoSpeechError("no speech"),
            ) as transcribe,
            mock.patch.object(dictation, "notify"),
        ):
            dictation.run("vad")

        transcribe.assert_called_once_with(generation)

    def test_stopped_vad_listener_abandons_owned_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = dictation.recorder.RecorderPaths(
                root,
                root / "recording.wav",
                root / "recording.pid",
                root / "recording.log",
            )

            def cancel_capture(*_args: object, **kwargs: object) -> None:
                paths.audio.write_bytes(b"partial")
                stop_requested = kwargs["stop_requested"]
                assert isinstance(stop_requested, threading.Event)
                stop_requested.set()
                raise DictationError("cancelled")

            with (
                mock.patch.object(dictation, "PATHS", paths),
                mock.patch.object(
                    dictation.recorder, "request_vad_stop", return_value=False
                ),
                mock.patch.object(dictation, "_ensure_dictation_allowed"),
                mock.patch.object(dictation, "socket_ready", return_value=True),
                mock.patch.object(
                    dictation.VadCaptureSettings,
                    "from_dictation",
                    return_value=mock.Mock(minimum_rms=1100),
                ),
                mock.patch.object(dictation, "SpeechDetector"),
                mock.patch.object(dictation.signal, "getsignal", return_value=None),
                mock.patch.object(dictation.signal, "signal"),
                mock.patch.object(dictation.recorder, "claim_current_process"),
                mock.patch.object(
                    dictation, "capture_vad_audio", side_effect=cancel_capture
                ),
                mock.patch.object(
                    dictation.recorder,
                    "abandon_current_recording",
                    side_effect=lambda *_args, **_kwargs: paths.audio.unlink(
                        missing_ok=True
                    ),
                ) as abandon,
                mock.patch.object(dictation, "notify"),
            ):
                dictation.run("vad")

            abandon.assert_called_once_with(paths, mode="vad")
            self.assertFalse(paths.audio.exists())

    def test_transcribes_typed_dictation_audio(self) -> None:
        generation = Path("/runtime/dictation/recordings/recording-generation.wav")
        with (
            mock.patch.object(
                dictation, "transcribe", return_value="hello"
            ) as transcribe,
            mock.patch.object(dictation, "inject") as inject,
        ):
            dictation.transcribe_and_type(generation)
        transcribe.assert_called_once_with(generation)
        inject.assert_called_once_with("hello")

    def test_direct_transcribe_hands_off_before_stt(self) -> None:
        generation = Path("/runtime/dictation/recordings/recording-generation.wav")
        with (
            mock.patch.object(dictation, "_ensure_dictation_allowed"),
            mock.patch.object(
                dictation.recorder,
                "handoff_recording",
                return_value=generation,
            ) as handoff,
            mock.patch.object(dictation, "transcribe_and_type") as transcribe,
        ):
            dictation.run("transcribe")

        handoff.assert_called_once_with(
            dictation.PATHS,
            active_message="cannot transcribe while dictation is recording",
        )
        transcribe.assert_called_once_with(generation)

    def test_runelite_dictation_is_rejected(self) -> None:
        desktop = mock.Mock()
        desktop.active_window.return_value = Window(
            "1", "net-runelite-client-runelite", 10
        )
        with (
            mock.patch.object(dictation, "get_desktop", return_value=desktop),
            self.assertRaisesRegex(DictationError, "disabled for RuneLite"),
        ):
            dictation.inject("buying fish")

    def test_runelite_recording_is_rejected_before_capture(self) -> None:
        with (
            mock.patch.object(
                dictation,
                "_active_window",
                return_value=Window("1", "net-runelite-client-runelite", 10),
            ),
            mock.patch.object(dictation, "socket_ready") as socket_ready,
            self.assertRaisesRegex(DictationError, "disabled for RuneLite"),
        ):
            dictation.start_recording()
        socket_ready.assert_not_called()

    def test_stdout_injection(self) -> None:
        output = io.StringIO()
        settings = replace(default_user_config().dictation, inject="stdout")
        with (
            mock.patch.object(dictation, "_ensure_dictation_allowed"),
            redirect_stdout(output),
        ):
            dictation.inject("hello world", settings)
        self.assertEqual(output.getvalue(), "hello world\n")

    def test_invalid_injection_mode_is_rejected(self) -> None:
        settings = replace(default_user_config().dictation, inject="invalid")
        with (
            mock.patch.object(dictation, "_ensure_dictation_allowed"),
            self.assertRaises(DictationError),
        ):
            dictation.inject("hello", settings)

    def test_copy_uses_desktop_clipboard(self) -> None:
        desktop = mock.Mock()
        desktop.active_window.return_value = Window("42", "firefox", 10)
        desktop.write_clipboard.return_value = True
        window = Window("42", "firefox", 10)
        dictation._copy_to_clipboard(desktop, window, "hello")
        desktop.write_clipboard.assert_called_once_with("hello")

    def test_clipboard_failure_is_reported(self) -> None:
        desktop = mock.Mock()
        desktop.active_window.return_value = Window("42", "firefox", 10)
        desktop.write_clipboard.return_value = False
        window = Window("42", "firefox", 10)
        with self.assertRaisesRegex(DictationError, "could not copy"):
            dictation._copy_to_clipboard(desktop, window, "hello")

    def test_auto_injection_pastes_after_copying_to_clipboard(self) -> None:
        operations: list[tuple[str, object]] = []
        window = Window("42", "firefox", 10)
        desktop = mock.Mock()
        desktop.active_window.return_value = window
        desktop.has_clipboard.return_value = True
        desktop.read_clipboard.side_effect = [
            (True, "previous"),
            (True, "hello"),
        ]
        desktop.write_clipboard.return_value = True
        with (
            mock.patch.dict("os.environ", {"DICTATION_INJECT": "auto"}),
            mock.patch.object(dictation, "_send_to_herdr", return_value=False),
            mock.patch.object(dictation, "get_desktop", return_value=desktop),
            mock.patch.object(
                dictation,
                "_copy_to_clipboard",
                side_effect=lambda _desktop, _window, text: operations.append(
                    ("copy", text)
                ),
            ),
            mock.patch.object(
                dictation,
                "_send_key",
                side_effect=lambda _desktop, _window, key: operations.append(
                    ("key", key)
                ),
            ),
            mock.patch.object(dictation.time, "sleep"),
        ):
            dictation.inject("hello")
        self.assertEqual(
            operations,
            [
                ("copy", "hello"),
                ("key", "ctrl+v"),
            ],
        )
        desktop.write_clipboard.assert_called_with("previous")

    def test_wayland_terminal_uses_ctrl_shift_v(self) -> None:
        window = Window("42", "foot", 10)
        desktop = mock.Mock()
        desktop.active_window.return_value = window
        desktop.has_clipboard.return_value = True
        desktop.read_clipboard.return_value = (False, "")
        desktop.write_clipboard.return_value = True
        settings = replace(default_user_config().dictation, inject="paste")
        with (
            mock.patch.object(dictation, "_send_to_herdr", return_value=False),
            mock.patch.object(dictation, "get_desktop", return_value=desktop),
            mock.patch.object(dictation, "_copy_to_clipboard"),
            mock.patch.object(dictation, "_send_key") as send_key,
            mock.patch.object(dictation.time, "sleep"),
        ):
            dictation.inject("hello", settings)
        send_key.assert_called_once_with(desktop, window, "ctrl+shift+v")

    def test_unsupported_wayland_session_is_reported(self) -> None:
        with (
            mock.patch.dict("os.environ", {"DICTATION_INJECT": "paste"}),
            mock.patch.object(dictation, "_send_to_herdr", return_value=False),
            mock.patch.object(dictation, "get_desktop", return_value=None),
            self.assertRaisesRegex(DictationError, "Hyprland, and Sway"),
        ):
            dictation.inject("hello")

    def test_gnome_wayland_reports_capability_degradation(self) -> None:
        from local_dictation.desktop import DegradedWaylandDesktop

        backend = DegradedWaylandDesktop(
            "gnome",
            "GNOME Wayland does not expose a supported focused-window or "
            "keyboard-injection API; dictation insertion degrades to stdout",
        )
        with (
            mock.patch.dict("os.environ", {"DICTATION_INJECT": "paste"}),
            mock.patch.object(dictation, "_send_to_herdr", return_value=False),
            mock.patch.object(dictation, "get_desktop", return_value=backend),
            self.assertRaisesRegex(DictationError, "GNOME Wayland"),
        ):
            dictation.inject("hello")

    def test_auto_injection_keeps_native_herdr_path(self) -> None:
        with (
            mock.patch.dict("os.environ", {"DICTATION_INJECT": "auto"}),
            mock.patch.object(dictation, "_send_to_herdr", return_value=True) as send,
            mock.patch.object(dictation, "_copy_to_clipboard") as copy,
            mock.patch.object(dictation, "_send_key") as send_key,
            mock.patch.object(dictation, "_type_text") as type_text,
        ):
            dictation.inject("hello")
        send.assert_called_once_with("hello")
        copy.assert_not_called()
        send_key.assert_not_called()
        type_text.assert_not_called()

    def test_focus_change_before_copy_cancels_paste(self) -> None:
        window = Window("42", "firefox", 10)
        other = Window("99", "foot", 11)
        desktop = mock.Mock()
        desktop.active_window.side_effect = [window, other]
        desktop.has_clipboard.return_value = True
        desktop.read_clipboard.return_value = (True, "previous")
        with (
            mock.patch.dict("os.environ", {"DICTATION_INJECT": "paste"}),
            mock.patch.object(dictation, "_send_to_herdr", return_value=False),
            mock.patch.object(dictation, "get_desktop", return_value=desktop),
            mock.patch.object(dictation, "_send_key") as send_key,
            self.assertRaisesRegex(DictationError, "focused window changed"),
        ):
            dictation.inject("hello")
        desktop.write_clipboard.assert_not_called()
        send_key.assert_not_called()

    def test_focus_change_after_copy_cancels_paste_and_restores_clipboard(
        self,
    ) -> None:
        window = Window("42", "firefox", 10)
        other = Window("99", "foot", 11)
        desktop = mock.Mock()
        desktop.active_window.side_effect = [window, window, other]
        desktop.has_clipboard.return_value = True
        desktop.read_clipboard.side_effect = [
            (True, "previous"),
            (True, "hello"),
        ]
        desktop.write_clipboard.return_value = True
        with (
            mock.patch.dict("os.environ", {"DICTATION_INJECT": "paste"}),
            mock.patch.object(dictation, "_send_to_herdr", return_value=False),
            mock.patch.object(dictation, "get_desktop", return_value=desktop),
            mock.patch.object(dictation.time, "sleep"),
            self.assertRaisesRegex(DictationError, "focused window changed"),
        ):
            dictation.inject("hello")

        desktop.send_key.assert_not_called()
        desktop.write_clipboard.assert_any_call("hello")
        desktop.write_clipboard.assert_called_with("previous")

    def test_focus_change_before_type_cancels_typing(self) -> None:
        window = Window("42", "alacritty", 10)
        other = Window("99", "foot", 11)
        desktop = mock.Mock()
        desktop.active_window.side_effect = [window, other]
        desktop.has_clipboard.return_value = False
        with (
            mock.patch.dict("os.environ", {"DICTATION_INJECT": "type"}),
            mock.patch.object(dictation, "_send_to_herdr", return_value=False),
            mock.patch.object(dictation, "get_desktop", return_value=desktop),
            mock.patch.object(dictation.time, "sleep"),
            self.assertRaisesRegex(DictationError, "focused window changed"),
        ):
            dictation.inject("hello")
        desktop.type_text.assert_not_called()

    def test_allowed_to_denied_focus_change_cannot_bypass_policy(self) -> None:
        allowed = Window("42", "firefox", 10)
        denied = Window("99", "net-runelite-client-runelite", 11)
        desktop = mock.Mock()
        desktop.active_window.side_effect = [allowed, allowed, denied]
        desktop.has_clipboard.return_value = True
        desktop.read_clipboard.side_effect = [(False, ""), (True, "buying fish")]
        desktop.write_clipboard.return_value = True
        with (
            mock.patch.dict("os.environ", {"DICTATION_INJECT": "paste"}),
            mock.patch.object(dictation, "_send_to_herdr", return_value=False),
            mock.patch.object(dictation, "get_desktop", return_value=desktop),
            mock.patch.object(dictation.time, "sleep"),
            self.assertRaisesRegex(DictationError, "focused window changed"),
        ):
            dictation.inject("buying fish")
        desktop.send_key.assert_not_called()
        desktop.write_clipboard.assert_called_with("")


if __name__ == "__main__":
    unittest.main()
