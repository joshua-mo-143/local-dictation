from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import tempfile
import unittest
from collections.abc import Iterable
from pathlib import Path
from unittest import mock

from local_dictation import config_management
from local_dictation.config import SERVICE_NAME
from local_dictation.user_config import (
    ComputeDevice,
    UserConfig,
    UserConfigurationError,
    default_user_config,
    load_user_config,
    render_user_config,
)

CONFIG_KEYS = (
    "stt.device",
    "stt.backend",
    "stt.model",
    "stt.quantization",
    "stt.compute",
    "stt.language",
    "dictation.source",
    "dictation.inject",
    "dictation.prompt",
    "dictation.replacements",
    "dictation.vad_end_silence_ms",
    "dictation.vad_max_seconds",
    "dictation.vad_min_speech_rms",
    "dictation.vad_start_speech_frames",
)
STT_KEYS = CONFIG_KEYS[:6]
DICTATION_KEYS = CONFIG_KEYS[6:]
SERVER_KEYS = frozenset((*STT_KEYS, "dictation.prompt", "dictation.replacements"))
VALID_VALUES: dict[str, tuple[str, object]] = {
    "stt.device": ("CUDA", ComputeDevice.CUDA),
    "stt.backend": ("Whisper", "whisper"),
    "stt.model": (" large-v3 ", "large-v3"),
    "stt.quantization": ("fp16", "fp16"),
    "stt.compute": ("int8", "int8"),
    "stt.language": ("EN", "en"),
    "dictation.source": (" usb-mic ", "usb-mic"),
    "dictation.inject": ("Paste", "paste"),
    "dictation.prompt": ("  Rust and Python terms ", "Rust and Python terms"),
    "dictation.replacements": (
        "pie test:pytest; rust c : rustc ;",
        (("pie test", "pytest"), ("rust c", "rustc")),
    ),
    "dictation.vad_end_silence_ms": ("650", 650.0),
    "dictation.vad_max_seconds": ("30.5", 30.5),
    "dictation.vad_min_speech_rms": ("0", 0.0),
    "dictation.vad_start_speech_frames": ("5", 5),
}
SOURCES = ("usb-mic", "headset-mic")


def setting(config: UserConfig, key: str) -> object:
    section, attribute = key.split(".")
    return getattr(getattr(config, section), attribute)


class ConfigManagementTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "local-dictation" / "config.toml"
        self.supervisor = mock.Mock()
        self.supervisor.binary_available.return_value = True
        self.supervisor.is_active.return_value = "inactive"
        services = mock.patch.object(
            config_management, "user_services", return_value=self.supervisor
        )
        services.start()
        self.addCleanup(services.stop)

    def commit(self, key: str, value: str) -> config_management.ConfigChangeResult:
        return config_management.commit_config_setting(key, value, path=self.path)

    def stored(self) -> UserConfig:
        return load_user_config({}, path=self.path)

    def test_set_stores_typed_values_for_every_key(self) -> None:
        self.assertEqual(tuple(VALID_VALUES), CONFIG_KEYS)
        for key, (raw, expected) in VALID_VALUES.items():
            with self.subTest(key=key):
                result = self.commit(key, raw)
                self.assertEqual(result.changed_keys, (key,))
                self.assertEqual(setting(result.config, key), expected)

        stored = self.stored()
        for key, (_raw, expected) in VALID_VALUES.items():
            self.assertEqual(setting(stored, key), expected)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_invalid_values_are_rejected_before_writing(self) -> None:
        cases = (
            ("stt.device", "tpu", "one of: auto, cpu, cuda"),
            ("stt.backend", "vosk", "one of"),
            ("stt.model", "   ", "empty"),
            ("stt.language", "klingon", "one of"),
            ("dictation.inject", "telepathy", "one of"),
            ("dictation.prompt", "first line\nsecond line", "single line"),
            ("dictation.replacements", "pytest", "spoken:written"),
            ("dictation.replacements", "rustc:rustc; :pytest", "spoken:written"),
            ("dictation.vad_end_silence_ms", "soon", "number"),
            ("dictation.vad_end_silence_ms", "0", "positive"),
            ("dictation.vad_max_seconds", "inf", "finite"),
            ("dictation.vad_min_speech_rms", "-1", "non-negative"),
            ("dictation.vad_start_speech_frames", "2.5", "integer"),
            ("dictation.vad_start_speech_frames", "0", "positive integer"),
        )
        for key, raw, message in cases:
            with self.subTest(key=key, value=raw):
                with self.assertRaisesRegex(
                    UserConfigurationError, f"^{re.escape(key)} .*{message}"
                ):
                    self.commit(key, raw)
                self.assertFalse(self.path.exists())

    def test_keys_ignore_case_and_surrounding_space(self) -> None:
        result = self.commit(" STT.Device ", "cpu")

        self.assertEqual(result.changed_keys, ("stt.device",))
        self.assertIs(self.stored().stt.device, ComputeDevice.CPU)

    def test_unknown_keys_list_the_supported_keys(self) -> None:
        message = r"unknown configuration key 'dictation\.hotkey'; .*stt\.device"
        with self.assertRaisesRegex(UserConfigurationError, message):
            self.commit("dictation.hotkey", "f9")
        with self.assertRaisesRegex(UserConfigurationError, message):
            config_management.show_config(key="dictation.hotkey", path=self.path)
        self.assertFalse(self.path.exists())

    def test_value_may_be_omitted_only_for_the_capture_source(self) -> None:
        prompt = mock.Mock(return_value="1")

        with self.assertRaisesRegex(
            UserConfigurationError, "stt.model requires a value"
        ):
            config_management.commit_config_setting(
                "stt.model", None, path=self.path, input_fn=prompt
            )

        prompt.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_source_aliases_select_the_pipewire_system_default(self) -> None:
        for alias in ("default", " System "):
            with self.subTest(alias=alias):
                self.commit("dictation.source", "usb-mic")
                result = self.commit("dictation.source", alias)
                self.assertEqual(result.config.dictation.source, "")
                self.assertEqual(self.stored().dictation.source, "")

    def test_environment_overrides_are_never_written_back(self) -> None:
        overrides = {
            "DICTATION_MODEL": "override-model",
            "DICTATION_SOURCE": "override-mic",
        }
        with mock.patch.dict(os.environ, overrides):
            result = self.commit("stt.device", "cpu")
            shown = config_management.show_config(key="stt.model", path=self.path)

        defaults = default_user_config()
        self.assertEqual(result.config.stt.model, defaults.stt.model)
        self.assertEqual(shown, defaults.stt.model)
        self.assertEqual(self.stored().dictation.source, defaults.dictation.source)
        self.assertNotIn("override", self.path.read_text())

    def test_default_path_follows_xdg_config_home(self) -> None:
        config_home = self.root / "xdg"
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(config_home)}):
            config_management.commit_config_setting("stt.device", "cpu")
            shown = config_management.show_config(key="stt.device")

        expected = config_home / "local-dictation" / "config.toml"
        self.assertIs(load_user_config({}, path=expected).stt.device, ComputeDevice.CPU)
        self.assertEqual(shown, "cpu")

    def test_restart_is_reported_only_for_server_settings_while_it_runs(
        self,
    ) -> None:
        self.supervisor.is_active.return_value = "active"
        for key, (raw, _expected) in VALID_VALUES.items():
            with self.subTest(key=key):
                self.supervisor.is_active.reset_mock()
                result = self.commit(key, raw)
                if key in SERVER_KEYS:
                    self.assertEqual(result.restart_services, (SERVICE_NAME,))
                    self.supervisor.is_active.assert_called_once_with(SERVICE_NAME)
                else:
                    self.assertEqual(result.restart_services, ())
                    self.supervisor.is_active.assert_not_called()

    def test_stopped_service_or_missing_systemctl_needs_no_restart(self) -> None:
        self.assertEqual(self.commit("stt.device", "cpu").restart_services, ())

        self.supervisor.binary_available.return_value = False
        self.supervisor.is_active.reset_mock()
        self.assertEqual(self.commit("stt.device", "cuda").restart_services, ())
        self.supervisor.is_active.assert_not_called()

    def test_restart_notice_names_services_and_the_restart_command(self) -> None:
        notice = config_management.format_restart_notice((SERVICE_NAME,))

        self.assertIn(SERVICE_NAME, notice)
        self.assertIn("`dictate services restart`", notice)
        self.assertEqual(
            config_management.format_restart_notice(()), "No restart needed."
        )

    def test_show_prints_stored_toml_and_formatted_values(self) -> None:
        self.commit("stt.device", "cpu")
        self.commit("dictation.replacements", "pie test:pytest;rust c:rustc")
        self.commit("dictation.vad_max_seconds", "45")

        self.assertEqual(
            config_management.show_config(path=self.path),
            render_user_config(self.stored()).rstrip("\n"),
        )
        shown = {
            key: config_management.show_config(key=key, path=self.path)
            for key in (
                "stt.device",
                "dictation.source",
                "dictation.replacements",
                "dictation.vad_max_seconds",
            )
        }
        self.assertEqual(
            shown,
            {
                "stt.device": "cpu",
                "dictation.source": "",
                "dictation.replacements": "pie test:pytest;rust c:rustc",
                "dictation.vad_max_seconds": "45.0",
            },
        )

    def test_show_json_covers_every_key_with_native_types(self) -> None:
        self.commit("stt.device", "cuda")
        self.commit("dictation.replacements", "pie test:pytest")
        self.commit("dictation.vad_start_speech_frames", "4")

        payload = json.loads(
            config_management.show_config(json_output=True, path=self.path)
        )
        single = json.loads(
            config_management.show_config(
                key="STT.Device", json_output=True, path=self.path
            )
        )

        self.assertEqual(tuple(payload), CONFIG_KEYS)
        self.assertEqual(payload["stt.device"], "cuda")
        self.assertEqual(payload["dictation.replacements"], [["pie test", "pytest"]])
        self.assertEqual(payload["dictation.vad_start_speech_frames"], 4)
        self.assertEqual(single, {"stt.device": "cuda"})

    def test_show_uses_defaults_without_creating_a_file(self) -> None:
        self.assertEqual(
            config_management.show_config(path=self.path),
            render_user_config(default_user_config()).rstrip("\n"),
        )
        self.assertFalse(self.path.parent.exists())

    def test_reset_section_restores_only_that_section(self) -> None:
        self.commit("stt.device", "cpu")
        self.commit("stt.model", "custom-model")
        self.commit("dictation.source", "usb-mic")

        result = config_management.reset_config(section=" STT ", path=self.path)

        self.assertEqual(result.changed_keys, STT_KEYS)
        self.assertEqual(result.restart_services, ())
        self.assertEqual(self.stored().stt, default_user_config().stt)
        self.assertEqual(self.stored().dictation.source, "usb-mic")

    def test_reset_dictation_section_reports_a_running_server(self) -> None:
        self.commit("stt.device", "cpu")
        self.commit("dictation.vad_max_seconds", "30")
        self.supervisor.is_active.return_value = "active"

        result = config_management.reset_config(section="dictation", path=self.path)

        self.assertEqual(result.changed_keys, DICTATION_KEYS)
        self.assertEqual(result.restart_services, (SERVICE_NAME,))
        self.assertEqual(self.stored().dictation, default_user_config().dictation)
        self.assertIs(self.stored().stt.device, ComputeDevice.CPU)

    def test_reset_rejects_unknown_sections(self) -> None:
        with self.assertRaisesRegex(
            UserConfigurationError, "section must be one of: stt, dictation"
        ):
            config_management.reset_config(section="hotkeys", path=self.path)
        self.assertFalse(self.path.exists())

    def test_full_reset_replaces_a_config_that_no_longer_loads(self) -> None:
        self.path.parent.mkdir(parents=True)
        for contents in ("[stt\n", "[obsolete]\nenabled = true\n"):
            with self.subTest(contents=contents):
                self.path.write_text(contents)
                with self.assertRaises(UserConfigurationError):
                    config_management.show_config(path=self.path)

                result = config_management.reset_config(path=self.path)

                self.assertEqual(result.changed_keys, CONFIG_KEYS)
                self.assertEqual(self.stored(), default_user_config())

    def test_writes_happen_while_the_config_lock_is_held(self) -> None:
        lock_path = self.path.with_name("config.toml.lock")
        real_write = config_management.write_user_config
        written: list[Path] = []

        def write_while_locked(config: UserConfig, path: Path) -> None:
            probe = os.open(lock_path, os.O_RDWR)
            try:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(probe)
            written.append(path)
            real_write(config, path)

        with mock.patch.object(
            config_management, "write_user_config", write_while_locked
        ):
            self.commit("stt.device", "cpu")
            config_management.reset_config(section="stt", path=self.path)
            config_management.reset_config(path=self.path)

        self.assertEqual(written, [self.path] * 3)
        self.assertEqual(stat.S_IMODE(lock_path.stat().st_mode), 0o600)

    def test_set_source_without_value_opens_the_picker(self) -> None:
        self.commit("dictation.source", "headset-mic")
        output: list[str] = []

        result = config_management.commit_config_setting(
            "dictation.source",
            None,
            path=self.path,
            input_fn=lambda _prompt: "1",
            print_fn=output.append,
            list_sources=lambda: SOURCES,
        )

        self.assertIn(" * 2. headset-mic", output)
        self.assertEqual(result.changed_keys, ("dictation.source",))
        self.assertEqual(self.stored().dictation.source, "usb-mic")

    def test_picker_keeps_settings_written_while_it_was_open(self) -> None:
        def answer(_prompt: str) -> str:
            self.commit("stt.device", "cpu")
            return "2"

        config_management.commit_config_setting(
            "dictation.source",
            None,
            path=self.path,
            input_fn=answer,
            print_fn=lambda _message: None,
            list_sources=lambda: SOURCES,
        )

        stored = self.stored()
        self.assertEqual(stored.dictation.source, "headset-mic")
        self.assertIs(stored.stt.device, ComputeDevice.CPU)


class CaptureSourcePickerTests(unittest.TestCase):
    def pick(
        self, answers: Iterable[str], *, current: str = ""
    ) -> tuple[str, list[str], list[str]]:
        pending = iter(answers)
        prompts: list[str] = []
        output: list[str] = []

        def answer(prompt: str) -> str:
            prompts.append(prompt)
            try:
                return next(pending)
            except StopIteration:
                raise EOFError from None

        selected = config_management.pick_capture_source(
            current=current,
            input_fn=answer,
            print_fn=output.append,
            list_sources=lambda: SOURCES,
        )
        return selected, prompts, output

    def test_numbered_choice_selects_a_listed_source(self) -> None:
        selected, prompts, output = self.pick(["1"])

        self.assertEqual(selected, "usb-mic")
        self.assertEqual(prompts, ["Capture source [0]: "])
        self.assertEqual(
            output,
            [
                "Capture sources:",
                " * 0. PipeWire system default",
                "   1. usb-mic",
                "   2. headset-mic",
            ],
        )

    def test_enter_keeps_the_current_source(self) -> None:
        selected, prompts, output = self.pick([""], current="headset-mic")

        self.assertEqual(selected, "headset-mic")
        self.assertEqual(prompts, ["Capture source [2]: "])
        self.assertIn(" * 2. headset-mic", output)

    def test_names_and_aliases_are_accepted(self) -> None:
        cases = (
            ("headset-mic", "headset-mic"),
            ("0", ""),
            ("default", ""),
            ("SYSTEM", ""),
        )
        for answer, expected in cases:
            with self.subTest(answer=answer):
                selected, _prompts, _output = self.pick([answer], current="usb-mic")
                self.assertEqual(selected, expected)

    def test_invalid_choices_prompt_again(self) -> None:
        selected, prompts, output = self.pick(["3", "²", "mic", "2"])

        self.assertEqual(selected, "headset-mic")
        self.assertEqual(len(prompts), 4)
        self.assertEqual(output.count("Choose 0-2 or a listed source name."), 3)

    def test_unlisted_current_source_falls_back_to_the_system_default(self) -> None:
        selected, prompts, output = self.pick([""], current="unplugged-mic")

        self.assertEqual(selected, "")
        self.assertEqual(prompts, ["Capture source [0]: "])
        self.assertIn(
            "Current source unplugged-mic is not among the listed devices.", output
        )

    def test_end_of_input_points_to_an_explicit_value(self) -> None:
        with self.assertRaisesRegex(
            UserConfigurationError,
            "interactive terminal.*`dictate config set dictation.source <node-name>`",
        ):
            self.pick([])

    def test_listing_failure_points_to_an_explicit_value(self) -> None:
        failure = UserConfigurationError("wpctl not found")

        with self.assertRaisesRegex(
            UserConfigurationError,
            "^wpctl not found; .*`dictate config set dictation.source <node-name>`",
        ):
            config_management.pick_capture_source(
                input_fn=mock.Mock(side_effect=AssertionError("unexpected prompt")),
                print_fn=lambda _message: None,
                list_sources=mock.Mock(side_effect=failure),
            )

    def test_sources_come_from_pipewire_by_default(self) -> None:
        with mock.patch.object(
            config_management,
            "list_pipewire_capture_sources",
            return_value=("usb-mic",),
        ) as list_sources:
            selected = config_management.pick_capture_source(
                input_fn=lambda _prompt: "1",
                print_fn=lambda _message: None,
            )

        list_sources.assert_called_once_with()
        self.assertEqual(selected, "usb-mic")


if __name__ == "__main__":
    unittest.main()
