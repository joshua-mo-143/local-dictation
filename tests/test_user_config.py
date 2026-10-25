from __future__ import annotations

import stat
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from local_dictation import user_config
from local_dictation.user_config import ComputeDevice, UserConfigurationError


class UserConfigDefaultsTests(unittest.TestCase):
    HOME = Path("/home/example")

    def test_defaults_select_local_parakeet_and_auto_injection(self) -> None:
        config = user_config.default_user_config(home=self.HOME)

        self.assertIs(config.stt.device, ComputeDevice.AUTO)
        self.assertEqual(config.stt.backend, "parakeet")
        self.assertEqual(config.stt.model, "nemo-parakeet-tdt-0.6b-v2")
        self.assertEqual(config.stt.quantization, "int8")
        self.assertEqual(config.stt.compute, "float16")
        self.assertEqual(config.stt.language, "auto")
        self.assertEqual(config.dictation.source, "")
        self.assertEqual(config.dictation.inject, "auto")
        self.assertEqual(config.dictation.prompt, user_config.DEFAULT_DICTATION_PROMPT)
        self.assertEqual(
            config.dictation.replacements,
            (("herder", "herdr"), ("cursa", "Cursor"), ("curser", "Cursor")),
        )
        self.assertEqual(config.dictation.vad_end_silence_ms, 900.0)
        self.assertEqual(config.dictation.vad_max_seconds, 120.0)
        self.assertEqual(config.dictation.vad_min_speech_rms, 1100.0)
        self.assertEqual(config.dictation.vad_start_speech_frames, 3)

    def test_config_path_lives_under_xdg_config_home(self) -> None:
        path = user_config.user_config_path({"XDG_CONFIG_HOME": "/cfg"}, home=self.HOME)
        self.assertEqual(path, Path("/cfg/local-dictation/config.toml"))

    def test_relative_xdg_config_home_falls_back_to_home(self) -> None:
        path = user_config.user_config_path({"XDG_CONFIG_HOME": "cfg"}, home=self.HOME)
        self.assertEqual(path, self.HOME / ".config/local-dictation/config.toml")


class UserConfigPrecedenceTests(unittest.TestCase):
    HOME = Path("/home/example")

    def test_config_file_overrides_defaults_and_environment_overrides_both(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            path.write_text(
                "[stt]\n"
                'device = "cpu"\n'
                'backend = "whisper"\n'
                'model = "config-model"\n'
                'language = "english"\n'
                "[dictation]\n"
                'source = "config-source"\n'
                'inject = "paste"\n'
                'prompt = "config prompt"\n'
                'replacements = "spoken:written"\n'
                "vad_end_silence_ms = 750\n"
            )
            config = user_config.load_user_config(
                {
                    "DICTATION_DEVICE": "CUDA",
                    "DICTATION_MODEL": "environment-model",
                    "DICTATION_INJECT": "stdout",
                    "DICTATION_PROMPT": "environment prompt",
                    "DICTATION_VAD_START_SPEECH_FRAMES": "5",
                },
                path=path,
                home=self.HOME,
            )

        self.assertIs(config.stt.device, ComputeDevice.CUDA)
        self.assertEqual(config.stt.backend, "whisper")
        self.assertEqual(config.stt.model, "environment-model")
        self.assertEqual(config.stt.language, "english")
        self.assertEqual(config.dictation.source, "config-source")
        self.assertEqual(config.dictation.inject, "stdout")
        self.assertEqual(config.dictation.prompt, "environment prompt")
        self.assertEqual(config.dictation.replacements, (("spoken", "written"),))
        self.assertEqual(config.dictation.vad_end_silence_ms, 750)
        self.assertEqual(config.dictation.vad_start_speech_frames, 5)

    def test_empty_replacements_disable_static_corrections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = user_config.load_user_config(
                {"DICTATION_REPLACEMENTS": ""},
                path=Path(temporary) / "missing.toml",
                home=self.HOME,
            )

        self.assertEqual(config.dictation.replacements, ())

    def test_replacements_accept_a_toml_table_and_skip_malformed_entries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            path.write_text('[dictation]\nreplacements = { herder = "herdr" }\n')
            from_table = user_config.load_user_config({}, path=path, home=self.HOME)
            from_text = user_config.load_user_config(
                {"DICTATION_REPLACEMENTS": " a : b ;missing-colon; :empty"},
                path=path,
                home=self.HOME,
            )

        self.assertEqual(from_table.dictation.replacements, (("herder", "herdr"),))
        self.assertEqual(from_text.dictation.replacements, (("a", "b"),))

    def test_default_path_resolves_from_xdg_config_home(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_dir = root / "local-dictation"
            config_dir.mkdir()
            (config_dir / "config.toml").write_text('[stt]\nbackend = "whisper"\n')

            config = user_config.load_user_config(
                {"XDG_CONFIG_HOME": str(root)}, home=self.HOME
            )

        self.assertEqual(config.stt.backend, "whisper")

    def test_running_snapshot_retains_values_and_restart_observes_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            path.write_text('[stt]\nmodel = "startup-model"\n')
            running = user_config.load_user_config({}, path=path, home=self.HOME)

            path.write_text('[stt]\nmodel = "restart-model"\n')
            restarted = user_config.load_user_config({}, path=path, home=self.HOME)

        self.assertEqual(running.stt.model, "startup-model")
        self.assertEqual(restarted.stt.model, "restart-model")


class UserConfigValidationTests(unittest.TestCase):
    HOME = Path("/home/example")

    def _load(
        self, text: str, environment: dict[str, str] | None = None
    ) -> user_config.UserConfig:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            path.write_text(text)
            return user_config.load_user_config(
                environment or {}, path=path, home=self.HOME
            )

    def test_unknown_section_is_rejected(self) -> None:
        with self.assertRaisesRegex(UserConfigurationError, "configuration section"):
            self._load("[providers]\nllm = 1\n")

    def test_unknown_key_within_section_is_rejected(self) -> None:
        with self.assertRaisesRegex(UserConfigurationError, r"\[stt\] key"):
            self._load("[stt]\ncuda_device = 0\n")
        with self.assertRaisesRegex(UserConfigurationError, r"\[dictation\] key"):
            self._load("[dictation]\nsink = 'speakers'\n")

    def test_section_must_be_a_table(self) -> None:
        with self.assertRaisesRegex(UserConfigurationError, "TOML table"):
            self._load("stt = 1\n")

    def test_invalid_choices_name_the_setting(self) -> None:
        cases = {
            "stt.device": "[stt]\ndevice = 'tpu'\n",
            "stt.backend": "[stt]\nbackend = 'vosk'\n",
            "stt.language": "[stt]\nlanguage = 'klingon'\n",
            "dictation.inject": "[dictation]\ninject = 'telepathy'\n",
        }
        for label, text in cases.items():
            with (
                self.subTest(label),
                self.assertRaisesRegex(UserConfigurationError, label),
            ):
                self._load(text)

    def test_invalid_environment_override_is_rejected(self) -> None:
        with self.assertRaisesRegex(UserConfigurationError, "stt.device"):
            self._load("", {"DICTATION_DEVICE": "gpu"})

    def test_empty_engine_selectors_are_rejected(self) -> None:
        for key in ("model", "quantization", "compute"):
            with (
                self.subTest(key),
                self.assertRaisesRegex(UserConfigurationError, f"stt.{key}"),
            ):
                self._load(f"[stt]\n{key} = '  '\n")

    def test_numeric_vad_settings_are_validated(self) -> None:
        cases = {
            "vad_start_speech_frames": ("0", "true", "1.5"),
            "vad_end_silence_ms": ("0", "-1", "'nan'"),
            "vad_max_seconds": ("0", "'inf'"),
            "vad_min_speech_rms": ("-1", "'loud'"),
        }
        for key, values in cases.items():
            for value in values:
                with (
                    self.subTest(key=key, value=value),
                    self.assertRaisesRegex(UserConfigurationError, f"dictation.{key}"),
                ):
                    self._load(f"[dictation]\n{key} = {value}\n")

    def test_zero_speech_rms_is_allowed(self) -> None:
        config = self._load("[dictation]\nvad_min_speech_rms = 0\n")
        self.assertEqual(config.dictation.vad_min_speech_rms, 0.0)

    def test_malformed_config_names_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.toml"
            config_path.write_text("[stt\n")

            with self.assertRaisesRegex(
                UserConfigurationError, rf"user configuration {config_path}"
            ):
                user_config.load_user_config({}, path=config_path, home=self.HOME)

    def test_invalid_utf8_config_names_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.toml"
            config_path.write_bytes(b"\xff")

            with self.assertRaisesRegex(
                UserConfigurationError, rf"user configuration {config_path}"
            ):
                user_config.load_user_config({}, path=config_path, home=self.HOME)

    def test_unreadable_config_names_source_path(self) -> None:
        config_path = Path("/config/config.toml")
        with (
            mock.patch.object(
                Path, "read_text", side_effect=PermissionError("permission denied")
            ),
            self.assertRaisesRegex(
                UserConfigurationError, rf"user configuration {config_path}"
            ),
        ):
            user_config.load_user_config({}, path=config_path, home=self.HOME)


class LocalComputeResolutionTests(unittest.TestCase):
    def test_cpu_never_requires_cuda(self) -> None:
        self.assertEqual(
            user_config.resolve_local_compute(
                ComputeDevice.CPU, cuda_available=True, label="dictation"
            ),
            "cpu",
        )

    def test_auto_follows_availability(self) -> None:
        for available, expected in ((True, "cuda"), (False, "cpu")):
            with self.subTest(available=available):
                self.assertEqual(
                    user_config.resolve_local_compute(
                        ComputeDevice.AUTO, cuda_available=available, label="dictation"
                    ),
                    expected,
                )

    def test_explicit_cuda_requires_availability(self) -> None:
        self.assertEqual(
            user_config.resolve_local_compute(
                ComputeDevice.CUDA, cuda_available=True, label="dictation"
            ),
            "cuda",
        )
        with self.assertRaisesRegex(RuntimeError, "install the cuda extra"):
            user_config.resolve_local_compute(
                ComputeDevice.CUDA, cuda_available=False, label="dictation"
            )


class UserConfigWriteTests(unittest.TestCase):
    HOME = Path("/home/example")

    def test_atomic_write_round_trips_and_is_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = user_config.default_user_config(home=self.HOME)
            path = root / "nested" / "config.toml"

            user_config.write_user_config(config, path)

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(
                user_config.load_user_config({}, path=path, home=self.HOME), config
            )
            self.assertEqual(
                [entry.name for entry in path.parent.iterdir()], ["config.toml"]
            )

    def test_non_default_values_survive_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            defaults = user_config.default_user_config(home=self.HOME)
            config = replace(
                defaults,
                stt=replace(defaults.stt, device=ComputeDevice.CUDA, backend="whisper"),
                dictation=replace(
                    defaults.dictation,
                    source="alsa_input.usb-example",
                    prompt='Say "quoted" C:\\paths',
                    replacements=(("a", "b"),),
                    vad_end_silence_ms=650.5,
                ),
            )

            user_config.write_user_config(config, path)

            self.assertEqual(
                user_config.load_user_config({}, path=path, home=self.HOME), config
            )

    def test_control_characters_in_text_survive_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            defaults = user_config.default_user_config(home=self.HOME)
            config = replace(
                defaults,
                dictation=replace(
                    defaults.dictation,
                    prompt="line one\nline two\ttabbed\r\x00\x1f\x7f\b\f end",
                ),
            )

            user_config.write_user_config(config, path)

            self.assertEqual(
                user_config.load_user_config({}, path=path, home=self.HOME), config
            )

    def test_failed_write_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            config = user_config.default_user_config(home=self.HOME)
            with (
                mock.patch.object(
                    user_config.os, "replace", side_effect=OSError("disk full")
                ),
                self.assertRaises(OSError),
            ):
                user_config.write_user_config(config, path)

            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
