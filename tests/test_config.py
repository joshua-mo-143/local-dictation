from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_dictation import config


class RuntimePathTests(unittest.TestCase):
    def test_runtime_directory_defaults_under_xdg_runtime_dir(self) -> None:
        self.assertEqual(
            config.runtime_dir({"XDG_RUNTIME_DIR": "/run/user/1000"}),
            Path("/run/user/1000/local-dictation"),
        )
        self.assertEqual(config.runtime_dir({}), Path("/tmp/local-dictation"))

    def test_absolute_override_isolates_a_development_server(self) -> None:
        environment = {
            "XDG_RUNTIME_DIR": "/run/user/1000",
            "DICTATION_RUNTIME_DIR": "/run/user/1000/local-dictation-dev",
        }

        directory = config.runtime_dir(environment)

        self.assertEqual(directory, Path("/run/user/1000/local-dictation-dev"))
        self.assertEqual(
            config.socket_path(directory),
            Path("/run/user/1000/local-dictation-dev.sock"),
        )

    def test_relative_override_is_ignored(self) -> None:
        environment = {
            "XDG_RUNTIME_DIR": "/run/user/1000",
            "DICTATION_RUNTIME_DIR": "relative",
        }

        self.assertEqual(
            config.runtime_dir(environment), Path("/run/user/1000/local-dictation")
        )

    def test_recording_paths_share_the_runtime_directory(self) -> None:
        self.assertEqual(config.STT_SOCKET, config.socket_path(config.STATE_DIR))
        for path in (config.WAV_PATH, config.PID_PATH, config.RECORDER_LOG):
            self.assertEqual(path.parent, config.STATE_DIR)
        self.assertEqual(config.SERVICE_NAME, "local-dictation.service")


class ConfigDirectoryTests(unittest.TestCase):
    HOME = Path("/home/example")

    def test_config_directory_follows_xdg_config_home(self) -> None:
        self.assertEqual(
            config.config_dir({"XDG_CONFIG_HOME": "/cfg"}, home=self.HOME),
            Path("/cfg/local-dictation"),
        )
        self.assertEqual(
            config.vocabulary_path({"XDG_CONFIG_HOME": "/cfg"}, home=self.HOME),
            Path("/cfg/local-dictation/vocabulary.json"),
        )

    def test_missing_or_relative_xdg_config_home_uses_home(self) -> None:
        for environment in ({}, {"XDG_CONFIG_HOME": "relative"}):
            with self.subTest(environment=environment):
                self.assertEqual(
                    config.xdg_config_home(environment, home=self.HOME),
                    self.HOME / ".config",
                )


class TomlTableTests(unittest.TestCase):
    def test_missing_file_is_an_empty_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(
                config.read_toml_table(Path(temporary) / "missing.toml"), {}
            )

    def test_malformed_file_raises_the_requested_error_with_its_path(self) -> None:
        class CustomError(ValueError):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            path.write_text("[stt\n")

            with self.assertRaisesRegex(CustomError, rf"settings {path}"):
                config.read_toml_table(path, error=CustomError, label="settings")


if __name__ == "__main__":
    unittest.main()
