from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from local_dictation.stt import launcher
from local_dictation.user_config import ComputeDevice


def _passwd(home: Path) -> mock.Mock:
    entry = mock.Mock()
    entry.pw_dir = str(home)
    return entry


class ProtectedEnvironmentTests(unittest.TestCase):
    def test_paths_derive_from_account_home_and_session_runtime(self) -> None:
        home = Path("/home/example")
        with mock.patch.object(launcher.pwd, "getpwuid", return_value=_passwd(home)):
            environment = launcher.protected_environment(
                {"HOME": "/attacker", "XDG_RUNTIME_DIR": "/attacker/run"}, uid=1000
            )

        self.assertEqual(set(environment), launcher.PROTECTED_ENVIRONMENT_KEYS)
        self.assertEqual(
            environment,
            {
                "CUDA_CACHE_PATH": "/run/user/1000/local-dictation/cuda-cache",
                "HF_HOME": "/home/example/.cache/huggingface",
                "TMPDIR": "/run/user/1000/local-dictation",
                "HOME": "/home/example",
                "XDG_CACHE_HOME": "/home/example/.cache",
                "XDG_RUNTIME_DIR": "/run/user/1000",
                "ORT_DISABLE_TELEMETRY": "1",
            },
        )

    def test_onnx_runtime_telemetry_stays_off_when_the_caller_enables_it(
        self,
    ) -> None:
        with mock.patch.object(
            launcher.pwd, "getpwuid", return_value=_passwd(Path("/home/example"))
        ):
            environment = launcher.protected_environment(
                {"ORT_DISABLE_TELEMETRY": "0"}, uid=1000
            )

        self.assertEqual(environment["ORT_DISABLE_TELEMETRY"], "1")

    def test_development_runtime_override_moves_scratch_paths(self) -> None:
        with mock.patch.object(
            launcher.pwd, "getpwuid", return_value=_passwd(Path("/home/example"))
        ):
            environment = launcher.protected_environment(
                {"DICTATION_RUNTIME_DIR": "/run/user/1000/local-dictation-dev"},
                uid=1000,
            )

        self.assertEqual(environment["TMPDIR"], "/run/user/1000/local-dictation-dev")
        self.assertEqual(
            environment["CUDA_CACHE_PATH"],
            "/run/user/1000/local-dictation-dev/cuda-cache",
        )


class NvidiaLibraryPathTests(unittest.TestCase):
    def test_wheel_libraries_are_prepended_to_existing_search_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            packages = Path(temporary)
            library = packages / "nvidia" / "cublas" / "lib"
            library.mkdir(parents=True)
            not_a_directory = packages / "nvidia" / "nccl" / "lib"
            not_a_directory.parent.mkdir()
            not_a_directory.write_text("")
            environment = {"LD_LIBRARY_PATH": "/opt/lib"}
            with mock.patch.object(
                launcher.site, "getsitepackages", return_value=[str(packages)]
            ):
                launcher.add_nvidia_library_paths(environment)

        self.assertEqual(environment["LD_LIBRARY_PATH"], f"{library}:/opt/lib")

    def test_missing_wheel_libraries_leave_environment_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment: dict[str, str] = {}
            with mock.patch.object(
                launcher.site, "getsitepackages", return_value=[temporary]
            ):
                launcher.add_nvidia_library_paths(environment)

        self.assertEqual(environment, {})


class LauncherMainTests(unittest.TestCase):
    def _launch(
        self,
        inherited: dict[str, str],
        *,
        device: ComputeDevice,
        owned: dict[str, str] | None = None,
    ) -> tuple[list[str], dict[str, str], mock.Mock]:
        captured: dict[str, object] = {}
        user_config = mock.Mock()
        user_config.stt.device = device

        def inspect_execve(
            executable: str, arguments: list[str], environment: dict[str, str]
        ) -> None:
            captured.update(
                executable=executable, arguments=arguments, environment=environment
            )
            raise RuntimeError("exec inspected")

        with (
            mock.patch.dict(os.environ, inherited, clear=True),
            mock.patch.object(
                launcher, "protected_environment", return_value=owned or {}
            ),
            mock.patch.object(launcher, "load_user_config", return_value=user_config),
            mock.patch.object(launcher, "add_nvidia_library_paths") as discover,
            mock.patch.object(launcher.os, "execve", side_effect=inspect_execve),
            mock.patch.object(launcher.sys, "argv", ["local-dictation-server"]),
            self.assertRaisesRegex(RuntimeError, "exec inspected"),
        ):
            launcher.main()

        self.assertEqual(captured["executable"], launcher.sys.executable)
        arguments = captured["arguments"]
        environment = captured["environment"]
        assert isinstance(arguments, list) and isinstance(environment, dict)
        return arguments, environment, discover

    def test_check_reports_ready_without_launching(self) -> None:
        output = io.StringIO()
        with (
            mock.patch.object(
                launcher.sys, "argv", ["local-dictation-server", "--check"]
            ),
            mock.patch.object(launcher.os, "execve") as execve,
            redirect_stdout(output),
        ):
            launcher.main()

        self.assertEqual(output.getvalue(), "local-dictation-server: ok\n")
        execve.assert_not_called()

    def test_owned_paths_replace_inherited_values_and_overrides_pass_through(
        self,
    ) -> None:
        owned = {
            key: f"/owned/{key.casefold()}"
            for key in launcher.PROTECTED_ENVIRONMENT_KEYS
        }
        inherited = {
            **{key: "/attacker" for key in launcher.PROTECTED_ENVIRONMENT_KEYS},
            "XDG_CONFIG_HOME": "/config",
            "DICTATION_BACKEND": "whisper",
        }

        arguments, environment, _discover = self._launch(
            inherited, device=ComputeDevice.AUTO, owned=owned
        )

        self.assertEqual(arguments[1:], ["-m", "local_dictation.stt.server"])
        for key, expected in owned.items():
            self.assertEqual(environment[key], expected)
        self.assertEqual(environment["XDG_CONFIG_HOME"], "/config")
        self.assertEqual(environment["DICTATION_BACKEND"], "whisper")

    def test_cpu_launch_does_not_discover_nvidia_libraries(self) -> None:
        _arguments, _environment, discover = self._launch({}, device=ComputeDevice.CPU)

        discover.assert_not_called()

    def test_auto_and_cuda_launches_discover_nvidia_libraries(self) -> None:
        for device in (ComputeDevice.AUTO, ComputeDevice.CUDA):
            with self.subTest(device=device):
                _arguments, _environment, discover = self._launch({}, device=device)

                discover.assert_called_once()


if __name__ == "__main__":
    unittest.main()
