from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_dictation import production_extras
from local_dictation.config import PROJECT_ROOT


class ProductionExtraSpecTests(unittest.TestCase):
    def test_every_declared_extra_has_an_install_and_entry_point_spec(self) -> None:
        pyproject = PROJECT_ROOT / "pyproject.toml"
        extras = production_extras.declared_extras(pyproject)
        self.assertEqual(extras, ("cpu", "cuda", "whisper"))
        self.assertEqual(extras, tuple(sorted(production_extras.EXTRA_SPECS)))
        self.assertEqual(production_extras.missing_extra_specs(pyproject), ())
        for extra, spec in production_extras.EXTRA_SPECS.items():
            with self.subTest(extra=extra):
                self.assertTrue(spec.imports)
                self.assertEqual(
                    spec.commands, (("local-dictation-server", ("--check",)),)
                )

    def test_entry_point_checks_avoid_model_loads(self) -> None:
        self.assertEqual(production_extras.CORE_COMMANDS, (("dictate", ("--help",)),))
        for name, arguments in (
            *production_extras.CORE_COMMANDS,
            *(
                command
                for spec in production_extras.EXTRA_SPECS.values()
                for command in spec.commands
            ),
        ):
            with self.subTest(entry_point=name):
                self.assertIn(arguments[0], {"--check", "--help"})

    def test_cuda_extra_must_resolve_the_gpu_runtime(self) -> None:
        self.assertEqual(
            production_extras.EXTRA_SPECS["cuda"].required_distributions,
            ("onnxruntime-gpu",),
        )
        self.assertIn("onnxruntime", production_extras.EXTRA_SPECS["cpu"].imports)


class ProductionExtraCommandTests(unittest.TestCase):
    def test_export_and_install_commands_keep_extras_in_separate_environments(
        self,
    ) -> None:
        wheel = Path("/tmp/local_dictation-0.1.0-py3-none-any.whl")
        python = Path("/tmp/extra-cpu/bin/python")
        requirements = Path("/tmp/cpu.txt")
        self.assertEqual(
            production_extras.export_command("cpu"),
            [
                "uv",
                "export",
                "--locked",
                "--no-dev",
                "--no-emit-project",
                "--extra",
                "cpu",
            ],
        )
        self.assertEqual(
            production_extras.install_commands(python, wheel, requirements),
            [
                ["uv", "pip", "install", "--python", str(python), str(wheel)],
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "-r",
                    str(requirements),
                ],
            ],
        )
        self.assertEqual(
            production_extras.import_command(python, ("onnx_asr", "onnxruntime")),
            [str(python), "-c", "import onnx_asr; import onnxruntime"],
        )
        self.assertEqual(
            production_extras.show_distribution_command(python, "onnxruntime-gpu"),
            ["uv", "pip", "show", "--python", str(python), "onnxruntime-gpu"],
        )

    def test_unknown_extra_fails_before_installing(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(production_extras, "_run") as run,
            self.assertRaises(SystemExit) as raised,
        ):
            production_extras.check_environments(
                project=PROJECT_ROOT,
                python="3.11",
                extras=("dictation",),
                work=Path(temporary),
            )
        self.assertIn("no extra spec for dictation", str(raised.exception))
        run.assert_not_called()

    def test_check_environments_installs_core_then_each_extra(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            wheel = work / "dist" / "pkg.whl"
            recorded: list[list[str]] = []

            def fake_run(command: list[str], **_options: object) -> mock.Mock:
                recorded.append(list(command))
                if command[:3] == ["uv", "build", "--wheel"]:
                    wheel.parent.mkdir(parents=True, exist_ok=True)
                    wheel.write_text("wheel")
                if command[:2] == ["uv", "export"]:
                    Path(command[-1]).write_text("deps\n")
                if command[:2] == ["uv", "venv"]:
                    destination = Path(command[-1])
                    (destination / "bin").mkdir(parents=True)
                    for name in ("python", "dictate", "local-dictation-server"):
                        (destination / "bin" / name).write_text(name)
                return mock.Mock(returncode=0, stdout="", stderr="")

            with (
                mock.patch.object(production_extras, "_run", side_effect=fake_run),
                mock.patch("builtins.print"),
            ):
                production_extras.check_environments(
                    project=PROJECT_ROOT,
                    python="3.11",
                    extras=("cpu", "cuda"),
                    work=work,
                )

        exported = [command for command in recorded if command[:2] == ["uv", "export"]]
        self.assertEqual(exported[0][-2:], ["-o", str(work / "core.txt")])
        self.assertEqual(exported[1][-4:-2], ["--extra", "cpu"])
        self.assertEqual(exported[2][-4:-2], ["--extra", "cuda"])
        self.assertIn([str(work / "core/bin/dictate"), "--help"], recorded)
        for extra in ("cpu", "cuda"):
            self.assertIn(
                [str(work / extra / "bin/local-dictation-server"), "--check"],
                recorded,
            )
            self.assertIn(
                [
                    str(work / extra / "bin/python"),
                    "-m",
                    "local_dictation.vad",
                    "--check",
                ],
                recorded,
            )
        self.assertIn(
            [
                str(work / "cpu/bin/python"),
                "-c",
                "import onnx_asr; import onnxruntime; import huggingface_hub",
            ],
            recorded,
        )
        self.assertIn(
            production_extras.show_distribution_command(
                work / "cuda/bin/python", "onnxruntime-gpu"
            ),
            recorded,
        )

    def test_imports_run_with_telemetry_off_in_scratch_directories(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.dict(
                production_extras.os.environ,
                {
                    "HOME": "/home/example",
                    "XDG_CACHE_HOME": "/home/example/.cache",
                    "XDG_CONFIG_HOME": "/home/example/.config",
                    "ORT_DISABLE_TELEMETRY": "0",
                    "PATH": "/usr/bin",
                },
                clear=True,
            ),
            mock.patch.object(production_extras, "_run") as run,
        ):
            scratch = Path(temporary) / "cpu-imports"
            python = Path("/tmp/extra-cpu/bin/python")
            production_extras._check_imports(python, ("onnxruntime",), scratch)

        run.assert_called_once_with(
            [str(python), "-c", "import onnxruntime"],
            env={
                "ORT_DISABLE_TELEMETRY": "1",
                "PATH": "/usr/bin",
                "HOME": str(scratch / "home"),
                "TMPDIR": str(scratch / "tmp"),
            },
        )

    def test_imports_that_leave_files_behind_fail_the_check(self) -> None:
        def write_device_id(_command: list[str], *, env: dict[str, str]) -> None:
            state = Path(env["HOME"]) / ".cache/Microsoft"
            state.mkdir(parents=True)
            (state / "deviceid").write_text("id")

        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(production_extras, "_run", side_effect=write_device_id),
            self.assertRaises(SystemExit) as raised,
        ):
            production_extras._check_imports(
                Path("/tmp/python"), ("onnxruntime",), Path(temporary) / "scratch"
            )

        message = str(raised.exception)
        self.assertIn("importing onnxruntime wrote", message)
        self.assertIn("home/.cache/Microsoft/deviceid", message)

    def test_vad_check_leaves_telemetry_to_the_vad_and_downloads_elsewhere(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.dict(
                production_extras.os.environ,
                {
                    "HOME": "/home/example",
                    "XDG_CACHE_HOME": "/home/example/.cache",
                    "ORT_DISABLE_TELEMETRY": "1",
                    "PATH": "/usr/bin",
                },
                clear=True,
            ),
            mock.patch.object(production_extras, "_run") as run,
        ):
            scratch = Path(temporary) / "cpu-vad"
            models = Path(temporary) / "huggingface"
            python = Path("/tmp/extra-cpu/bin/python")
            production_extras._check_vad(python, scratch, models)

        run.assert_called_once_with(
            [str(python), "-m", "local_dictation.vad", "--check"],
            env={
                "PATH": "/usr/bin",
                "HOME": str(scratch / "home"),
                "TMPDIR": str(scratch / "tmp"),
                "HF_HOME": str(models),
            },
        )

    def test_vad_check_that_leaves_files_behind_fails(self) -> None:
        def write_session(_command: list[str], *, env: dict[str, str]) -> None:
            (Path(env["TMPDIR"]) / ".ses").write_text("session")

        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(production_extras, "_run", side_effect=write_session),
            self.assertRaises(SystemExit) as raised,
        ):
            production_extras._check_vad(
                Path("/tmp/python"),
                Path(temporary) / "scratch",
                Path(temporary) / "huggingface",
            )

        self.assertIn("the VAD check wrote tmp/.ses", str(raised.exception))

    def test_missing_entry_point_fails_the_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            venv = Path(temporary)
            (venv / "bin").mkdir()
            with self.assertRaises(SystemExit) as raised:
                production_extras._check_commands(
                    venv, production_extras.EXTRA_SPECS["cpu"].commands
                )
        self.assertIn("local-dictation-server is missing", str(raised.exception))

    def test_main_uses_a_local_dictation_scratch_directory(self) -> None:
        with (
            mock.patch.object(
                production_extras.tempfile,
                "TemporaryDirectory",
                wraps=tempfile.TemporaryDirectory,
            ) as temporary,
            mock.patch.object(production_extras, "check_environments") as check,
        ):
            result = production_extras.main(["--python", "3.11", "--extra", "cpu"])

        self.assertEqual(result, 0)
        self.assertTrue(
            temporary.call_args.kwargs["prefix"].startswith("local-dictation-")
        )
        self.assertEqual(check.call_args.kwargs["extras"], ("cpu",))
        self.assertEqual(check.call_args.kwargs["python"], "3.11")


if __name__ == "__main__":
    unittest.main()
