from __future__ import annotations

import contextlib
import io
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_dictation import platform_services, service_units
from local_dictation.config import SERVICE_NAME
from local_dictation.user_config import default_user_config

SHIPPED_UNIT = service_units.PROJECT_ROOT / service_units.SOURCE_RELATIVE / SERVICE_NAME
MINIMAL_PYPROJECT = """\
[project]
name = "local-dictation"

[project.scripts]
local-dictation-server = "local_dictation.stt.launcher:main"

[project.optional-dependencies]
cpu = ["onnx-asr[hub]>=0.7.0", "onnxruntime>=1.20"]
cuda = ["onnx-asr[gpu,hub]>=0.7.0"]
"""


def _project(root: Path, *, unit: str | None = None, pyproject: str = "") -> Path:
    """Stage a scratch project root with the shipped unit and a pyproject."""

    source = root / service_units.SOURCE_RELATIVE
    shutil.copytree(SHIPPED_UNIT.parent, source)
    if unit is not None:
        (source / SERVICE_NAME).write_text(unit)
    (root / "pyproject.toml").write_text(pyproject or MINIMAL_PYPROJECT)
    return root


def _security_errors(unit: str) -> list[str]:
    with tempfile.TemporaryDirectory() as temporary:
        return service_units.security_errors(_project(Path(temporary), unit=unit))


def _consistency_errors(*, unit: str | None = None, pyproject: str = "") -> list[str]:
    with tempfile.TemporaryDirectory() as temporary:
        root = _project(Path(temporary), unit=unit, pyproject=pyproject)
        return service_units.consistency_errors(root)


class CiWorkflowTests(unittest.TestCase):
    def test_ci_runs_staged_verifier_with_pinned_actions(self) -> None:
        workflows = sorted(
            (service_units.PROJECT_ROOT / ".github" / "workflows").glob("*.y*ml")
        )
        if not workflows:
            self.skipTest("this checkout has no CI workflows")
        verifier = re.compile(
            r"python -m local_dictation\.service_units\s*(?:\\\s*)?"
            r"--require-systemd-analyze"
        )
        running = [path for path in workflows if verifier.search(path.read_text())]
        self.assertTrue(running, "no CI workflow runs the staged unit verifier")
        for path in running:
            workflow = path.read_text()
            self.assertNotIn("--audit-installed", workflow)
            action_lines = re.findall(
                r"^\s*(?:-\s*)?uses:\s*(\S+)\s+#\s*(\S+)$", workflow, re.M
            )
            self.assertEqual(len(action_lines), workflow.count("uses:"))
            for reference, version in action_lines:
                with self.subTest(workflow=path.name, action=reference):
                    self.assertRegex(reference, r"^[^@]+@[0-9a-f]{40}$")
                    self.assertRegex(version, r"^v\d")


class ShippedUnitTests(unittest.TestCase):
    def test_shipped_unit_has_parity_consistency_and_policy(self) -> None:
        self.assertEqual(service_units.parity_errors(), [])
        self.assertEqual(service_units.consistency_errors(), [])
        self.assertEqual(service_units.security_errors(), [])

    def test_unit_runs_the_server_from_the_service_environment(self) -> None:
        unit = SHIPPED_UNIT.read_text()

        self.assertEqual(service_units.INSTALL_ROOT, "%h/local-dictation")
        self.assertEqual(
            service_units.EXPECTED_EXECSTART,
            "%h/local-dictation/.venv-service/bin/local-dictation-server",
        )
        self.assertIn("Description=Local dictation speech-to-text server\n", unit)
        self.assertIn("WantedBy=default.target\n", unit)
        self.assertEqual(
            service_units.REQUIRED_ENVIRONMENT,
            {
                "CUDA_CACHE_PATH": "%t/local-dictation/cuda-cache",
                "HF_HOME": "%h/.cache/huggingface",
                "TMPDIR": "%t/local-dictation",
            },
        )
        self.assertNotIn("DICTATION_SOCKET", unit)

    def test_policy_keeps_the_shared_runtime_directory_and_resource_bounds(
        self,
    ) -> None:
        policy = service_units.SECURITY_POLICY

        self.assertEqual(policy["RuntimeDirectory"], "local-dictation")
        self.assertEqual(policy["RuntimeDirectoryMode"], "0700")
        self.assertEqual(policy["RuntimeDirectoryPreserve"], "yes")
        self.assertEqual(policy["ReadWritePaths"], "%t")
        self.assertEqual(policy["ProtectHome"], "read-only")
        self.assertEqual((policy["MemoryHigh"], policy["MemoryMax"]), ("4G", "6G"))
        self.assertTrue(
            service_units.REQUIRED_ENVIRONMENT["CUDA_CACHE_PATH"].startswith(
                f"%t/{policy['RuntimeDirectory']}/"
            )
        )
        self.assertFalse(service_units.CUDA_DEVICE_RESTRICTIONS & set(policy))
        self.assertIn("DevicePolicy", service_units.INTENTIONALLY_UNUSED_DIRECTIVES)


class ParityTests(unittest.TestCase):
    def test_reports_unexpected_and_differing_source_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _project(
                Path(temporary), unit=SHIPPED_UNIT.read_text() + "# edited\n"
            )
            (root / service_units.SOURCE_RELATIVE / "dictation.service").write_text(
                "[Unit]\n"
            )

            errors = service_units.parity_errors(root)

        self.assertIn("source units unexpected: dictation.service", errors)
        self.assertIn(f"source and packaged units differ: {SERVICE_NAME}", errors)

    def test_reports_missing_source_unit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _project(Path(temporary))
            (root / service_units.SOURCE_RELATIVE / SERVICE_NAME).unlink()

            errors = service_units.parity_errors(root)

        self.assertEqual(errors, [f"source units missing: {SERVICE_NAME}"])


class SecurityPolicyTests(unittest.TestCase):
    def test_rejects_weakened_or_incompatible_unit(self) -> None:
        errors = _security_errors(
            SHIPPED_UNIT.read_text()
            .replace("MemoryMax=6G", "MemoryMax=infinity")
            .replace(
                "ProtectSystem=strict",
                "ProtectSystem=strict\nPrivateDevices=true\n"
                "SystemCallFilter=@system-service",
            )
        )

        self.assertIn(f"{SERVICE_NAME} MemoryMax must be '6G', got 'infinity'", errors)
        self.assertIn(
            f"{SERVICE_NAME} must not restrict CUDA devices with PrivateDevices",
            errors,
        )
        self.assertIn(
            f"{SERVICE_NAME} must not set compatibility-sensitive SystemCallFilter",
            errors,
        )

    def test_rejects_duplicates_resets_and_last_overrides(self) -> None:
        errors = _security_errors(
            SHIPPED_UNIT.read_text()
            .replace("MemoryMax=6G", "MemoryMax=6G\nMemoryMax=6G")
            .replace(
                "ProtectHome=read-only",
                "ProtectHome=read-only\nProtectHome=\nProtectHome=false",
            )
            .replace(
                "NoNewPrivileges=true", "NoNewPrivileges=true\nNoNewPrivileges=false"
            )
            .replace(
                "Environment=CUDA_CACHE_PATH=%t/local-dictation/cuda-cache",
                "Environment=CUDA_CACHE_PATH=%t/local-dictation/cuda-cache\n"
                "Environment=\n"
                "Environment=CUDA_CACHE_PATH=/tmp/cuda",
            )
        )

        self.assertIn(
            f"{SERVICE_NAME} MemoryMax must have exactly one assignment, "
            "got ['6G', '6G']",
            errors,
        )
        self.assertIn(
            f"{SERVICE_NAME} ProtectHome must have exactly one assignment, "
            "got ['read-only', '', 'false']",
            errors,
        )
        self.assertIn(
            f"{SERVICE_NAME} NoNewPrivileges must have exactly one assignment, "
            "got ['true', 'false']",
            errors,
        )
        self.assertIn(
            f"{SERVICE_NAME} Environment=CUDA_CACHE_PATH must be assigned exactly "
            "once to '%t/local-dictation/cuda-cache', got "
            "['%t/local-dictation/cuda-cache', '<reset>', '/tmp/cuda']",
            errors,
        )

    def test_rejects_unrestricted_environment_files(self) -> None:
        errors = _security_errors(
            SHIPPED_UNIT.read_text().replace(
                "[Service]", "[Service]\nEnvironmentFile=-%h/.config/unsafe.env"
            )
        )

        self.assertIn(
            f"{SERVICE_NAME} must not import unrestricted EnvironmentFile values",
            errors,
        )

    def test_rejects_user_settings_and_socket_overrides_in_the_unit(self) -> None:
        errors = _security_errors(
            SHIPPED_UNIT.read_text().replace(
                "[Service]",
                "[Service]\nEnvironment=DICTATION_DEVICE=cuda\n"
                "Environment=DICTATION_SOCKET=%t/dictation.sock",
            )
        )

        self.assertIn(
            f"{SERVICE_NAME} Environment keys must be exactly "
            "['CUDA_CACHE_PATH', 'HF_HOME', 'TMPDIR'], got "
            "['CUDA_CACHE_PATH', 'DICTATION_DEVICE', 'DICTATION_SOCKET', "
            "'HF_HOME', 'TMPDIR']",
            errors,
        )

    def test_rejects_unsandboxed_replaced_or_extended_execstart(self) -> None:
        expected = service_units.EXPECTED_EXECSTART
        for replacement in (
            f"ExecStart=+{expected}",
            f"ExecStart={expected} --listen 0.0.0.0",
            f"ExecStart={expected}\nExecStart=\nExecStart=/usr/bin/python -m x",
            f"ExecStart={expected}\nExecStart={expected}",
        ):
            with self.subTest(replacement=replacement):
                errors = _security_errors(
                    SHIPPED_UNIT.read_text().replace(
                        f"ExecStart={expected}", replacement
                    )
                )
                self.assertTrue(
                    any("effective ExecStart must be" in error for error in errors),
                    errors,
                )


class ConsistencyTests(unittest.TestCase):
    def test_minimal_project_is_consistent(self) -> None:
        self.assertEqual(_consistency_errors(), [])

    def test_requires_the_server_entry_point_the_unit_executes(self) -> None:
        errors = _consistency_errors(
            pyproject=MINIMAL_PYPROJECT.replace(
                "local_dictation.stt.launcher:main", "local_dictation.stt.server:main"
            )
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("local-dictation-server", errors[0])

    def test_requires_service_extras_to_install_the_default_backend(self) -> None:
        errors = _consistency_errors(
            pyproject=MINIMAL_PYPROJECT.replace(
                'cuda = ["onnx-asr[gpu,hub]>=0.7.0"]', 'cuda = ["onnxruntime-gpu"]'
            )
        )

        self.assertEqual(
            errors, ["cuda extra does not install the default parakeet backend"]
        )

    def test_runtime_directory_must_be_the_one_client_and_server_share(self) -> None:
        errors = _consistency_errors(
            unit=SHIPPED_UNIT.read_text().replace(
                "RuntimeDirectory=local-dictation", "RuntimeDirectory=dictation"
            )
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("RuntimeDirectory", errors[0])
        self.assertIn("'dictation'", errors[0])

    def test_environment_must_match_what_the_launcher_forces(self) -> None:
        errors = _consistency_errors(
            unit=SHIPPED_UNIT.read_text().replace(
                "Environment=TMPDIR=%t/local-dictation", "Environment=TMPDIR=%t/tmp"
            )
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("Environment=TMPDIR", errors[0])


class StageExecutableStubTests(unittest.TestCase):
    def test_stages_normal_absolute_executable_inside_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            root.mkdir()

            staged = service_units.stage_executable_stub(root, "/usr/local/bin/tool")

            self.assertEqual(staged, root.resolve() / "usr/local/bin/tool")
            self.assertTrue(staged.is_file())
            self.assertTrue(staged.stat().st_mode & 0o111)

    def test_rejects_lexical_traversal_and_non_normalized_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "root"
            root.mkdir()
            outside = base / "outside"

            for executable in (
                "relative/tool",
                "/../outside/tool",
                "/usr/../outside/tool",
                "/usr/./bin/tool",
                "/usr//bin/tool",
            ):
                with (
                    self.subTest(executable=executable),
                    self.assertRaisesRegex(ValueError, "unsafe staged executable"),
                ):
                    service_units.stage_executable_stub(root, executable)

            self.assertFalse(outside.exists())

    def test_rejects_parent_and_destination_symlink_escapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            outside = base / "outside"
            outside.mkdir()

            parent_root = base / "parent-root"
            parent_root.mkdir()
            (parent_root / "usr").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                service_units.stage_executable_stub(parent_root, "/usr/bin/tool")
            self.assertFalse((outside / "bin/tool").exists())

            destination_root = base / "destination-root"
            (destination_root / "usr/bin").mkdir(parents=True)
            outside_target = outside / "tool"
            (destination_root / "usr/bin/tool").symlink_to(outside_target)
            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                service_units.stage_executable_stub(destination_root, "/usr/bin/tool")
            self.assertFalse(outside_target.exists())


class SystemdAnalyzeTests(unittest.TestCase):
    def test_is_optional_when_not_installed(self) -> None:
        with mock.patch.object(service_units.shutil, "which", return_value=None):
            self.assertIsNone(service_units.systemd_analyze([SHIPPED_UNIT]))

    def test_stages_the_server_executable_and_dependency_stubs(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "", "")

        def inspect_staging(
            arguments: list[str], **options: object
        ) -> subprocess.CompletedProcess[str]:
            self.assertEqual(arguments[:2], ["/tool", "--user"])
            root = Path(arguments[2].removeprefix("--root="))
            self.assertEqual(
                arguments[3:], ["--man=no", "--generators=no", "verify", SERVICE_NAME]
            )
            self.assertEqual(
                options, {"capture_output": True, "text": True, "check": False}
            )
            server = Path(
                service_units.EXPECTED_EXECSTART.replace("%h", str(Path.home()))
            )
            staged = root / server.relative_to("/")
            self.assertTrue(staged.is_file())
            self.assertTrue(staged.stat().st_mode & 0o111)
            for name in service_units.EXTERNAL_UNIT_STUBS:
                self.assertTrue((root / "etc/systemd/user" / name).is_file())
            staged_unit = (root / "etc/systemd/user" / SERVICE_NAME).read_text()
            self.assertTrue(staged_unit.startswith(SHIPPED_UNIT.read_text()))
            self.assertTrue(staged_unit.endswith("DefaultDependencies=no\n"))
            return completed

        with mock.patch.object(
            service_units.subprocess, "run", side_effect=inspect_staging
        ):
            result = service_units.systemd_analyze([SHIPPED_UNIT], executable="/tool")

        assert result is not None
        self.assertIs(result.process, completed)
        self.assertEqual(result.context, "staged-user")
        self.assertEqual(result.rooted_user_context, "supported")

    def test_does_not_stub_an_invented_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / SERVICE_NAME
            source.write_text(
                SHIPPED_UNIT.read_text().replace(
                    "bin/local-dictation-server", "bin/local-dictation-sever"
                )
            )

            def inspect_staging(
                arguments: list[str], **_options: object
            ) -> subprocess.CompletedProcess[str]:
                root = Path(arguments[2].removeprefix("--root="))
                self.assertFalse(
                    any(path.is_file() for path in root.rglob("local-dictation-se*"))
                )
                return subprocess.CompletedProcess([], 1, "", "missing executable")

            with mock.patch.object(
                service_units.subprocess, "run", side_effect=inspect_staging
            ):
                result = service_units.systemd_analyze([source], executable="/tool")

        assert result is not None
        self.assertNotEqual(result.returncode, 0)

    def test_unsupported_user_context_markers(self) -> None:
        self.assertTrue(
            service_units.user_context_unsupported(
                "Failed to lookup RuntimeDirectory path: No such device or address\n"
                "Failed to initialize manager: No such device or address\n"
            )
        )
        self.assertTrue(
            service_units.user_context_unsupported("Failed to connect to bus")
        )
        self.assertFalse(service_units.user_context_unsupported("missing executable"))

    def test_falls_back_to_system_context_when_rooted_user_mode_fails(self) -> None:
        for reason in (
            "Failed to lookup RuntimeDirectory path: No such device or address",
            "Failed to initialize unit search paths for root directory "
            "/tmp/root: Invalid argument",
        ):
            with self.subTest(reason=reason):
                unsupported = subprocess.CompletedProcess([], 1, "", reason)
                completed = subprocess.CompletedProcess([], 0, "", "")
                with mock.patch.object(
                    service_units.subprocess,
                    "run",
                    side_effect=[unsupported, completed],
                ) as run:
                    result = service_units.systemd_analyze(
                        [SHIPPED_UNIT], executable="/tool"
                    )

                assert result is not None
                self.assertIs(result.process, completed)
                self.assertEqual(result.context, "staged-system")
                self.assertEqual(result.rooted_user_context, "unsupported")
                self.assertEqual(result.reason, reason)
                self.assertEqual(run.call_args_list[0].args[0][1], "--user")
                self.assertTrue(run.call_args_list[1].args[0][1].startswith("--root="))

    @unittest.skipUnless(shutil.which("systemd-analyze"), "systemd-analyze unavailable")
    def test_shipped_unit_passes_staged_verification(self) -> None:
        result = service_units.systemd_analyze([SHIPPED_UNIT])

        assert result is not None
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which("systemd-analyze"), "systemd-analyze unavailable")
    def test_rejects_genuine_unit_syntax_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary) / "invalid.service"
            invalid.write_text(
                "[Unit]\n"
                "Description=Invalid fixture\n"
                "[Service]\n"
                "Type=oneshot\n"
                "ExecStart=relative/path/is/not/valid\n"
            )

            result = service_units.systemd_analyze([invalid])

        assert result is not None
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "Neither a valid executable name nor an absolute path", result.stderr
        )

    @unittest.skipUnless(shutil.which("systemd-analyze"), "systemd-analyze unavailable")
    def test_rejects_unknown_required_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary) / "invalid-dependency.service"
            invalid.write_text(
                "[Unit]\n"
                "Description=Invalid dependency fixture\n"
                "Requires=misspelled-runtime.service\n"
                "[Service]\n"
                "Type=oneshot\n"
                "ExecStart=/missing/on/clean/runner\n"
            )

            result = service_units.systemd_analyze([invalid])

        assert result is not None
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("misspelled-runtime.service", result.stderr)


class MainTests(unittest.TestCase):
    def run_main(self, arguments: list[str]) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = service_units.main(arguments)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_validation_errors_fail_before_systemd_analyze(self) -> None:
        with (
            mock.patch.object(service_units, "parity_errors", return_value=["bad"]),
            mock.patch.object(service_units, "systemd_analyze") as analyze,
        ):
            result, _stdout, stderr = self.run_main(["--require-systemd-analyze"])

        self.assertEqual(result, 1)
        self.assertIn("error: bad", stderr)
        analyze.assert_not_called()

    def test_missing_systemd_analyze_fails_only_when_required(self) -> None:
        with mock.patch.object(service_units, "systemd_analyze", return_value=None):
            optional = self.run_main([])
            required = self.run_main(["--require-systemd-analyze"])

        self.assertEqual(optional[0], 0)
        self.assertIn("skipped", optional[1])
        self.assertEqual(required[0], 1)
        self.assertIn("systemd-analyze is unavailable", required[2])

    def test_require_user_context_fails_on_honest_unsupported_result(self) -> None:
        verification = service_units.StagedVerification(
            subprocess.CompletedProcess([], 0, "", ""),
            context="staged-system",
            rooted_user_context="unsupported",
            reason="Invalid argument",
        )
        with mock.patch.object(
            service_units, "systemd_analyze", return_value=verification
        ):
            result, stdout, stderr = self.run_main(
                ["--require-systemd-analyze", "--require-user-context"]
            )

        self.assertEqual(result, 1)
        self.assertIn('"context": "staged-system"', stdout)
        self.assertIn("user-context verification is unsupported", stderr)

    def test_verification_failure_is_reported(self) -> None:
        verification = service_units.StagedVerification(
            subprocess.CompletedProcess([], 1, "", "Command is not executable"),
            context="staged-user",
            rooted_user_context="supported",
        )
        with mock.patch.object(
            service_units, "systemd_analyze", return_value=verification
        ):
            result, _stdout, stderr = self.run_main(["--require-systemd-analyze"])

        self.assertEqual(result, 1)
        self.assertIn("Command is not executable", stderr)

    def test_audit_installed_flag_delegates_to_the_read_only_audit(self) -> None:
        with mock.patch.object(
            service_units, "audit_installed", return_value=1
        ) as audit:
            result, _stdout, _stderr = self.run_main(["--audit-installed"])

        self.assertEqual(result, 1)
        audit.assert_called_once_with()


class InstalledAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        home = Path(temporary.name) / "home"
        runtime = Path(temporary.name) / "runtime"
        home.mkdir()
        runtime.mkdir()
        environment = mock.patch.dict(
            os.environ, {"HOME": str(home), "XDG_RUNTIME_DIR": str(runtime)}
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.server = Path(
            service_units._expanded_unit_value(service_units.EXPECTED_EXECSTART)
        )
        self.server.parent.mkdir(parents=True)
        self.server.write_text("#!/bin/sh\n")
        self.server.chmod(0o755)

    def fixture(
        self,
        *,
        environment: dict[str, str] | None = None,
        unit_suffix: str = "",
        property_overrides: dict[str, str] | None = None,
        omit: tuple[str, ...] = (),
    ) -> list[subprocess.CompletedProcess[str]]:
        assignments = (
            service_units.REQUIRED_ENVIRONMENT if environment is None else environment
        )
        properties = {
            **service_units._audit_expected_properties(),
            "ActiveState": "inactive",
            "SubState": "dead",
            "Result": "success",
            "NRestarts": "0",
            "ExecMainStatus": "0",
            "MemoryCurrent": "[not set]",
            "TasksCurrent": "[not set]",
            "Environment": " ".join(
                f"{key}={service_units._expanded_unit_value(value)}"
                for key, value in assignments.items()
            ),
            "ExecStart": (
                f"{{ path={self.server} ; argv[]={self.server} ; ignore_errors=no ; "
                "start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; "
                "status=0/0 }"
            ),
            **(property_overrides or {}),
        }
        shown = "\n".join(
            f"{key}={value}" for key, value in properties.items() if key not in omit
        )
        return [
            subprocess.CompletedProcess(
                [], 0, SHIPPED_UNIT.read_text() + unit_suffix, ""
            ),
            subprocess.CompletedProcess([], 0, shown, ""),
        ]

    def audit(
        self, completed: list[subprocess.CompletedProcess[str]]
    ) -> tuple[int, str, str, mock.MagicMock]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(
                platform_services.subprocess, "run", side_effect=completed
            ) as run,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = service_units.audit_installed(config=default_user_config())
        return result, stdout.getvalue(), stderr.getvalue(), run

    def test_expected_properties_cover_every_non_observational_property(
        self,
    ) -> None:
        observational = {
            "ActiveState",
            "SubState",
            "Result",
            "NRestarts",
            "ExecMainStatus",
            "MemoryCurrent",
            "TasksCurrent",
            "Environment",
            "ExecStart",
        }
        self.assertEqual(
            set(service_units._audit_expected_properties()),
            set(service_units.AUDIT_SHOW_PROPERTIES) - observational,
        )

    def test_reads_effective_unit_and_runtime_state(self) -> None:
        result, stdout, stderr, run = self.audit(self.fixture())

        self.assertEqual(result, 0, stderr)
        self.assertEqual(
            run.call_args_list[0].args[0],
            ["systemctl", "--user", "cat", SERVICE_NAME],
        )
        show = run.call_args_list[1].args[0]
        self.assertEqual(show[:4], ["systemctl", "--user", "show", SERVICE_NAME])
        self.assertIn("--property=EnvironmentFiles", show)
        self.assertIn(f"{SERVICE_NAME}: active=inactive sub=dead", stdout)
        self.assertIn(
            "configuration: device=auto backend=parakeet "
            "model=nemo-parakeet-tdt-0.6b-v2",
            stdout,
        )

    def test_empty_properties_that_systemd_leaves_out_count_as_empty(self) -> None:
        result, _stdout, stderr, _run = self.audit(
            self.fixture(omit=("EnvironmentFiles", "InaccessiblePaths"))
        )

        self.assertEqual(result, 0, stderr)

    def test_missing_required_property_still_fails(self) -> None:
        result, _stdout, stderr, _run = self.audit(
            self.fixture(omit=("ProtectSystem",))
        )

        self.assertEqual(result, 1)
        self.assertIn("effective ProtectSystem must be 'strict', got None", stderr)

    def test_unit_that_is_not_installed_fails_without_reading_properties(
        self,
    ) -> None:
        missing = subprocess.CompletedProcess(
            [], 1, "", f"No files found for {SERVICE_NAME}."
        )
        result, _stdout, stderr, run = self.audit([missing])

        self.assertEqual(result, 1)
        self.assertEqual(run.call_count, 1)
        self.assertIn("systemctl cat failed: No files found", stderr)

    def test_reports_a_missing_service_environment(self) -> None:
        self.server.unlink()

        result, _stdout, stderr, _run = self.audit(self.fixture())

        self.assertEqual(result, 1)
        self.assertIn("missing or not executable", stderr)
        self.assertIn("scripts/install.sh", stderr)

    def test_healthy_restart_history_is_observational(self) -> None:
        result, _stdout, stderr, _run = self.audit(
            self.fixture(
                property_overrides={
                    "ActiveState": "active",
                    "SubState": "running",
                    "NRestarts": "5",
                }
            )
        )

        self.assertEqual(result, 0, stderr)

    def test_rejects_active_restart_loop_and_failed_state(self) -> None:
        for overrides, message in (
            (
                {"ActiveState": "activating", "SubState": "auto-restart"},
                "active crash loop",
            ),
            ({"ActiveState": "failed", "Result": "exit-code"}, "unhealthy state"),
        ):
            with self.subTest(message=message):
                result, _stdout, stderr, _run = self.audit(
                    self.fixture(property_overrides={**overrides, "NRestarts": "5"})
                )

                self.assertEqual(result, 1)
                self.assertIn(message, stderr)

    def test_rejects_environment_files(self) -> None:
        result, _stdout, stderr, _run = self.audit(
            self.fixture(
                unit_suffix=(
                    "\n# drop-in\n[Service]\nEnvironmentFile=-%h/.config/unsafe.env\n"
                ),
                property_overrides={
                    "EnvironmentFiles": "/home/test/.config/unsafe.env"
                },
            )
        )

        self.assertEqual(result, 1)
        self.assertIn("effective EnvironmentFile must be empty", stderr)
        self.assertIn("effective EnvironmentFiles must be ''", stderr)

    def test_rejects_user_settings_and_process_injection_overrides(self) -> None:
        overrides = {
            "LD_PRELOAD": "/tmp/inject.so",
            "DICTATION_RUNTIME_DIR": "/tmp/runtime",
            "DICTATION_DEVICE": "cuda",
            "DICTATION_SOCKET": "/tmp/dictation.sock",
        }
        result, _stdout, stderr, _run = self.audit(
            self.fixture(
                environment={**service_units.REQUIRED_ENVIRONMENT, **overrides}
            )
        )

        self.assertEqual(result, 1)
        self.assertIn("unsupported keys", stderr)
        for variable in overrides:
            self.assertIn(variable, stderr)

    def test_rejects_changed_or_missing_required_environment(self) -> None:
        for variable in service_units.REQUIRED_ENVIRONMENT:
            with self.subTest(variable=variable, change="changed"):
                result, _stdout, stderr, _run = self.audit(
                    self.fixture(
                        environment={
                            **service_units.REQUIRED_ENVIRONMENT,
                            variable: "/tmp/elsewhere",
                        }
                    )
                )
                self.assertEqual(result, 1)
                self.assertIn(f"effective {variable} must be", stderr)
            with self.subTest(variable=variable, change="missing"):
                environment = dict(service_units.REQUIRED_ENVIRONMENT)
                del environment[variable]
                result, _stdout, stderr, _run = self.audit(
                    self.fixture(environment=environment)
                )
                self.assertEqual(result, 1)
                self.assertIn(f"lacks required keys ['{variable}']", stderr)

    def test_rejects_effective_weakening_drop_in(self) -> None:
        weakened = f"+{self.server}"
        result, _stdout, stderr, _run = self.audit(
            self.fixture(
                unit_suffix=(
                    "\n# weakening drop-in\n[Service]\n"
                    "NoNewPrivileges=false\nMemoryMax=infinity\nSystemCallFilter=@raw-io\n"
                    f"ExecStart=\nExecStart={weakened}\n"
                ),
                property_overrides={
                    "MemoryMax": "infinity",
                    "NoNewPrivileges": "no",
                    "ExecStart": (
                        f"{{ path={self.server} ; argv[]=/usr/bin/env {self.server} ; "
                        "ignore_errors=no ; }"
                    ),
                },
            )
        )

        self.assertEqual(result, 1)
        self.assertIn("effective MemoryMax must be '6442450944'", stderr)
        self.assertIn("effective NoNewPrivileges must be 'yes'", stderr)
        self.assertIn("effective SystemCallFilter must be empty", stderr)
        self.assertIn("effective ExecStart must be", stderr)
        self.assertIn("effective ExecStart argv must be", stderr)


if __name__ == "__main__":
    unittest.main()
