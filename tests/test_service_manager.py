from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest import mock

from local_dictation import service_manager
from local_dictation.config import SERVICE_NAME
from local_dictation.errors import DictationError
from local_dictation.service_units import packaged_unit_text
from local_dictation.user_config import default_user_config

SHIPPED = (service_manager.PROJECT_ROOT / "systemd" / "user" / SERVICE_NAME).read_text()
OK = subprocess.CompletedProcess([], 0, "", "")


class ServiceManagerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.systemd = Path(temporary.name) / "systemd" / "user"
        self.destination = self.systemd / SERVICE_NAME
        directory = mock.patch.object(service_manager, "SYSTEMD_USER_DIR", self.systemd)
        directory.start()
        self.addCleanup(directory.stop)

    @contextlib.contextmanager
    def systemctl(
        self, *results: subprocess.CompletedProcess[str]
    ) -> Iterator[mock.MagicMock]:
        responses = iter(results)
        with mock.patch.object(
            service_manager,
            "systemctl",
            side_effect=lambda *_arguments, check=True: next(responses, OK),
        ) as systemctl:
            yield systemctl

    def calls(self, systemctl: mock.MagicMock) -> list[tuple[str, ...]]:
        return [call.args for call in systemctl.call_args_list]

    def output(self, action: Callable[[], object]) -> str:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            action()
        return stdout.getvalue()


class UnitTextTests(ServiceManagerTestCase):
    def test_checkout_and_packaged_copies_are_identical(self) -> None:
        self.assertEqual(service_manager.unit_text(), SHIPPED)
        self.assertEqual(packaged_unit_text(), SHIPPED)

    def test_falls_back_to_the_packaged_copy_outside_a_checkout(self) -> None:
        with mock.patch.object(
            service_manager, "PROJECT_ROOT", Path("/definitely/not/a/checkout")
        ):
            self.assertEqual(service_manager.unit_text(), packaged_unit_text())


class InstallTests(ServiceManagerTestCase):
    def test_installs_reloads_and_enables_only_this_unit(self) -> None:
        with self.systemctl() as systemctl:
            output = self.output(service_manager.install_service)

        self.assertEqual(self.destination.read_text(), SHIPPED)
        self.assertEqual(
            self.calls(systemctl), [("daemon-reload",), ("enable", SERVICE_NAME)]
        )
        self.assertIn("dictate services start", output)
        self.assertNotIn("disable --now", output)

    def test_identical_unit_is_left_in_place_but_still_enabled(self) -> None:
        self.systemd.mkdir(parents=True)
        self.destination.write_text(SHIPPED)
        before = self.destination.stat()

        with self.systemctl() as systemctl:
            output = self.output(service_manager.install_service)

        after = self.destination.stat()
        self.assertEqual(
            (before.st_ino, before.st_mtime_ns), (after.st_ino, after.st_mtime_ns)
        )
        self.assertIn("already up to date", output)
        self.assertEqual(
            self.calls(systemctl), [("daemon-reload",), ("enable", SERVICE_NAME)]
        )

    def test_differing_unit_requires_force(self) -> None:
        self.systemd.mkdir(parents=True)
        self.destination.write_text("[Service]\nExecStart=/custom\n")

        with (
            self.systemctl() as systemctl,
            self.assertRaisesRegex(DictationError, "dictate services install --force"),
        ):
            service_manager.install_service()

        self.assertEqual(self.destination.read_text(), "[Service]\nExecStart=/custom\n")
        systemctl.assert_not_called()

        with self.systemctl():
            output = self.output(lambda: service_manager.install_service(force=True))

        self.assertEqual(self.destination.read_text(), SHIPPED)
        self.assertIn("Replaced", output)

    def test_force_replaces_a_symlink_without_touching_its_target(self) -> None:
        self.systemd.mkdir(parents=True)
        target = self.systemd.parent / "elsewhere.service"
        target.write_text("[Service]\nExecStart=/custom\n")
        self.destination.symlink_to(target)

        with self.assertRaises(DictationError), self.systemctl():
            service_manager.install_service()
        with self.systemctl():
            self.output(lambda: service_manager.install_service(force=True))

        self.assertFalse(self.destination.is_symlink())
        self.assertEqual(self.destination.read_text(), SHIPPED)
        self.assertEqual(target.read_text(), "[Service]\nExecStart=/custom\n")

    def test_refuses_to_replace_a_directory_even_with_force(self) -> None:
        self.destination.mkdir(parents=True)
        (self.destination / "keep").write_text("data")

        for force in (False, True):
            with (
                self.subTest(force=force),
                self.systemctl() as systemctl,
                self.assertRaisesRegex(DictationError, "refusing to replace directory"),
            ):
                service_manager.install_service(force=force)
            systemctl.assert_not_called()

        self.assertEqual((self.destination / "keep").read_text(), "data")

    def test_leaves_legacy_and_other_units_alone(self) -> None:
        self.systemd.mkdir(parents=True)
        legacy = self.systemd / "dictation.service"
        legacy.write_text("legacy dictation\n")
        other = self.systemd / "other.service"
        other.write_text("other\n")

        with self.systemctl() as systemctl:
            installed = self.output(lambda: service_manager.install_service(force=True))
            self.output(service_manager.uninstall_service)

        self.assertIn(f"Left {legacy} as it is", installed)
        self.assertIn("systemctl --user disable --now dictation.service", installed)
        self.assertEqual(legacy.read_text(), "legacy dictation\n")
        self.assertEqual(other.read_text(), "other\n")
        for arguments in self.calls(systemctl):
            self.assertTrue(set(arguments[1:]) <= {SERVICE_NAME}, arguments)

    def test_systemctl_failure_is_reported_with_its_message(self) -> None:
        failed = subprocess.CompletedProcess(
            [], 1, "", "Failed to connect to bus: No medium found\n"
        )
        with (
            self.systemctl(failed),
            self.assertRaisesRegex(
                DictationError,
                "systemctl --user daemon-reload failed: Failed to connect to bus",
            ),
        ):
            self.output(service_manager.install_service)


class LifecycleTests(ServiceManagerTestCase):
    def test_start_stop_and_restart_target_only_this_unit(self) -> None:
        for action, verb in (
            (service_manager.start_service, "start"),
            (service_manager.stop_service, "stop"),
            (service_manager.restart_service, "restart"),
        ):
            with self.subTest(verb=verb), self.systemctl() as systemctl:
                output = self.output(action)

                self.assertEqual(self.calls(systemctl), [(verb, SERVICE_NAME)])
                self.assertIn(SERVICE_NAME, output)

    def test_lifecycle_failures_raise_with_the_systemctl_message(self) -> None:
        missing = subprocess.CompletedProcess(
            [], 5, "", f"Failed to start {SERVICE_NAME}: Unit not found.\n"
        )
        with (
            self.systemctl(missing),
            self.assertRaisesRegex(DictationError, "Unit not found"),
        ):
            service_manager.start_service()

    def test_status_reports_active_and_enabled_state(self) -> None:
        with self.systemctl(
            subprocess.CompletedProcess([], 0, "active\n", ""),
            subprocess.CompletedProcess([], 0, "enabled\n", ""),
        ) as systemctl:
            output = self.output(service_manager.status)

        self.assertEqual(output, f"{SERVICE_NAME}: active (enabled)\n")
        self.assertEqual(
            self.calls(systemctl),
            [("is-active", SERVICE_NAME), ("is-enabled", SERVICE_NAME)],
        )

    def test_status_of_a_unit_that_is_not_installed(self) -> None:
        with self.systemctl(
            subprocess.CompletedProcess([], 4, "inactive\n", ""),
            subprocess.CompletedProcess([], 4, "", "Failed to get unit file state"),
        ):
            output = self.output(service_manager.status)

        self.assertEqual(output, f"{SERVICE_NAME}: inactive (not-found)\n")

    def test_audit_passes_the_loaded_configuration(self) -> None:
        config = default_user_config()
        with (
            mock.patch.object(service_manager, "load_user_config", return_value=config),
            mock.patch.object(
                service_manager, "audit_installed", return_value=1
            ) as audit,
        ):
            self.assertEqual(service_manager.audit_service(), 1)

        audit.assert_called_once_with(config=config)

    def test_logs_reads_this_units_journal_and_exits_with_its_status(self) -> None:
        for follow, expected in (
            (False, ["journalctl", "--user", "-u", SERVICE_NAME, "-n", "40"]),
            (True, ["journalctl", "--user", "-u", SERVICE_NAME, "-n", "40", "-f"]),
        ):
            with (
                self.subTest(follow=follow),
                mock.patch.object(
                    service_manager.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 3),
                ) as run,
                self.assertRaises(SystemExit) as exited,
            ):
                service_manager.logs(follow=follow, lines=40)

            run.assert_called_once_with(expected, check=False)
            self.assertEqual(exited.exception.code, 3)


class UninstallTests(ServiceManagerTestCase):
    def test_stops_disables_and_removes_the_shipped_unit(self) -> None:
        self.systemd.mkdir(parents=True)
        self.destination.write_text(SHIPPED)

        with self.systemctl() as systemctl:
            output = self.output(service_manager.uninstall_service)

        self.assertFalse(self.destination.exists())
        self.assertIn("Removed", output)
        self.assertEqual(
            self.calls(systemctl),
            [("stop", SERVICE_NAME), ("disable", SERVICE_NAME), ("daemon-reload",)],
        )

    def test_keeps_a_customized_unit(self) -> None:
        self.systemd.mkdir(parents=True)
        self.destination.write_text("[Service]\nExecStart=/custom\n")

        with self.systemctl():
            output = self.output(service_manager.uninstall_service)

        self.assertEqual(self.destination.read_text(), "[Service]\nExecStart=/custom\n")
        self.assertIn("Kept", output)

    def test_is_idempotent_when_nothing_is_installed(self) -> None:
        not_loaded = subprocess.CompletedProcess([], 5, "", "Unit not loaded.")
        with self.systemctl(not_loaded, not_loaded) as systemctl:
            output = self.output(service_manager.uninstall_service)

        self.assertIn("is not installed", output)
        self.assertEqual(self.calls(systemctl)[-1], ("daemon-reload",))


class ServerEntryPointTests(unittest.TestCase):
    def test_service_executable_is_installed_and_passes_its_check(self) -> None:
        server = Path(sys.executable).parent / "local-dictation-server"
        process = subprocess.run(
            [str(server), "--check"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout, "local-dictation-server: ok\n")


if __name__ == "__main__":
    unittest.main()
