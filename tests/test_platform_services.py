from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from local_dictation import desktop, platform_services
from local_dictation.diagnostics import checks
from local_dictation.diagnostics.model import Severity
from local_dictation.platform_services import SystemdUserSupervisor, user_services
from local_dictation.user_config import default_user_config

ORCHESTRATION_MODULES = (
    "local_dictation.config_management",
    "local_dictation.desktop",
    "local_dictation.service_manager",
    "local_dictation.diagnostics.checks",
)
UNIT = "local-dictation.service"


def _completed(
    returncode: int = 0, stdout: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, "")


class SystemdUserSupervisorTests(unittest.TestCase):
    def test_lifecycle_commands_target_the_user_manager(self) -> None:
        supervisor = SystemdUserSupervisor()
        with mock.patch.object(
            platform_services.subprocess, "run", return_value=_completed()
        ) as run:
            supervisor.start(UNIT)
            supervisor.stop(UNIT)
            supervisor.enable(UNIT)
            supervisor.disable(UNIT)
            supervisor.restart(UNIT)
            supervisor.reload()

        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [
                ["systemctl", "--user", "start", UNIT],
                ["systemctl", "--user", "stop", UNIT],
                ["systemctl", "--user", "enable", UNIT],
                ["systemctl", "--user", "disable", UNIT],
                ["systemctl", "--user", "restart", UNIT],
                ["systemctl", "--user", "daemon-reload"],
            ],
        )
        self.assertEqual(
            [call.kwargs["check"] for call in run.call_args_list],
            [True, False, True, False, False, True],
        )
        self.assertTrue(all(call.kwargs["text"] for call in run.call_args_list))

    def test_configured_systemctl_executable_is_used(self) -> None:
        supervisor = SystemdUserSupervisor(systemctl="/opt/systemd/bin/systemctl")
        with mock.patch.object(
            platform_services.subprocess, "run", return_value=_completed()
        ) as run:
            supervisor.run("list-units", check=False)

        self.assertEqual(
            run.call_args.args[0],
            ["/opt/systemd/bin/systemctl", "--user", "list-units"],
        )

    def test_is_active_reports_state_and_defaults_to_inactive(self) -> None:
        supervisor = SystemdUserSupervisor()
        for process, expected in (
            (_completed(0, "active\n"), "active"),
            (_completed(3, ""), "inactive"),
        ):
            with self.subTest(expected=expected):
                with mock.patch.object(
                    platform_services.subprocess, "run", return_value=process
                ) as run:
                    self.assertEqual(supervisor.is_active(UNIT), expected)
                self.assertEqual(
                    run.call_args.args[0],
                    ["systemctl", "--user", "is-active", UNIT],
                )
                self.assertFalse(run.call_args.kwargs["check"])

    def test_show_requests_properties_and_parses_assignments(self) -> None:
        supervisor = SystemdUserSupervisor()
        output = "LoadState=loaded\nUnitFileState=\nunparseable line\n"
        with mock.patch.object(
            platform_services.subprocess, "run", return_value=_completed(0, output)
        ) as run:
            properties = supervisor.show(UNIT, ["LoadState", "UnitFileState"])

        self.assertEqual(properties, {"LoadState": "loaded", "UnitFileState": ""})
        self.assertEqual(
            run.call_args.args[0],
            [
                "systemctl",
                "--user",
                "show",
                UNIT,
                "--no-pager",
                "--property=LoadState",
                "--property=UnitFileState",
            ],
        )

    def test_show_failure_reports_no_properties(self) -> None:
        with mock.patch.object(
            platform_services.subprocess,
            "run",
            return_value=_completed(1, "LoadState=loaded\n"),
        ):
            self.assertEqual(SystemdUserSupervisor().show(UNIT, ["LoadState"]), {})

    def test_user_environment_keeps_non_empty_assignments(self) -> None:
        output = "DISPLAY=:0\nEMPTY=\nWAYLAND_DISPLAY=wayland-1\n"
        with mock.patch.object(
            platform_services.subprocess, "run", return_value=_completed(0, output)
        ) as run:
            environment = SystemdUserSupervisor().user_environment()

        self.assertEqual(environment, {"DISPLAY": ":0", "WAYLAND_DISPLAY": "wayland-1"})
        self.assertEqual(
            run.call_args.args[0], ["systemctl", "--user", "show-environment"]
        )

    def test_user_environment_failure_is_empty(self) -> None:
        with mock.patch.object(
            platform_services.subprocess,
            "run",
            return_value=_completed(1, "DISPLAY=:0\n"),
        ):
            self.assertEqual(SystemdUserSupervisor().user_environment(), {})

    def test_availability_requires_the_systemctl_binary(self) -> None:
        supervisor = SystemdUserSupervisor()
        with (
            mock.patch.object(platform_services.shutil, "which", return_value=None),
            mock.patch.object(platform_services.subprocess, "run") as run,
        ):
            self.assertFalse(supervisor.binary_available())
            self.assertFalse(supervisor.available())
        run.assert_not_called()

    def test_systemd_availability_probes_user_bus_with_timeout(self) -> None:
        supervisor = SystemdUserSupervisor()
        with (
            mock.patch.object(
                platform_services.shutil, "which", return_value="/usr/bin/systemctl"
            ),
            mock.patch.object(
                platform_services.subprocess, "run", return_value=_completed()
            ) as run,
        ):
            self.assertTrue(supervisor.binary_available())
            self.assertTrue(supervisor.available())

        self.assertEqual(
            run.call_args.args[0],
            ["systemctl", "--user", "show-environment"],
        )
        self.assertEqual(
            run.call_args.kwargs["timeout"],
            supervisor.CAPABILITY_PROBE_TIMEOUT_SECONDS,
        )

    def test_systemd_availability_rejects_unusable_or_stalled_user_bus(self) -> None:
        supervisor = SystemdUserSupervisor()
        for outcome in (
            {"return_value": _completed(1)},
            {"side_effect": subprocess.TimeoutExpired("systemctl", 2)},
            {"side_effect": PermissionError("systemctl")},
        ):
            with self.subTest(outcome=outcome):
                with (
                    mock.patch.object(
                        platform_services.shutil,
                        "which",
                        return_value="/usr/bin/systemctl",
                    ),
                    mock.patch.object(platform_services.subprocess, "run") as run,
                ):
                    run.configure_mock(**outcome)
                    self.assertFalse(supervisor.available())

    def test_user_services_returns_the_systemd_supervisor(self) -> None:
        self.assertIsInstance(user_services(), SystemdUserSupervisor)


class PlatformServiceBoundaryTests(unittest.TestCase):
    def test_orchestration_does_not_invoke_platform_tools_directly(self) -> None:
        forbidden = ('["systemctl"', "['systemctl'", '"notify-send"')
        for name in ORCHESTRATION_MODULES:
            with self.subTest(module=name):
                spec = importlib.util.find_spec(name)
                self.assertIsNotNone(spec, f"{name} no longer exists")
                assert spec is not None and spec.origin is not None
                source = Path(spec.origin).read_text()
                for token in forbidden:
                    self.assertNotIn(token, source, f"{name} still constructs {token}")


class PlatformCapabilityDiagnosticTests(unittest.TestCase):
    def test_missing_features_are_reported_not_required(self) -> None:
        supervisor = mock.Mock()
        supervisor.available.return_value = False
        supervisor.binary_available.return_value = True
        backend = desktop.DegradedWaylandDesktop(
            "gnome",
            "GNOME Wayland does not expose a supported focused-window or "
            "keyboard-injection API; dictation insertion degrades to stdout",
        )
        snapshot = checks.DiagnosticSnapshot(config=default_user_config())
        with (
            mock.patch.object(checks, "user_services", return_value=supervisor),
            mock.patch.object(
                checks, "notification_service_available", return_value=False
            ),
            mock.patch.object(
                checks, "notification_service_binary_available", return_value=True
            ),
            mock.patch.object(checks, "get_desktop", return_value=backend),
            mock.patch.object(desktop.shutil, "which", return_value=None),
        ):
            results = checks.check_platform_capabilities()
            units = checks.check_service_unit()
            focus = checks.check_focus_automation(snapshot)

        names = {result.name: result for result in results}
        self.assertEqual(
            set(names),
            {"platform:services", "platform:notifications", "platform:desktop"},
        )
        self.assertEqual(names["platform:services"].severity, Severity.WARNING)
        self.assertEqual(names["platform:notifications"].severity, Severity.WARNING)
        self.assertEqual(names["platform:desktop"].severity, Severity.WARNING)
        self.assertIn("installed", names["platform:services"].detail)
        self.assertIn(UNIT, names["platform:services"].detail)
        self.assertIn("installed", names["platform:notifications"].detail)
        self.assertIn("GNOME", names["platform:desktop"].detail)
        self.assertEqual(units[0].name, f"unit:{UNIT}")
        self.assertEqual(units[0].severity, Severity.WARNING)
        self.assertNotIn("not installed", units[0].detail)
        supervisor.show.assert_not_called()
        self.assertEqual(focus[0].severity, Severity.WARNING)
        self.assertIn("GNOME", focus[0].detail)

    def test_missing_platform_binaries_are_named(self) -> None:
        supervisor = mock.Mock()
        supervisor.available.return_value = False
        supervisor.binary_available.return_value = False
        with (
            mock.patch.object(checks, "user_services", return_value=supervisor),
            mock.patch.object(
                checks, "notification_service_available", return_value=False
            ),
            mock.patch.object(
                checks, "notification_service_binary_available", return_value=False
            ),
            mock.patch.object(checks, "get_desktop", return_value=None),
        ):
            results = checks.check_platform_capabilities()

        names = {result.name: result for result in results}
        self.assertIn("systemctl is not installed", names["platform:services"].detail)
        self.assertIn(
            "notify-send is not installed", names["platform:notifications"].detail
        )
        self.assertEqual(names["platform:desktop"].severity, Severity.WARNING)
        self.assertIn("no desktop session", names["platform:desktop"].detail)

    def test_supported_desktops_remain_ok(self) -> None:
        supervisor = mock.Mock()
        supervisor.available.return_value = True
        snapshot = checks.DiagnosticSnapshot(config=default_user_config())
        with (
            mock.patch.object(checks, "user_services", return_value=supervisor),
            mock.patch.object(
                checks, "notification_service_available", return_value=True
            ),
            mock.patch.object(checks, "get_desktop", return_value=desktop.X11Desktop()),
            mock.patch.object(desktop.shutil, "which", return_value="/usr/bin/tool"),
        ):
            results = checks.check_platform_capabilities()
            focus = checks.check_focus_automation(snapshot)

        names = {result.name: result for result in results}
        self.assertTrue(all(result.severity is Severity.OK for result in results))
        self.assertEqual(focus[0].severity, Severity.OK)
        self.assertIn("X11", names["platform:desktop"].detail)


class DesktopCapabilityTests(unittest.TestCase):
    def test_x11_hyprland_and_sway_report_full_capabilities(self) -> None:
        with mock.patch.object(desktop.shutil, "which", return_value="/usr/bin/tool"):
            x11 = desktop.X11Desktop().capabilities()
            hyprland = desktop.HyprlandDesktop().capabilities()
            sway = desktop.SwayDesktop().capabilities()

        for capabilities in (x11, hyprland, sway):
            with self.subTest(desktop=capabilities.name):
                self.assertTrue(capabilities.active_window)
                self.assertTrue(capabilities.clipboard)
                self.assertTrue(capabilities.type_text)
                self.assertTrue(capabilities.send_key)
        self.assertEqual(x11.name, "x11")
        self.assertEqual(x11.session, "x11")
        self.assertEqual(hyprland.name, "hyprland")
        self.assertEqual(hyprland.session, "wayland")
        self.assertEqual(sway.name, "sway")
        self.assertEqual(sway.session, "wayland")


if __name__ == "__main__":
    unittest.main()
