from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from local_dictation import (
    cli,
    config_management,
    dictation,
    service_manager,
    user_config,
    vocabulary,
)
from local_dictation.errors import DictationError


class RecordingCommandTests(unittest.TestCase):
    def test_recording_commands_run_with_loaded_configuration(self) -> None:
        loaded = user_config.default_user_config()
        for command in cli.RECORDING_COMMANDS:
            with (
                self.subTest(command=command),
                mock.patch.object(user_config, "load_user_config", return_value=loaded),
                mock.patch.object(dictation, "run") as run,
            ):
                cli.main([command])

                run.assert_called_once_with(command, loaded)

    def test_retry_passes_an_absolute_recording_path(self) -> None:
        loaded = user_config.default_user_config()
        with (
            mock.patch.object(user_config, "load_user_config", return_value=loaded),
            mock.patch.object(dictation, "retry") as retry,
        ):
            cli.main(["retry", "recording.wav"])

        retry.assert_called_once_with(Path(os.getcwd()) / "recording.wav", loaded)

    def test_invalid_command_exits_before_dispatch(self) -> None:
        with (
            mock.patch.object(cli, "dispatch") as dispatch,
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as caught,
        ):
            cli.main(["unknown"])

        self.assertEqual(caught.exception.code, 2)
        dispatch.assert_not_called()

    def test_entrypoint_import_loads_no_other_package_modules(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        code = (
            "import json, sys; "
            "import local_dictation.cli; "
            "print(json.dumps(sorted(name for name in sys.modules "
            "if name.startswith('local_dictation.'))))"
        )

        process = subprocess.run(
            [sys.executable, "-c", code],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )

        self.assertEqual(json.loads(process.stdout), ["local_dictation.cli"])


class FailureReportingTests(unittest.TestCase):
    def test_failures_are_logged_redacted_reported_and_notified(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(
                cli,
                "dispatch",
                side_effect=DictationError("STT failed: token=abc123"),
            ),
            mock.patch("local_dictation.notifications.notify") as notify,
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as caught,
        ):
            cli.main(["toggle"])

        self.assertEqual(caught.exception.code, 1)
        diagnostic, summary = stderr.getvalue().splitlines()
        self.assertEqual(
            json.loads(diagnostic),
            {
                "component": "cli",
                "event": "command_failed",
                "diagnostic": "DictationError: STT failed: token=[REDACTED]",
            },
        )
        self.assertEqual(
            summary, "dictate: The command failed. Check the logs for details."
        )
        notify.assert_called_once_with(
            "The command failed. Check the logs for details.", error=True
        )

    def test_exit_requests_pass_through_unreported(self) -> None:
        with (
            mock.patch.object(cli, "dispatch", side_effect=SystemExit(3)),
            mock.patch("local_dictation.notifications.notify") as notify,
            self.assertRaises(SystemExit) as caught,
        ):
            cli.main(["doctor"])

        self.assertEqual(caught.exception.code, 3)
        notify.assert_not_called()


class VocabularyCommandTests(unittest.TestCase):
    def test_add_and_remove_dispatch(self) -> None:
        with mock.patch.object(vocabulary, "add_replacement") as add:
            cli.main(["vocabulary", "add", "herder", "herdr", "--force"])
        add.assert_called_once_with("herder", "herdr", force=True)

        with mock.patch.object(vocabulary, "remove_replacement") as remove:
            cli.main(["vocabulary", "remove", "herder"])
        remove.assert_called_once_with("herder")

    def test_list_export_and_import_dispatch(self) -> None:
        with mock.patch.object(vocabulary, "list_entries") as list_entries:
            cli.main(["vocabulary", "list"])
        list_entries.assert_called_once_with()

        with mock.patch.object(vocabulary, "export_entries") as export_entries:
            cli.main(["vocabulary", "export", "--output", "out.json"])
        export_entries.assert_called_once_with(Path("out.json"))

        with mock.patch.object(vocabulary, "import_entries") as import_entries:
            cli.main(["vocabulary", "import", "backup.json", "--replace"])
        import_entries.assert_called_once_with(Path("backup.json"), replace=True)


class ConfigCommandTests(unittest.TestCase):
    def _result(self, *services: str) -> config_management.ConfigChangeResult:
        return config_management.ConfigChangeResult(
            config=user_config.default_user_config(),
            changed_keys=("stt.device",),
            restart_services=services,
        )

    def test_show_prints_requested_value(self) -> None:
        output = io.StringIO()
        with (
            mock.patch.object(
                config_management, "show_config", return_value='"cuda"'
            ) as show,
            redirect_stdout(output),
        ):
            cli.main(["config", "show", "stt.device", "--json"])

        show.assert_called_once_with(key="stt.device", json_output=True)
        self.assertEqual(output.getvalue(), '"cuda"\n')

    def test_set_prints_restart_notice(self) -> None:
        output = io.StringIO()
        with (
            mock.patch.object(
                config_management,
                "commit_config_setting",
                return_value=self._result("local-dictation.service"),
            ) as commit,
            redirect_stdout(output),
        ):
            cli.main(["config", "set", "stt.device", "cpu"])

        commit.assert_called_once_with("stt.device", "cpu")
        self.assertIn("dictate services restart", output.getvalue())

    def test_source_can_be_picked_interactively(self) -> None:
        with (
            mock.patch.object(
                config_management,
                "commit_config_setting",
                return_value=self._result(),
            ) as commit,
            redirect_stdout(io.StringIO()),
        ):
            cli.main(["config", "set", "dictation.source"])

        commit.assert_called_once_with("dictation.source", None)

    def test_reset_accepts_known_sections_only(self) -> None:
        output = io.StringIO()
        with (
            mock.patch.object(
                config_management, "reset_config", return_value=self._result()
            ) as reset,
            redirect_stdout(output),
        ):
            cli.main(["config", "reset", "--section", "dictation"])
            cli.main(["config", "reset"])

        self.assertEqual(
            reset.call_args_list,
            [mock.call(section="dictation"), mock.call(section=None)],
        )
        self.assertEqual(output.getvalue(), "No restart needed.\n" * 2)
        with (
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as caught,
        ):
            cli.main(["config", "reset", "--section", "providers"])
        self.assertEqual(caught.exception.code, 2)


class ServiceCommandTests(unittest.TestCase):
    def test_lifecycle_commands_dispatch(self) -> None:
        cases = {
            ("install",): ("install_service", mock.call(force=False)),
            ("install", "--force"): ("install_service", mock.call(force=True)),
            ("start",): ("start_service", mock.call()),
            ("stop",): ("stop_service", mock.call()),
            ("restart",): ("restart_service", mock.call()),
            ("status",): ("status", mock.call()),
            ("uninstall",): ("uninstall_service", mock.call()),
            ("logs", "-f", "-n", "0"): ("logs", mock.call(follow=True, lines=1)),
            ("logs",): ("logs", mock.call(follow=False, lines=100)),
        }
        for arguments, (name, expected) in cases.items():
            with (
                self.subTest(arguments=arguments),
                mock.patch.object(service_manager, name) as function,
            ):
                cli.main(["services", *arguments])

                self.assertEqual(function.call_args_list, [expected])

    def test_audit_exit_status_is_preserved(self) -> None:
        with (
            mock.patch.object(service_manager, "audit_service", return_value=4),
            self.assertRaises(SystemExit) as caught,
        ):
            cli.main(["services", "audit"])

        self.assertEqual(caught.exception.code, 4)


class DoctorCommandTests(unittest.TestCase):
    def test_doctor_exit_status_and_options_are_preserved(self) -> None:
        with (
            mock.patch("local_dictation.diagnostics.doctor", return_value=2) as doctor,
            self.assertRaises(SystemExit) as caught,
        ):
            cli.main(["doctor", "--json", "--fix"])

        self.assertEqual(caught.exception.code, 2)
        doctor.assert_called_once_with(json_output=True, fix=True)


if __name__ == "__main__":
    unittest.main()
