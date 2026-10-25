from __future__ import annotations

import inspect
import io
import json
import os
import socket
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from local_dictation import desktop, diagnostics
from local_dictation.diagnostics import checks, runner
from local_dictation.diagnostics.model import (
    PROBLEM_SEVERITIES,
    SEVERITY_RANK,
    CheckResult,
    Repair,
    Severity,
)
from local_dictation.user_config import (
    ComputeDevice,
    UserConfigurationError,
    default_user_config,
)

UNIT = "local-dictation.service"


def _result(severity: Severity, name: str = "x", **kwargs: object) -> CheckResult:
    return CheckResult(
        name=name,
        category="test",
        severity=severity,
        detail="detail",
        **kwargs,  # type: ignore[arg-type]
    )


def _completed(
    returncode: int, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["stub"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _snapshot(
    *,
    inject: str = "auto",
    source: str = "",
    device: ComputeDevice = ComputeDevice.AUTO,
) -> checks.DiagnosticSnapshot:
    config = default_user_config()
    return checks.DiagnosticSnapshot(
        config=replace(
            config,
            stt=replace(config.stt, device=device),
            dictation=replace(config.dictation, inject=inject, source=source),
        )
    )


def _broken_snapshot(message: str = "bad config.toml") -> checks.DiagnosticSnapshot:
    return checks.DiagnosticSnapshot(config=None, error=UserConfigurationError(message))


def _by_name(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {result.name: result for result in results}


class ModelTests(unittest.TestCase):
    def test_severity_rank_orders_fatal_first(self) -> None:
        ordered = sorted(Severity, key=lambda s: SEVERITY_RANK[s])
        self.assertEqual(ordered[0], Severity.FATAL)

    def test_check_result_to_dict_serializes_repair_summary(self) -> None:
        repair = Repair(summary="restart x", action=lambda: "done")
        result = _result(Severity.WARNING, suggestion="do it", repair=repair)
        payload = result.to_dict()
        self.assertEqual(payload["severity"], "warning")
        self.assertEqual(payload["repair"], "restart x")
        self.assertEqual(payload["suggestion"], "do it")

    def test_problem_severities_exclude_ok(self) -> None:
        self.assertNotIn(Severity.OK, PROBLEM_SEVERITIES)
        self.assertIn(Severity.FATAL, PROBLEM_SEVERITIES)


class RunnerTests(unittest.TestCase):
    def test_run_diagnostics_converts_crash_into_fatal(self) -> None:
        def boom() -> list[CheckResult]:
            raise RuntimeError("kaboom")

        results = runner.run_diagnostics([boom], snapshot=_snapshot())
        self.assertEqual(len(results), 1)
        self.assertIs(results[0].severity, Severity.FATAL)
        self.assertEqual(results[0].category, "internal")
        self.assertIn("kaboom", results[0].detail)

    def test_run_diagnostics_redacts_crash_details(self) -> None:
        def boom() -> list[CheckResult]:
            raise RuntimeError("token=supersecretvalue")

        results = runner.run_diagnostics([boom], snapshot=_snapshot())

        self.assertNotIn("supersecretvalue", results[0].detail)
        self.assertIn("[REDACTED]", results[0].detail)

    def test_run_diagnostics_aggregates_all_checks(self) -> None:
        results = runner.run_diagnostics(
            [
                lambda: [_result(Severity.OK, "a")],
                lambda: [_result(Severity.WARNING, "b")],
            ],
            snapshot=_snapshot(),
        )
        self.assertEqual([r.name for r in results], ["a", "b"])

    def test_run_diagnostics_resolves_one_snapshot_for_every_check(self) -> None:
        snapshot = _snapshot()
        observed: list[checks.DiagnosticSnapshot] = []

        def observe(value: checks.DiagnosticSnapshot) -> list[CheckResult]:
            observed.append(value)
            return []

        with mock.patch.object(
            checks.DiagnosticSnapshot, "load", return_value=snapshot
        ) as load:
            runner.run_diagnostics([observe, observe])

        load.assert_called_once_with()
        self.assertEqual(observed, [snapshot, snapshot])

    def test_snapshot_load_captures_configuration_errors(self) -> None:
        error = UserConfigurationError("stt.device must be one of: auto, cpu, cuda")
        with mock.patch.object(checks, "load_user_config", side_effect=error):
            snapshot = checks.DiagnosticSnapshot.load()

        self.assertIsNone(snapshot.config)
        self.assertIs(snapshot.error, error)

    def test_malformed_config_is_one_direct_fatal_beside_independent_checks(
        self,
    ) -> None:
        results = runner.run_diagnostics(
            [
                checks.check_configuration,
                lambda _snapshot: [_result(Severity.OK, "independent")],
                checks.check_compute_mode,
                checks.check_cuda,
            ],
            snapshot=_broken_snapshot(),
        )

        self.assertEqual(
            [result.name for result in results],
            ["configuration:user", "independent"],
        )
        self.assertIn("bad config.toml", results[0].detail)

    def test_render_json_reports_summary_and_health(self) -> None:
        results = [_result(Severity.FATAL, "a"), _result(Severity.OK, "b")]
        payload = json.loads(runner.render_json(results))
        self.assertFalse(payload["healthy"])
        self.assertEqual(payload["summary"]["fatal"], 1)
        self.assertEqual(payload["summary"]["ok"], 1)
        self.assertEqual(len(payload["checks"]), 2)

    def test_render_json_healthy_when_no_fatal(self) -> None:
        payload = json.loads(runner.render_json([_result(Severity.WARNING, "a")]))
        self.assertTrue(payload["healthy"])

    def test_render_human_groups_and_headline(self) -> None:
        repair = Repair(summary="restart svc", action=lambda: "ok")
        results = [
            _result(Severity.FATAL, "fatal-one", suggestion="dictate services logs"),
            _result(Severity.WARNING, "warn-one", repair=repair),
            _result(Severity.UNAVAILABLE, "opt-one"),
            _result(Severity.OK, "ok-one"),
        ]
        text = runner.render_human(results)
        self.assertIn("FATAL (1):", text)
        self.assertIn("WARNING (1):", text)
        self.assertIn("UNAVAILABLE (1):", text)
        self.assertNotIn("ok-one", text)
        self.assertIn("suggested: dictate services logs", text)
        self.assertIn("repair available: restart svc", text)
        self.assertIn("Dictation is unhealthy", text)
        self.assertIn("Summary: 1 fatal, 1 warning, 1 unavailable, 1 ok", text)
        self.assertIn("Run `dictate doctor --fix`", text)

    def test_render_human_healthy_message(self) -> None:
        text = runner.render_human([_result(Severity.OK, "a")])
        self.assertIn("Dictation looks healthy", text)

    def test_render_human_degraded_message(self) -> None:
        text = runner.render_human([_result(Severity.WARNING, "a")])
        self.assertIn("usable but some features are degraded", text)

    def test_render_human_omits_fix_hint_for_healthy_repairable_results(self) -> None:
        repair = Repair(summary="restart svc", action=lambda: "ok")
        text = runner.render_human([_result(Severity.OK, "a", repair=repair)])
        self.assertNotIn("doctor --fix", text)


class ApplyRepairTests(unittest.TestCase):
    def test_confirmed_repair_runs_and_reports(self) -> None:
        calls: list[str] = []
        repair = Repair(summary="fix it", action=lambda: calls.append("ran") or "done")
        out = io.StringIO()
        runner.apply_repairs(
            [_result(Severity.FATAL, "svc", repair=repair)],
            confirm=lambda _prompt: True,
            out=out,
        )
        self.assertEqual(calls, ["ran"])
        self.assertIn("repaired svc: done", out.getvalue())

    def test_declined_repair_is_skipped(self) -> None:
        ran: list[str] = []
        repair = Repair(summary="fix it", action=lambda: ran.append("x") or "done")
        out = io.StringIO()
        runner.apply_repairs(
            [_result(Severity.FATAL, "svc", repair=repair)],
            confirm=lambda _prompt: False,
            out=out,
        )
        self.assertEqual(ran, [])
        self.assertIn("skipped svc", out.getvalue())

    def test_failed_repair_does_not_raise(self) -> None:
        def action() -> str:
            raise RuntimeError("nope")

        out = io.StringIO()
        runner.apply_repairs(
            [_result(Severity.FATAL, "svc", repair=Repair("fix", action))],
            confirm=lambda _prompt: True,
            out=out,
        )
        self.assertIn("repair failed for svc: nope", out.getvalue())

    def test_failed_repair_redacts_diagnostic(self) -> None:
        def action() -> str:
            raise RuntimeError("password=supersecretvalue")

        out = io.StringIO()
        runner.apply_repairs(
            [_result(Severity.FATAL, "svc", repair=Repair("fix", action))],
            confirm=lambda _prompt: True,
            out=out,
        )

        self.assertNotIn("supersecretvalue", out.getvalue())
        self.assertIn("[REDACTED]", out.getvalue())

    def test_ok_results_with_repair_are_ignored(self) -> None:
        ran: list[str] = []
        repair = Repair(summary="fix", action=lambda: ran.append("x") or "done")
        out = io.StringIO()
        runner.apply_repairs(
            [_result(Severity.OK, "svc", repair=repair)],
            confirm=lambda _prompt: True,
            out=out,
        )
        self.assertEqual(ran, [])

    def test_default_confirmation_accepts_only_yes(self) -> None:
        for answer, expected in (
            ("y", True),
            (" YES ", True),
            ("", False),
            ("n", False),
        ):
            with self.subTest(answer=answer):
                with mock.patch("builtins.input", return_value=answer) as prompt:
                    self.assertIs(runner._default_confirm("Apply?"), expected)
                prompt.assert_called_once_with("Apply? [y/N] ")

    def test_default_confirmation_declines_without_input(self) -> None:
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertFalse(runner._default_confirm("Apply?"))


class DoctorEntryTests(unittest.TestCase):
    def test_package_exports_keyword_only_doctor(self) -> None:
        self.assertIs(diagnostics.doctor, runner.doctor)
        self.assertIs(diagnostics.ALL_CHECKS, checks.ALL_CHECKS)
        parameters = inspect.signature(diagnostics.doctor).parameters
        for name in ("json_output", "fix"):
            self.assertIs(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIs(parameters[name].default, False)

    def test_doctor_json_mode_returns_zero_without_fatal(self) -> None:
        out = io.StringIO()
        code = runner.doctor(
            json_output=True,
            checks=[lambda: [_result(Severity.WARNING, "a")]],
            out=out,
        )
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["healthy"])

    def test_doctor_human_mode_prints_report(self) -> None:
        out = io.StringIO()
        code = runner.doctor(checks=[lambda: [_result(Severity.OK, "a")]], out=out)
        self.assertEqual(code, 0)
        self.assertIn("Dictation looks healthy", out.getvalue())

    def test_doctor_returns_one_on_fatal(self) -> None:
        out = io.StringIO()
        code = runner.doctor(
            checks=[lambda: [_result(Severity.FATAL, "a")]],
            out=out,
        )
        self.assertEqual(code, 1)

    def test_doctor_fix_applies_confirmed_repairs(self) -> None:
        ran: list[str] = []
        repair = Repair(summary="restart", action=lambda: ran.append("x") or "ok")
        out = io.StringIO()
        code = runner.doctor(
            fix=True,
            checks=[lambda: [_result(Severity.FATAL, "svc", repair=repair)]],
            confirm=lambda _prompt: True,
            out=out,
        )
        self.assertEqual(code, 1)
        self.assertEqual(ran, ["x"])
        self.assertIn("Guided recovery", out.getvalue())

    def test_doctor_fix_reports_when_no_repairs(self) -> None:
        out = io.StringIO()
        runner.doctor(
            fix=True,
            checks=[lambda: [_result(Severity.WARNING, "a")]],
            confirm=lambda _prompt: True,
            out=out,
        )
        self.assertIn("No confirmation-gated repairs", out.getvalue())

    def test_doctor_json_mode_never_prompts_for_repairs(self) -> None:
        repair = Repair(summary="restart", action=lambda: "ok")
        confirm = mock.Mock(return_value=True)
        out = io.StringIO()
        runner.doctor(
            json_output=True,
            fix=True,
            checks=[lambda: [_result(Severity.FATAL, "svc", repair=repair)]],
            confirm=confirm,
            out=out,
        )
        confirm.assert_not_called()


class ConfigurationCheckTests(unittest.TestCase):
    def test_malformed_configuration_is_fatal_with_generic_path(self) -> None:
        results = checks.check_configuration(_broken_snapshot())

        self.assertEqual(len(results), 1)
        self.assertIs(results[0].severity, Severity.FATAL)
        self.assertIn(
            "$XDG_CONFIG_HOME/local-dictation/config.toml", results[0].suggestion or ""
        )

    def test_malformed_configuration_diagnostic_is_redacted(self) -> None:
        results = checks.check_configuration(
            _broken_snapshot("bad config.toml: token=supersecretvalue")
        )

        self.assertNotIn("supersecretvalue", results[0].detail)
        self.assertIn("[REDACTED]", results[0].detail)

    def test_valid_configuration_summarizes_stt_and_injection(self) -> None:
        results = checks.check_configuration(
            _snapshot(inject="type", device=ComputeDevice.CPU)
        )

        self.assertIs(results[0].severity, Severity.OK)
        self.assertIn("backend=parakeet", results[0].detail)
        self.assertIn("device=cpu", results[0].detail)
        self.assertIn("inject=type", results[0].detail)


class ExecutableCheckTests(unittest.TestCase):
    def _check(
        self,
        snapshot: checks.DiagnosticSnapshot,
        session: desktop.Desktop | None,
        *,
        location: str | None = None,
    ) -> tuple[list[CheckResult], mock.Mock]:
        with (
            mock.patch.object(
                checks, "get_desktop", return_value=session
            ) as get_desktop,
            mock.patch.object(checks, "_which", return_value=location),
        ):
            return checks.check_executables(snapshot), get_desktop

    def test_x11_requires_xdotool_and_recommends_xclip(self) -> None:
        results, _ = self._check(_snapshot(), desktop.X11Desktop())

        by_name = _by_name(results)
        self.assertEqual(
            list(by_name),
            ["executable:pw-record", "executable:xdotool", "executable:xclip"],
        )
        self.assertIs(by_name["executable:pw-record"].severity, Severity.FATAL)
        self.assertEqual(
            by_name["executable:pw-record"].suggestion,
            "paru -S --needed pipewire-audio",
        )
        self.assertIs(by_name["executable:xdotool"].severity, Severity.FATAL)
        self.assertEqual(
            by_name["executable:xdotool"].suggestion, "paru -S --needed xdotool"
        )
        self.assertIs(by_name["executable:xclip"].severity, Severity.WARNING)
        self.assertIn("types it instead", by_name["executable:xclip"].detail)

    def test_wayland_compositors_require_window_tool_and_wtype(self) -> None:
        for session, window_tool in (
            (desktop.HyprlandDesktop(), "hyprctl"),
            (desktop.SwayDesktop(), "swaymsg"),
        ):
            with self.subTest(window_tool=window_tool):
                results, _ = self._check(_snapshot(inject="paste"), session)

                by_name = _by_name(results)
                self.assertEqual(
                    list(by_name),
                    [
                        "executable:pw-record",
                        f"executable:{window_tool}",
                        "executable:wtype",
                        "executable:wl-copy",
                        "executable:wl-paste",
                    ],
                )
                self.assertIs(
                    by_name[f"executable:{window_tool}"].severity, Severity.FATAL
                )
                self.assertIsNone(by_name[f"executable:{window_tool}"].suggestion)
                self.assertIs(by_name["executable:wtype"].severity, Severity.FATAL)
                for clipboard in ("executable:wl-copy", "executable:wl-paste"):
                    self.assertIs(by_name[clipboard].severity, Severity.WARNING)
                    self.assertEqual(
                        by_name[clipboard].suggestion, "paru -S --needed wl-clipboard"
                    )

    def test_type_mode_does_not_need_clipboard_tools(self) -> None:
        results, _ = self._check(_snapshot(inject="type"), desktop.X11Desktop())

        self.assertEqual(
            [result.name for result in results],
            ["executable:pw-record", "executable:xdotool"],
        )

    def test_stdout_mode_needs_only_capture_tools(self) -> None:
        results, get_desktop = self._check(
            _snapshot(inject="stdout"), desktop.X11Desktop()
        )

        self.assertEqual([result.name for result in results], ["executable:pw-record"])
        get_desktop.assert_not_called()

    def test_malformed_configuration_still_checks_capture_tools(self) -> None:
        results, get_desktop = self._check(_broken_snapshot(), desktop.X11Desktop())

        self.assertEqual([result.name for result in results], ["executable:pw-record"])
        self.assertIs(results[0].severity, Severity.FATAL)
        get_desktop.assert_not_called()

    def test_desktops_without_supported_automation_add_no_tools(self) -> None:
        degraded = desktop.DegradedWaylandDesktop("gnome", "GNOME Wayland is limited")
        for session in (degraded, None):
            with self.subTest(session=session):
                results, _ = self._check(_snapshot(), session)
                self.assertEqual(
                    [result.name for result in results], ["executable:pw-record"]
                )

    def test_present_executables_are_ok(self) -> None:
        results, _ = self._check(
            _snapshot(), desktop.X11Desktop(), location="/usr/bin/tool"
        )

        self.assertTrue(all(r.severity is Severity.OK for r in results))
        self.assertTrue(all("/usr/bin/tool" in r.detail for r in results))


class FocusAutomationTests(unittest.TestCase):
    def _check(
        self,
        session: desktop.Desktop | None,
        *,
        snapshot: checks.DiagnosticSnapshot | None = None,
        tool: str | None = "/usr/bin/tool",
    ) -> tuple[CheckResult, mock.Mock]:
        with (
            mock.patch.object(
                checks, "get_desktop", return_value=session
            ) as get_desktop,
            mock.patch.object(desktop.shutil, "which", return_value=tool),
        ):
            results = checks.check_focus_automation(snapshot or _snapshot())
        self.assertEqual(len(results), 1)
        return results[0], get_desktop

    def test_focus_automation_ok_when_x11_stack_present(self) -> None:
        result, _ = self._check(desktop.X11Desktop())
        self.assertIs(result.severity, Severity.OK)
        self.assertIn("X11", result.detail)

    def test_focus_automation_ok_when_wayland_stack_present(self) -> None:
        result, _ = self._check(desktop.HyprlandDesktop())
        self.assertIs(result.severity, Severity.OK)
        self.assertIn("hyprland", result.detail)

    def test_focus_automation_warns_without_any_stack(self) -> None:
        result, _ = self._check(desktop.X11Desktop(), tool=None)
        self.assertIs(result.severity, Severity.WARNING)
        self.assertIn(
            "dictate config set dictation.inject stdout", result.suggestion or ""
        )

    def test_focus_automation_warns_without_desktop_session(self) -> None:
        result, _ = self._check(None)
        self.assertIs(result.severity, Severity.WARNING)
        self.assertIn("no desktop session", result.detail)
        self.assertIn("dictation.inject stdout", result.suggestion or "")

    def test_stdout_injection_does_not_need_focus_automation(self) -> None:
        result, get_desktop = self._check(
            desktop.DegradedWaylandDesktop("gnome", "GNOME Wayland is limited"),
            snapshot=_snapshot(inject="stdout"),
        )
        self.assertIs(result.severity, Severity.OK)
        self.assertIn("stdout", result.detail)
        get_desktop.assert_not_called()


class ServiceEnvironmentTests(unittest.TestCase):
    def _check(self, environment: Path) -> CheckResult:
        with (
            mock.patch.object(checks, "SERVICE_ENVIRONMENT", environment),
            mock.patch.object(
                checks,
                "SERVICE_EXECUTABLE",
                environment / "bin" / "local-dictation-server",
            ),
        ):
            results = checks.check_service_environment()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "venv:service")
        return results[0]

    def test_present_environment_is_ok(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = Path(temporary) / ".venv-service"
            (environment / "bin").mkdir(parents=True)
            (environment / "bin" / "local-dictation-server").touch()
            result = self._check(environment)
        self.assertIs(result.severity, Severity.OK)

    def test_incomplete_environment_is_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = Path(temporary) / ".venv-service"
            environment.mkdir()
            result = self._check(environment)
        self.assertIs(result.severity, Severity.WARNING)
        self.assertIn("local-dictation-server", result.detail)
        self.assertIn("scripts/install.sh", result.suggestion or "")

    def test_missing_environment_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self._check(Path(temporary) / ".venv-service")
        self.assertIs(result.severity, Severity.FATAL)
        self.assertIn(f"{UNIT} cannot start", result.detail)
        self.assertIn("scripts/install.sh", result.suggestion or "")


class ModelCacheTests(unittest.TestCase):
    def test_model_cache_present(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(checks, "HUGGINGFACE_CACHE", Path(temporary)):
                results = checks.check_model_cache(_snapshot())
        self.assertIs(results[0].severity, Severity.OK)

    def test_model_cache_absent_names_the_configured_model(self) -> None:
        with mock.patch.object(checks, "HUGGINGFACE_CACHE", Path("/nonexistent/hf")):
            results = checks.check_model_cache(_snapshot())
        self.assertIs(results[0].severity, Severity.WARNING)
        self.assertIn("parakeet model nemo-parakeet-tdt-0.6b-v2", results[0].detail)
        self.assertIn("dictate services start", results[0].suggestion or "")

    def test_model_cache_absent_without_configuration_is_skipped(self) -> None:
        with mock.patch.object(checks, "HUGGINGFACE_CACHE", Path("/nonexistent/hf")):
            self.assertEqual(checks.check_model_cache(_broken_snapshot()), [])


class ComputeAndCudaTests(unittest.TestCase):
    def _cuda(
        self,
        device: ComputeDevice,
        *,
        location: str | None = "/usr/bin/nvidia-smi",
        process: subprocess.CompletedProcess[str] | None = None,
    ) -> tuple[CheckResult, mock.Mock]:
        with (
            mock.patch.object(checks, "_which", return_value=location),
            mock.patch.object(checks, "_run", return_value=process) as run,
        ):
            results = checks.check_cuda(_snapshot(device=device))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "gpu:cuda")
        return results[0], run

    def test_cpu_compute_resolves_without_probing_cuda(self) -> None:
        snapshot = _snapshot(device=ComputeDevice.CPU)
        with (
            mock.patch.object(checks, "_which") as which,
            mock.patch.object(checks, "_run") as run,
        ):
            compute = checks.check_compute_mode(snapshot)
            cuda = checks.check_cuda(snapshot)

        self.assertIn("effective device=cpu", compute[0].detail)
        self.assertIs(cuda[0].severity, Severity.OK)
        which.assert_not_called()
        run.assert_not_called()

    def test_auto_compute_is_resolved_at_service_start(self) -> None:
        results = checks.check_compute_mode(_snapshot())
        self.assertEqual(results[0].name, "compute:stt")
        self.assertIs(results[0].severity, Severity.OK)
        self.assertIn("device=auto", results[0].detail)
        self.assertIn("resolved at service start", results[0].detail)

    def test_compute_checks_skip_malformed_configuration(self) -> None:
        self.assertEqual(checks.check_compute_mode(_broken_snapshot()), [])
        self.assertEqual(checks.check_cuda(_broken_snapshot()), [])

    def test_auto_without_nvidia_smi_warns_and_offers_cpu(self) -> None:
        result, run = self._cuda(ComputeDevice.AUTO, location=None)
        self.assertIs(result.severity, Severity.WARNING)
        self.assertIn("falls back", result.detail)
        self.assertIn("dictate config set stt.device cpu", result.suggestion or "")
        run.assert_not_called()

    def test_explicit_cuda_without_nvidia_smi_is_fatal(self) -> None:
        result, _ = self._cuda(ComputeDevice.CUDA, location=None)
        self.assertIs(result.severity, Severity.FATAL)
        self.assertEqual(result.suggestion, checks.INSTALL_HINTS["nvidia-smi"])

    def test_failing_nvidia_smi_reports_first_error_line(self) -> None:
        for device, severity in (
            (ComputeDevice.AUTO, Severity.WARNING),
            (ComputeDevice.CUDA, Severity.FATAL),
        ):
            with self.subTest(device=device):
                result, _ = self._cuda(
                    device,
                    process=_completed(9, "", "driver mismatch\nsecond line"),
                )
                self.assertIs(result.severity, severity)
                self.assertIn("failed to query the GPU: driver mismatch", result.detail)
                self.assertNotIn("second line", result.detail)
                self.assertIn("nvidia-smi", result.suggestion or "")

    def test_unrunnable_nvidia_smi_is_reported(self) -> None:
        result, _ = self._cuda(ComputeDevice.AUTO, process=None)
        self.assertIs(result.severity, Severity.WARNING)
        self.assertIn("nvidia-smi failed to query the GPU;", result.detail)

    def test_working_nvidia_smi_is_ok(self) -> None:
        result, run = self._cuda(ComputeDevice.CUDA, process=_completed(0, "GPU"))
        self.assertIs(result.severity, Severity.OK)
        run.assert_called_once_with(["nvidia-smi"], timeout=10)


class PipewireSourceTests(unittest.TestCase):
    SOURCES = ("alsa_input.pci-0000_00_1f.3.analog-stereo", "echo_cancel_source")

    def _check(
        self,
        snapshot: checks.DiagnosticSnapshot,
        *,
        wpctl: str | None = "/usr/bin/wpctl",
        sources: tuple[str, ...] | Exception = SOURCES,
    ) -> tuple[CheckResult, mock.Mock]:
        listing = (
            mock.patch.object(
                checks, "list_pipewire_capture_sources", side_effect=sources
            )
            if isinstance(sources, Exception)
            else mock.patch.object(
                checks, "list_pipewire_capture_sources", return_value=sources
            )
        )
        with (
            mock.patch.object(checks, "_which", return_value=wpctl),
            listing as list_sources,
        ):
            results = checks.check_pipewire_source(snapshot)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "audio:pipewire")
        return results[0], list_sources

    def test_missing_wpctl_is_fatal_with_install_hint(self) -> None:
        result, list_sources = self._check(_snapshot(), wpctl=None)
        self.assertIs(result.severity, Severity.FATAL)
        self.assertIn("system default source", result.detail)
        self.assertEqual(result.suggestion, "paru -S --needed wireplumber")
        list_sources.assert_not_called()

    def test_unconfigured_source_uses_system_default(self) -> None:
        result, _ = self._check(_snapshot())
        self.assertIs(result.severity, Severity.OK)
        self.assertIn("system default source", result.detail)

    def test_configured_filter_node_is_a_valid_capture_target(self) -> None:
        result, _ = self._check(_snapshot(source="echo_cancel_source"))
        self.assertIs(result.severity, Severity.OK)
        self.assertIn("echo_cancel_source", result.detail)

    def test_missing_configured_source_is_fatal(self) -> None:
        result, _ = self._check(_snapshot(source="missing-dictation-mic"))
        self.assertIs(result.severity, Severity.FATAL)
        self.assertIn("missing-dictation-mic", result.detail)
        self.assertIn("echo_cancel_source", result.detail)
        self.assertEqual(result.suggestion, "dictate config set dictation.source")

    def test_unusable_pipewire_is_fatal(self) -> None:
        error = UserConfigurationError(
            "wpctl status failed; PipeWire may not be running"
        )
        result, _ = self._check(_snapshot(), sources=error)
        self.assertIs(result.severity, Severity.FATAL)
        self.assertIn("wpctl status failed", result.detail)
        self.assertIn("pipewire wireplumber", result.suggestion or "")

    def test_malformed_configuration_reports_pipewire_status(self) -> None:
        result, _ = self._check(_broken_snapshot())
        self.assertIs(result.severity, Severity.OK)


_HEALTHY_UNIT = {
    "LoadState": "loaded",
    "ActiveState": "active",
    "SubState": "running",
    "Result": "success",
    "NRestarts": "0",
    "UnitFileState": "enabled",
}


class ServiceUnitTests(unittest.TestCase):
    def _check(
        self,
        properties: dict[str, str],
        *,
        installed: bool = True,
        available: bool = True,
    ) -> tuple[CheckResult, mock.Mock]:
        supervisor = mock.Mock()
        supervisor.available.return_value = available
        supervisor.show.return_value = properties
        with tempfile.TemporaryDirectory() as temporary:
            unit_dir = Path(temporary)
            if installed:
                (unit_dir / UNIT).write_text("[Unit]\n")
            with (
                mock.patch.object(checks, "SYSTEMD_USER_DIR", unit_dir),
                mock.patch.object(checks, "user_services", return_value=supervisor),
            ):
                results = checks.check_service_unit()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, f"unit:{UNIT}")
        return results[0], supervisor

    def test_unavailable_supervision_skips_unit_state(self) -> None:
        result, supervisor = self._check({}, available=False)
        self.assertIs(result.severity, Severity.WARNING)
        self.assertNotIn("not installed", result.detail)
        supervisor.show.assert_not_called()

    def test_unknown_unit_is_not_installed(self) -> None:
        result, _ = self._check(
            {
                "LoadState": "not-found",
                "ActiveState": "inactive",
                "SubState": "dead",
                "Result": "success",
                "NRestarts": "0",
                "UnitFileState": "",
            },
            installed=False,
        )
        self.assertIs(result.severity, Severity.FATAL)
        self.assertIn("is not installed", result.detail)
        self.assertEqual(result.suggestion, "dictate services install")

    def test_unit_file_that_systemd_has_not_loaded_needs_install(self) -> None:
        result, _ = self._check({**_HEALTHY_UNIT, "LoadState": "not-found"})
        self.assertIs(result.severity, Severity.FATAL)
        self.assertIn("systemd has not loaded it", result.detail)
        self.assertEqual(result.suggestion, "dictate services install")

    def test_unqueryable_missing_unit_is_not_installed(self) -> None:
        result, _ = self._check({}, installed=False)
        self.assertIs(result.severity, Severity.FATAL)
        self.assertIn("is not installed", result.detail)

    def test_restart_loop_is_fatal_with_repair(self) -> None:
        result, _ = self._check(
            {**_HEALTHY_UNIT, "ActiveState": "activating", "NRestarts": "9"}
        )
        self.assertIs(result.severity, Severity.FATAL)
        self.assertIn("restarted 9 times", result.detail)
        self.assertEqual(result.suggestion, "dictate services logs")
        assert result.repair is not None
        self.assertEqual(result.repair.summary, f"restart {UNIT}")

    def test_failed_unit_is_fatal_with_repair(self) -> None:
        result, _ = self._check(
            {
                **_HEALTHY_UNIT,
                "ActiveState": "failed",
                "SubState": "failed",
                "Result": "exit-code",
            }
        )
        self.assertIs(result.severity, Severity.FATAL)
        self.assertIn("result=exit-code", result.detail)
        self.assertIsNotNone(result.repair)

    def test_disabled_unit_is_warning(self) -> None:
        result, _ = self._check({**_HEALTHY_UNIT, "UnitFileState": "disabled"})
        self.assertIs(result.severity, Severity.WARNING)
        self.assertIn("should start at login", result.detail)
        self.assertEqual(result.suggestion, "dictate services install")
        self.assertIsNone(result.repair)

    def test_inactive_unit_is_warning_with_restart_repair(self) -> None:
        result, _ = self._check(
            {**_HEALTHY_UNIT, "ActiveState": "inactive", "SubState": "dead"}
        )
        self.assertIs(result.severity, Severity.WARNING)
        self.assertEqual(result.suggestion, "dictate services start")
        self.assertIsNotNone(result.repair)

    def test_healthy_unit_is_ok(self) -> None:
        result, supervisor = self._check(_HEALTHY_UNIT)
        self.assertIs(result.severity, Severity.OK)
        self.assertIn("active=active", result.detail)
        name, properties = supervisor.show.call_args.args
        self.assertEqual(name, UNIT)
        self.assertIn("LoadState", properties)

    def test_unparseable_restart_count_is_treated_as_zero(self) -> None:
        result, _ = self._check({**_HEALTHY_UNIT, "NRestarts": "[not set]"})
        self.assertIs(result.severity, Severity.OK)
        self.assertIn("restarts=0", result.detail)


class SttSocketTests(unittest.TestCase):
    def _check(self, socket_path: Path, *, ready: bool, active: str) -> CheckResult:
        with (
            mock.patch.object(checks, "STT_SOCKET", socket_path),
            mock.patch.object(checks, "socket_ready", return_value=ready),
            mock.patch.object(checks, "_service_active_state", return_value=active),
        ):
            results = checks.check_stt_socket()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "socket:stt")
        return results[0]

    def test_socket_ready_is_ok(self) -> None:
        result = self._check(Path("/run/x.sock"), ready=True, active="active")
        self.assertIs(result.severity, Severity.OK)

    def test_socket_down_while_service_active_offers_restart(self) -> None:
        result = self._check(Path("/run/x.sock"), ready=False, active="active")
        self.assertIs(result.severity, Severity.WARNING)
        self.assertIn("may still be loading", result.detail)
        assert result.repair is not None
        self.assertEqual(result.repair.summary, f"restart {UNIT}")

    def test_stale_socket_offers_removal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            socket_path = Path(temporary) / "x.sock"
            socket_path.write_text("stale")
            result = self._check(socket_path, ready=False, active="inactive")
        self.assertIs(result.severity, Severity.WARNING)
        self.assertIn("stale", result.detail)
        assert result.repair is not None
        self.assertIn("remove stale socket", result.repair.summary)

    def test_absent_socket_suggests_starting_the_service(self) -> None:
        result = self._check(Path("/nonexistent/x.sock"), ready=False, active="failed")
        self.assertIs(result.severity, Severity.WARNING)
        self.assertIn(f"{UNIT} is failed", result.detail)
        self.assertEqual(result.suggestion, "dictate services start")
        self.assertIsNone(result.repair)

    def test_service_state_is_unknown_when_systemctl_cannot_run(self) -> None:
        supervisor = mock.Mock()
        supervisor.show.side_effect = FileNotFoundError("systemctl")
        with mock.patch.object(checks, "user_services", return_value=supervisor):
            self.assertEqual(checks._service_active_state(UNIT), "unknown")


class RepairActionTests(unittest.TestCase):
    def test_restart_service_repair_success(self) -> None:
        supervisor = mock.Mock()
        supervisor.restart.return_value = _completed(0)
        with mock.patch.object(checks, "user_services", return_value=supervisor):
            message = checks._restart_service_repair(UNIT).action()
        supervisor.restart.assert_called_once_with(UNIT)
        self.assertEqual(message, f"restarted {UNIT}")

    def test_restart_service_repair_failure_raises(self) -> None:
        supervisor = mock.Mock()
        supervisor.restart.return_value = _completed(1, "", "boom")
        with mock.patch.object(checks, "user_services", return_value=supervisor):
            repair = checks._restart_service_repair(UNIT)
            with self.assertRaises(checks.DictationError) as caught:
                repair.action()
        self.assertIn("boom", str(caught.exception))

    def test_restart_service_repair_spawn_failure_raises(self) -> None:
        supervisor = mock.Mock()
        supervisor.restart.side_effect = OSError("gone")
        with mock.patch.object(checks, "user_services", return_value=supervisor):
            repair = checks._restart_service_repair(UNIT)
            with self.assertRaises(checks.DictationError):
                repair.action()

    def _stale_socket(self, directory: str) -> Path:
        path = Path(directory) / "stale.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        listener.close()
        return path

    def test_remove_stale_socket_unlinks_unresponsive_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._stale_socket(temporary)
            message = checks._remove_stale_socket_repair(path).action()
            self.assertFalse(path.exists())
        self.assertIn("removed stale socket", message)

    def test_remove_stale_socket_keeps_a_socket_that_responds_again(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._stale_socket(temporary)
            with mock.patch.object(checks, "socket_ready", return_value=True):
                message = checks._remove_stale_socket_repair(path).action()
            self.assertTrue(path.is_socket())
        self.assertIn("responding again", message)

    def test_remove_stale_socket_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "gone.sock"
            message = checks._remove_stale_socket_repair(missing).action()
        self.assertIn("nothing to do", message)

    def test_remove_stale_socket_failure_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._stale_socket(temporary)
            with (
                mock.patch.object(Path, "unlink", side_effect=PermissionError("no")),
                self.assertRaisesRegex(checks.DictationError, "could not remove"),
            ):
                checks._remove_stale_socket_repair(path).action()


class RuntimeDirectoryTests(unittest.TestCase):
    def _check(
        self, runtime: Path, state: Path, *, xdg: str | None
    ) -> dict[str, CheckResult]:
        with mock.patch.dict(os.environ):
            os.environ.pop("XDG_RUNTIME_DIR", None)
            if xdg is not None:
                os.environ["XDG_RUNTIME_DIR"] = xdg
            with (
                mock.patch.object(checks, "RUNTIME", runtime),
                mock.patch.object(checks, "STATE_DIR", state),
            ):
                results = checks.check_runtime_directories()
        return _by_name(results)

    def test_writable_runtime_is_ok_and_missing_state_is_created_later(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            results = self._check(runtime, runtime / "state", xdg=temporary)
        self.assertIs(results["runtime:xdg"].severity, Severity.OK)
        self.assertIs(results["runtime:state"].severity, Severity.OK)
        self.assertIn("will be created", results["runtime:state"].detail)

    def test_missing_runtime_is_fatal(self) -> None:
        missing = Path("/nonexistent/runtime")
        results = self._check(missing, missing / "state", xdg=str(missing))
        self.assertIs(results["runtime:xdg"].severity, Severity.FATAL)

    def test_unset_runtime_variable_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            results = self._check(runtime, runtime / "state", xdg=None)
        self.assertIs(results["runtime:xdg"].severity, Severity.FATAL)
        self.assertIn("XDG_RUNTIME_DIR is not set", results["runtime:xdg"].detail)

    def test_private_state_directory_is_ok(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            state.mkdir(mode=0o700)
            results = self._check(Path(temporary), state, xdg=temporary)
        self.assertIs(results["runtime:state"].severity, Severity.OK)
        self.assertIn("private permissions", results["runtime:state"].detail)

    def test_shared_state_directory_is_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            state.mkdir()
            state.chmod(0o750)
            results = self._check(Path(temporary), state, xdg=temporary)
        self.assertIs(results["runtime:state"].severity, Severity.WARNING)
        self.assertIn("mode 750", results["runtime:state"].detail)

    def test_directory_owned_by_other_uid_is_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            state.mkdir(mode=0o700)
            fake_stat = mock.Mock(st_uid=999999, st_mode=0o040700)
            with mock.patch.object(Path, "stat", return_value=fake_stat):
                result = checks._directory_result("runtime:state", state)
        self.assertIs(result.severity, Severity.WARNING)
        self.assertIn("owned by uid", result.detail)

    def test_unreadable_directory_is_warning_with_redacted_detail(self) -> None:
        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(
                Path, "stat", side_effect=PermissionError("token=supersecretvalue")
            ),
        ):
            result = checks._directory_result("runtime:state", Path("/run/state"))
        self.assertIs(result.severity, Severity.WARNING)
        self.assertIn("could not stat", result.detail)
        self.assertNotIn("supersecretvalue", result.detail)


class ProcessCapabilityTests(unittest.TestCase):
    def test_strong_process_ownership_is_ok(self) -> None:
        with mock.patch.object(checks, "capability_diagnostics", return_value=[]):
            results = checks.check_process_capabilities()
        self.assertIs(results[0].severity, Severity.OK)

    def test_missing_process_capabilities_are_warnings(self) -> None:
        issues = ["boot identity is unavailable", "pidfd_open is unavailable"]
        with mock.patch.object(checks, "capability_diagnostics", return_value=issues):
            results = checks.check_process_capabilities()
        self.assertIs(results[0].severity, Severity.WARNING)
        self.assertEqual(results[0].detail, "; ".join(issues))


class SmokeTests(unittest.TestCase):
    def test_all_checks_run_without_crashing(self) -> None:
        results = runner.run_diagnostics()
        self.assertTrue(results)
        self.assertFalse(
            any(r.category == "internal" for r in results),
            [r.detail for r in results if r.category == "internal"],
        )


if __name__ == "__main__":
    unittest.main()
