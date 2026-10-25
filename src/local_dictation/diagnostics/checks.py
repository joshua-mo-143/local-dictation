from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import (
    PROJECT_ROOT,
    RUNTIME,
    SERVICE_NAME,
    STATE_DIR,
    STT_SOCKET,
    SYSTEMD_USER_DIR,
)
from ..desktop import get_desktop
from ..diagnostic_safety import redact_diagnostic
from ..errors import DictationError
from ..ipc import socket_ready
from ..notifications import (
    notification_service_available,
    notification_service_binary_available,
)
from ..pipewire import list_pipewire_capture_sources
from ..platform_services import user_services
from ..process import capability_diagnostics
from ..user_config import (
    ComputeDevice,
    UserConfig,
    UserConfigurationError,
    load_user_config,
    resolve_local_compute,
)
from .model import CheckResult, Repair, Severity


@dataclass(frozen=True)
class DiagnosticSnapshot:
    """One immutable configuration view for a doctor run."""

    config: UserConfig | None
    error: UserConfigurationError | None = None

    @classmethod
    def load(cls) -> DiagnosticSnapshot:
        try:
            config = load_user_config()
        except UserConfigurationError as exc:
            return cls(config=None, error=exc)
        return cls(config=config)


Check = Callable[..., list[CheckResult]]


def _resolved(snapshot: DiagnosticSnapshot | None) -> DiagnosticSnapshot:
    return snapshot if snapshot is not None else DiagnosticSnapshot.load()


# systemd stops restarting the unit at this burst; matches its StartLimitBurst.
RESTART_LOOP_THRESHOLD = 5

SERVICE_ENVIRONMENT = PROJECT_ROOT / ".venv-service"
SERVICE_EXECUTABLE = SERVICE_ENVIRONMENT / "bin" / "local-dictation-server"
HUGGINGFACE_CACHE = Path.home() / ".cache" / "huggingface"

CAPTURE_EXECUTABLES: tuple[tuple[str, str], ...] = (
    ("pw-record", "record microphone audio through PipeWire"),
)
# Keyed by DesktopCapabilities.name; desktops without supported focused-window
# automation have no entry.
INSERTION_EXECUTABLES: dict[str, tuple[tuple[str, str], ...]] = {
    "x11": (("xdotool", "find the focused X11 window and insert recognized text"),),
    "hyprland": (
        ("hyprctl", "find the focused Hyprland window"),
        ("wtype", "insert recognized text on Wayland"),
    ),
    "sway": (
        ("swaymsg", "find the focused Sway window"),
        ("wtype", "insert recognized text on Wayland"),
    ),
}
_PASTE = "paste recognized text (dictation types it instead)"
CLIPBOARD_EXECUTABLES: dict[str, tuple[tuple[str, str], ...]] = {
    "x11": (("xclip", _PASTE),),
    "hyprland": (("wl-copy", _PASTE), ("wl-paste", _PASTE)),
    "sway": (("wl-copy", _PASTE), ("wl-paste", _PASTE)),
}
INSTALL_HINTS: dict[str, str] = {
    "pw-record": "paru -S --needed pipewire-audio",
    "wpctl": "paru -S --needed wireplumber",
    "xdotool": "paru -S --needed xdotool",
    "xclip": "paru -S --needed xclip",
    "wtype": "paru -S --needed wtype",
    "wl-copy": "paru -S --needed wl-clipboard",
    "wl-paste": "paru -S --needed wl-clipboard",
    "nvidia-smi": "install the proprietary NVIDIA driver for your kernel",
}
_STDOUT_SUGGESTION = (
    "dictate config set dictation.inject stdout  "
    "# if focused-window insertion is not required"
)


def _which(name: str) -> str | None:
    return shutil.which(name)


def _run(
    command: Sequence[str], *, timeout: float = 5.0
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _systemctl_show(name: str, properties: Sequence[str]) -> dict[str, str]:
    try:
        return user_services().show(name, properties)
    except OSError:
        return {}


def _service_active_state(name: str) -> str:
    return _systemctl_show(name, ["ActiveState"]).get("ActiveState", "unknown")


def _restart_service_repair(name: str) -> Repair:
    def action() -> str:
        try:
            process = user_services().restart(name)
        except OSError as exc:
            raise DictationError(f"could not restart {name}") from exc
        if process.returncode:
            raise DictationError(process.stderr.strip() or f"could not restart {name}")
        return f"restarted {name}"

    return Repair(summary=f"restart {name}", action=action)


def _remove_stale_socket_repair(path: Path) -> Repair:
    def action() -> str:
        if socket_ready(path):
            return f"{path} is responding again; nothing to do"
        try:
            if path.is_socket():
                path.unlink(missing_ok=True)
                return f"removed stale socket {path}"
        except OSError as exc:
            raise DictationError(f"could not remove {path}: {exc}") from exc
        return f"{path} is no longer a stale socket; nothing to do"

    return Repair(summary=f"remove stale socket {path}", action=action)


def check_configuration(
    snapshot: DiagnosticSnapshot | None = None,
) -> list[CheckResult]:
    resolved = _resolved(snapshot)
    if resolved.config is None:
        return [
            CheckResult(
                name="configuration:user",
                category="configuration",
                severity=Severity.FATAL,
                detail=redact_diagnostic(resolved.error),
                suggestion=(
                    "fix $XDG_CONFIG_HOME/local-dictation/config.toml or the "
                    "DICTATION_* environment overrides"
                ),
            )
        ]
    stt = resolved.config.stt
    return [
        CheckResult(
            name="configuration:user",
            category="configuration",
            severity=Severity.OK,
            detail=(
                f"STT backend={stt.backend} model={stt.model} device={stt.device}; "
                f"dictation inject={resolved.config.dictation.inject}"
            ),
        )
    ]


def _executable_result(name: str, purpose: str, *, required: bool) -> CheckResult:
    location = _which(name)
    if location is not None:
        return CheckResult(
            name=f"executable:{name}",
            category="executables",
            severity=Severity.OK,
            detail=f"{name} found at {location}",
        )
    return CheckResult(
        name=f"executable:{name}",
        category="executables",
        severity=Severity.FATAL if required else Severity.WARNING,
        detail=(
            f"{name} is not on PATH; {'required' if required else 'needed'} "
            f"to {purpose}"
        ),
        suggestion=INSTALL_HINTS.get(name),
    )


def check_executables(
    snapshot: DiagnosticSnapshot | None = None,
) -> list[CheckResult]:
    """Check capture tools and the insertion tools the inject mode will use."""

    config = _resolved(snapshot).config
    results = [
        _executable_result(name, purpose, required=True)
        for name, purpose in CAPTURE_EXECUTABLES
    ]
    if config is None or config.dictation.inject == "stdout":
        return results
    desktop = get_desktop()
    if desktop is None:
        return results
    backend = desktop.capabilities().name
    results.extend(
        _executable_result(name, purpose, required=True)
        for name, purpose in INSERTION_EXECUTABLES.get(backend, ())
    )
    if config.dictation.inject != "type":
        results.extend(
            _executable_result(name, purpose, required=False)
            for name, purpose in CLIPBOARD_EXECUTABLES.get(backend, ())
        )
    return results


def check_focus_automation(
    snapshot: DiagnosticSnapshot | None = None,
) -> list[CheckResult]:
    """Report detected desktop capabilities without requiring unavailable APIs."""

    config = _resolved(snapshot).config
    if config is not None and config.dictation.inject == "stdout":
        return [
            CheckResult(
                name="focus-automation",
                category="desktop",
                severity=Severity.OK,
                detail=(
                    "dictation.inject is stdout; recognized text is printed "
                    "instead of inserted into the focused window"
                ),
            )
        ]
    desktop = get_desktop()
    if desktop is None:
        return [
            CheckResult(
                name="focus-automation",
                category="desktop",
                severity=Severity.WARNING,
                detail=(
                    "no desktop session was detected; recognized text cannot be "
                    "inserted into a focused window"
                ),
                suggestion=_STDOUT_SUGGESTION,
            )
        ]
    capabilities = desktop.capabilities()
    if capabilities.active_window and capabilities.type_text and capabilities.send_key:
        return [
            CheckResult(
                name="focus-automation",
                category="desktop",
                severity=Severity.OK,
                detail=capabilities.detail,
            )
        ]
    return [
        CheckResult(
            name="focus-automation",
            category="desktop",
            severity=Severity.WARNING,
            detail=capabilities.detail,
            suggestion=_STDOUT_SUGGESTION,
        )
    ]


def check_service_environment(
    _snapshot: DiagnosticSnapshot | None = None,
) -> list[CheckResult]:
    suggestion = "scripts/install.sh  # rerun from the checkout to rebuild it"
    if SERVICE_EXECUTABLE.exists():
        return [
            CheckResult(
                name="venv:service",
                category="python-env",
                severity=Severity.OK,
                detail=f"service environment is present ({SERVICE_EXECUTABLE})",
            )
        ]
    if SERVICE_ENVIRONMENT.exists():
        return [
            CheckResult(
                name="venv:service",
                category="python-env",
                severity=Severity.WARNING,
                detail=(
                    f"service environment {SERVICE_ENVIRONMENT} exists but its "
                    f"console script is missing ({SERVICE_EXECUTABLE}); the "
                    "environment may be incomplete"
                ),
                suggestion=suggestion,
            )
        ]
    return [
        CheckResult(
            name="venv:service",
            category="python-env",
            severity=Severity.FATAL,
            detail=(
                f"service environment {SERVICE_ENVIRONMENT} is missing; "
                f"{SERVICE_NAME} cannot start"
            ),
            suggestion=suggestion,
        )
    ]


def check_model_cache(
    snapshot: DiagnosticSnapshot | None = None,
) -> list[CheckResult]:
    if HUGGINGFACE_CACHE.is_dir():
        return [
            CheckResult(
                name="cache:huggingface",
                category="models",
                severity=Severity.OK,
                detail=f"Hugging Face cache present at {HUGGINGFACE_CACHE}",
            )
        ]
    resolved = _resolved(snapshot)
    if resolved.config is None:
        return []
    stt = resolved.config.stt
    return [
        CheckResult(
            name="cache:huggingface",
            category="models",
            severity=Severity.WARNING,
            detail=(
                f"Hugging Face cache {HUGGINGFACE_CACHE} is absent; the "
                f"{stt.backend} model {stt.model} downloads on first service start"
            ),
            suggestion="dictate services start  # downloads the model before use",
        )
    ]


def check_compute_mode(
    snapshot: DiagnosticSnapshot | None = None,
) -> list[CheckResult]:
    resolved = _resolved(snapshot)
    if resolved.config is None:
        return []
    requested = resolved.config.stt.device
    if requested is ComputeDevice.CPU:
        effective = resolve_local_compute(requested, cuda_available=False, label="STT")
        detail = f"STT configured device={requested}; effective device={effective}"
    else:
        detail = (
            f"STT configured device={requested}; the effective device is "
            "resolved at service start without changing the configured mode"
        )
    return [
        CheckResult(
            name="compute:stt",
            category="compute",
            severity=Severity.OK,
            detail=detail,
        )
    ]


def check_cuda(snapshot: DiagnosticSnapshot | None = None) -> list[CheckResult]:
    resolved = _resolved(snapshot)
    if resolved.config is None:
        return []
    requested = resolved.config.stt.device
    if requested is ComputeDevice.CPU:
        return [
            CheckResult(
                name="gpu:cuda",
                category="gpu",
                severity=Severity.OK,
                detail="CUDA tools were not invoked; STT is configured for the CPU",
            )
        ]
    explicit = requested is ComputeDevice.CUDA
    unavailable_severity = Severity.FATAL if explicit else Severity.WARNING
    unavailable_behavior = (
        "stt.device=cuda cannot start"
        if explicit
        else "stt.device=auto falls back to substantially slower CPU transcription"
    )
    if _which("nvidia-smi") is None:
        return [
            CheckResult(
                name="gpu:cuda",
                category="gpu",
                severity=unavailable_severity,
                detail=(
                    "nvidia-smi not found; GPU acceleration for STT is "
                    f"unavailable; {unavailable_behavior}"
                ),
                suggestion=(
                    INSTALL_HINTS["nvidia-smi"]
                    if explicit
                    else "dictate config set stt.device cpu  "
                    "# or install the NVIDIA driver to use CUDA"
                ),
            )
        ]
    process = _run(["nvidia-smi"], timeout=10)
    if process is None or process.returncode:
        detail = "nvidia-smi failed to query the GPU"
        if process is not None and process.stderr.strip():
            detail = f"{detail}: {process.stderr.strip().splitlines()[0]}"
        return [
            CheckResult(
                name="gpu:cuda",
                category="gpu",
                severity=unavailable_severity,
                detail=f"{detail}; {unavailable_behavior}",
                suggestion="nvidia-smi  # confirm the driver is loaded",
            )
        ]
    return [
        CheckResult(
            name="gpu:cuda",
            category="gpu",
            severity=Severity.OK,
            detail=f"nvidia-smi reports a working GPU for STT (device={requested})",
        )
    ]


def check_pipewire_source(
    snapshot: DiagnosticSnapshot | None = None,
) -> list[CheckResult]:
    config = _resolved(snapshot).config
    source = config.dictation.source if config is not None else ""
    source_label = source or "the PipeWire system default source"
    if _which("wpctl") is None:
        return [
            CheckResult(
                name="audio:pipewire",
                category="audio",
                severity=Severity.FATAL,
                detail=(
                    "wpctl not found; cannot enumerate PipeWire capture sources. "
                    f"Dictation captures from {source_label}"
                ),
                suggestion=INSTALL_HINTS["wpctl"],
            )
        ]
    try:
        sources = list_pipewire_capture_sources()
    except UserConfigurationError as exc:
        return [
            CheckResult(
                name="audio:pipewire",
                category="audio",
                severity=Severity.FATAL,
                detail=f"{exc}. Dictation captures from {source_label}",
                suggestion="systemctl --user status pipewire wireplumber",
            )
        ]
    if source and source not in sources:
        return [
            CheckResult(
                name="audio:pipewire",
                category="audio",
                severity=Severity.FATAL,
                detail=(
                    f"configured dictation source {source} was not found among "
                    f"PipeWire capture sources: {', '.join(sources)}"
                ),
                suggestion="dictate config set dictation.source",
            )
        ]
    return [
        CheckResult(
            name="audio:pipewire",
            category="audio",
            severity=Severity.OK,
            detail=f"PipeWire is responding; dictation captures from {source_label}",
        )
    ]


def check_platform_capabilities(
    _snapshot: DiagnosticSnapshot | None = None,
) -> list[CheckResult]:
    """Report service, notification, and desktop session capabilities."""

    results: list[CheckResult] = []
    services = user_services()
    if services.available():
        results.append(
            CheckResult(
                name="platform:services",
                category="platform",
                severity=Severity.OK,
                detail="systemd user service supervision is available",
            )
        )
    else:
        detail = (
            "systemctl is installed but the systemd user bus is not responding"
            if services.binary_available()
            else "systemctl is not installed"
        )
        results.append(
            CheckResult(
                name="platform:services",
                category="platform",
                severity=Severity.WARNING,
                detail=f"{detail}; {SERVICE_NAME} cannot be supervised",
            )
        )
    if notification_service_available():
        results.append(
            CheckResult(
                name="platform:notifications",
                category="platform",
                severity=Severity.OK,
                detail="desktop notifications are available",
            )
        )
    else:
        detail = (
            "notify-send is installed but no usable notification service responded"
            if notification_service_binary_available()
            else "notify-send is not installed"
        )
        results.append(
            CheckResult(
                name="platform:notifications",
                category="platform",
                severity=Severity.WARNING,
                detail=(f"{detail}; dictation continues without desktop notifications"),
            )
        )
    desktop = get_desktop()
    if desktop is None:
        results.append(
            CheckResult(
                name="platform:desktop",
                category="platform",
                severity=Severity.WARNING,
                detail="no desktop session was detected",
            )
        )
        return results
    capabilities = desktop.capabilities()
    results.append(
        CheckResult(
            name="platform:desktop",
            category="platform",
            severity=(
                Severity.OK
                if capabilities.active_window and capabilities.type_text
                else Severity.WARNING
            ),
            detail=capabilities.detail,
        )
    )
    return results


def check_service_unit(
    _snapshot: DiagnosticSnapshot | None = None,
) -> list[CheckResult]:
    name = f"unit:{SERVICE_NAME}"
    if not user_services().available():
        return [
            CheckResult(
                name=name,
                category="systemd",
                severity=Severity.WARNING,
                detail=(
                    "systemd user service supervision is unavailable; "
                    f"{SERVICE_NAME} state was not queried"
                ),
            )
        ]
    installed = (SYSTEMD_USER_DIR / SERVICE_NAME).exists()
    properties = _systemctl_show(
        SERVICE_NAME,
        [
            "LoadState",
            "ActiveState",
            "SubState",
            "Result",
            "NRestarts",
            "UnitFileState",
        ],
    )
    # systemctl show succeeds for unknown units and reports LoadState=not-found.
    if properties.get("LoadState") == "not-found" or (not properties and not installed):
        return [
            CheckResult(
                name=name,
                category="systemd",
                severity=Severity.FATAL,
                detail=(
                    f"{SERVICE_NAME} is installed but systemd has not loaded it"
                    if installed
                    else f"{SERVICE_NAME} is not installed"
                ),
                suggestion="dictate services install",
            )
        ]

    active = properties.get("ActiveState", "unknown")
    sub = properties.get("SubState", "unknown")
    result = properties.get("Result", "unknown")
    unit_file_state = properties.get("UnitFileState", "unknown")
    try:
        restarts = int(properties.get("NRestarts", "0") or "0")
    except ValueError:
        restarts = 0

    if restarts >= RESTART_LOOP_THRESHOLD and active in {"activating", "failed"}:
        return [
            CheckResult(
                name=name,
                category="systemd",
                severity=Severity.FATAL,
                detail=(
                    f"{SERVICE_NAME} has restarted {restarts} times "
                    f"(result={result}); it is likely crash-looping"
                ),
                suggestion="dictate services logs",
                repair=_restart_service_repair(SERVICE_NAME),
            )
        ]
    if active == "failed" or result not in {"success", ""}:
        return [
            CheckResult(
                name=name,
                category="systemd",
                severity=Severity.FATAL,
                detail=f"{SERVICE_NAME} is failed (result={result}, sub={sub})",
                suggestion="dictate services logs",
                repair=_restart_service_repair(SERVICE_NAME),
            )
        ]
    if unit_file_state not in {"enabled", "enabled-runtime"}:
        return [
            CheckResult(
                name=name,
                category="systemd",
                severity=Severity.WARNING,
                detail=f"{SERVICE_NAME} should start at login but is {unit_file_state!r}",
                suggestion="dictate services install",
            )
        ]
    if active != "active":
        return [
            CheckResult(
                name=name,
                category="systemd",
                severity=Severity.WARNING,
                detail=f"{SERVICE_NAME} is an always-on service but is {active}",
                suggestion="dictate services start",
                repair=_restart_service_repair(SERVICE_NAME),
            )
        ]
    return [
        CheckResult(
            name=name,
            category="systemd",
            severity=Severity.OK,
            detail=(
                f"{SERVICE_NAME} active={active} sub={sub} restarts={restarts} "
                f"({unit_file_state})"
            ),
        )
    ]


def check_stt_socket(
    _snapshot: DiagnosticSnapshot | None = None,
) -> list[CheckResult]:
    if socket_ready(STT_SOCKET):
        return [
            CheckResult(
                name="socket:stt",
                category="runtime",
                severity=Severity.OK,
                detail=f"STT socket {STT_SOCKET} is responding",
            )
        ]
    active = _service_active_state(SERVICE_NAME)
    if active == "active":
        return [
            CheckResult(
                name="socket:stt",
                category="runtime",
                severity=Severity.WARNING,
                detail=(
                    f"STT socket {STT_SOCKET} is not responding although "
                    f"{SERVICE_NAME} is active; the model may still be loading"
                ),
                suggestion="dictate services logs",
                repair=_restart_service_repair(SERVICE_NAME),
            )
        ]
    if STT_SOCKET.exists():
        return [
            CheckResult(
                name="socket:stt",
                category="runtime",
                severity=Severity.WARNING,
                detail=(
                    f"STT socket {STT_SOCKET} is stale: the file exists but "
                    f"{SERVICE_NAME} is {active}"
                ),
                suggestion="dictate services start",
                repair=_remove_stale_socket_repair(STT_SOCKET),
            )
        ]
    return [
        CheckResult(
            name="socket:stt",
            category="runtime",
            severity=Severity.WARNING,
            detail=f"STT socket absent; {SERVICE_NAME} is {active} (not running)",
            suggestion="dictate services start",
        )
    ]


def _directory_result(name: str, path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult(
            name=name,
            category="runtime",
            severity=Severity.OK,
            detail=f"{path} does not exist (will be created on next use)",
        )
    try:
        info = path.stat()
    except OSError as exc:
        return CheckResult(
            name=name,
            category="runtime",
            severity=Severity.WARNING,
            detail=redact_diagnostic(f"could not stat {path}: {exc}"),
        )
    if info.st_uid != os.getuid():
        return CheckResult(
            name=name,
            category="runtime",
            severity=Severity.WARNING,
            detail=(
                f"{path} is owned by uid {info.st_uid}, not the current user "
                f"({os.getuid()}); it may be stale state from another session"
            ),
        )
    if info.st_mode & 0o077:
        return CheckResult(
            name=name,
            category="runtime",
            severity=Severity.WARNING,
            detail=f"{path} is group/other accessible (mode {info.st_mode & 0o777:o})",
        )
    return CheckResult(
        name=name,
        category="runtime",
        severity=Severity.OK,
        detail=f"{path} exists with private permissions",
    )


def check_runtime_directories(
    _snapshot: DiagnosticSnapshot | None = None,
) -> list[CheckResult]:
    if not os.environ.get("XDG_RUNTIME_DIR"):
        runtime = CheckResult(
            name="runtime:xdg",
            category="runtime",
            severity=Severity.FATAL,
            detail=(
                "XDG_RUNTIME_DIR is not set, so the dictation client cannot find "
                f"the runtime directory where {SERVICE_NAME} listens"
            ),
            suggestion=(
                "run dictate from your desktop login session, where pam_systemd "
                "sets XDG_RUNTIME_DIR"
            ),
        )
    elif not RUNTIME.exists() or not os.access(RUNTIME, os.W_OK):
        runtime = CheckResult(
            name="runtime:xdg",
            category="runtime",
            severity=Severity.FATAL,
            detail=(
                f"runtime directory {RUNTIME} is missing or not writable; "
                "the STT socket and recordings cannot be created"
            ),
        )
    else:
        runtime = CheckResult(
            name="runtime:xdg",
            category="runtime",
            severity=Severity.OK,
            detail=f"runtime directory {RUNTIME} is writable",
        )
    return [runtime, _directory_result("runtime:state", STATE_DIR)]


def check_process_capabilities(
    _snapshot: DiagnosticSnapshot | None = None,
) -> list[CheckResult]:
    issues = capability_diagnostics()
    if not issues:
        return [
            CheckResult(
                name="runtime:process-ownership",
                category="runtime",
                severity=Severity.OK,
                detail="Linux process identity and pidfd ownership are available",
            )
        ]
    return [
        CheckResult(
            name="runtime:process-ownership",
            category="runtime",
            severity=Severity.WARNING,
            detail="; ".join(issues),
        )
    ]


ALL_CHECKS: tuple[Check, ...] = (
    check_configuration,
    check_executables,
    check_focus_automation,
    check_service_environment,
    check_model_cache,
    check_compute_mode,
    check_cuda,
    check_pipewire_source,
    check_service_unit,
    check_stt_socket,
    check_runtime_directories,
    check_process_capabilities,
    check_platform_capabilities,
)
