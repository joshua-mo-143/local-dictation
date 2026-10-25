"""Validate the shipped systemd user unit and audit the installed one."""

from __future__ import annotations

import argparse
import importlib.resources
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .config import PROJECT_ROOT, SERVICE_NAME, runtime_dir, socket_path
from .platform_services import user_services
from .stt import launcher
from .user_config import UserConfig, load_user_config

SOURCE_RELATIVE = Path("systemd/user")
PACKAGED_RELATIVE = ("data", "systemd")
INSTALL_ROOT = "%h/local-dictation"
EXPECTED_EXECSTART = f"{INSTALL_ROOT}/.venv-service/bin/local-dictation-server"
SERVER_ENTRY_POINT = "local_dictation.stt.launcher:main"
SERVICE_EXTRAS = ("cpu", "cuda")
EXTERNAL_UNIT_STUBS = {"default.target", "graphical-session.target"}
SECURITY_POLICY = {
    "Type": "simple",
    "WorkingDirectory": INSTALL_ROOT,
    "UMask": "0077",
    "Restart": "on-failure",
    "RestartSec": "5",
    "TimeoutStopSec": "15",
    "TasksMax": "512",
    "LimitNOFILE": "4096",
    "LimitCORE": "0",
    "MemoryHigh": "4G",
    "MemoryMax": "6G",
    "MemorySwapMax": "infinity",
    "OOMPolicy": "stop",
    "OOMScoreAdjust": "200",
    "NoNewPrivileges": "true",
    "CapabilityBoundingSet": "",
    "AmbientCapabilities": "",
    "PrivateTmp": "true",
    "ProtectSystem": "strict",
    "ProtectHome": "read-only",
    "ReadWritePaths": "%t",
    "ProtectControlGroups": "true",
    "ProtectKernelModules": "true",
    "ProtectKernelTunables": "true",
    "ProtectKernelLogs": "true",
    "ProtectClock": "true",
    "ProtectHostname": "true",
    "ProtectProc": "invisible",
    "LockPersonality": "true",
    "RestrictNamespaces": "true",
    "RestrictRealtime": "true",
    "RestrictSUIDSGID": "true",
    "RestrictAddressFamilies": "AF_UNIX AF_INET AF_INET6 AF_NETLINK",
    "SystemCallArchitectures": "native",
    "StartLimitIntervalSec": "60",
    "StartLimitBurst": "5",
    "RuntimeDirectory": "local-dictation",
    "RuntimeDirectoryMode": "0700",
    "RuntimeDirectoryPreserve": "yes",
    "CacheDirectory": "huggingface",
    "CacheDirectoryMode": "0700",
}
UNIT_DIRECTIVES = {"StartLimitIntervalSec", "StartLimitBurst"}
INTENTIONALLY_UNUSED_DIRECTIVES = {
    "DevicePolicy",
    "IPAddressDeny",
    "MemoryDenyWriteExecute",
    "SystemCallFilter",
}
CUDA_DEVICE_RESTRICTIONS = {"DeviceAllow", "DeviceDeny", "PrivateDevices"}
REQUIRED_ENVIRONMENT = {
    "CUDA_CACHE_PATH": "%t/local-dictation/cuda-cache",
    "HF_HOME": "%h/.cache/huggingface",
    "TMPDIR": "%t/local-dictation",
}
AUDIT_SHOW_PROPERTIES = (
    "ActiveState",
    "SubState",
    "Result",
    "NRestarts",
    "ExecMainStatus",
    "MemoryCurrent",
    "TasksCurrent",
    "MemoryHigh",
    "MemoryMax",
    "MemorySwapMax",
    "TasksMax",
    "LimitNOFILE",
    "LimitCORE",
    "UMask",
    "Restart",
    "RestartUSec",
    "TimeoutStopUSec",
    "WorkingDirectory",
    "Type",
    "PrivateTmp",
    "NoNewPrivileges",
    "ProtectSystem",
    "ProtectHome",
    "ProtectProc",
    "ProcSubset",
    "ProtectControlGroups",
    "ProtectKernelModules",
    "ProtectKernelTunables",
    "ProtectKernelLogs",
    "ProtectClock",
    "ProtectHostname",
    "PrivateDevices",
    "DevicePolicy",
    "CapabilityBoundingSet",
    "AmbientCapabilities",
    "LockPersonality",
    "MemoryDenyWriteExecute",
    "RestrictNamespaces",
    "RestrictSUIDSGID",
    "RestrictRealtime",
    "RestrictAddressFamilies",
    "SystemCallArchitectures",
    "RuntimeDirectory",
    "RuntimeDirectoryMode",
    "RuntimeDirectoryPreserve",
    "StateDirectory",
    "StateDirectoryMode",
    "CacheDirectory",
    "CacheDirectoryMode",
    "ReadWritePaths",
    "InaccessiblePaths",
    "OOMPolicy",
    "OOMScoreAdjust",
    "StartLimitIntervalUSec",
    "StartLimitBurst",
    "Environment",
    "EnvironmentFiles",
    "ExecStart",
)
_SET_VALUED_PROPERTIES = {
    "RestrictAddressFamilies",
    "RuntimeDirectory",
    "ReadWritePaths",
    "InaccessiblePaths",
    "CapabilityBoundingSet",
    "AmbientCapabilities",
}


_USER_CONTEXT_UNSUPPORTED = re.compile(
    r"Failed to initialize unit search paths.*Invalid argument"
    r"|Failed to lookup RuntimeDirectory path"
    r"|Failed to initialize manager"
    r"|Failed to connect to bus",
    re.DOTALL,
)


def user_context_unsupported(stderr: str) -> bool:
    """Return whether systemd-analyze could not initialize rooted user context."""

    return bool(_USER_CONTEXT_UNSUPPORTED.search(stderr))


@dataclass(frozen=True)
class StagedVerification:
    process: subprocess.CompletedProcess[str]
    context: str
    rooted_user_context: str
    reason: str | None = None

    @property
    def returncode(self) -> int:
        return self.process.returncode

    @property
    def stdout(self) -> str:
        return self.process.stdout

    @property
    def stderr(self) -> str:
        return self.process.stderr


def packaged_unit_text() -> str:
    resource = importlib.resources.files("local_dictation").joinpath(
        *PACKAGED_RELATIVE, SERVICE_NAME
    )
    return resource.read_text()


def _source_unit(project_root: Path) -> str:
    return (project_root / SOURCE_RELATIVE / SERVICE_NAME).read_text()


def parity_errors(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Check that the checkout and wheel ship exactly one identical unit."""

    source_names = {
        path.name for path in (project_root / SOURCE_RELATIVE).glob("*.service")
    }
    packaged_names = {
        entry.name
        for entry in importlib.resources.files("local_dictation")
        .joinpath(*PACKAGED_RELATIVE)
        .iterdir()
        if entry.name.endswith(".service")
    }
    errors: list[str] = []
    for label, actual in (("source", source_names), ("packaged", packaged_names)):
        if SERVICE_NAME not in actual:
            errors.append(f"{label} units missing: {SERVICE_NAME}")
        extra = sorted(actual - {SERVICE_NAME})
        if extra:
            errors.append(f"{label} units unexpected: {', '.join(extra)}")
    if SERVICE_NAME in source_names & packaged_names:
        if _source_unit(project_root) != packaged_unit_text():
            errors.append(f"source and packaged units differ: {SERVICE_NAME}")
    return errors


def _directive_values(text: str, name: str, *, section: str | None = None) -> list[str]:
    """Collect assignments in file order, optionally within one unit section."""

    current_section: str | None = None
    values: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            continue
        if section is not None and current_section != section:
            continue
        key, separator, value = line.partition("=")
        if separator and key == name:
            values.append(value.strip())
    return values


def _directive(text: str, name: str, *, section: str | None = None) -> str | None:
    """Return the effective value for a scalar systemd assignment."""

    values = _directive_values(text, name, section=section)
    return values[-1] if values else None


def _environment_values(text: str, variable: str) -> list[str]:
    """Collect every explicit assignment of one Environment= variable."""

    values: list[str] = []
    for directive in _directive_values(text, "Environment", section="Service"):
        if not directive:
            values.append("<reset>")
            continue
        for assignment in shlex.split(directive):
            key, separator, value = assignment.partition("=")
            if separator and key == variable:
                values.append(value)
    return values


def _environment_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for directive in _directive_values(text, "Environment", section="Service"):
        for assignment in shlex.split(directive):
            key, separator, _value = assignment.partition("=")
            if separator:
                keys.add(key)
    return keys


def _effective_list_values(text: str, name: str, *, section: str) -> list[str]:
    """Apply systemd's empty-assignment reset semantics for list directives."""

    effective: list[str] = []
    for value in _directive_values(text, name, section=section):
        if value:
            effective.append(value)
        else:
            effective.clear()
    return effective


def _expand_specifiers(value: str, *, home: str, runtime: str) -> str:
    return value.replace("%h", home).replace("%t", runtime)


def consistency_errors(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Check the unit against the package metadata and code it launches."""

    text = _source_unit(project_root)
    project = tomllib.loads((project_root / "pyproject.toml").read_text())["project"]
    errors: list[str] = []

    executable = Path(EXPECTED_EXECSTART).name
    if project.get("scripts", {}).get(executable) != SERVER_ENTRY_POINT:
        errors.append(
            f"pyproject.toml must declare {executable} = {SERVER_ENTRY_POINT!r} "
            f"for {SERVICE_NAME} ExecStart"
        )
    extras = project.get("optional-dependencies", {})
    for extra in SERVICE_EXTRAS:
        if not any(str(item).startswith("onnx-asr") for item in extras.get(extra, [])):
            errors.append(
                f"{extra} extra does not install the default parakeet backend"
            )

    protected = launcher.protected_environment({}, uid=os.getuid())
    runtime = Path(protected["XDG_RUNTIME_DIR"])
    state = runtime_dir({"XDG_RUNTIME_DIR": str(runtime)})
    runtime_directory = _directive(text, "RuntimeDirectory", section="Service")
    if runtime_directory is None or runtime / runtime_directory != state:
        errors.append(
            f"{SERVICE_NAME} RuntimeDirectory must be the directory dictate and "
            f"the server derive ({state.name}), got {runtime_directory!r}"
        )
    if socket_path(state).parent != runtime:
        errors.append(
            f"the server socket {socket_path(state)} must be directly in %t, "
            "which ReadWritePaths=%t keeps writable"
        )
    for variable in sorted(
        _environment_keys(text) & launcher.PROTECTED_ENVIRONMENT_KEYS
    ):
        value = _expand_specifiers(
            _environment_values(text, variable)[-1],
            home=protected["HOME"],
            runtime=str(runtime),
        )
        if value != protected[variable]:
            errors.append(
                f"{SERVICE_NAME} Environment={variable} must match the value the "
                f"launcher forces ({protected[variable]!r}), got {value!r}"
            )
    return errors


def security_errors(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Validate controls without ignoring later systemd assignments or resets."""

    text = _source_unit(project_root)
    errors: list[str] = []
    for directive, expected in SECURITY_POLICY.items():
        section = "Unit" if directive in UNIT_DIRECTIVES else "Service"
        values = _directive_values(text, directive, section=section)
        if len(values) != 1:
            errors.append(
                f"{SERVICE_NAME} {directive} must have exactly one assignment, "
                f"got {values!r}"
            )
        elif values[0] != expected:
            errors.append(
                f"{SERVICE_NAME} {directive} must be {expected!r}, got {values[0]!r}"
            )
    for directive in sorted(INTENTIONALLY_UNUSED_DIRECTIVES):
        if _directive_values(text, directive, section="Service"):
            errors.append(
                f"{SERVICE_NAME} must not set compatibility-sensitive {directive}"
            )
    for directive in sorted(CUDA_DEVICE_RESTRICTIONS):
        if _directive_values(text, directive, section="Service"):
            errors.append(
                f"{SERVICE_NAME} must not restrict CUDA devices with {directive}"
            )
    for variable, expected in REQUIRED_ENVIRONMENT.items():
        values = _environment_values(text, variable)
        if values != [expected]:
            errors.append(
                f"{SERVICE_NAME} Environment={variable} must be assigned exactly "
                f"once to {expected!r}, got {values!r}"
            )
    environment_keys = _environment_keys(text)
    if environment_keys != set(REQUIRED_ENVIRONMENT):
        errors.append(
            f"{SERVICE_NAME} Environment keys must be exactly "
            f"{sorted(REQUIRED_ENVIRONMENT)!r}, got {sorted(environment_keys)!r}"
        )
    if _directive_values(text, "EnvironmentFile", section="Service"):
        errors.append(
            f"{SERVICE_NAME} must not import unrestricted EnvironmentFile values"
        )
    commands = _effective_list_values(text, "ExecStart", section="Service")
    if commands != [EXPECTED_EXECSTART]:
        errors.append(
            f"{SERVICE_NAME} effective ExecStart must be "
            f"{EXPECTED_EXECSTART!r}, got {commands!r}"
        )
    return errors


def stage_executable_stub(root: Path, executable: str) -> Path:
    """Create an executable stub without permitting traversal outside ``root``."""

    resolved_root = root.resolve(strict=True)
    components = executable.split("/")
    if (
        not executable.startswith("/")
        or any(component in {"", ".", ".."} for component in components[1:])
        or Path(executable).as_posix() != executable
    ):
        raise ValueError(f"unsafe staged executable path: {executable!r}")

    destination = resolved_root.joinpath(*components[1:])
    current = resolved_root
    for component in components[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(
                f"staged executable path contains a symlink: {executable!r}"
            )

    resolved_destination = destination.resolve(strict=False)
    if not resolved_destination.is_relative_to(resolved_root):
        raise ValueError(f"staged executable escapes verification root: {executable!r}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError(f"staged executable destination is a symlink: {executable!r}")
    destination.touch()
    destination.chmod(0o755)
    return destination


def systemd_analyze(
    paths: Iterable[Path], *, executable: str | None = None
) -> StagedVerification | None:
    """Verify units in an isolated root with deterministic dependency/command stubs."""

    command = executable or shutil.which("systemd-analyze")
    if command is None:
        return None
    unit_paths = tuple(paths)
    expected_executable = shlex.split(EXPECTED_EXECSTART)[0]
    with tempfile.TemporaryDirectory(prefix="local-dictation-systemd-") as temporary:
        root = Path(temporary)
        unit_dir = root / "etc/systemd/user"
        unit_dir.mkdir(parents=True)
        system_unit_dir = root / "etc/systemd/system"
        system_unit_dir.mkdir(parents=True)
        staged_paths: list[Path] = []
        command_paths: set[str] = set()
        for source in unit_paths:
            text = source.read_text()
            staged = unit_dir / source.name
            staged_text = f"{text}\n[Unit]\nDefaultDependencies=no\n"
            staged.write_text(staged_text)
            (system_unit_dir / source.name).write_text(staged_text)
            staged_paths.append(staged)
            for match in re.finditer(
                r"^Exec(?:Condition|StartPre|Start|StartPost|Reload|Stop|StopPost)=(.*)$",
                text,
                re.MULTILINE,
            ):
                arguments = shlex.split(match.group(1))
                if not arguments:
                    continue
                if arguments[0].lstrip("-+!:@") == expected_executable:
                    executable_path = expected_executable.replace(
                        "%h", str(Path.home()), 1
                    )
                    if Path(executable_path).is_absolute():
                        command_paths.add(executable_path)

        for name in EXTERNAL_UNIT_STUBS:
            stub_text = (
                "[Unit]\n"
                f"Description=Verification stub for {name}\n"
                "DefaultDependencies=no\n"
            )
            for directory in (unit_dir, system_unit_dir):
                (directory / name).write_text(stub_text)
        for path in command_paths:
            stage_executable_stub(root, path)

        common_arguments = [
            f"--root={root}",
            "--man=no",
            "--generators=no",
            "verify",
            *(path.name for path in staged_paths),
        ]
        result = subprocess.run(
            [command, "--user", *common_arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode and user_context_unsupported(result.stderr):
            user_error = result.stderr.strip()
            fallback = subprocess.run(
                [command, *common_arguments],
                capture_output=True,
                text=True,
                check=False,
            )
            return StagedVerification(
                fallback,
                context="staged-system",
                rooted_user_context="unsupported",
                reason=user_error,
            )
        return StagedVerification(
            result,
            context="staged-user",
            rooted_user_context="supported",
        )


def _memory_bytes(value: str) -> str:
    match = re.fullmatch(r"(\d+)([KMGT])", value)
    if match is None:
        return value
    powers = {"K": 1, "M": 2, "G": 3, "T": 4}
    return str(int(match.group(1)) * 1024 ** powers[match.group(2)])


def _expanded_unit_value(value: str) -> str:
    runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return _expand_specifiers(value, home=str(Path.home()), runtime=runtime)


def _audit_expected_properties() -> dict[str, str]:
    policy = SECURITY_POLICY

    def yes_no(directive: str) -> str:
        return "yes" if policy[directive] == "true" else "no"

    return {
        "Type": policy["Type"],
        "MemoryHigh": _memory_bytes(policy["MemoryHigh"]),
        "MemoryMax": _memory_bytes(policy["MemoryMax"]),
        "MemorySwapMax": policy["MemorySwapMax"],
        "TasksMax": policy["TasksMax"],
        "LimitNOFILE": policy["LimitNOFILE"],
        "LimitCORE": policy["LimitCORE"],
        "UMask": policy["UMask"],
        "Restart": policy["Restart"],
        "RestartUSec": f"{policy['RestartSec']}s",
        "TimeoutStopUSec": f"{policy['TimeoutStopSec']}s",
        "WorkingDirectory": _expanded_unit_value(policy["WorkingDirectory"]),
        "OOMPolicy": policy["OOMPolicy"],
        "OOMScoreAdjust": policy["OOMScoreAdjust"],
        "PrivateTmp": yes_no("PrivateTmp"),
        "NoNewPrivileges": yes_no("NoNewPrivileges"),
        "ProtectSystem": policy["ProtectSystem"],
        "ProtectHome": policy["ProtectHome"],
        "ProtectProc": policy["ProtectProc"],
        "ProcSubset": "all",
        "ProtectControlGroups": yes_no("ProtectControlGroups"),
        "ProtectKernelModules": yes_no("ProtectKernelModules"),
        "ProtectKernelTunables": yes_no("ProtectKernelTunables"),
        "ProtectKernelLogs": yes_no("ProtectKernelLogs"),
        "ProtectClock": yes_no("ProtectClock"),
        "ProtectHostname": yes_no("ProtectHostname"),
        "PrivateDevices": "no",
        "DevicePolicy": "auto",
        "CapabilityBoundingSet": policy["CapabilityBoundingSet"],
        "AmbientCapabilities": policy["AmbientCapabilities"],
        "LockPersonality": yes_no("LockPersonality"),
        "MemoryDenyWriteExecute": "no",
        "RestrictNamespaces": yes_no("RestrictNamespaces"),
        "RestrictSUIDSGID": yes_no("RestrictSUIDSGID"),
        "RestrictRealtime": yes_no("RestrictRealtime"),
        "RestrictAddressFamilies": policy["RestrictAddressFamilies"],
        "SystemCallArchitectures": policy["SystemCallArchitectures"],
        "RuntimeDirectory": policy["RuntimeDirectory"],
        "RuntimeDirectoryMode": policy["RuntimeDirectoryMode"],
        "RuntimeDirectoryPreserve": policy["RuntimeDirectoryPreserve"],
        "StateDirectory": "",
        "StateDirectoryMode": "0755",
        "CacheDirectory": policy["CacheDirectory"],
        "CacheDirectoryMode": policy["CacheDirectoryMode"],
        "ReadWritePaths": _expanded_unit_value(policy["ReadWritePaths"]),
        "InaccessiblePaths": "",
        "EnvironmentFiles": "",
        "StartLimitIntervalUSec": "1min",
        "StartLimitBurst": policy["StartLimitBurst"],
    }


def _show_properties(text: str) -> dict[str, str]:
    properties: dict[str, str] = {}
    for line in text.splitlines():
        name, separator, value = line.partition("=")
        if separator:
            properties[name] = value
    return properties


def _show_exec_argv(value: str) -> list[str] | None:
    match = re.search(r"(?:^|;\s*)argv\[\]=(.*?)(?:\s*;\s*ignore_errors=|$)", value)
    if match is None:
        return None
    return shlex.split(match.group(1))


def _installed_unit_errors() -> list[str]:
    services = user_services()
    cat = services.run("cat", SERVICE_NAME, check=False)
    if cat.returncode:
        return [f"{SERVICE_NAME}: systemctl cat failed: {cat.stderr.strip()}"]
    errors: list[str] = []
    commands = _effective_list_values(cat.stdout, "ExecStart", section="Service")
    if commands != [EXPECTED_EXECSTART]:
        errors.append(
            f"{SERVICE_NAME}: effective ExecStart must be "
            f"{EXPECTED_EXECSTART!r}, got {commands!r}"
        )
    if _effective_list_values(cat.stdout, "EnvironmentFile", section="Service"):
        errors.append(
            f"{SERVICE_NAME}: effective EnvironmentFile must be empty; "
            "the shipped service does not allow environment files"
        )
    for directive in ("IPAddressDeny", "SystemCallFilter"):
        if _effective_list_values(cat.stdout, directive, section="Service"):
            errors.append(f"{SERVICE_NAME}: effective {directive} must be empty")
    for directive in ("DeviceAllow", "DeviceDeny"):
        if _effective_list_values(cat.stdout, directive, section="Service"):
            errors.append(
                f"{SERVICE_NAME}: effective {directive} must not restrict CUDA"
            )

    shown = services.run(
        "show",
        SERVICE_NAME,
        *(f"--property={prop}" for prop in AUDIT_SHOW_PROPERTIES),
        check=False,
    )
    if shown.returncode:
        errors.append(f"{SERVICE_NAME}: systemctl show failed: {shown.stderr.strip()}")
        return errors
    properties = _show_properties(shown.stdout)
    for prop, expected in _audit_expected_properties().items():
        # systemctl show leaves out some empty properties, such as
        # EnvironmentFiles, even with --all.
        actual = properties.get(prop, "" if expected == "" else None)
        if prop in _SET_VALUED_PROPERTIES:
            matches = set((actual or "").split()) == set(expected.split())
        else:
            matches = actual == expected
        if not matches:
            errors.append(
                f"{SERVICE_NAME}: effective {prop} must be {expected!r}, got {actual!r}"
            )
    expected_argv = shlex.split(_expanded_unit_value(EXPECTED_EXECSTART))
    actual_argv = _show_exec_argv(properties.get("ExecStart", ""))
    if actual_argv != expected_argv:
        errors.append(
            f"{SERVICE_NAME}: effective ExecStart argv must be {expected_argv!r}, "
            f"got {actual_argv!r}"
        )
    elif not os.access(expected_argv[0], os.X_OK):
        errors.append(
            f"{SERVICE_NAME}: {expected_argv[0]} is missing or not executable; "
            "run scripts/install.sh to create the service environment"
        )

    actual_environment = {
        key: value
        for assignment in shlex.split(properties.get("Environment", ""))
        for key, separator, value in [assignment.partition("=")]
        if separator
    }
    unknown_environment = set(actual_environment) - set(REQUIRED_ENVIRONMENT)
    missing_environment = set(REQUIRED_ENVIRONMENT) - set(actual_environment)
    if unknown_environment:
        errors.append(
            f"{SERVICE_NAME}: effective Environment has unsupported keys "
            f"{sorted(unknown_environment)!r}"
        )
    if missing_environment:
        errors.append(
            f"{SERVICE_NAME}: effective Environment lacks required keys "
            f"{sorted(missing_environment)!r}"
        )
    for variable, expected in REQUIRED_ENVIRONMENT.items():
        actual = actual_environment.get(variable)
        if actual is not None and actual not in {
            expected,
            _expanded_unit_value(expected),
        }:
            errors.append(
                f"{SERVICE_NAME}: effective {variable} must be {expected}, "
                f"got {actual!r}"
            )

    active = properties.get("ActiveState", "unknown")
    result = properties.get("Result", "unknown")
    restarts = properties.get("NRestarts", "unknown")
    if active == "failed" or result not in {"success", ""}:
        errors.append(
            f"{SERVICE_NAME}: unhealthy state ActiveState={active} Result={result}"
        )
    try:
        if int(restarts) >= int(SECURITY_POLICY["StartLimitBurst"]) and active in {
            "activating",
            "failed",
        }:
            errors.append(
                f"{SERVICE_NAME}: restart count {restarts} indicates an active "
                "crash loop"
            )
    except ValueError:
        errors.append(f"{SERVICE_NAME}: invalid NRestarts={restarts!r}")
    print(
        f"{SERVICE_NAME}: active={active} sub={properties.get('SubState', 'unknown')} "
        f"result={result} restarts={restarts} "
        f"memory={properties.get('MemoryCurrent', 'unknown')} "
        f"tasks={properties.get('TasksCurrent', 'unknown')}"
    )
    return errors


def audit_installed(*, config: UserConfig | None = None) -> int:
    """Read the effective installed unit and state without changing the manager."""

    resolved = config if config is not None else load_user_config()
    errors = _installed_unit_errors()
    print(
        "configuration: "
        f"device={resolved.stt.device} "
        f"backend={resolved.stt.backend} "
        f"model={resolved.stt.model}"
    )
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Validate the shipped {SERVICE_NAME} systemd unit"
    )
    parser.add_argument(
        "--require-systemd-analyze",
        action="store_true",
        help="fail instead of skipping when systemd-analyze is unavailable",
    )
    parser.add_argument(
        "--audit-installed",
        action="store_true",
        help="read and validate the effective installed unit without changing it",
    )
    parser.add_argument(
        "--require-user-context",
        action="store_true",
        help="fail unless rooted staged verification ran in systemd user context",
    )
    options = parser.parse_args(arguments)

    if options.audit_installed:
        return audit_installed()

    errors = [*parity_errors(), *consistency_errors(), *security_errors()]
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    analyzed = systemd_analyze([PROJECT_ROOT / SOURCE_RELATIVE / SERVICE_NAME])
    if analyzed is None:
        if options.require_systemd_analyze or options.require_user_context:
            print("error: systemd-analyze is unavailable", file=sys.stderr)
            return 1
        print("systemd-analyze unavailable; syntax verification skipped")
        return 0
    print(
        json.dumps(
            {
                "systemd_verification": {
                    "context": analyzed.context,
                    "rooted_user_context": analyzed.rooted_user_context,
                    "reason": analyzed.reason,
                }
            },
            sort_keys=True,
        )
    )
    if options.require_user_context and analyzed.rooted_user_context != "supported":
        print(
            "error: rooted staged systemd user-context verification is unsupported",
            file=sys.stderr,
        )
        return 1
    if analyzed.returncode:
        sys.stderr.write(analyzed.stdout)
        sys.stderr.write(analyzed.stderr)
        return analyzed.returncode
    print(f"validated source and packaged {SERVICE_NAME} in {analyzed.context} context")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
