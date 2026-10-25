"""systemd user-session supervision."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence


class SystemdUserSupervisor:
    """Keeps systemd command construction in one place."""

    CAPABILITY_PROBE_TIMEOUT_SECONDS = 2

    def __init__(self, *, systemctl: str = "systemctl") -> None:
        self.systemctl = systemctl

    def binary_available(self) -> bool:
        return shutil.which(self.systemctl) is not None

    def available(self) -> bool:
        if not self.binary_available():
            return False
        try:
            process = subprocess.run(
                [self.systemctl, "--user", "show-environment"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=self.CAPABILITY_PROBE_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return process.returncode == 0

    def run(
        self, *arguments: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.systemctl, "--user", *arguments],
            capture_output=True,
            text=True,
            check=check,
        )

    def start(
        self, *units: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return self.run("start", *units, check=check)

    def stop(
        self, *units: str, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return self.run("stop", *units, check=check)

    def enable(
        self, *units: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return self.run("enable", *units, check=check)

    def disable(
        self, *units: str, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return self.run("disable", *units, check=check)

    def reload(self, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return self.run("daemon-reload", check=check)

    def is_active(self, unit: str) -> str:
        process = self.run("is-active", unit, check=False)
        return process.stdout.strip() or "inactive"

    def show(self, unit: str, properties: Sequence[str]) -> dict[str, str]:
        arguments = ["show", unit, "--no-pager"]
        arguments.extend(f"--property={name}" for name in properties)
        process = self.run(*arguments, check=False)
        values: dict[str, str] = {}
        if process.returncode:
            return values
        for line in process.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value
        return values

    def user_environment(self) -> dict[str, str]:
        process = self.run("show-environment", check=False)
        values: dict[str, str] = {}
        if process.returncode:
            return values
        for line in process.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator and key and value:
                values[key] = value
        return values

    def restart(
        self, unit: str, *, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return self.run("restart", unit, check=check)


def user_services() -> SystemdUserSupervisor:
    return SystemdUserSupervisor()
