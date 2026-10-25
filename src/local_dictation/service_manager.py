"""Install and control the local-dictation systemd user service."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .config import PROJECT_ROOT, SERVICE_NAME, SYSTEMD_USER_DIR
from .errors import DictationError
from .platform_services import user_services
from .service_units import SOURCE_RELATIVE, audit_installed, packaged_unit_text
from .user_config import load_user_config

LEGACY_SERVICE_NAME = "dictation.service"


def systemctl(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Delegate user-service commands through the platform supervisor."""

    return user_services().run(*arguments, check=check)


def _require_systemctl(*arguments: str) -> None:
    process = systemctl(*arguments, check=False)
    if process.returncode:
        detail = (
            process.stderr.strip()
            or process.stdout.strip()
            or f"exit status {process.returncode}"
        )
        raise DictationError(f"systemctl --user {' '.join(arguments)} failed: {detail}")


def unit_text() -> str:
    """Return the shipped unit: the checkout copy if present, else the packaged one."""

    source = PROJECT_ROOT / SOURCE_RELATIVE / SERVICE_NAME
    if source.is_file():
        return source.read_text()
    return packaged_unit_text()


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError):
        return None


def install_service(*, force: bool = False) -> None:
    destination = SYSTEMD_USER_DIR / SERVICE_NAME
    expected = unit_text()
    if destination.is_dir() and not destination.is_symlink():
        raise DictationError(f"refusing to replace directory {destination}")
    if destination.exists() or destination.is_symlink():
        if _read(destination) == expected:
            print(f"{destination} is already up to date.")
        elif not force:
            raise DictationError(
                f"{destination} differs from the shipped unit; review it, then run "
                "`dictate services install --force` to replace it"
            )
        else:
            destination.unlink()
            destination.write_text(expected)
            print(f"Replaced {destination}.")
    else:
        SYSTEMD_USER_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)
        destination.write_text(expected)
        print(f"Installed {destination}.")
    _require_systemctl("daemon-reload")
    _require_systemctl("enable", SERVICE_NAME)
    print(f"Enabled {SERVICE_NAME}. Start it with `dictate services start`.")
    legacy = SYSTEMD_USER_DIR / LEGACY_SERVICE_NAME
    if legacy.exists() or legacy.is_symlink():
        print(
            f"Left {legacy} as it is. Once local-dictation replaces it, stop it "
            f"with `systemctl --user disable --now {LEGACY_SERVICE_NAME}`."
        )


def start_service() -> None:
    _require_systemctl("start", SERVICE_NAME)
    print(f"Started {SERVICE_NAME}.")


def stop_service() -> None:
    _require_systemctl("stop", SERVICE_NAME)
    print(f"Stopped {SERVICE_NAME}.")


def restart_service() -> None:
    _require_systemctl("restart", SERVICE_NAME)
    print(f"Restarted {SERVICE_NAME}.")


def status() -> None:
    active = systemctl("is-active", SERVICE_NAME, check=False).stdout.strip()
    enabled = systemctl("is-enabled", SERVICE_NAME, check=False).stdout.strip()
    print(f"{SERVICE_NAME}: {active or 'unknown'} ({enabled or 'not-found'})")


def audit_service() -> int:
    """Read the effective installed unit and runtime state without changing them."""

    return audit_installed(config=load_user_config())


def logs(*, follow: bool, lines: int) -> None:
    command = ["journalctl", "--user", "-u", SERVICE_NAME, "-n", str(lines)]
    if follow:
        command.append("-f")
    raise SystemExit(subprocess.run(command, check=False).returncode)


def uninstall_service() -> None:
    systemctl("stop", SERVICE_NAME, check=False)
    systemctl("disable", SERVICE_NAME, check=False)
    destination = SYSTEMD_USER_DIR / SERVICE_NAME
    if not (destination.exists() or destination.is_symlink()):
        print(f"{destination} is not installed.")
    elif _read(destination) == unit_text():
        destination.unlink()
        print(f"Removed {destination}.")
    else:
        print(
            f"Kept {destination} because it differs from the shipped unit; "
            "remove it yourself after reviewing it."
        )
    _require_systemctl("daemon-reload")
