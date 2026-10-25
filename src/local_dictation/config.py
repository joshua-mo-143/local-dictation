from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path

APP_NAME = "local-dictation"
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
RUNTIME = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))


def runtime_dir(environment: Mapping[str, str] = os.environ) -> Path:
    """Directory for recordings, server state, and the recording lock.

    ``DICTATION_RUNTIME_DIR`` gives a development checkout its own directory so
    it never touches an installed service.
    """

    configured = environment.get("DICTATION_RUNTIME_DIR", "")
    path = Path(configured) if configured else None
    if path is not None and path.is_absolute():
        return path
    return Path(environment.get("XDG_RUNTIME_DIR", "/tmp")) / APP_NAME


def socket_path(directory: Path) -> Path:
    """The server socket lives next to its runtime directory."""

    return directory.with_name(f"{directory.name}.sock")


def xdg_config_home(
    environment: Mapping[str, str] = os.environ,
    *,
    home: Path | None = None,
) -> Path:
    configured = environment.get("XDG_CONFIG_HOME")
    path = Path(configured) if configured else None
    if path is not None and path.is_absolute():
        return path
    return (home or Path.home()) / ".config"


def config_dir(
    environment: Mapping[str, str] = os.environ,
    *,
    home: Path | None = None,
) -> Path:
    return xdg_config_home(environment, home=home) / APP_NAME


def vocabulary_path(
    environment: Mapping[str, str] = os.environ,
    *,
    home: Path | None = None,
) -> Path:
    return config_dir(environment, home=home) / "vocabulary.json"


def read_toml_table(
    config_path: Path,
    *,
    error: type[ValueError] = ValueError,
    label: str = "configuration",
) -> dict[str, object]:
    """Read ``config_path`` into a TOML table, tolerating a missing file."""

    try:
        raw = tomllib.loads(config_path.read_text())
    except FileNotFoundError:
        raw = {}
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise error(f"could not read {label} {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise error(f"{label} must be a TOML table")
    return raw


STATE_DIR = runtime_dir()
STT_SOCKET = socket_path(STATE_DIR)
WAV_PATH = STATE_DIR / "recording.wav"
PID_PATH = STATE_DIR / "recording.pid"
RECORDER_LOG = STATE_DIR / "pw-record.log"
VOCABULARY_PATH = vocabulary_path()
DEFAULT_SOURCE = ""

SYSTEMD_USER_DIR = xdg_config_home() / "systemd" / "user"
SERVICE_NAME = f"{APP_NAME}.service"
