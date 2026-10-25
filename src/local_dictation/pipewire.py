"""PipeWire capture-source discovery through ``wpctl``."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence

from .user_config import UserConfigurationError


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


def pipewire_section_devices(status: str, section: str) -> tuple[str, ...]:
    """Return device names listed under a ``wpctl status`` Audio section."""

    lines = status.splitlines()
    collecting = False
    names: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Audio"):
            collecting = False
        if stripped.rstrip(":") in {
            section,
            f"├─ {section}",
            f"└─ {section}",
            f"│  {section}",
        }:
            collecting = True
            continue
        if collecting and (
            stripped.startswith("├─")
            or stripped.startswith("└─")
            or stripped == "Video"
        ):
            break
        if not collecting:
            continue
        label = stripped.lstrip("│").strip()
        if not label or label.startswith("├") or label.startswith("└"):
            continue
        label = label.lstrip("* ").strip()
        if label[:1].isdigit():
            _, _, remainder = label.partition(".")
            label = remainder.strip() or label
        name, _, _volume = label.partition("[")
        name = name.strip()
        if name:
            names.append(name)
    return tuple(names)


def list_pipewire_capture_sources() -> tuple[str, ...]:
    """Return PipeWire node names usable as ``pw-record --target``.

    Includes Audio Sources and Filters so echo-cancelled capture nodes remain
    selectable. Empty means the caller should keep PipeWire's system default.
    """

    if _which("wpctl") is None:
        raise UserConfigurationError(
            "wpctl not found; install WirePlumber to list capture sources"
        )
    process = _run(["wpctl", "status", "--name"], timeout=5)
    if process is None or process.returncode:
        raise UserConfigurationError(
            "wpctl status failed; PipeWire may not be running for this session"
        )
    names: list[str] = []
    seen: set[str] = set()
    for name in (
        *pipewire_section_devices(process.stdout, "Sources"),
        *pipewire_section_devices(process.stdout, "Filters"),
    ):
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    if not names:
        raise UserConfigurationError("PipeWire reported no capture sources")
    return tuple(names)
