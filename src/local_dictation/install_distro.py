"""Installation plan for ``scripts/install.sh``.

Resolves the requested compute device to the ``cpu`` or ``cuda`` extra, maps
the system packages dictation needs onto the host's package manager, and
discovers the install paths. The installer runs this with the system Python
before the project environment exists, so it must import only the standard
library and stdlib-only package modules.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from .config import config_dir, xdg_config_home
from .service_units import INSTALL_ROOT
from .user_config import ComputeDevice

DICTATION_PACKAGES = (
    "pipewire",
    "pipewire-tools",
    "wireplumber",
    "libnotify",
    "xdotool",
    "xclip",
    "wl-clipboard",
    "wtype",
    "git",
    "curl",
    "uv",
)
CUDA_PACKAGES = ("cuda",)


class DistroFamily(StrEnum):
    ARCH = "arch"
    DEBIAN = "debian"
    FEDORA = "fedora"


class InstallPlanError(ValueError):
    """The requested install cannot be planned on this host."""


@dataclass(frozen=True)
class InstallPaths:
    """Discovered checkout, command, service, and configuration locations."""

    checkout: Path
    service_checkout: Path
    user_bin: Path
    dictate: Path
    systemd_user_dir: Path
    config_dir: Path


@dataclass(frozen=True)
class InstallPlan:
    """Extra, distro packages, and paths for one install."""

    device: str
    extra: str
    family: str
    distro_id: str
    package_manager: str
    install_command: tuple[str, ...]
    packages: tuple[str, ...]
    cuda_packages: tuple[str, ...]
    skipped_packages: tuple[str, ...]
    uv_bootstrap: bool
    paths: InstallPaths


_ARCH_PACKAGES: dict[str, str | None] = {
    **{name: name for name in (*DICTATION_PACKAGES, *CUDA_PACKAGES)},
    "pipewire-tools": "pipewire-audio",
}

_DEBIAN_PACKAGES: dict[str, str | None] = {
    "pipewire": "pipewire",
    "pipewire-tools": "pipewire-bin",
    "wireplumber": "wireplumber",
    "libnotify": "libnotify-bin",
    "xdotool": "xdotool",
    "xclip": "xclip",
    "wl-clipboard": "wl-clipboard",
    "wtype": "wtype",
    "git": "git",
    "curl": "curl",
    "uv": None,
    "cuda": None,
}

_FEDORA_PACKAGES: dict[str, str | None] = {
    "pipewire": "pipewire",
    "pipewire-tools": "pipewire-utils",
    "wireplumber": "wireplumber",
    "libnotify": "libnotify",
    "xdotool": "xdotool",
    "xclip": "xclip",
    "wl-clipboard": "wl-clipboard",
    "wtype": "wtype",
    "git": "git",
    "curl": "curl",
    "uv": None,
    "cuda": None,
}

_PACKAGE_MAPS = {
    DistroFamily.ARCH: _ARCH_PACKAGES,
    DistroFamily.DEBIAN: _DEBIAN_PACKAGES,
    DistroFamily.FEDORA: _FEDORA_PACKAGES,
}

_INSTALL_COMMANDS = {
    DistroFamily.ARCH: ("paru", "-S", "--needed", "--noconfirm"),
    DistroFamily.DEBIAN: (
        "sudo",
        "apt-get",
        "install",
        "-y",
        "--no-install-recommends",
    ),
    DistroFamily.FEDORA: ("sudo", "dnf", "install", "-y"),
}

_PACKAGE_MANAGERS = {
    DistroFamily.ARCH: "paru",
    DistroFamily.DEBIAN: "apt-get",
    DistroFamily.FEDORA: "dnf",
}


def resolve_device(value: str | None) -> ComputeDevice:
    selected = (value or ComputeDevice.AUTO).strip().casefold()
    try:
        return ComputeDevice(selected)
    except ValueError:
        raise InstallPlanError("device must be auto, cpu, or cuda") from None


def resolve_extra(device: ComputeDevice, *, cuda_available: bool) -> str:
    """Pick the extra; an explicit ``cuda`` request does not depend on the probe."""

    if device is ComputeDevice.CUDA or (
        device is ComputeDevice.AUTO and cuda_available
    ):
        return "cuda"
    return "cpu"


def parse_os_release(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        name, separator, value = line.partition("=")
        if not separator:
            continue
        values[name] = value.strip().strip('"')
    return values


def detect_distro_family(
    os_release: Mapping[str, str] | None = None,
) -> DistroFamily:
    values = dict(os_release or {})
    identity = " ".join(
        (
            values.get("ID", ""),
            values.get("ID_LIKE", ""),
            values.get("NAME", ""),
        )
    ).casefold()
    if any(
        token in identity for token in ("arch", "cachyos", "manjaro", "endeavouros")
    ):
        return DistroFamily.ARCH
    if any(
        token in identity
        for token in ("ubuntu", "debian", "pop", "linuxmint", "elementary")
    ):
        return DistroFamily.DEBIAN
    if any(token in identity for token in ("fedora", "rhel", "centos", "nobara")):
        return DistroFamily.FEDORA
    raise InstallPlanError(
        "unsupported distribution; installation supports Arch, Ubuntu/Debian, "
        "and Fedora"
    )


def discover_install_paths(
    *,
    checkout: Path,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> InstallPaths:
    env = environment if environment is not None else os.environ
    resolved_home = home or Path(env.get("HOME") or Path.home())
    user_bin = resolved_home / ".local" / "bin"
    return InstallPaths(
        checkout=checkout.resolve(),
        service_checkout=resolved_home / INSTALL_ROOT.removeprefix("%h/"),
        user_bin=user_bin,
        dictate=user_bin / "dictate",
        systemd_user_dir=xdg_config_home(env, home=resolved_home) / "systemd" / "user",
        config_dir=config_dir(env, home=resolved_home),
    )


def _map_packages(
    family: DistroFamily, names: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    mapping = _PACKAGE_MAPS[family]
    mapped: list[str] = []
    skipped: list[str] = []
    for name in names:
        package = mapping.get(name)
        if package is None:
            skipped.append(name)
        elif package not in mapped:
            mapped.append(package)
    return tuple(mapped), tuple(skipped)


def resolve_install_plan(
    *,
    device: str | None,
    cuda_available: bool,
    checkout: Path,
    os_release: Mapping[str, str] | None = None,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
    install_packages: bool = True,
) -> InstallPlan:
    """Return the extra, packages, and paths for one install choice.

    Packages without an adapter for this distro fail the plan unless the
    installer is skipping system packages, in which case the user provides them.
    """

    requested = resolve_device(device)
    extra = resolve_extra(requested, cuda_available=cuda_available)
    family = detect_distro_family(os_release)
    packages, skipped_system = _map_packages(family, DICTATION_PACKAGES)
    cuda_packages, skipped_cuda = _map_packages(
        family, CUDA_PACKAGES if extra == "cuda" else ()
    )
    skipped = (*skipped_system, *skipped_cuda)
    unmapped = tuple(name for name in skipped if name != "uv")
    if unmapped and install_packages:
        raise InstallPlanError(
            f"{family.value} has no supported package adapter for: "
            f"{', '.join(unmapped)}; install them yourself and rerun with "
            "SKIP_SYSTEM_PACKAGES=1, or choose --device cpu"
        )
    return InstallPlan(
        device=requested.value,
        extra=extra,
        family=family.value,
        distro_id=(os_release or {}).get("ID", family.value),
        package_manager=_PACKAGE_MANAGERS[family],
        install_command=_INSTALL_COMMANDS[family],
        packages=packages,
        cuda_packages=cuda_packages,
        skipped_packages=skipped,
        uv_bootstrap="uv" in skipped,
        paths=discover_install_paths(
            checkout=checkout, home=home, environment=environment
        ),
    )


def plan_to_env(plan: InstallPlan) -> str:
    """Render shell assignments for ``scripts/install.sh``."""

    values: Mapping[str, object] = {
        "INSTALL_DEVICE": plan.device,
        "INSTALL_EXTRA": plan.extra,
        "INSTALL_DISTRO_FAMILY": plan.family,
        "INSTALL_DISTRO_ID": plan.distro_id,
        "INSTALL_PACKAGE_MANAGER": plan.package_manager,
        "INSTALL_PACKAGE_COMMAND": " ".join(plan.install_command),
        "INSTALL_DISTRO_PACKAGES": " ".join(plan.packages),
        "INSTALL_DISTRO_CUDA_PACKAGES": " ".join(plan.cuda_packages),
        "INSTALL_SKIPPED_PACKAGES": " ".join(plan.skipped_packages),
        "INSTALL_UV_BOOTSTRAP": "1" if plan.uv_bootstrap else "0",
        "INSTALL_CHECKOUT": str(plan.paths.checkout),
        "INSTALL_SERVICE_CHECKOUT": str(plan.paths.service_checkout),
        "INSTALL_USER_BIN": str(plan.paths.user_bin),
        "INSTALL_DICTATE": str(plan.paths.dictate),
        "INSTALL_SYSTEMD_USER_DIR": str(plan.paths.systemd_user_dir),
        "INSTALL_CONFIG_DIR": str(plan.paths.config_dir),
    }
    return "".join(f"{name}={_shell_value(value)}\n" for name, value in values.items())


def _shell_value(value: object) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _read_os_release(path: Path) -> dict[str, str]:
    try:
        return parse_os_release(path.read_text())
    except OSError as exc:
        raise InstallPlanError(f"could not read {path}: {exc}") from exc


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve the extra, distro packages, and paths for an install"
    )
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--os-release", default="/etc/os-release")
    parser.add_argument("--device", default="", help="auto, cpu, or cuda")
    parser.add_argument(
        "--cuda-available",
        action="store_true",
        help="a bounded runtime probe confirmed usable CUDA on this host",
    )
    parser.add_argument(
        "--skip-system-packages",
        action="store_true",
        help="the user provides system packages; do not require package adapters",
    )
    parser.add_argument("--format", choices=("env", "json"), default="env")
    options = parser.parse_args(arguments)
    try:
        plan = resolve_install_plan(
            device=options.device or None,
            cuda_available=options.cuda_available,
            checkout=Path(options.checkout),
            os_release=_read_os_release(Path(options.os_release)),
            install_packages=not options.skip_system_packages,
        )
    except InstallPlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if options.format == "json":
        payload = asdict(plan)
        payload["paths"] = {key: str(value) for key, value in payload["paths"].items()}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(plan_to_env(plan), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
