from __future__ import annotations

import os
import pwd
import site
import sys
from collections.abc import Mapping
from pathlib import Path

from .. import config
from ..user_config import ComputeDevice, load_user_config

# Recent ONNX Runtime releases keep a device ID and upload usage events from
# Linux unless this is "1".
TELEMETRY_OPT_OUTS: Mapping[str, str] = {"ORT_DISABLE_TELEMETRY": "1"}

PROTECTED_ENVIRONMENT_KEYS = frozenset(
    {
        "CUDA_CACHE_PATH",
        "HF_HOME",
        "TMPDIR",
        "HOME",
        "XDG_CACHE_HOME",
        "XDG_RUNTIME_DIR",
        *TELEMETRY_OPT_OUTS,
    }
)


def protected_environment(
    environment: Mapping[str, str] = os.environ,
    *,
    uid: int | None = None,
) -> dict[str, str]:
    """Return service-owned settings that user configuration cannot replace."""

    resolved_uid = os.getuid() if uid is None else uid
    home = Path(pwd.getpwuid(resolved_uid).pw_dir)
    runtime = Path(f"/run/user/{resolved_uid}")
    state = config.runtime_dir({**environment, "XDG_RUNTIME_DIR": str(runtime)})
    return {
        "CUDA_CACHE_PATH": str(state / "cuda-cache"),
        "HF_HOME": str(home / ".cache/huggingface"),
        "TMPDIR": str(state),
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_RUNTIME_DIR": str(runtime),
        **TELEMETRY_OPT_OUTS,
    }


def add_nvidia_library_paths(environment: dict[str, str]) -> None:
    """Expose wheel-provided CUDA libraries without changing CPU-only launches."""

    libraries = [
        str(path)
        for packages in map(Path, site.getsitepackages())
        for path in (packages / "nvidia").glob("*/lib")
        if path.is_dir()
    ]
    existing = environment.get("LD_LIBRARY_PATH", "")
    if libraries:
        environment["LD_LIBRARY_PATH"] = ":".join(
            [*libraries, *([existing] if existing else [])]
        )


def main() -> None:
    if "--check" in sys.argv[1:]:
        print("local-dictation-server: ok")
        return
    environment = os.environ.copy()
    environment.update(protected_environment(environment))
    device = load_user_config(environment).stt.device
    if device is not ComputeDevice.CPU:
        add_nvidia_library_paths(environment)
    os.execve(
        sys.executable,
        [sys.executable, "-m", "local_dictation.stt.server"],
        environment,
    )


if __name__ == "__main__":
    main()
