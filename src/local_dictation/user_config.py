"""Validated user configuration.

Settings live in ``$XDG_CONFIG_HOME/local-dictation/config.toml`` in two
tables: ``[stt]`` selects the speech-to-text engine, and ``[dictation]`` covers
capture, injection, transcription prompting, and VAD tuning.

Precedence, from lowest to highest, is::

    built-in defaults < config.toml < DICTATION_* environment variables
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from . import config
from .config import read_toml_table

_TOP_LEVEL_SECTIONS = ("stt", "dictation")
_STT_KEYS = (
    "device",
    "backend",
    "model",
    "quantization",
    "compute",
    "language",
)
_DICTATION_KEYS = (
    "source",
    "inject",
    "prompt",
    "replacements",
    "vad_end_silence_ms",
    "vad_max_seconds",
    "vad_min_speech_rms",
    "vad_start_speech_frames",
)

_STT_BACKENDS = {"parakeet", "whisper"}
_STT_DEVICES = {"auto", "cpu", "cuda"}
_STT_LANGUAGES = {"en", "zh", "english", "chinese", "auto"}
_DICTATION_INJECT_MODES = {"auto", "paste", "type", "stdout"}


class UserConfigurationError(ValueError):
    """A configuration value is missing, malformed, or out of range."""


class ComputeDevice(StrEnum):
    """Where the speech-to-text engine runs."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True)
class SttSettings:
    """Speech-to-text engine selectors."""

    device: ComputeDevice = ComputeDevice.AUTO
    backend: str = "parakeet"
    model: str = "nemo-parakeet-tdt-0.6b-v2"
    quantization: str = "int8"
    compute: str = "float16"
    language: str = "auto"


def resolve_local_compute(
    requested: ComputeDevice,
    *,
    cuda_available: bool,
    label: str,
) -> Literal["cpu", "cuda"]:
    """Resolve a typed local compute selector without probing on the CPU path.

    Callers must not query CUDA when ``requested`` is ``cpu``. ``auto`` and
    ``cuda`` receive a precomputed availability flag so tests can stay
    hardware-free.
    """

    if requested is ComputeDevice.CPU:
        return "cpu"
    if requested is ComputeDevice.CUDA and not cuda_available:
        raise RuntimeError(
            f"CUDA {label} was requested, but CUDA is unavailable; install the "
            "cuda extra and verify the NVIDIA driver"
        )
    if requested is ComputeDevice.CUDA:
        return "cuda"
    return "cuda" if cuda_available else "cpu"


DEFAULT_DICTATION_PROMPT = (
    "Technical software engineering dictation that may mention Cursor, Herdr, "
    "code, files, functions, terminals, and command-line tools."
)
DEFAULT_DICTATION_REPLACEMENTS = (
    ("herder", "herdr"),
    ("cursa", "Cursor"),
    ("curser", "Cursor"),
)


@dataclass(frozen=True)
class DictationSettings:
    """Focused-window injection, transcription prompting, and VAD tuning."""

    source: str = config.DEFAULT_SOURCE
    inject: str = "auto"
    prompt: str = DEFAULT_DICTATION_PROMPT
    replacements: tuple[tuple[str, str], ...] = DEFAULT_DICTATION_REPLACEMENTS
    vad_end_silence_ms: float = 900.0
    vad_max_seconds: float = 120.0
    vad_min_speech_rms: float = 1100.0
    vad_start_speech_frames: int = 3


@dataclass(frozen=True)
class UserConfig:
    """The complete, validated user configuration."""

    stt: SttSettings
    dictation: DictationSettings


def user_config_path(
    environment: Mapping[str, str] = os.environ,
    *,
    home: Path | None = None,
) -> Path:
    """Return the path to ``config.toml``."""

    return config.config_dir(environment, home=home) / "config.toml"


def _section(raw: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise UserConfigurationError(f"[{name}] must be a TOML table")
    return value


def _reject_unknown(
    section: Mapping[str, object], allowed: Iterable[str], *, label: str
) -> None:
    permitted = set(allowed)
    unknown = sorted(key for key in section if key not in permitted)
    if unknown:
        raise UserConfigurationError(f"unknown {label} key(s): {', '.join(unknown)}")


def _resolve(
    environment: Mapping[str, str],
    env_key: str,
    section: Mapping[str, object],
    key: str,
    default: object,
) -> object:
    if env_key in environment:
        return environment[env_key]
    if key in section:
        return section[key]
    return default


def _as_positive_int(value: object, *, label: str) -> int:
    number = _as_int(value, label=label)
    if number < 1:
        raise UserConfigurationError(f"{label} must be a positive integer")
    return number


def _as_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise UserConfigurationError(f"{label} must be an integer")
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise UserConfigurationError(f"{label} must be an integer") from exc


def _as_nonnegative_float(value: object, *, label: str) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError) as exc:
        raise UserConfigurationError(
            f"{label} must be a finite non-negative number"
        ) from exc
    if not math.isfinite(number) or number < 0:
        raise UserConfigurationError(f"{label} must be a finite non-negative number")
    return number


def _as_positive_float(value: object, *, label: str) -> float:
    number = _as_nonnegative_float(value, label=label)
    if number <= 0:
        raise UserConfigurationError(f"{label} must be a finite positive number")
    return number


def _as_nonempty(value: object, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise UserConfigurationError(f"{label} must not be empty")
    return text


def _as_choice(value: object, choices: set[str], *, label: str) -> str:
    text = str(value).strip().casefold()
    if text not in choices:
        options = ", ".join(sorted(choices))
        raise UserConfigurationError(f"{label} must be one of: {options}")
    return text


def _as_replacements(value: object, *, label: str) -> tuple[tuple[str, str], ...]:
    if isinstance(value, Mapping):
        entries = ((str(source), str(target)) for source, target in value.items())
    else:
        entries = (
            tuple(entry.split(":", 1))
            for entry in str(value).split(";")
            if entry.strip() and ":" in entry
        )
    replacements = tuple(
        (source.strip(), target.strip()) for source, target in entries if source.strip()
    )
    return replacements


def _load_stt(
    section: Mapping[str, object], environment: Mapping[str, str]
) -> SttSettings:
    _reject_unknown(section, _STT_KEYS, label="[stt]")
    defaults = SttSettings()
    return SttSettings(
        device=ComputeDevice(
            _as_choice(
                _resolve(
                    environment, "DICTATION_DEVICE", section, "device", defaults.device
                ),
                _STT_DEVICES,
                label="stt.device",
            )
        ),
        backend=_as_choice(
            _resolve(
                environment, "DICTATION_BACKEND", section, "backend", defaults.backend
            ),
            _STT_BACKENDS,
            label="stt.backend",
        ),
        model=_as_nonempty(
            _resolve(environment, "DICTATION_MODEL", section, "model", defaults.model),
            label="stt.model",
        ),
        quantization=_as_nonempty(
            _resolve(
                environment,
                "DICTATION_QUANTIZATION",
                section,
                "quantization",
                defaults.quantization,
            ),
            label="stt.quantization",
        ),
        compute=_as_nonempty(
            _resolve(
                environment, "DICTATION_COMPUTE", section, "compute", defaults.compute
            ),
            label="stt.compute",
        ),
        language=_as_choice(
            _resolve(
                environment,
                "DICTATION_LANGUAGE",
                section,
                "language",
                defaults.language,
            ),
            _STT_LANGUAGES,
            label="stt.language",
        ),
    )


def _load_dictation(
    section: Mapping[str, object], environment: Mapping[str, str]
) -> DictationSettings:
    _reject_unknown(section, _DICTATION_KEYS, label="[dictation]")
    return DictationSettings(
        source=str(
            _resolve(
                environment,
                "DICTATION_SOURCE",
                section,
                "source",
                config.DEFAULT_SOURCE,
            ),
        ).strip(),
        inject=_as_choice(
            _resolve(environment, "DICTATION_INJECT", section, "inject", "auto"),
            _DICTATION_INJECT_MODES,
            label="dictation.inject",
        ),
        prompt=str(
            _resolve(
                environment,
                "DICTATION_PROMPT",
                section,
                "prompt",
                DEFAULT_DICTATION_PROMPT,
            ),
        ).strip(),
        replacements=_as_replacements(
            _resolve(
                environment,
                "DICTATION_REPLACEMENTS",
                section,
                "replacements",
                ";".join(
                    f"{source}:{target}"
                    for source, target in DEFAULT_DICTATION_REPLACEMENTS
                ),
            ),
            label="dictation.replacements",
        ),
        vad_end_silence_ms=_as_positive_float(
            _resolve(
                environment,
                "DICTATION_VAD_END_SILENCE_MS",
                section,
                "vad_end_silence_ms",
                900,
            ),
            label="dictation.vad_end_silence_ms",
        ),
        vad_max_seconds=_as_positive_float(
            _resolve(
                environment,
                "DICTATION_VAD_MAX_SECONDS",
                section,
                "vad_max_seconds",
                120,
            ),
            label="dictation.vad_max_seconds",
        ),
        vad_min_speech_rms=_as_nonnegative_float(
            _resolve(
                environment,
                "DICTATION_VAD_MIN_SPEECH_RMS",
                section,
                "vad_min_speech_rms",
                1100,
            ),
            label="dictation.vad_min_speech_rms",
        ),
        vad_start_speech_frames=_as_positive_int(
            _resolve(
                environment,
                "DICTATION_VAD_START_SPEECH_FRAMES",
                section,
                "vad_start_speech_frames",
                3,
            ),
            label="dictation.vad_start_speech_frames",
        ),
    )


def load_user_config(
    environment: Mapping[str, str] = os.environ,
    *,
    path: Path | None = None,
    home: Path | None = None,
) -> UserConfig:
    """Load and validate ``config.toml`` with environment overrides applied."""

    config_path = path or user_config_path(environment, home=home)
    raw = read_toml_table(
        config_path, error=UserConfigurationError, label="user configuration"
    )
    _reject_unknown(raw, _TOP_LEVEL_SECTIONS, label="configuration section")
    return UserConfig(
        stt=_load_stt(_section(raw, "stt"), environment),
        dictation=_load_dictation(_section(raw, "dictation"), environment),
    )


def default_user_config(home: Path | None = None) -> UserConfig:
    """Return the built-in configuration with no files or environment applied."""

    return load_user_config({}, path=Path(os.devnull), home=home)


_TOML_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _toml_escape(character: str) -> str:
    if character in _TOML_ESCAPES:
        return _TOML_ESCAPES[character]
    if ord(character) < 0x20 or ord(character) == 0x7F:
        return f"\\u{ord(character):04x}"
    return character


def _toml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return '"' + "".join(map(_toml_escape, str(value))) + '"'


def _render_table(name: str, entries: Mapping[str, object]) -> list[str]:
    lines = [f"[{name}]"]
    for key, value in entries.items():
        lines.append(f"{key} = {_toml_scalar(value)}")
    lines.append("")
    return lines


def render_user_config(user_config: UserConfig) -> str:
    """Serialize a :class:`UserConfig` to deterministic TOML text."""

    stt = user_config.stt
    dictation = user_config.dictation
    lines: list[str] = []
    lines += _render_table(
        "stt",
        {
            "device": stt.device,
            "backend": stt.backend,
            "model": stt.model,
            "quantization": stt.quantization,
            "compute": stt.compute,
            "language": stt.language,
        },
    )
    lines += _render_table(
        "dictation",
        {
            "source": dictation.source,
            "inject": dictation.inject,
            "prompt": dictation.prompt,
            "replacements": ";".join(
                f"{source}:{target}" for source, target in dictation.replacements
            ),
            "vad_end_silence_ms": dictation.vad_end_silence_ms,
            "vad_max_seconds": dictation.vad_max_seconds,
            "vad_min_speech_rms": dictation.vad_min_speech_rms,
            "vad_start_speech_frames": dictation.vad_start_speech_frames,
        },
    )
    return "\n".join(lines).rstrip("\n") + "\n"


def write_user_config(user_config: UserConfig, path: Path) -> None:
    """Atomically write ``user_config`` to ``path`` with owner-only permissions."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = render_user_config(user_config)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _fsync_directory(directory: Path) -> None:
    """Flush a directory entry so a rename survives a crash (best effort)."""

    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
