"""Configuration management for the ``dictate`` CLI.

Shows, sets, and resets the ``[stt]`` and ``[dictation]`` settings stored in
``config.toml``. Every write is validated by rendering the candidate and
reloading it through :mod:`local_dictation.user_config`, and happens under a
file lock that serialises concurrent ``dictate config`` commands.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from .config import APP_NAME, SERVICE_NAME
from .pipewire import list_pipewire_capture_sources
from .platform_services import user_services
from .user_config import (
    ComputeDevice,
    UserConfig,
    UserConfigurationError,
    default_user_config,
    load_user_config,
    render_user_config,
    user_config_path,
    write_user_config,
)

_SECTIONS = ("stt", "dictation")
_SYSTEM_DEFAULT_LABEL = "PipeWire system default"
_SYSTEM_DEFAULT_ALIASES = frozenset({"default", "system"})
_EXPLICIT_SOURCE_HINT = "run `dictate config set dictation.source <node-name>`"


@dataclass(frozen=True)
class ConfigChangeResult:
    """A stored change; ``restart_services`` lists only units running now."""

    config: UserConfig
    changed_keys: tuple[str, ...]
    restart_services: tuple[str, ...]


def _parse_text(key: str, value: str) -> str:
    text = value.strip()
    if any(unicodedata.category(character) == "Cc" for character in text):
        raise UserConfigurationError(f"{key} must be a single line of text")
    return text


def _parse_source(key: str, value: str) -> str:
    source = _parse_text(key, value)
    return "" if source.casefold() in _SYSTEM_DEFAULT_ALIASES else source


def _parse_device(key: str, value: str) -> ComputeDevice:
    try:
        return ComputeDevice(value.strip().casefold())
    except ValueError as exc:
        options = ", ".join(device.value for device in ComputeDevice)
        raise UserConfigurationError(f"{key} must be one of: {options}") from exc


def _parse_number(key: str, value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise UserConfigurationError(f"{key} must be a number") from exc


def _parse_integer(key: str, value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise UserConfigurationError(f"{key} must be an integer") from exc


def _parse_replacements(key: str, value: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for entry in _parse_text(key, value).split(";"):
        if not entry.strip():
            continue
        spoken, separator, written = entry.partition(":")
        if not separator or not spoken.strip():
            raise UserConfigurationError(
                f"{key} must be spoken:written pairs separated by ';', "
                f"got {entry.strip()!r}"
            )
        pairs.append((spoken.strip(), written.strip()))
    return tuple(pairs)


@dataclass(frozen=True)
class _Field:
    parse: Callable[[str, str], object]
    services: tuple[str, ...] = ()


# The STT server reads these once at startup; `dictate` rereads the others on
# every run.
_SERVER = (SERVICE_NAME,)

_FIELDS: dict[str, _Field] = {
    "stt.device": _Field(_parse_device, _SERVER),
    "stt.backend": _Field(_parse_text, _SERVER),
    "stt.model": _Field(_parse_text, _SERVER),
    "stt.quantization": _Field(_parse_text, _SERVER),
    "stt.compute": _Field(_parse_text, _SERVER),
    "stt.language": _Field(_parse_text, _SERVER),
    "dictation.source": _Field(_parse_source),
    "dictation.inject": _Field(_parse_text),
    "dictation.prompt": _Field(_parse_text, _SERVER),
    "dictation.replacements": _Field(_parse_replacements, _SERVER),
    "dictation.vad_end_silence_ms": _Field(_parse_number),
    "dictation.vad_max_seconds": _Field(_parse_number),
    "dictation.vad_min_speech_rms": _Field(_parse_number),
    "dictation.vad_start_speech_frames": _Field(_parse_integer),
}


def _known_key(key: str) -> str:
    name = key.strip().casefold()
    if name not in _FIELDS:
        raise UserConfigurationError(
            f"unknown configuration key {key!r}; expected one of: {', '.join(_FIELDS)}"
        )
    return name


def _describe(config: UserConfig, key: str) -> object:
    section, attribute = key.split(".")
    return getattr(getattr(config, section), attribute)


def _apply(config: UserConfig, key: str, value: object) -> UserConfig:
    section, attribute = key.split(".")
    settings = replace(getattr(config, section), **{attribute: value})
    return replace(config, **{section: settings})


def _format_value(value: object) -> str:
    if isinstance(value, tuple):
        return ";".join(f"{spoken}:{written}" for spoken, written in value)
    return str(value)


def _load_stored(path: Path) -> UserConfig:
    # DICTATION_* overrides must never be written back into the file.
    return load_user_config({}, path=path)


def _validated(config: UserConfig) -> UserConfig:
    with tempfile.TemporaryDirectory(prefix=f"{APP_NAME}-") as directory:
        candidate = Path(directory) / "config.toml"
        candidate.write_text(render_user_config(config), encoding="utf-8")
        return _load_stored(candidate)


@contextmanager
def _config_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = os.open(path.with_name(f"{path.name}.lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield
    finally:
        os.close(lock)


def _running_services(keys: Iterable[str]) -> tuple[str, ...]:
    services = dict.fromkeys(
        service for key in keys for service in _FIELDS[key].services
    )
    if not services:
        return ()
    supervisor = user_services()
    if not supervisor.binary_available():
        return ()
    return tuple(name for name in services if supervisor.is_active(name) == "active")


def _store(
    path: Path, keys: tuple[str, ...], build: Callable[[], UserConfig]
) -> ConfigChangeResult:
    """Validate and write ``build()`` while holding the config lock.

    Anything ``build`` keeps from the stored file must be read inside it, not
    before, so a concurrent change to another key is not lost.
    """

    with _config_lock(path):
        config = _validated(build())
        write_user_config(config, path)
    return ConfigChangeResult(
        config=config,
        changed_keys=keys,
        restart_services=_running_services(keys),
    )


def show_config(
    *,
    key: str | None = None,
    json_output: bool = False,
    path: Path | None = None,
) -> str:
    """Return the stored settings as TOML, or one value, optionally as JSON.

    Values come from ``config.toml`` over built-in defaults; ``DICTATION_*``
    environment overrides are not applied.
    """

    names = tuple(_FIELDS) if key is None else (_known_key(key),)
    config = _load_stored(path or user_config_path())
    if json_output:
        return json.dumps({name: _describe(config, name) for name in names}, indent=2)
    if key is None:
        return render_user_config(config).rstrip("\n")
    return _format_value(_describe(config, names[0]))


def commit_config_setting(
    key: str,
    value: str | None,
    *,
    path: Path | None = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    list_sources: Callable[[], Sequence[str]] | None = None,
) -> ConfigChangeResult:
    """Validate and store one setting.

    Only ``dictation.source`` may omit ``value``; it then opens the PipeWire
    capture-source picker.
    """

    name = _known_key(key)
    config_path = path or user_config_path()
    parsed: object
    if value is not None:
        parsed = _FIELDS[name].parse(name, value)
    elif name == "dictation.source":
        parsed = pick_capture_source(
            current=_load_stored(config_path).dictation.source,
            input_fn=input_fn,
            print_fn=print_fn,
            list_sources=list_sources,
        )
    else:
        raise UserConfigurationError(f"{name} requires a value")
    return _store(
        config_path,
        (name,),
        lambda: _apply(_load_stored(config_path), name, parsed),
    )


def reset_config(
    *,
    section: str | None = None,
    path: Path | None = None,
) -> ConfigChangeResult:
    """Restore built-in defaults for one section, or for the whole file.

    A whole-file reset never reads the existing file, so it also replaces one
    that no longer loads.
    """

    config_path = path or user_config_path()
    defaults = default_user_config()
    if section is None:
        return _store(config_path, tuple(_FIELDS), lambda: defaults)
    name = section.strip().casefold()
    if name not in _SECTIONS:
        raise UserConfigurationError(f"section must be one of: {', '.join(_SECTIONS)}")
    return _store(
        config_path,
        tuple(key for key in _FIELDS if key.startswith(f"{name}.")),
        lambda: replace(_load_stored(config_path), **{name: getattr(defaults, name)}),
    )


def format_restart_notice(services: Sequence[str]) -> str:
    if not services:
        return "No restart needed."
    return (
        f"Run `dictate services restart` to apply this change to {', '.join(services)}."
    )


def pick_capture_source(
    *,
    current: str = "",
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    list_sources: Callable[[], Sequence[str]] | None = None,
) -> str:
    """Prompt for a PipeWire capture node; ``""`` means the system default."""

    try:
        sources = tuple((list_sources or list_pipewire_capture_sources)())
    except UserConfigurationError as exc:
        raise UserConfigurationError(
            f"{exc}; to name a node directly, {_EXPLICIT_SOURCE_HINT}"
        ) from exc
    labels = (_SYSTEM_DEFAULT_LABEL, *sources)
    default_index = 0
    if current in sources:
        default_index = sources.index(current) + 1
    elif current:
        print_fn(f"Current source {current} is not among the listed devices.")
    print_fn("Capture sources:")
    for index, label in enumerate(labels):
        marker = "*" if index == default_index else " "
        print_fn(f" {marker} {index}. {label}")
    last = len(labels) - 1
    while True:
        try:
            answer = input_fn(f"Capture source [{default_index}]: ").strip()
        except EOFError as exc:
            raise UserConfigurationError(
                "choosing a capture source needs an interactive terminal; "
                f"instead, {_EXPLICIT_SOURCE_HINT}"
            ) from exc
        if not answer:
            index = default_index
        elif answer.isdecimal() and int(answer) <= last:
            index = int(answer)
        elif answer.casefold() in _SYSTEM_DEFAULT_ALIASES:
            index = 0
        elif answer in sources:
            index = sources.index(answer) + 1
        else:
            print_fn(f"Choose 0-{last} or a listed source name.")
            continue
        return sources[index - 1] if index else ""
