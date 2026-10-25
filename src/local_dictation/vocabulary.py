"""Local personal vocabulary of speech-recognition corrections.

This module keeps a user-owned, on-disk store of replacements applied to every
transcript. It is intentionally local: the store is never uploaded anywhere.

Storage
-------
The store is a single JSON document (schema ``version`` 3) at
``$XDG_CONFIG_HOME/local-dictation/vocabulary.json`` (default
``~/.config/local-dictation/vocabulary.json``). JSON round-trips losslessly with
only the standard library. The file is written privately (``0o600``) with
sorted keys so backups and diffs are stable. Documents written by
local-voice-harness may also hold ``aliases`` and ``pronunciations``; those
entries are ignored, so its exports import cleanly.

Replacements
------------
``spoken`` (the recognized text) is replaced with ``written`` wherever it
appears as a whole, whitespace- and case-insensitive phrase. Matching never
fires inside a longer word. In the STT correction layer the precedence is:

1. user vocabulary replacements (highest),
2. the process-start ``dictation.replacements`` snapshot,
3. built-in defaults.

Adding a replacement whose phrase already maps to a different written form is
rejected as a conflict (pass ``force`` to overwrite), so the store never
silently learns or overrides a correction. A stored file that contains the same
phrase twice with different written forms is ambiguous and rejected on load.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import config
from .errors import DictationError

SCHEMA_VERSION = 3
_SUPPORTED_SCHEMA_VERSIONS = {1, 2, SCHEMA_VERSION}


class VocabularyError(DictationError):
    """A vocabulary store could not be read, validated, or updated."""


def _normalize_phrase(value: str) -> str:
    """Collapse whitespace and case-fold a spoken phrase for comparison."""

    collapsed = " ".join(value.split())
    if not collapsed:
        raise VocabularyError("phrase must not be empty")
    return collapsed.casefold()


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    body = r"\s+".join(re.escape(word) for word in phrase.split())
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


@dataclass(frozen=True)
class Replacement:
    spoken: str
    written: str

    def apply(self, text: str) -> str:
        return _phrase_pattern(self.spoken).sub(
            lambda _match, replacement=self.written: replacement, text
        )


@dataclass(frozen=True)
class Vocabulary:
    version: int = SCHEMA_VERSION
    replacements: tuple[Replacement, ...] = ()

    def replacement_for(self, spoken: str) -> Replacement | None:
        folded = _normalize_phrase(spoken)
        for replacement in self.replacements:
            if replacement.spoken.casefold() == folded:
                return replacement
        return None

    def to_document(self) -> dict[str, object]:
        return {
            "version": self.version,
            "replacements": [
                {"spoken": item.spoken, "written": item.written}
                for item in sorted(self.replacements, key=lambda item: item.spoken)
            ],
        }


def _require_mapping(value: object, message: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise VocabularyError(message)
    return value


def _parse_replacements(value: object) -> tuple[Replacement, ...]:
    if not isinstance(value, list):
        raise VocabularyError("'replacements' must be a list")
    replacements: list[Replacement] = []
    seen: dict[str, str] = {}
    for entry in value:
        record = _require_mapping(entry, "each replacement must be an object")
        spoken_raw = record.get("spoken")
        written_raw = record.get("written")
        if not isinstance(spoken_raw, str) or not isinstance(written_raw, str):
            raise VocabularyError("replacement 'spoken' and 'written' must be strings")
        spoken = " ".join(spoken_raw.split())
        written = written_raw.strip()
        if not spoken or not written:
            raise VocabularyError(
                "replacement 'spoken' and 'written' must not be empty"
            )
        folded = spoken.casefold()
        if folded in seen and seen[folded] != written:
            raise VocabularyError(
                f"replacement {spoken!r} is ambiguous: it maps to both "
                f"{seen[folded]!r} and {written!r}"
            )
        if folded in seen:
            continue
        seen[folded] = written
        replacements.append(Replacement(spoken=spoken, written=written))
    return tuple(replacements)


def _resolve_path(path: Path | None) -> Path:
    return path if path is not None else config.VOCABULARY_PATH


def load(path: Path | None = None) -> Vocabulary:
    """Load the vocabulary store, returning an empty store when absent."""

    location = _resolve_path(path)
    try:
        raw = location.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Vocabulary()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VocabularyError(f"vocabulary store {location} is not valid JSON") from exc
    document = _require_mapping(
        document, f"vocabulary store {location} must be an object"
    )
    version = document.get("version", SCHEMA_VERSION)
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise VocabularyError(
            f"unsupported vocabulary schema version {version!r}; "
            f"expected one of {sorted(_SUPPORTED_SCHEMA_VERSIONS)}"
        )
    return Vocabulary(
        version=SCHEMA_VERSION,
        replacements=_parse_replacements(document.get("replacements", [])),
    )


def save(vocabulary: Vocabulary, path: Path | None = None) -> Path:
    """Persist the vocabulary store atomically with private permissions."""

    location = _resolve_path(path)
    location.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = json.dumps(vocabulary.to_document(), indent=2, sort_keys=True) + "\n"
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=location.parent,
        prefix=".vocabulary-",
        suffix=".json",
        delete=False,
    )
    try:
        with handle:
            handle.write(payload)
        os.chmod(handle.name, 0o600)
        os.replace(handle.name, location)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return location


def add_replacement(
    spoken: str, written: str, *, force: bool = False, path: Path | None = None
) -> None:
    vocabulary = load(path)
    normalized = " ".join(spoken.split())
    target = written.strip()
    if not normalized or not target:
        raise VocabularyError("both a spoken phrase and a written form are required")
    existing = vocabulary.replacement_for(normalized)
    if existing is not None and existing.written != target and not force:
        raise VocabularyError(
            f"replacement {normalized!r} already maps to {existing.written!r}; "
            "pass --force to overwrite"
        )
    kept = tuple(
        item
        for item in vocabulary.replacements
        if item.spoken.casefold() != normalized.casefold()
    )
    updated = Vocabulary(
        version=SCHEMA_VERSION,
        replacements=(*kept, Replacement(spoken=normalized, written=target)),
    )
    save(updated, path)
    verb = "updated" if existing is not None else "added"
    print(f"{verb} replacement {normalized!r} -> {target!r}")


def remove_replacement(spoken: str, *, path: Path | None = None) -> None:
    vocabulary = load(path)
    normalized = " ".join(spoken.split())
    if vocabulary.replacement_for(normalized) is None:
        raise VocabularyError(f"no replacement is stored for {normalized!r}")
    kept = tuple(
        item
        for item in vocabulary.replacements
        if item.spoken.casefold() != normalized.casefold()
    )
    save(Vocabulary(version=SCHEMA_VERSION, replacements=kept), path)
    print(f"removed replacement {normalized!r}")


def list_entries(*, path: Path | None = None) -> None:
    vocabulary = load(path)
    print("replacements:")
    if not vocabulary.replacements:
        print("  (none)")
    for item in sorted(vocabulary.replacements, key=lambda item: item.spoken):
        print(f"  {item.spoken!r} -> {item.written!r}")


def export_entries(output: Path | None = None, *, path: Path | None = None) -> None:
    """Print or write a portable JSON backup of the store."""

    vocabulary = load(path)
    payload = json.dumps(vocabulary.to_document(), indent=2, sort_keys=True) + "\n"
    if output is None:
        print(payload, end="")
        return
    output.write_text(payload, encoding="utf-8")
    print(f"exported vocabulary to {output}")


def import_entries(
    source: Path, *, replace: bool = False, path: Path | None = None
) -> None:
    """Merge (default) or replace the store with a backup document."""

    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise VocabularyError(f"could not read {source}: {exc}") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VocabularyError(f"{source} is not valid JSON") from exc
    document = _require_mapping(document, f"{source} must be a JSON object")
    incoming = _parse_replacements(document.get("replacements", []))
    if replace:
        save(Vocabulary(version=SCHEMA_VERSION, replacements=incoming), path)
        print(f"replaced vocabulary with {len(incoming)} replacement(s) from {source}")
        return
    replacements = {item.spoken.casefold(): item for item in load(path).replacements}
    for item in incoming:
        replacements[item.spoken.casefold()] = item
    save(
        Vocabulary(version=SCHEMA_VERSION, replacements=tuple(replacements.values())),
        path,
    )
    print(f"imported {len(incoming)} replacement(s) from {source}")
