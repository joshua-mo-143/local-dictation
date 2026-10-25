from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

_AMBIENT_PREFIXES = ("DICTATION_",)
_AMBIENT_KEYS = ("STATE_DIRECTORY",)


@pytest.fixture(autouse=True)
def isolate_user_environment(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Keep default tests off the invoking user's HOME, XDG, and DICTATION_* env."""

    root = tmp_path_factory.mktemp("isolated-home")
    home = root / "home"
    config_home = root / "config"
    state_home = root / "state"
    cache_home = root / "cache"
    runtime_dir = root / "runtime"
    for path in (home, config_home, state_home, cache_home, runtime_dir):
        path.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    for key in list(os.environ):
        if key.startswith(_AMBIENT_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    for key in _AMBIENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield
