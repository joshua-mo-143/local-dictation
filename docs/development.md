# Development

## Run the current checkout

`scripts/dev.sh` runs the STT server and the `dictate` client from the checkout
without touching the installed service. Start the server in one terminal:

```fish
scripts/dev.sh server
```

Wait for `model ready` in its log, then run client commands from another
terminal:

```fish
scripts/dev.sh dictate config set dictation.inject stdout
scripts/dev.sh dictate begin
scripts/dev.sh dictate end
scripts/dev.sh dictate vocabulary list
scripts/dev.sh dictate doctor
```

The launcher keeps the development copy apart from the installed one:

| What | Development | Installed |
| --- | --- | --- |
| Runtime directory | `$XDG_RUNTIME_DIR/local-dictation-dev/` (through `DICTATION_RUNTIME_DIR`) | `$XDG_RUNTIME_DIR/local-dictation/` |
| Socket | `$XDG_RUNTIME_DIR/local-dictation-dev.sock` | `$XDG_RUNTIME_DIR/local-dictation.sock` |
| Configuration and vocabulary | `.dev/config/local-dictation/` | `~/.config/local-dictation/` |
| Virtual environment | `.dev/venv` | `.venv-service` |

Both share the Hugging Face model cache in `~/.cache/huggingface`, so the
development server doesn't download the model again. The development server
installs the `cpu` extra by default; set `DICTATION_DEV_EXTRA=cuda` to test on
the GPU, and `PYTHON_VERSION` to pick another interpreter.

The launcher never starts, stops, installs, or edits a systemd unit, and it
refuses `dictate services`. Both copies can run side by side, but they share the
microphone, so record with one at a time. Setting
`dictation.inject` to `stdout` in the development configuration prints
transcripts instead of typing them into whatever window has focus.

## Smoke tests that use the microphone

Smoke tests that record real audio need someone to speak. Before starting one,
say which commands will run and wait for the person at the microphone to
confirm they're ready. Don't infer readiness from an earlier message. Use
`stdout` injection unless the test is about inserting text.

## Quality checks

The development environment holds the package and quality tools only. It doesn't
install an engine extra, download models, or start services. Run every check for
each Python version in CI (3.11 and 3.12):

```fish
set python_version 3.12
set -gx UV_PROJECT_ENVIRONMENT ".dev/quality-$python_version"
uv sync --python $python_version
uv run ruff format --check .
uv run ruff check .
uv run pyright --pythonversion $python_version
uv run python -m local_dictation.service_units --require-systemd-analyze
uv run pytest
uv run coverage json -o coverage.json
uv run python -m local_dictation.coverage_gate coverage.json
```

Use `uv run ruff format .` to apply formatting. Pytest measures branch coverage
and fails below 75% overall. `coverage_gate` adds per-module floors from
`[tool.local-dictation.coverage.module-minimums]` in `pyproject.toml`.

The service-unit check makes sure the checkout and the wheel ship the same unit,
that the unit agrees with the package (the server entry point, the engine
extras, the runtime directory, and the environment the launcher sets), and that
its hardening directives are intact. When `systemd-analyze` is installed, it
verifies the unit in an isolated root with stubs for its executables, so it
doesn't need a model environment. It checks the shipped template only; run
`dictate services audit` to check an installed unit and its drop-ins.

The production-extras check builds the wheel, installs each engine extra into a
clean environment, and confirms that the entry points work. It imports each
engine with the launcher's telemetry opt-outs and a scratch home and temporary
directory, and fails if the import writes anything there, which catches a
dependency upgrade that starts keeping telemetry state. It then runs
`python -m local_dictation.vad --check` the same way, but without the opt-out,
so the check also proves that the VAD sets it before loading ONNX Runtime. That
check downloads the 2 MB Silero model and confirms that loud white noise doesn't
register as speech. CI checks every extra in a separate job. The cuda and
whisper extras are large downloads, so locally you can check one extra at a
time:

```fish
env PYTHONPATH=src uv run --no-project python -m local_dictation.production_extras \
    --python 3.12 --extra cpu
```

GPU, microphone, and desktop checks are manual. See the
[hardware smoke checklist](hardware-smoke.md).
