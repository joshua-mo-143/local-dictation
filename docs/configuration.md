# Configuration

Settings live in `$XDG_CONFIG_HOME/local-dictation/config.toml` (by default
`~/.config/local-dictation/config.toml`). The file is optional: a missing file
or key falls back to the built-in default. The systemd unit only sets
operational values such as the runtime directory and cache paths, never engine,
device, or audio choices.

Values resolve in this order, from lowest to highest precedence:

1. built-in defaults
2. `config.toml`
3. `DICTATION_*` environment variables

## Settings

This file shows every key with its default:

```toml
[stt]
device = "auto"
backend = "parakeet"
model = "nemo-parakeet-tdt-0.6b-v2"
quantization = "int8"
compute = "float16"
language = "auto"

[dictation]
source = ""
inject = "auto"
prompt = "Technical software engineering dictation that may mention Cursor, Herdr, code, files, functions, terminals, and command-line tools."
replacements = "herder:herdr;cursa:Cursor;curser:Cursor"
vad_end_silence_ms = 900.0
vad_max_seconds = 120.0
vad_min_speech_rms = 1100.0
vad_start_speech_frames = 3
```

| Key | Environment override | Meaning |
| --- | --- | --- |
| `stt.device` | `DICTATION_DEVICE` | `auto`, `cpu`, or `cuda`; see [Compute devices](#compute-devices) |
| `stt.backend` | `DICTATION_BACKEND` | `parakeet` or `whisper` |
| `stt.model` | `DICTATION_MODEL` | Backend model name |
| `stt.quantization` | `DICTATION_QUANTIZATION` | Parakeet ONNX quantization; `none` disables it |
| `stt.compute` | `DICTATION_COMPUTE` | faster-whisper compute type |
| `stt.language` | `DICTATION_LANGUAGE` | `en`, `zh`, `english`, `chinese`, or `auto` |
| `dictation.source` | `DICTATION_SOURCE` | PipeWire capture node; empty uses the system default |
| `dictation.inject` | `DICTATION_INJECT` | `auto`, `paste`, `type`, or `stdout`; see [Inserting text](architecture.md#inserting-text) |
| `dictation.prompt` | `DICTATION_PROMPT` | Initial faster-whisper prompt |
| `dictation.replacements` | `DICTATION_REPLACEMENTS` | `spoken:written` pairs separated by `;`; empty disables the defaults |
| `dictation.vad_end_silence_ms` | `DICTATION_VAD_END_SILENCE_MS` | Silence that finishes a VAD utterance |
| `dictation.vad_max_seconds` | `DICTATION_VAD_MAX_SECONDS` | Longest VAD utterance |
| `dictation.vad_min_speech_rms` | `DICTATION_VAD_MIN_SPEECH_RMS` | Loudness floor below which VAD ignores sound; `0` disables it |
| `dictation.vad_start_speech_frames` | `DICTATION_VAD_START_SPEECH_FRAMES` | Consecutive 80 ms speech frames that start a VAD utterance |

`dictation.replacements` may also be a TOML table, such as
`replacements = { herder = "herdr" }`. For corrections you add and remove over
time, the [vocabulary store](vocabulary.md) is usually more convenient, because
it applies without restarting the service.

Invalid values fail with an error that names the setting, rather than being
ignored: an unknown section or key, a choice outside the list, a non-positive
duration, a frame count that isn't a positive whole number, and so on. A
malformed, non-UTF-8, or unreadable file fails with an error that names its
path.

## Commands

`dictate config` validates every change before writing it. Writes are atomic,
owner-only (`0600`), and serialized with a lock file, so two commands can't lose
each other's changes. They never write environment overrides into the file.

```fish
dictate config show
dictate config show stt.device
dictate config show --json
dictate config set stt.device cpu
dictate config set dictation.inject paste
dictate config set dictation.source
dictate config reset --section dictation
dictate config reset
```

`show` prints the stored settings (the file over built-in defaults) without
environment overrides. Leave the value off `config set dictation.source` to pick
from the PipeWire capture sources that `wpctl status` reports, or set it to
`default` for the system default. A whole-file `reset` never reads the existing
file, so it also replaces one that no longer loads.

## When changes apply

The service reads its settings once at startup. `dictate` reads them again on
every run.

| Settings | Read by | Apply changes by |
| --- | --- | --- |
| Every `stt.*` setting, `dictation.prompt`, `dictation.replacements` | `local-dictation.service` | `dictate services restart` |
| `dictation.source`, `dictation.inject`, and the `dictation.vad_*` settings | Each `dictate` command | Running the next command; a running `dictate vad` keeps the values it started with |

When a change needs a restart and the service is running, `dictate config set`
and `reset` say so. Environment variables follow the same rule: changing one
has no effect on a process that's already running.

## Compute devices

`stt.device` picks where the model runs:

- `cpu` is a strict CPU path. Parakeet gets only ONNX Runtime's CPU provider,
  faster-whisper gets `device="cpu"`, half-precision Whisper compute types drop
  to `int8`, and startup never probes CUDA or searches for NVIDIA libraries.
- `cuda` requires the backend to report a usable CUDA device. If it can't, the
  service fails at startup with an actionable error instead of quietly using the
  CPU.
- `auto` uses CUDA when the installed runtime exposes it and otherwise falls back
  to the CPU.

The service environment needs an engine extra that supports the device and
backend: `cpu` for Parakeet on the CPU, `cuda` for Parakeet on an NVIDIA GPU,
and `whisper` for faster-whisper. Switching `stt.backend` to `whisper` also
needs a Whisper model in `stt.model`, such as `large-v3-turbo`. See
[Engine extras](installation.md#engine-extras).
