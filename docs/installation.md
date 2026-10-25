# Installation

`scripts/install.sh` is the supported way to install on Arch, Ubuntu/Debian, and
Fedora. This page covers what it does, how to choose the compute device and
speech engine, how to update, and how to install by hand on other systems.

## Requirements

- Linux on x86-64 with a systemd user session and PipeWire.
- `python3` 3.11 or newer for the installer's planning step. If it's missing or
  too old, the installer installs the distro's `python3` package first.
- For CUDA: an NVIDIA GPU with a working driver, plus the CUDA and cuDNN
  libraries. See [CUDA](#cuda).
- Network access the first time the service starts, to download the model from
  Hugging Face. The Parakeet model takes less than 1 GB in
  `~/.cache/huggingface`. The first `dictate vad` also downloads a 2 MB voice
  activity model.
- Memory for the model: systemd throttles the service above 4 GB and stops it
  above 6 GB.

## Run the installer

```fish
git clone https://github.com/joshua-mo-143/local-dictation.git ~/local-dictation
cd ~/local-dictation
./scripts/install.sh
```

The installer:

1. Resolves the plan for this machine: the device, the engine extra, the distro
   packages, and the install paths.
2. Links `~/local-dictation` to the checkout if you cloned it somewhere else,
   because the shipped unit runs the server from
   `~/local-dictation/.venv-service`. It won't replace a real directory at that
   path.
3. Installs the [system packages](#system-packages) with `paru`, `apt-get`, or
   `dnf`.
4. Creates the service environment, `.venv-service`, with uv and the chosen
   engine extra.
5. Links `~/.local/bin/dictate` to the `dictate` command in that environment.
   It won't replace a different program at that path.
6. Saves the device with `dictate config set stt.device`.
7. Installs, audits, and starts `local-dictation.service`.

The first start downloads the model. Watch `dictate services logs --follow`
until it says `model ready`, then pick your microphone and check the setup:

```fish
dictate config set dictation.source
dictate doctor
```

### Options

| Option | Effect |
| --- | --- |
| `--device auto` | The default. Installs the `cuda` extra when `nvidia-smi` finds a working driver within 10 seconds, and the `cpu` extra otherwise |
| `--device cpu` | Installs the `cpu` extra without probing the GPU |
| `--device cuda` | Installs the `cuda` extra without probing the GPU |
| `DICTATION_DEVICE` | Same as `--device`; the flag wins when you give both |
| `PYTHON_VERSION` | Python for the service environment; defaults to 3.11, and 3.12 also works |
| `SKIP_SYSTEM_PACKAGES=1` | Don't install distro packages; you provide them |

For example:

```fish
./scripts/install.sh --device cpu
env PYTHON_VERSION=3.12 SKIP_SYSTEM_PACKAGES=1 ./scripts/install.sh
```

### System packages

| Purpose | Arch | Ubuntu/Debian | Fedora |
| --- | --- | --- | --- |
| Audio capture | `pipewire`, `pipewire-audio`, `wireplumber` | `pipewire`, `pipewire-bin`, `wireplumber` | `pipewire`, `pipewire-utils`, `wireplumber` |
| Notifications | `libnotify` | `libnotify-bin` | `libnotify` |
| Inserting text | `xdotool`, `xclip`, `wl-clipboard`, `wtype` | the same | the same |
| Tools | `git`, `curl`, `uv` | `git`, `curl` | `git`, `curl` |
| CUDA, with the `cuda` extra only | `cuda` | none | none |

On Ubuntu/Debian and Fedora, the installer gets uv from
[its official installer](https://docs.astral.sh/uv/getting-started/installation/)
when it isn't already installed.

## CUDA

The `cuda` extra runs Parakeet on ONNX Runtime's CUDA provider, which loads the
CUDA and cuDNN libraries from the system. The installer covers only part of
that:

- On Arch it installs `cuda`, but not cuDNN. Install that yourself:

  ```fish
  paru -S --needed cudnn
  ```

- On Ubuntu/Debian and Fedora it can't install CUDA at all. If `nvidia-smi`
  works, the default `auto` plan stops with an error instead of guessing.
  Either install CUDA and cuDNN from NVIDIA and re-run with
  `SKIP_SYSTEM_PACKAGES=1`, or choose `--device cpu`.

If transcription is slow on a machine with a GPU, check
`dictate services logs` for CUDA provider errors.

## Engine extras

The service environment holds one Parakeet extra, and optionally faster-whisper
next to it:

| Extra | Engine | Configuration |
| --- | --- | --- |
| `cpu` | Parakeet on ONNX Runtime's CPU provider | `stt.device` is `cpu` or `auto` |
| `cuda` | Parakeet on ONNX Runtime's CUDA provider | `stt.device` is `cuda` or `auto` |
| `whisper` | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | `stt.backend` is `whisper` |

`cpu` and `cuda` conflict, so the installer picks one. To add faster-whisper,
sync it next to the Parakeet extra you installed, with the same Python version,
then choose a Whisper model and restart the service:

```fish
cd ~/local-dictation
env UV_PROJECT_ENVIRONMENT=.venv-service \
    uv sync --python 3.11 --no-dev --extra cpu --extra whisper
dictate config set stt.backend whisper
dictate config set stt.model large-v3-turbo
dictate services restart
```

Use `--extra cuda` instead of `--extra cpu` if you installed for CUDA. On an
NVIDIA GPU, faster-whisper needs CUDA's cuBLAS and cuDNN libraries too. To switch
back, set `stt.backend` to `parakeet`: Parakeet ignores Whisper model names and
uses its default model.

The installer syncs only the Parakeet extra, so re-running it removes
faster-whisper. Run the `uv sync` command above again afterwards.

## Updating

Pull the changes, then re-run the installer with the device you chose before.
It saves that device to `stt.device` again, so leaving `--device` off resets
the setting to `auto`.

```fish
cd ~/local-dictation
git pull
./scripts/install.sh --device cpu
dictate services restart
```

The restart matters: the installer starts the service, but a service that's
already running keeps its old code and settings until it restarts.

If an update changed the shipped unit, the installer stops at
`services install`, because the installed unit no longer matches. Compare the
two, then replace the installed unit and finish by hand:

```fish
diff ~/.config/systemd/user/local-dictation.service \
    ~/local-dictation/systemd/user/local-dictation.service
dictate services install --force
dictate services audit
dictate services restart
```

## Install by hand

On other distributions, install the equivalents of the
[system packages](#system-packages) and uv yourself. The shipped unit expects
the checkout at `~/local-dictation`; a symlink there works too.

```fish
git clone https://github.com/joshua-mo-143/local-dictation.git ~/local-dictation
cd ~/local-dictation
env UV_PROJECT_ENVIRONMENT=.venv-service \
    uv sync --python 3.11 --no-dev --extra cpu
mkdir -p ~/.local/bin
ln -sfn ~/local-dictation/.venv-service/bin/dictate ~/.local/bin/dictate
dictate config set stt.device cpu
dictate services install
dictate services audit
dictate services start
```

Use `--extra cuda` and `stt.device cuda` for CUDA. Make sure `~/.local/bin` is on
your `PATH`.

The service gets its own environment because `uv sync` keeps `.venv` in step
with the development setup, which would remove the engine extra.
[Development](development.md) covers `.venv` and the isolated development
launcher.

## Moving from local-voice-harness

local-dictation has its own unit, socket, runtime directory, and settings, so
it doesn't share state with the dictation service from
[local-voice-harness](https://github.com/joshua-mo-143/local-voice-harness).
To switch:

1. Stop the old service, so the two don't both hold a model in memory:

   ```fish
   systemctl --user disable --now dictation.service
   ```

2. Install local-dictation as above.
3. Set your preferences again with `dictate config set`. local-dictation
   doesn't read local-voice-harness configuration or `backend.env`; see
   [Configuration](configuration.md) for the keys.
4. Import your vocabulary, as described in
   [Vocabulary](vocabulary.md#importing-from-local-voice-harness).
5. Point your key bindings at `~/.local/bin/dictate`, for example
   `dictate toggle` instead of `voice-harness-dictate toggle`.
