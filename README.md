# local-dictation

Local push-to-talk and voice-activated dictation for Linux desktops. Press a
key, say something, and the text lands in whatever window has focus. Speech
recognition runs on your own machine with
[Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
through ONNX Runtime, or with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), on the CPU or an
NVIDIA GPU. Audio and transcripts never leave the machine.

This started life as the dictation half of
[local-voice-harness](https://github.com/joshua-mo-143/local-voice-harness) and
now stands on its own.

## How it works

A small systemd user service, `local-dictation.service`, keeps the speech model
loaded and listens on a private Unix socket. The `dictate` command records from
PipeWire, hands the finished recording to the service, and inserts the
transcript into the focused window by pasting or typing it. In a nutshell, the
service does the heavy lifting once, so each key press stays quick.

Recordings that fail to transcribe aren't thrown away: they stay in the runtime
directory, and the error tells you how to retry them. The
[architecture notes](docs/architecture.md) cover the recording, hand-off, and
recovery details.

## Requirements

- Linux with a systemd user session and PipeWire (`pw-record` and `wpctl`)
- X11 with `xdotool` and `xclip`, or Hyprland or Sway with `wtype` and
  `wl-clipboard`
- `notify-send` for desktop notifications
- [uv](https://docs.astral.sh/uv/) and Python 3.11 or 3.12
- Optionally, an NVIDIA GPU with a working driver, CUDA, and cuDNN for GPU
  inference

GNOME and KDE Wayland sessions don't let other programs see the focused window,
so there `dictate` can only print transcripts (`inject = "stdout"`).

## Install

```fish
git clone https://github.com/joshua-mo-143/local-dictation.git ~/local-dictation
cd ~/local-dictation
./scripts/install.sh
```

The installer handles Arch, Ubuntu/Debian, and Fedora. It installs the system
packages, creates the service environment, links `dictate` into `~/.local/bin`,
and installs and starts the user service. It picks CUDA when `nvidia-smi` finds
a working driver and the CPU otherwise; pass `--device cpu` or `--device cuda`
to choose yourself. The first start downloads the model, so give it a minute.
See [Installation](docs/installation.md) for the details and a manual route.

Then pick your microphone and check everything:

```fish
dictate config set dictation.source
dictate doctor
```

## Usage

Bind the recording commands to keys in your window manager. For example, in
i3 or Sway:

```text
bindsym $mod+F9 exec ~/.local/bin/dictate toggle
bindsym $mod+F10 exec ~/.local/bin/dictate vad
```

| Command | What it does |
| --- | --- |
| `dictate begin` | Start recording |
| `dictate end` | Stop recording, then transcribe and insert the text |
| `dictate toggle` | `end` when recording, otherwise `begin` |
| `dictate cancel` | Stop recording and discard the audio |
| `dictate vad` | Start or stop hands-free dictation, which transcribes each utterance after a pause |
| `dictate retry <path>` | Transcribe and insert a recording that a failed request kept |
| `dictate transcribe` | Transcribe and insert a finished recording that was never handed off |
| `dictate vocabulary ...` | Manage your own transcript corrections |
| `dictate config ...` | Show, set, or reset settings |
| `dictate services ...` | Install, start, stop, restart, inspect, or remove the service |
| `dictate doctor` | Check the setup and suggest fixes |

Dictation won't record or type while RuneLite has focus, because simulated
input may break Jagex's rules.

## Configure

Settings live in `~/.config/local-dictation/config.toml`, and `dictate config`
edits them safely:

```fish
dictate config show
dictate config set stt.device cpu
dictate config set dictation.inject paste
dictate services restart
```

Recurring mistakes are easiest to fix with the vocabulary store, which applies
without a restart:

```fish
dictate vocabulary add herder herdr
```

## Documentation

- [Installation](docs/installation.md)
- [Configuration](docs/configuration.md)
- [Vocabulary](docs/vocabulary.md)
- [Service management](docs/service-management.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Architecture](docs/architecture.md)
- [Security notes](docs/security.md)
- [Service hardening](docs/service-hardening.md)
- [Development](docs/development.md)
- [Hardware smoke checks](docs/hardware-smoke.md)
