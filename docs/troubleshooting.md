# Troubleshooting

Start with the doctor. It checks the configuration, required executables, focus
automation, the service environment, the model cache, the compute device, the
PipeWire source, desktop support, the service, the socket, and the runtime
directory, and it suggests a command for anything that fails:

```fish
dictate doctor
dictate doctor --fix
```

`--fix` offers the repairs it knows (restarting a failed service or removing a
stale socket) and asks before each one.

## A command says "The command failed"

Failures show a generic notification, and the details go to standard error as
one JSON line. Commands started from a key binding write that line to your
window manager's log, so the quickest way to see it is to run the same command
in a terminal:

```fish
dictate toggle
```

## "STT is stopped or still loading"

The service isn't accepting connections yet. Right after a start or restart,
it's still loading the model; the log ends with `model ready` once it's done.

```fish
dictate services status
dictate services logs
```

## A transcription failed and kept the recording

When the service is busy for longer than two minutes, stops mid-request, or
fails, the recording stays under
`$XDG_RUNTIME_DIR/local-dictation/recordings/` and the error names it. Once the
service is healthy again, insert it with:

```fish
dictate retry $XDG_RUNTIME_DIR/local-dictation/recordings/recording-<uuid>.wav
```

If the service quarantined a file, the log names its path under
`stt-quarantine/`. The service never removes quarantined audio, so inspect or
delete it yourself. Like everything else in the runtime directory, it's gone
after you log out.

## The service won't start or keeps restarting

```fish
dictate services logs
dictate services audit
```

A missing Python module usually means the installed engine extra doesn't match
the configuration. `cpu` serves Parakeet on the CPU, `cuda` serves Parakeet on
an NVIDIA GPU, and `whisper` adds faster-whisper. Re-run the installer with the
device you want, or see
[Installation](installation.md#engine-extras) to add faster-whisper.

Explicit `cuda` fails at startup when the runtime can't see a CUDA device:

```fish
nvidia-smi
dictate config set stt.device cpu
dictate services restart
```

`cpu` never probes CUDA, which makes it a good way to rule the GPU out. CUDA
provider errors in the log usually mean the CUDA or cuDNN libraries are
missing; see [CUDA](installation.md#cuda).

## Nothing gets typed

- Check the desktop row in `dictate doctor`. GNOME and KDE Wayland sessions
  can't report the focused window, so only `stdout` works there.
- Insertion cancels if focus changes between recording and typing. Keep the
  target window focused until the text appears.
- Dictation refuses to record or type while RuneLite has focus.
- Set `dictation.inject` to `stdout` and run `dictate toggle` twice in a
  terminal to check that transcription works on its own.

## The wrong microphone, or VAD misbehaves

Pick the capture source from the devices PipeWire reports:

```fish
wpctl status
dictate config set dictation.source
```

If `dictate vad` triggers on background noise, raise
`dictation.vad_min_speech_rms` or `dictation.vad_start_speech_frames`. If it
cuts you off mid-sentence, raise `dictation.vad_end_silence_ms`. These apply to
the next `dictate vad` you start.

## The installed unit drifted

After pulling changes to the unit file, or if `dictate services audit` reports
a mismatch:

```fish
dictate services install --force
dictate services audit
dictate services restart
```
