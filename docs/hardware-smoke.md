# Opt-in hardware smoke checks

These checks are manual and never run in CI. They use the microphone, load models
on the GPU, and type into real windows. Run only the section whose side effects
you want, on the machine you're testing. Every section except the first records
speech, so make sure whoever is at the microphone is ready before starting.

Install the project as described in [Installation](installation.md) first, and
keep a second terminal open for logs:

```fish
journalctl --user -u local-dictation -f
```

## Service and environment

```fish
dictate services status
dictate doctor
```

Checklist:

- The service is `active` and the log ends with `model ready`.
- `dictate doctor` reports no failures, and the desktop row matches the session
  (X11, Hyprland, or Sway can insert text; GNOME and KDE Wayland can't).

## PipeWire capture

```fish
wpctl status
dictate config set dictation.source
dictate config set dictation.inject stdout
dictate begin
# Say a short sentence.
dictate end
```

Checklist:

- The chosen source appears under the audio sources in `wpctl status`.
- `begin` prints `recording` without a PipeWire error.
- `end` prints the sentence you said.
- `$XDG_RUNTIME_DIR/local-dictation/recordings/` is empty afterwards.

## Inserting text

Set `dictation.inject` back to `auto`, then dictate into a text editor and into
a terminal with `dictate toggle` bound to a key.

Checklist:

- The editor receives pasted text, and its previous clipboard contents come back
  afterwards.
- The terminal receives typed (or <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>V</kbd>
  pasted) text.
- Switching windows between the end of recording and insertion cancels the
  insertion with a notification instead of typing into the new window.

## Voice-activated dictation

```fish
dictate vad
```

Say two separate sentences with a pause after each, stay quiet for a few
seconds, then run `dictate vad` again from another terminal or key binding.

Checklist:

- Each sentence is transcribed and inserted after its pause, and the listener
  rearms between them.
- Silence and background noise don't produce empty insertions.
- With steady background noise, such as a fan, each sentence still finishes
  after its pause instead of running on to `dictation.vad_max_seconds`.
- The second `dictate vad` stops the listener, and the first process exits.

## CUDA model loading

```fish
nvidia-smi
dictate config set stt.device cuda
dictate services restart
journalctl --user -u local-dictation -n 50
```

Checklist:

- The log shows Parakeet loading on `CUDA` with no CPU fallback, missing library,
  or out-of-memory error.
- `nvidia-smi` lists the server process with plausible VRAM use.
- A dictation round trip still works.

Return to your usual device setting afterwards.

## Recovery and retry

```fish
dictate config set dictation.inject stdout
dictate begin
# Say a short sentence.
dictate services stop
dictate end
dictate services start
# Wait for "model ready" in the log.
dictate retry <path from the error>
```

Checklist:

- `end` fails, and its diagnostic names the preserved recording and the
  `dictate retry` command.
- The recording stays in `recordings/` while the service is stopped.
- `retry` prints the sentence and the recording is removed afterwards.
