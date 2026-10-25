# Architecture

```text
PipeWire microphone
  -> dictate (begin/end/toggle, or vad)
       -> pw-record, or a raw stream framed by Silero VAD
       -> immutable recording generation
  -> local-dictation.service over a private Unix socket
       -> Parakeet TDT 0.6B v2 via ONNX Runtime (CPU or CUDA), or
       -> faster-whisper (CPU or CUDA)
       -> transcript corrections (vocabulary, then configured replacements)
  -> dictate
       -> herdr pane, clipboard paste, typed input, or stdout
  -> focused window
```

There are two processes. `local-dictation.service` is a systemd user service that
loads the speech model once and serves transcriptions over
`$XDG_RUNTIME_DIR/local-dictation.sock`. `dictate` is a short-lived command that
desktop key bindings run. It owns the microphone while recording, asks the
service for a transcript, and inserts the text into the focused window. Audio and
text never leave the machine.

## Recording

`dictate begin` starts `pw-record` on the configured PipeWire source, and
`dictate end` stops it. `dictate toggle` does whichever of the two applies. The
recorder doesn't start until the service socket accepts connections, so a key
press while the model is still loading fails straight away instead of recording
audio that can't be transcribed.

Recorder ownership is a JSON file holding the recorder's PID and its Linux process
start time, written only after `pw-record` is ready. Stop and cancel compare both
values before signalling, so a reused PID is never signalled. A cross-process
`recording.lock` serializes every start, stop, cancel, and hand-off.

`dictate vad` enables hands-free dictation. The invoking process stays running,
repeatedly owns a raw PipeWire stream, and runs each 80 ms frame through the
[Silero VAD](https://github.com/snakers4/silero-vad) model on one CPU thread.
Frames quieter than `dictation.vad_min_speech_rms` never count as speech, but
steady background noise above that level doesn't count either, because the
model listens for speech rather than loudness. It needs several consecutive
speech frames before an utterance starts, finishes the utterance after the
configured silence, then transcribes it and listens again, including when STT
finds no recognizable speech. Running `dictate vad` again signals the owner to
stop instead of touching its audio.

The first `dictate vad` downloads the 2 MB model from Hugging Face at a pinned
revision. Later runs load it from `~/.cache/huggingface` without a network
request.

## Hand-off and transcription

Stopping capture atomically renames the writable `recording.wav` to an immutable
generation, `recordings/recording-<uuid>.wav`, while the recorder lock is still
held. A later recording only ever replaces the writable path, so a slow
transcription can't be overwritten by the next key press. VAD performs the same
hand-off at the end of each utterance.

The service accepts only strictly named generations in its own `recordings/`
directory. After acquiring the model slot, it moves the generation to a unique
private path under `stt-processing/` and transcribes it there.

The client and server speak version 2 of a newline-delimited JSON protocol:

1. The client sends `{"type": "transcribe", "audio_path": ...}`.
2. The server replies with the complete transcript and a unique delivery ID.
3. The client acknowledges that ID only after it has parsed the transcript.
4. A matching acknowledgment moves the claim to `stt-delivered/`, and the
   server then removes it.

Backend failures, response failures, disconnects, timeouts, and invalid
acknowledgments restore the original generation so it can be retried, without
overwriting another file. Clients that send only a raw path still receive a
transcript, but they can't authorize deletion, so the server restores those
generations too.

Only one transcription runs at a time. A second request gets a structured
`server_busy` error straight away instead of queueing behind a possibly hung model
call, and its generation stays where it is. The client retries the same
generation with bounded backoff for up to 120 seconds. If STT is still busy, the
error names the preserved file and the command to retry it:

```fish
dictate retry $XDG_RUNTIME_DIR/local-dictation/recordings/recording-<uuid>.wav
```

`dictate retry` re-checks the generation (it must be in `recordings/`, owned by
you, private, and non-empty) and then transcribes and inserts it through the
normal path, including the focus and blocked-window checks. `dictate transcribe`
does the same for a finished `recording.wav` that was never handed off.

Python can't safely cancel a hung native GPU call, so the accepted call stays
synchronous. Recovering from a hung model means restarting the service.

## Startup recovery

Before loading the model, the service restores valid `stt-processing/` claims to
`recordings/` and finishes cleaning up claims that had already been delivered. It
recognizes a hard-link restore that was interrupted halfway. Malformed claims and
conflicting destination files move to a private `stt-quarantine/` directory, and
the service logs their exact paths. The runtime directory survives service
restarts so that this recovery can run. Everything in it is session-scoped and
disappears when `$XDG_RUNTIME_DIR` is retired at logout.

## Transcript corrections

The service applies corrections to every transcript in this order:

1. replacements from the personal vocabulary store,
2. `dictation.replacements` from the configuration the service started with,
3. built-in defaults.

Matching is case-insensitive, whitespace-flexible, and only fires on whole
phrases. The service reads the vocabulary store for each transcription, so
`dictate vocabulary` edits apply without a restart. See
[Vocabulary](vocabulary.md).

## Inserting text

`dictation.inject` picks how the transcript reaches the focused window:

- `stdout` prints it and doesn't touch the desktop.
- `type` simulates keystrokes with xdotool on X11 or wtype on Wayland.
- `paste` copies it to the clipboard, sends <kbd>Ctrl</kbd>+<kbd>V</kbd>
  (<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>V</kbd> in terminals), and restores
  the previous clipboard if nothing else changed it in the meantime.
- `auto`, the default, types into terminals and pastes everywhere else.

When the focused terminal runs [herdr](https://herdr.dev) and its focused pane is
a Cursor agent, `auto` and `paste` send the text to that pane with
`herdr pane send-text` instead.

Every insertion records the focused window first and cancels if focus changes
before the text goes in. Dictation refuses to record or insert while RuneLite has
focus, because simulated input may break Jagex's rules.

Window detection uses `xdotool` on X11, `hyprctl` on Hyprland, and `swaymsg` on
Sway. GNOME and KDE Wayland sessions don't expose the focused window to other
programs, so they can only use `stdout`. `dictate doctor` reports what the
current session supports.
