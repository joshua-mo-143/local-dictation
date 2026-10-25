# Security notes

- Speech recognition runs locally. Audio, transcripts, and the vocabulary store
  never leave the machine. Models download once from Hugging Face into
  `~/.cache/huggingface`.
- Transcription can be wrong, and dictated text goes straight into the focused
  window. Check what was typed before you press <kbd>Enter</kbd> in a terminal.
- Recordings, recorder ownership, logs, and the service socket are session data
  under `$XDG_RUNTIME_DIR`. The runtime directory and its subdirectories are
  private to your user (`0700`), audio claims are `0600`, and the socket is
  `0600`.
- The service only transcribes strictly named recording generations inside its
  own runtime directory. It rejects any other path a client sends, so a process
  that can reach the socket can't make the service read arbitrary files.
- Every insertion re-checks the focused window and cancels if focus changed after
  recording. Paste mode restores the previous clipboard unless something else
  replaced it first.
- Dictation refuses to record or insert while RuneLite has focus, because
  simulated input may break Jagex's rules.
- Service logs and command-failure diagnostics pass through a redaction filter
  that removes common credential formats, and failure notifications show a
  generic message instead of the error text. The service logs model loading,
  recovery, and quarantine paths, but not transcripts.
- `config.toml` and `vocabulary.json` are written atomically and owner-only.
- The service runs with service-specific systemd hardening and bounded memory;
  see the [hardening policy](service-hardening.md) for the deliberate
  exceptions and host checks.
