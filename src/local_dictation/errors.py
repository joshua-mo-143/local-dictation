class DictationError(RuntimeError):
    """User-facing dictation failure."""


class NoSpeechError(DictationError):
    """The speech-to-text backend found no recognizable speech."""
