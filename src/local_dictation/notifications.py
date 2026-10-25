from __future__ import annotations

import shutil
import subprocess
from enum import StrEnum

APP_NAME = "local-dictation"
SUMMARY = "Dictation"


class NotificationResult(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


CAPABILITY_PROBE_TIMEOUT_SECONDS = 2


def notification_service_binary_available() -> bool:
    return shutil.which("notify-send") is not None


def notification_service_available() -> bool:
    executable = shutil.which("notify-send")
    if executable is None:
        return False
    try:
        process = subprocess.run(
            [
                executable,
                "--print-id",
                "--expire-time=1",
                "--transient",
                f"--app-name={APP_NAME}",
                f"{SUMMARY} capability probe",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=CAPABILITY_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return process.returncode == 0


def notify(message: str, *, error: bool = False) -> NotificationResult:
    urgency = "critical" if error else "normal"
    try:
        result = subprocess.run(
            ["notify-send", "-u", urgency, "-t", "5000", SUMMARY, message],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return NotificationResult.FAILED
    if result.returncode != 0:
        return NotificationResult.FAILED
    return NotificationResult.SUCCEEDED
