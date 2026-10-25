from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from local_dictation import pipewire
from local_dictation.user_config import UserConfigurationError

_WPCTL_STATUS = """\
Audio
 ├─ Devices:
 │      40. alsa_card.pci-0000_00_1f.3
 ├─ Sinks:
 │  *   62. alsa_output.pci-0000_00_1f.3.analog-stereo [vol: 0.50]
 ├─ Sources:
 │  *   63. alsa_input.pci-0000_00_1f.3.analog-stereo [vol: 1.00]
 ├─ Filters:
"""

_WPCTL_FILTER_STATUS = _WPCTL_STATUS.replace(
    " ├─ Filters:\n",
    " ├─ Filters:\n │      72. echo_cancel_sink\n │  *   73. echo_cancel_source\n",
)


def _completed(returncode: int, stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, "")


class PipewireTests(unittest.TestCase):
    def test_terminal_section_header_is_parsed(self) -> None:
        status = """\
Audio
 ├─ Sinks:
 │  *   62. output.node
 └─ Sources:
    *   63. input.node
"""
        self.assertEqual(
            pipewire.pipewire_section_devices(status, "Sources"),
            ("input.node",),
        )

    def test_list_capture_sources_includes_sources_and_filters(self) -> None:
        with (
            mock.patch.object(pipewire, "_which", return_value="/usr/bin/wpctl"),
            mock.patch.object(
                pipewire, "_run", return_value=_completed(0, _WPCTL_FILTER_STATUS)
            ),
        ):
            names = pipewire.list_pipewire_capture_sources()
        self.assertEqual(
            names,
            (
                "alsa_input.pci-0000_00_1f.3.analog-stereo",
                "echo_cancel_sink",
                "echo_cancel_source",
            ),
        )

    def test_list_capture_sources_requires_wpctl(self) -> None:
        with (
            mock.patch.object(pipewire, "_which", return_value=None),
            self.assertRaisesRegex(UserConfigurationError, "wpctl not found"),
        ):
            pipewire.list_pipewire_capture_sources()

    def test_failed_wpctl_status_is_reported(self) -> None:
        with (
            mock.patch.object(pipewire, "_which", return_value="/usr/bin/wpctl"),
            mock.patch.object(pipewire, "_run", return_value=_completed(1)),
            self.assertRaisesRegex(UserConfigurationError, "wpctl status failed"),
        ):
            pipewire.list_pipewire_capture_sources()

    def test_empty_capture_listing_is_rejected(self) -> None:
        with (
            mock.patch.object(pipewire, "_which", return_value="/usr/bin/wpctl"),
            mock.patch.object(pipewire, "_run", return_value=_completed(0, "Audio\n")),
            self.assertRaisesRegex(UserConfigurationError, "no capture sources"),
        ):
            pipewire.list_pipewire_capture_sources()


if __name__ == "__main__":
    unittest.main()
