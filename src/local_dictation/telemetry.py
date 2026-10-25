"""Environment settings that keep bundled engines from reporting usage."""

from __future__ import annotations

from collections.abc import Mapping

# Recent ONNX Runtime releases keep a device ID and upload usage events from
# Linux unless this is "1".
TELEMETRY_OPT_OUTS: Mapping[str, str] = {"ORT_DISABLE_TELEMETRY": "1"}
