"""Local opt-out telemetry for GUI usage events."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ENV_FLAG = "MULTIVERSE_GUI_TELEMETRY"
_DEFAULT_PATH = Path.home() / ".multiverse" / "gui_events.jsonl"


def telemetry_enabled() -> bool:
    """Report whether GUI telemetry is enabled.

    Telemetry is on by default and opted out via ``MULTIVERSE_GUI_TELEMETRY``
    set to a falsy value (``0``/``false``/``no``/``off``).

    Returns:
        True unless the opt-out flag is set.
    """
    return os.environ.get(_ENV_FLAG, "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def track(event_name: str, **fields: Any) -> None:
    """Append a usage event as one JSON line to the local telemetry log.

    A no-op when telemetry is opted out. Writes are local-only and best-effort:
    I/O failures are swallowed so telemetry never disrupts the GUI.

    Args:
        event_name: Identifier for the event being recorded.
        **fields: Arbitrary event payload merged into the record alongside the
            event name and a UTC timestamp.
    """
    if not telemetry_enabled():
        return
    record = {
        "event": event_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    try:
        _DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEFAULT_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    except OSError:
        return
