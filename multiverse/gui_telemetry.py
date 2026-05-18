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
    return os.environ.get(_ENV_FLAG, "1").strip().lower() not in {"0", "false", "no", "off"}


def track(event_name: str, **fields: Any) -> None:
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
