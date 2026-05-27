"""Kernel event stream.

``stream_events`` returns server-sent events: state transitions and log
tail. Milestone 7 implements state transitions only; log-tail follow is
Milestone 9 (GUI cutover) territory.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class EventKind(str, Enum):
    STATE_TRANSITION = "STATE_TRANSITION"
    PROJECTION_STATUS = "PROJECTION_STATUS"
    SUBMITTED = "SUBMITTED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    EXECUTOR_LOG = "EXECUTOR_LOG"


@dataclass
class KernelEvent:
    kind: EventKind
    physical_attempt_id: str
    payload: Dict[str, Any]
    seq: Optional[int] = None
    """Journal seq this event was derived from, when applicable."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "physical_attempt_id": self.physical_attempt_id,
            "payload": dict(self.payload),
            "seq": self.seq,
        }
