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
    """Kinds of events emitted on :meth:`~multiverse.mvd.kernel.Kernel.stream_events`."""

    STATE_TRANSITION = "STATE_TRANSITION"
    PROJECTION_STATUS = "PROJECTION_STATUS"
    SUBMITTED = "SUBMITTED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    EXECUTOR_LOG = "EXECUTOR_LOG"


@dataclass
class KernelEvent:
    """One server-sent event for GUI or transport subscribers.

    Attributes:
        kind: Discriminator that classifies the event for routing.
        physical_attempt_id: The attempt this event belongs to.
        payload: Free-form data specific to the event kind.
        seq: Journal sequence number this event was derived from, when
            applicable; ``None`` for synthetic (non-journal) events.
    """

    kind: EventKind
    physical_attempt_id: str
    payload: Dict[str, Any]
    seq: Optional[int] = None
    """Journal seq this event was derived from, when applicable."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the wire format used by the socket transport.

        Returns:
            A plain dict with string keys; safe for JSON serialization over
            the Unix-domain socket transport.
        """
        return {
            "kind": self.kind.value,
            "physical_attempt_id": self.physical_attempt_id,
            "payload": dict(self.payload),
            "seq": self.seq,
        }
