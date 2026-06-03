"""Journal record schema (STRATEGY R3).

Records are encoded as newline-delimited JSON, one per line. A record body
embeds its payload inline up to ``INLINE_BLOB_SPILL_THRESHOLD`` bytes; larger
payloads spill to ``store/journal/blobs/<sha256>.json`` and the record
references the blob hash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

# 256 KiB — the inline-spill threshold from R4.
INLINE_BLOB_SPILL_THRESHOLD = 256 * 1024


class JournalKind(str, Enum):
    """Canonical record kinds. Adding a kind requires a strategy/ADR update.

    The set covers the surfaces the kernel will write into the journal during
    the saga lifecycle (S3, S8, R3).
    """

    JOB_INTENT = "JOB_INTENT"  # user submitted a run
    ADMITTED = "ADMITTED"  # resource broker admitted
    RESERVATION_GRANTED = "RESERVATION_GRANTED"  # M3: durable ledger grant
    RESERVATION_RELEASED = "RESERVATION_RELEASED"  # M3: durable ledger release
    CONTAINER_LAUNCH = "CONTAINER_LAUNCH"  # docker container created
    STATE_TRANSITION = "STATE_TRANSITION"  # generic primary-state move
    PROMOTE_PREPARE = "PROMOTE_PREPARE"  # promotion saga step 1
    PROMOTE_VALIDATE = "PROMOTE_VALIDATE"  # promotion saga step 2
    PROMOTE_STAGE = "PROMOTE_STAGE"  # promotion saga step 3
    PROMOTE_COMMIT_MANIFEST = "PROMOTE_COMMIT_MANIFEST"  # step 4
    PROMOTE_COMMIT_INDEX = "PROMOTE_COMMIT_INDEX"  # step 5
    PROMOTE_COMMIT_TRACKING = "PROMOTE_COMMIT_TRACKING"  # step 6 (projection)
    PROMOTION_QUARANTINE = "PROMOTION_QUARANTINE"  # R5 quarantine
    CANCEL_REQUESTED = "CANCEL_REQUESTED"  # S8 step 1
    CANCEL_STOPPED = "CANCEL_STOPPED"  # docker stop succeeded
    CANCEL_KILLED = "CANCEL_KILLED"  # docker kill after grace
    CANCELLED = "CANCELLED"  # S8 terminal
    ARTIFACT_MANIFEST_COMMIT = "ARTIFACT_MANIFEST_COMMIT"  # R4
    RECOVERY_NOTE = "RECOVERY_NOTE"  # rebuild-index annotation
    PROJECTION_STATUS = "PROJECTION_STATUS"  # MLflow/Optuna sync status


@dataclass
class JournalRecord:
    """One entry in the kernel's append-only intent record.

    ``payload`` may be either an inline mapping or ``{"blob_sha256": "..."}``
    after spill. Callers normally hand in the inline mapping and the writer
    decides whether to spill.
    """

    seq: int
    kind: JournalKind
    monotonic_ns: int
    wall_iso: str
    mvd_boot_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    physical_attempt_id: Optional[str] = None
    logical_run_id: Optional[str] = None
    prev_state: Optional[str] = None
    next_state: Optional[str] = None
    user_id: Optional[str] = None
    """Resolved owner of the run. Absent from pre-G2 records (reads as None).
    Stamped by JournalWriter on every record when a user_id is configured."""

    # ---- serialization ----

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "seq": int(self.seq),
            "kind": self.kind.value,
            "monotonic_ns": int(self.monotonic_ns),
            "wall_iso": self.wall_iso,
            "mvd_boot_id": self.mvd_boot_id,
            "payload": dict(self.payload),
        }
        if self.physical_attempt_id is not None:
            out["physical_attempt_id"] = self.physical_attempt_id
        if self.logical_run_id is not None:
            out["logical_run_id"] = self.logical_run_id
        if self.prev_state is not None:
            out["prev_state"] = self.prev_state
        if self.next_state is not None:
            out["next_state"] = self.next_state
        if self.user_id is not None:
            out["user_id"] = self.user_id
        return out

    def to_line(self) -> bytes:
        """Encode to one ND-JSON line including the trailing ``\\n``.

        ``sort_keys=True`` so that a record written twice (group commit
        replay) is byte-equal.
        """
        return (
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JournalRecord":
        try:
            return cls(
                seq=int(data["seq"]),
                kind=JournalKind(data["kind"]),
                monotonic_ns=int(data["monotonic_ns"]),
                wall_iso=str(data["wall_iso"]),
                mvd_boot_id=str(data["mvd_boot_id"]),
                payload=dict(data.get("payload") or {}),
                physical_attempt_id=data.get("physical_attempt_id"),
                logical_run_id=data.get("logical_run_id"),
                prev_state=data.get("prev_state"),
                next_state=data.get("next_state"),
                user_id=data.get("user_id"),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"malformed journal record: {exc}") from exc
