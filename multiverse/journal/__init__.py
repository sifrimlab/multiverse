"""Journal core — STRATEGY.md Milestone 4 / R3.

The journal is the kernel's *intent* record: the only thing the kernel writes
synchronously before any side effect. Per ADR §8 the kernel is single-process
and uses asyncio; the journal API is therefore plain synchronous Python that
runs in the kernel's event-loop thread (no I/O threads).

Layout (per R3)::

    store/journal/
        current.log                   active append-only segment
        rotated/<seq>-<iso>.log[.zst] rotated segments
        blobs/<sha256>.json           content-addressed spill for large payloads
        checkpoint.json               last fully reconciled seq + offset

Record format (newline-delimited JSON, one record per line)::

    {"seq": 42,
     "monotonic_ns": ...,
     "wall_iso": "...",
     "mvd_boot_id": "...",
     "kind": "PROMOTE_PREPARE",
     "physical_attempt_id": "...",
     "prev_state": "TRAINING_SUCCEEDED",
     "next_state": "PROMOTING",
     "payload": {...} | {"blob_sha256": "..."}
    }

Durability boundary: a record is *acknowledged* (returned to the caller) only
after the segment file's data has been fsynced and — when the segment was
newly created — the parent directory has been fsynced. Group commit may batch
multiple appends in one fsync, but it MUST NOT ack any of them before the
batch fsync succeeds.
"""

from .errors import (JournalCorruptError, JournalError, JournalLocked,
                     JournalReplayError)
from .layout import JournalLayout
from .reader import JournalReader, ReplayResult
from .record import INLINE_BLOB_SPILL_THRESHOLD, JournalKind, JournalRecord
from .writer import JournalWriter, SegmentInfo

__all__ = [
    "INLINE_BLOB_SPILL_THRESHOLD",
    "JournalCorruptError",
    "JournalError",
    "JournalKind",
    "JournalLayout",
    "JournalLocked",
    "JournalReader",
    "JournalRecord",
    "JournalReplayError",
    "JournalWriter",
    "ReplayResult",
    "SegmentInfo",
]
