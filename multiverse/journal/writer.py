"""Append-only journal writer with group commit, rotation, and blob spill.

Durability protocol (R3):
    * Open the segment with append semantics. Buffer record bytes in memory
      only between ``append`` and ``flush`` calls in the same group-commit
      cycle; never lose a record across an ack boundary.
    * ``commit()`` fsyncs the file data, then (if the segment is new) the
      parent directory inode. Only after both fsyncs return success does it
      ack the pending appends.
    * Rotation: when ``current.log`` reaches the size threshold, rename it
      to ``rotated/<base-seq>-<iso>.log`` (optionally zstd-compressed) and
      open a fresh ``current.log``. Rotation is itself fsynced before any
      record from the new segment may be acked.
    * Blob spill: a record whose ``payload`` JSON exceeds
      ``INLINE_BLOB_SPILL_THRESHOLD`` bytes is rewritten to reference a
      content-addressed blob; the blob file is written atomically (tmp →
      fsync → rename → fsync parent) before the journal record is appended.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..artifact.checksums import atomic_write_bytes, fsync_path, sha256_bytes
from .errors import JournalError
from .layout import JournalLayout
from .record import INLINE_BLOB_SPILL_THRESHOLD, JournalKind, JournalRecord

DEFAULT_SEGMENT_MAX_BYTES = 16 * 1024 * 1024   # 16 MiB per segment


@dataclass(frozen=True)
class SegmentInfo:
    path: Path
    base_seq: int  # seq of the first record in this segment
    size_bytes: int


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _now_monotonic_ns() -> int:
    return time.monotonic_ns()


class JournalWriter:
    """Single-writer journal. Construction opens the current segment for
    append; ``close()`` flushes any pending records and closes the file.

    The writer is single-threaded by contract: callers are responsible for
    serialising ``append``/``commit`` calls. The kernel's asyncio event loop
    runs the writer in its own task, so the single-threaded constraint is
    free.
    """

    def __init__(
        self,
        layout: JournalLayout,
        *,
        boot_id: str,
        starting_seq: Optional[int] = None,
        segment_max_bytes: int = DEFAULT_SEGMENT_MAX_BYTES,
        inline_blob_threshold: int = INLINE_BLOB_SPILL_THRESHOLD,
        fsync_enabled: bool = True,
    ) -> None:
        self._layout = layout.ensure()
        self._boot_id = boot_id
        self._segment_max_bytes = int(segment_max_bytes)
        self._inline_blob_threshold = int(inline_blob_threshold)
        self._fsync_enabled = bool(fsync_enabled)

        # ``_pending`` are records appended-but-not-yet-acked. ``commit``
        # forces them to disk and returns their seq numbers.
        self._pending: List[JournalRecord] = []
        self._pending_bytes: List[bytes] = []

        self._segment_size: int = 0
        self._segment_fd: int = -1
        self._segment_is_new: bool = False
        self._segment_base_seq: int = 0
        self._segment_path: Path = self._layout.current_segment

        # Seq counter is global across boot lifetimes; replay determines the
        # next free seq from disk if ``starting_seq`` is not supplied.
        if starting_seq is None:
            starting_seq = _scan_max_seq_plus_one(self._layout)
        self._next_seq = int(starting_seq)

        self._open_current_segment()

    # ---- public API --------------------------------------------------

    @property
    def boot_id(self) -> str:
        return self._boot_id

    @property
    def next_seq(self) -> int:
        return self._next_seq

    def append(
        self,
        kind: JournalKind,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        physical_attempt_id: Optional[str] = None,
        logical_run_id: Optional[str] = None,
        prev_state: Optional[str] = None,
        next_state: Optional[str] = None,
    ) -> int:
        """Stage a record for the next group commit. Returns the seq it was
        assigned.

        The record is NOT durable until ``commit()`` is called and returns.
        Callers must not externalise the seq before commit succeeds.
        """
        payload_dict: Dict[str, Any] = dict(payload or {})
        encoded_payload = self._encode_payload(payload_dict)

        record = JournalRecord(
            seq=self._next_seq,
            kind=kind,
            monotonic_ns=_now_monotonic_ns(),
            wall_iso=_now_iso(),
            mvd_boot_id=self._boot_id,
            payload=encoded_payload,
            physical_attempt_id=physical_attempt_id,
            logical_run_id=logical_run_id,
            prev_state=prev_state,
            next_state=next_state,
        )
        line = record.to_line()

        # Rotate eagerly so the *new* record never spans two segments.
        prospective_size = self._segment_size + sum(
            len(b) for b in self._pending_bytes
        ) + len(line)
        if (
            prospective_size > self._segment_max_bytes
            and (self._segment_size > 0 or self._pending_bytes)
        ):
            # Commit anything already pending so it stays in the *current*
            # segment, then rotate before adding the new record.
            self.commit()
            self._rotate_segment()

        self._pending.append(record)
        self._pending_bytes.append(line)
        self._next_seq += 1
        return record.seq

    def commit(self) -> List[int]:
        """Flush pending records, fsync, and acknowledge them.

        Returns the seq numbers that were durably persisted by this call
        (empty if no records were pending). On error the writer raises and
        keeps the pending records — callers can retry ``commit``.
        """
        if not self._pending_bytes:
            return []

        # Concatenate so the write hits the disk in one syscall; this is
        # the group-commit batch.
        batch = b"".join(self._pending_bytes)
        try:
            os.write(self._segment_fd, batch)
        except OSError as exc:
            raise JournalError(f"journal write failed: {exc}") from exc

        if self._fsync_enabled:
            try:
                os.fsync(self._segment_fd)
            except OSError as exc:
                raise JournalError(f"journal fsync failed: {exc}") from exc
            if self._segment_is_new:
                fsync_path(self._layout.root)
                self._segment_is_new = False

        # Records are now durable. Ack them.
        committed_seqs = [r.seq for r in self._pending]
        self._segment_size += len(batch)
        self._pending.clear()
        self._pending_bytes.clear()
        return committed_seqs

    def write_checkpoint(self, *, last_seq: int) -> None:
        """Atomically rewrite ``checkpoint.json`` with the latest fully-
        reconciled seq.

        Replay uses the checkpoint to skip records that the index already
        absorbed; it must never be advanced past a seq that has not been
        durably acked.
        """
        payload = {
            "last_seq": int(last_seq),
            "mvd_boot_id": self._boot_id,
            "wall_iso": _now_iso(),
        }
        atomic_write_bytes(
            self._layout.checkpoint,
            json.dumps(payload, sort_keys=True, indent=2).encode("utf-8"),
            fsync=self._fsync_enabled,
        )

    def close(self) -> None:
        if self._pending_bytes:
            self.commit()
        if self._segment_fd >= 0:
            try:
                os.close(self._segment_fd)
            finally:
                self._segment_fd = -1

    # ---- internal ----------------------------------------------------

    def _open_current_segment(self) -> None:
        path = self._layout.current_segment
        new_segment = not path.exists()
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        self._segment_fd = os.open(str(path), flags, 0o644)
        self._segment_size = os.fstat(self._segment_fd).st_size
        self._segment_path = path
        self._segment_is_new = new_segment
        if new_segment:
            self._segment_base_seq = self._next_seq
        else:
            self._segment_base_seq = _first_seq_in(path) or self._next_seq

    def _rotate_segment(self) -> None:
        """Rename current.log to a stamped rotated file, open a fresh one."""
        os.close(self._segment_fd)
        rotated_name = (
            f"{self._segment_base_seq:020d}"
            f"-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log"
        )
        rotated_path = self._layout.rotated_dir / rotated_name
        os.replace(str(self._layout.current_segment), str(rotated_path))
        if self._fsync_enabled:
            fsync_path(self._layout.rotated_dir)
            fsync_path(self._layout.root)
        self._segment_fd = -1
        self._segment_size = 0
        self._open_current_segment()

    def _encode_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Spill large payloads into ``blobs/<sha256>.json``.

        The journal record then records ``{"blob_sha256": "...", "blob_path":
        "..."}`` instead of the inline body.
        """
        body = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if len(body) <= self._inline_blob_threshold:
            return payload

        digest = sha256_bytes(body)
        blob_path = self._layout.blobs_dir / f"{digest}.json"
        if not blob_path.exists():
            atomic_write_bytes(blob_path, body, fsync=self._fsync_enabled)
        return {
            "blob_sha256": digest,
            "blob_path": str(blob_path.relative_to(self._layout.root)),
            "_original_size": len(body),
        }


# ---------------------------------------------------------------------------
# Helpers for replay / startup
# ---------------------------------------------------------------------------


def _iter_segment_files(layout: JournalLayout) -> Iterable[Path]:
    """Yield rotated segments in seq-base order, then current.log."""
    rotated = sorted(layout.rotated_dir.glob("*.log"), key=lambda p: p.name)
    yield from rotated
    if layout.current_segment.exists():
        yield layout.current_segment


def _scan_max_seq_plus_one(layout: JournalLayout) -> int:
    """Walk every segment to find the highest seq, return seq+1.

    Used at writer construction when ``starting_seq`` is not supplied. Linear
    in journal size at startup; the kernel calls it once per boot.
    """
    max_seq = -1
    for segment in _iter_segment_files(layout):
        try:
            with segment.open("rb") as fp:
                # Read lines in chunks to bound RSS on huge segments.
                for line in fp:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        seq = int(obj.get("seq", -1))
                        if seq > max_seq:
                            max_seq = seq
                    except (ValueError, TypeError):
                        # Truncated final line from a crash: stop scanning
                        # this segment; everything after a corrupt line is
                        # treated as not-yet-acked. R3 lets us ignore the
                        # tail without quarantining the segment.
                        break
        except FileNotFoundError:
            continue
    return max_seq + 1


def _first_seq_in(segment: Path) -> Optional[int]:
    try:
        with segment.open("rb") as fp:
            for line in fp:
                if not line.strip():
                    continue
                try:
                    return int(json.loads(line).get("seq"))
                except (ValueError, TypeError):
                    return None
    except FileNotFoundError:
        return None
    return None


@contextmanager
def open_writer(layout: JournalLayout, *, boot_id: str, **kwargs: Any):
    """Convenience CM that ensures the writer is closed on exit."""
    writer = JournalWriter(layout, boot_id=boot_id, **kwargs)
    try:
        yield writer
    finally:
        writer.close()
