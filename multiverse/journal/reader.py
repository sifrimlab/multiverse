"""Journal reader / replay.

Per R3 replay is the *first* phase of rebuild: it produces the kernel's
authoritative intent stream. Replay is forward-only, deterministic, and
tolerant of a truncated tail in ``current.log`` (the common crash-mid-write
shape).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from .errors import JournalCorruptError, JournalReplayError
from .layout import JournalLayout
from .record import JournalRecord


@dataclass
class ReplayResult:
    """Aggregate of a full replay walk."""

    records: List[JournalRecord] = field(default_factory=list)
    last_seq: int = -1
    truncated_tail_at: Optional[Path] = None
    """If the active segment ended in a truncated record, the path of that
    segment is recorded here for the kernel to surface to ``doctor``."""

    @property
    def is_empty(self) -> bool:
        return not self.records


@dataclass
class JournalReader:
    """Replay reader.

    ``replay()`` is the eager all-in-memory form used by rebuild-index and
    kernel boot reconciliation. ``stream()`` yields records lazily for
    callers (e.g. log-tail) that want to follow new appends — though the
    follow-tail feature itself is deferred to Milestone 8.
    """

    layout: JournalLayout

    # ---- public API --------------------------------------------------

    def replay(self, *, from_seq: int = 0) -> ReplayResult:
        """Eagerly walk all segments into an in-memory ``ReplayResult``.

        Forward-only and deterministic. A truncated tail in the active segment
        is treated as not-yet-acked and stops the scan without raising; a
        backwards seq across committed records is genuine corruption.

        Args:
            from_seq: Skip records below this seq (e.g. resume past a
                checkpoint the index already absorbed).

        Returns:
            The replayed records, the highest seq seen, and the path of any
            truncated tail segment.

        Raises:
            JournalReplayError: If the seq counter goes backwards across
                committed records, indicating an incoherent stream.
        """
        result = ReplayResult()
        last_seq = -1
        truncated_at: Optional[Path] = None
        for segment in _ordered_segments(self.layout):
            for raw, ok in _iter_lines(segment):
                if not ok:
                    # Truncated final line. Per R3 the tail is treated as
                    # not-yet-acked; we stop scanning this segment but do
                    # not raise.
                    truncated_at = segment
                    break
                record = _decode(raw, segment)
                if record.seq < from_seq:
                    continue
                if record.seq <= last_seq:
                    raise JournalReplayError(
                        f"seq went backwards across segments at {segment}: "
                        f"saw {record.seq} after {last_seq}"
                    )
                result.records.append(record)
                last_seq = record.seq
        result.last_seq = last_seq
        result.truncated_tail_at = truncated_at
        return result

    def stream(self, *, from_seq: int = 0) -> Iterator[JournalRecord]:
        """Lazy iterator over records (no follow-tail; one-shot)."""
        last_seq = -1
        for segment in _ordered_segments(self.layout):
            for raw, ok in _iter_lines(segment):
                if not ok:
                    return  # truncated tail; stop
                record = _decode(raw, segment)
                if record.seq < from_seq:
                    continue
                if record.seq <= last_seq:
                    raise JournalReplayError(
                        f"seq went backwards across segments at {segment}: "
                        f"saw {record.seq} after {last_seq}"
                    )
                yield record
                last_seq = record.seq

    def read_checkpoint(self) -> int:
        """Return the last fully-reconciled seq, or -1 if none recorded."""
        path = self.layout.checkpoint
        if not path.is_file():
            return -1
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return int(data.get("last_seq", -1))
        except (OSError, ValueError, TypeError):
            return -1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ordered_segments(layout: JournalLayout) -> Iterable[Path]:
    """Yield rotated segments in seq-base order, then the active segment."""
    rotated = sorted(layout.rotated_dir.glob("*.log"), key=lambda p: p.name)
    yield from rotated
    if layout.current_segment.exists():
        yield layout.current_segment


def _iter_lines(segment: Path) -> Iterable[tuple[bytes, bool]]:
    """Yield ``(line_bytes, ok)`` for each line in a segment.

    ``ok=False`` for the *last* line if it lacks a trailing ``\\n`` — i.e.
    the writer crashed mid-line. After such a yield, callers must stop
    scanning this segment.
    """
    with segment.open("rb") as fp:
        leftover = b""
        for chunk in iter(lambda: fp.read(1 << 16), b""):
            data = leftover + chunk
            *complete, leftover = data.split(b"\n")
            for line in complete:
                if line.strip():
                    yield line, True
        # ``leftover`` is whatever came after the last newline. If it is
        # non-empty, the writer crashed mid-record.
        if leftover.strip():
            yield leftover, False


def _decode(raw: bytes, segment: Path) -> JournalRecord:
    """Parse one ND-JSON line into a record, or raise ``JournalCorruptError``."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JournalCorruptError(f"unparseable record in {segment}: {exc}") from exc
    try:
        return JournalRecord.from_dict(obj)
    except ValueError as exc:
        raise JournalCorruptError(f"malformed record in {segment}: {exc}") from exc
