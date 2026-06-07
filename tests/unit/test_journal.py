"""Milestone-4 exit-gate tests for the journal core.

Coverage:
    1. ``append`` returns a seq, and only ``commit`` makes it durable.
    2. Crash-after-commit: a record acked by commit survives process death
       (simulated by abandoning the writer without close()).
    3. Crash-before-commit: pending-but-not-acked records are absent on
       replay.
    4. Rotation: a segment that exceeds the size threshold rotates, and
       replay across rotated + current is monotonic in seq.
    5. Truncated tail in the active segment is tolerated; replay stops at
       the corrupt line but does not raise.
    6. Blob spill: a record with a payload above the threshold lands in
       ``blobs/<sha256>.json`` and the journal record carries the digest.
    7. Multi-boot replay: writer started a second time picks up next_seq
       correctly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from multiverse.journal import (INLINE_BLOB_SPILL_THRESHOLD, JournalKind,
                                JournalLayout, JournalReader, JournalRecord,
                                JournalWriter)
from multiverse.journal.errors import JournalReplayError


def _writer(tmp_path: Path, **kwargs) -> JournalWriter:
    return JournalWriter(
        JournalLayout.at(tmp_path / "journal"), boot_id="boot-A", **kwargs
    )


def _records(reader: JournalReader) -> list[JournalRecord]:
    return reader.replay().records


# ---------------------------------------------------------------------------
# 1. append vs commit semantics
# ---------------------------------------------------------------------------


def test_commit_required_before_record_is_durable(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    seq = writer.append(JournalKind.JOB_INTENT, payload={"x": 1})
    assert seq == 0

    # Snapshot the segment without committing — the bytes should not yet be
    # on disk.
    seg = JournalLayout.at(tmp_path / "journal").current_segment
    assert seg.stat().st_size == 0

    writer.commit()
    assert seg.stat().st_size > 0
    writer.close()

    reader = JournalReader(JournalLayout.at(tmp_path / "journal"))
    records = _records(reader)
    assert [r.seq for r in records] == [0]
    assert records[0].kind is JournalKind.JOB_INTENT
    assert records[0].payload == {"x": 1}


def test_commit_returns_acked_seq_list(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    a = writer.append(JournalKind.JOB_INTENT)
    b = writer.append(JournalKind.ADMITTED)
    assert writer.commit() == [a, b]
    assert writer.commit() == []  # no pending
    writer.close()


# ---------------------------------------------------------------------------
# 2-3. Crash semantics
# ---------------------------------------------------------------------------


def test_committed_record_survives_writer_abandonment(tmp_path: Path) -> None:
    """Simulate process kill *after* commit: the file descriptor is leaked
    and the writer is dropped without close()."""
    writer = _writer(tmp_path)
    writer.append(JournalKind.JOB_INTENT, payload={"v": "ok"})
    writer.commit()
    # Drop reference without closing. Python's GC may close the FD but our
    # contract is that the data is already durable on disk by the time
    # commit returns.
    del writer

    reader = JournalReader(JournalLayout.at(tmp_path / "journal"))
    records = _records(reader)
    assert len(records) == 1
    assert records[0].payload == {"v": "ok"}


def test_uncommitted_records_do_not_survive_abandonment(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.append(JournalKind.JOB_INTENT)
    # No commit. Drop the writer.
    del writer

    reader = JournalReader(JournalLayout.at(tmp_path / "journal"))
    assert _records(reader) == []


# ---------------------------------------------------------------------------
# 4. Rotation
# ---------------------------------------------------------------------------


def test_rotation_preserves_seq_monotonicity(tmp_path: Path) -> None:
    # Pick a tiny segment size to force rotation.
    writer = _writer(tmp_path, segment_max_bytes=512)

    # Each record is around 200 bytes; write enough to trigger ≥2 rotations.
    for i in range(20):
        writer.append(JournalKind.STATE_TRANSITION, payload={"i": i})
        writer.commit()

    writer.close()

    layout = JournalLayout.at(tmp_path / "journal")
    rotated = sorted(layout.rotated_dir.glob("*.log"))
    assert rotated, "rotation must have produced at least one rotated segment"

    reader = JournalReader(layout)
    records = _records(reader)
    seqs = [r.seq for r in records]
    assert seqs == list(range(20)), f"seq must be monotonic across rotation; got {seqs}"


def test_rotated_segment_filename_is_sortable(tmp_path: Path) -> None:
    writer = _writer(tmp_path, segment_max_bytes=256)
    for i in range(10):
        writer.append(JournalKind.STATE_TRANSITION, payload={"x": i})
        writer.commit()
    writer.close()

    layout = JournalLayout.at(tmp_path / "journal")
    rotated = sorted(layout.rotated_dir.glob("*.log"), key=lambda p: p.name)
    # File names start with a zero-padded base seq so lexical sort = seq
    # sort.
    names = [p.name for p in rotated]
    base_seqs = [int(n.split("-", 1)[0]) for n in names]
    assert base_seqs == sorted(base_seqs)


# ---------------------------------------------------------------------------
# 5. Truncated tail tolerance
# ---------------------------------------------------------------------------


def test_truncated_tail_is_tolerated_on_replay(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    for i in range(3):
        writer.append(JournalKind.STATE_TRANSITION, payload={"i": i})
    writer.commit()
    writer.close()

    layout = JournalLayout.at(tmp_path / "journal")
    seg = layout.current_segment
    # Append a partial JSON line — simulating the writer dying mid-write.
    with seg.open("ab") as fp:
        fp.write(b'{"seq":99,"kind":"PROMOTE_PREPARE",')  # no newline

    reader = JournalReader(layout)
    result = reader.replay()
    # The three good records are present.
    assert [r.seq for r in result.records] == [0, 1, 2]
    # And the reader flagged the truncated segment.
    assert result.truncated_tail_at == seg


def test_replay_raises_on_seq_going_backwards(tmp_path: Path) -> None:
    """Adversarial input: hand-edit a segment so seq decreases. Replay must
    detect this and raise rather than silently emit the bad ordering."""
    writer = _writer(tmp_path)
    writer.append(JournalKind.STATE_TRANSITION, payload={"a": 1})
    writer.append(JournalKind.STATE_TRANSITION, payload={"a": 2})
    writer.commit()
    writer.close()

    layout = JournalLayout.at(tmp_path / "journal")
    seg = layout.current_segment
    # Replace the second record's seq with a smaller value.
    lines = seg.read_bytes().splitlines()
    obj = json.loads(lines[1])
    obj["seq"] = 0
    lines[1] = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    seg.write_bytes(b"\n".join(lines) + b"\n")

    reader = JournalReader(layout)
    with pytest.raises(JournalReplayError):
        reader.replay()


# ---------------------------------------------------------------------------
# 6. Blob spill
# ---------------------------------------------------------------------------


def test_large_payload_spills_to_blob(tmp_path: Path) -> None:
    writer = _writer(tmp_path, inline_blob_threshold=1024)
    big_payload = {"body": "x" * 2048}
    seq = writer.append(JournalKind.ARTIFACT_MANIFEST_COMMIT, payload=big_payload)
    writer.commit()
    writer.close()

    layout = JournalLayout.at(tmp_path / "journal")
    reader = JournalReader(layout)
    records = _records(reader)
    assert len(records) == 1
    record = records[0]
    assert record.seq == seq
    # Payload was rewritten to a blob reference.
    assert "blob_sha256" in record.payload
    digest = record.payload["blob_sha256"]
    blob_path = layout.blobs_dir / f"{digest}.json"
    assert blob_path.is_file()
    # Blob contents round-trip to the original payload.
    blob_obj = json.loads(blob_path.read_bytes())
    assert blob_obj == big_payload


def test_small_payload_does_not_spill(tmp_path: Path) -> None:
    writer = _writer(tmp_path, inline_blob_threshold=1024)
    writer.append(JournalKind.STATE_TRANSITION, payload={"small": "ok"})
    writer.commit()
    writer.close()
    layout = JournalLayout.at(tmp_path / "journal")
    assert not list(layout.blobs_dir.iterdir()), "small payload must not spill"


# ---------------------------------------------------------------------------
# 7. Multi-boot continuity
# ---------------------------------------------------------------------------


def test_second_boot_continues_seq(tmp_path: Path) -> None:
    boot_a = JournalWriter(JournalLayout.at(tmp_path / "journal"), boot_id="A")
    boot_a.append(JournalKind.JOB_INTENT, payload={})
    boot_a.append(JournalKind.JOB_INTENT, payload={})
    boot_a.commit()
    boot_a.close()

    # New writer must continue at seq=2, not restart at 0.
    boot_b = JournalWriter(JournalLayout.at(tmp_path / "journal"), boot_id="B")
    assert boot_b.next_seq == 2
    boot_b.append(JournalKind.JOB_INTENT, payload={})
    boot_b.commit()
    boot_b.close()

    reader = JournalReader(JournalLayout.at(tmp_path / "journal"))
    records = _records(reader)
    seqs = [r.seq for r in records]
    boot_ids = {r.mvd_boot_id for r in records}
    assert seqs == [0, 1, 2]
    assert boot_ids == {"A", "B"}


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.append(JournalKind.JOB_INTENT)
    writer.commit()
    writer.write_checkpoint(last_seq=0)
    writer.close()

    reader = JournalReader(JournalLayout.at(tmp_path / "journal"))
    assert reader.read_checkpoint() == 0


# ---------------------------------------------------------------------------
# 8. Threshold is exactly the strategy value
# ---------------------------------------------------------------------------


def test_inline_blob_threshold_matches_strategy() -> None:
    assert INLINE_BLOB_SPILL_THRESHOLD == 256 * 1024
