"""Doctor probe: projection ↔ journal consistency (STRATEGY M5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from multiverse.doctor import probe_projection_consistency
from multiverse.doctor.health_probes import ProbeOutcome
from multiverse.index import INDEX_FILENAME, open_index
from multiverse.journal import JournalKind, JournalLayout, JournalWriter

pytestmark = pytest.mark.control_plane


def _write_journal(state_root: Path, records: list[tuple]) -> None:
    layout = JournalLayout.at(state_root / "journal").ensure()
    writer = JournalWriter(layout, boot_id="boot-test")
    for kind, attempt, payload in records:
        writer.append(kind, payload=payload, physical_attempt_id=attempt)
    writer.commit()
    writer.close()


def test_probe_skipped_with_no_journal(tmp_path: Path) -> None:
    report = probe_projection_consistency(tmp_path)
    assert report.probe is ProbeOutcome.SKIPPED


def test_probe_passes_when_in_sync(tmp_path: Path) -> None:
    _write_journal(
        tmp_path,
        [
            (JournalKind.JOB_INTENT, "r1", {"manifest_path": "/m"}),
            (
                JournalKind.STATE_TRANSITION,
                "r1",
                {"from_state": "PENDING", "to_state": "ARTIFACT_SUCCESS"},
            ),
        ],
    )
    with open_index(tmp_path / INDEX_FILENAME) as idx:
        idx.upsert_run(
            {
                "physical_attempt_id": "r1",
                "primary_state": "ARTIFACT_SUCCESS",
                "options": {},
            }
        )
    report = probe_projection_consistency(tmp_path)
    assert report.probe is ProbeOutcome.PASS
    assert "runs_in_journal=1" in (report.detail or "")
    assert "runs_in_projection=1" in (report.detail or "")


def test_probe_fails_when_projection_missing_a_run(tmp_path: Path) -> None:
    _write_journal(
        tmp_path,
        [
            (JournalKind.JOB_INTENT, "r-missing", {"manifest_path": "/m"}),
        ],
    )
    report = probe_projection_consistency(tmp_path)
    assert report.probe is ProbeOutcome.FAIL
    assert report.leak_count >= 1
    assert "rebuild-index" in (report.detail or "")


def test_probe_fails_on_stale_state(tmp_path: Path) -> None:
    _write_journal(
        tmp_path,
        [
            (JournalKind.JOB_INTENT, "r1", {"manifest_path": "/m"}),
            (
                JournalKind.STATE_TRANSITION,
                "r1",
                {"from_state": "PENDING", "to_state": "FAILED"},
            ),
        ],
    )
    with open_index(tmp_path / INDEX_FILENAME) as idx:
        idx.upsert_run(
            {
                "physical_attempt_id": "r1",
                "primary_state": "RUNNING",
                "options": {},
            }
        )
    report = probe_projection_consistency(tmp_path)
    assert report.probe is ProbeOutcome.FAIL
    assert "stale_state" in (report.detail or "")


def test_probe_truncates_long_drift_list(tmp_path: Path) -> None:
    """Detail line should sample up to 3 attempts so it stays readable
    even when the projection has wildly drifted."""
    records = []
    for i in range(7):
        records.append((JournalKind.JOB_INTENT, f"r{i:02d}", {"manifest_path": "/m"}))
    _write_journal(tmp_path, records)
    report = probe_projection_consistency(tmp_path)
    assert report.probe is ProbeOutcome.FAIL
    assert report.leak_count == 7
    assert "+4 more" in (report.detail or "")
