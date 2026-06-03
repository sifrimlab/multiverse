"""Tests for the read-mostly projection facade (STRATEGY M5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from multiverse.index import INDEX_FILENAME, open_index
from multiverse.index_projection import (ProjectionDrift,
                                         ProjectionVerifyReport, get_run,
                                         list_runs, projections_for,
                                         verify_projection_against_journal)
from multiverse.journal import JournalKind, JournalLayout, JournalWriter

pytestmark = pytest.mark.control_plane


# ---------------------------------------------------------------------------
# Read facade
# ---------------------------------------------------------------------------


def test_get_run_returns_none_when_projection_missing(tmp_path: Path) -> None:
    assert get_run(tmp_path, physical_attempt_id="x") is None
    assert list_runs(tmp_path) == []
    assert projections_for(tmp_path, physical_attempt_id="x") == {}


def test_get_run_round_trips_through_projection(tmp_path: Path) -> None:
    db_path = tmp_path / INDEX_FILENAME
    with open_index(db_path) as idx:
        idx.upsert_run(
            {
                "physical_attempt_id": "r1",
                "logical_run_id": "lr1",
                "primary_state": "ARTIFACT_SUCCESS",
                "failure_reason": None,
                "artifact_dir": "/store/a",
                "workspace_dir": "/store/w",
                "manifest_path": "/m.yaml",
                "cancel_requested": False,
                "submitted_wall_iso": "2026-01-01T00:00:00+00:00",
                "last_seq": 12,
                "options": {"k": "v"},
            }
        )

    row = get_run(tmp_path, physical_attempt_id="r1")
    assert row is not None
    assert row["primary_state"] == "ARTIFACT_SUCCESS"
    assert row["logical_run_id"] == "lr1"
    rows = list_runs(tmp_path, primary_state="ARTIFACT_SUCCESS")
    assert [r["physical_attempt_id"] for r in rows] == ["r1"]
    assert list_runs(tmp_path, primary_state="FAILED") == []


def test_projections_for_returns_status_map(tmp_path: Path) -> None:
    db_path = tmp_path / INDEX_FILENAME
    with open_index(db_path) as idx:
        idx.upsert_run(
            {
                "physical_attempt_id": "r1",
                "logical_run_id": "lr1",
                "primary_state": "ARTIFACT_SUCCESS",
                "options": {},
            }
        )
        idx.set_projection(
            physical_attempt_id="r1",
            plugin="mlflow",
            status="TRACKING_PENDING",
        )

    assert projections_for(tmp_path, physical_attempt_id="r1") == {
        "mlflow": "TRACKING_PENDING"
    }


# ---------------------------------------------------------------------------
# verify_projection_against_journal
# ---------------------------------------------------------------------------


def _write_journal(state_root: Path, records: list[tuple]) -> None:
    """Each record is ``(kind, attempt_id, payload)``."""
    layout = JournalLayout.at(state_root / "journal").ensure()
    writer = JournalWriter(layout, boot_id="boot-test")
    for kind, attempt, payload in records:
        writer.append(kind, payload=payload, physical_attempt_id=attempt)
    writer.commit()
    writer.close()


def test_verify_no_journal_returns_empty_report(tmp_path: Path) -> None:
    report = verify_projection_against_journal(tmp_path)
    assert isinstance(report, ProjectionVerifyReport)
    assert report.in_sync
    assert report.runs_in_journal == 0
    assert report.runs_in_projection == 0


def test_verify_in_sync_when_both_agree(tmp_path: Path) -> None:
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
                "logical_run_id": None,
                "primary_state": "ARTIFACT_SUCCESS",
                "options": {},
            }
        )

    report = verify_projection_against_journal(tmp_path)
    assert report.in_sync
    assert report.runs_in_journal == 1
    assert report.runs_in_projection == 1


def test_verify_missing_in_projection(tmp_path: Path) -> None:
    _write_journal(
        tmp_path,
        [(JournalKind.JOB_INTENT, "r-missing", {"manifest_path": "/m"})],
    )
    # Projection file does not exist yet.
    report = verify_projection_against_journal(tmp_path)
    kinds = {d.kind for d in report.drifts}
    assert "missing_in_projection" in kinds
    assert report.drifts[0].physical_attempt_id == "r-missing"


def test_verify_stale_state(tmp_path: Path) -> None:
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
                "primary_state": "RUNNING",  # stale
                "options": {},
            }
        )
    report = verify_projection_against_journal(tmp_path)
    assert not report.in_sync
    drift = report.drifts[0]
    assert drift.kind == "stale_state"
    assert drift.journal_state == "FAILED"
    assert drift.projection_state == "RUNNING"


def test_verify_orphan_in_projection(tmp_path: Path) -> None:
    # Journal has no records at all for r-orphan.
    layout = JournalLayout.at(tmp_path / "journal").ensure()
    writer = JournalWriter(layout, boot_id="boot-test")
    writer.append(
        JournalKind.JOB_INTENT,
        payload={"manifest_path": "/m"},
        physical_attempt_id="r-real",
    )
    writer.commit()
    writer.close()
    with open_index(tmp_path / INDEX_FILENAME) as idx:
        idx.upsert_run(
            {
                "physical_attempt_id": "r-real",
                "primary_state": "PENDING",
                "options": {},
            }
        )
        idx.upsert_run(
            {
                "physical_attempt_id": "r-orphan",
                "primary_state": "FAILED",
                "options": {},
            }
        )
    report = verify_projection_against_journal(tmp_path)
    drift_kinds = {d.kind for d in report.drifts}
    assert "orphan_in_projection" in drift_kinds
    orphan = next(d for d in report.drifts if d.kind == "orphan_in_projection")
    assert orphan.physical_attempt_id == "r-orphan"


def test_delete_run_removes_row_and_cascade(tmp_path: Path) -> None:
    """delete_run must remove the run row and cascade to child tables."""
    db_path = tmp_path / INDEX_FILENAME
    with open_index(db_path) as idx:
        idx.upsert_run(
            {
                "physical_attempt_id": "r-del",
                "logical_run_id": "lr-del",
                "primary_state": "ARTIFACT_SUCCESS",
                "failure_reason": None,
                "artifact_dir": "/store/a",
                "workspace_dir": "/store/w",
                "manifest_path": "/m.yaml",
                "cancel_requested": False,
                "submitted_wall_iso": "2026-01-01T00:00:00+00:00",
                "last_seq": 1,
                "options": {},
            }
        )
        idx.set_projection(physical_attempt_id="r-del", plugin="mlflow", status="ok")
        assert idx.delete_run("r-del") is True

    # Row is gone from the index.
    with open_index(db_path) as idx:
        assert idx.get_run("r-del") is None
        # Cascade removes the projection row.
        rows = idx.conn.execute(
            "SELECT 1 FROM run_projections WHERE physical_attempt_id = 'r-del'"
        ).fetchall()
        assert rows == []


def test_delete_run_returns_false_for_missing(tmp_path: Path) -> None:
    db_path = tmp_path / INDEX_FILENAME
    with open_index(db_path) as idx:
        assert idx.delete_run("nonexistent") is False


def test_drift_to_dict_round_trips() -> None:
    d = ProjectionDrift(
        physical_attempt_id="r1",
        kind="stale_state",
        journal_state="FAILED",
        projection_state="RUNNING",
        detail="x",
    )
    out = d.to_dict()
    assert out["kind"] == "stale_state"
    assert out["journal_state"] == "FAILED"
    assert out["projection_state"] == "RUNNING"
