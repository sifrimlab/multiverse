"""``multiverse rebuild-index --verify`` CLI (STRATEGY M5).

The --verify flag is a read-only consistency check: it reports drift
between the journal and the projection, and exits non-zero if any
drift is found. Doctor uses the same code path but surfaces the result
in human/JSON form alongside the other probes.
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pytest

from multiverse.cli_entrypoints import rebuild_index_main
from multiverse.index import INDEX_FILENAME, open_index
from multiverse.journal import (
    JournalKind,
    JournalLayout,
    JournalWriter,
)


pytestmark = pytest.mark.control_plane


def _journal(state_root: Path, *records) -> None:
    layout = JournalLayout.at(state_root / "journal").ensure()
    writer = JournalWriter(layout, boot_id="boot-test")
    for kind, attempt, payload in records:
        writer.append(kind, payload=payload, physical_attempt_id=attempt)
    writer.commit()
    writer.close()


def _capture(state_root: Path, *args: str) -> tuple[int, str]:
    buf = StringIO()
    saved_stdout = sys.stdout
    sys.stdout = buf
    try:
        rc = rebuild_index_main(
            ["--state-root", str(state_root), "--verify", *args]
        )
    finally:
        sys.stdout = saved_stdout
    return rc, buf.getvalue()


def test_verify_returns_zero_when_in_sync(tmp_path: Path) -> None:
    _journal(
        tmp_path,
        (JournalKind.JOB_INTENT, "r1", {"manifest_path": "/m"}),
        (
            JournalKind.STATE_TRANSITION,
            "r1",
            {"from_state": "PENDING", "to_state": "ARTIFACT_SUCCESS"},
        ),
    )
    with open_index(tmp_path / INDEX_FILENAME) as idx:
        idx.upsert_run(
            {
                "physical_attempt_id": "r1",
                "primary_state": "ARTIFACT_SUCCESS",
                "options": {},
            }
        )
    rc, stdout = _capture(tmp_path)
    assert rc == 0, stdout
    payload = json.loads(stdout)
    assert payload["drift_count"] == 0


def test_verify_returns_nonzero_on_drift(tmp_path: Path) -> None:
    _journal(
        tmp_path,
        (JournalKind.JOB_INTENT, "r-missing", {"manifest_path": "/m"}),
    )
    rc, stdout = _capture(tmp_path)
    assert rc == 1, stdout
    payload = json.loads(stdout)
    assert payload["drift_count"] >= 1
    assert any(
        d["physical_attempt_id"] == "r-missing" for d in payload["drifts"]
    )


def test_verify_does_not_write_to_index(tmp_path: Path) -> None:
    """--verify is read-only: a non-existent projection must stay
    non-existent after the command runs."""
    _journal(
        tmp_path,
        (JournalKind.JOB_INTENT, "r1", {"manifest_path": "/m"}),
    )
    assert not (tmp_path / INDEX_FILENAME).exists()
    rc, _ = _capture(tmp_path)
    assert rc == 1
    assert not (tmp_path / INDEX_FILENAME).exists()
