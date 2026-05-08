"""
Verification: zero SQLite 'database is locked' errors under 50 concurrent writers.

Tests the single-writer DB actor pattern:
  - db_writer_task holds the only write connection.
  - Worker coroutines enqueue _DbWriteOp items and await lastrowid via Future.
  - No concurrent direct writes ever happen; WAL reads are non-blocking.
"""

import asyncio
import os
import sqlite3
import pytest

import multiverse.runner.docker_runner as dr
from multiverse import registry_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_test_db(db_path: str) -> None:
    """Create a minimal runs table in a fresh SQLite file."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id    INTEGER,
            model_slug    TEXT,
            model_version TEXT,
            model_name    TEXT,
            status        TEXT NOT NULL,
            output_path   TEXT
        )
        """
    )
    conn.commit()
    conn.close()


async def _run_50_concurrent_writes(n: int = 50) -> list[int]:
    """Fire n concurrent _db_write() calls and collect all lastrowid values."""
    lock_errors: list[Exception] = []

    async def one_write(i: int) -> int:
        try:
            return await dr._db_write(
                "INSERT INTO runs "
                "(dataset_id, model_slug, model_version, model_name, status, output_path) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (i, f"model_{i}", "1.0.0", f"model_{i}", "SUCCESS", f"/artifacts/run_{i}"),
            )
        except sqlite3.OperationalError as exc:
            lock_errors.append(exc)
            raise

    row_ids = await asyncio.gather(*[one_write(i) for i in range(n)])
    assert not lock_errors, f"Unexpected lock errors: {lock_errors}"
    return list(row_ids)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_50_concurrent_writes_no_lock_errors(tmp_path):
    """50 concurrent INSERT coroutines must all succeed with no lock errors."""
    db_path = str(tmp_path / "test_registry.db")
    _init_test_db(db_path)

    # Redirect the module-level DB_NAME so get_db_connection() hits our temp file.
    original_db_name = registry_db.DB_NAME
    registry_db.DB_NAME = db_path
    try:
        dr.start_db_writer()
        try:
            row_ids = await _run_50_concurrent_writes(50)
        finally:
            await dr.stop_db_writer()
    finally:
        registry_db.DB_NAME = original_db_name

    # All 50 writes must have landed.
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    conn.close()

    assert len(row_ids) == 50, f"Expected 50 row IDs, got {len(row_ids)}"
    assert count == 50, f"Expected 50 rows in DB, found {count}"
    # All row IDs must be distinct (no collisions).
    assert len(set(row_ids)) == 50, "Duplicate lastrowid values — writes collided"


@pytest.mark.asyncio
async def test_db_write_returns_correct_lastrowid(tmp_path):
    """lastrowid returned by _db_write must match what SQLite assigned."""
    db_path = str(tmp_path / "test_registry.db")
    _init_test_db(db_path)

    original_db_name = registry_db.DB_NAME
    registry_db.DB_NAME = db_path
    try:
        dr.start_db_writer()
        rid1 = await dr._db_write(
            "INSERT INTO runs (model_slug, model_version, model_name, status) "
            "VALUES (?, ?, ?, ?)",
            ("pca", "1.0.0", "pca", "SUCCESS"),
        )
        rid2 = await dr._db_write(
            "INSERT INTO runs (model_slug, model_version, model_name, status) "
            "VALUES (?, ?, ?, ?)",
            ("mofa", "1.0.0", "mofa", "SUCCESS"),
        )
        await dr.stop_db_writer()
    finally:
        registry_db.DB_NAME = original_db_name

    assert rid2 == rid1 + 1, (
        f"Expected consecutive lastrowids ({rid1}, {rid1+1}), got ({rid1}, {rid2})"
    )


@pytest.mark.asyncio
async def test_db_write_raises_before_start():
    """_db_write must raise RuntimeError if called before start_db_writer()."""
    # Ensure queue is cleared (stop if previously started)
    if dr._db_write_queue is not None:
        await dr.stop_db_writer()

    with pytest.raises(RuntimeError, match="start_db_writer"):
        await dr._db_write("SELECT 1", ())


@pytest.mark.asyncio
async def test_3_phase_commit_leaves_success_in_db(tmp_path):
    """Simulate the 3-phase commit sequence and verify the DB ends at SUCCESS.

    Phase 1: INSERT with status=PROMOTING, capture run_id.
    Phase 2: filesystem 'promotion' (simulated by creating a directory).
    Phase 3: UPDATE to SUCCESS.
    """
    db_path = str(tmp_path / "test_registry.db")
    _init_test_db(db_path)
    artifact_dir = str(tmp_path / "artifact")
    os.makedirs(artifact_dir)

    original_db_name = registry_db.DB_NAME
    registry_db.DB_NAME = db_path
    try:
        dr.start_db_writer()

        # Phase 1
        run_id = await dr._db_write(
            "INSERT INTO runs "
            "(dataset_id, model_slug, model_version, model_name, status, output_path) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (1, "pca", "1.0.0", "pca", "PROMOTING", artifact_dir),
        )

        # Phase 2 — filesystem already present (simulated above)

        # Phase 3
        await dr._db_write(
            "UPDATE runs SET status=?, output_path=? WHERE run_id=?",
            ("SUCCESS", artifact_dir, run_id),
        )

        await dr.stop_db_writer()
    finally:
        registry_db.DB_NAME = original_db_name

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT status FROM runs WHERE run_id=?", (run_id,)
    ).fetchone()
    conn.close()

    assert row is not None, "Row not found after 3-phase commit"
    assert row[0] == "SUCCESS", f"Expected SUCCESS, got {row[0]}"


@pytest.mark.asyncio
async def test_recover_orphaned_runs_heals_promoting_rows(tmp_path):
    """recover_orphaned_runs() must reconcile PROMOTING rows against the filesystem."""
    db_path = str(tmp_path / "test_registry.db")
    _init_test_db(db_path)

    # Artifact dir that actually exists on disk (Phase 2 succeeded).
    existing_dir = str(tmp_path / "existing_artifact")
    os.makedirs(existing_dir)

    # Artifact dir that does NOT exist (Phase 2 never ran).
    missing_dir = str(tmp_path / "missing_artifact")

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO runs (model_slug, model_version, model_name, status, output_path) "
        "VALUES (?, ?, ?, ?, ?)",
        ("pca", "1.0.0", "pca", "PROMOTING", existing_dir),
    )
    conn.execute(
        "INSERT INTO runs (model_slug, model_version, model_name, status, output_path) "
        "VALUES (?, ?, ?, ?, ?)",
        ("mofa", "1.0.0", "mofa", "PROMOTING", missing_dir),
    )
    conn.commit()
    conn.close()

    original_db_name = registry_db.DB_NAME
    registry_db.DB_NAME = db_path
    try:
        healed = registry_db.recover_orphaned_runs()
    finally:
        registry_db.DB_NAME = original_db_name

    assert healed == 2, f"Expected 2 healed rows, got {healed}"

    conn = sqlite3.connect(db_path)
    rows = {
        r[0]: r[1]
        for r in conn.execute("SELECT output_path, status FROM runs").fetchall()
    }
    conn.close()

    assert rows[existing_dir] == "SUCCESS", (
        f"Row with existing artifact should be SUCCESS, got {rows[existing_dir]}"
    )
    assert rows[missing_dir] == "FAILED", (
        f"Row with missing artifact should be FAILED, got {rows[missing_dir]}"
    )


@pytest.mark.asyncio
async def test_stop_db_writer_is_idempotent():
    """Calling stop_db_writer() when no writer is running must not raise."""
    # Ensure clean state
    if dr._db_write_queue is not None:
        await dr.stop_db_writer()

    # Should be a no-op, not an error.
    await dr.stop_db_writer()
    await dr.stop_db_writer()
