import sqlite3

from multiverse import registry_db


def _init_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE runs ("
        "run_id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT NOT NULL, "
        "output_path TEXT, container_id TEXT, failure_reason TEXT)"
    )
    conn.commit()
    conn.close()


def test_sigterm_mid_promote_recovers(tmp_path):
    db_path = str(tmp_path / "registry.db")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / ".promotion_complete").write_text("", encoding="utf-8")
    _init_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO runs (status, output_path) VALUES (?, ?)",
        ("PROMOTING", str(artifact)),
    )
    conn.commit()
    conn.close()

    old = registry_db.DB_NAME
    registry_db.DB_NAME = db_path
    try:
        healed = registry_db.recover_orphaned_runs()
    finally:
        registry_db.DB_NAME = old

    conn = sqlite3.connect(db_path)
    status, reason = conn.execute("SELECT status, failure_reason FROM runs").fetchone()
    conn.close()
    assert healed == 1
    assert status == "SUCCESS"
    assert reason is None


def test_sigterm_drains_queue_direct_fallback(tmp_path):
    from multiverse import registry_db as rdb
    from multiverse.runner.docker_runner import mark_active_runs_failed_direct

    db_path = str(tmp_path / "registry.db")
    _init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO runs (status, output_path) VALUES (?, ?)", ("RUNNING", "/tmp/work")
    )
    conn.commit()
    conn.close()

    old = rdb.DB_NAME
    rdb.DB_NAME = db_path
    try:
        marked = mark_active_runs_failed_direct("CANCELLED")
    finally:
        rdb.DB_NAME = old

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT status, failure_reason FROM runs").fetchone()
    conn.close()
    assert marked == 1
    assert row == ("FAILED", "CANCELLED")
