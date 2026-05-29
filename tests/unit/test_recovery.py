import os
import sqlite3
from types import SimpleNamespace

from multiverse import registry_db


def _init_db(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT NOT NULL, output_path TEXT, container_id TEXT, failure_reason TEXT)")
    conn.commit()
    conn.close()


def test_promoting_no_marker_marks_failed(tmp_path):
    db_path = str(tmp_path / "registry.db")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO runs (status, output_path) VALUES (?, ?)", ("PROMOTING", str(artifact)))
    conn.commit(); conn.close()

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
    assert status == "FAILED"
    assert reason == "INCOMPLETE_PROMOTION"
    assert not artifact.exists()


def test_running_dead_container_marks_orphaned(tmp_path):
    db_path = str(tmp_path / "registry.db")
    _init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO runs (status, output_path, container_id) VALUES (?, ?, ?)", ("RUNNING", "/tmp/work", "missing"))
    conn.commit(); conn.close()

    class Containers:
        def get(self, container_id):
            raise RuntimeError("missing")

    old = registry_db.DB_NAME
    registry_db.DB_NAME = db_path
    try:
        healed = registry_db.recover_orphaned_runs(docker_client=SimpleNamespace(containers=Containers()))
    finally:
        registry_db.DB_NAME = old

    conn = sqlite3.connect(db_path)
    status, reason = conn.execute("SELECT status, failure_reason FROM runs").fetchone()
    conn.close()
    assert healed == 1
    assert (status, reason) == ("FAILED", "ORPHANED")


def test_running_live_container_invokes_reattach_callback(tmp_path):
    db_path = str(tmp_path / "registry.db")
    _init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO runs (status, output_path, container_id) VALUES (?, ?, ?)", ("RUNNING", "/tmp/work", "abc"))
    conn.commit(); conn.close()

    container = SimpleNamespace(status="running", reload=lambda: None)
    class Containers:
        def get(self, container_id):
            return container

    calls = []
    old = registry_db.DB_NAME
    registry_db.DB_NAME = db_path
    try:
        healed = registry_db.recover_orphaned_runs(
            docker_client=SimpleNamespace(containers=Containers()),
            reattach_callback=lambda *args: calls.append(args),
        )
    finally:
        registry_db.DB_NAME = old

    assert healed == 1
    assert calls and calls[0][0] is container
