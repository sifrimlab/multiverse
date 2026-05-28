import os
import sqlite3
from types import SimpleNamespace

from multiverse import registry_db


def _init_db(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT NOT NULL, output_path TEXT, container_id TEXT, failure_reason TEXT)")
    conn.commit()
    conn.close()


def test_promoting_no_marker_quarantines_without_deletion(tmp_path):
    """STRATEGY v2 §3: recovery must classify, never delete result-like
    data. An incomplete promotion is moved into ``store/quarantine/`` and
    a tombstone is left at the original path."""
    db_path = str(tmp_path / "registry.db")
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    artifact = store_dir / "artifacts" / "demo"
    artifact.mkdir(parents=True)
    (artifact / "container.log").write_text("partial output")
    _init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO runs (status, output_path) VALUES (?, ?)",
        ("PROMOTING", str(artifact)),
    )
    conn.commit()
    conn.close()

    old_db = registry_db.DB_NAME
    old_store = registry_db.STORE_DIR
    registry_db.DB_NAME = db_path
    registry_db.STORE_DIR = str(store_dir)
    try:
        healed = registry_db.recover_orphaned_runs()
    finally:
        registry_db.DB_NAME = old_db
        registry_db.STORE_DIR = old_store

    conn = sqlite3.connect(db_path)
    status, reason = conn.execute("SELECT status, failure_reason FROM runs").fetchone()
    conn.close()
    assert healed == 1
    assert status == "FAILED"
    assert reason == "INCOMPLETE_PROMOTION"
    # Recovery never deletes. The artifact directory was moved into
    # quarantine and a tombstone marks the original path.
    assert not artifact.exists(), "original path no longer hosts the dir"
    quarantine_dir = store_dir / "quarantine"
    assert quarantine_dir.is_dir()
    moved_files = list(quarantine_dir.rglob("container.log"))
    assert moved_files, "the partial workspace must be preserved in quarantine"
    tombstone = list(artifact.parent.glob("*.quarantined"))
    assert tombstone, ".quarantined tombstone must be left at the original location"


def test_running_dead_container_without_injected_client_uses_default_client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "registry.db")
    _init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO runs (status, output_path, container_id) VALUES (?, ?, ?)",
        ("RUNNING", "/tmp/work", "missing"),
    )
    conn.commit(); conn.close()

    class Containers:
        def get(self, container_id):
            raise RuntimeError("missing")

    monkeypatch.setattr(
        registry_db,
        "_build_recovery_docker_client",
        lambda: SimpleNamespace(containers=Containers()),
    )
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
    assert (status, reason) == ("FAILED", "ORPHANED")


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
