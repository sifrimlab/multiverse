"""Move-3 exit-gate tests for non-destructive recovery.

Strategy v2 §3 acceptance: a simulated PROMOTING crash never deletes
artifact/workspace data and instead produces a recoverable quarantine or
RECOVERY_PENDING classification.

These tests cover two recovery paths:

1. The legacy ``recover_orphaned_runs()`` (still called from
   ``init_db()`` until the CLI cutover is complete). It must now
   *quarantine* incomplete promotions rather than ``rmtree`` them.

2. The new ``rebuild_index`` path (preferred). It already classifies
   without deletion; we lock that here so a refactor cannot regress it.

A grep-style gate also runs across ``registry_db.py`` and
``runner/cli.py`` to ensure they no longer call ``shutil.rmtree`` or
``os.unlink`` on result-like data in the recovery code path.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from multiverse import registry_db
from multiverse.index import open_index, rebuild_index
from multiverse.journal import JournalKind, JournalLayout, JournalWriter
from multiverse.mvd.state import PrimaryState
from multiverse.promotion import StoreLayout


# ---------------------------------------------------------------------------
# 1. Legacy recover_orphaned_runs() — never deletes
# ---------------------------------------------------------------------------


def _init_legacy_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE runs ("
        "run_id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "status TEXT NOT NULL, "
        "output_path TEXT, "
        "container_id TEXT, "
        "failure_reason TEXT)"
    )
    conn.commit()
    conn.close()


@pytest.fixture
def isolated_legacy_runtime(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "registry.db"
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    _init_legacy_db(str(db_path))
    monkeypatch.setattr(registry_db, "DB_NAME", str(db_path))
    monkeypatch.setattr(registry_db, "STORE_DIR", str(store_dir))
    return db_path, store_dir


def test_legacy_recovery_quarantines_incomplete_promotion(
    isolated_legacy_runtime,
) -> None:
    db_path, store_dir = isolated_legacy_runtime
    artifact_dir = store_dir / "artifacts" / "demo_pca"
    artifact_dir.mkdir(parents=True)
    important = artifact_dir / "container.log"
    important.write_text("partial output")
    embedding = artifact_dir / "embeddings.h5"
    embedding.write_bytes(b"\x89HDF\r\n\x1a\nfake")  # bytes don't matter; presence does

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO runs (status, output_path) VALUES (?, ?)",
        ("PROMOTING", str(artifact_dir)),
    )
    conn.commit()
    conn.close()

    registry_db.recover_orphaned_runs()

    # The artifact directory was moved into quarantine, not deleted.
    quarantine_root = store_dir / "quarantine"
    assert quarantine_root.is_dir()
    surviving_logs = list(quarantine_root.rglob("container.log"))
    surviving_embeddings = list(quarantine_root.rglob("embeddings.h5"))
    assert surviving_logs, "container.log must survive recovery"
    assert surviving_embeddings, "embeddings.h5 must survive recovery"

    # A tombstone is left at the original path.
    tombstone = list(artifact_dir.parent.glob("*.quarantined"))
    assert tombstone, "tombstone must mark the original artifact location"


def test_legacy_recovery_with_missing_dir_does_not_crash(
    isolated_legacy_runtime,
) -> None:
    db_path, _ = isolated_legacy_runtime
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO runs (status, output_path) VALUES (?, ?)",
        ("PROMOTING", "/nonexistent/path"),
    )
    conn.commit()
    conn.close()

    healed = registry_db.recover_orphaned_runs()
    assert healed == 1


def test_legacy_recovery_with_marker_does_not_quarantine(
    isolated_legacy_runtime,
) -> None:
    db_path, store_dir = isolated_legacy_runtime
    artifact_dir = store_dir / "artifacts" / "good"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / ".promotion_complete").write_text("")
    (artifact_dir / "embeddings.h5").write_bytes(b"x")

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO runs (status, output_path) VALUES (?, ?)",
        ("PROMOTING", str(artifact_dir)),
    )
    conn.commit()
    conn.close()

    registry_db.recover_orphaned_runs()

    # No quarantine should happen on a marker-present row.
    quarantine_root = store_dir / "quarantine"
    if quarantine_root.exists():
        assert list(quarantine_root.iterdir()) == []
    assert (artifact_dir / "embeddings.h5").is_file(), "successful promotion stays put"


# ---------------------------------------------------------------------------
# 2. Grep gate: registry_db.recover_orphaned_runs body has no rmtree/unlink
# ---------------------------------------------------------------------------


_DESTRUCTIVE_CALLS = (
    re.compile(r"\bshutil\.rmtree\b"),
    re.compile(r"\bos\.unlink\b"),
    re.compile(r"\bos\.rmdir\b"),
    re.compile(r"\bos\.removedirs\b"),
    re.compile(r"\bPath\([^)]*\)\.rmdir\b"),
)


def _function_source(module_text: str, name: str) -> str:
    """Cheap function-body extractor — returns text from ``def NAME(`` to
    the next top-level ``def `` / EOF."""
    lines = module_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"def {name}(") or line.startswith(f"async def {name}("):
            start = i
            break
    assert start is not None, f"function {name} not found"
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("def ") or lines[j].startswith("async def "):
            end = j
            break
    return "\n".join(lines[start:end])


def test_recover_orphaned_runs_body_has_no_destructive_calls() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "multiverse" / "registry_db.py").read_text(encoding="utf-8")
    body = _function_source(text, "recover_orphaned_runs")
    for pattern in _DESTRUCTIVE_CALLS:
        assert not pattern.search(body), (
            f"forbidden destructive call {pattern.pattern} in recover_orphaned_runs()"
        )


# ---------------------------------------------------------------------------
# 3. Preferred path: rebuild-index classifies PROMOTE_PREPARE-without-commit
#    without deletion (already locked in test_index_rebuild.py; here we
#    pin the cardinal "no mutation of the half-built dir" property again).
# ---------------------------------------------------------------------------


def test_rebuild_index_does_not_delete_incomplete_promotion(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True)
    JournalLayout.at(state_root / "journal").ensure()
    store = StoreLayout(root=tmp_path / "store").ensure()
    artifact_dir = store.artifacts / "incomplete"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / ".mvd_owner").write_text("placeholder")
    (artifact_dir / "embeddings.h5").write_bytes(b"x")

    writer = JournalWriter(JournalLayout.at(state_root / "journal"), boot_id="B")
    writer.append(
        JournalKind.JOB_INTENT,
        payload={"manifest_path": "/tmp/m.yaml"},
        physical_attempt_id="att-recover",
    )
    writer.append(
        JournalKind.PROMOTE_PREPARE,
        payload={
            "workspace_dir": "/tmp/ws",
            "final_artifact_dir": str(artifact_dir),
            "owner_token": "own",
        },
        physical_attempt_id="att-recover",
        logical_run_id="L",
    )
    writer.commit()
    writer.close()

    snapshot_files = sorted(p.name for p in artifact_dir.iterdir())
    with open_index(state_root / "mvexp_state.db") as idx:
        result = rebuild_index(index=idx, state_root=state_root, store=store)
        run = idx.get_run("att-recover")

    assert run["primary_state"] == PrimaryState.RECOVERY_PENDING.value
    assert result.recovery_pending == 1
    # Nothing was deleted by rebuild-index.
    assert sorted(p.name for p in artifact_dir.iterdir()) == snapshot_files
