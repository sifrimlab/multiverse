"""Milestone-8 exit-gate tests for the SQLite index + rebuilder.

Coverage:
    1. ``open_index`` creates the schema and is idempotent.
    2. ``upsert_run`` round-trip; ``list_runs`` filters by state.
    3. Three-source merge — happy path: journal-replayed promotion + a
       sidecar-verified artifact dir → ARTIFACT_SUCCESS.
    4. Promote-prepare without commit → RECOVERY_PENDING (S4).
    5. Corrupt manifest (sidecar mismatch) → RECOVERY_PENDING and the
       rebuilder does NOT delete or move the corrupt dir.
    6. RUNNING in journal with no container → RECOVERY_PENDING (disappeared).
    7. RUNNING in journal with live container → reattached.
    8. Deleting the index db and rebuilding restores promoted runs (S2
       acceptance).
    9. Truncated journal tail is reported but does not abort the rebuild.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multiverse.artifact import (ARTIFACT_MANIFEST_FILENAME, ArtifactManifest,
                                 BootContext, ImageIdentity, ProducedAt,
                                 ProducedBy, compute_manifest_hash,
                                 compute_params_hash, produced_at_now,
                                 write_manifest)
from multiverse.docker_supervisor import InMemoryContainerEngine
from multiverse.index import (INDEX_FILENAME, SCHEMA_VERSION, RebuildOutcome,
                              open_index, rebuild_index)
from multiverse.journal import JournalKind, JournalLayout, JournalWriter
from multiverse.mvd.state import PrimaryState
from multiverse.promotion import StoreLayout

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def state_root(tmp_path: Path) -> Path:
    root = tmp_path / "state"
    root.mkdir(parents=True, exist_ok=True)
    JournalLayout.at(root / "journal").ensure()
    return root


@pytest.fixture
def store(tmp_path: Path) -> StoreLayout:
    return StoreLayout(root=tmp_path / "store").ensure()


@pytest.fixture
def index_path(state_root: Path) -> Path:
    return state_root / INDEX_FILENAME


def _journal(state_root: Path, boot_id: str = "B") -> JournalWriter:
    return JournalWriter(JournalLayout.at(state_root / "journal"), boot_id=boot_id)


def _make_artifact_dir(
    store: StoreLayout,
    name: str,
    *,
    logical: str,
    attempt: str,
    boot: BootContext,
) -> Path:
    image = ImageIdentity.registry_digest("sha256:" + "a" * 64)
    manifest_hash = compute_manifest_hash("jobs: []\n")
    params_hash = compute_params_hash({"x": 1})
    manifest = ArtifactManifest(
        logical_run_id=logical,
        physical_attempt_id=attempt,
        manifest_hash=manifest_hash,
        dataset_fingerprint={"slug": "demo", "n_obs": 4},
        image_identity=image,
        params_hash=params_hash,
        mv_contract_version="1",
        produced_at=ProducedAt.from_dict(produced_at_now(boot)),
        produced_by=ProducedBy(mvd_version="0.1.0-test"),
        artifacts=[],
        owner_token="owner-test",
    )
    target = store.artifacts / name
    write_manifest(target, manifest)
    return target


# ---------------------------------------------------------------------------
# 1. Schema and open_index
# ---------------------------------------------------------------------------


def test_open_index_creates_schema_and_is_idempotent(index_path: Path) -> None:
    with open_index(index_path) as idx:
        cur = idx.conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        )
        assert cur.fetchone()[0] == SCHEMA_VERSION
    # Reopening is fine.
    with open_index(index_path):
        pass


def test_open_index_refuses_mismatched_schema(index_path: Path) -> None:
    with open_index(index_path) as idx:
        idx.conn.execute(
            "UPDATE schema_meta SET value=? WHERE key='schema_version'",
            ("999",),
        )
        idx.conn.commit()
    with pytest.raises(RuntimeError):
        open_index(index_path)


# ---------------------------------------------------------------------------
# 2. upsert / list
# ---------------------------------------------------------------------------


def test_upsert_and_list_round_trip(index_path: Path) -> None:
    with open_index(index_path) as idx:
        idx.upsert_run(
            {
                "physical_attempt_id": "att-1",
                "logical_run_id": "L",
                "primary_state": PrimaryState.ARTIFACT_SUCCESS.value,
                "submitted_wall_iso": "2026-01-01T00:00:00+00:00",
                "last_seq": 10,
            }
        )
        idx.upsert_run(
            {
                "physical_attempt_id": "att-2",
                "logical_run_id": "L",
                "primary_state": PrimaryState.FAILED.value,
                "submitted_wall_iso": "2026-01-02T00:00:00+00:00",
                "last_seq": 20,
            }
        )
        assert idx.get_run("att-1")["primary_state"] == "ARTIFACT_SUCCESS"
        runs = idx.list_runs(primary_state="ARTIFACT_SUCCESS")
        assert [r["physical_attempt_id"] for r in runs] == ["att-1"]


def test_projection_round_trip(index_path: Path) -> None:
    with open_index(index_path) as idx:
        idx.upsert_run(
            {
                "physical_attempt_id": "a",
                "logical_run_id": "L",
                "primary_state": PrimaryState.ARTIFACT_SUCCESS.value,
            }
        )
        idx.set_projection(
            physical_attempt_id="a",
            plugin="mlflow",
            status="TRACKING_SYNC_FAILED",
            details={"error": "denied"},
        )
        assert idx.projections_for("a") == {"mlflow": "TRACKING_SYNC_FAILED"}


# ---------------------------------------------------------------------------
# 3. Happy-path rebuild: journal + verified manifest → ARTIFACT_SUCCESS
# ---------------------------------------------------------------------------


def _seed_promotion_journal(
    state_root: Path,
    *,
    attempt: str,
    logical: str,
    artifact_dir: Path,
) -> None:
    writer = _journal(state_root)
    writer.append(
        JournalKind.JOB_INTENT,
        payload={"manifest_path": "/tmp/m.yaml"},
        physical_attempt_id=attempt,
    )
    writer.append(
        JournalKind.STATE_TRANSITION,
        payload={"from_state": "PENDING", "to_state": "RUNNING"},
        physical_attempt_id=attempt,
        logical_run_id=logical,
    )
    writer.append(
        JournalKind.PROMOTE_PREPARE,
        payload={
            "workspace_dir": "/tmp/ws",
            "final_artifact_dir": str(artifact_dir),
            "owner_token": "own",
        },
        physical_attempt_id=attempt,
        logical_run_id=logical,
    )
    writer.append(
        JournalKind.PROMOTE_COMMIT_MANIFEST,
        payload={"artifact_dir": str(artifact_dir), "manifest_sha256": "deadbeef"},
        physical_attempt_id=attempt,
        logical_run_id=logical,
    )
    writer.commit()
    writer.close()


def test_rebuild_happy_path_promotes(
    state_root: Path, store: StoreLayout, index_path: Path
) -> None:
    boot = BootContext.new(mvd_version="0.1.0-test")
    artifact_dir = _make_artifact_dir(
        store, "demo_pca", logical="LOG", attempt="att-success", boot=boot
    )
    _seed_promotion_journal(
        state_root, attempt="att-success", logical="LOG", artifact_dir=artifact_dir
    )
    with open_index(index_path) as idx:
        result = rebuild_index(index=idx, state_root=state_root, store=store)
        run = idx.get_run("att-success")
    assert result.total_runs == 1
    assert result.artifact_success == 1
    assert run["primary_state"] == "ARTIFACT_SUCCESS"
    assert run["artifact_dir"] == str(artifact_dir)
    # One classification, outcome PROMOTED.
    [classification] = result.classifications
    assert classification.outcome is RebuildOutcome.PROMOTED


# ---------------------------------------------------------------------------
# 4. Promote-prepare without commit → RECOVERY_PENDING
# ---------------------------------------------------------------------------


def test_rebuild_classifies_incomplete_promotion_as_recovery_pending(
    state_root: Path, store: StoreLayout, index_path: Path
) -> None:
    artifact_dir = store.artifacts / "incomplete"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / ".mvd_owner").write_text("placeholder")

    writer = _journal(state_root)
    writer.append(
        JournalKind.JOB_INTENT,
        payload={"manifest_path": "/tmp/m.yaml"},
        physical_attempt_id="att-inc",
    )
    writer.append(
        JournalKind.PROMOTE_PREPARE,
        payload={
            "workspace_dir": "/tmp/ws",
            "final_artifact_dir": str(artifact_dir),
            "owner_token": "own",
        },
        physical_attempt_id="att-inc",
        logical_run_id="L",
    )
    writer.commit()
    writer.close()

    with open_index(index_path) as idx:
        result = rebuild_index(index=idx, state_root=state_root, store=store)
        run = idx.get_run("att-inc")
    assert run["primary_state"] == PrimaryState.RECOVERY_PENDING.value
    assert result.recovery_pending == 1
    # And the rebuilder did NOT touch the artifact dir on disk.
    assert artifact_dir.is_dir()
    assert (artifact_dir / ".mvd_owner").is_file()


# ---------------------------------------------------------------------------
# 5. Corrupt manifest → RECOVERY_PENDING, no mutation
# ---------------------------------------------------------------------------


def test_rebuild_corrupt_manifest_classifies_recovery_pending_without_mutation(
    state_root: Path, store: StoreLayout, index_path: Path
) -> None:
    boot = BootContext.new(mvd_version="0.1.0-test")
    artifact_dir = _make_artifact_dir(
        store, "corrupt_one", logical="L", attempt="att-corrupt", boot=boot
    )
    # Tamper with the manifest body — sidecar no longer matches.
    body_path = artifact_dir / ARTIFACT_MANIFEST_FILENAME
    body_path.write_text(body_path.read_text() + "\n# tampered\n")
    snapshot_files = sorted(p.name for p in artifact_dir.iterdir())

    _seed_promotion_journal(
        state_root, attempt="att-corrupt", logical="L", artifact_dir=artifact_dir
    )
    with open_index(index_path) as idx:
        rebuild_index(index=idx, state_root=state_root, store=store)
        run = idx.get_run("att-corrupt")

    assert run["primary_state"] == PrimaryState.RECOVERY_PENDING.value
    # The rebuilder must NOT have renamed/moved/deleted anything.
    assert sorted(p.name for p in artifact_dir.iterdir()) == snapshot_files


# ---------------------------------------------------------------------------
# 6. RUNNING in journal with no live container → RECOVERY_PENDING
# ---------------------------------------------------------------------------


def test_rebuild_running_without_container_marks_recovery_pending(
    state_root: Path, store: StoreLayout, index_path: Path
) -> None:
    writer = _journal(state_root)
    writer.append(
        JournalKind.JOB_INTENT,
        payload={"manifest_path": "/tmp/m.yaml"},
        physical_attempt_id="att-gone",
    )
    writer.append(
        JournalKind.STATE_TRANSITION,
        payload={"from_state": "PENDING", "to_state": "RUNNING"},
        physical_attempt_id="att-gone",
        logical_run_id="L",
    )
    writer.commit()
    writer.close()

    engine = InMemoryContainerEngine()
    with open_index(index_path) as idx:
        rebuild_index(index=idx, state_root=state_root, store=store, engine=engine)
        run = idx.get_run("att-gone")
    assert run["primary_state"] == PrimaryState.RECOVERY_PENDING.value


def test_rebuild_running_with_live_container_reattaches(
    state_root: Path, store: StoreLayout, index_path: Path
) -> None:
    engine = InMemoryContainerEngine()
    # Pre-create a running container labelled for this attempt.
    info = engine.launch(
        image="x:1",
        labels={"multiverse.run_id": "att-alive"},
        env={},
        volumes={},
        mem_limit=None,
        name=None,
    )
    assert info  # ensure the fake produced one

    writer = _journal(state_root)
    writer.append(
        JournalKind.JOB_INTENT,
        payload={"manifest_path": "/tmp/m.yaml"},
        physical_attempt_id="att-alive",
    )
    writer.append(
        JournalKind.STATE_TRANSITION,
        payload={"from_state": "PENDING", "to_state": "RUNNING"},
        physical_attempt_id="att-alive",
        logical_run_id="L",
    )
    writer.commit()
    writer.close()

    with open_index(index_path) as idx:
        result = rebuild_index(
            index=idx, state_root=state_root, store=store, engine=engine
        )
        run = idx.get_run("att-alive")
    assert run["primary_state"] == PrimaryState.RUNNING.value
    [classification] = result.classifications
    assert classification.outcome is RebuildOutcome.RUNNING_REATTACHED


# ---------------------------------------------------------------------------
# 8. Deleting the index db and rebuilding restores promoted runs (S2)
# ---------------------------------------------------------------------------


def test_deleting_index_db_and_rebuilding_restores_promoted_runs(
    state_root: Path, store: StoreLayout, index_path: Path
) -> None:
    boot = BootContext.new(mvd_version="0.1.0-test")
    artifact_dir = _make_artifact_dir(
        store, "demo_full", logical="LFULL", attempt="att-1", boot=boot
    )
    _seed_promotion_journal(
        state_root, attempt="att-1", logical="LFULL", artifact_dir=artifact_dir
    )

    # First rebuild — record exists.
    with open_index(index_path) as idx:
        rebuild_index(index=idx, state_root=state_root, store=store)
        assert idx.get_run("att-1")["primary_state"] == "ARTIFACT_SUCCESS"

    # User deletes the db (the audit-window failure mode).
    index_path.unlink()

    # Second rebuild — record is restored exclusively from journal + artifact tree.
    with open_index(index_path) as idx2:
        result = rebuild_index(index=idx2, state_root=state_root, store=store)
        assert idx2.get_run("att-1")["primary_state"] == "ARTIFACT_SUCCESS"
    assert result.artifact_success == 1


# ---------------------------------------------------------------------------
# 9. Truncated journal tail tolerated
# ---------------------------------------------------------------------------


def test_rebuild_tolerates_truncated_journal_tail(
    state_root: Path, store: StoreLayout, index_path: Path
) -> None:
    boot = BootContext.new(mvd_version="0.1.0-test")
    artifact_dir = _make_artifact_dir(
        store, "demo_trunc", logical="LT", attempt="att-trunc", boot=boot
    )
    _seed_promotion_journal(
        state_root, attempt="att-trunc", logical="LT", artifact_dir=artifact_dir
    )
    # Append a partial line to simulate a crash mid-write.
    journal_path = state_root / "journal" / "current.log"
    with journal_path.open("ab") as fp:
        fp.write(b'{"seq":99,"kind":"PROMOTE_PREPARE",')  # no newline

    with open_index(index_path) as idx:
        result = rebuild_index(index=idx, state_root=state_root, store=store)
    assert result.truncated_journal_tail is not None
    # The completed promotion is still classified PROMOTED.
    assert result.artifact_success == 1
