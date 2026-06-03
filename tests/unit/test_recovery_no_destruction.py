"""``rebuild_index`` must never delete result-like data (STRATEGY v2 §3).

The legacy ``recover_orphaned_runs`` tests were removed in G6 along with
the docker_runner path they tested. This file retains the mvd-era
assertion: a PROMOTE_PREPARE-without-commit is classified as
RECOVERY_PENDING and the half-built artifact directory is left intact.
"""

from __future__ import annotations

from pathlib import Path

from multiverse.index import open_index, rebuild_index
from multiverse.journal import JournalKind, JournalLayout, JournalWriter
from multiverse.mvd.state import PrimaryState
from multiverse.promotion import StoreLayout

# ---------------------------------------------------------------------------
# Preferred path: rebuild-index classifies PROMOTE_PREPARE-without-commit
# without deletion.
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
