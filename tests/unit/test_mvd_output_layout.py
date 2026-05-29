from pathlib import Path

from multiverse.runner.mvd_entrypoint import _state_root_for_output, _store_for_output


def test_output_dir_is_promoted_artifact_root(tmp_path: Path) -> None:
    output_root = tmp_path / "store" / "artifacts" / "run_output"
    state_root = _state_root_for_output(output_root)

    store = _store_for_output(state_root=state_root, artifact_root=output_root)

    assert state_root == output_root / ".mvd-state"
    assert store.artifacts == output_root
    assert store.workspaces == state_root / "store" / "workspaces"
    assert output_root.is_dir()
    assert store.workspaces.is_dir()
    assert not (output_root / "store" / "artifacts").exists()


def test_store_layout_default_is_unchanged(tmp_path: Path) -> None:
    state_root = tmp_path / "state"

    store = _store_for_output(state_root=state_root)

    assert store.artifacts == state_root / "store" / "artifacts"
    assert store.workspaces == state_root / "store" / "workspaces"


def test_journal_snapshot_fallback_does_not_take_writer_lock(tmp_path: Path) -> None:
    from multiverse.journal import JournalKind, JournalLayout, JournalWriter
    from multiverse.runner.mvd_inprocess import snapshots_from_journal

    state_root = tmp_path / "state"
    layout = JournalLayout.at(state_root / "journal").ensure()
    writer = JournalWriter(layout, boot_id="boot-test")
    try:
        writer.append(
            JournalKind.JOB_INTENT,
            payload={
                "manifest_path": "/tmp/run_manifest.yaml",
                "options": {"dataset_slug": "ds", "model_slug": "pca"},
            },
            physical_attempt_id="attempt-1",
        )
        writer.append(
            JournalKind.STATE_TRANSITION,
            payload={"from_state": "PENDING", "to_state": "RUNNING", "reason": None},
            physical_attempt_id="attempt-1",
            prev_state="PENDING",
            next_state="RUNNING",
        )
        writer.commit()

        snapshots = snapshots_from_journal(
            state_root=state_root, attempt_ids=["attempt-1"]
        )
    finally:
        writer.close()

    assert snapshots == [
        {
            "physical_attempt_id": "attempt-1",
            "logical_run_id": None,
            "primary_state": "RUNNING",
            "cancel_requested": False,
            "failure_reason": None,
            "artifact_dir": None,
            "workspace_dir": None,
            "manifest_path": "/tmp/run_manifest.yaml",
            "submitted_wall_iso": snapshots[0]["submitted_wall_iso"],
            "projections": {
                "mlflow": "TRACKING_NOT_CONFIGURED",
                "optuna": "TRACKING_NOT_APPLICABLE",
            },
            "options": {"dataset_slug": "ds", "model_slug": "pca"},
        }
    ]
