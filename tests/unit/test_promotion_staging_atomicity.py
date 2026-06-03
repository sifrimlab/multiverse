"""Move-5 exit-gate tests for the staging-then-atomic-swap semantics.

Strategy v2 §5 acceptance: fault injection between every promotion step
cannot leave a split workspace/artifact truth.

The invariant tested here: the final artifact path (``store/artifacts/
<name>/``) does NOT exist on disk until ``PROMOTE_COMMIT_MANIFEST``'s
single ``os.replace(staging, final)`` syscall succeeds. Therefore any
crash before that rename leaves either:

    * the workspace untouched (PREPARE crash);
    * the workspace untouched + an identifiable staging dir
      (VALIDATE / STAGE crash);

never both a half-populated artifact dir AND a half-emptied workspace.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from multiverse.artifact import (ArtifactManifest, BootContext, ImageIdentity,
                                 ModelOutputContract, ProducedAt, ProducedBy,
                                 ValidationLevel, compute_logical_run_id,
                                 compute_manifest_hash, compute_params_hash,
                                 new_physical_attempt_id, produced_at_now,
                                 read_manifest)
from multiverse.journal import JournalLayout, JournalWriter
from multiverse.promotion import (PromotionOutcome, PromotionSaga,
                                  PromotionStep, StoreLayout)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> StoreLayout:
    return StoreLayout(root=tmp_path / "store").ensure()


@pytest.fixture
def journal_writer(tmp_path: Path):
    writer = JournalWriter(JournalLayout.at(tmp_path / "journal"), boot_id="B")
    yield writer
    writer.close()


def _workspace(tmp_path: Path, n_obs: int = 4) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    with h5py.File(ws / "embeddings.h5", "w") as f:
        f.create_dataset(
            "latent",
            data=np.random.default_rng(0)
            .standard_normal((n_obs, 4))
            .astype(np.float32),
        )
    # A nested subdir to verify the staging copy preserves structure.
    sub = ws / "logs"
    sub.mkdir()
    (sub / "model.log").write_text("hi\n")
    return ws


def _manifest() -> ArtifactManifest:
    boot = BootContext.new(mvd_version="0.1.0-test")
    image = ImageIdentity.registry_digest("sha256:" + "a" * 64)
    return ArtifactManifest(
        logical_run_id=compute_logical_run_id(
            manifest_hash=compute_manifest_hash("jobs: []\n"),
            dataset_fingerprint={"slug": "demo", "n_obs": 4},
            image_identity=image,
            params_hash=compute_params_hash({"x": 1}),
            mv_contract_version="1",
        ),
        physical_attempt_id=new_physical_attempt_id(),
        manifest_hash=compute_manifest_hash("jobs: []\n"),
        dataset_fingerprint={"slug": "demo", "n_obs": 4},
        image_identity=image,
        params_hash=compute_params_hash({"x": 1}),
        mv_contract_version="1",
        produced_at=ProducedAt.from_dict(produced_at_now(boot)),
        produced_by=ProducedBy(mvd_version=boot.mvd_version),
        artifacts=[],
        owner_token="t",
    )


def _saga(
    *,
    journal_writer: JournalWriter,
    store: StoreLayout,
    workspace: Path,
    manifest: ArtifactManifest,
    target_name: str = "demo_pca",
    after_step_hook=None,
) -> PromotionSaga:
    return PromotionSaga(
        journal=journal_writer,
        layout=store,
        physical_attempt_id=manifest.physical_attempt_id,
        logical_run_id=manifest.logical_run_id,
        workspace_dir=workspace,
        target_artifact_dir=store.artifacts / target_name,
        manifest=manifest,
        contract=ModelOutputContract.default(expected_n_obs=4),
        validators=ValidationLevel.BASIC,
        after_step_hook=after_step_hook,
    )


def _staging_for(store: StoreLayout, target_name: str, attempt_id: str) -> Path:
    return store.artifacts / f".{target_name}.staging.{attempt_id}"


# ---------------------------------------------------------------------------
# 1. The final artifact dir never appears before the atomic swap
# ---------------------------------------------------------------------------


class _Abort(Exception):
    pass


@pytest.mark.parametrize(
    "kill_after",
    [
        PromotionStep.PREPARE,
        PromotionStep.VALIDATE,
        PromotionStep.STAGE,
    ],
)
def test_final_path_absent_until_commit_manifest(
    kill_after: PromotionStep,
    tmp_path: Path,
    store: StoreLayout,
    journal_writer: JournalWriter,
) -> None:
    ws = _workspace(tmp_path)
    manifest = _manifest()
    saga = _saga(
        journal_writer=journal_writer,
        store=store,
        workspace=ws,
        manifest=manifest,
        after_step_hook=lambda s: (
            (_ for _ in ()).throw(_Abort) if s is kill_after else None
        ),
    )
    with pytest.raises(_Abort):
        saga.run()

    # Cardinal invariant: final artifact path never exists before
    # PROMOTE_COMMIT_MANIFEST's atomic swap.
    final = store.artifacts / "demo_pca"
    assert (
        not final.exists()
    ), f"final path {final} must not exist after crash post-{kill_after.value}"
    # Staging dir is the saga's recognisable scratch.
    staging = _staging_for(store, "demo_pca", manifest.physical_attempt_id)
    assert staging.is_dir()


# ---------------------------------------------------------------------------
# 2. STAGE crash: workspace truth is preserved (workspace files remain
#    where the saga decided to put them, no half-emptied original
#    workspace that ALSO failed to populate a target)
# ---------------------------------------------------------------------------


def test_stage_crash_does_not_split_truth(
    tmp_path: Path, store: StoreLayout, journal_writer: JournalWriter
) -> None:
    ws = _workspace(tmp_path)
    manifest = _manifest()
    saga = _saga(
        journal_writer=journal_writer,
        store=store,
        workspace=ws,
        manifest=manifest,
        after_step_hook=lambda s: (
            (_ for _ in ()).throw(_Abort) if s is PromotionStep.STAGE else None
        ),
    )
    with pytest.raises(_Abort):
        saga.run()

    # The final artifact path is absent.
    final = store.artifacts / "demo_pca"
    assert not final.exists()

    # Files live in staging; none escaped into the final artifact path.
    staging = _staging_for(store, "demo_pca", manifest.physical_attempt_id)
    staged_files = sorted(
        p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file()
    )
    assert "embeddings.h5" in staged_files
    assert "logs/model.log" in staged_files
    assert ".mvd_owner" in staged_files

    # Resume completes the saga via the same workspace path; the same-FS
    # rename branch already moved the originals into staging, but resume
    # tolerates a now-empty workspace and goes straight to COMMIT_MANIFEST.
    resume_saga = _saga(
        journal_writer=journal_writer,
        store=store,
        workspace=ws,
        manifest=manifest,
    )
    result = resume_saga.resume(last_committed_step=PromotionStep.VALIDATE)
    # Resume should re-run STAGE (idempotent over already-staged files).
    assert result.outcome is PromotionOutcome.PROMOTED
    loaded = read_manifest(final)
    assert loaded.logical_run_id == manifest.logical_run_id


# ---------------------------------------------------------------------------
# 3. COMMIT_MANIFEST is a single atomic rename
# ---------------------------------------------------------------------------


def test_commit_manifest_atomic_rename_only_appears_after_swap(
    tmp_path: Path, store: StoreLayout, journal_writer: JournalWriter
) -> None:
    ws = _workspace(tmp_path)
    manifest = _manifest()

    captured: dict = {}

    def _hook(step: PromotionStep) -> None:
        # At each step check whether the final path exists.
        captured[step.value] = (store.artifacts / "demo_pca").exists()

    saga = _saga(
        journal_writer=journal_writer,
        store=store,
        workspace=ws,
        manifest=manifest,
        after_step_hook=_hook,
    )
    result = saga.run()
    assert result.outcome is PromotionOutcome.PROMOTED
    assert captured["PROMOTE_PREPARE"] is False
    assert captured["PROMOTE_VALIDATE"] is False
    assert captured["PROMOTE_STAGE"] is False
    assert captured["PROMOTE_COMMIT_MANIFEST"] is True


# ---------------------------------------------------------------------------
# 4. Saga refuses to overwrite a pre-existing final artifact path
# ---------------------------------------------------------------------------


def test_commit_manifest_refuses_to_overwrite_an_existing_target(
    tmp_path: Path, store: StoreLayout, journal_writer: JournalWriter
) -> None:
    ws = _workspace(tmp_path)
    manifest = _manifest()
    saga = _saga(
        journal_writer=journal_writer,
        store=store,
        workspace=ws,
        manifest=manifest,
    )

    # Race: another writer creates the target between STAGE and COMMIT.
    interrupted: dict = {"raised": False}

    def _hook(step: PromotionStep) -> None:
        if step is PromotionStep.STAGE:
            (store.artifacts / "demo_pca").mkdir()
            (store.artifacts / "demo_pca" / "intruder").write_bytes(b"x")

    saga.after_step_hook = _hook
    with pytest.raises(FileExistsError):
        saga.run()
    interrupted["raised"] = True
    assert interrupted["raised"]

    # Staging dir still exists; the intruder's data is untouched.
    staging = _staging_for(store, "demo_pca", manifest.physical_attempt_id)
    assert staging.is_dir()
    assert (store.artifacts / "demo_pca" / "intruder").is_file()
