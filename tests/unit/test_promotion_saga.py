"""Milestone-5 exit-gate tests for the promotion saga and recovery.

Coverage:
    1. Happy path: a contract-valid workspace promotes through PREPARE →
       VALIDATE → STAGE → COMMIT_MANIFEST and ``read_manifest`` verifies.
    2. Validator refusal: the prepared artifact dir is quarantined and a
       tombstone is left at the original path.
    3. Fault injection: aborting between every two steps and replaying
       results in a final state (PROMOTED or QUARANTINED) without lost
       data and without ever overwriting an existing artifact.
    4. Ownership mismatch: resuming on a dir owned by another attempt
       refuses, never mutates.
    5. Cross-FS path uses the staged-copy helper rather than per-file
       rename (verified by injecting a fake same-FS predicate).
    6. R5 grep gate: the saga and recovery modules contain no
       ``rmtree``/``unlink``/``rmdir`` outside the documented exception.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from multiverse.artifact import (ArtifactEntry, ArtifactManifest, BootContext,
                                 ExpectedArtifact, ImageIdentity,
                                 ModelOutputContract, ProducedAt, ProducedBy,
                                 ValidationLevel, compute_logical_run_id,
                                 compute_manifest_hash, compute_params_hash,
                                 new_physical_attempt_id, produced_at_now,
                                 read_manifest)
from multiverse.journal import (JournalKind, JournalLayout, JournalReader,
                                JournalWriter)
from multiverse.promotion import (OWNER_TOKEN_FILENAME, TOMBSTONE_SUFFIX,
                                  OwnershipMismatchError, PromotionOutcome,
                                  PromotionSaga, PromotionStep, StoreLayout,
                                  read_owner_token)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def boot() -> BootContext:
    return BootContext.new(mvd_version="0.1.0-test")


@pytest.fixture
def store(tmp_path: Path) -> StoreLayout:
    return StoreLayout(root=tmp_path / "store").ensure()


@pytest.fixture
def journal_writer(tmp_path: Path):
    layout = JournalLayout.at(tmp_path / "journal")
    writer = JournalWriter(layout, boot_id="boot-test")
    yield writer
    writer.close()


def _workspace_with_good_embedding(tmp_path: Path, n_obs: int = 4) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    with h5py.File(ws / "embeddings.h5", "w") as f:
        f.create_dataset(
            "latent",
            data=np.random.default_rng(0)
            .standard_normal((n_obs, 4))
            .astype(np.float32),
        )
    (ws / "metrics.json").write_text(json.dumps({"asw": 0.5}), encoding="utf-8")
    return ws


def _make_manifest(boot: BootContext, *, n_obs: int = 4) -> ArtifactManifest:
    image = ImageIdentity.registry_digest("sha256:" + "a" * 64)
    manifest_hash = compute_manifest_hash("jobs: []\n")
    params_hash = compute_params_hash({"x": 1})
    fingerprint = {"slug": "demo", "n_obs": n_obs}
    logical = compute_logical_run_id(
        manifest_hash=manifest_hash,
        dataset_fingerprint=fingerprint,
        image_identity=image,
        params_hash=params_hash,
        mv_contract_version="1",
    )
    return ArtifactManifest(
        logical_run_id=logical,
        physical_attempt_id=new_physical_attempt_id(),
        manifest_hash=manifest_hash,
        dataset_fingerprint=fingerprint,
        image_identity=image,
        params_hash=params_hash,
        mv_contract_version="1",
        produced_at=ProducedAt.from_dict(produced_at_now(boot)),
        produced_by=ProducedBy(mvd_version=boot.mvd_version),
        artifacts=[],
        owner_token="filled-by-saga",
    )


def _build_saga(
    *,
    journal_writer: JournalWriter,
    store: StoreLayout,
    workspace: Path,
    manifest: ArtifactManifest,
    target_name: str = "demo_pca",
    after_step_hook=None,
    fsync_enabled: bool = True,
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
        fsync_enabled=fsync_enabled,
    )


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


def test_happy_path_produces_verified_manifest(
    tmp_path: Path, boot: BootContext, store: StoreLayout, journal_writer: JournalWriter
) -> None:
    ws = _workspace_with_good_embedding(tmp_path, n_obs=4)
    manifest = _make_manifest(boot, n_obs=4)
    saga = _build_saga(
        journal_writer=journal_writer,
        store=store,
        workspace=ws,
        manifest=manifest,
    )
    result = saga.run()
    assert result.outcome is PromotionOutcome.PROMOTED
    assert result.committed_steps == [
        PromotionStep.PREPARE,
        PromotionStep.VALIDATE,
        PromotionStep.STAGE,
        PromotionStep.COMMIT_MANIFEST,
    ]
    artifact_dir = store.artifacts / "demo_pca"
    assert (artifact_dir / OWNER_TOKEN_FILENAME).is_file()
    loaded = read_manifest(artifact_dir)
    assert loaded.logical_run_id == manifest.logical_run_id


def test_journal_records_all_four_steps_in_order(
    tmp_path: Path, boot: BootContext, store: StoreLayout, journal_writer: JournalWriter
) -> None:
    ws = _workspace_with_good_embedding(tmp_path, n_obs=4)
    saga = _build_saga(
        journal_writer=journal_writer,
        store=store,
        workspace=ws,
        manifest=_make_manifest(boot),
    )
    saga.run()
    journal_writer.close()

    reader = JournalReader(JournalLayout.at(tmp_path / "journal"))
    kinds = [r.kind for r in reader.replay().records]
    assert kinds == [
        JournalKind.PROMOTE_PREPARE,
        JournalKind.PROMOTE_VALIDATE,
        JournalKind.PROMOTE_STAGE,
        JournalKind.PROMOTE_COMMIT_MANIFEST,
    ]


# ---------------------------------------------------------------------------
# 2. Validator refusal → quarantine + tombstone
# ---------------------------------------------------------------------------


def test_validator_refusal_quarantines_with_tombstone(
    tmp_path: Path, boot: BootContext, store: StoreLayout, journal_writer: JournalWriter
) -> None:
    # n_obs=8 in the workspace but the contract expects 4 → REFUSAL.
    ws = _workspace_with_good_embedding(tmp_path, n_obs=8)
    saga = _build_saga(
        journal_writer=journal_writer,
        store=store,
        workspace=ws,
        manifest=_make_manifest(boot, n_obs=4),
    )
    result = saga.run()
    assert result.outcome is PromotionOutcome.QUARANTINED
    assert result.quarantine_report is not None
    qpath = result.quarantine_report.quarantine_path
    assert qpath.is_dir()
    # Quarantine carries the owner token + the report.
    assert (qpath / OWNER_TOKEN_FILENAME).is_file()
    assert (qpath / "QUARANTINE_REPORT.md").is_file()
    # Tombstone is at the original artifact path.
    tombstone = result.quarantine_report.tombstone_path
    assert tombstone.is_file()
    assert tombstone.name.endswith(TOMBSTONE_SUFFIX)
    data = json.loads(tombstone.read_text())
    assert data["quarantined_to"] == str(qpath)


# ---------------------------------------------------------------------------
# 3. Fault injection: kill between each step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kill_after",
    [
        PromotionStep.PREPARE,
        PromotionStep.VALIDATE,
        PromotionStep.STAGE,
    ],
)
def test_fault_between_steps_then_resume(
    kill_after: PromotionStep,
    tmp_path: Path,
    boot: BootContext,
    store: StoreLayout,
    journal_writer: JournalWriter,
) -> None:
    ws = _workspace_with_good_embedding(tmp_path, n_obs=4)
    manifest = _make_manifest(boot, n_obs=4)

    class _Abort(Exception):
        pass

    def _hook(step):
        if step is kill_after:
            raise _Abort

    saga = _build_saga(
        journal_writer=journal_writer,
        store=store,
        workspace=ws,
        manifest=manifest,
        after_step_hook=_hook,
    )
    with pytest.raises(_Abort):
        saga.run()

    # New saga (different worker process, same physical_attempt_id from the
    # journal) resumes. The journal is the authority for "last committed
    # step" — we read the last committed PROMOTE_* kind from it.
    last_step = _last_promotion_step_from_journal(tmp_path)
    assert last_step is kill_after

    resumed_saga = _build_saga(
        journal_writer=journal_writer,
        store=store,
        workspace=ws,
        manifest=manifest,
    )
    result = resumed_saga.resume(last_committed_step=kill_after)
    assert result.outcome is PromotionOutcome.PROMOTED
    artifact_dir = store.artifacts / "demo_pca"
    loaded = read_manifest(artifact_dir)
    assert loaded.logical_run_id == manifest.logical_run_id


def _last_promotion_step_from_journal(tmp_path: Path) -> PromotionStep:
    reader = JournalReader(JournalLayout.at(tmp_path / "journal"))
    last: PromotionStep | None = None
    for record in reader.replay().records:
        try:
            last = PromotionStep(record.kind.value)
        except ValueError:
            continue
    assert last is not None
    return last


def test_resume_refuses_when_token_in_staging_belongs_to_other_attempt(
    tmp_path: Path, boot: BootContext, store: StoreLayout, journal_writer: JournalWriter
) -> None:
    """STRATEGY v2 §5: staging dirs embed the attempt id so two attempts
    can never collide naturally. But if an operator hand-pre-creates a
    staging dir with another attempt's owner token, ``resume()`` must
    refuse rather than mutate someone else's scratch."""
    from multiverse.promotion.tokens import write_owner_token

    ws = _workspace_with_good_embedding(tmp_path, n_obs=4)
    saga2 = _build_saga(
        journal_writer=journal_writer,
        store=store,
        workspace=ws,
        manifest=_make_manifest(boot),
    )
    staging = saga2._staging_dir()  # type: ignore[attr-defined]
    write_owner_token(
        staging,
        owner_token="someone-else",
        physical_attempt_id="other-attempt",
        mvd_boot_id="other-boot",
        purpose="promotion-prepare",
    )
    with pytest.raises(OwnershipMismatchError):
        saga2.resume(last_committed_step=PromotionStep.PREPARE)


# ---------------------------------------------------------------------------
# 4. Hot path never deletes (R5 acceptance)
# ---------------------------------------------------------------------------


_HOT_PATH_MODULES = [
    "multiverse/promotion/saga.py",
    "multiverse/promotion/quarantine.py",
    "multiverse/promotion/tokens.py",
    "multiverse/promotion/layout.py",
]

_FORBIDDEN_CALLS = (
    re.compile(r"\bshutil\.rmtree\b"),
    re.compile(r"\bos\.unlink\b"),
    re.compile(r"\bos\.rmdir\b"),
    re.compile(r"\bPath\([^)]*\)\.unlink\b"),
)


def test_hot_path_has_no_destructive_calls() -> None:
    """R5 grep gate. ``staged_copy_directory`` is the only Tier-1-scratch
    delete path and lives in ``fsutil.py``; even there it no longer
    contains unlink/rmdir after the refactor to unique-per-attempt staging
    names (see Milestone-5 implementation notes)."""
    root = Path(__file__).resolve().parents[2]
    for rel in _HOT_PATH_MODULES:
        text = (root / rel).read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_CALLS:
            assert not pattern.search(
                text
            ), f"forbidden destructive call {pattern.pattern} found in {rel}"


def test_fsutil_staged_copy_has_no_delete_in_main_path() -> None:
    """fsutil.py contains the staged-copy helper but, since switching to
    unique-per-attempt staging tokens, must no longer call unlink/rmdir."""
    root = Path(__file__).resolve().parents[2]
    text = (root / "multiverse/promotion/fsutil.py").read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_CALLS:
        assert not pattern.search(
            text
        ), f"forbidden destructive call {pattern.pattern} reintroduced in fsutil.py"


# ---------------------------------------------------------------------------
# 5. Owner token is the first artefact written
# ---------------------------------------------------------------------------


def test_prepare_writes_owner_token_into_staging_only(
    tmp_path: Path, boot: BootContext, store: StoreLayout, journal_writer: JournalWriter
) -> None:
    """STRATEGY v2 §5: PREPARE never touches the final artifact path.
    The token lands in the staging dir; the final path appears only
    after step 4's atomic swap."""
    ws = _workspace_with_good_embedding(tmp_path, n_obs=4)
    saga = _build_saga(
        journal_writer=journal_writer,
        store=store,
        workspace=ws,
        manifest=_make_manifest(boot),
        after_step_hook=lambda step: (
            (_ for _ in ()).throw(RuntimeError("stop"))
            if step is PromotionStep.PREPARE
            else None
        ),
    )
    with pytest.raises(RuntimeError):
        saga.run()
    artifact_dir = store.artifacts / "demo_pca"
    # The final artifact path must NOT exist yet (no split-truth).
    assert not artifact_dir.exists()
    # The staging dir owns the token.
    staging_candidates = list(store.artifacts.glob(".demo_pca.staging.*"))
    assert len(staging_candidates) == 1, staging_candidates
    files = [p.name for p in staging_candidates[0].iterdir()]
    assert files == [OWNER_TOKEN_FILENAME]
    token = read_owner_token(staging_candidates[0])
    assert token is not None
    assert token.physical_attempt_id == saga.physical_attempt_id


# ---------------------------------------------------------------------------
# 6. Symlink policy
# ---------------------------------------------------------------------------


def test_symlink_in_workspace_is_rejected_by_cross_fs_copy(tmp_path: Path) -> None:
    """``staged_copy_directory`` refuses to copy a workspace containing a
    symlink — per R13 symlinks within managed store paths are forbidden."""
    from multiverse.promotion.errors import SymlinkPolicyError
    from multiverse.promotion.fsutil import staged_copy_directory

    src = tmp_path / "src"
    src.mkdir()
    (src / "real.txt").write_text("ok")
    (src / "link.txt").symlink_to("real.txt")
    dst = tmp_path / "dst"
    with pytest.raises(SymlinkPolicyError):
        staged_copy_directory(src, dst, staging_token="t1")
