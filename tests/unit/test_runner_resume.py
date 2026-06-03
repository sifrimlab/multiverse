"""Tests for opt-in, mvd-backed manifest resume (STRATEGY: MVD Manifest
Resume and Dedupe Strategy).

Covers the resume policy precedence, the canonical logical-run identity used to
match completed work, and that only durable mvd ``ARTIFACT_SUCCESS`` state (not
the legacy ``runs`` table, not other states) suppresses a job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from multiverse.index import INDEX_FILENAME, open_index
from multiverse.runner.resume import (SKIP_REASON, completed_logical_runs,
                                      decorate_plan_with_resume,
                                      resolve_manifest_job_identity,
                                      resolve_skip_completed)

MANIFEST_HASH = "manifesthash0001"


def _job(**overrides: Any) -> Dict[str, Any]:
    job = {
        "model_slug": "pca",
        "model_name": "pca",
        "model_image": "multiverse-pca:1.0.0",
        "model_version": "1.0",
        "dataset_slug": "demo",
        "dataset_name": "demo",
        "dataset_path": "/tmp/data.h5mu",
        "dataset_n_obs": 100,
        "dataset_n_vars": 50,
        "model_params": {"n_components": 4},
        "params_hash": "abc123",
    }
    job.update(overrides)
    return job


def _seed_completed(
    state_root: Path,
    *,
    logical_run_id: str,
    artifact_dir: Path,
    state: str = "ARTIFACT_SUCCESS",
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with open_index(state_root / INDEX_FILENAME) as index:
        index.upsert_run(
            {
                "physical_attempt_id": "att-" + logical_run_id[:8],
                "logical_run_id": logical_run_id,
                "primary_state": state,
                "artifact_dir": str(artifact_dir),
                "options": {},
            }
        )


# ---------------------------------------------------------------------------
# Policy precedence
# ---------------------------------------------------------------------------


def test_skip_completed_defaults_off() -> None:
    assert resolve_skip_completed() is False
    assert resolve_skip_completed(manifest_data={"globals": {}}) is False


def test_skip_completed_reads_manifest_global() -> None:
    assert (
        resolve_skip_completed(manifest_data={"globals": {"skip_completed": True}})
        is True
    )


def test_cli_flag_overrides_manifest_global() -> None:
    data = {"globals": {"skip_completed": True}}
    assert resolve_skip_completed(cli_flag=False, manifest_data=data) is False
    assert (
        resolve_skip_completed(
            cli_flag=True, manifest_data={"globals": {"skip_completed": False}}
        )
        is True
    )


# ---------------------------------------------------------------------------
# Canonical identity sensitivity
# ---------------------------------------------------------------------------


def test_identity_stable_for_identical_jobs() -> None:
    a = resolve_manifest_job_identity(_job(), manifest_hash=MANIFEST_HASH, seed=42)
    b = resolve_manifest_job_identity(_job(), manifest_hash=MANIFEST_HASH, seed=42)
    assert a == b


@pytest.mark.parametrize(
    "overrides,seed",
    [
        ({"model_params": {"n_components": 8}}, 42),  # params
        ({}, 7),  # seed
        ({"preprocessing": {"n_top_genes": 250}}, 42),  # preprocessing
        ({"model_version": "2.0"}, 42),  # model version
        ({"model_image": "multiverse-pca:2.0.0"}, 42),  # image identity
    ],
)
def test_identity_changes_when_recipe_changes(
    overrides: Dict[str, Any], seed: int
) -> None:
    base = resolve_manifest_job_identity(_job(), manifest_hash=MANIFEST_HASH, seed=42)
    other = resolve_manifest_job_identity(
        _job(**overrides), manifest_hash=MANIFEST_HASH, seed=seed
    )
    assert base != other


# ---------------------------------------------------------------------------
# Completed-state reading
# ---------------------------------------------------------------------------


def test_completed_logical_runs_only_artifact_success(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    art = tmp_path / "art-ok"
    _seed_completed(state_root, logical_run_id="lrok", artifact_dir=art)
    for bad in ("FAILED", "CANCELLED", "RECOVERY_PENDING"):
        _seed_completed(
            state_root,
            logical_run_id="lr" + bad,
            artifact_dir=tmp_path / ("art-" + bad),
            state=bad,
        )
    completed = completed_logical_runs(state_root)
    assert set(completed) == {"lrok"}
    assert completed["lrok"]["artifact_dir"] == str(art)


def test_completed_requires_existing_artifact_dir(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    # ARTIFACT_SUCCESS but the artifact dir does not exist on disk.
    with open_index(state_root / INDEX_FILENAME) as index:
        index.upsert_run(
            {
                "physical_attempt_id": "att1",
                "logical_run_id": "lrghost",
                "primary_state": "ARTIFACT_SUCCESS",
                "artifact_dir": str(tmp_path / "does-not-exist"),
                "options": {},
            }
        )
    assert completed_logical_runs(state_root) == {}


def test_completed_empty_when_no_state(tmp_path: Path) -> None:
    assert completed_logical_runs(tmp_path / "missing") == {}


# ---------------------------------------------------------------------------
# Plan decoration / integration
# ---------------------------------------------------------------------------


def test_decorate_default_does_not_skip(tmp_path: Path) -> None:
    # No completed state at all: every job stays runnable.
    state_root = tmp_path / "state"
    plan = [_job()]
    decorated = decorate_plan_with_resume(
        plan, state_root=state_root, manifest_hash=MANIFEST_HASH, seed=42
    )
    assert not decorated[0].get("_skipped")
    assert decorated[0]["_logical_run_id"]


def test_decorate_skips_completed_with_provenance(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    job = _job()
    logical = resolve_manifest_job_identity(job, manifest_hash=MANIFEST_HASH, seed=42)
    art = tmp_path / "artifacts" / "done"
    _seed_completed(state_root, logical_run_id=logical, artifact_dir=art)

    decorated = decorate_plan_with_resume(
        [job], state_root=state_root, manifest_hash=MANIFEST_HASH, seed=42
    )
    assert decorated[0]["_skipped"] is True
    assert decorated[0]["_skip_reason"] == SKIP_REASON
    assert decorated[0]["_completed_artifact_dir"] == str(art)
    assert decorated[0]["_completed_attempt_id"]


def test_resume_roundtrip_skip_false_then_true(tmp_path: Path) -> None:
    """Integration-style: a completed ARTIFACT_SUCCESS attempt is runnable when
    skip is off and skipped when skip is on for the same manifest."""
    state_root = tmp_path / "state"
    state_root.mkdir()
    job = _job()
    logical = resolve_manifest_job_identity(job, manifest_hash=MANIFEST_HASH, seed=42)
    _seed_completed(state_root, logical_run_id=logical, artifact_dir=tmp_path / "done")

    # skip_completed = False: caller does not decorate -> job is runnable.
    assert resolve_skip_completed(cli_flag=False) is False

    # skip_completed = True: decoration marks the job skipped.
    decorated = decorate_plan_with_resume(
        [job], state_root=state_root, manifest_hash=MANIFEST_HASH, seed=42
    )
    assert decorated[0]["_skipped"] is True


def test_different_seed_does_not_match_completed(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    job = _job()
    logical_seed42 = resolve_manifest_job_identity(
        job, manifest_hash=MANIFEST_HASH, seed=42
    )
    _seed_completed(
        state_root, logical_run_id=logical_seed42, artifact_dir=tmp_path / "done"
    )

    # Re-plan with a different seed: identity differs, so nothing is skipped.
    decorated = decorate_plan_with_resume(
        [job], state_root=state_root, manifest_hash=MANIFEST_HASH, seed=99
    )
    assert not decorated[0].get("_skipped")


def test_preexisting_skipped_jobs_passthrough(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    validation_failure = {
        **_job(),
        "_skipped": True,
        "_skip_reason": "dataset unreadable",
    }
    decorated = decorate_plan_with_resume(
        [validation_failure],
        state_root=state_root,
        manifest_hash=MANIFEST_HASH,
        seed=42,
    )
    assert decorated[0]["_skip_reason"] == "dataset unreadable"
