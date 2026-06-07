"""Milestone-10 exit-gate tests for MLflow projection sync.

Coverage:
    1. Happy-path sync: a verified artifact bundle round-trips into the
       MLflow target with tags, params, metrics, and artifacts logged.
    2. MLflow outage at ``create_run`` → SYNC_FAILED; kernel's primary
       state stays ARTIFACT_SUCCESS (R6).
    3. MLflow outage at ``log_metrics`` → SYNC_FAILED.
    4. Corrupt manifest → SYNC_FAILED without contacting the target.
    5. Re-sync after a transient failure flips the projection back to
       TRACKING_SYNCED, primary state unchanged.
    6. Import-graph: the projection package does NOT import the real
       ``mlflow`` SDK at module load time.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from multiverse.artifact import (ARTIFACT_MANIFEST_FILENAME, ArtifactManifest,
                                 BootContext, BundleInputs, ImageIdentity,
                                 ModelOutputContract, ProducedAt, ProducedBy,
                                 ValidationLevel, compute_logical_run_id,
                                 compute_manifest_hash, compute_params_hash,
                                 new_physical_attempt_id, produced_at_now,
                                 validate_output_bundle, write_bundle)
from multiverse.mvd import (Kernel, KernelConfig, PrimaryState,
                            SyntheticRunExecutor)
from multiverse.projection import (DEFAULT_PROJECTION_PLUGIN, MLflowSyncPlugin,
                                   SyncOutcome)
from multiverse.projection.base import InMemoryMLflowTarget

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_bundle(tmp_path: Path) -> Path:
    """Produce a contract-valid bundle for a fake workspace."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with h5py.File(workspace / "embeddings.h5", "w") as f:
        f.create_dataset(
            "latent",
            data=np.random.default_rng(0).standard_normal((4, 4)).astype(np.float32),
        )
    (workspace / "metrics.json").write_text(
        json.dumps({"asw": 0.5, "ari": 0.7, "note": "ignored"})
    )

    boot = BootContext.new(mvd_version="0.1.0-test")
    image = ImageIdentity.registry_digest("sha256:" + "a" * 64)
    manifest_hash = compute_manifest_hash("jobs: []\n")
    params_hash = compute_params_hash({"n_components": 4})
    fingerprint = {"slug": "demo", "n_obs": 4}
    logical = compute_logical_run_id(
        manifest_hash=manifest_hash,
        dataset_fingerprint=fingerprint,
        image_identity=image,
        params_hash=params_hash,
        mv_contract_version="1",
    )
    contract = ModelOutputContract.default(expected_n_obs=4)
    report = validate_output_bundle(workspace, contract, ValidationLevel.BASIC)
    assert report.passed

    manifest = ArtifactManifest(
        logical_run_id=logical,
        physical_attempt_id=new_physical_attempt_id(),
        manifest_hash=manifest_hash,
        dataset_fingerprint=fingerprint,
        image_identity=image,
        params_hash=params_hash,
        mv_contract_version="1",
        produced_at=ProducedAt.from_dict(produced_at_now(boot)),
        produced_by=ProducedBy(mvd_version=boot.mvd_version),
        artifacts=list(report.artifact_entries),
        owner_token="t",
    )

    bundle = tmp_path / "bundle"
    write_bundle(
        bundle,
        BundleInputs(
            artifact_manifest=manifest,
            outputs={
                "embeddings.h5": workspace / "embeddings.h5",
                "metrics.json": workspace / "metrics.json",
            },
            environment={"mvd_version": "0.1.0-test"},
            validation_report=report.to_dict(),
        ),
    )
    return bundle


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


def test_happy_path_syncs_bundle(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    target = InMemoryMLflowTarget()
    plugin = MLflowSyncPlugin(target=target)

    result = plugin.sync_bundle(bundle)
    assert result.outcome is SyncOutcome.SYNCED
    assert result.target_run_id in target.runs
    run = target.runs[result.target_run_id]
    assert run.terminal_status == "FINISHED"
    assert run.metrics == {"asw": 0.5, "ari": 0.7}
    assert any(p.endswith("embeddings.h5") for p in run.artifacts)
    assert any(p.endswith("metrics.json") for p in run.artifacts)
    assert any(p.endswith(ARTIFACT_MANIFEST_FILENAME) for p in run.artifacts)


def test_tags_carry_image_identity_kind_and_value(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    target = InMemoryMLflowTarget()
    plugin = MLflowSyncPlugin(target=target)
    result = plugin.sync_bundle(bundle)
    run = target.runs[result.target_run_id]
    assert run.tags["multiverse.image_kind"] == "registry_digest"
    assert run.tags["multiverse.image_value"].startswith("sha256:")


# ---------------------------------------------------------------------------
# 2-3. Outage paths
# ---------------------------------------------------------------------------


def test_outage_at_create_run_yields_sync_failed_without_side_effects(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path)
    target = InMemoryMLflowTarget(fail_on_create=True)
    plugin = MLflowSyncPlugin(target=target)
    result = plugin.sync_bundle(bundle)
    assert result.outcome is SyncOutcome.SYNC_FAILED
    assert "MLflow tracking server unreachable" in result.failure_reason
    assert target.runs == {}


def test_outage_at_log_metrics_yields_sync_failed(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    target = InMemoryMLflowTarget(fail_on_log=True)
    plugin = MLflowSyncPlugin(target=target)
    result = plugin.sync_bundle(bundle)
    assert result.outcome is SyncOutcome.SYNC_FAILED


# ---------------------------------------------------------------------------
# 4. Corrupt manifest never contacts the target
# ---------------------------------------------------------------------------


def test_corrupt_manifest_sync_failed_no_target_calls(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    (bundle / ARTIFACT_MANIFEST_FILENAME).write_text(
        (bundle / ARTIFACT_MANIFEST_FILENAME).read_text() + "\n# tampered\n"
    )
    target = InMemoryMLflowTarget()
    plugin = MLflowSyncPlugin(target=target)
    result = plugin.sync_bundle(bundle)
    assert result.outcome is SyncOutcome.SYNC_FAILED
    assert target.runs == {}  # never created a run; sidecar gate held


# ---------------------------------------------------------------------------
# 5. Kernel projection-status independence (R6 acceptance)
# ---------------------------------------------------------------------------


def test_kernel_primary_state_unchanged_by_projection_outcomes(
    tmp_path: Path,
) -> None:
    kernel = Kernel(
        KernelConfig(state_root=tmp_path / "state"),
        executor=SyntheticRunExecutor("success"),
    )

    async def _scenario() -> None:
        attempt = await kernel.submit_run(manifest_path="/tmp/m.yaml")
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        # Outage: report TRACKING_SYNC_FAILED.
        await kernel.report_projection_status(
            plugin=DEFAULT_PROJECTION_PLUGIN,
            physical_attempt_id=attempt,
            status="TRACKING_SYNC_FAILED",
            details={"error": "outage"},
        )
        snap = await kernel.query_run(physical_attempt_id=attempt)
        assert snap["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value
        assert snap["projections"]["mlflow"] == "TRACKING_SYNC_FAILED"

        # Later: re-sync succeeds.
        await kernel.report_projection_status(
            plugin=DEFAULT_PROJECTION_PLUGIN,
            physical_attempt_id=attempt,
            status="TRACKING_SYNCED",
        )
        snap2 = await kernel.query_run(physical_attempt_id=attempt)
        assert snap2["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value
        assert snap2["projections"]["mlflow"] == "TRACKING_SYNCED"
        await kernel.shutdown()

    asyncio.run(_scenario())


# ---------------------------------------------------------------------------
# 6. Import-graph: projection package does not load mlflow at import time
# ---------------------------------------------------------------------------


def test_projection_package_does_not_load_real_mlflow_at_import() -> None:
    import subprocess

    script = (
        "import sys\n"
        "import multiverse.projection  # noqa\n"
        "from multiverse.projection import MLflowSyncPlugin  # noqa\n"
        "if 'mlflow' in sys.modules:\n"
        "    print('mlflow loaded')\n"
        "    raise SystemExit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"projection package eagerly loaded mlflow: {result.stdout.strip()!r}\n"
        f"stderr: {result.stderr}"
    )


def test_sync_many_collects_per_bundle_outcomes(tmp_path: Path) -> None:
    b1 = _build_bundle(tmp_path / "a")
    b2 = _build_bundle(tmp_path / "b")
    target = InMemoryMLflowTarget()
    plugin = MLflowSyncPlugin(target=target)
    results = plugin.sync_many([b1, b2])
    assert [r.outcome for r in results] == [SyncOutcome.SYNCED, SyncOutcome.SYNCED]
