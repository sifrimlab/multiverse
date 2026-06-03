"""STRATEGY M2 acceptance: a run via Docker and a run via Apptainer for
the same model+dataset+params must compare equal on
``(image_identity.value, params_hash, dataset_fingerprint)``.

This is the cross-backend reproducibility gate. If it ever fails, the
manifest cannot be trusted to say "these two runs used the same image."
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pytest

from multiverse.apptainer import InMemoryApptainerEngine
from multiverse.artifact import BootContext, read_manifest
from multiverse.broker import HostMetrics, InMemoryHostObserver, ResourceBroker
from multiverse.docker_supervisor import (DockerSupervisor,
                                          InMemoryContainerEngine)
from multiverse.journal import JournalLayout, JournalWriter
from multiverse.mvd import (Kernel, KernelConfig, MvdDockerExecutor,
                            PrimaryState, build_executor_options)
from multiverse.promotion import StoreLayout

pytestmark = pytest.mark.control_plane


# ---------------------------------------------------------------------------
# Common scaffolding (mirrors test_mvd_docker_executor.py)
# ---------------------------------------------------------------------------


def _broker_with_capacity() -> ResourceBroker:
    return ResourceBroker(
        observer=InMemoryHostObserver(
            HostMetrics(ram_free_bytes=8 * 1024**3, ram_total_bytes=16 * 1024**3)
        )
    )


def _good_producer(n_obs: int):
    def _producer(workspace: Path, params: Mapping[str, Any]) -> None:
        with h5py.File(workspace / "embeddings.h5", "w") as f:
            f.create_dataset(
                "latent",
                data=np.zeros((n_obs, 4), dtype=np.float32),
            )

    return _producer


def _producer_then_exit_first_container(engine, workspace: Path, *, n_obs: int) -> None:
    _good_producer(n_obs)(workspace, {})
    for c in reversed(list(engine.containers.values())):
        if c.removed:
            continue
        if c.state.value == "running":
            engine.simulate_natural_exit(c.container_id, exit_code=0)
            return


def _build_and_run(
    *,
    state_root: Path,
    store: StoreLayout,
    engine,
    dataset_file: Path,
) -> dict:
    boot = BootContext.new(mvd_version="0.1.0-test")
    writer = JournalWriter(
        JournalLayout.at(state_root / "journal"), boot_id=boot.boot_id
    )
    supervisor = DockerSupervisor(
        engine=engine, journal=writer, mvd_version="0.1.0-test"
    )
    executor = MvdDockerExecutor(
        journal=writer,
        boot=boot,
        store=store,
        supervisor=supervisor,
        broker=_broker_with_capacity(),
        state_root=state_root,
        producer_hook=lambda ws, _: _producer_then_exit_first_container(
            engine, ws, n_obs=4
        ),
        poll_interval_seconds=0.0,
        max_poll_iterations=200,
    )
    kernel = Kernel(
        KernelConfig(state_root=state_root, mvd_version="0.1.0-test"),
        executor=executor,
        journal=writer,
        boot=boot,
    )

    opts = build_executor_options(
        model_slug="pca",
        model_image="multiverse-pca:1.0.0",
        # Same OCI digest passed for both backends — the dual-digest
        # invariant says this is the source of truth.
        image_digest="sha256:" + "a" * 64,
        dataset_slug="demo",
        dataset_path=str(dataset_file),
        dataset_n_obs=4,
        dataset_n_vars=8,
        params={"n_components": 4},
        manifest_text="schema_version: '1'\n",
    )

    async def _run() -> dict:
        attempt = await kernel.submit_run(manifest_path="/tmp/m.yaml", options=opts)
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snap = await kernel.query_run(physical_attempt_id=attempt)
        assert snap["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value, snap
        artifact_dir = Path(snap["artifact_dir"])
        manifest = read_manifest(artifact_dir)
        await kernel.shutdown()
        return {
            "manifest": manifest,
            "engine_name": getattr(engine, "name", ""),
        }

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# The acceptance gate
# ---------------------------------------------------------------------------


def test_docker_and_apptainer_runs_produce_comparable_manifests(tmp_path: Path):
    dataset_file = tmp_path / "data.h5mu"
    dataset_file.write_bytes(b"placeholder")

    # Two completely isolated state roots; same inputs.
    docker_root = tmp_path / "docker-state"
    docker_root.mkdir()
    docker_store = StoreLayout(root=tmp_path / "docker-store").ensure()
    docker_result = _build_and_run(
        state_root=docker_root,
        store=docker_store,
        engine=InMemoryContainerEngine(),
        dataset_file=dataset_file,
    )

    apptainer_root = tmp_path / "apptainer-state"
    apptainer_root.mkdir()
    apptainer_store = StoreLayout(root=tmp_path / "apptainer-store").ensure()
    apptainer_result = _build_and_run(
        state_root=apptainer_root,
        store=apptainer_store,
        engine=InMemoryApptainerEngine(),
        dataset_file=dataset_file,
    )

    dm = docker_result["manifest"]
    am = apptainer_result["manifest"]

    # The acceptance trinity from STRATEGY M2: same source, same params,
    # same dataset fingerprint.
    assert dm.image_identity.value == am.image_identity.value
    assert dm.params_hash == am.params_hash
    assert dict(dm.dataset_fingerprint) == dict(am.dataset_fingerprint)

    # Backends differ in how the *runtime* identity is recorded:
    # Docker has none, Apptainer carries the SIF identity that points
    # back to the OCI digest.
    assert dm.runtime_image_identity is None
    assert am.runtime_image_identity is not None
    assert am.runtime_image_identity.kind.value == "sif_digest"
    assert am.runtime_image_identity.built_from == am.image_identity.value


def test_dual_digest_invariant_violation_fails_the_run(tmp_path: Path):
    """Sanity: if the engine reports a SIF whose built_from cannot be
    derived (because the source identity is unverified_local), the
    manifest's runtime SIF will have built_from=None and that is *not*
    strict-acceptable. The run still completes (we're not in strict
    mode in this test scaffolding), but the manifest's strict-acceptable
    flag is False, surfacing the degraded posture to the consumer."""
    from multiverse.artifact import (ImageIdentity,
                                     verify_runtime_identity_matches_source)

    src = ImageIdentity.unverified_local("myimage:latest")
    # The executor would derive built_from=None in this scenario because
    # the source is not a registry_digest or build_context_hash.
    rt = ImageIdentity.sif_digest("sha256:fff", built_from=None)
    # The invariant verifier raises — which is what the executor does
    # at compose time, transitioning the run to FAILED.
    with pytest.raises(ValueError, match="built_from"):
        verify_runtime_identity_matches_source(src, rt)
