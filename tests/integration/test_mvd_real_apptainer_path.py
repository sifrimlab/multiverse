"""Real-Apptainer mvd end-to-end tests (STRATEGY G5 / M2 / R1).

These tests wire the full path:
    Kernel → MvdDockerExecutor → DockerSupervisor → RealApptainerEngine
    → PromotionSaga → rebuild_index

They skip entirely when:
  * ``apptainer`` (or ``singularity``) is not on PATH, OR
  * the binary cannot actually exec a container (user-namespace/setuid not
    configured) — detected by a functional probe at session start.

They do NOT skip on OOM classification failures. If memory-limit enforcement
regresses, the OOM test FAILS (not skips). That is the R1 contract.

Environment overrides
---------------------
* ``MVD_REAL_APPTAINER_SIF``       — path to a pre-built busybox-equivalent SIF
                                      (replaces the auto-built fixture).
* ``MVD_REAL_APPTAINER_BIN``       — override the apptainer binary (default:
                                      ``apptainer``, fallback ``singularity``).
* ``MVD_REAL_APPTAINER_OOM_SIF``   — path to a pre-built OOM-fixture SIF.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
from hashlib import sha256
from pathlib import Path
from typing import Optional
from uuid import uuid4

import h5py
import numpy as np
import pytest

from multiverse.apptainer import RealApptainerEngine
from multiverse.artifact import BootContext, ImageIdentityKind, read_manifest
from multiverse.broker import HostMetrics, InMemoryHostObserver, ResourceBroker
from multiverse.docker_supervisor import DockerSupervisor
from multiverse.index import open_index, rebuild_index
from multiverse.index.sqlite_index import INDEX_FILENAME
from multiverse.journal import JournalLayout, JournalWriter
from multiverse.mvd import (
    Kernel,
    KernelConfig,
    MvdDockerExecutor,
    PrimaryState,
    build_executor_options,
)
from multiverse.promotion import StoreLayout


# ---------------------------------------------------------------------------
# Apptainer availability probe
# ---------------------------------------------------------------------------


def _apptainer_bin() -> Optional[str]:
    override = os.environ.get("MVD_REAL_APPTAINER_BIN")
    if override:
        return override if shutil.which(override) else None
    return shutil.which("apptainer") or shutil.which("singularity")


def _apptainer_can_exec(sif_path: Path) -> bool:
    """Return True iff apptainer exec of a minimal SIF succeeds."""
    bin_name = _apptainer_bin()
    if not bin_name:
        return False
    try:
        result = subprocess.run(
            [bin_name, "exec", str(sif_path), "sh", "-c", "echo probe-ok"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0 and "probe-ok" in result.stdout
    except Exception:
        return False


# ---------------------------------------------------------------------------
# SIF building helpers
# ---------------------------------------------------------------------------


def _docker_available() -> bool:
    return shutil.which("docker") is not None and _docker_daemon_up()


def _docker_daemon_up() -> bool:
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5, check=False
        )
        return r.returncode == 0
    except Exception:
        return False


def _build_sif_from_docker(docker_image: str, sif_path: Path, bin_name: str) -> bool:
    """Convert a local Docker image to a SIF. Returns True on success."""
    try:
        result = subprocess.run(
            [bin_name, "build", str(sif_path), f"docker-daemon://{docker_image}"],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        return result.returncode == 0 and sif_path.is_file()
    except Exception:
        return False


def _find_local_docker_shell_image() -> Optional[str]:
    """Return the name of a small locally-available Docker image with ``sh``."""
    override = os.environ.get("MVD_REAL_DOCKER_BASE_IMAGE")
    candidates = (override,) if override else (
        "busybox:latest",
        "alpine:latest",
        "multiverse-pca:1.0.0",
    )
    try:
        import docker  # type: ignore[import-untyped]
        client = docker.from_env()
    except Exception:
        return None
    for img in candidates:
        if not img:
            continue
        try:
            client.images.get(img)
            return img
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Session-scoped SIF fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _session_sif_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("apptainer-sifs")


@pytest.fixture(scope="session")
def apptainer_busybox_sif(_session_sif_dir: Path):
    """Return path to a busybox-equivalent SIF, skipping if not obtainable."""
    explicit = os.environ.get("MVD_REAL_APPTAINER_SIF")
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            pytest.skip(f"MVD_REAL_APPTAINER_SIF not found: {p}")
        return p

    bin_name = _apptainer_bin()
    if not bin_name:
        pytest.skip(
            "apptainer/singularity not on PATH; install Apptainer to run G5 tests"
        )

    sif_path = _session_sif_dir / "busybox.sif"
    if sif_path.is_file():
        return sif_path

    docker_img = _find_local_docker_shell_image() if _docker_available() else None
    if docker_img is None:
        pytest.skip(
            "No local Docker shell image available and MVD_REAL_APPTAINER_SIF not set; "
            "pre-load busybox:latest or set MVD_REAL_APPTAINER_SIF to a busybox SIF path"
        )

    built = _build_sif_from_docker(docker_img, sif_path, bin_name)
    if not built:
        pytest.skip(
            f"apptainer build from docker-daemon://{docker_img} failed; "
            "check Docker daemon and apptainer install"
        )
    return sif_path


@pytest.fixture(scope="session")
def apptainer_available(apptainer_busybox_sif: Path):
    """Skip all G5 tests if the apptainer exec probe fails (e.g. user-ns blocked)."""
    if not _apptainer_can_exec(apptainer_busybox_sif):
        pytest.skip(
            "apptainer exec probe failed — user namespaces or setuid not configured; "
            "check /proc/sys/kernel/unprivileged_userns_clone and /etc/subuid entries, "
            "or use a setuid-installed Apptainer"
        )


# ---------------------------------------------------------------------------
# OOM fixture SIF
# ---------------------------------------------------------------------------

# A pure-shell memory allocator: doubles a shell variable until the cgroup
# kills the process. Mirrors the Docker integration-test OOM fixture.
_OOM_FIXTURE_CMD = (
    "set +e; "
    "echo oom-started > /output/oom-heartbeat; "
    "x=A; while :; do x=$x$x; done"
)

_OOM_MEM_LIMIT = "64m"


@pytest.fixture(scope="session")
def apptainer_oom_sif(_session_sif_dir: Path, apptainer_available):
    """Return a SIF whose default CMD allocates memory until OOM-killed."""
    explicit = os.environ.get("MVD_REAL_APPTAINER_OOM_SIF")
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            pytest.skip(f"MVD_REAL_APPTAINER_OOM_SIF not found: {p}")
        return p

    bin_name = _apptainer_bin()
    docker_img = _find_local_docker_shell_image() if _docker_available() else None
    if docker_img is None:
        pytest.skip(
            "No Docker shell image available to build the OOM fixture SIF; "
            "set MVD_REAL_APPTAINER_OOM_SIF to a pre-built SIF"
        )

    # Tag the SIF by the hash of the OOM command so any change forces a rebuild.
    digest = sha256(_OOM_FIXTURE_CMD.encode()).hexdigest()[:12]
    sif_path = _session_sif_dir / f"oom-fixture-{digest}.sif"
    if sif_path.is_file():
        return sif_path

    # Build a custom Docker image that runs the OOM allocator by default,
    # then convert to SIF.
    with tempfile.TemporaryDirectory(prefix="mvd-apptainer-oom-") as workdir:
        import json

        ctx = Path(workdir)
        (ctx / "Dockerfile").write_text(
            "\n".join([
                f"FROM {docker_img}",
                "RUN mkdir -p /input /output",
                f'CMD ["sh", "-c", {json.dumps(_OOM_FIXTURE_CMD)}]',
                "",
            ]),
            encoding="utf-8",
        )
        tag = f"mvd-apptainer-oom-fixture:{digest}"
        r = subprocess.run(
            ["docker", "build", "--pull=false", "-t", tag, str(ctx)],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            pytest.skip(
                f"OOM Docker fixture build failed: {r.stderr.strip()[-400:]}"
            )
        built = _build_sif_from_docker(tag, sif_path, bin_name)
        # Clean up the ephemeral Docker image.
        subprocess.run(["docker", "rmi", tag], capture_output=True, check=False)

    if not built:
        pytest.skip("apptainer build of OOM fixture SIF failed")
    return sif_path


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------


def _dataset_file(tmp_path: Path, *, n_obs: int = 4) -> Path:
    dataset = tmp_path / "fixture.h5mu"
    with h5py.File(dataset, "w") as f:
        f.create_dataset(
            "latent",
            data=np.random.default_rng(0).standard_normal((n_obs, 4)).astype(np.float32),
        )
    return dataset


def _kernel_with_real_apptainer(
    *,
    state_root: Path,
    store: StoreLayout,
    poll_interval_seconds: float = 0.1,
    max_poll_iterations: int = 300,
) -> Kernel:
    boot = BootContext.new(mvd_version="0.1.0-real-apptainer-it")
    journal = JournalWriter(
        JournalLayout.at(state_root / "journal").ensure(),
        boot_id=boot.boot_id,
    )
    engine = RealApptainerEngine(
        state_dir=state_root / "apptainer-engine",
        apptainer_bin=_apptainer_bin() or "apptainer",
    )
    supervisor = DockerSupervisor(
        engine=engine,
        journal=journal,
        mvd_version="0.1.0-real-apptainer-it",
    )
    executor = MvdDockerExecutor(
        journal=journal,
        boot=boot,
        store=store,
        supervisor=supervisor,
        broker=ResourceBroker(
            observer=InMemoryHostObserver(
                HostMetrics(
                    ram_free_bytes=8 * 1024**3, ram_total_bytes=16 * 1024**3
                )
            )
        ),
        state_root=state_root,
        poll_interval_seconds=poll_interval_seconds,
        max_poll_iterations=max_poll_iterations,
        accept_degraded=True,  # integration tests use local SIFs without OCI digests
    )
    return Kernel(
        KernelConfig(state_root=state_root, mvd_version="0.1.0-real-apptainer-it"),
        executor=executor,
        journal=journal,
        boot=boot,
    )


def _opts(
    *,
    image: str,
    dataset: Path,
    n_obs: int = 4,
    mem_limit: Optional[str] = None,
    command: Optional[list] = None,
    entrypoint: Optional[str] = None,
) -> dict:
    return build_executor_options(
        model_slug="real-apptainer-model",
        model_image=image,
        image_digest=None,
        dataset_slug="demo",
        dataset_path=str(dataset),
        dataset_n_obs=n_obs,
        dataset_n_vars=8,
        params={},
        manifest_text="schema_version: '1'\n",
        artifact_dir_name=f"artifact-{uuid4().hex[:8]}",
        mem_limit=mem_limit,
        container_command=command,
        container_entrypoint=entrypoint,
    )


async def _wait_for_state(
    kernel: Kernel, attempt: str, state: str, timeout: float = 30.0
) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = await kernel.query_run(physical_attempt_id=attempt)
        if last["primary_state"] == state:
            return last
        await asyncio.sleep(0.1)
    pytest.fail(
        f"attempt {attempt} did not reach {state} within {timeout}s; last={last}"
    )


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_apptainer_happy_path_reaches_artifact_success(
    tmp_path: Path, apptainer_busybox_sif: Path, apptainer_available
) -> None:
    """Happy path: SIF copies /input/data.h5mu → /output/embeddings.h5, then
    the promotion saga produces ARTIFACT_SUCCESS with a sif_digest manifest.
    """
    state_root = tmp_path / "state"
    store = StoreLayout(root=state_root / "store").ensure()
    dataset = _dataset_file(tmp_path, n_obs=4)
    kernel = _kernel_with_real_apptainer(state_root=state_root, store=store)

    async def _scenario() -> dict:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/manifest.yaml",
            options=_opts(
                image=str(apptainer_busybox_sif),
                dataset=dataset,
                n_obs=4,
                entrypoint="sh",
                command=["-c", "cp /input/data.h5mu /output/embeddings.h5"],
            ),
        )
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snap = await kernel.query_run(physical_attempt_id=attempt)
        await kernel.shutdown()
        return snap

    snap = asyncio.run(_scenario())
    assert snap["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value, snap

    artifact_dir = Path(snap["artifact_dir"])
    manifest = read_manifest(artifact_dir)
    assert manifest.physical_attempt_id
    assert {e.name for e in manifest.artifacts} >= {"embeddings.h5"}
    # Apptainer engine stamps a SIF digest into the manifest.
    assert manifest.image_identity is not None
    # runtime_image_identity carries the SIF digest when accept_degraded=True
    # and source is unverified_local — the engine stamps it via sif_digest_for().
    # No dual-digest link (no oci_digest source), but the SIF digest is present.
    if manifest.runtime_image_identity is not None:
        assert manifest.runtime_image_identity.kind is ImageIdentityKind.SIF_DIGEST


@pytest.mark.integration
def test_real_apptainer_happy_path_rebuild_index(
    tmp_path: Path, apptainer_busybox_sif: Path, apptainer_available
) -> None:
    """rebuild_index restores ARTIFACT_SUCCESS after the DB is deleted."""
    state_root = tmp_path / "state"
    store = StoreLayout(root=state_root / "store").ensure()
    dataset = _dataset_file(tmp_path, n_obs=4)
    kernel = _kernel_with_real_apptainer(state_root=state_root, store=store)

    async def _scenario() -> str:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/manifest.yaml",
            options=_opts(
                image=str(apptainer_busybox_sif),
                dataset=dataset,
                n_obs=4,
                entrypoint="sh",
                command=["-c", "cp /input/data.h5mu /output/embeddings.h5"],
            ),
        )
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        await kernel.shutdown()
        return attempt

    attempt_id = asyncio.run(_scenario())

    index_path = state_root / INDEX_FILENAME
    if index_path.exists():
        index_path.unlink()

    with open_index(index_path) as index:
        result = rebuild_index(index=index, state_root=state_root, store=store)
        row = index.get_run(attempt_id)

    assert result.artifact_success == 1
    assert row is not None
    assert row["primary_state"] == PrimaryState.ARTIFACT_SUCCESS.value
    assert row["artifact_dir"]


# ---------------------------------------------------------------------------
# 2. Container fail
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_apptainer_container_fail_produces_failed_state(
    tmp_path: Path, apptainer_busybox_sif: Path, apptainer_available
) -> None:
    """Container that exits non-zero → run ends in FAILED with a usable reason."""
    state_root = tmp_path / "state"
    store = StoreLayout(root=state_root / "store").ensure()
    dataset = _dataset_file(tmp_path, n_obs=4)
    kernel = _kernel_with_real_apptainer(state_root=state_root, store=store)

    async def _scenario() -> dict:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/manifest.yaml",
            options=_opts(
                image=str(apptainer_busybox_sif),
                dataset=dataset,
                n_obs=4,
                entrypoint="sh",
                command=["-c", "exit 42"],
            ),
        )
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snap = await kernel.query_run(physical_attempt_id=attempt)
        await kernel.shutdown()
        return snap

    snap = asyncio.run(_scenario())
    assert snap["primary_state"] == PrimaryState.FAILED.value, snap
    reason = snap.get("failure_reason") or ""
    # Must contain a usable signal about the exit code.
    assert "42" in reason or "exit" in reason.lower() or "failed" in reason.lower(), (
        f"failure_reason should mention exit code 42; got: {reason!r}"
    )


# ---------------------------------------------------------------------------
# 3. OOM (STRATEGY R1)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_apptainer_oom_produces_failed_with_oom_reason(
    tmp_path: Path,
    apptainer_oom_sif: Path,
    apptainer_available,
) -> None:
    """R1: OOM fixture killed by memory limit → FAILED with 'OOM' in reason.

    This test FAILS (does not skip) if OOM classification regresses on a host
    where cgroup v2 memory limits are enforced. If the test environment cannot
    enforce memory limits at all, the test skips with an explicit precondition
    message — do not hide a real regression behind unconditional skip.

    Preconditions for this test to pass:
    * cgroup v2 with the memory controller enabled
      (``cat /sys/fs/cgroup/cgroup.controllers`` must include ``memory``)
    * The Apptainer installation must honour ``--memory`` (either via user-ns
      cgroups or a setuid install with cgroup delegation).
    """
    # Detect cgroup v2 memory support at test time.
    cgroup_ok = Path("/sys/fs/cgroup/cgroup.controllers").is_file() and (
        "memory" in Path("/sys/fs/cgroup/cgroup.controllers").read_text()
    )
    if not cgroup_ok:
        pytest.skip(
            "cgroup v2 memory controller not available on this host; "
            "R1 OOM test requires 'memory' in /sys/fs/cgroup/cgroup.controllers"
        )

    state_root = tmp_path / "state"
    store = StoreLayout(root=state_root / "store").ensure()
    dataset = _dataset_file(tmp_path, n_obs=4)
    kernel = _kernel_with_real_apptainer(
        state_root=state_root,
        store=store,
        poll_interval_seconds=0.2,
        max_poll_iterations=150,  # 30s max
    )

    async def _scenario() -> dict:
        attempt = await kernel.submit_run(
            manifest_path="/tmp/manifest.yaml",
            options=_opts(
                image=str(apptainer_oom_sif),
                dataset=dataset,
                n_obs=4,
                mem_limit=_OOM_MEM_LIMIT,
                # No explicit command: uses the SIF's default CMD (the allocator).
            ),
        )
        await kernel._execution_tasks[attempt]  # type: ignore[attr-defined]
        snap = await kernel.query_run(physical_attempt_id=attempt)
        await kernel.shutdown()
        return snap

    snap = asyncio.run(_scenario())
    assert snap["primary_state"] == PrimaryState.FAILED.value, (
        f"OOM run should reach FAILED; got {snap['primary_state']!r}"
    )
    reason = snap.get("failure_reason") or ""
    # The engine marks oom_killed=True when exit_code==137 + mem_limit set;
    # the executor turns that into "container OOM_KILLED" in failure_reason.
    assert "OOM" in reason.upper(), (
        f"failure_reason must contain 'OOM' when cgroup kills the container; "
        f"got: {reason!r}. "
        f"If --memory is silently ignored on this kernel, classify R1 as open "
        f"and document the precondition."
    )
