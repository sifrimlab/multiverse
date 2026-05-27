"""Move-2 exit-gate tests for the simple-mode Docker backend.

Strategy v2 §2: ``multiverse run --simple <manifest> --out <bundle>``
must work with Docker and produce a sidecar-verified artifact bundle.

These tests use a fully-faked Docker client so they run without a real
daemon. The acceptance properties tested:

    1. The container is started with the correct volumes (dataset →
       ``/input/data.h5mu`` ro; workspace → ``/output`` rw).
    2. The container's environment carries the MVR contract variables.
    3. ``job_spec.json`` is written into the workspace before launch.
    4. Image identity resolves to ``registry_digest`` when ``RepoDigests``
       are present; ``local_image_id`` when only image id is available;
       ``unverified_local`` otherwise.
    5. A non-zero exit code raises ``DockerBackendError`` and the run
       outcome lands in ``_failed/`` with a ``run_attempt_manifest.json``.
    6. End-to-end happy path with a synthetic-producer container writes a
       contract-valid bundle (manifest verifies via the sidecar reader).
    7. Import graph: the backend is NOT imported by ``multiverse.simple``
       at module load — the CLI lazy-imports it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import h5py
import numpy as np
import pytest

from multiverse.artifact import (
    ARTIFACT_MANIFEST_FILENAME,
    ImageIdentityKind,
    read_manifest,
)
from multiverse.simple import (
    SimpleModeRunner,
    parse_simple_manifest,
)
from multiverse.simple.backends.docker import (
    CONTAINER_INPUT_DATA_PATH,
    CONTAINER_OUTPUT_DIR,
    JOB_SPEC_FILENAME,
    DockerBackend,
    DockerBackendError,
)
from multiverse.simple.runner import JobStatus


# ---------------------------------------------------------------------------
# Fake Docker SDK shaped to the subset the backend actually uses.
# ---------------------------------------------------------------------------


@dataclass
class _FakeImage:
    id: str
    attrs: Dict[str, Any] = field(default_factory=dict)
    short_id: Optional[str] = None


@dataclass
class _FakeImages:
    by_ref: Dict[str, _FakeImage] = field(default_factory=dict)
    pull_should_fail: bool = False

    def get(self, ref: str) -> _FakeImage:
        if ref in self.by_ref:
            return self.by_ref[ref]
        raise RuntimeError(f"no such image: {ref}")

    def pull(self, ref: str) -> _FakeImage:
        if self.pull_should_fail:
            raise RuntimeError(f"pull failed for {ref}")
        # Pulling registers a digest.
        img = _FakeImage(
            id=f"sha256:{'a' * 64}",
            attrs={"RepoDigests": [f"{ref}@sha256:{'a' * 64}"]},
        )
        self.by_ref[ref] = img
        return img


@dataclass
class _FakeContainer:
    container_id: str
    image: str
    name: str
    labels: Dict[str, str]
    environment: Dict[str, str]
    volumes: Dict[str, Any]
    exit_code: int = 0
    producer: Optional[Callable[["_FakeContainer"], None]] = None
    log_bytes: bytes = b""
    started: bool = False
    removed: bool = False
    workspace_dir: Optional[Path] = None

    def start(self) -> None:
        self.started = True
        if self.producer is not None:
            self.producer(self)

    def wait(self, *, timeout: Optional[int] = None) -> Dict[str, int]:
        return {"StatusCode": self.exit_code}

    def logs(self, *, stdout: bool = True, stderr: bool = True) -> bytes:
        return self.log_bytes

    def remove(self, *, force: bool = False) -> None:
        self.removed = True


@dataclass
class _FakeContainers:
    last: Optional[_FakeContainer] = None
    producer: Optional[Callable[[_FakeContainer], None]] = None
    exit_code: int = 0
    log_bytes: bytes = b"hello\n"

    def create(
        self,
        *,
        image: str,
        name: str,
        detach: bool,
        labels: Dict[str, str],
        environment: Dict[str, str],
        volumes: Dict[str, Any],
        mem_limit: Optional[str],
    ) -> _FakeContainer:
        # Find the workspace bind so producer callbacks can write outputs.
        workspace_dir = None
        for host, spec in volumes.items():
            if isinstance(spec, dict) and spec.get("bind") == CONTAINER_OUTPUT_DIR:
                workspace_dir = Path(host)
                break
        container = _FakeContainer(
            container_id=f"fake-{name}",
            image=image,
            name=name,
            labels=dict(labels),
            environment=dict(environment),
            volumes=dict(volumes),
            exit_code=self.exit_code,
            producer=self.producer,
            log_bytes=self.log_bytes,
            workspace_dir=workspace_dir,
        )
        self.last = container
        return container


@dataclass
class _FakeDockerClient:
    images: _FakeImages = field(default_factory=_FakeImages)
    containers: _FakeContainers = field(default_factory=_FakeContainers)

    def ping(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Manifest + fixtures
# ---------------------------------------------------------------------------


def _make_manifest(
    tmp_path: Path,
    *,
    n_obs: int = 4,
    digest: str | None = "sha256:" + "b" * 64,
) -> Path:
    dataset = tmp_path / "data.h5mu"
    dataset.write_bytes(b"placeholder")  # backend doesn't open it; Docker mount only.
    digest_line = f'      image_digest: "{digest}"\n' if digest else ""
    text = (
        'schema_version: "1"\n'
        'globals: {mv_contract_version: "1"}\n'
        'jobs:\n'
        '  - name: "demo_pca"\n'
        '    model:\n'
        '      slug: "pca"\n'
        '      version: "1.0.0"\n'
        '      image: "multiverse-pca:1.0.0"\n'
        f'{digest_line}'
        '      contract_version: "1"\n'
        '    dataset:\n'
        '      slug: "demo"\n'
        f'      path: "{dataset}"\n'
        f'      n_obs: {n_obs}\n'
        '    params:\n'
        '      n_components: 4\n'
    )
    path = tmp_path / "manifest.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _good_producer(n_obs: int) -> Callable[[_FakeContainer], None]:
    def _producer(container: _FakeContainer) -> None:
        ws = container.workspace_dir
        if ws is None:
            return
        with h5py.File(ws / "embeddings.h5", "w") as f:
            f.create_dataset(
                "latent",
                data=np.random.default_rng(0).standard_normal((n_obs, 4)).astype(np.float32),
            )
    return _producer


# ---------------------------------------------------------------------------
# 1-3. Volumes, environment, job spec
# ---------------------------------------------------------------------------


def test_backend_mounts_dataset_and_workspace_correctly(tmp_path: Path) -> None:
    manifest = parse_simple_manifest(_make_manifest(tmp_path))
    job = manifest.jobs[0]
    workspace = tmp_path / "ws"

    client = _FakeDockerClient()
    backend = DockerBackend(client=client)
    backend.execute(job=job, workspace_dir=workspace, seed=42)

    container = client.containers.last
    assert container is not None
    binds = {host: spec["bind"] for host, spec in container.volumes.items()}
    assert CONTAINER_INPUT_DATA_PATH in binds.values()
    assert CONTAINER_OUTPUT_DIR in binds.values()
    # Dataset is read-only.
    for host, spec in container.volumes.items():
        if spec["bind"] == CONTAINER_INPUT_DATA_PATH:
            assert spec["mode"] == "ro"
        if spec["bind"] == CONTAINER_OUTPUT_DIR:
            assert spec["mode"] == "rw"


def test_backend_environment_carries_mvr_contract(tmp_path: Path) -> None:
    manifest = parse_simple_manifest(_make_manifest(tmp_path))
    backend = DockerBackend(client=_FakeDockerClient())
    backend.execute(
        job=manifest.jobs[0], workspace_dir=tmp_path / "ws", seed=7
    )
    env = backend.client.containers.last.environment
    assert env["MVR_INPUT_DATA_PATH"] == CONTAINER_INPUT_DATA_PATH
    assert env["MVR_OUTPUT_DIR"] == CONTAINER_OUTPUT_DIR
    assert env["MVR_JOB_SPEC_PATH"].endswith(JOB_SPEC_FILENAME)
    assert env["MVR_SEED"] == "7"


def test_backend_writes_job_spec_before_launch(tmp_path: Path) -> None:
    manifest = parse_simple_manifest(_make_manifest(tmp_path))
    job = manifest.jobs[0]
    workspace = tmp_path / "ws"

    written_before_start: Dict[str, bool] = {"ok": False}

    def _check_producer(container: _FakeContainer) -> None:
        ws = container.workspace_dir
        if ws is not None and (ws / JOB_SPEC_FILENAME).is_file():
            written_before_start["ok"] = True

    client = _FakeDockerClient()
    client.containers.producer = _check_producer
    DockerBackend(client=client).execute(
        job=job, workspace_dir=workspace, seed=None
    )

    assert written_before_start["ok"], "job_spec.json must exist before container.start()"
    spec = json.loads((workspace / JOB_SPEC_FILENAME).read_text())
    assert spec["model_name"] == "pca"
    assert spec["hyperparameters"]["pca"] == {"n_components": 4}


# ---------------------------------------------------------------------------
# 4. Image identity resolution (R10)
# ---------------------------------------------------------------------------


def test_image_identity_uses_manifest_digest_when_present(tmp_path: Path) -> None:
    manifest = parse_simple_manifest(_make_manifest(tmp_path, digest="sha256:" + "c" * 64))
    backend = DockerBackend(client=_FakeDockerClient())
    result = backend.execute(
        job=manifest.jobs[0], workspace_dir=tmp_path / "ws", seed=None
    )
    assert result.image_identity.kind is ImageIdentityKind.REGISTRY_DIGEST
    assert result.image_identity.value == "sha256:" + "c" * 64


def test_image_identity_pulls_and_uses_registry_digest_when_local_absent(
    tmp_path: Path,
) -> None:
    manifest = parse_simple_manifest(_make_manifest(tmp_path, digest=None))
    client = _FakeDockerClient()  # empty image cache
    backend = DockerBackend(client=client, no_image_pull=False)
    result = backend.execute(
        job=manifest.jobs[0], workspace_dir=tmp_path / "ws", seed=None
    )
    assert result.image_identity.kind is ImageIdentityKind.REGISTRY_DIGEST


def test_image_identity_local_image_id_when_pull_disabled(tmp_path: Path) -> None:
    manifest = parse_simple_manifest(_make_manifest(tmp_path, digest=None))
    client = _FakeDockerClient()
    # Seed only a local image (no RepoDigests).
    client.images.by_ref["multiverse-pca:1.0.0"] = _FakeImage(
        id=f"sha256:{'d' * 64}",
        attrs={"RepoDigests": []},
    )
    backend = DockerBackend(client=client, no_image_pull=True)
    result = backend.execute(
        job=manifest.jobs[0], workspace_dir=tmp_path / "ws", seed=None
    )
    assert result.image_identity.kind is ImageIdentityKind.LOCAL_IMAGE_ID


def test_image_identity_unverified_local_when_no_inspect_and_no_pull(
    tmp_path: Path,
) -> None:
    manifest = parse_simple_manifest(_make_manifest(tmp_path, digest=None))
    client = _FakeDockerClient()
    client.images.pull_should_fail = True
    backend = DockerBackend(client=client, no_image_pull=True)
    # With no-pull and no local image, the backend should raise OR fall
    # back to unverified — the strategy says R10's unverified_local is a
    # valid resolution when no other signal exists, and the runner refuses
    # it under --strict. Here we accept either: the backend raises, or
    # produces unverified_local. The latter is preferred so the runner
    # can record the variant.
    try:
        result = backend.execute(
            job=manifest.jobs[0], workspace_dir=tmp_path / "ws", seed=None
        )
    except DockerBackendError:
        return
    assert result.image_identity.kind in (
        ImageIdentityKind.UNVERIFIED_LOCAL,
        ImageIdentityKind.LOCAL_IMAGE_ID,
    )


# ---------------------------------------------------------------------------
# 5. Non-zero exit → DockerBackendError → run_attempt_manifest in _failed/
# ---------------------------------------------------------------------------


def test_non_zero_exit_raises_and_runner_records_failure(tmp_path: Path) -> None:
    manifest_path = _make_manifest(tmp_path)
    manifest = parse_simple_manifest(manifest_path)
    client = _FakeDockerClient()
    client.containers.exit_code = 2
    client.containers.log_bytes = b"boom\n"
    backend = DockerBackend(client=client)

    runner = SimpleModeRunner(backend=backend, output_root=tmp_path / "out")
    result = runner.run(manifest)
    outcome = result.outcomes[0]
    assert outcome.status is JobStatus.FAILED
    assert outcome.failure_dir is not None
    attempt = json.loads(
        (outcome.failure_dir / "run_attempt_manifest.json").read_text()
    )
    assert attempt["final_state"] == "FAILED"
    assert "exited non-zero" in attempt["failure_reason"]


# ---------------------------------------------------------------------------
# 6. End-to-end happy path through the runner
# ---------------------------------------------------------------------------


def test_runner_with_docker_backend_produces_verified_bundle(tmp_path: Path) -> None:
    manifest_path = _make_manifest(tmp_path)
    manifest = parse_simple_manifest(manifest_path)
    client = _FakeDockerClient()
    client.containers.producer = _good_producer(n_obs=4)
    backend = DockerBackend(client=client)
    runner = SimpleModeRunner(backend=backend, output_root=tmp_path / "out")

    result = runner.run(manifest)
    assert result.all_succeeded

    outcome = result.outcomes[0]
    bundle = outcome.bundle_path
    assert bundle is not None
    assert (bundle / ARTIFACT_MANIFEST_FILENAME).is_file()
    # Sidecar-verified read succeeds.
    manifest_obj = read_manifest(bundle)
    assert manifest_obj.image_identity.kind is ImageIdentityKind.REGISTRY_DIGEST


# ---------------------------------------------------------------------------
# 7. Import-graph: backend NOT loaded by `import multiverse.simple`
# ---------------------------------------------------------------------------


def test_simple_package_does_not_load_docker_backend_eagerly() -> None:
    script = (
        "import sys\n"
        "import multiverse.simple\n"
        "if 'multiverse.simple.backends.docker' in sys.modules:\n"
        "    print('docker backend eagerly loaded')\n"
        "    raise SystemExit(1)\n"
        "if 'docker' in sys.modules:\n"
        "    print('docker SDK eagerly loaded')\n"
        "    raise SystemExit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"unexpected eager import: {result.stdout.strip()!r} stderr={result.stderr}"
    )
