"""Real Docker backend for simple-mode (STRATEGY v2 §2 / R10).

Honours the same model-container contract as the future ``MvdDockerExecutor``:

* ``/input/data.h5mu`` is the read-only dataset mount.
* ``/output`` is the writable workspace mount.
* ``job_spec.json`` is dropped into the workspace before launch.

Image identity is resolved per R10:

1. Try ``docker image inspect`` for the requested ref. If a registry digest
   is present in ``RepoDigests``, use ``registry_digest``.
2. Otherwise, if ``--no-image-pull`` was *not* passed, try ``client.images
   .pull(tag)`` to fetch a registry digest.
3. Otherwise, fall back to ``local_image_id`` if the image exists locally.
4. Otherwise, ``unverified_local`` — strict mode in the runner refuses this.

The Docker SDK is imported lazily so the simple-mode package's import graph
stays clean (R1 grep gate in ``test_client_cutover.py``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import docker

from ...artifact import ImageIdentity
from ..manifest import SimpleJob
from .base import ExecutionResult


CONTAINER_INPUT_DATA_PATH = "/input/data.h5mu"
CONTAINER_OUTPUT_DIR = "/output"
JOB_SPEC_FILENAME = "job_spec.json"


import subprocess

def gpu_available():
    try:
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except Exception:
        return False
class DockerBackendError(RuntimeError):
    """Raised on hard Docker failures (image missing, container non-zero
    exit, daemon unreachable)."""


@dataclass
class DockerBackend:
    """``ExecutionBackend`` that runs the model inside a real Docker
    container.

    Construction does not contact Docker; the daemon is opened on the
    first ``execute()`` call so ``--help`` and dry-run paths work without
    a daemon. Pass ``client`` to inject a custom Docker client (used by
    integration tests).
    """

    no_image_pull: bool = False
    mem_limit: Optional[str] = None
    name_prefix: str = "multiverse-simple"
    timeout_seconds: int = 60 * 60 * 6  # 6 h
    client: Any = None  # ``docker.DockerClient`` once opened
    name: str = "docker"

    # ------------------------------------------------------------------
    # ExecutionBackend surface
    # ------------------------------------------------------------------

    def execute(
        self,
        *,
        job: SimpleJob,
        workspace_dir: Path,
        seed: Optional[int],
    ) -> ExecutionResult:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        self._write_job_spec(job, workspace_dir, seed=seed)

        client = self._client()
        image_identity = self._resolve_image_identity(client, job)
        image_ref = self._image_ref_for_run(image_identity, job)

        container_log = workspace_dir / "container.log"
        labels = {
            "multiverse.simple_mode": "1",
            "multiverse.run_name": job.name,
            "multiverse.dataset_slug": job.dataset_slug,
            "multiverse.model_slug": job.model_slug,
            "multiverse.image_kind": image_identity.kind.value,
        }
        device_requests = None
         # TODO: make it from the config file
        if gpu_available():
            device_requests = [
                docker.types.DeviceRequest(
                    count=-1,
                    capabilities=[["gpu"]]
                )
            ]
        environment = self._environment_for(seed)
        volumes = self._volumes_for(job, workspace_dir)

        container_name = f"{self.name_prefix}-{job.name}"
        container = None
        try:
            kwargs = dict(
                image=image_ref,
                name=container_name,
                detach=True,
                labels=labels,
                environment=environment,
                volumes=volumes,
                mem_limit=self.mem_limit,
            )
            if device_requests is not None:
                kwargs["device_requests"] = device_requests
            container = client.containers.create(**kwargs)
            container.start()
            exit_status = container.wait(timeout=self.timeout_seconds)
            exit_code = (
                int(exit_status.get("StatusCode", exit_status.get("status_code", 0)))
                if isinstance(exit_status, dict)
                else 0
            )
            log_bytes = container.logs(stdout=True, stderr=True) or b""
            container_log.write_bytes(log_bytes)
        except Exception as exc:
            raise DockerBackendError(
                f"Docker backend failure for job {job.name!r}: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

        if exit_code != 0:
            raise DockerBackendError(
                f"container for {job.name!r} exited non-zero ({exit_code}); "
                f"see {container_log}"
            )

        return ExecutionResult(
            image_identity=image_identity,
            container_log_path=container_log,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _client(self):
        if self.client is not None:
            return self.client
        try:
            import docker  # type: ignore  # lazy import; ADR §8
        except ImportError as exc:
            raise DockerBackendError(
                "the docker Python package is required for simple-mode Docker "
                "execution; install the docker extra or use a synthetic backend"
            ) from exc
        try:
            client = docker.from_env()
            client.ping()
        except Exception as exc:
            raise DockerBackendError(
                f"Docker daemon is not reachable: {exc}"
            ) from exc
        self.client = client
        return client

    def _resolve_image_identity(self, client, job: SimpleJob) -> ImageIdentity:
        """R10 resolution: registry_digest > build_context_hash >
        local_image_id > unverified_local.

        ``build_context_hash`` is not produced here (it requires the
        build context; production ``mvd-register`` is responsible for
        stamping it). Simple-mode reaches the daemon via the manifest's
        ``image_digest`` field which trumps the live image inspect.
        """
        if job.image_digest:
            return ImageIdentity.registry_digest(job.image_digest)

        image_ref = job.model_image
        try:
            image = client.images.get(image_ref)
        except Exception:
            image = None

        if image is None and not self.no_image_pull:
            try:
                image = client.images.pull(image_ref)
            except Exception as exc:
                raise DockerBackendError(
                    f"image {image_ref!r} not found locally and pull failed: {exc}"
                ) from exc

        if image is None:
            return ImageIdentity.unverified_local(image_ref)

        # Prefer a registry digest from RepoDigests for strict-acceptable
        # identity.
        repo_digests = list(getattr(image, "attrs", {}).get("RepoDigests", []) or [])
        for entry in repo_digests:
            if "@sha256:" in entry:
                return ImageIdentity.registry_digest(entry.split("@", 1)[1])

        image_id = getattr(image, "id", None) or getattr(image, "short_id", None)
        if image_id and image_id.startswith("sha256:"):
            return ImageIdentity.local_image_id(image_id)
        if image_id:
            return ImageIdentity.local_image_id(f"sha256:{image_id}")
        return ImageIdentity.unverified_local(image_ref)

    def _image_ref_for_run(
        self, identity: ImageIdentity, job: SimpleJob
    ) -> str:
        """Choose the runtime image reference that Docker will actually
        run. Always prefer the manifest tag — digests on a registry are
        addressable, but ``image@sha256:...`` references are not always
        cached locally after a fresh pull. The tag is fine because we
        record the digest in the manifest separately.
        """
        return job.model_image

    def _write_job_spec(
        self,
        job: SimpleJob,
        workspace_dir: Path,
        *,
        seed: Optional[int],
    ) -> None:
        """Write the model-contract job spec into the workspace so the
        container's ``mvr-worker`` SDK can read it."""
        spec: Dict[str, Any] = {
            "model_name": job.model_slug,
            "model_version": job.model_version,
            "dataset_slug": job.dataset_slug,
            "dataset_path_in_container": CONTAINER_INPUT_DATA_PATH,
            "hyperparameters": {job.model_slug: dict(job.params)},
            "seed": seed,
        }
        (workspace_dir / JOB_SPEC_FILENAME).write_text(
            json.dumps(spec, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    def _environment_for(self, seed: Optional[int]) -> Dict[str, str]:
        env: Dict[str, str] = {
            "MVR_INPUT_DATA_PATH": CONTAINER_INPUT_DATA_PATH,
            "MVR_OUTPUT_DIR": CONTAINER_OUTPUT_DIR,
            "MVR_JOB_SPEC_PATH": os.path.join(CONTAINER_OUTPUT_DIR, JOB_SPEC_FILENAME),
        }
        if seed is not None:
            env["MVR_SEED"] = str(int(seed))
        return env

    def _volumes_for(self, job: SimpleJob, workspace_dir: Path) -> Dict[str, Any]:
        dataset_abs = str(Path(job.dataset_path).expanduser().resolve())
        return {
            dataset_abs: {"bind": CONTAINER_INPUT_DATA_PATH, "mode": "ro"},
            str(workspace_dir.resolve()): {"bind": CONTAINER_OUTPUT_DIR, "mode": "rw"},
        }
