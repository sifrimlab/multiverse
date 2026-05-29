"""In-memory Apptainer engine fake for tests (STRATEGY M2).

Mirrors :class:`multiverse.docker_supervisor.client.InMemoryContainerEngine`
but exposes the Apptainer-specific bits the executor and manifest tests
need to assert against — most importantly the ``sif_digest`` of the
"running" container so the dual-digest manifest can be exercised
without an Apptainer binary.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..docker_supervisor.client import ContainerInfo, ContainerState
from ..docker_supervisor.errors import NoSuchContainerError


def _iso(epoch: Optional[float]) -> Optional[str]:
    if epoch is None:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _synthetic_sif_digest(image: str, oci_digest: Optional[str]) -> str:
    """Produce a deterministic SIF digest for an image string.

    Tests can rely on this being identical across runs: the same
    ``(image, oci_digest)`` pair always yields the same digest. The
    function lives here (not in production code) because the real
    engine reads the SIF bytes — this is a stand-in for tests.
    """
    h = hashlib.sha256(b"fake-sif:")
    if oci_digest:
        h.update(oci_digest.encode("utf-8"))
        h.update(b":")
    h.update(image.encode("utf-8"))
    return f"sha256:{h.hexdigest()}"


@dataclass
class _FakeContainer:
    container_id: str
    image: str
    labels: Dict[str, str]
    state: ContainerState
    started_at: float
    sif_digest: str
    source_oci_digest: Optional[str]
    exit_code: Optional[int] = None
    oom_killed: bool = False
    finished_at: Optional[float] = None
    env: Dict[str, str] = field(default_factory=dict)
    volumes: Dict[str, str] = field(default_factory=dict)
    mem_limit: Optional[str] = None
    name: Optional[str] = None
    removed: bool = False
    log_output: bytes = b""

    def to_info(self) -> ContainerInfo:
        return ContainerInfo(
            container_id=self.container_id,
            state=self.state,
            labels=dict(self.labels),
            exit_code=self.exit_code,
            oom_killed=self.oom_killed,
            started_at=_iso(self.started_at),
            finished_at=_iso(self.finished_at),
            image=self.image,
        )


@dataclass
class InMemoryApptainerEngine:
    """Deterministic fake mirroring ``InMemoryContainerEngine``."""

    name: str = "apptainer-in-memory"
    containers: Dict[str, _FakeContainer] = field(default_factory=dict)

    # ---- ContainerEngine surface ----

    def launch(
        self,
        *,
        image: str,
        command: Optional[List[str]] = None,
        labels: Optional[Dict[str, str]] = None,
        env: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, str]] = None,
        mem_limit: Optional[str] = None,
        name: Optional[str] = None,
        entrypoint: Optional[str] = None,
    ) -> ContainerInfo:
        labels = dict(labels or {})
        oci = labels.get("multiverse.image_digest")
        container_id = uuid.uuid4().hex
        container = _FakeContainer(
            container_id=container_id,
            image=image,
            labels=labels,
            state=ContainerState.RUNNING,
            started_at=time.time(),
            sif_digest=_synthetic_sif_digest(image, oci),
            source_oci_digest=oci,
            env=dict(env or {}),
            volumes=dict(volumes or {}),
            mem_limit=mem_limit,
            name=name,
        )
        self.containers[container_id] = container
        return container.to_info()

    def list_by_labels(self, *, labels: Dict[str, str]) -> List[ContainerInfo]:
        out: List[ContainerInfo] = []
        for c in self.containers.values():
            if c.removed:
                continue
            if all(c.labels.get(k) == v for k, v in labels.items()):
                out.append(c.to_info())
        return out

    def inspect(self, container_id: str) -> ContainerInfo:
        c = self.containers.get(container_id)
        if c is None or c.removed:
            raise NoSuchContainerError(f"no such container: {container_id}")
        return c.to_info()

    def logs(self, container_id: str) -> bytes:
        c = self.containers.get(container_id)
        if c is None or c.removed:
            raise NoSuchContainerError(f"no such container: {container_id}")
        return c.log_output

    def stop(self, container_id: str, *, timeout: int) -> None:
        c = self._require(container_id)
        c.state = ContainerState.EXITED
        c.exit_code = 0
        c.finished_at = time.time()

    def kill(self, container_id: str) -> None:
        c = self._require(container_id)
        c.state = ContainerState.EXITED
        c.exit_code = 137
        c.finished_at = time.time()

    def remove(self, container_id: str, *, force: bool = False) -> None:
        c = self.containers.get(container_id)
        if c is None:
            return
        c.removed = True

    # ---- helpers ----

    def sif_digest_for(self, container_id: str) -> Optional[str]:
        c = self.containers.get(container_id)
        return c.sif_digest if c else None

    def source_oci_digest_for(self, container_id: str) -> Optional[str]:
        c = self.containers.get(container_id)
        return c.source_oci_digest if c else None

    def simulate_natural_exit(
        self,
        container_id: str,
        *,
        exit_code: int = 0,
        oom_killed: bool = False,
    ) -> None:
        c = self._require(container_id)
        c.state = ContainerState.EXITED
        c.exit_code = exit_code
        c.oom_killed = oom_killed
        c.finished_at = time.time()

    def _require(self, container_id: str) -> _FakeContainer:
        c = self.containers.get(container_id)
        if c is None or c.removed:
            raise NoSuchContainerError(f"no such container: {container_id}")
        return c
