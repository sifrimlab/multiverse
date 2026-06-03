"""Container engine protocol and in-memory fake (STRATEGY S7).

The supervisor talks to an object that satisfies ``ContainerEngine``. In
production this is a thin wrapper around the Docker SDK; in tests it is the
``InMemoryContainerEngine`` defined below.

Keeping Docker behind a protocol means:

* the kernel hot path's import graph is grep-checked clean of ``docker``;
* CI does not need a Docker daemon for unit tests;
* a future Podman / containerd backend is a drop-in.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (Any, Dict, Iterable, List, Optional, Protocol,
                    runtime_checkable)


class ContainerState(str, Enum):
    """Coarse state recognised by the supervisor.

    Specific engine status strings (``created``, ``running``, ``paused``,
    ``exited``) are normalised into these four buckets.
    """

    PENDING = "pending"
    RUNNING = "running"
    EXITED = "exited"
    UNKNOWN = "unknown"


@dataclass
class ContainerInfo:
    """Snapshot returned by the engine for one container."""

    container_id: str
    state: ContainerState
    labels: Dict[str, str]
    exit_code: Optional[int] = None
    oom_killed: bool = False
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    image: Optional[str] = None


@runtime_checkable
class ContainerEngine(Protocol):
    """Minimum surface the kernel needs from a container engine."""

    name: str

    def launch(
        self,
        *,
        image: str,
        command: Optional[List[str]],
        labels: Dict[str, str],
        env: Optional[Dict[str, str]],
        volumes: Optional[Dict[str, str]],
        mem_limit: Optional[str],
        name: Optional[str],
        entrypoint: Optional[str] = None,
        gpu_requested: bool = False,
    ) -> ContainerInfo: ...

    def list_by_labels(self, *, labels: Dict[str, str]) -> List[ContainerInfo]: ...

    def inspect(self, container_id: str) -> ContainerInfo: ...

    def logs(self, container_id: str) -> bytes: ...

    def stop(self, container_id: str, *, timeout: int) -> None: ...

    def kill(self, container_id: str) -> None: ...

    def remove(self, container_id: str, *, force: bool = False) -> None: ...


# ---------------------------------------------------------------------------
# Real Docker engine adapter
# ---------------------------------------------------------------------------


def _normalise_state(status: Optional[str]) -> ContainerState:
    s = (status or "").lower()
    if s in {"created", "restarting"}:
        return ContainerState.PENDING
    if s in {"running", "paused"}:
        return ContainerState.RUNNING
    if s in {"exited", "dead", "removing"}:
        return ContainerState.EXITED
    return ContainerState.UNKNOWN


def _is_not_found(exc: Exception) -> bool:
    if exc.__class__.__name__ in {"NotFound", "NoSuchContainerError"}:
        return True
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code == 404:
        return True
    if getattr(response, "status_code", None) == 404:
        return True
    return False


def _is_not_running_conflict(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) != 409:
        return False
    msg = str(exc).lower()
    return "not running" in msg or "is not running" in msg


def _docker_volumes(volumes: Optional[Dict[str, str]]) -> Dict[str, Any]:
    """Accept the supervisor's simple host->container mapping and the
    Docker SDK's expanded mapping. Dataset-like file mounts default to
    read-only; workspace-like directory mounts default to read-write.
    """
    out: Dict[str, Any] = {}
    for host, target in dict(volumes or {}).items():
        if isinstance(target, dict):
            out[str(host)] = dict(target)
            continue
        host_s = str(host)
        target_s = str(target)
        mode = (
            "ro"
            if target_s.endswith(".h5mu") or target_s.startswith("/input/")
            else "rw"
        )
        out[host_s] = {"bind": target_s, "mode": mode}
    return out


import subprocess


def gpu_available():
    try:
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


@dataclass
class RealDockerEngine:
    """Thin Docker SDK adapter implementing :class:`ContainerEngine`.

    The Docker SDK is imported lazily so importing ``multiverse.mvd`` and
    ``multiverse.docker_supervisor`` remains clean in environments without
    Docker. Construction does not contact the daemon; the first operation
    does, and failures surface as ``ContainerEngineError``.
    """

    client: Any = None
    name: str = "docker"

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
        gpu_requested: bool = False,
    ) -> ContainerInfo:
        from .errors import ContainerEngineError

        device_requests = None

        # GPU is opt-in: the run/model must explicitly request it AND a GPU
        # must be present. gpu_available() alone is only a guard against
        # requesting an unavailable device — it must not force GPU allocation
        # onto every container (issue #30).
        if gpu_requested and gpu_available():
            # Lazy import via importlib keeps the kernel import graph
            # docker-free (the grep gate forbids any `import docker` line).
            import importlib

            docker = importlib.import_module("docker")
            device_requests = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]
        try:
            kwargs = dict(
                image=image,
                command=command,
                detach=True,
                labels=dict(labels or {}),
                environment=dict(env or {}),
                volumes=_docker_volumes(volumes),
                name=name,
                mem_limit=mem_limit,
                entrypoint=entrypoint,
            )
            if device_requests is not None:
                kwargs["device_requests"] = device_requests

            container = self._client().containers.create(**kwargs)
            container.start()
            container.reload()
            return self._info(container)
        except Exception as exc:
            raise ContainerEngineError(
                f"Docker launch failed for image {image!r}: {type(exc).__name__}: {exc}"
            ) from exc

    def list_by_labels(self, *, labels: Dict[str, str]) -> List[ContainerInfo]:
        from .errors import ContainerEngineError

        filters = {"label": [f"{k}={v}" for k, v in labels.items()]}
        try:
            containers = self._client().containers.list(all=True, filters=filters)
            result: List[ContainerInfo] = []
            for container in containers:
                try:
                    container.reload()
                    result.append(self._info(container))
                except Exception as exc:
                    if _is_not_found(exc):
                        continue
                    raise
            return result
        except Exception as exc:
            raise ContainerEngineError(
                f"Docker label query failed: {type(exc).__name__}: {exc}"
            ) from exc

    def inspect(self, container_id: str) -> ContainerInfo:
        from .errors import ContainerEngineError, NoSuchContainerError

        try:
            container = self._client().containers.get(container_id)
            container.reload()
            return self._info(container)
        except Exception as exc:
            if _is_not_found(exc):
                raise NoSuchContainerError(
                    f"no such container: {container_id}"
                ) from exc
            raise ContainerEngineError(
                f"Docker inspect failed for {container_id}: {type(exc).__name__}: {exc}"
            ) from exc

    def logs(self, container_id: str) -> bytes:
        """Return the container's combined stdout/stderr as raw bytes.

        Best-effort host-side capture: callers persist this to
        ``container.log`` so a run that never reached the worker SDK (early
        crash, OOM, non-SDK image) still leaves debuggable evidence.
        """
        from .errors import ContainerEngineError, NoSuchContainerError

        try:
            container = self._client().containers.get(container_id)
            return container.logs(stdout=True, stderr=True) or b""
        except Exception as exc:
            if _is_not_found(exc):
                raise NoSuchContainerError(
                    f"no such container: {container_id}"
                ) from exc
            raise ContainerEngineError(
                f"Docker logs failed for {container_id}: {type(exc).__name__}: {exc}"
            ) from exc

    def stop(self, container_id: str, *, timeout: int) -> None:
        from .errors import ContainerEngineError, NoSuchContainerError

        try:
            self._client().containers.get(container_id).stop(timeout=timeout)
        except Exception as exc:
            if _is_not_found(exc):
                raise NoSuchContainerError(
                    f"no such container: {container_id}"
                ) from exc
            raise ContainerEngineError(
                f"Docker stop failed for {container_id}: {type(exc).__name__}: {exc}"
            ) from exc

    def kill(self, container_id: str) -> None:
        from .errors import ContainerEngineError, NoSuchContainerError

        try:
            self._client().containers.get(container_id).kill()
        except Exception as exc:
            if _is_not_found(exc):
                raise NoSuchContainerError(
                    f"no such container: {container_id}"
                ) from exc
            if _is_not_running_conflict(exc):
                return
            raise ContainerEngineError(
                f"Docker kill failed for {container_id}: {type(exc).__name__}: {exc}"
            ) from exc

    def remove(self, container_id: str, *, force: bool = False) -> None:
        from .errors import ContainerEngineError

        try:
            self._client().containers.get(container_id).remove(force=force)
        except Exception as exc:
            if _is_not_found(exc):
                return
            raise ContainerEngineError(
                f"Docker remove failed for {container_id}: {type(exc).__name__}: {exc}"
            ) from exc

    def _client(self):
        from .errors import ContainerEngineError

        if self.client is not None:
            return self.client
        try:
            import importlib

            docker = importlib.import_module("docker")
        except ImportError as exc:
            raise ContainerEngineError(
                "the docker Python package is required for mvd Docker execution"
            ) from exc
        try:
            client = docker.from_env()
            client.ping()
        except Exception as exc:
            raise ContainerEngineError(
                f"Docker daemon is not reachable: {exc}"
            ) from exc
        self.client = client
        return client

    def _info(self, container: Any) -> ContainerInfo:
        attrs = getattr(container, "attrs", {}) or {}
        state = attrs.get("State", {}) or {}
        config = attrs.get("Config", {}) or {}
        labels = dict(config.get("Labels") or {})
        status = state.get("Status") or getattr(container, "status", None)
        return ContainerInfo(
            container_id=str(getattr(container, "id", "") or attrs.get("Id", "")),
            state=_normalise_state(status),
            labels=labels,
            exit_code=state.get("ExitCode"),
            oom_killed=bool(state.get("OOMKilled", False)),
            started_at=state.get("StartedAt"),
            finished_at=state.get("FinishedAt"),
            image=(config.get("Image") or attrs.get("Image")),
        )


# ---------------------------------------------------------------------------
# In-memory fake for tests
# ---------------------------------------------------------------------------


@dataclass
class _InMemoryContainer:
    container_id: str
    image: str
    labels: Dict[str, str]
    state: ContainerState
    started_at: float
    exit_code: Optional[int] = None
    oom_killed: bool = False
    finished_at: Optional[float] = None
    env: Dict[str, str] = field(default_factory=dict)
    volumes: Dict[str, str] = field(default_factory=dict)
    mem_limit: Optional[str] = None
    name: Optional[str] = None
    gpu_requested: bool = False
    stops: int = 0
    kills: int = 0
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
            finished_at=_iso(self.finished_at) if self.finished_at else None,
            image=self.image,
        )


def _iso(t: Optional[float]) -> Optional[str]:
    if t is None:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


@dataclass
class InMemoryContainerEngine:
    """Deterministic fake engine for tests.

    ``simulate_*`` helpers let tests model real-world events (engine
    restart, OOM kill, container disappeared due to ``docker rm``) without
    a real daemon.
    """

    name: str = "in-memory"
    containers: Dict[str, _InMemoryContainer] = field(default_factory=dict)

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
        gpu_requested: bool = False,
    ) -> ContainerInfo:
        container_id = uuid.uuid4().hex
        container = _InMemoryContainer(
            container_id=container_id,
            image=image,
            labels=dict(labels or {}),
            state=ContainerState.RUNNING,
            started_at=time.time(),
            env=dict(env or {}),
            volumes=dict(volumes or {}),
            mem_limit=mem_limit,
            name=name,
            gpu_requested=gpu_requested,
        )
        self.containers[container_id] = container
        return container.to_info()

    def list_by_labels(self, *, labels: Dict[str, str]) -> List[ContainerInfo]:
        result: List[ContainerInfo] = []
        for c in self.containers.values():
            if c.removed:
                continue
            if all(c.labels.get(k) == v for k, v in labels.items()):
                result.append(c.to_info())
        return result

    def inspect(self, container_id: str) -> ContainerInfo:
        c = self.containers.get(container_id)
        if c is None or c.removed:
            from .errors import NoSuchContainerError

            raise NoSuchContainerError(f"no such container: {container_id}")
        return c.to_info()

    def logs(self, container_id: str) -> bytes:
        c = self.containers.get(container_id)
        if c is None or c.removed:
            from .errors import NoSuchContainerError

            raise NoSuchContainerError(f"no such container: {container_id}")
        return c.log_output

    def stop(self, container_id: str, *, timeout: int) -> None:
        c = self._require(container_id)
        c.stops += 1
        # Simulate a graceful exit.
        c.state = ContainerState.EXITED
        c.exit_code = 0
        c.finished_at = time.time()

    def kill(self, container_id: str) -> None:
        c = self._require(container_id)
        c.kills += 1
        c.state = ContainerState.EXITED
        c.exit_code = 137  # SIGKILL conventional exit code
        c.finished_at = time.time()

    def remove(self, container_id: str, *, force: bool = False) -> None:
        c = self.containers.get(container_id)
        if c is None:
            return
        c.removed = True

    # ---- test helpers ----

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

    def simulate_logs(self, container_id: str, output: bytes) -> None:
        """Seed the stdout/stderr bytes returned by :meth:`logs`."""
        c = self._require(container_id)
        c.log_output = output

    def simulate_docker_rm(self, container_id: str) -> None:
        """Model the user (or ``docker system prune``) removing the
        container without going through the supervisor."""
        c = self._require(container_id)
        c.removed = True

    def _require(self, container_id: str) -> _InMemoryContainer:
        c = self.containers.get(container_id)
        if c is None or c.removed:
            from .errors import NoSuchContainerError

            raise NoSuchContainerError(f"no such container: {container_id}")
        return c
