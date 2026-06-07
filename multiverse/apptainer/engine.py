"""Real Apptainer engine — subprocess-driven (STRATEGY M2).

Spawns ``apptainer exec`` as a child process and tracks the PID. The
synthetic ``container_id`` is a random hex string; ``ContainerInfo``
maps to subprocess state via the JSON sidecar.

OOM detection caveat (STRATEGY R1 — explicit risk):

Apptainer has no equivalent of Docker's ``OOMKilled`` flag. The heuristic
used here is conservative — we mark ``oom_killed`` only when:

* exit_code == 137 (SIGKILL) *and*
* a memory limit was set on launch (mem_limit non-empty), *and*
* the systemd-run wrapper (when enabled) reports the kill cause.

Without ``systemd-run --user``, mem_limit is honored only via
``--memory`` on the ``apptainer`` command itself (which Apptainer maps
to a cgroup when running under a user-namespace cgroup mount). On HPC,
the canonical answer is to run under Slurm (M4) and read OOM
classification from sacct.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..docker_supervisor.client import ContainerInfo, ContainerState
from .images import classify_image_ref, compute_sif_digest, sif_cache_path_for
from .state import ApptainerContainerRecord, ApptainerSidecar


def _iso(epoch: Optional[float]) -> Optional[str]:
    if epoch is None:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _pid_alive(pid: Optional[int]) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The PID exists; we just don't own it. Treat as alive.
        return True
    return True


@dataclass
class RealApptainerEngine:
    """Subprocess-driven Apptainer adapter for :class:`ContainerEngine`.

    Construction does *not* invoke ``apptainer``; ``available()`` and
    ``launch()`` do. The engine writes its sidecar under ``state_dir``
    (default ``$state_root/apptainer-engine``) and pulls SIFs into
    ``$state_dir/sif-cache``.
    """

    state_dir: Path
    apptainer_bin: str = "apptainer"
    use_systemd_run: bool = False
    """When True and ``systemd-run --user`` is available, wrap the
    apptainer invocation so memory limits map to a transient unit
    whose ``OOMKilled=yes`` property the engine can read on exit.
    Disabled by default; opt in once the spike from STRATEGY R1
    confirms it works on the deployment site."""
    name: str = "apptainer"

    _sidecar: ApptainerSidecar = field(init=False)
    _logs_dir: Path = field(init=False)
    _sif_cache: Path = field(init=False)
    # In-memory Popen cache keyed by container_id — enables accurate exit-code
    # capture and OOM heuristics for containers launched in the current process.
    # Entries survive only for the lifetime of this engine instance.
    _procs: Dict[str, subprocess.Popen] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.state_dir = Path(self.state_dir).expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir = self.state_dir / "logs"
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._sif_cache = self.state_dir / "sif-cache"
        self._sif_cache.mkdir(parents=True, exist_ok=True)
        self._sidecar = ApptainerSidecar.load_or_empty(
            self.state_dir / "apptainer-engine.json"
        )

    # ------------------------------------------------------------------
    # capability detection
    # ------------------------------------------------------------------

    @classmethod
    def available(cls, bin_name: str = "apptainer") -> bool:
        """True iff the ``apptainer`` binary is on PATH. Does not invoke it."""
        return (
            shutil.which(bin_name) is not None
            or shutil.which("singularity") is not None
        )

    # ------------------------------------------------------------------
    # image acquisition
    # ------------------------------------------------------------------

    def acquire_image(self, image: str, *, oci_digest: Optional[str] = None) -> Path:
        """Return a path to a SIF for ``image``.

        ``image`` may be a SIF path or any reference Apptainer can pull
        (``docker://...``, ``oras://...``, etc.). ``oci_digest``, when
        provided, is used to name the cache entry so two pulls of the
        same digest dedupe.
        """
        ref = classify_image_ref(image)
        if ref.is_local_file:
            return Path(ref.locator)
        target = sif_cache_path_for(self._sif_cache, oci_digest, ref.locator)
        if target.is_file():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        # ``apptainer pull <target> <ref>`` writes the SIF atomically when
        # the target does not exist; the binary handles its own tmp file.
        cmd = [self.apptainer_bin, "pull", str(target), ref.locator]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            from ..docker_supervisor.errors import ContainerEngineError

            raise ContainerEngineError(
                f"apptainer pull failed for {ref.locator!r}: "
                f"rc={result.returncode} stderr={result.stderr.strip()!r}"
            )
        return target

    # ------------------------------------------------------------------
    # ContainerEngine surface
    # ------------------------------------------------------------------

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
        """Acquire the SIF, spawn ``apptainer exec``, and record the sidecar.

        Args:
            image: SIF path or any reference Apptainer can pull.
            command: Argv passed to the container after the entrypoint.
            labels: Reconciliation labels; ``multiverse.image_digest``
                supplies the OCI half of the dual-digest pair and names
                the SIF cache entry.
            env: Extra environment overlaid on the engine's environment.
            volumes: Host->target bind mounts (e.g. ``/input/data.h5mu``
                read-only, ``/output`` writable).
            mem_limit: ``--memory`` value; advisory unless a user cgroup
                namespace is available (feeds the OOM heuristic).
            name: Optional human label recorded in the sidecar.
            entrypoint: Optional executable to run instead of the SIF
                default.
            gpu_requested: When True, pass ``--nv`` for GPU passthrough.

        Returns:
            The launched container's info snapshot.

        Raises:
            ContainerEngineError: If image acquisition, SIF hashing, or
                the subprocess spawn fails.
        """
        from ..docker_supervisor.errors import ContainerEngineError

        labels = dict(labels or {})
        # Pull (or accept) the SIF.
        try:
            sif_path = self.acquire_image(
                image, oci_digest=labels.get("multiverse.image_digest")
            )
        except ContainerEngineError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ContainerEngineError(
                f"apptainer acquire_image failed for {image!r}: {exc}"
            ) from exc

        try:
            sif_digest = compute_sif_digest(sif_path)
        except OSError as exc:
            raise ContainerEngineError(
                f"could not hash SIF at {str(sif_path)!r}: {exc}"
            ) from exc

        cmd = self._build_apptainer_command(
            sif_path=sif_path,
            command=command,
            volumes=volumes,
            mem_limit=mem_limit,
            entrypoint=entrypoint,
            gpu_requested=gpu_requested,
        )
        container_id = uuid.uuid4().hex
        log_path = self._logs_dir / f"{container_id}.log"
        try:
            log_file = log_path.open("ab")
            proc = subprocess.Popen(  # noqa: S603 - command list is built above
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env={**os.environ, **(env or {})},
                start_new_session=True,
            )
        except OSError as exc:
            raise ContainerEngineError(
                f"apptainer launch failed for {image!r}: {exc}"
            ) from exc

        record = ApptainerContainerRecord(
            container_id=container_id,
            pid=proc.pid,
            image=image,
            labels=labels,
            command=list(cmd),
            name=name,
            started_at=time.time(),
            log_file=str(log_path),
            sif_digest=sif_digest,
            source_oci_digest=labels.get("multiverse.image_digest"),
            mem_limit=mem_limit,
        )
        self._procs[container_id] = proc
        self._sidecar.put(record)
        return self._info(record)

    def list_by_labels(self, *, labels: Dict[str, str]) -> List[ContainerInfo]:
        """Return live snapshots of containers whose labels match ``labels``.

        Args:
            labels: Subset of labels every returned container must carry.

        Returns:
            Info snapshots for matching, non-removed containers.
        """
        # Reap exited PIDs first so the snapshot reflects current liveness,
        # not the state at last launch.
        for rec in list(self._sidecar.containers.values()):
            self._reap_if_exited(rec)
        return [self._info(rec) for rec in self._sidecar.matching_labels(labels)]

    def inspect(self, container_id: str) -> ContainerInfo:
        """Return the current info snapshot for one container.

        Args:
            container_id: Synthetic id assigned at launch.

        Returns:
            The container's info, with liveness refreshed.

        Raises:
            NoSuchContainerError: If the id is unknown or removed.
        """
        rec = self._sidecar.get(container_id)
        if rec is None or rec.removed:
            from ..docker_supervisor.errors import NoSuchContainerError

            raise NoSuchContainerError(f"no such container: {container_id}")
        self._reap_if_exited(rec)
        return self._info(rec)

    def logs(self, container_id: str) -> bytes:
        """Return the captured stdout/stderr for ``container_id``.

        The engine redirects each container's output to a per-container log
        file at launch, so the bytes are read straight off disk.
        """
        rec = self._sidecar.get(container_id)
        if rec is None or rec.removed:
            from ..docker_supervisor.errors import NoSuchContainerError

            raise NoSuchContainerError(f"no such container: {container_id}")
        log_file = getattr(rec, "log_file", None)
        if not log_file:
            return b""
        try:
            return Path(log_file).read_bytes()
        except OSError:
            return b""

    def stop(self, container_id: str, *, timeout: int) -> None:
        rec = self._require(container_id)
        if rec.pid is None or not _pid_alive(rec.pid):
            return
        try:
            os.killpg(os.getpgid(rec.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.time() + max(timeout, 1)
        while time.time() < deadline:
            if not _pid_alive(rec.pid):
                self._reap_if_exited(rec)
                return
            time.sleep(0.1)
        # Escalate.
        self.kill(container_id)

    def kill(self, container_id: str) -> None:
        rec = self._require(container_id)
        if rec.pid is None or not _pid_alive(rec.pid):
            return
        try:
            os.killpg(os.getpgid(rec.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return
        # Give the kernel a moment to flip the state.
        time.sleep(0.05)
        self._reap_if_exited(rec)

    def remove(self, container_id: str, *, force: bool = False) -> None:
        rec = self._sidecar.get(container_id)
        if rec is None:
            return
        if force and rec.pid and _pid_alive(rec.pid):
            self.kill(container_id)
        rec.removed = True
        self._sidecar.save()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def sif_digest_for(self, container_id: str) -> Optional[str]:
        rec = self._sidecar.get(container_id)
        return rec.sif_digest if rec else None

    def _require(self, container_id: str) -> ApptainerContainerRecord:
        from ..docker_supervisor.errors import NoSuchContainerError

        rec = self._sidecar.get(container_id)
        if rec is None or rec.removed:
            raise NoSuchContainerError(f"no such container: {container_id}")
        return rec

    def _reap_if_exited(self, rec: ApptainerContainerRecord) -> None:
        if rec.exit_code is not None:
            return
        if rec.pid is None:
            return
        if _pid_alive(rec.pid):
            return
        rec.finished_at = time.time()
        # For in-process launches we have the Popen object; poll() returns
        # the actual exit code without blocking.  For containers from a prior
        # boot (not in _procs) we fall back to -1 (unknown).
        proc = self._procs.get(rec.container_id)
        if proc is not None:
            rc = proc.poll()
            rec.exit_code = rc if rc is not None else -1
        else:
            rec.exit_code = -1
        # OOM heuristic: SIGKILL (137) + a memory limit was set at launch.
        if rec.exit_code == 137 and rec.mem_limit:
            rec.oom_killed = True
        self._sidecar.save()

    def _info(self, rec: ApptainerContainerRecord) -> ContainerInfo:
        if rec.exit_code is not None:
            state = ContainerState.EXITED
        elif rec.pid and _pid_alive(rec.pid):
            state = ContainerState.RUNNING
        else:
            state = ContainerState.UNKNOWN
        return ContainerInfo(
            container_id=rec.container_id,
            state=state,
            labels=dict(rec.labels),
            exit_code=rec.exit_code,
            oom_killed=rec.oom_killed,
            started_at=_iso(rec.started_at),
            finished_at=_iso(rec.finished_at),
            image=rec.image,
        )

    def _build_apptainer_command(
        self,
        *,
        sif_path: Path,
        command: Optional[List[str]],
        volumes: Optional[Dict[str, str]],
        mem_limit: Optional[str],
        entrypoint: Optional[str],
        gpu_requested: bool = False,
    ) -> List[str]:
        bin_name = (
            self.apptainer_bin if shutil.which(self.apptainer_bin) else "singularity"
        )
        argv: List[str] = [bin_name, "exec"]
        # GPU is opt-in (issue #30): only pass --nv when the run requests it.
        if gpu_requested:
            argv.append("--nv")
        for host, target in (volumes or {}).items():
            target_path = target if isinstance(target, str) else target.get("bind", "")
            argv.extend(["--bind", f"{host}:{target_path}"])
        if mem_limit:
            # Apptainer maps this to a cgroup when running under a user
            # cgroup namespace (kernel ≥ 4.6 + cgroup v2). On older
            # systems it is silently ignored — the engine documents that
            # mem_limit becomes advisory in that case.
            argv.extend(["--memory", mem_limit])
        argv.append(str(sif_path))
        if entrypoint:
            argv.append(entrypoint)
        if command:
            argv.extend(command)
        return argv

    def command_for_debug(self, **kwargs: Any) -> str:
        """Render the exact ``apptainer`` command that ``launch`` would
        produce, for doctor diagnostics."""
        cmd = self._build_apptainer_command(**kwargs)
        return " ".join(shlex.quote(p) for p in cmd)
