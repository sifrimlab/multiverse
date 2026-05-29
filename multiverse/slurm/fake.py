"""In-memory Slurm engine for tests (STRATEGY M4).

Mirrors :class:`multiverse.docker_supervisor.client.InMemoryContainerEngine`
in shape: ``submit`` stages a job in PENDING, ``simulate_*`` helpers
move it through the lifecycle deterministically, and ``query`` reports
the current observation.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .engine import SlurmSubmission
from .state import SlurmJobInfo, SlurmJobState
from .template import SlurmJobSpec, render_sbatch_script


@dataclass
class _Job:
    job_id: str
    spec: SlurmJobSpec
    state: SlurmJobState = SlurmJobState.PENDING
    exit_code: Optional[int] = None
    reason: Optional[str] = None
    script_path: Optional[Path] = None
    cancelled: bool = False


@dataclass
class InMemorySlurmEngine:
    """Deterministic Slurm engine fake.

    The fake assigns sequential numeric job ids starting at ``next_id``
    so multiple submits from one test produce predictable values.
    ``simulate_running`` / ``simulate_completed`` / ``simulate_failed`` /
    ``simulate_oom`` / ``simulate_timeout`` drive jobs through their
    states; tests call them at the points where they want the kernel to
    observe a transition.
    """

    name: str = "slurm-in-memory"
    next_id: int = 1
    jobs: Dict[str, _Job] = field(default_factory=dict)
    submit_count: int = 0
    """Total number of ``submit`` calls; tests use this to assert that
    the executor did not flood ``sbatch`` past ``max_inflight``."""

    _id_iter: Optional[itertools.count] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._id_iter = itertools.count(start=self.next_id)

    # ---- protocol surface -------------------------------------------

    def submit(self, spec: SlurmJobSpec, *, script_dir: Path) -> SlurmSubmission:
        assert self._id_iter is not None
        self.submit_count += 1
        job_id = str(next(self._id_iter))
        script_dir = Path(script_dir)
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / f"{spec.job_name}.sbatch"
        script_path.write_text(render_sbatch_script(spec), encoding="utf-8")
        self.jobs[job_id] = _Job(
            job_id=job_id, spec=spec, script_path=script_path
        )
        return SlurmSubmission(job_id=job_id, script_path=script_path)

    def query(self, job_id: str) -> SlurmJobInfo:
        job = self.jobs.get(str(job_id))
        if job is None:
            # The real engine would surface as PENDING for an unknown
            # accounting entry; mirror that here so the executor's
            # polling loop doesn't crash on a missing job.
            return SlurmJobInfo(
                job_id=str(job_id), state=SlurmJobState.PENDING
            )
        return SlurmJobInfo(
            job_id=job.job_id,
            state=job.state,
            exit_code=job.exit_code,
            reason=job.reason,
        )

    def cancel(self, job_id: str) -> None:
        job = self.jobs.get(str(job_id))
        if job is None:
            return
        job.cancelled = True
        if not job.state.is_terminal:
            job.state = SlurmJobState.CANCELLED
            job.exit_code = 0
            job.reason = "cancelled"

    def sif_digest_for_submission(self, spec: SlurmJobSpec) -> Optional[str]:
        """Deterministic synthetic digest — mirrors InMemoryApptainerEngine's
        convention so unit tests can assert the dual-digest fields without
        touching the filesystem."""
        raw = b"fake-sif:" + str(spec.image_sif).encode()
        return "sha256:" + hashlib.sha256(raw).hexdigest()

    # ---- test helpers ------------------------------------------------

    def simulate_running(self, job_id: str) -> None:
        self._mutate(job_id, state=SlurmJobState.RUNNING)

    def simulate_completed(self, job_id: str, *, exit_code: int = 0) -> None:
        self._mutate(
            job_id,
            state=SlurmJobState.COMPLETED,
            exit_code=exit_code,
        )

    def simulate_failed(
        self, job_id: str, *, exit_code: int = 1, reason: Optional[str] = None
    ) -> None:
        self._mutate(
            job_id,
            state=SlurmJobState.FAILED,
            exit_code=exit_code,
            reason=reason,
        )

    def simulate_oom(self, job_id: str) -> None:
        self._mutate(
            job_id,
            state=SlurmJobState.OUT_OF_MEMORY,
            exit_code=137,
            reason="oom-kill",
        )

    def simulate_timeout(self, job_id: str) -> None:
        self._mutate(
            job_id,
            state=SlurmJobState.TIMEOUT,
            exit_code=124,
            reason="time limit",
        )

    def simulate_node_fail(self, job_id: str) -> None:
        self._mutate(
            job_id,
            state=SlurmJobState.NODE_FAIL,
            exit_code=None,
            reason="node failure",
        )

    # ---- internals ---------------------------------------------------

    def _mutate(
        self,
        job_id: str,
        *,
        state: SlurmJobState,
        exit_code: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> None:
        job = self.jobs.get(str(job_id))
        if job is None:
            raise KeyError(f"unknown job_id {job_id!r}")
        job.state = state
        if exit_code is not None or state.is_terminal:
            job.exit_code = exit_code
        if reason is not None:
            job.reason = reason
