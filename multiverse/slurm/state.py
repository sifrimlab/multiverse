"""Slurm job state model (STRATEGY M4).

Slurm reports a free-form state string per job (``sacct --format=State``).
The strings we care about are a small enumerated set; everything else is
collapsed to ``UNKNOWN`` and treated as a terminal failure by the
executor (we never optimistically retry on an unrecognized state).

The ``is_terminal`` / ``is_failure`` properties drive the executor's
classification step as set-membership lookups rather than a branching
ladder.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SlurmJobState(str, Enum):
    """Subset of Slurm job states the executor recognizes.

    The values are the literal strings emitted by ``sacct`` (case-
    sensitive); use :func:`from_sacct_state` to parse.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    NODE_FAIL = "NODE_FAIL"
    PREEMPTED = "PREEMPTED"
    BOOT_FAIL = "BOOT_FAIL"
    DEADLINE = "DEADLINE"
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal(self) -> bool:
        """True for any state other than ``PENDING``/``RUNNING``."""
        return self not in _NON_TERMINAL

    @property
    def is_failure(self) -> bool:
        """True for terminal states the executor treats as a failure.

        ``UNKNOWN`` is included: an unparseable state is never
        optimistically retried.
        """
        return self in _FAILURE


_NON_TERMINAL = frozenset({SlurmJobState.PENDING, SlurmJobState.RUNNING})

_FAILURE = frozenset(
    {
        SlurmJobState.FAILED,
        SlurmJobState.CANCELLED,
        SlurmJobState.TIMEOUT,
        SlurmJobState.OUT_OF_MEMORY,
        SlurmJobState.NODE_FAIL,
        SlurmJobState.PREEMPTED,
        SlurmJobState.BOOT_FAIL,
        SlurmJobState.DEADLINE,
        SlurmJobState.UNKNOWN,
    }
)


def from_sacct_state(raw: str) -> SlurmJobState:
    """Parse a raw ``sacct`` State string into a :class:`SlurmJobState`.

    Slurm sometimes annotates states like ``CANCELLED by 1001`` or
    appends ``+`` for derived array jobs; we strip the suffix and look
    up the prefix. Unknown strings collapse to ``UNKNOWN`` rather than
    raising — a job whose state we cannot parse is treated as a terminal
    failure by the executor.
    """
    token = (raw or "").strip().split()[0] if raw else ""
    token = token.rstrip("+")
    try:
        return SlurmJobState(token)
    except ValueError:
        return SlurmJobState.UNKNOWN


@dataclass(frozen=True)
class SlurmJobInfo:
    """One observation of a Slurm job's lifecycle."""

    job_id: str
    state: SlurmJobState
    exit_code: Optional[int] = None
    reason: Optional[str] = None
    """Free-form Slurm reason string (e.g. ``oom-kill``, ``NODE_FAIL``);
    surfaced into the failure message but not load-bearing for state
    classification."""

    @property
    def is_terminal(self) -> bool:
        """True when the observed state is terminal."""
        return self.state.is_terminal

    @property
    def oom_killed(self) -> bool:
        """True when Slurm classified the job as out-of-memory."""
        return self.state is SlurmJobState.OUT_OF_MEMORY
