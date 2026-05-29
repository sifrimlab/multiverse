"""Slurm backend (STRATEGY M4).

Implements the Slurm-based execution path. Mode A only: the kernel
runs on a login node and dispatches each run as one ``sbatch``.

Public surface:

* :class:`SlurmEngine` — protocol the executor depends on.
* :class:`RealSlurmEngine` — production subprocess wrapper around
  ``sbatch`` / ``sacct`` / ``scancel``.
* :class:`InMemorySlurmEngine` — deterministic fake for tests.
* :class:`SlurmJobSpec` / :func:`render_sbatch_script` — config-driven
  template generation. No DSL.
* :class:`SlurmJobInfo` / :class:`SlurmJobState` — lifecycle model used
  by the executor's classification step.
"""

from __future__ import annotations

from .engine import (
    RealSlurmEngine,
    SlurmEngine,
    SlurmEngineError,
    SlurmSubmission,
)
from .fake import InMemorySlurmEngine
from .state import SlurmJobInfo, SlurmJobState, from_sacct_state
from .template import SlurmJobSpec, render_sbatch_script

__all__ = [
    "InMemorySlurmEngine",
    "RealSlurmEngine",
    "SlurmEngine",
    "SlurmEngineError",
    "SlurmJobInfo",
    "SlurmJobSpec",
    "SlurmJobState",
    "SlurmSubmission",
    "from_sacct_state",
    "render_sbatch_script",
]
