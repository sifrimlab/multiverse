"""Simple-mode runner — STRATEGY.md Milestone 3 / R7.

Hot-path-clean: imports only the artifact-contract library and pyyaml. No
MLflow, no Optuna, no Docker (unless the explicit Docker backend is
selected), no SQLite, no GUI, no daemon kernel.

The simple-mode runner defines the artifact contract that every later phase
consumes. It is the *user* of ``multiverse.artifact`` and the *producer* of
the bundle layout that ``multiverse export-run`` will later mirror.
"""

from .backends.base import ExecutionBackend, ExecutionResult
from .backends.synthetic import SyntheticBackend
from .manifest import (SimpleJob, SimpleManifest, SimpleManifestError,
                       parse_simple_manifest)
from .runner import (JobOutcome, SimpleModeResult, SimpleModeRunner,
                     StrictModeViolation)

__all__ = [
    "ExecutionBackend",
    "ExecutionResult",
    "JobOutcome",
    "SimpleJob",
    "SimpleManifest",
    "SimpleManifestError",
    "SimpleModeResult",
    "SimpleModeRunner",
    "StrictModeViolation",
    "SyntheticBackend",
    "parse_simple_manifest",
]
