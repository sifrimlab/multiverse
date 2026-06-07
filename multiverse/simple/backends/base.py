"""Execution backend protocol.

Backends abstract *how* a model is invoked. The simple-mode runner does not
depend on any one backend; tests run the synthetic backend in-process,
production runs the Docker backend, and a contributor adding a new model
can wire a local-process backend without touching the bundle writer.

Per ADR 0001 §8 the kernel hot path may never depend on Docker; this
protocol keeps that boundary clean — every backend is a separate module the
runner pulls in by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from ...artifact import ImageIdentity


@dataclass
class ExecutionResult:
    """What a backend reports back after running a single job."""

    image_identity: ImageIdentity
    container_log_path: Optional[Path] = None
    model_log_path: Optional[Path] = None
    exit_code: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ExecutionBackend(Protocol):
    """Pluggable execution backend.

    A backend resolves the image identity (per R10) and runs the model so
    that its declared artifacts land in ``workspace_dir``. The runner
    validates the workspace afterwards; backends do not call validators.
    """

    name: str

    def execute(
        self,
        *,
        job: Any,  # SimpleJob; typed loose to avoid circular imports
        workspace_dir: Path,
        seed: Optional[int],
    ) -> ExecutionResult:
        """Run one job and return identity + log pointers.

        Raise ``RuntimeError`` for hard backend failures (image pull failed,
        container exited non-zero before producing artifacts, …). The
        runner classifies those as ``FAILED`` and writes a
        ``run_attempt_manifest.json`` in the failure directory.
        """
        ...
