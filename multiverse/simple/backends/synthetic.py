"""In-process synthetic backend for fast tests and developer iteration.

A test (or a contributor adding a new model) passes a callable
``producer(workspace_dir, job)`` that writes the expected outputs into the
workspace. The synthetic backend does no Docker work; it is the foundation
for the fast Milestone-3 fixture tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ...artifact import ImageIdentity
from .base import ExecutionResult

Producer = Callable[[Path, Any], None]


@dataclass
class SyntheticBackend:
    """Backend whose ``execute`` invokes a user-supplied producer callable.

    The backend resolves image identity from the job's declared digest if
    present, otherwise falls back to ``unverified_local`` per R10's
    resolution table.
    """

    producer: Producer
    name: str = "synthetic"

    def execute(
        self,
        *,
        job: Any,
        workspace_dir: Path,
        seed: Optional[int],
    ) -> ExecutionResult:
        """Resolve image identity and invoke the producer to fill the workspace.

        Args:
            job: The job whose declared image identity is resolved.
            workspace_dir: Workspace the producer writes outputs into.
            seed: Accepted for protocol parity; the synthetic backend leaves
                seed handling to the producer.

        Returns:
            Execution result carrying the resolved image identity.
        """
        workspace_dir.mkdir(parents=True, exist_ok=True)
        identity = self._resolve_identity(job)
        self.producer(workspace_dir, job)
        return ExecutionResult(image_identity=identity)

    @staticmethod
    def _resolve_identity(job: Any) -> ImageIdentity:
        """Resolve image identity from the job's digest, else unverified_local."""
        digest = getattr(job, "image_digest", None)
        if digest:
            return ImageIdentity.registry_digest(digest)
        image = getattr(job, "model_image", "synthetic:unknown")
        return ImageIdentity.unverified_local(image)
