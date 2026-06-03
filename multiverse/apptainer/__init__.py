"""Apptainer (Singularity) backend (STRATEGY M2).

Implements :class:`multiverse.docker_supervisor.client.ContainerEngine`
against the Apptainer CLI. Two adapters:

* :class:`RealApptainerEngine` — subprocess-driven; spawns ``apptainer
  exec`` and tracks the child PID. Sidecar JSON records ``labels`` and
  liveness so the supervisor's label-based reconciliation still works.
* :class:`InMemoryApptainerEngine` — deterministic fake for tests,
  mirroring the shape of ``InMemoryContainerEngine``.

The Apptainer-specific image-acquisition logic (``apptainer pull`` for
OCI refs, SIF passthrough for local files) lives in :mod:`.images`.
"""

from __future__ import annotations

from .engine import RealApptainerEngine
from .fake import InMemoryApptainerEngine
from .images import ApptainerImageRef, classify_image_ref, compute_sif_digest

__all__ = [
    "ApptainerImageRef",
    "InMemoryApptainerEngine",
    "RealApptainerEngine",
    "classify_image_ref",
    "compute_sif_digest",
]
