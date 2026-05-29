"""Device resolution for container entrypoints.

The orchestrator may schedule a job with ``device="cuda"`` even when the
container has not been granted GPU access (for example, because the host's
Docker daemon has no NVIDIA runtime registered). Calling ``.to("cuda")``
in that situation triggers ``torch._C._cuda_init()`` which raises
``RuntimeError: Found no NVIDIA driver``. ``resolve_device`` performs the
``torch.cuda.is_available()`` check inside the container and downgrades to
CPU with a warning when CUDA is unreachable, so model code never crashes on
a missing driver.
"""

from __future__ import annotations

from .logging import get_logger

logger = get_logger(__name__)


def resolve_device(requested: str | None, default: str = "cpu") -> str:
    device = (requested or default).strip().lower()
    if not device.startswith("cuda"):
        return device

    try:
        import torch
    except ImportError:
        logger.warning("torch not importable; falling back to CPU.")
        return "cpu"

    if not torch.cuda.is_available():
        logger.warning(
            "device=%r requested but CUDA is not available in this container "
            "(no NVIDIA driver or container started without --gpus). Falling back to CPU.",
            requested,
        )
        return "cpu"
    return device
