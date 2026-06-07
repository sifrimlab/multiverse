"""Host metrics observer protocol + in-memory fake."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class HostMetrics:
    """A snapshot of host resources at one instant."""

    ram_free_bytes: int
    ram_total_bytes: int
    disk_free_bytes_per_path: Dict[str, int] = field(default_factory=dict)
    inodes_free_per_path: Dict[str, int] = field(default_factory=dict)
    vram_free_per_gpu: Dict[int, int] = field(default_factory=dict)
    vram_total_per_gpu: Dict[int, int] = field(default_factory=dict)
    fd_open: int = 0
    fd_limit: int = 0


@dataclass(frozen=True)
class ResourceRequest:
    """What a job declares it needs at admission.

    ``ram_bytes`` is required; everything else is optional and the broker
    treats omission as "no constraint".
    """

    ram_bytes: int
    vram_bytes: int = 0
    gpu_index: Optional[int] = None
    disk_bytes_per_path: Dict[str, int] = field(default_factory=dict)


@runtime_checkable
class HostObserver(Protocol):
    """Anything the broker can ask for a current ``HostMetrics``."""

    name: str

    def observe(self) -> HostMetrics:
        """Return a fresh snapshot of host resources."""
        ...


@dataclass
class InMemoryHostObserver:
    """Deterministic observer for tests. Mutate ``current`` to simulate
    pressure and OOM events without touching the OS."""

    current: HostMetrics
    name: str = "in-memory"

    def observe(self) -> HostMetrics:
        """Return the currently configured ``current`` snapshot verbatim."""
        return self.current
