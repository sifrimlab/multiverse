"""Kernel API — the **seven** verbs (STRATEGY R2).

The API surface is frozen at exactly seven verbs. Adding a verb requires a
strategy edit; the test in ``tests/unit/test_mvd_kernel.py`` reads this
file and asserts that exactly the seven names are present.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable

from .events import KernelEvent


# Frozen list. Order is documentation order from R2; the test asserts the
# *set* matches, so reordering is acceptable.
KERNEL_VERBS: tuple[str, ...] = (
    "submit_run",
    "cancel_run",
    "query_run",
    "list_runs",
    "stream_events",
    "health",
    "report_projection_status",
)


@runtime_checkable
class KernelAPI(Protocol):
    """The seven verbs the kernel exposes."""

    async def submit_run(
        self,
        *,
        manifest_path: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Validate the manifest, write ``JOB_INTENT`` to the journal,
        return the new ``physical_attempt_id``.

        Idempotency: a client-supplied ``options['idempotency_key']`` is
        accepted; submitting the same key returns the same id.
        """
        ...

    async def cancel_run(self, *, physical_attempt_id: str) -> None:
        """Append ``CANCEL_REQUESTED`` to the journal. Returns immediately;
        the kernel drives the saga in the background."""
        ...

    async def query_run(self, *, physical_attempt_id: str) -> Dict[str, Any]:
        """Read-only state snapshot."""
        ...

    async def list_runs(
        self,
        *,
        state: Optional[str] = None,
        logical_run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Read-only summary list."""
        ...

    def stream_events(
        self,
        *,
        physical_attempt_id: str,
    ) -> AsyncIterator[KernelEvent]:
        """Server-sent state transitions and log tail (Milestone 9 fills
        the log-tail half). Returns an async iterator; the caller drives
        it with ``async for``."""
        ...

    async def health(self) -> Dict[str, Any]:
        """Kernel self-check — no external probes (those live in
        ``multiverse doctor``)."""
        ...

    async def report_projection_status(
        self,
        *,
        plugin: str,
        physical_attempt_id: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Projection plugin reports its sync status. The kernel validates
        the plugin and status name and updates the side table."""
        ...
