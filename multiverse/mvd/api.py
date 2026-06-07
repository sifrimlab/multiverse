"""Kernel API — the **seven** verbs (STRATEGY R2).

The API surface is frozen at exactly seven verbs. Adding a verb requires a
strategy edit; the test in ``tests/unit/test_mvd_kernel.py`` reads this
file and asserts that exactly the seven names are present.
"""

from __future__ import annotations

from typing import (Any, AsyncIterator, Dict, List, Optional, Protocol,
                    runtime_checkable)

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
        """Validate the manifest and enqueue a new run for execution.

        Writes ``JOB_INTENT`` to the journal before scheduling, so a crash
        can replay the intent idempotently. A client-supplied
        ``options['idempotency_key']`` is honoured: submitting the same key
        again returns the same id rather than starting a second attempt.

        Args:
            manifest_path: Path to the run manifest to validate and execute.
            options: Executor-specific job options; may carry
                ``idempotency_key`` to dedupe resubmissions.

        Returns:
            The ``physical_attempt_id`` of the new (or deduped) attempt.
        """
        ...

    async def cancel_run(self, *, physical_attempt_id: str) -> None:
        """Request cancellation of a run.

        Appends ``CANCEL_REQUESTED`` to the journal and returns immediately;
        the kernel drives the cancellation saga in the background.

        Args:
            physical_attempt_id: The attempt to cancel.
        """
        ...

    async def query_run(self, *, physical_attempt_id: str) -> Dict[str, Any]:
        """Return a read-only state snapshot for one attempt.

        Args:
            physical_attempt_id: The attempt to inspect.

        Returns:
            The run record's serialized snapshot.
        """
        ...

    async def list_runs(
        self,
        *,
        state: Optional[str] = None,
        logical_run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return a read-only summary list of runs.

        Args:
            state: Optional primary-state name to filter by.
            logical_run_id: Optional logical run to group attempts under.

        Returns:
            One serialized summary per matching run.
        """
        ...

    def stream_events(
        self,
        *,
        physical_attempt_id: str,
    ) -> AsyncIterator[KernelEvent]:
        """Subscribe to server-sent state transitions and log tail.

        Milestone 9 (GUI cutover) fills the log-tail half; today this carries
        state transitions only.

        Args:
            physical_attempt_id: The attempt to subscribe to.

        Returns:
            An async iterator the caller drives with ``async for``.
        """
        ...

    async def health(self) -> Dict[str, Any]:
        """Run a kernel self-check.

        Performs no external probes — those live in ``multiverse doctor``.

        Returns:
            A snapshot of kernel liveness and counters.
        """
        ...

    async def report_projection_status(
        self,
        *,
        plugin: str,
        physical_attempt_id: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a projection plugin's sync status for one attempt.

        The kernel validates the plugin and status name, then updates the
        projection side table. Projections are never the source of run truth.

        Args:
            plugin: Projection plugin name (e.g. ``"mlflow"``, ``"optuna"``).
            physical_attempt_id: The attempt whose projection status changed.
            status: One of the plugin's allowed status names.
            details: Optional free-form details about the sync.
        """
        ...
