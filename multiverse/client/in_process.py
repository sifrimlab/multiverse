"""In-process client.

Wraps a ``Kernel`` instance and exposes the seven verbs so the GUI/CLI can
talk to it without going through the Unix socket. Used by:

* tests that don't want to spin up a socket server;
* the simple-mode runner integration when it wants kernel semantics in
  the same process;
* future GUI smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from ..mvd import Kernel
from ..mvd.events import KernelEvent


@dataclass
class InProcessClient:
    """Minimal seven-verb facade around a Kernel.

    Each method delegates straight to the kernel coroutine of the same name,
    bypassing the wire protocol. Behaviour and error semantics therefore
    match :class:`KernelSocketClient`, minus the socket transport.

    Attributes:
        kernel: The mvd kernel instance this client drives directly.
    """

    kernel: Kernel

    async def submit_run(
        self,
        *,
        manifest_path: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a manifest for execution.

        Args:
            manifest_path: Path to the run manifest describing what to run.
            options: Optional submission overrides passed through to the kernel.

        Returns:
            The ``physical_attempt_id`` of the newly created attempt.
        """
        return await self.kernel.submit_run(
            manifest_path=manifest_path, options=options
        )

    async def cancel_run(self, *, physical_attempt_id: str) -> None:
        """Request cancellation of an in-flight attempt.

        Args:
            physical_attempt_id: The attempt to cancel.
        """
        await self.kernel.cancel_run(physical_attempt_id=physical_attempt_id)

    async def query_run(self, *, physical_attempt_id: str) -> Dict[str, Any]:
        """Fetch the current state snapshot for one attempt.

        Args:
            physical_attempt_id: The attempt to query.

        Returns:
            A snapshot dict of the attempt's state and metadata.
        """
        return await self.kernel.query_run(physical_attempt_id=physical_attempt_id)

    async def list_runs(
        self,
        *,
        state: Optional[str] = None,
        logical_run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List run snapshots, optionally filtered.

        Args:
            state: Restrict to attempts in this state when given.
            logical_run_id: Restrict to attempts under this logical run
                (i.e. one group of retries/resumes) when given.

        Returns:
            A list of run snapshot dicts matching the filters.
        """
        return await self.kernel.list_runs(state=state, logical_run_id=logical_run_id)

    def stream_events(self, *, physical_attempt_id: str) -> AsyncIterator[KernelEvent]:
        """Open a live event stream for one attempt.

        Args:
            physical_attempt_id: The attempt whose events to stream.

        Returns:
            An async iterator of kernel events until the attempt terminates.
        """
        return self.kernel.stream_events(physical_attempt_id=physical_attempt_id)

    async def health(self) -> Dict[str, Any]:
        """Return the kernel's health/status snapshot."""
        return await self.kernel.health()

    async def report_projection_status(
        self,
        *,
        plugin: str,
        physical_attempt_id: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Report a projection plugin's sync outcome back to the kernel.

        Args:
            plugin: Projection plugin name (e.g. ``mlflow``).
            physical_attempt_id: The attempt the projection covers.
            status: Outcome string (e.g. ``TRACKING_SYNCED``).
            details: Optional structured context for the status.
        """
        await self.kernel.report_projection_status(
            plugin=plugin,
            physical_attempt_id=physical_attempt_id,
            status=status,
            details=details,
        )
