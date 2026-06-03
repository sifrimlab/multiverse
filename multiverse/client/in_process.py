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
    """Minimal seven-verb facade around a Kernel."""

    kernel: Kernel

    async def submit_run(
        self,
        *,
        manifest_path: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        return await self.kernel.submit_run(
            manifest_path=manifest_path, options=options
        )

    async def cancel_run(self, *, physical_attempt_id: str) -> None:
        await self.kernel.cancel_run(physical_attempt_id=physical_attempt_id)

    async def query_run(self, *, physical_attempt_id: str) -> Dict[str, Any]:
        return await self.kernel.query_run(physical_attempt_id=physical_attempt_id)

    async def list_runs(
        self,
        *,
        state: Optional[str] = None,
        logical_run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await self.kernel.list_runs(state=state, logical_run_id=logical_run_id)

    def stream_events(self, *, physical_attempt_id: str) -> AsyncIterator[KernelEvent]:
        return self.kernel.stream_events(physical_attempt_id=physical_attempt_id)

    async def health(self) -> Dict[str, Any]:
        return await self.kernel.health()

    async def report_projection_status(
        self,
        *,
        plugin: str,
        physical_attempt_id: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self.kernel.report_projection_status(
            plugin=plugin,
            physical_attempt_id=physical_attempt_id,
            status=status,
            details=details,
        )
