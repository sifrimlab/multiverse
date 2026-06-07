"""Unix-socket client (STRATEGY R2).

The CLI and GUI use this client to call the kernel. The client owns a
single persistent connection; requests are correlated by ``id``.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from .protocol import ApiError, RpcRequest, decode_response, encode_request


class KernelSocketClient:
    """Async client. Construct, ``await connect()``, then call any verb."""

    def __init__(self, socket_path: Path) -> None:
        """Configure the client without connecting.

        Args:
            socket_path: Filesystem path to the kernel's Unix socket
                (``${state_root}/mvd.sock``).
        """
        self._socket_path = Path(socket_path)
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def connect(self) -> None:
        """Open the persistent connection to the kernel socket."""
        self._reader, self._writer = await asyncio.open_unix_connection(
            path=str(self._socket_path)
        )

    async def close(self) -> None:
        """Close the connection, swallowing errors from an already-dead peer."""
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except OSError:
                pass
            self._writer = None
            self._reader = None

    async def __aenter__(self) -> "KernelSocketClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    # ---- verbs ----

    async def submit_run(
        self,
        *,
        manifest_path: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a manifest for execution over the socket.

        Args:
            manifest_path: Path to the run manifest describing what to run.
            options: Optional submission overrides.

        Returns:
            The ``physical_attempt_id`` of the newly created attempt.
        """
        return await self._call(
            "submit_run",
            {"manifest_path": manifest_path, "options": options or {}},
        )

    async def cancel_run(self, *, physical_attempt_id: str) -> None:
        """Request cancellation of an in-flight attempt.

        Args:
            physical_attempt_id: The attempt to cancel.
        """
        await self._call("cancel_run", {"physical_attempt_id": physical_attempt_id})

    async def query_run(self, *, physical_attempt_id: str) -> Dict[str, Any]:
        """Fetch the current state snapshot for one attempt.

        Args:
            physical_attempt_id: The attempt to query.

        Returns:
            A snapshot dict of the attempt's state and metadata.
        """
        return await self._call(
            "query_run", {"physical_attempt_id": physical_attempt_id}
        )

    async def list_runs(
        self,
        *,
        state: Optional[str] = None,
        logical_run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List run snapshots, optionally filtered.

        Args:
            state: Restrict to attempts in this state when given.
            logical_run_id: Restrict to attempts under this logical run when given.

        Returns:
            A list of run snapshot dicts matching the filters.
        """
        return await self._call(
            "list_runs", {"state": state, "logical_run_id": logical_run_id}
        )

    async def health(self) -> Dict[str, Any]:
        """Return the kernel's health/status snapshot."""
        return await self._call("health", {})

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
        await self._call(
            "report_projection_status",
            {
                "plugin": plugin,
                "physical_attempt_id": physical_attempt_id,
                "status": status,
                "details": details or {},
            },
        )

    def stream_events(
        self, *, physical_attempt_id: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """Open a live event stream for one attempt.

        Args:
            physical_attempt_id: The attempt whose events to stream.

        Returns:
            An async iterator yielding event dicts until the stream ends.
        """
        return self._stream(
            "stream_events", {"physical_attempt_id": physical_attempt_id}
        )

    # ---- internals ----

    async def _call(self, verb: str, kwargs: Dict[str, Any]) -> Any:
        """Send one request and await its single correlated response.

        Raises:
            ApiError: ``DISCONNECTED`` if the kernel closed the connection,
                or the server-supplied error code on a non-OK response.
        """
        assert self._writer is not None and self._reader is not None
        req_id = uuid.uuid4().hex
        self._writer.write(
            encode_request(RpcRequest(verb=verb, kwargs=kwargs, id=req_id))
        )
        await self._writer.drain()
        line = await self._reader.readline()
        if not line:
            raise ApiError("DISCONNECTED", "kernel closed the connection")
        response = decode_response(line)
        if response.error is not None:
            raise ApiError(
                code=str(response.error.get("code", "UNKNOWN")),
                message=str(response.error.get("message", "")),
                details=response.error.get("details") or {},
            )
        return response.result

    async def _stream(
        self, verb: str, kwargs: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Send a streaming request and yield result frames until stream end.

        Terminates on the ``stream_end`` frame or a closed connection.

        Raises:
            ApiError: With the server-supplied code if an error frame arrives.

        Yields:
            The ``result`` payload of each intermediate stream frame.
        """
        assert self._writer is not None and self._reader is not None
        req_id = uuid.uuid4().hex
        self._writer.write(
            encode_request(RpcRequest(verb=verb, kwargs=kwargs, id=req_id))
        )
        await self._writer.drain()
        while True:
            line = await self._reader.readline()
            if not line:
                return
            response = decode_response(line)
            if response.error is not None:
                raise ApiError(
                    code=str(response.error.get("code", "UNKNOWN")),
                    message=str(response.error.get("message", "")),
                )
            if response.stream_end:
                return
            yield response.result
