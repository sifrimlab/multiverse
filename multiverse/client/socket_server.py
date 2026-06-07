"""Unix-domain-socket transport for the kernel (STRATEGY R2 / ADR §4).

The kernel speaks line-delimited JSON over a Unix socket mounted at
``${MULTIVERSE_STATE_ROOT}/mvd.sock``. Auth is filesystem-permission-based:
the socket is created with mode 0600 and is owned by the user running
``mvd``. No tokens, no TLS — see ADR 0001 §4.

The server is one task per connection; the kernel is single-threaded by
contract so handler tasks await the kernel's coroutines directly.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from ..mvd import Kernel
from .protocol import (ApiError, RpcRequest, RpcResponse, decode_request,
                       encode_response)

SOCKET_FILENAME = "mvd.sock"
SOCKET_MODE = 0o600


class KernelSocketServer:
    """Async Unix-socket server fronting a Kernel.

    Construction does not bind; ``start()`` creates the socket file and
    accepts connections until ``stop()`` is called.
    """

    def __init__(
        self,
        kernel: Kernel,
        *,
        socket_path: Path,
    ) -> None:
        """Configure the server without binding the socket.

        Args:
            kernel: The mvd kernel whose verbs this server exposes.
            socket_path: Filesystem path at which to create the Unix socket.
        """
        self._kernel = kernel
        self._socket_path = socket_path
        self._server: Optional[asyncio.AbstractServer] = None

    @property
    def socket_path(self) -> Path:
        """Path of the Unix socket this server binds (or will bind)."""
        return self._socket_path

    async def start(self) -> None:
        """Create the socket file and begin accepting connections.

        Removes a stale socket left by a previously dead kernel and applies
        mode 0600 so authorization is purely filesystem-permission-based.
        """
        if self._socket_path.exists():
            # A previous mvd died holding the socket file. The socket file
            # is *our* property when we are the kernel for this state root,
            # so removing a stale socket is allowed.
            self._socket_path.unlink()
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._server = await asyncio.start_unix_server(
            self._handle, path=str(self._socket_path)
        )
        os.chmod(str(self._socket_path), SOCKET_MODE)

    async def serve_forever(self) -> None:
        """Serve connections until cancelled.

        Raises:
            AssertionError: If called before :meth:`start`.
        """
        assert self._server is not None, "call start() before serve_forever()"
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        """Stop accepting connections and remove the socket file."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._socket_path.exists():
            try:
                self._socket_path.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # connection handler
    # ------------------------------------------------------------------

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Serve one connection: read frames, dispatch, reply, until EOF.

        Malformed or unknown-verb frames produce an error response and the
        loop continues — a bad request never tears down the connection.
        """
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return
                try:
                    request = decode_request(line)
                except ApiError as exc:
                    writer.write(
                        encode_response(
                            RpcResponse(
                                id="",
                                error={
                                    "code": exc.code,
                                    "message": exc.message,
                                    "details": exc.details,
                                },
                            )
                        )
                    )
                    await writer.drain()
                    continue
                except (json.JSONDecodeError, ValueError):
                    writer.write(
                        encode_response(
                            RpcResponse(
                                id="",
                                error={
                                    "code": "BAD_REQUEST",
                                    "message": "could not parse request",
                                },
                            )
                        )
                    )
                    await writer.drain()
                    continue
                await self._dispatch(request, writer)
        except (asyncio.CancelledError, ConnectionResetError):
            return
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (OSError, asyncio.CancelledError):
                pass

    async def _dispatch(
        self,
        request: RpcRequest,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Invoke a non-streaming verb and write its result or mapped error.

        Kernel exceptions are translated into wire error codes:
        ``KeyError`` -> ``NOT_FOUND``, ``ValueError`` -> ``INVALID_ARGUMENT``,
        ``RuntimeError`` -> ``INTERNAL``. Streaming verbs are routed to
        :meth:`_dispatch_stream` instead.
        """
        if request.verb == "stream_events":
            await self._dispatch_stream(request, writer)
            return
        try:
            result = await self._invoke(request)
            writer.write(encode_response(RpcResponse(id=request.id, result=result)))
        except KeyError as exc:
            writer.write(
                encode_response(
                    RpcResponse(
                        id=request.id,
                        error={
                            "code": "NOT_FOUND",
                            "message": str(exc),
                        },
                    )
                )
            )
        except ValueError as exc:
            writer.write(
                encode_response(
                    RpcResponse(
                        id=request.id,
                        error={"code": "INVALID_ARGUMENT", "message": str(exc)},
                    )
                )
            )
        except RuntimeError as exc:
            writer.write(
                encode_response(
                    RpcResponse(
                        id=request.id,
                        error={"code": "INTERNAL", "message": str(exc)},
                    )
                )
            )
        await writer.drain()

    async def _invoke(self, request: RpcRequest) -> Any:
        """Call the kernel method named by the request verb with its kwargs."""
        method = getattr(self._kernel, request.verb)
        return await method(**request.kwargs)

    async def _dispatch_stream(
        self,
        request: RpcRequest,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Stream kernel events for one attempt, closing with a stream-end frame.

        Each event is written as a ``stream`` frame; a final ``stream_end``
        frame always terminates the stream, including on cancellation.
        """
        try:
            iterator = self._kernel.stream_events(**request.kwargs)
        except KeyError as exc:
            writer.write(
                encode_response(
                    RpcResponse(
                        id=request.id,
                        error={"code": "NOT_FOUND", "message": str(exc)},
                    )
                )
            )
            await writer.drain()
            return
        try:
            async for event in iterator:
                writer.write(
                    encode_response(
                        RpcResponse(
                            id=request.id,
                            result=event.to_dict(),
                            stream=True,
                        )
                    )
                )
                await writer.drain()
        except asyncio.CancelledError:
            pass
        writer.write(encode_response(RpcResponse(id=request.id, stream_end=True)))
        await writer.drain()


@asynccontextmanager
async def serve_kernel(kernel: Kernel, *, socket_path: Path):
    """Convenience CM: start the server, yield it, stop on exit."""
    server = KernelSocketServer(kernel, socket_path=socket_path)
    await server.start()
    try:
        yield server
    finally:
        await server.stop()
