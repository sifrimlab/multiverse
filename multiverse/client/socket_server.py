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
from .protocol import (
    ApiError,
    RpcRequest,
    RpcResponse,
    decode_request,
    encode_response,
)


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
        self._kernel = kernel
        self._socket_path = socket_path
        self._server: Optional[asyncio.AbstractServer] = None

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    async def start(self) -> None:
        if self._socket_path.exists():
            # The previous mvd died holding the socket file. Per the spec
            # the socket file is *our* property when we're the kernel for
            # this state root, so removing a stale socket is allowed.
            self._socket_path.unlink()
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._server = await asyncio.start_unix_server(
            self._handle, path=str(self._socket_path)
        )
        os.chmod(str(self._socket_path), SOCKET_MODE)

    async def serve_forever(self) -> None:
        assert self._server is not None, "call start() before serve_forever()"
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
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
        method = getattr(self._kernel, request.verb)
        return await method(**request.kwargs)

    async def _dispatch_stream(
        self,
        request: RpcRequest,
        writer: asyncio.StreamWriter,
    ) -> None:
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
        writer.write(
            encode_response(RpcResponse(id=request.id, stream_end=True))
        )
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
