"""Kernel client + Unix-socket transport (STRATEGY Milestone 9 / R2).

This package is what the GUI and CLI talk through. It exposes exactly the
seven verbs the kernel provides and does **nothing else**:

* it does not import Docker — the GUI never opens a Docker client;
* it does not import SQLite — the GUI does not write directly to the index;
* it does not import MLflow, Optuna, or Streamlit.

The grep gate in ``tests/unit/test_client_cutover.py`` enforces those
invariants. The Unix-socket transport handles the wire protocol; tests can
also use ``InProcessClient`` which wires a Kernel directly without going
through the socket.
"""

from .in_process import InProcessClient
from .protocol import (ApiError, RpcRequest, RpcResponse, decode_request,
                       decode_response, encode_request, encode_response)
from .socket_client import KernelSocketClient
from .socket_server import KernelSocketServer, serve_kernel

__all__ = [
    "ApiError",
    "InProcessClient",
    "KernelSocketClient",
    "KernelSocketServer",
    "RpcRequest",
    "RpcResponse",
    "decode_request",
    "decode_response",
    "encode_request",
    "encode_response",
    "serve_kernel",
]
