"""Wire protocol — line-delimited JSON for kernel ↔ client.

One JSON object per line, ``utf-8`` encoded. Every request carries a
``verb``, a ``kwargs`` dict, and a client-supplied ``id`` so responses can
be correlated. Server responses carry the same ``id``, plus either a
``result`` or an ``error`` block.

Streaming verbs (``stream_events``) emit responses with ``stream: true``
until terminated with ``stream_end: true``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..mvd.api import KERNEL_VERBS


class ApiError(Exception):
    """Raised on a non-OK server response."""

    def __init__(
        self, code: str, message: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass
class RpcRequest:
    verb: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {"verb": self.verb, "kwargs": self.kwargs, "id": self.id},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass
class RpcResponse:
    id: str
    result: Any = None
    error: Optional[Dict[str, Any]] = None
    stream: bool = False
    stream_end: bool = False

    def to_json(self) -> str:
        payload: Dict[str, Any] = {"id": self.id}
        if self.error is not None:
            payload["error"] = self.error
        else:
            payload["result"] = self.result
        if self.stream:
            payload["stream"] = True
        if self.stream_end:
            payload["stream_end"] = True
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )


def encode_request(req: RpcRequest) -> bytes:
    return (req.to_json() + "\n").encode("utf-8")


def decode_request(line: bytes) -> RpcRequest:
    data = json.loads(line)
    verb = str(data.get("verb", ""))
    if verb not in KERNEL_VERBS:
        raise ApiError(
            "UNKNOWN_VERB",
            f"verb {verb!r} not in the seven-verb kernel API",
            {"verb": verb, "known": list(KERNEL_VERBS)},
        )
    return RpcRequest(
        verb=verb,
        kwargs=dict(data.get("kwargs") or {}),
        id=str(data.get("id", "")),
    )


def encode_response(resp: RpcResponse) -> bytes:
    return (resp.to_json() + "\n").encode("utf-8")


def decode_response(line: bytes) -> RpcResponse:
    data = json.loads(line)
    return RpcResponse(
        id=str(data.get("id", "")),
        result=data.get("result"),
        error=data.get("error"),
        stream=bool(data.get("stream", False)),
        stream_end=bool(data.get("stream_end", False)),
    )
