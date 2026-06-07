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
    """Raised on a non-OK server response.

    Carries the structured error block from the wire so callers can branch
    on ``code`` rather than parse the human-readable message.

    Attributes:
        code: Stable machine-readable error code (e.g. ``UNKNOWN_VERB``,
            ``NOT_FOUND``, ``DISCONNECTED``).
        message: Human-readable explanation.
        details: Optional structured context for the error.
    """

    def __init__(
        self, code: str, message: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Build an error from a wire error block.

        Args:
            code: Machine-readable error code.
            message: Human-readable explanation.
            details: Optional structured context; stored as ``{}`` if omitted.
        """
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass
class RpcRequest:
    """One client-to-kernel call on the wire.

    Attributes:
        verb: One of the seven kernel verbs to invoke.
        kwargs: Keyword arguments forwarded to the kernel method.
        id: Client-supplied correlation id echoed back on the response.
    """

    verb: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def to_json(self) -> str:
        """Serialize to canonical single-line JSON.

        Keys are sorted and separators tightened so the encoding is
        deterministic (one object per line, no embedded newlines).
        """
        return json.dumps(
            {"verb": self.verb, "kwargs": self.kwargs, "id": self.id},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass
class RpcResponse:
    """One kernel-to-client reply on the wire.

    Exactly one of ``result``/``error`` is meaningful: ``error`` is set on
    failure, otherwise ``result`` carries the verb's return value. The
    stream flags drive the streaming protocol for ``stream_events``.

    Attributes:
        id: Correlation id matching the originating request.
        result: Verb return value on success.
        error: Structured error block (``code``/``message``/``details``)
            on failure; mutually exclusive with a meaningful ``result``.
        stream: True for each intermediate streamed event.
        stream_end: True on the final, value-less frame that closes a stream.
    """

    id: str
    result: Any = None
    error: Optional[Dict[str, Any]] = None
    stream: bool = False
    stream_end: bool = False

    def to_json(self) -> str:
        """Serialize to canonical single-line JSON.

        Emits ``error`` when present, otherwise ``result``; the ``stream``
        and ``stream_end`` flags are only included when true.
        """
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
    """Encode a request as a newline-terminated UTF-8 frame for the socket.

    Args:
        req: The request to serialize.

    Returns:
        The wire bytes, including the trailing ``\\n`` delimiter.
    """
    return (req.to_json() + "\n").encode("utf-8")


def decode_request(line: bytes) -> RpcRequest:
    """Parse one wire frame into a validated request.

    Args:
        line: A single JSON frame read from the socket (newline optional).

    Returns:
        The decoded request with defaults filled in for missing fields.

    Raises:
        ApiError: With code ``UNKNOWN_VERB`` if the verb is not one of the
            seven kernel verbs — rejected before any dispatch occurs.
    """
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
    """Encode a response as a newline-terminated UTF-8 frame for the socket.

    Args:
        resp: The response to serialize.

    Returns:
        The wire bytes, including the trailing ``\\n`` delimiter.
    """
    return (resp.to_json() + "\n").encode("utf-8")


def decode_response(line: bytes) -> RpcResponse:
    """Parse one wire frame into a response.

    Args:
        line: A single JSON frame read from the socket.

    Returns:
        The decoded response with stream flags coerced to bools.
    """
    data = json.loads(line)
    return RpcResponse(
        id=str(data.get("id", "")),
        result=data.get("result"),
        error=data.get("error"),
        stream=bool(data.get("stream", False)),
        stream_end=bool(data.get("stream_end", False)),
    )
