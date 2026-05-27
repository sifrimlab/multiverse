"""Checksum and atomic-write helpers.

Atomic write protocol (R4):
    1. Write ``<path>.tmp`` and ``fsync`` the file descriptor.
    2. ``rename`` ``<path>.tmp`` → ``<path>``.
    3. ``fsync`` the parent directory inode.

These primitives intentionally avoid any external dependency so the artifact
contract is usable in the simple-mode runner without pulling in the rest of
the platform.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, Union

PathLike = Union[str, os.PathLike[str], Path]

_CHUNK = 1 << 20  # 1 MiB streaming buffer; bounded RSS regardless of file size


def sha256_bytes(data: bytes) -> str:
    """Return the hex-encoded sha256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_iter(chunks: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk)
    return h.hexdigest()


def sha256_file(path: PathLike) -> str:
    """Stream a file and return its hex-encoded sha256.

    Streaming so that gigabyte-scale embeddings checksum within bounded RSS.
    """
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fp:
        while True:
            chunk = fp.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def fsync_path(path: PathLike) -> None:
    """``fsync`` a file or directory by opening it and flushing the FD.

    On platforms where directory fsync is not meaningful, the call is silently
    a no-op rather than an error; the durability claim degrades to file-only.
    See ADR 0001 §2 and R8 graded storage health.
    """
    p = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY") and p.is_dir():
        flags |= os.O_DIRECTORY
    fd = os.open(str(p), flags)
    try:
        try:
            os.fsync(fd)
        except OSError:
            # Some filesystems (notably some network mounts) refuse fsync on
            # directory FDs. The graded storage probe reports this as
            # `degraded`; the caller's durability claim is degraded but the
            # rename has still happened.
            pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: PathLike, data: bytes, *, fsync: bool = True) -> None:
    """Atomically replace the file at ``path`` with ``data``.

    Sequence: write ``<path>.tmp``, ``fsync`` file, ``rename`` over ``<path>``,
    ``fsync`` parent directory.

    Setting ``fsync=False`` is intended ONLY for tests that need to verify the
    rename sequence without paying for the disk barrier. Production callers
    must leave ``fsync`` at its default.
    """
    p = Path(path)
    parent = p.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = parent / f"{p.name}.tmp"
    # Open with O_CREAT|O_WRONLY|O_TRUNC so we never read a previous tmp.
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, data)
        if fsync:
            os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(str(tmp), str(p))
    if fsync:
        fsync_path(parent)
