"""Filesystem helpers used by the promotion saga.

Implements:
    * descriptor-based traversal (``open_dir_fd``) for the destructive /
      promotion boundary — per R13 promotion's pre-rename re-check is done
      against descriptor-relative paths so a TOCTOU swap cannot redirect
      it.
    * symlink policy enforcement inside managed store paths (R13 point 2).
    * cross-filesystem detection so the saga can choose between
      ``os.rename`` and a staged copy (S3 step 3).
    * a safe-relative-path helper used by registration hardening (S19) and
      by promotion to refuse path-escape attempts.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union

from ..artifact.checksums import fsync_path, sha256_file
from .errors import SymlinkPolicyError

PathLike = Union[str, os.PathLike[str], Path]


class StoreRoot:
    """Hold an ``O_DIRECTORY`` descriptor open on the store root.

    Production callers create one ``StoreRoot`` at kernel boot and pass it to
    every destructive helper; tests construct one per-temp-store. Keeping the
    descriptor open lets the saga's pre-rename re-check (R13) canonicalise
    against a fixed directory handle rather than a path that an attacker could
    swap out from under it.

    Args:
        root: Store root directory; created (with parents) if absent.
    """

    def __init__(self, root: PathLike) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "O_DIRECTORY"):
            self._fd: Optional[int] = os.open(
                str(self._root), os.O_RDONLY | os.O_DIRECTORY
            )
        else:  # pragma: no cover — non-POSIX
            self._fd = None

    @property
    def path(self) -> Path:
        """Resolved absolute path of the artifact store root."""
        return self._root

    @property
    def fd(self) -> Optional[int]:
        """Open ``O_DIRECTORY`` file descriptor, or ``None`` on non-POSIX."""
        return self._fd

    def close(self) -> None:
        """Close the held directory descriptor, if open."""
        if self._fd is not None and self._fd >= 0:
            try:
                os.close(self._fd)
            finally:
                self._fd = None

    def __enter__(self) -> "StoreRoot":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


@contextmanager
def open_dir_fd(path: PathLike) -> Iterator[int]:
    """Open ``path`` as an ``O_DIRECTORY`` descriptor for the duration of a block.

    Used by the saga's pre-rename re-check (R13 point 3) to canonicalise
    against the live filesystem at the exact moment of the destructive
    operation, closing a TOCTOU window where the path could be swapped.

    Args:
        path: Directory to open.

    Yields:
        An open directory file descriptor, closed when the block exits.
    """
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(str(path), flags)
    try:
        yield fd
    finally:
        os.close(fd)


def is_same_filesystem(a: PathLike, b: PathLike) -> bool:
    """Report whether two paths reside on the same filesystem.

    The saga uses this to choose between a same-FS atomic ``os.rename`` and a
    cross-FS staged copy. Each path is resolved to its nearest existing
    ancestor before comparing device ids, so a not-yet-created destination
    still classifies correctly.

    Args:
        a: First path.
        b: Second path.

    Returns:
        ``True`` iff both paths' nearest existing ancestors share an
        ``st_dev``.
    """
    return _stat_dev(a) == _stat_dev(b)


def _stat_dev(p: PathLike) -> int:
    """Return the device id of ``p``'s nearest existing ancestor."""
    path = Path(p)
    while True:
        if path.exists():
            return path.stat().st_dev
        parent = path.parent
        if parent == path:
            raise FileNotFoundError(f"no existing ancestor for {p}")
        path = parent


def safe_relative_path(base: PathLike, candidate: PathLike) -> Path:
    """Resolve ``candidate`` under ``base``, refusing any path escape.

    Used during registration (S19) and inside the saga's recovery branch to
    reject path-traversal attempts. Resolution follows symlinks so a link
    that points outside ``base`` is also caught.

    Args:
        base: Managed root the candidate must stay within.
        candidate: Path to resolve; relative candidates are joined onto
            ``base``.

    Returns:
        The canonical (resolved) absolute path of ``candidate``.

    Raises:
        ValueError: If the resolved path falls outside ``base``.
    """
    base_resolved = Path(base).resolve(strict=False)
    target = Path(candidate)
    if not target.is_absolute():
        target = base_resolved / target
    resolved = target.resolve(strict=False)
    try:
        resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError(f"path {candidate!r} escapes managed root {base!r}") from exc
    return resolved


def assert_no_symlinks_in_tree(root: PathLike) -> None:
    """Verify no symlink exists at or under ``root``.

    R13 forbids symlinks inside managed store paths because a swap between
    validation and use is a real TOCTOU vector. A missing ``root`` is a
    no-op.

    Args:
        root: Directory tree to scan.

    Raises:
        SymlinkPolicyError: If ``root`` is itself a symlink or any descendant
            is a symlink.
    """
    base = Path(root)
    if not base.exists():
        return
    if base.is_symlink():
        raise SymlinkPolicyError(f"{base} is itself a symlink")
    for child in base.rglob("*"):
        if child.is_symlink():
            raise SymlinkPolicyError(
                f"symlink in managed store path: {child} -> {os.readlink(child)}"
            )


# ---------------------------------------------------------------------------
# Staged copy for cross-filesystem promotion (S3 step 3)
# ---------------------------------------------------------------------------


def staged_copy_directory(
    src: PathLike,
    dst: PathLike,
    *,
    staging_token: str,
    fsync_enabled: bool = True,
) -> dict:
    """Copy a tree cross-filesystem via a staging dir, then atomic-rename onto ``dst``.

    Populates a per-attempt staging directory, fsyncs every file and
    directory, then publishes the whole tree with a single ``os.replace`` so
    ``dst`` never exists half-written (S3 step 3, atomic workspace promotion).

    The staging dir name embeds ``staging_token`` (typically the saga's
    ``physical_attempt_id``) so two retries never collide and no cleanup is
    needed on retry. Stale staging dirs from crashed attempts are reclaimed by
    Tier-1 GC (R12), not by the hot path — this package never deletes.

    Args:
        src: Source tree to copy.
        dst: Final destination path; must not already exist.
        staging_token: Per-attempt suffix making the staging dir name unique
            across retries.
        fsync_enabled: When ``True`` (default), fsync each file and directory
            and the parent before returning, trading throughput for crash
            durability.

    Returns:
        A map from each copied file's path (relative to ``src``) to its
        sha256 digest, as computed on the staged copy.

    Raises:
        FileExistsError: If ``dst`` already exists, or if the staging dir for
            this token is already present (journal/disk disagreement — a
            diagnostic event, not a hot-path fix-up).
        SymlinkPolicyError: If the source tree contains a symlink (R13).
    """
    src_path = Path(src)
    dst_path = Path(dst)
    if dst_path.exists():
        raise FileExistsError(f"staged_copy refuses to overwrite {dst_path}")
    assert_no_symlinks_in_tree(src_path)

    parent = dst_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{dst_path.name}.staging.{staging_token}"
    if staging.exists():
        # Same-token retry on the same destination — refuse loudly rather
        # than mutate. The saga's idempotency is at the *seq* level; if
        # this happens it means the journal and disk disagree, which is a
        # diagnostic event, not a hot-path fix-up.
        raise FileExistsError(
            f"staged_copy refuses to reuse existing staging dir {staging}; "
            "previous attempt with this token left scratch on disk — "
            "Tier-1 GC will reclaim it."
        )

    checksums: dict = {}
    staging.mkdir(parents=True, exist_ok=False)
    for src_file in src_path.rglob("*"):
        if src_file.is_symlink():
            raise SymlinkPolicyError(f"symlink in promotion source: {src_file}")
        rel = src_file.relative_to(src_path)
        dst_file = staging / rel
        if src_file.is_dir():
            dst_file.mkdir(parents=True, exist_ok=True)
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        if fsync_enabled:
            fd = os.open(str(dst_file), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        checksums[str(rel)] = sha256_file(dst_file)
    if fsync_enabled:
        for d in staging.rglob("*"):
            if d.is_dir():
                fsync_path(d)
        fsync_path(staging)

    os.replace(str(staging), str(dst_path))
    if fsync_enabled:
        fsync_path(parent)
    return checksums
