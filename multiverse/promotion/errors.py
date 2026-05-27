"""Promotion exception taxonomy."""

from __future__ import annotations


class PromotionError(Exception):
    """Base class for promotion-saga failures."""


class OwnershipMismatchError(PromotionError):
    """Raised when a destructive or promotion operation finds a directory
    whose ``.mvd_owner`` token does not match the expected owner.

    Per S3 / R5 the kernel refuses to mutate such a directory; the user is
    the only authority that can decide what to do with it.
    """


class SymlinkPolicyError(PromotionError):
    """Raised when a symlink is discovered inside ``store/artifacts/``,
    ``store/workspaces/``, ``store/quarantine/``, or ``store/journal/``.

    Per R13 symlinks within managed store paths are forbidden as policy.
    """
