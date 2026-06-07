"""Exception taxonomy for the promotion saga."""

from __future__ import annotations


class PromotionError(Exception):
    """Base class for all promotion saga failures."""


class OwnershipMismatchError(PromotionError):
    """Signal that a directory's ``.mvd_owner`` token did not match.

    Raised when a destructive or promotion operation finds a directory whose
    ``.mvd_owner`` token does not match the expected owner. Per S3 / R5 the
    kernel refuses to mutate such a directory; the user is the only authority
    that can decide what to do with it.
    """


class SymlinkPolicyError(PromotionError):
    """Signal that a symlink was found inside a managed store path.

    Raised when a symlink is discovered inside ``store/artifacts/``,
    ``store/workspaces/``, ``store/quarantine/``, or ``store/journal/``. Per
    R13 symlinks within managed store paths are forbidden as policy.
    """
