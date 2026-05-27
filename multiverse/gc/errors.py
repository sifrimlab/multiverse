"""GC exception taxonomy."""

from __future__ import annotations


class GcError(Exception):
    """Base class for GC failures."""


class GcGateError(GcError):
    """A deletion was refused by a gate (retention age, missing export,
    dry-run flag)."""


class NotOwnedError(GcGateError):
    """The candidate directory's ``.mvd_owner`` does not match an
    expected owner-token. Per R5 the GC plugin refuses to delete it."""
