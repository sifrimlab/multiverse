"""Registration exception taxonomy."""

from __future__ import annotations


class RegistrationError(Exception):
    """Base class for registration failures."""


class PathEscapeError(RegistrationError):
    """A declared path resolves outside the configured store root after
    symlink canonicalisation.

    The error message includes the offending path so the user can fix
    their manifest.
    """


class PrivilegedRegistrationError(RegistrationError):
    """A model manifest requested a privileged Docker flag and the
    caller did not pass ``allow_elevated=True``."""
