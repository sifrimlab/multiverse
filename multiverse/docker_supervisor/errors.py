"""Docker supervisor exception taxonomy."""

from __future__ import annotations


class SupervisorError(Exception):
    """Base class for supervisor failures."""


class ContainerEngineError(SupervisorError):
    """Underlying container engine (Docker, Podman, mock) returned an error
    the kernel cannot translate into a more specific failure."""


class NoSuchContainerError(SupervisorError):
    """The supervisor was asked about a container the engine does not
    know — typically a stale label query after the user pruned containers."""


class LeaseExpiredError(SupervisorError):
    """A lease was found expired on a still-running container. The kernel
    interprets this as "a previous mvd died holding this lease"; replay
    decides whether to reattach or to mark the run failed."""
