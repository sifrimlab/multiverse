"""Container lease ledger (STRATEGY S7).

The supervisor holds a lease per running container, refreshed periodically.
If the kernel dies, the lease expires; on the next boot the journal is
replayed, every active lease's container is queried by label, and the
kernel either reattaches (alive) or transitions to a terminal failure
(exited while we were dead).

The ledger is an in-memory data structure backed by journal records:
``CONTAINER_LAUNCH`` opens a lease, ``STATE_TRANSITION`` to a terminal
state closes it. On boot the ledger is reconstructed from the journal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

DEFAULT_LEASE_TTL_SECONDS = 60


@dataclass
class ContainerLease:
    """One row in the lease ledger.

    ``last_renewed_monotonic_ns`` is the most recent supervisor tick that
    refreshed this lease. ``mvd_boot_id`` is the boot ID that owns it; a
    different boot reading this lease infers the prior mvd died.
    """

    physical_attempt_id: str
    container_id: str
    workspace: str
    owner_token: str
    mvd_boot_id: str
    issued_at_monotonic_ns: int
    last_renewed_monotonic_ns: int
    ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS
    closed: bool = False

    def renew(self) -> None:
        """Stamp the lease as freshly held by the current supervisor tick."""
        self.last_renewed_monotonic_ns = time.monotonic_ns()

    def close(self) -> None:
        """Mark the lease released; a closed lease is always treated expired."""
        self.closed = True

    def is_expired(self, *, now_monotonic_ns: Optional[int] = None) -> bool:
        """Report whether the lease has lapsed past its TTL.

        An expired lease on a still-running container is the signal that a
        previous mvd died holding it.

        Args:
            now_monotonic_ns: Monotonic-clock reading to compare against;
                defaults to the current time. Injectable for deterministic
                tests.

        Returns:
            ``True`` if the lease is closed or the time since the last renew
            exceeds ``ttl_seconds``.
        """
        if self.closed:
            return True
        now = now_monotonic_ns if now_monotonic_ns is not None else time.monotonic_ns()
        delta_ns = now - self.last_renewed_monotonic_ns
        return delta_ns > (self.ttl_seconds * 1_000_000_000)


@dataclass
class LeaseLedger:
    """In-memory set of container leases, keyed by physical_attempt_id.

    Rebuilt from the journal on boot: ``CONTAINER_LAUNCH`` opens a lease and
    a terminal ``STATE_TRANSITION`` closes it. Indexed by attempt id for O(1)
    lookups.

    Attributes:
        leases: Mapping of physical_attempt_id to its lease, including closed
            ones (so reconciliation can still see their last-known state).
    """

    leases: Dict[str, ContainerLease] = field(default_factory=dict)

    def open(
        self,
        *,
        physical_attempt_id: str,
        container_id: str,
        workspace: str,
        owner_token: str,
        mvd_boot_id: str,
        ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
    ) -> ContainerLease:
        """Open and register a new lease for a launched container.

        Mirrors a ``CONTAINER_LAUNCH`` journal record; on boot the ledger is
        rebuilt by replaying those records through this method.

        Args:
            physical_attempt_id: The attempt this lease covers; the ledger key.
            container_id: Engine-assigned id of the running container.
            workspace: Absolute path of the run's workspace directory.
            owner_token: Token proving this boot owns the workspace.
            mvd_boot_id: Boot id of the kernel taking the lease.
            ttl_seconds: Lifetime after which an un-renewed lease is expired.

        Returns:
            The newly created lease, also stored in the ledger.
        """
        now = time.monotonic_ns()
        lease = ContainerLease(
            physical_attempt_id=physical_attempt_id,
            container_id=container_id,
            workspace=workspace,
            owner_token=owner_token,
            mvd_boot_id=mvd_boot_id,
            issued_at_monotonic_ns=now,
            last_renewed_monotonic_ns=now,
            ttl_seconds=ttl_seconds,
        )
        self.leases[physical_attempt_id] = lease
        return lease

    def renew(self, physical_attempt_id: str) -> None:
        """Refresh the lease for an attempt; no-op if it is unknown."""
        lease = self.leases.get(physical_attempt_id)
        if lease is not None:
            lease.renew()

    def close(self, physical_attempt_id: str) -> None:
        """Release the lease for an attempt; no-op if it is unknown."""
        lease = self.leases.get(physical_attempt_id)
        if lease is not None:
            lease.close()

    def active(self) -> List[ContainerLease]:
        """Return every lease that is still open (not closed)."""
        return [l for l in self.leases.values() if not l.closed]

    def expired(
        self, *, now_monotonic_ns: Optional[int] = None
    ) -> List[ContainerLease]:
        """Return open leases whose TTL has lapsed.

        Args:
            now_monotonic_ns: Monotonic-clock reading to evaluate against;
                defaults to the current time.

        Returns:
            Open leases that are past their TTL — candidates for "prior mvd
            died holding this lease" handling on boot.
        """
        return [
            l
            for l in self.leases.values()
            if not l.closed and l.is_expired(now_monotonic_ns=now_monotonic_ns)
        ]
