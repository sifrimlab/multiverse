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
        self.last_renewed_monotonic_ns = time.monotonic_ns()

    def close(self) -> None:
        self.closed = True

    def is_expired(self, *, now_monotonic_ns: Optional[int] = None) -> bool:
        if self.closed:
            return True
        now = now_monotonic_ns if now_monotonic_ns is not None else time.monotonic_ns()
        delta_ns = now - self.last_renewed_monotonic_ns
        return delta_ns > (self.ttl_seconds * 1_000_000_000)


@dataclass
class LeaseLedger:
    """Indexed by physical_attempt_id for O(1) lookups."""

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
        lease = self.leases.get(physical_attempt_id)
        if lease is not None:
            lease.renew()

    def close(self, physical_attempt_id: str) -> None:
        lease = self.leases.get(physical_attempt_id)
        if lease is not None:
            lease.close()

    def active(self) -> List[ContainerLease]:
        return [l for l in self.leases.values() if not l.closed]

    def expired(self, *, now_monotonic_ns: Optional[int] = None) -> List[ContainerLease]:
        return [
            l
            for l in self.leases.values()
            if not l.closed and l.is_expired(now_monotonic_ns=now_monotonic_ns)
        ]
