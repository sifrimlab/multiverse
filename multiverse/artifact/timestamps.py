"""Time discipline (STRATEGY S14 and ADR 0001 §11).

Three time surfaces are recorded for every state transition and every artifact:

* ``monotonic_ns`` — monotonic counter from the *current daemon process*. Use
  for ordering within a boot. Never compared across daemon lifetimes without
  ``mvd_boot_id``.
* ``mvd_boot_id`` — UUID generated at daemon start (or at simple-mode CLI
  start). Disambiguates monotonic counters from different processes.
* ``wall_iso`` — ISO 8601 with explicit timezone offset. Used for display and
  forensic correlation.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


def new_boot_id() -> str:
    """Generate an mvd_boot_id (UUID4)."""
    return str(uuid.uuid4())


def _now_iso() -> str:
    """Return a timezone-aware ISO 8601 string.

    Tries to honour the host's local timezone; falls back to UTC if the
    local tz cannot be resolved. The string is always offset-aware.
    """
    try:
        return datetime.now(
            tz=datetime.now(timezone.utc).astimezone().tzinfo
        ).isoformat()
    except (ValueError, OSError):  # pragma: no cover — defensive
        return datetime.now(tz=timezone.utc).isoformat()


def _local_tz_name() -> str:
    try:
        return datetime.now(timezone.utc).astimezone().tzname() or "UTC"
    except (ValueError, OSError):  # pragma: no cover — defensive
        return "UTC"


@dataclass(frozen=True)
class BootContext:
    """Process-lifetime identity surface.

    A simple-mode CLI invocation, a daemon process, and the test harness each
    create a fresh ``BootContext`` at startup. The contained ``boot_id`` is
    written into every journal record and every artifact manifest produced
    during that lifetime.
    """

    boot_id: str
    mvd_version: str
    git_commit: Optional[str] = None
    process_started_monotonic_ns: int = 0

    @classmethod
    def new(
        cls,
        mvd_version: str,
        git_commit: Optional[str] = None,
    ) -> "BootContext":
        return cls(
            boot_id=new_boot_id(),
            mvd_version=mvd_version,
            git_commit=git_commit,
            process_started_monotonic_ns=time.monotonic_ns(),
        )


def timestamp_now_struct(boot: BootContext) -> dict:
    """Return the three-surface timestamp dict written into journal records.

    The shape matches STRATEGY R3's record format::

        {"monotonic_ns": ..., "wall_iso": "...", "mvd_boot_id": "..."}
    """
    return {
        "monotonic_ns": time.monotonic_ns(),
        "wall_iso": _now_iso(),
        "mvd_boot_id": boot.boot_id,
    }


def produced_at_now(boot: BootContext) -> dict:
    """Return the ``produced_at`` struct written into ``artifact_manifest.json``.

    Per S2 the manifest's ``produced_at`` field is a struct, not a string,
    so a laptop suspend during a run produces correctly-ordered transitions
    on resume (S14 acceptance criterion).
    """
    return {
        "wall": _now_iso(),
        "monotonic_ns": time.monotonic_ns(),
        "tz": _local_tz_name(),
        "mvd_boot_id": boot.boot_id,
    }
