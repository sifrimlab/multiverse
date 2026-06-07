"""Quarantine and tombstone (STRATEGY S4 / R5).

When the promotion saga refuses to promote — bad validator, ownership
mismatch, mid-saga abort — the half-built artifact directory is moved into
``store/quarantine/<date>/<physical_attempt_id>/`` and a tombstone is left
at the original path so users following the original location can see what
happened.

The recovery subsystem (S4) never deletes; this module is its only mutation
power, and it is strictly *move + tombstone*. Deletion is reserved to
``multiverse gc`` (R12 / Milestone 12).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..artifact.checksums import atomic_write_bytes, fsync_path
from .errors import OwnershipMismatchError
from .layout import StoreLayout
from .tokens import OwnerTokenFile, read_owner_token

QUARANTINE_REPORT_FILENAME = "QUARANTINE_REPORT.md"
TOMBSTONE_SUFFIX = ".quarantined"


@dataclass(frozen=True)
class QuarantineReport:
    """Record of a single quarantine move: where a directory went and why.

    Attributes:
        source_path: Original path the directory occupied before the move.
        quarantine_path: Destination under ``store/quarantine/<date>/<id>/``.
        reason: Human-readable cause of the quarantine.
        physical_attempt_id: Attempt that owned the moved directory, if known.
        tombstone_path: Marker left at ``source_path`` pointing at the move.
        owner_token: ``.mvd_owner`` token found at the source, if any.
    """

    source_path: Path
    quarantine_path: Path
    reason: str
    physical_attempt_id: Optional[str]
    tombstone_path: Path
    owner_token: Optional[OwnerTokenFile]

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation for journaling and GUI surfacing."""
        return {
            "source_path": str(self.source_path),
            "quarantine_path": str(self.quarantine_path),
            "reason": self.reason,
            "physical_attempt_id": self.physical_attempt_id,
            "tombstone_path": str(self.tombstone_path),
            "owner_token": (
                {
                    "owner_token": self.owner_token.owner_token,
                    "physical_attempt_id": self.owner_token.physical_attempt_id,
                    "purpose": self.owner_token.purpose,
                }
                if self.owner_token is not None
                else None
            ),
        }


def _today_partition() -> str:
    """Return today's UTC date as the ``YYYY-MM-DD`` quarantine partition."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def quarantine_directory(
    *,
    source: Path,
    layout: StoreLayout,
    reason: str,
    expected_owner_token: Optional[str] = None,
    physical_attempt_id: Optional[str] = None,
    extra_report: str = "",
) -> QuarantineReport:
    """Move ``source`` into the quarantine tree and drop a tombstone.

    The recovery subsystem never deletes; this move-plus-tombstone is its only
    mutation power (S4 / R5). The directory lands at
    ``store/quarantine/<date>/<attempt_id>/`` (uniquified if that path already
    exists, since quarantined data is never overwritten), a report is written
    beside it, and a ``.quarantined`` tombstone is left at the original path so
    users following the old location can see what happened.

    Args:
        source: Directory to quarantine.
        layout: Store layout supplying the quarantine root; ``ensure()``-d.
        reason: Human-readable cause, recorded in the report and tombstone.
        expected_owner_token: If given, the source's ``.mvd_owner`` token must
            match; otherwise the move is refused (R5).
        physical_attempt_id: Attempt id for the destination directory name;
            falls back to the source token's id, then ``"unknown-attempt"``.
        extra_report: Optional free-text appended to the report's notes.

    Returns:
        A ``QuarantineReport`` describing the move; the caller is expected to
        journal it and surface it via the GUI.

    Raises:
        FileNotFoundError: If ``source`` does not exist.
        OwnershipMismatchError: If ``expected_owner_token`` is supplied and the
            source token is absent or does not match.
    """
    layout.ensure()
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"quarantine source does not exist: {source}")

    token = read_owner_token(source_path)
    if expected_owner_token is not None:
        if token is None or token.owner_token != expected_owner_token:
            raise OwnershipMismatchError(
                f"refusing to quarantine {source_path}: "
                f"expected owner token {expected_owner_token!r}, "
                f"found {token.owner_token if token else None!r}"
            )

    attempt_id = (
        physical_attempt_id
        or (token.physical_attempt_id if token is not None else None)
        or "unknown-attempt"
    )

    date_dir = layout.quarantine / _today_partition()
    date_dir.mkdir(parents=True, exist_ok=True)
    destination = date_dir / attempt_id
    # If the destination already exists (a same-attempt re-run), uniquify
    # rather than collide. The kernel never overwrites quarantined data.
    if destination.exists():
        suffix = datetime.now(timezone.utc).strftime("%H%M%S%f")
        destination = date_dir / f"{attempt_id}.{suffix}"

    os.replace(str(source_path), str(destination))
    fsync_path(date_dir)

    report_body = _render_report(
        source=source_path,
        destination=destination,
        reason=reason,
        token=token,
        extra=extra_report,
    )
    atomic_write_bytes(destination / QUARANTINE_REPORT_FILENAME, report_body)

    tombstone = source_path.with_suffix(source_path.suffix + TOMBSTONE_SUFFIX)
    tombstone_payload = json.dumps(
        {
            "quarantined_to": str(destination),
            "reason": reason,
            "physical_attempt_id": attempt_id,
            "at": datetime.now(timezone.utc).astimezone().isoformat(),
        },
        sort_keys=True,
        indent=2,
    ).encode("utf-8")
    atomic_write_bytes(tombstone, tombstone_payload)

    return QuarantineReport(
        source_path=source_path,
        quarantine_path=destination,
        reason=reason,
        physical_attempt_id=attempt_id,
        tombstone_path=tombstone,
        owner_token=token,
    )


def _render_report(
    *,
    source: Path,
    destination: Path,
    reason: str,
    token: Optional[OwnerTokenFile],
    extra: str,
) -> bytes:
    """Render the human-readable ``QUARANTINE_REPORT.md`` body as UTF-8 bytes."""
    lines = [
        "# Quarantine report",
        "",
        f"- **source:** `{source}`",
        f"- **quarantined to:** `{destination}`",
        f"- **reason:** {reason}",
        f"- **time (UTC):** {datetime.now(timezone.utc).isoformat()}",
    ]
    if token is not None:
        lines += [
            "",
            "## Owner token",
            f"- owner_token: `{token.owner_token}`",
            f"- physical_attempt_id: `{token.physical_attempt_id}`",
            f"- created: `{token.created_wall_iso}`",
            f"- purpose: `{token.purpose}`",
        ]
    else:
        lines += ["", "## Owner token", "_no `.mvd_owner` was found_"]
    if extra:
        lines += ["", "## Notes", extra.strip()]
    lines += [
        "",
        "## Next steps",
        "- Use `multiverse runs adopt-quarantine <id>` to roll this back into the artifact tree if false-positive.",
        "- Use `multiverse export-run` to bundle this for archival before invoking gc.",
        "",
    ]
    return ("\n".join(lines)).encode("utf-8")
