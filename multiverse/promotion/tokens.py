"""Ownership tokens for the promotion saga (STRATEGY S3 / R5).

Every staging directory and artifact bundle directory created by the promotion
saga carries a single ``.mvd_owner`` file. The file contains:

    owner_token: <stable uuid>
    physical_attempt_id: <attempt that created the directory>
    boot_id: <mvd kernel boot that wrote the token>
    created_wall_iso: <timestamp>
    purpose: <e.g. "promotion-prepare", "quarantine">

The owner token answers "is this staging directory mine to continue?" after a
crash. Per R5 the mvd kernel never mutates or quarantines a directory without
an owner-token match. Per S3 step 1 the token is written *before* any rename,
so a crash mid-saga leaves a staging directory the saga can recognise as its
own on replay.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..artifact.checksums import atomic_write_bytes

OWNER_TOKEN_FILENAME = ".mvd_owner"


@dataclass(frozen=True)
class OwnerTokenFile:
    """Parsed contents of a ``.mvd_owner`` token file.

    Attributes:
        owner_token: Stable UUID identifying the owner of the directory.
        physical_attempt_id: Attempt that created the directory.
        mvd_boot_id: mvd kernel boot that wrote the token.
        created_wall_iso: Wall-clock timestamp the token was written.
        purpose: What the directory is for (e.g. ``"promotion-prepare"``,
            ``"quarantine"``).
    """

    owner_token: str
    physical_attempt_id: str
    mvd_boot_id: str
    created_wall_iso: str
    purpose: str

    def to_bytes(self) -> bytes:
        """Serialise the token to canonical (sorted-key) JSON bytes."""
        return json.dumps(
            {
                "owner_token": self.owner_token,
                "physical_attempt_id": self.physical_attempt_id,
                "mvd_boot_id": self.mvd_boot_id,
                "created_wall_iso": self.created_wall_iso,
                "purpose": self.purpose,
            },
            sort_keys=True,
            indent=2,
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, raw: bytes) -> "OwnerTokenFile":
        """Parse token bytes back into an ``OwnerTokenFile``.

        Args:
            raw: UTF-8 JSON bytes as produced by ``to_bytes``.

        Returns:
            The reconstructed token.
        """
        data = json.loads(raw.decode("utf-8"))
        return cls(
            owner_token=str(data["owner_token"]),
            physical_attempt_id=str(data["physical_attempt_id"]),
            mvd_boot_id=str(data["mvd_boot_id"]),
            created_wall_iso=str(data["created_wall_iso"]),
            purpose=str(data["purpose"]),
        )


def new_owner_token() -> str:
    """Generate a fresh owner token (UUID4)."""
    return str(uuid.uuid4())


def write_owner_token(
    directory: Path,
    *,
    owner_token: str,
    physical_attempt_id: str,
    mvd_boot_id: str,
    purpose: str = "promotion-prepare",
) -> OwnerTokenFile:
    """Atomically write the ``.mvd_owner`` file inside ``directory``.

    The directory itself is created if needed. The token is written *before*
    any rename (S3 step 1), so a crash mid-saga leaves a directory the saga
    can recognise as its own on replay; it answers "is this staging dir mine
    to continue?".

    Args:
        directory: Target directory; created with parents if absent.
        owner_token: Stable UUID owner identity to record.
        physical_attempt_id: Attempt writing the token.
        mvd_boot_id: mvd kernel boot id writing the token.
        purpose: Why the directory exists (default ``"promotion-prepare"``).

    Returns:
        The ``OwnerTokenFile`` that was written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    payload = OwnerTokenFile(
        owner_token=owner_token,
        physical_attempt_id=physical_attempt_id,
        mvd_boot_id=mvd_boot_id,
        created_wall_iso=datetime.now(timezone.utc).astimezone().isoformat(),
        purpose=purpose,
    )
    atomic_write_bytes(directory / OWNER_TOKEN_FILENAME, payload.to_bytes())
    return payload


def read_owner_token(directory: Path) -> Optional[OwnerTokenFile]:
    """Read the ``.mvd_owner`` token from ``directory``.

    Args:
        directory: Directory expected to hold a token file.

    Returns:
        The parsed token, or ``None`` if no token file is present.

    Raises:
        Exception: Propagated from JSON parsing if the token file exists but
            is corrupt — a corrupt token is never silently treated as absent.
    """
    path = directory / OWNER_TOKEN_FILENAME
    if not path.is_file():
        return None
    return OwnerTokenFile.from_bytes(path.read_bytes())
