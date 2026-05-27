"""Ownership tokens.

Every artifact directory created by the promotion saga carries a single
``.mvd_owner`` file. The file contains:

    owner_token: <stable uuid>
    physical_attempt_id: <attempt that created the directory>
    boot_id: <daemon boot that wrote the token>
    created_wall_iso: <timestamp>
    purpose: <e.g. "promotion-prepare", "quarantine">

The token is the answer to "is this dir mine to mutate?" — per R5 the kernel
never deletes anything without an owner-token match, and per S3 step 1 the
token is written *before* any rename so a crash mid-saga leaves a directory
the saga can recognise as its own on replay.
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
    owner_token: str
    physical_attempt_id: str
    mvd_boot_id: str
    created_wall_iso: str
    purpose: str

    def to_bytes(self) -> bytes:
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

    The directory itself is created if needed. The token file is the first
    artefact written into a newly-prepared artifact directory.
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
    """Read the token file, or ``None`` if absent. Corruption raises."""
    path = directory / OWNER_TOKEN_FILENAME
    if not path.is_file():
        return None
    return OwnerTokenFile.from_bytes(path.read_bytes())
