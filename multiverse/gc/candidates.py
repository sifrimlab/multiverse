"""Enumerate Tier-2 GC candidates.

A candidate is any directory under ``store/`` that could *in principle* be
deleted by ``multiverse gc``. Promoted artifacts, failed/cancelled
workspaces, and quarantine entries are all candidates — the gate logic
in :mod:`apply` decides whether the user's flags + retention policy +
owner-token state allow deletion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

from ..promotion.layout import StoreLayout
from ..promotion.tokens import OwnerTokenFile, read_owner_token


class CandidateKind(str, Enum):
    PROMOTED_ARTIFACT = "promoted_artifact"
    FAILED_WORKSPACE = "failed_workspace"
    CANCELLED_WORKSPACE = "cancelled_workspace"
    QUARANTINE = "quarantine"


@dataclass
class GcCandidate:
    path: Path
    kind: CandidateKind
    age_seconds: float
    owner_token: Optional[OwnerTokenFile] = None
    has_export: bool = False
    has_manifest: bool = False


def enumerate_candidates(store: StoreLayout) -> List[GcCandidate]:
    """Walk store/ and return every Tier-2 candidate."""
    import time

    out: List[GcCandidate] = []
    now = time.time()

    def _entries(root: Path, kind: CandidateKind, *, recurse_one: bool) -> None:
        if not root.is_dir():
            return
        # quarantine/cancelled are partitioned by date.
        if recurse_one:
            for date_dir in root.iterdir():
                if date_dir.is_dir():
                    for entry in date_dir.iterdir():
                        if entry.is_dir():
                            out.append(_make(entry, kind, now))
        else:
            for entry in root.iterdir():
                if entry.is_dir():
                    out.append(_make(entry, kind, now))

    _entries(store.artifacts, CandidateKind.PROMOTED_ARTIFACT, recurse_one=False)
    _entries(store.failed, CandidateKind.FAILED_WORKSPACE, recurse_one=False)
    _entries(store.cancelled, CandidateKind.CANCELLED_WORKSPACE, recurse_one=True)
    _entries(store.quarantine, CandidateKind.QUARANTINE, recurse_one=True)
    return out


def _make(path: Path, kind: CandidateKind, now: float) -> GcCandidate:
    stat = path.stat()
    age = max(0.0, now - stat.st_mtime)
    token = read_owner_token(path)
    has_manifest = (path / "artifact_manifest.json").is_file()
    has_export = (path / "EXPORTED").is_file()
    return GcCandidate(
        path=path,
        kind=kind,
        age_seconds=age,
        owner_token=token,
        has_export=has_export,
        has_manifest=has_manifest,
    )
