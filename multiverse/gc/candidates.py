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
    """Classification of a Tier-2 GC candidate directory."""

    PROMOTED_ARTIFACT = "promoted_artifact"
    FAILED_WORKSPACE = "failed_workspace"
    CANCELLED_WORKSPACE = "cancelled_workspace"
    QUARANTINE = "quarantine"


@dataclass
class GcCandidate:
    """A single Tier-2 directory under ``store/`` that GC has considered.

    Attributes:
        path: Absolute path to the candidate directory.
        kind: Classification of the directory (artifact bundle, failed
            workspace, cancelled workspace, or quarantine entry).
        age_seconds: Elapsed seconds since the directory's mtime; used
            against the :class:`~plan.RetentionPolicy` threshold.
        owner_token: Parsed ``.mvd_owner`` token, or ``None`` if absent.
            GC refuses deletion when this is ``None`` (sole-writer invariant).
        has_export: Whether an ``EXPORTED`` marker file exists; required by
            default before an artifact bundle may be deleted.
        has_manifest: Whether an ``artifact_manifest.json`` is present;
            recorded for informational purposes in the dry-run report.
    """

    path: Path
    kind: CandidateKind
    age_seconds: float
    owner_token: Optional[OwnerTokenFile] = None
    has_export: bool = False
    has_manifest: bool = False


def enumerate_candidates(store: StoreLayout) -> List[GcCandidate]:
    """Walk the artifact store and return every Tier-2 GC candidate.

    Scans ``store/artifacts/``, ``store/failed/``, ``store/cancelled/``,
    and ``store/quarantine/``. Promoted artifact bundles, failed/cancelled
    workspaces, and quarantine entries are all included — the gate logic
    in :mod:`apply` decides whether each entry may actually be deleted.

    Args:
        store: Resolved :class:`~promotion.layout.StoreLayout` whose
            ``artifacts``, ``failed``, ``cancelled``, and ``quarantine``
            directories are walked.

    Returns:
        List of :class:`GcCandidate` objects, one per qualifying directory
        found. Order is not guaranteed.
    """
    import time

    out: List[GcCandidate] = []
    now = time.time()

    def _entries(root: Path, kind: CandidateKind, *, recurse_one: bool) -> None:
        if not root.is_dir():
            return
        # Quarantine and cancelled directories are partitioned by date — one
        # extra level of nesting before the actual candidate entries.
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
    """Construct a :class:`GcCandidate` by reading stat, owner token, and markers."""
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
