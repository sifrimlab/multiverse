"""Artifact store directory layout for the promotion saga (STRATEGY S3 / S4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ARTIFACTS_SUBDIR = "artifacts"
WORKSPACES_SUBDIR = "workspaces"
QUARANTINE_SUBDIR = "quarantine"
CANCELLED_SUBDIR = "cancelled"
FAILED_SUBDIR = "failed"


@dataclass(frozen=True)
class StoreLayout:
    """Resolved subdirectory paths under the artifact store ``store/`` tree.

    The kernel treats these directories as its property; symlinks within them
    are policy-rejected (R13). Each ``*_root`` override defaults to ``None``,
    in which case the corresponding property derives the path from ``root``;
    this lets tests or split-volume deployments redirect individual subtrees.

    Attributes:
        root: Store root; all defaulted subdirectories hang off it.
        artifacts_root: Override for the artifact-bundle tree.
        workspaces_root: Override for the in-flight workspace tree.
        quarantine_root: Override for the quarantine (recovery-evidence) tree.
        cancelled_root: Override for the cancelled-run tree.
        failed_root: Override for the failed-run tree.
    """

    root: Path
    artifacts_root: Optional[Path] = None
    workspaces_root: Optional[Path] = None
    quarantine_root: Optional[Path] = None
    cancelled_root: Optional[Path] = None
    failed_root: Optional[Path] = None

    @property
    def artifacts(self) -> Path:
        """Root of the artifact bundle tree (``store/artifacts/``)."""
        return self.artifacts_root or self.root / ARTIFACTS_SUBDIR

    @property
    def workspaces(self) -> Path:
        """Root of the in-flight workspace tree (``store/workspaces/``)."""
        return self.workspaces_root or self.root / WORKSPACES_SUBDIR

    @property
    def quarantine(self) -> Path:
        """Root of the quarantine tree (``store/quarantine/``)."""
        return self.quarantine_root or self.root / QUARANTINE_SUBDIR

    @property
    def cancelled(self) -> Path:
        """Root of the cancelled-run tree (``store/cancelled/``)."""
        return self.cancelled_root or self.root / CANCELLED_SUBDIR

    @property
    def failed(self) -> Path:
        """Root of the failed-run tree (``store/failed/``)."""
        return self.failed_root or self.root / FAILED_SUBDIR

    def ensure(self) -> "StoreLayout":
        """Create every store subdirectory if absent.

        Returns:
            ``self``, so callers can chain ``StoreLayout(...).ensure()``.
        """
        for sub in (
            self.artifacts,
            self.workspaces,
            self.quarantine,
            self.cancelled,
            self.failed,
        ):
            sub.mkdir(parents=True, exist_ok=True)
        return self
