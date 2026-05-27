"""Store layout for the promotion saga (STRATEGY S3 / S4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ARTIFACTS_SUBDIR = "artifacts"
WORKSPACES_SUBDIR = "workspaces"
QUARANTINE_SUBDIR = "quarantine"
CANCELLED_SUBDIR = "cancelled"
FAILED_SUBDIR = "failed"


@dataclass(frozen=True)
class StoreLayout:
    """Resolved paths under ``store/``.

    The kernel treats these directories as its property. Symlinks within
    them are policy-rejected (R13). ``ensure()`` creates each directory.
    """

    root: Path

    @property
    def artifacts(self) -> Path:
        return self.root / ARTIFACTS_SUBDIR

    @property
    def workspaces(self) -> Path:
        return self.root / WORKSPACES_SUBDIR

    @property
    def quarantine(self) -> Path:
        return self.root / QUARANTINE_SUBDIR

    @property
    def cancelled(self) -> Path:
        return self.root / CANCELLED_SUBDIR

    @property
    def failed(self) -> Path:
        return self.root / FAILED_SUBDIR

    def ensure(self) -> "StoreLayout":
        for sub in (
            self.artifacts,
            self.workspaces,
            self.quarantine,
            self.cancelled,
            self.failed,
        ):
            sub.mkdir(parents=True, exist_ok=True)
        return self
