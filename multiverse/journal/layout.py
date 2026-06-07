"""Journal directory layout (STRATEGY R3 / ADR §10)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CURRENT_SEGMENT_NAME = "current.log"
ROTATED_SUBDIR = "rotated"
BLOBS_SUBDIR = "blobs"
CHECKPOINT_FILENAME = "checkpoint.json"
WRITER_LOCK_FILENAME = ".writer.lock"


@dataclass(frozen=True)
class JournalLayout:
    """Resolved filesystem paths inside a journal root.

    ``ensure()`` creates the directories. Tests pass a tmp path; production
    uses ``${MULTIVERSE_STATE_ROOT}/journal``.
    """

    root: Path

    @property
    def current_segment(self) -> Path:
        return self.root / CURRENT_SEGMENT_NAME

    @property
    def rotated_dir(self) -> Path:
        return self.root / ROTATED_SUBDIR

    @property
    def blobs_dir(self) -> Path:
        return self.root / BLOBS_SUBDIR

    @property
    def checkpoint(self) -> Path:
        return self.root / CHECKPOINT_FILENAME

    @property
    def writer_lock(self) -> Path:
        return self.root / WRITER_LOCK_FILENAME

    def ensure(self) -> "JournalLayout":
        """Create the journal root and its subdirectories; return ``self``."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.rotated_dir.mkdir(parents=True, exist_ok=True)
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def at(cls, root: str | os.PathLike[str] | Path) -> "JournalLayout":
        """Construct a layout rooted at ``root`` (does not create directories).

        Args:
            root: The journal root directory, e.g. ``<state root>/journal``.
        """
        return cls(root=Path(root))
