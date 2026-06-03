"""Tier-1 GC: closed list of daemon-owned scratch paths (STRATEGY R12).

The set of paths Tier-1 GC may touch is enumerated here. Adding a path
requires a strategy edit and updating the corresponding test:
``tests/unit/test_gc.py::test_tier1_paths_are_a_closed_list``.

Tier-1 may NOT touch:
    * ``store/artifacts/``
    * ``store/workspaces/`` (except the ``__mvd_health_probe__`` subdir)
    * ``store/quarantine/``
    * ``store/cancelled/``
    * ``store/failed/``
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# Closed list of *relative* paths under the state root (or the store root)
# Tier-1 GC is permitted to operate on. Format is
# ``(root_kind, relative_path, max_age_seconds)``.
#
# root_kind ∈ {"state", "store"}
TIER1_PATHS: tuple[tuple[str, str, int], ...] = (
    # Rotated journal segments (S2 / R3) — default 90 days.
    ("state", "journal/rotated", 90 * 24 * 3600),
    # Health-probe workspace namespace (R9) — 1 h.
    ("store", "workspaces/__mvd_health_probe__", 3600),
)


@dataclass
class Tier1Result:
    """Per-path tally of how many entries Tier-1 reclaimed."""

    removed_per_path: Dict[str, int] = field(default_factory=dict)
    refused_paths: List[str] = field(default_factory=list)

    @property
    def total_removed(self) -> int:
        return sum(self.removed_per_path.values())


def sweep_tier1(
    *,
    state_root: Path,
    store_root: Path,
    now: float | None = None,
) -> Tier1Result:
    """Walk the closed list and remove entries older than each path's TTL.

    Refuses any path that resolves outside its declared root after symlink
    canonicalisation — defence-in-depth against a malformed closed list.
    """
    now_ts = time.time() if now is None else float(now)
    result = Tier1Result()
    for root_kind, rel, ttl in TIER1_PATHS:
        base = state_root if root_kind == "state" else store_root
        target = (base / rel).resolve()
        try:
            target.relative_to(base.resolve())
        except ValueError:
            result.refused_paths.append(str(target))
            continue
        if not target.is_dir():
            result.removed_per_path[rel] = 0
            continue
        n = 0
        for entry in target.iterdir():
            try:
                stat = entry.stat()
            except OSError:
                continue
            if now_ts - stat.st_mtime <= ttl:
                continue
            try:
                if entry.is_dir():
                    _shallow_remove_tree(entry)
                else:
                    entry.unlink()
                n += 1
            except OSError:
                continue
        result.removed_per_path[rel] = n
    return result


def _shallow_remove_tree(path: Path) -> None:
    """Remove a Tier-1 subtree. Tier-1 entries are kernel-owned scratch
    so removal is allowed (R12 acceptance: "auto-clean is limited to
    enumerated daemon scratch")."""
    for child in sorted(path.rglob("*"), reverse=True):
        try:
            if child.is_dir():
                child.rmdir()
            else:
                child.unlink()
        except OSError:
            continue
    try:
        path.rmdir()
    except OSError:
        pass
