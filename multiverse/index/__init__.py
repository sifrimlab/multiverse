"""SQLite index + rebuild (STRATEGY S2 / S17 / R3 / Milestone 8).

The journal is the kernel's authoritative intent record. The artifact tree
is the authoritative outcome record. SQLite is a *projection* — fast to
query for the GUI, fully rebuildable from the other two surfaces.

``RebuildResult.classified`` describes what was found in each source so
``multiverse doctor`` and the GUI can surface partial-success states.
"""

from .rebuilder import (RebuildClassification, RebuildOutcome, RebuildResult,
                        rebuild_index)
from .sqlite_index import (INDEX_FILENAME, SCHEMA_VERSION, SqliteIndex,
                           open_index)

__all__ = [
    "INDEX_FILENAME",
    "RebuildClassification",
    "RebuildOutcome",
    "RebuildResult",
    "SCHEMA_VERSION",
    "SqliteIndex",
    "open_index",
    "rebuild_index",
]
