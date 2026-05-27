"""Docker label conventions (STRATEGY S7).

Every container launched by the kernel carries a fixed set of labels. The
labels are the single source of truth for "is this container ours, and what
run does it belong to?" — reconciliation queries the engine with
``label=multiverse.run_id=...`` rather than relying on Python-held IDs.

The label set is frozen here; adding a label requires a strategy edit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

LABEL_RUN_ID = "multiverse.run_id"
LABEL_LOGICAL_RUN_ID = "multiverse.logical_run_id"
LABEL_MANIFEST_HASH = "multiverse.manifest_hash"
LABEL_WORKSPACE = "multiverse.workspace"
LABEL_OWNER_TOKEN = "multiverse.owner_token"
LABEL_MVD_VERSION = "multiverse.mvd_version"
LABEL_HOST_PID = "multiverse.host_pid"
LABEL_HEALTH_PROBE = "multiverse.health_probe"


_ALL_RUN_LABELS = (
    LABEL_RUN_ID,
    LABEL_LOGICAL_RUN_ID,
    LABEL_MANIFEST_HASH,
    LABEL_WORKSPACE,
    LABEL_OWNER_TOKEN,
    LABEL_MVD_VERSION,
    LABEL_HOST_PID,
)


@dataclass(frozen=True)
class MultiverseLabels:
    """Run-launch label set.

    The kernel constructs one of these per ``CONTAINER_LAUNCH`` journal
    record. The dict form is what gets handed to the container engine.
    """

    run_id: str
    logical_run_id: str
    manifest_hash: str
    workspace: str
    owner_token: str
    mvd_version: str
    host_pid: int

    def to_dict(self) -> Dict[str, str]:
        return {
            LABEL_RUN_ID: self.run_id,
            LABEL_LOGICAL_RUN_ID: self.logical_run_id,
            LABEL_MANIFEST_HASH: self.manifest_hash,
            LABEL_WORKSPACE: self.workspace,
            LABEL_OWNER_TOKEN: self.owner_token,
            LABEL_MVD_VERSION: self.mvd_version,
            LABEL_HOST_PID: str(int(self.host_pid)),
        }

    @classmethod
    def from_dict(cls, labels: Dict[str, str]) -> Optional["MultiverseLabels"]:
        """Build labels from an engine response. Returns None if any required
        label is missing — i.e. the container is not ours."""
        try:
            return cls(
                run_id=labels[LABEL_RUN_ID],
                logical_run_id=labels[LABEL_LOGICAL_RUN_ID],
                manifest_hash=labels[LABEL_MANIFEST_HASH],
                workspace=labels[LABEL_WORKSPACE],
                owner_token=labels[LABEL_OWNER_TOKEN],
                mvd_version=labels[LABEL_MVD_VERSION],
                host_pid=int(labels[LABEL_HOST_PID]),
            )
        except (KeyError, ValueError, TypeError):
            return None


def multiverse_labels(
    *,
    run_id: str,
    logical_run_id: str,
    manifest_hash: str,
    workspace: str,
    owner_token: str,
    mvd_version: str,
    host_pid: Optional[int] = None,
) -> MultiverseLabels:
    return MultiverseLabels(
        run_id=run_id,
        logical_run_id=logical_run_id,
        manifest_hash=manifest_hash,
        workspace=workspace,
        owner_token=owner_token,
        mvd_version=mvd_version,
        host_pid=host_pid if host_pid is not None else os.getpid(),
    )


def label_query(run_id: str) -> Dict[str, str]:
    """Helper returning the label filter dict for one run."""
    return {LABEL_RUN_ID: run_id}


def all_run_labels() -> tuple[str, ...]:
    return _ALL_RUN_LABELS
