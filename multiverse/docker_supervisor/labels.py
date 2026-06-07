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
        """Render the label set as the engine-facing string dict.

        Returns:
            Mapping of canonical ``multiverse.*`` label keys to string
            values, ready to hand to the container engine's ``launch``.
            ``host_pid`` is stringified since engine labels are strings.
        """
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
        """Reconstruct the label set from an engine container's labels.

        Used during reconciliation to decide whether a container the engine
        reports is one of ours and which run it belongs to.

        Args:
            labels: Raw label mapping as returned by the container engine
                for a single container.

        Returns:
            The parsed label set, or ``None`` if any required ``multiverse.*``
            label is absent or malformed — i.e. the container is not ours.
        """
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
    """Construct a run-launch label set, defaulting ``host_pid`` to this process.

    Args:
        run_id: The physical attempt id; the value of ``multiverse.run_id``
            and the primary key reconciliation queries the engine by.
        logical_run_id: Identifier of the logical run grouping this attempt's
            retries/resumes.
        manifest_hash: Hash of the run manifest pinning this attempt's inputs.
        workspace: Absolute path of the in-flight workspace directory.
        owner_token: Token answering "is this workspace mine to continue?"
            after a crash.
        mvd_version: Version string of the mvd kernel that launched the
            container.
        host_pid: PID of the launching mvd process; defaults to the current
            process id when omitted.

    Returns:
        A frozen ``MultiverseLabels`` carrying the full run identity.
    """
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
    """Build the engine label filter selecting a single run's container.

    Args:
        run_id: The physical attempt id to filter on.

    Returns:
        A ``{LABEL_RUN_ID: run_id}`` mapping for ``list_by_labels``.
    """
    return {LABEL_RUN_ID: run_id}


def all_run_labels() -> tuple[str, ...]:
    """Return the frozen tuple of every required run-identity label key."""
    return _ALL_RUN_LABELS
