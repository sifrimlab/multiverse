"""Docker supervisor core (STRATEGY S7 / S8 / Milestone 6).

Hot-path module. Per ADR §8 the kernel imports only:

* the artifact contract;
* the journal;
* the promotion package (for the cancel saga's workspace-preservation step).

The actual Docker SDK is imported lazily inside ``client.RealDockerEngine``
so tests run without a Docker daemon. The kernel talks to whatever object
implements ``ContainerEngine``.

S7 binds every container the kernel launches to a fixed set of labels; S8
defines cancellation as a saga whose shape matches promotion (journal,
idempotent steps, replay-safe).
"""

from .cancel_saga import (DEFAULT_CANCEL_GRACE_SECONDS, CancelOutcome,
                          CancelResult, CancelSaga, CancelStep)
from .client import (ContainerEngine, ContainerInfo, ContainerState,
                     InMemoryContainerEngine, RealDockerEngine)
from .errors import (ContainerEngineError, LeaseExpiredError,
                     NoSuchContainerError, SupervisorError)
from .labels import (LABEL_HOST_PID, LABEL_LOGICAL_RUN_ID, LABEL_MANIFEST_HASH,
                     LABEL_MVD_VERSION, LABEL_OWNER_TOKEN, LABEL_RUN_ID,
                     LABEL_WORKSPACE, MultiverseLabels, multiverse_labels)
from .leases import ContainerLease, LeaseLedger
from .supervisor import DockerSupervisor, LaunchResult, ReconcileReport

__all__ = [
    "CancelOutcome",
    "CancelResult",
    "CancelSaga",
    "CancelStep",
    "ContainerEngine",
    "ContainerEngineError",
    "ContainerInfo",
    "ContainerLease",
    "ContainerState",
    "DEFAULT_CANCEL_GRACE_SECONDS",
    "DockerSupervisor",
    "InMemoryContainerEngine",
    "RealDockerEngine",
    "LABEL_HOST_PID",
    "LABEL_LOGICAL_RUN_ID",
    "LABEL_MANIFEST_HASH",
    "LABEL_MVD_VERSION",
    "LABEL_OWNER_TOKEN",
    "LABEL_RUN_ID",
    "LABEL_WORKSPACE",
    "LaunchResult",
    "LeaseExpiredError",
    "LeaseLedger",
    "MultiverseLabels",
    "NoSuchContainerError",
    "ReconcileReport",
    "SupervisorError",
    "multiverse_labels",
]
