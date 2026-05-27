"""mvd kernel — STRATEGY.md Milestone 7 / R1 / R2.

The minimum surface the kernel exposes is **seven verbs** over a Unix-domain
socket. Plugins (MLflow sync, GC, doctor, exporter, registration) run as
separate processes and talk to the kernel through the same socket and the
artifact filesystem. Per ADR §8 the kernel's import graph excludes MLflow,
Optuna, GC, exporter, and Streamlit; the import-graph test in
``tests/unit/test_mvd_kernel.py`` enforces this.

The kernel composes other Milestone packages:

* ``multiverse.journal`` — append-only durability record;
* ``multiverse.docker_supervisor`` — container labels, leases, cancel saga;
* ``multiverse.promotion`` — saga, quarantine, recovery;
* ``multiverse.artifact`` — manifests, validators, bundle writer.

Run execution is delegated to a pluggable ``RunExecutor`` so the kernel itself
contains no Docker, scvi-tools, or model-specific code.
"""

from .api import KernelAPI, KERNEL_VERBS
from .docker_executor import MvdDockerExecutor, build_executor_options
from .events import KernelEvent, EventKind
from .executor import (
    NullRunExecutor,
    RunExecutor,
    SyntheticRunExecutor,
)
from .kernel import Kernel, KernelConfig
from .runs import RunRecord, RunRegistry
from .state import (
    PrimaryState,
    PROJECTION_STATUSES,
    STATE_TRANSITIONS,
    assert_valid_transition,
)

__all__ = [
    "EventKind",
    "KERNEL_VERBS",
    "Kernel",
    "KernelAPI",
    "KernelConfig",
    "KernelEvent",
    "MvdDockerExecutor",
    "NullRunExecutor",
    "PROJECTION_STATUSES",
    "PrimaryState",
    "RunExecutor",
    "RunRecord",
    "RunRegistry",
    "STATE_TRANSITIONS",
    "SyntheticRunExecutor",
    "assert_valid_transition",
    "build_executor_options",
]
