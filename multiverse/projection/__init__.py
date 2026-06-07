"""Projection plugins (STRATEGY S13 / R1 / R6 / Milestone 10).

Projections are denormalized caches of the artifact store. MLflow and
Optuna are projections, not contracts: a run can be ``ARTIFACT_SUCCESS``
with ``TRACKING_SYNC_FAILED``. The artifact bundle is the publication-grade
record; MLflow is the cross-run dashboard.

The plugin lives in its own process (per R1 plugins do not share writable
state with the kernel). The kernel talks to it only by:

* writing artifact manifests to disk (the plugin's input);
* receiving ``report_projection_status`` updates over the kernel API.

This module's ``MLflowSyncPlugin`` is composable: tests inject a fake
``MLflowTarget``; production composes the real MLflow SDK.
"""

from .base import MLflowTarget, SyncOutcome, SyncResult
from .mlflow_sync import (DEFAULT_PROJECTION_PLUGIN, MLflowSyncPlugin,
                          sync_artifact_bundle)

__all__ = [
    "DEFAULT_PROJECTION_PLUGIN",
    "MLflowSyncPlugin",
    "MLflowTarget",
    "SyncOutcome",
    "SyncResult",
    "sync_artifact_bundle",
]
