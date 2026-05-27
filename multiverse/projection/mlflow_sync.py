"""MLflow projection sync plugin (STRATEGY S13).

The plugin reads an artifact bundle (or directory containing
``artifact_manifest.json``) and pushes its metrics + artifacts to MLflow.
On any failure the plugin reports ``TRACKING_SYNC_FAILED`` to the kernel
via the seven-verb API — the kernel never blocks ``ARTIFACT_SUCCESS`` on
projection sync (R6).

The plugin runs in its own process (R1). For tests / smoke checks, the
``sync_artifact_bundle`` function is callable in-process with an in-memory
``MLflowTarget``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..artifact import (
    ARTIFACT_MANIFEST_FILENAME,
    ArtifactManifest,
    ChecksumMismatchError,
    ManifestCorruptError,
    ManifestMissingError,
    read_manifest,
)
from .base import MLflowTarget, SyncOutcome, SyncResult


DEFAULT_PROJECTION_PLUGIN = "mlflow"
_TERMINAL_STATUS_FROM_PRIMARY = {
    "ARTIFACT_SUCCESS": "FINISHED",
    "FAILED": "FAILED",
    "CANCELLED": "KILLED",
    "EVALUATION_FAILED": "FAILED",
    "PROMOTION_FAILED": "FAILED",
}


def sync_artifact_bundle(
    *,
    bundle_dir: Path,
    target: MLflowTarget,
    experiment_name: str = "multiverse",
    final_state: str = "ARTIFACT_SUCCESS",
) -> SyncResult:
    """Push one artifact bundle into MLflow. Returns the outcome.

    Reads the manifest with full sidecar verification; a corrupt or
    missing manifest produces ``TRACKING_SYNC_FAILED`` without touching
    the target.
    """
    bundle = Path(bundle_dir)
    try:
        manifest = read_manifest(bundle)
    except (ManifestMissingError, ManifestCorruptError, ChecksumMismatchError) as exc:
        return SyncResult(
            physical_attempt_id="(unknown)",
            outcome=SyncOutcome.SYNC_FAILED,
            failure_reason=f"manifest read failed: {exc}",
        )

    attempt_id = manifest.physical_attempt_id
    tags = _build_tags(manifest)
    params = _build_params(manifest)
    metrics = _maybe_load_metrics(bundle, manifest)

    try:
        run_id = target.create_run(
            experiment_name=experiment_name,
            run_name=manifest.logical_run_id[:12],
            tags=tags,
        )
        target.log_params(run_id=run_id, params=params)
        if metrics:
            target.log_metrics(run_id=run_id, metrics=metrics)
        artifacts_logged = _log_bundle_artifacts(target, run_id=run_id, bundle=bundle)
        target.set_terminal_status(
            run_id=run_id,
            status=_TERMINAL_STATUS_FROM_PRIMARY.get(final_state, "FINISHED"),
        )
    except Exception as exc:
        return SyncResult(
            physical_attempt_id=attempt_id,
            outcome=SyncOutcome.SYNC_FAILED,
            failure_reason=f"{type(exc).__name__}: {exc}",
        )

    return SyncResult(
        physical_attempt_id=attempt_id,
        outcome=SyncOutcome.SYNCED,
        target_run_id=run_id,
        metrics_logged=len(metrics),
        artifacts_logged=artifacts_logged,
    )


def _build_tags(manifest: ArtifactManifest) -> Dict[str, str]:
    return {
        "multiverse.logical_run_id": manifest.logical_run_id,
        "multiverse.manifest_hash": manifest.manifest_hash,
        "multiverse.mv_contract_version": manifest.mv_contract_version,
        "multiverse.image_kind": manifest.image_identity.kind.value,
        "multiverse.image_value": manifest.image_identity.value,
    }


def _build_params(manifest: ArtifactManifest) -> Dict[str, Any]:
    return {
        "params_hash": manifest.params_hash,
        "dataset_slug": str(manifest.dataset_fingerprint.get("slug", "")),
    }


def _maybe_load_metrics(
    bundle: Path, manifest: ArtifactManifest
) -> Dict[str, float]:
    """Parse ``outputs/metrics.json`` if it is one of the declared
    artifacts. Non-fatal: returns an empty dict if absent or unparseable.
    """
    metrics_path: Optional[Path] = None
    for entry in manifest.artifacts:
        if entry.name == "metrics.json":
            candidate = bundle / "outputs" / entry.name
            if candidate.is_file():
                metrics_path = candidate
                break
    if metrics_path is None:
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}


def _log_bundle_artifacts(
    target: MLflowTarget, *, run_id: str, bundle: Path
) -> int:
    n = 0
    outputs_dir = bundle / "outputs"
    if outputs_dir.is_dir():
        for child in outputs_dir.iterdir():
            if child.is_file():
                target.log_artifact(run_id=run_id, path=str(child))
                n += 1
    manifest_path = bundle / ARTIFACT_MANIFEST_FILENAME
    if manifest_path.is_file():
        target.log_artifact(run_id=run_id, path=str(manifest_path))
        n += 1
    return n


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


@dataclass
class MLflowSyncPlugin:
    """Pluggable sync runner.

    Instantiate with an ``MLflowTarget`` (real or fake). Call
    ``sync_bundle`` for each artifact directory and feed the returned
    ``SyncResult`` to ``kernel.report_projection_status`` (typically via a
    client).
    """

    target: MLflowTarget
    experiment_name: str = "multiverse"

    def sync_bundle(
        self,
        bundle_dir: Path,
        *,
        final_state: str = "ARTIFACT_SUCCESS",
    ) -> SyncResult:
        return sync_artifact_bundle(
            bundle_dir=bundle_dir,
            target=self.target,
            experiment_name=self.experiment_name,
            final_state=final_state,
        )

    def sync_many(self, bundle_dirs: list[Path]) -> list[SyncResult]:
        return [self.sync_bundle(b) for b in bundle_dirs]
