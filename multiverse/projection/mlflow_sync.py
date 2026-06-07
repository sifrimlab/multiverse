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
from typing import Any, Dict, Optional

from ..artifact import (ArtifactManifest, ChecksumMismatchError,
                        ManifestCorruptError, ManifestMissingError,
                        read_manifest)
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
    existing_run_id: Optional[str] = None,
) -> SyncResult:
    """Push one artifact bundle into MLflow. Returns the outcome.

    Reads the manifest with full sidecar verification; a corrupt or
    missing manifest produces ``TRACKING_SYNC_FAILED`` without touching
    the target.

    If ``existing_run_id`` is provided the sync attaches to that pre-existing
    MLflow run (typically the parent run the container's EpochLogger
    already streamed into) and only appends final scalars + bundle artifacts.
    Otherwise a fresh run is created.
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
        if existing_run_id:
            run_id = existing_run_id
        else:
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


def _resolve_artifact_path(bundle: Path, artifact_name: str) -> Optional[Path]:
    """Find an artifact file inside a bundle.

    The export-bundle layout (``write_bundle``) places artifacts under
    ``<bundle>/outputs/``; the live promotion saga keeps them at the
    bundle root. Try both.
    """
    candidates = (bundle / "outputs" / artifact_name, bundle / artifact_name)
    for c in candidates:
        if c.is_file():
            return c
    return None


def _maybe_load_metrics(bundle: Path, manifest: ArtifactManifest) -> Dict[str, float]:
    """Parse ``metrics.json`` if it is one of the declared artifacts.

    Looks in both the export-bundle ``outputs/`` subdir and the bundle
    root. Non-fatal: returns an empty dict if absent or unparseable.
    """
    metrics_path: Optional[Path] = None
    for entry in manifest.artifacts:
        if entry.name == "metrics.json":
            metrics_path = _resolve_artifact_path(bundle, entry.name)
            break
    if metrics_path is None:
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return _flatten_scalar_metrics(data)


def _flatten_scalar_metrics(data: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Recursively flatten a metrics.json into MLflow scalar entries.

    Lists are reduced to their last element (matches the legacy tracker's
    behaviour for per-epoch history). Booleans are coerced to 0/1.
    """
    out: Dict[str, float] = {}
    for key, value in data.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if key == "history":
            continue
        if isinstance(value, bool):
            out[full] = float(int(value))
            continue
        if isinstance(value, (int, float)):
            out[full] = float(value)
            continue
        if isinstance(value, list) and value:
            try:
                out[full] = float(value[-1])
            except (TypeError, ValueError):
                continue
            continue
        if isinstance(value, dict):
            out.update(_flatten_scalar_metrics(value, prefix=full))
    return out


_ARTIFACT_SKIP_SUFFIXES = (".sha256",)


def _should_skip_artifact(name: str) -> bool:
    if name.startswith("."):
        # Bookkeeping markers (.promotion_complete, .mvd_owner, etc).
        return True
    if name.endswith(_ARTIFACT_SKIP_SUFFIXES):
        # Checksum sidecars travel inside the bundle for self-verification
        # but MLflow already content-addresses uploaded artifacts.
        return True
    return False


def _log_bundle_artifacts(target: MLflowTarget, *, run_id: str, bundle: Path) -> int:
    """Log every declared artifact plus the manifest itself.

    Supports both bundle layouts: the export-bundle ``outputs/`` subdir
    and the live promoted artifact dir where files live at the root.
    """
    n = 0
    seen: set[Path] = set()
    outputs_dir = bundle / "outputs"
    if outputs_dir.is_dir():
        for child in sorted(outputs_dir.iterdir()):
            if not child.is_file() or _should_skip_artifact(child.name):
                continue
            target.log_artifact(run_id=run_id, path=str(child))
            seen.add(child.resolve())
            n += 1
    for child in sorted(bundle.iterdir()):
        if not child.is_file() or _should_skip_artifact(child.name):
            continue
        if child.resolve() in seen:
            continue
        target.log_artifact(run_id=run_id, path=str(child))
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
