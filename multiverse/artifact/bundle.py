"""Export bundle layout (STRATEGY S18) and ``run_attempt_manifest.json`` writer.

The bundle is the canonical *publication-supplementary-material* artifact.
Per R7 (simple-mode first), the bundle layout is defined here and consumed
by both the simple-mode runner and the daemon's promotion saga. There is one
writer of artifact manifests in the codebase, and it lives in
``manifest.py``; this module composes it with the surrounding files needed
for a portable export.

Bundle layout::

    <bundle-dir>/
        artifact_manifest.json
        artifact_manifest.sha256
        outputs/                              <-- the model's declared artifacts
            embeddings.h5
            metrics.json
            umap.png
            ...
        inputs/
            run_manifest.yaml                 <-- input manifest (copy)
            model_manifest.yaml               <-- model contract (copy, optional)
        logs/
            container.log                    <-- copied from workspace if present
            model.log
        validation_report.json               <-- ValidationReport.to_dict()
        environment.json                     <-- mvd_version, python, OS, GPU
        manifest.txt                         <-- per-input checksum index
        README.md                            <-- short reproduce-from-this guide
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .checksums import PathLike, atomic_write_bytes, fsync_path, sha256_file
from .manifest import ArtifactManifest, write_manifest

OUTPUTS_SUBDIR = "outputs"
INPUTS_SUBDIR = "inputs"
LOGS_SUBDIR = "logs"
ENVIRONMENT_FILENAME = "environment.json"
INPUT_INDEX_FILENAME = "manifest.txt"
README_FILENAME = "README.md"
VALIDATION_REPORT_FILENAME = "validation_report.json"
RUN_ATTEMPT_MANIFEST_FILENAME = "run_attempt_manifest.json"


def _normalize_inputs(inputs: Mapping[str, PathLike]) -> Dict[str, Path]:
    return {str(k): Path(v) for k, v in inputs.items()}


@dataclass
class BundleInputs:
    """Inputs to ``write_bundle``.

    ``outputs`` is a mapping of bundle-relative names → workspace file paths,
    e.g. ``{"embeddings.h5": "/ws/.../embeddings.h5"}``. The bundle copies
    each entry into ``outputs/``.

    ``inputs`` is the same shape for manifest copies (``run_manifest.yaml``,
    ``model_manifest.yaml``).

    ``logs`` is the same shape for any log files to capture under ``logs/``.
    """

    artifact_manifest: ArtifactManifest
    outputs: Dict[str, Path] = field(default_factory=dict)
    inputs: Dict[str, Path] = field(default_factory=dict)
    logs: Dict[str, Path] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    validation_report: Optional[Dict[str, Any]] = None
    readme_extra: str = ""


def _copy_into(dst_dir: Path, files: Mapping[str, Path]) -> List[Dict[str, Any]]:
    """Copy a mapping of named files into ``dst_dir`` and return per-file checksum
    entries for the bundle's input index."""
    entries: List[Dict[str, Any]] = []
    if not files:
        return entries
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        src_path = Path(src)
        if not src_path.is_file():
            # Skip silently for optional inputs (logs are commonly missing).
            continue
        dst = dst_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src_path), str(dst))
        entries.append(
            {
                "name": name,
                "relpath": str(dst.relative_to(dst_dir.parent)),
                "sha256": sha256_file(dst),
                "size": dst.stat().st_size,
            }
        )
    return entries


def _default_environment() -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "os_name": os.name,
    }


def _readme_text(manifest: ArtifactManifest, extra: str) -> str:
    base = (
        "# multiverse export bundle\n"
        "\n"
        f"This bundle was produced by multiverse and contains a contract-valid "
        f"artifact for logical run `{manifest.logical_run_id}`.\n"
        "\n"
        "## Reproduce\n"
        "\n"
        "1. Install multiverse on a fresh clone of this repository.\n"
        "2. Verify the bundle: `multiverse import-run <this-directory>`.\n"
        "3. The import verifies every checksum in `artifact_manifest.json` "
        "and rebuilds the local registry entry.\n"
        "\n"
        "## What's in the bundle\n"
        "\n"
        "* `artifact_manifest.json` — primary contract (always present).\n"
        "* `artifact_manifest.sha256` — detached checksum sidecar (always "
        "present).\n"
        "* `outputs/` — the model's declared artifacts.\n"
        "* `inputs/` — input manifests.\n"
        "* `logs/` — container and model logs if available.\n"
        "* `environment.json` — environment record.\n"
        "* `validation_report.json` — output of the semantic validators.\n"
    )
    if extra:
        base += "\n" + extra.rstrip() + "\n"
    return base


def write_bundle(bundle_dir: PathLike, payload: BundleInputs) -> Path:
    """Write a full export bundle to ``bundle_dir``.

    Returns the resolved bundle path. The bundle is contract-equivalent to
    what ``multiverse export-run`` produces against the same recipe (R7
    acceptance criterion).
    """
    bundle = Path(bundle_dir)
    bundle.mkdir(parents=True, exist_ok=True)

    outputs_dir = bundle / OUTPUTS_SUBDIR
    inputs_dir = bundle / INPUTS_SUBDIR
    logs_dir = bundle / LOGS_SUBDIR

    output_entries = _copy_into(outputs_dir, payload.outputs)
    input_entries = _copy_into(inputs_dir, payload.inputs)
    _copy_into(logs_dir, payload.logs)

    environment = {**_default_environment(), **payload.environment}
    atomic_write_bytes(
        bundle / ENVIRONMENT_FILENAME,
        json.dumps(environment, indent=2, sort_keys=True).encode("utf-8"),
    )

    if payload.validation_report is not None:
        atomic_write_bytes(
            bundle / VALIDATION_REPORT_FILENAME,
            json.dumps(payload.validation_report, indent=2, sort_keys=True).encode(
                "utf-8"
            ),
        )

    index_payload = {
        "outputs": output_entries,
        "inputs": input_entries,
    }
    atomic_write_bytes(
        bundle / INPUT_INDEX_FILENAME,
        json.dumps(index_payload, indent=2, sort_keys=True).encode("utf-8"),
    )

    atomic_write_bytes(
        bundle / README_FILENAME,
        _readme_text(payload.artifact_manifest, payload.readme_extra).encode("utf-8"),
    )

    # Manifest is written *last* so a partial bundle never carries a valid
    # manifest. Crash-recovery readers therefore know: if manifest exists
    # and verifies, the bundle is complete.
    write_manifest(bundle, payload.artifact_manifest)
    fsync_path(bundle)

    return bundle


# ---------------------------------------------------------------------------
# run_attempt_manifest.json — recorded for non-success terminal outcomes (S5)
# ---------------------------------------------------------------------------


@dataclass
class RunAttemptManifest:
    """Diagnostic manifest for failed/cancelled/quarantined attempts (S5).

    The same identity fields a successful artifact manifest carries are
    stamped here so the failed attempt remains diagnosable, comparable, and
    exportable even though no artifact was promoted.
    """

    physical_attempt_id: str
    logical_run_id: str
    manifest_hash: str
    params_hash: str
    image_identity: Dict[str, Any]
    mv_contract_version: str
    final_state: str
    failure_reason: Optional[str]
    produced_at: Dict[str, Any]
    produced_by: Dict[str, Any]
    state_transitions: List[Dict[str, Any]] = field(default_factory=list)
    recovery_hint: Optional[str] = None
    validation_report: Optional[Dict[str, Any]] = None
    log_checksums: Dict[str, str] = field(default_factory=dict)
    resource_observations: Optional[Dict[str, Any]] = None
    schema_version: str = "1"

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "physical_attempt_id": self.physical_attempt_id,
            "logical_run_id": self.logical_run_id,
            "manifest_hash": self.manifest_hash,
            "params_hash": self.params_hash,
            "image_identity": dict(self.image_identity),
            "mv_contract_version": self.mv_contract_version,
            "final_state": self.final_state,
            "produced_at": dict(self.produced_at),
            "produced_by": dict(self.produced_by),
            "state_transitions": [dict(s) for s in self.state_transitions],
            "log_checksums": dict(self.log_checksums),
        }
        if self.failure_reason is not None:
            out["failure_reason"] = self.failure_reason
        if self.recovery_hint is not None:
            out["recovery_hint"] = self.recovery_hint
        if self.validation_report is not None:
            out["validation_report"] = self.validation_report
        if self.resource_observations is not None:
            out["resource_observations"] = self.resource_observations
        return out


def write_run_attempt_manifest(
    target_dir: PathLike,
    attempt: RunAttemptManifest,
) -> Path:
    """Write a ``run_attempt_manifest.json`` atomically into ``target_dir``.

    The target is typically the failed workspace, the cancellation directory,
    or the quarantine entry.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    body = json.dumps(attempt.to_dict(), indent=2, sort_keys=True).encode("utf-8")
    path = target / RUN_ATTEMPT_MANIFEST_FILENAME
    atomic_write_bytes(path, body)
    return path
