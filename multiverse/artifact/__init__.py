"""Artifact contract library — STRATEGY.md Milestone 1.

Hot-path-clean: this package does not import MLflow, Optuna, Docker, Streamlit,
or any other dependency outside the Python standard library (and pydantic for
schema validation). It is the durable contract that every other layer — the
simple-mode runner, the daemon kernel, the export tooling — reads and writes
through. See ADR 0001 §8 (process model) and §11 (time, identity, durability).
"""

from .checksums import sha256_file, sha256_bytes, atomic_write_bytes
from .errors import (
    ArtifactContractError,
    ManifestCorruptError,
    ManifestMissingError,
    ChecksumMismatchError,
)
from .ids import (
    PARAMS_HASH_CANONICAL_SEPARATORS,
    compute_logical_run_id,
    compute_manifest_hash,
    compute_params_hash,
    new_physical_attempt_id,
)
from .image_identity import (
    ImageIdentity,
    ImageIdentityKind,
    verify_runtime_identity_matches_source,
)
from .manifest import (
    ARTIFACT_MANIFEST_FILENAME,
    ARTIFACT_MANIFEST_SHA256_FILENAME,
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    ArtifactEntry,
    ArtifactManifest,
    ProducedAt,
    ProducedBy,
    ResourceObservations,
    StateTransition,
    read_manifest,
    write_manifest,
)
from .timestamps import (
    BootContext,
    new_boot_id,
    produced_at_now,
    timestamp_now_struct,
)
from .validation import (
    ExpectedArtifact,
    ExpectedArtifactRole,
    IssueSeverity,
    ModelOutputContract,
    ValidationIssue,
    ValidationLevel,
    ValidationReport,
    validate_output_bundle,
)
from .bundle import (
    BundleInputs,
    RunAttemptManifest,
    write_bundle,
    write_run_attempt_manifest,
)

__all__ = [
    "ARTIFACT_MANIFEST_FILENAME",
    "ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "ARTIFACT_MANIFEST_SHA256_FILENAME",
    "ArtifactContractError",
    "ArtifactEntry",
    "ArtifactManifest",
    "BootContext",
    "BundleInputs",
    "ChecksumMismatchError",
    "ExpectedArtifact",
    "ExpectedArtifactRole",
    "ImageIdentity",
    "ImageIdentityKind",
    "IssueSeverity",
    "ManifestCorruptError",
    "ManifestMissingError",
    "ModelOutputContract",
    "PARAMS_HASH_CANONICAL_SEPARATORS",
    "ProducedAt",
    "ProducedBy",
    "ResourceObservations",
    "RunAttemptManifest",
    "StateTransition",
    "ValidationIssue",
    "ValidationLevel",
    "ValidationReport",
    "atomic_write_bytes",
    "compute_logical_run_id",
    "compute_manifest_hash",
    "compute_params_hash",
    "new_boot_id",
    "new_physical_attempt_id",
    "produced_at_now",
    "read_manifest",
    "sha256_bytes",
    "sha256_file",
    "timestamp_now_struct",
    "validate_output_bundle",
    "verify_runtime_identity_matches_source",
    "write_bundle",
    "write_manifest",
    "write_run_attempt_manifest",
]
