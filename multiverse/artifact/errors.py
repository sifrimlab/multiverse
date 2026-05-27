"""Exception taxonomy for the artifact contract library."""

from __future__ import annotations


class ArtifactContractError(Exception):
    """Base class for every artifact-contract failure."""


class ManifestMissingError(ArtifactContractError):
    """Raised when an expected manifest or sidecar is absent."""


class ManifestCorruptError(ArtifactContractError):
    """Raised when a manifest exists but is structurally invalid (bad JSON,
    schema violation, or unexpected schema version)."""


class ChecksumMismatchError(ArtifactContractError):
    """Raised when an artifact-manifest body fails its sha256 sidecar check.

    Per STRATEGY R4, callers must NOT mutate the manifest directory in
    response to this error. They must report the directory as
    RECOVERY_PENDING and leave repair to ``multiverse doctor --repair`` or
    ``multiverse rebuild-index``.
    """

    def __init__(self, path: str, expected: str, observed: str) -> None:
        super().__init__(
            f"checksum mismatch for {path}: expected {expected}, observed {observed}"
        )
        self.path = path
        self.expected = expected
        self.observed = observed
