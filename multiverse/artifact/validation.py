"""Output semantic validators (STRATEGY S15, S6, R7).

The validators are the precondition gate between container exit and promotion.
Per S15 the only correct ``ARTIFACT_SUCCESS`` is one where every declared
artifact opens, has the right shape, the right dtype, finite-enough values,
and a per-artifact checksum stamped into the manifest.

Validation runs at three levels (S6):

* ``basic`` — default. Cheap checks (<1 s). Single-batch / NaN warnings do
  not refuse, they downgrade to warnings.
* ``strict`` — publication mode. The same checks but warnings become
  refusals; the finite-fraction floor rises to 1.0; only strict-acceptable
  image identities (R10) pass.
* ``developer`` — adds a synthetic round-trip; intended to be invoked from a
  contributor's machine when adding a new model. The synthetic step itself
  is out of this module — see ``tests/integration/`` — but the level is
  recognized here so the manifest stamps it.

The module lazy-imports ``h5py`` so the artifact contract library stays
importable in environments that have not installed the ML-legacy dependency
group.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .checksums import PathLike, sha256_file
from .manifest import ArtifactEntry


class ValidationLevel(str, Enum):
    """How strictly to validate an output bundle (S6).

    ``basic`` downgrades soft problems to warnings; ``strict`` (publication)
    turns those warnings into refusals; ``developer`` additionally exercises a
    synthetic round-trip driven from outside this module.
    """

    BASIC = "basic"
    STRICT = "strict"
    DEVELOPER = "developer"


class IssueSeverity(str, Enum):
    """Whether a validation issue blocks promotion.

    A ``refusal`` fails the run; a ``warning`` is recorded but does not block.
    """

    WARNING = "warning"
    REFUSAL = "refusal"


@dataclass
class ValidationIssue:
    """A single problem found while validating an artifact.

    Attributes:
        code: Stable machine-readable issue code (e.g. ``EMBEDDING_MISSING``).
        message: Human-readable explanation.
        severity: Whether the issue refuses promotion or is only a warning.
        artifact: Name of the offending artifact, when issue is artifact-scoped.
    """

    code: str
    message: str
    severity: IssueSeverity
    artifact: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-ready dict; ``artifact`` omitted when unset."""
        out: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
        }
        if self.artifact is not None:
            out["artifact"] = self.artifact
        return out


@dataclass
class ValidationReport:
    """Outcome of validating an output bundle at a given level.

    Attributes:
        level: The validation level that produced this report.
        issues: All issues found, both warnings and refusals.
        artifact_entries: Checksummed entries for artifacts that were located
            and stamped, ready to fold into the artifact manifest.
    """

    level: ValidationLevel
    issues: List[ValidationIssue] = field(default_factory=list)
    artifact_entries: List[ArtifactEntry] = field(default_factory=list)

    @property
    def refusals(self) -> List[ValidationIssue]:
        """The subset of issues that refuse promotion."""
        return [i for i in self.issues if i.severity is IssueSeverity.REFUSAL]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """The subset of issues that are warnings only."""
        return [i for i in self.issues if i.severity is IssueSeverity.WARNING]

    @property
    def passed(self) -> bool:
        """True when there are no refusals, i.e. the bundle may be promoted."""
        return not self.refusals

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the report to the ``validation_report.json`` shape."""
        return {
            "level": self.level.value,
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
        }


# ---------------------------------------------------------------------------
# Contract description
# ---------------------------------------------------------------------------


class ExpectedArtifactRole(str, Enum):
    """The kind of artifact a contract entry describes, selecting its validator."""

    EMBEDDING = "embedding"
    METRICS = "metrics"
    UMAP = "umap"
    GENERIC = "generic"


@dataclass
class ExpectedArtifact:
    """Declares one artifact a model contract requires."""

    name: str
    role: ExpectedArtifactRole
    required: bool = True
    # Embedding-specific knobs (ignored for other roles).
    expected_n_obs: Optional[int] = None
    finite_fraction_min: float = 1.0
    # PNG-specific knob.
    min_size_bytes: int = 256
    # Metrics-specific knob — a callable that returns issues; default trivial.
    metrics_schema_keys: Sequence[str] = ()

    @classmethod
    def embedding(
        cls,
        name: str = "embeddings.h5",
        *,
        expected_n_obs: Optional[int] = None,
        finite_fraction_min: float = 1.0,
        required: bool = True,
    ) -> "ExpectedArtifact":
        """Declare a required-by-default embedding artifact (``embeddings.h5``)."""
        return cls(
            name=name,
            role=ExpectedArtifactRole.EMBEDDING,
            required=required,
            expected_n_obs=expected_n_obs,
            finite_fraction_min=finite_fraction_min,
        )

    @classmethod
    def metrics(
        cls,
        name: str = "metrics.json",
        *,
        required: bool = False,
        schema_keys: Sequence[str] = (),
    ) -> "ExpectedArtifact":
        """Declare an optional-by-default metrics JSON artifact (``metrics.json``)."""
        return cls(
            name=name,
            role=ExpectedArtifactRole.METRICS,
            required=required,
            metrics_schema_keys=schema_keys,
        )

    @classmethod
    def umap(
        cls,
        name: str = "umap.png",
        *,
        required: bool = False,
        min_size_bytes: int = 256,
    ) -> "ExpectedArtifact":
        """Declare an optional-by-default UMAP plot artifact (``umap.png``)."""
        return cls(
            name=name,
            role=ExpectedArtifactRole.UMAP,
            required=required,
            min_size_bytes=min_size_bytes,
        )


@dataclass
class ModelOutputContract:
    """A model's declared output shape.

    ``mv_contract_version`` enters logical-run-ID derivation so changes to the
    contract version invalidate cached identities (S16).
    """

    mv_contract_version: str
    artifacts: Sequence[ExpectedArtifact]

    @classmethod
    def default(
        cls,
        *,
        expected_n_obs: Optional[int] = None,
        mv_contract_version: str = "1",
    ) -> "ModelOutputContract":
        """The default contract that every model in this repo currently honours."""
        return cls(
            mv_contract_version=mv_contract_version,
            artifacts=[
                ExpectedArtifact.embedding(expected_n_obs=expected_n_obs),
                ExpectedArtifact.metrics(required=False),
                ExpectedArtifact.umap(required=False),
            ],
        )


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _level_finite_floor(level: ValidationLevel, declared: float) -> float:
    """Strict mode raises the finite-fraction floor to 1.0 regardless of
    contract default; basic mode honours the contract default."""
    if level is ValidationLevel.STRICT:
        return max(declared, 1.0)
    return declared


def _refusal_or_warning(
    level: ValidationLevel,
    *,
    in_strict_only: bool,
) -> IssueSeverity:
    """Many checks are warnings in basic mode and refusals in strict mode.

    ``in_strict_only=True`` means "this check is only a refusal under
    strict"; warning otherwise.
    """
    if level is ValidationLevel.BASIC and in_strict_only:
        return IssueSeverity.WARNING
    return IssueSeverity.REFUSAL


def _png_header_ok(path: Path) -> bool:
    """Return True iff the file starts with the 8-byte PNG signature."""
    try:
        with path.open("rb") as fp:
            return fp.read(8) == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False


def _validate_embedding(
    path: Path,
    spec: ExpectedArtifact,
    level: ValidationLevel,
    issues: List[ValidationIssue],
) -> Optional[ArtifactEntry]:
    """Check embeddings.h5: one dataset 'latent', shape, dtype, finite fraction."""
    if not path.is_file():
        issues.append(
            ValidationIssue(
                code="EMBEDDING_MISSING",
                message=f"required embedding artifact missing: {path.name}",
                severity=IssueSeverity.REFUSAL,
                artifact=spec.name,
            )
        )
        return None

    try:
        import h5py  # type: ignore
    except ImportError:  # pragma: no cover — environment-dependent
        issues.append(
            ValidationIssue(
                code="EMBEDDING_VALIDATOR_UNAVAILABLE",
                message="h5py is not installed; cannot semantically validate "
                "the embedding. Install the ml-legacy extras.",
                severity=_refusal_or_warning(level, in_strict_only=True),
                artifact=spec.name,
            )
        )
        return ArtifactEntry.from_path(
            path, name=spec.name, role=spec.role.value, validated=False
        )

    try:
        with h5py.File(path, "r") as f:
            top_level = list(f.keys())
            if top_level != ["latent"]:
                issues.append(
                    ValidationIssue(
                        code="EMBEDDING_BAD_LAYOUT",
                        message=(
                            f"{spec.name} must contain exactly one top-level "
                            f"dataset 'latent'; found {top_level!r}"
                        ),
                        severity=IssueSeverity.REFUSAL,
                        artifact=spec.name,
                    )
                )
                return None

            latent = f["latent"]
            shape = tuple(int(d) for d in latent.shape)
            dtype = latent.dtype

            if (
                spec.expected_n_obs is not None
                and shape
                and shape[0] != spec.expected_n_obs
            ):
                issues.append(
                    ValidationIssue(
                        code="EMBEDDING_WRONG_N_OBS",
                        message=(
                            f"latent.shape[0]={shape[0]} but dataset fingerprint "
                            f"declares {spec.expected_n_obs} cells"
                        ),
                        severity=IssueSeverity.REFUSAL,
                        artifact=spec.name,
                    )
                )
                return None

            if dtype.kind != "f":
                issues.append(
                    ValidationIssue(
                        code="EMBEDDING_NOT_FLOAT",
                        message=f"latent.dtype={dtype.name}; floating dtype required",
                        severity=IssueSeverity.REFUSAL,
                        artifact=spec.name,
                    )
                )
                return None

            # Streaming finite-fraction check; bounded RSS regardless of size.
            import numpy as np  # local import to keep top-level clean

            arr = latent[...]
            finite = float(np.isfinite(arr).mean()) if arr.size > 0 else 0.0
            floor = _level_finite_floor(level, spec.finite_fraction_min)
            if finite < floor:
                issues.append(
                    ValidationIssue(
                        code="EMBEDDING_TOO_MANY_NONFINITE",
                        message=(
                            f"finite fraction {finite:.6f} < floor {floor:.6f}; "
                            f"NaN/inf values dominate the embedding"
                        ),
                        severity=IssueSeverity.REFUSAL,
                        artifact=spec.name,
                    )
                )
                return None
    except (OSError, KeyError, ValueError) as exc:
        issues.append(
            ValidationIssue(
                code="EMBEDDING_UNREADABLE",
                message=f"could not read {spec.name}: {exc}",
                severity=IssueSeverity.REFUSAL,
                artifact=spec.name,
            )
        )
        return None

    return ArtifactEntry.from_path(path, name=spec.name, role=spec.role.value)


def _validate_metrics(
    path: Path,
    spec: ExpectedArtifact,
    level: ValidationLevel,
    issues: List[ValidationIssue],
) -> Optional[ArtifactEntry]:
    """Check a metrics JSON: present (if required), parseable object, required keys."""
    if not path.is_file():
        if spec.required:
            issues.append(
                ValidationIssue(
                    code="METRICS_MISSING",
                    message=f"required metrics artifact missing: {spec.name}",
                    severity=IssueSeverity.REFUSAL,
                    artifact=spec.name,
                )
            )
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(
            ValidationIssue(
                code="METRICS_UNPARSABLE",
                message=f"{spec.name} is not valid JSON: {exc}",
                severity=IssueSeverity.REFUSAL,
                artifact=spec.name,
            )
        )
        return None

    if not isinstance(data, dict):
        issues.append(
            ValidationIssue(
                code="METRICS_NOT_MAPPING",
                message=f"{spec.name} top-level must be an object",
                severity=IssueSeverity.REFUSAL,
                artifact=spec.name,
            )
        )
        return None

    missing = [k for k in spec.metrics_schema_keys if k not in data]
    if missing:
        issues.append(
            ValidationIssue(
                code="METRICS_MISSING_KEYS",
                message=(
                    f"{spec.name} is missing required keys {missing!r}; declared "
                    f"contract requires {list(spec.metrics_schema_keys)!r}"
                ),
                severity=_refusal_or_warning(level, in_strict_only=True),
                artifact=spec.name,
            )
        )

    return ArtifactEntry.from_path(path, name=spec.name, role=spec.role.value)


def _validate_umap(
    path: Path,
    spec: ExpectedArtifact,
    level: ValidationLevel,
    issues: List[ValidationIssue],
) -> Optional[ArtifactEntry]:
    """Check a UMAP plot: present (if required), non-trivial size, PNG header."""
    if not path.is_file():
        if spec.required:
            issues.append(
                ValidationIssue(
                    code="UMAP_MISSING",
                    message=f"required plot artifact missing: {spec.name}",
                    severity=IssueSeverity.REFUSAL,
                    artifact=spec.name,
                )
            )
        return None

    size = path.stat().st_size
    if size < spec.min_size_bytes:
        issues.append(
            ValidationIssue(
                code="UMAP_TOO_SMALL",
                message=(
                    f"{spec.name} is only {size} bytes; minimum "
                    f"{spec.min_size_bytes}. The plot is likely truncated."
                ),
                severity=_refusal_or_warning(level, in_strict_only=True),
                artifact=spec.name,
            )
        )

    if not _png_header_ok(path):
        issues.append(
            ValidationIssue(
                code="UMAP_BAD_HEADER",
                message=f"{spec.name} does not have a PNG header",
                severity=IssueSeverity.REFUSAL,
                artifact=spec.name,
            )
        )
        return None

    return ArtifactEntry.from_path(path, name=spec.name, role=spec.role.value)


def _validate_generic(
    path: Path,
    spec: ExpectedArtifact,
    level: ValidationLevel,
    issues: List[ValidationIssue],
) -> Optional[ArtifactEntry]:
    if not path.is_file():
        if spec.required:
            issues.append(
                ValidationIssue(
                    code="ARTIFACT_MISSING",
                    message=f"required artifact missing: {spec.name}",
                    severity=IssueSeverity.REFUSAL,
                    artifact=spec.name,
                )
            )
        return None
    return ArtifactEntry.from_path(path, name=spec.name, role=spec.role.value)


_VALIDATORS = {
    ExpectedArtifactRole.EMBEDDING: _validate_embedding,
    ExpectedArtifactRole.METRICS: _validate_metrics,
    ExpectedArtifactRole.UMAP: _validate_umap,
    ExpectedArtifactRole.GENERIC: _validate_generic,
}


def validate_output_bundle(
    workspace_dir: PathLike,
    contract: ModelOutputContract,
    level: ValidationLevel = ValidationLevel.BASIC,
) -> ValidationReport:
    """Validate ``workspace_dir`` against ``contract`` and return a report.

    The function is read-only: it never mutates the workspace. Callers decide
    what to do with refusals (typically: transition to ``EVALUATION_FAILED``
    or ``PROMOTION_FAILED`` and quarantine — see S15, R5).
    """
    workspace = Path(workspace_dir)
    if not workspace.is_dir():
        return ValidationReport(
            level=level,
            issues=[
                ValidationIssue(
                    code="WORKSPACE_MISSING",
                    message=f"workspace directory does not exist: {workspace}",
                    severity=IssueSeverity.REFUSAL,
                )
            ],
        )

    issues: List[ValidationIssue] = []
    entries: List[ArtifactEntry] = []
    for spec in contract.artifacts:
        path = workspace / spec.name
        validator = _VALIDATORS[spec.role]
        entry = validator(path, spec, level, issues)
        if entry is not None:
            entries.append(entry)

    return ValidationReport(level=level, issues=issues, artifact_entries=entries)
