"""Top-level model / dataset registration validator (STRATEGY S19)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from .errors import PrivilegedRegistrationError
from .paths import validate_paths_in_mapping
from .privileges import PrivilegeAudit, audit_docker_flags
from .trust import TrustLevel, classify_trust


@dataclass
class ModelRegistrationReport:
    manifest_path: Path
    trust: TrustLevel
    privilege_audit: PrivilegeAudit
    resolved_paths: Dict[str, Path]
    manifest: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "trust": self.trust.value,
            "privilege_audit": self.privilege_audit.to_dict(),
            "resolved_paths": {k: str(v) for k, v in self.resolved_paths.items()},
        }


@dataclass
class DatasetRegistrationReport:
    manifest_path: Path
    resolved_paths: Dict[str, Path]
    manifest: Mapping[str, Any]


def _load(manifest_path: Path) -> Mapping[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, Mapping):
        raise ValueError(
            f"manifest {manifest_path} top-level must be a mapping"
        )
    return data


def validate_model_manifest(
    manifest_path: str | Path,
    *,
    store_root: str | Path,
    allow_elevated: bool = False,
    builtin_root: Optional[Path] = None,
) -> ModelRegistrationReport:
    """Parse, normalise, and audit a ``model.yaml``.

    Raises ``PathEscapeError`` on path-escape; ``PrivilegedRegistrationError``
    if the manifest requests elevated Docker flags and the caller did not
    pass ``allow_elevated=True``.
    """
    path = Path(manifest_path)
    data = _load(path)
    resolved = validate_paths_in_mapping(data, root=store_root)
    audit = audit_docker_flags(data)
    if audit.elevated and not allow_elevated:
        raise PrivilegedRegistrationError(
            "model manifest requests elevated Docker flags: "
            + ", ".join(audit.reasons)
            + "; re-run with allow_elevated=True after auditing"
        )
    trust = classify_trust(path, builtin_root=builtin_root)
    return ModelRegistrationReport(
        manifest_path=path,
        trust=trust,
        privilege_audit=audit,
        resolved_paths=resolved,
        manifest=data,
    )


def validate_dataset_manifest(
    manifest_path: str | Path,
    *,
    store_root: str | Path,
) -> DatasetRegistrationReport:
    """Parse and path-normalise a ``dataset.yaml``."""
    path = Path(manifest_path)
    data = _load(path)
    resolved = validate_paths_in_mapping(data, root=store_root)
    return DatasetRegistrationReport(
        manifest_path=path,
        resolved_paths=resolved,
        manifest=data,
    )
