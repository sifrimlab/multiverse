from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from .logging_utils import get_logger
from .registry_db import MODELS_DIR, get_db_connection, init_db

logger = get_logger(__name__)

SEMVER_PATTERN = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
DOCKER_IMAGE_TAG_PATTERN = (
    r"^(?:(?:[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?"
    r"(?::[0-9]+)?/)?"
    r"(?:[a-z0-9]+(?:(?:[._-]|__|[-]*)[a-z0-9]+)*/)*"
    r"[a-z0-9]+(?:(?:[._-]|__|[-]*)[a-z0-9]+)*)"
    r"(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})$"
)


class BuildSpec(BaseModel):
    context: str
    dockerfile: str

    @field_validator("context", "dockerfile")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("build context/dockerfile must not be empty.")
        return value


class RuntimeSpec(BaseModel):
    image: str
    entrypoint: Optional[List[str]] = None

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: str) -> str:
        if not re.match(DOCKER_IMAGE_TAG_PATTERN, value):
            raise ValueError(
                "runtime.image must be a valid Docker image reference with explicit tag "
                "(e.g., ghcr.io/org/model:1.2.3)."
            )
        return value


class ResourcesSpec(BaseModel):
    gpu: bool = False
    memory_limit: str = "16g"


class ContractSpec(BaseModel):
    input_path: str = "/input/data.h5mu"
    output_path: str = "/output"
    job_spec_path: str = "/output/job_spec.json"


class ModelManifest(BaseModel):
    name: str
    version: str
    description: Optional[str] = None
    contract_version: str = "1.0.0"
    supported_omics: List[str] = Field(min_length=1)
    runtime: RuntimeSpec
    hyperparameters_schema: Optional[str] = None
    resources: ResourcesSpec = Field(default_factory=ResourcesSpec)
    contract: ContractSpec = Field(default_factory=ContractSpec)
    build: Optional[BuildSpec] = None
    manifest_path: Optional[str] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("name must not be empty.")
        return value

    @field_validator("version", "contract_version")
    @classmethod
    def validate_semver(cls, value: str) -> str:
        if not re.match(SEMVER_PATTERN, value):
            raise ValueError("Must be a valid semantic version (e.g., 1.2.3).")
        return value

    @field_validator("supported_omics")
    @classmethod
    def validate_supported_omics(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("supported_omics must contain at least one modality or ['any'].")
        normalized = [v.strip().lower() for v in value]
        if "any" in normalized and len(normalized) > 1:
            raise ValueError("supported_omics cannot mix 'any' with specific modalities.")
        return normalized


def resolve_model_manifest_path(
    *, manifest_path: Optional[str] = None, slug: Optional[str] = None
) -> Path:
    if manifest_path:
        return Path(manifest_path).resolve()
    if not slug:
        raise ValueError("Either manifest_path or slug must be provided.")
    return (Path(MODELS_DIR) / slug / "model.yaml").resolve()


def _compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_slug(slug: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", slug).strip("-").lower()
    if not safe:
        raise ValueError("Invalid model slug derived from manifest path.")
    return safe


def load_model_manifest(manifest_path: str) -> ModelManifest:
    path = Path(manifest_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model manifest not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Model manifest must parse to a YAML mapping.")
    manifest = ModelManifest(**raw)
    manifest.manifest_path = str(path)
    return manifest


def register_model_from_manifest(
    manifest_path: str,
    *,
    build: bool = False,
    allow_elevated: bool = False,
) -> Dict[str, Any]:
    """Register a model manifest into SQLite.

    STRATEGY v2 §8: every model registration now goes through the
    hardening pipeline. Path-escaping ``raw_files`` entries are refused
    at parse time. Elevated Docker flags (``privileged``,
    ``--network host``, unauthorised volume mounts, …) require an
    explicit ``allow_elevated=True`` opt-in; otherwise registration
    raises :class:`multiverse.registration.PrivilegedRegistrationError`.
    """
    manifest_file = Path(manifest_path).resolve()
    # Activate hardening before any SQLite work.
    from .registration import audit_docker_flags, validate_paths_in_mapping
    from .registration.errors import PrivilegedRegistrationError

    if manifest_file.is_file():
        raw = yaml.safe_load(manifest_file.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict):
            validate_paths_in_mapping(raw, root=manifest_file.parent)
            audit = audit_docker_flags(raw)
            if audit.elevated and not allow_elevated:
                raise PrivilegedRegistrationError(
                    "model manifest requests elevated Docker flags: "
                    + ", ".join(audit.reasons)
                    + "; re-register with allow_elevated=True after auditing"
                )

    manifest = load_model_manifest(manifest_path)
    manifest_hash = _compute_file_sha256(manifest_file)
    model_slug = _sanitize_slug(manifest_file.parent.name)

    init_db()
    conn = get_db_connection()
    conn.row_factory = None
    cursor = conn.cursor()
    existing = cursor.execute(
        """
        SELECT manifest_hash
        FROM models
        WHERE slug = ? AND version = ?
        LIMIT 1
        """,
        (model_slug, manifest.version),
    ).fetchone()
    if existing and existing[0] == manifest_hash:
        conn.close()
        return {
            "action": "noop",
            "slug": model_slug,
            "version": manifest.version,
            "docker_image": manifest.runtime.image,
            "message": "Model manifest unchanged; skipping registration.",
        }

    cursor.execute(
        """
        INSERT OR REPLACE INTO models
        (slug, version, name, docker_image, image_digest, supported_omics, manifest_path, manifest_hash, hyperparameters_schema, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model_slug,
            manifest.version,
            manifest.name,
            manifest.runtime.image,
            None,
            json.dumps(manifest.supported_omics),
            str(manifest_file),
            manifest_hash,
            manifest.hyperparameters_schema,
            "ACTIVE",
        ),
    )
    conn.commit()
    conn.close()

    if build:
        from .builder import build_local_model

        build_local_model(manifest)

    return {
        "action": "inserted_or_updated",
        "slug": model_slug,
        "version": manifest.version,
        "name": manifest.name,
        "docker_image": manifest.runtime.image,
        "message": f"Model '{manifest.name}' registered from manifest.",
    }
