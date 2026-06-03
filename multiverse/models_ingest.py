from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

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


class PreprocessingSpec(BaseModel):
    """Optional preprocessing defaults a model declares in ``model.yaml``.

    Every field is optional: an unset field falls back to the container's
    built-in default (issue #22). Per-run overrides supplied through the run
    manifest / GUI are merged on top of these defaults inside the container.
    """

    n_top_genes: Optional[int] = None
    normalization_target_sum: Optional[float] = None
    log_normalization: Optional[bool] = None
    # Per-modality scaling, e.g. {"rna": False, "atac": True}.
    scale: Optional[Dict[str, bool]] = None

    def to_job_spec(self) -> Dict[str, Any]:
        """Return only the explicitly-set fields, ready for job_spec.json."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ContractSpec(BaseModel):
    input_path: str = "/input/data.h5mu"
    output_path: str = "/output"
    job_spec_path: str = "/output/job_spec.json"


class ApptainerSpec(BaseModel):
    sif_path: Optional[str] = None
    build_from: Optional[str] = None  # "dockerfile" | "def_file"
    def_file: Optional[str] = None
    gpu_required: bool = False


class ModelManifest(BaseModel):
    name: str
    version: str
    description: Optional[str] = None
    contract_version: str = "1.0.0"
    supported_omics: List[str] = Field(min_length=1)
    runtime: Optional[RuntimeSpec] = None
    apptainer: Optional[ApptainerSpec] = None
    hyperparameters_schema: Optional[str] = None
    preprocessing: Optional[PreprocessingSpec] = None
    resources: ResourcesSpec = Field(default_factory=ResourcesSpec)
    contract: ContractSpec = Field(default_factory=ContractSpec)
    build: Optional[BuildSpec] = None
    manifest_path: Optional[str] = None

    @model_validator(mode="after")
    def _require_at_least_one_runtime(self) -> "ModelManifest":
        if self.runtime is None and self.apptainer is None:
            raise ValueError(
                "model.yaml must specify at least one of 'runtime' (Docker) or 'apptainer' (SIF)."
            )
        return self

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
            raise ValueError(
                "supported_omics must contain at least one modality or ['any']."
            )
        normalized = [v.strip().lower() for v in value]
        if "any" in normalized and len(normalized) > 1:
            raise ValueError(
                "supported_omics cannot mix 'any' with specific modalities."
            )
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
    state_root: Optional["Path"] = None,
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
    # Apptainer-only models have no Docker image. The legacy registry_db schema
    # still constrains docker_image NOT NULL, so write "" there; the canonical
    # asset_registry schema is relaxed and stores NULL (see below).
    docker_image = manifest.runtime.image if manifest.runtime else ""
    docker_image_canonical = manifest.runtime.image if manifest.runtime else None

    existing = cursor.execute(
        """
        SELECT manifest_hash
        FROM models
        WHERE slug = ? AND version = ?
        LIMIT 1
        """,
        (model_slug, manifest.version),
    ).fetchone()
    legacy_unchanged = bool(existing and existing[0] == manifest_hash)

    # The legacy registry_db write is skippable when nothing changed; the
    # asset_registry upsert below always runs because that DB (the canonical
    # one) may be missing the row even when the legacy DB is unchanged.
    if not legacy_unchanged:
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
                docker_image,
                None,
                json.dumps(manifest.supported_omics),
                str(manifest_file),
                manifest_hash,
                manifest.hyperparameters_schema,
                "ACTIVE",
            ),
        )
        conn.commit()
    else:
        # Even when the manifest hash is unchanged, the row may have been
        # soft-deleted (status='INACTIVE') by a prior delete. Re-registration
        # must reactivate it, otherwise the GUI (which reads the legacy DB)
        # keeps hiding a model the user just re-added (issue #29).
        cursor.execute(
            "UPDATE models SET status = 'ACTIVE' WHERE slug = ? AND version = ?",
            (model_slug, manifest.version),
        )
        conn.commit()
    conn.close()

    # Always upsert into asset_registry (canonical writer per G6). Use an
    # UPSERT with COALESCE so a sif_path/image_digest already recorded by
    # ``build-sif``/``--set-sif-path`` is preserved when the manifest does not
    # itself carry one.
    from .asset_registry import (get_asset_registry_connection,
                                 init_asset_registry)

    init_asset_registry(state_root)
    ar_conn = get_asset_registry_connection(state_root)
    ar_cursor = ar_conn.cursor()
    sif_path = manifest.apptainer.sif_path if manifest.apptainer else None
    gpu_required = 1 if (manifest.apptainer and manifest.apptainer.gpu_required) else 0
    ar_cursor.execute(
        """
        INSERT INTO models
        (slug, version, name, docker_image, image_digest, supported_omics, manifest_path, manifest_hash, hyperparameters_schema, status, sif_path, gpu_required)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(slug, version) DO UPDATE SET
            name=excluded.name,
            docker_image=excluded.docker_image,
            supported_omics=excluded.supported_omics,
            manifest_path=excluded.manifest_path,
            manifest_hash=excluded.manifest_hash,
            hyperparameters_schema=excluded.hyperparameters_schema,
            status=excluded.status,
            gpu_required=excluded.gpu_required,
            sif_path=COALESCE(excluded.sif_path, models.sif_path),
            image_digest=COALESCE(excluded.image_digest, models.image_digest)
        """,
        (
            model_slug,
            manifest.version,
            manifest.name,
            docker_image_canonical,
            None,
            json.dumps(manifest.supported_omics),
            str(manifest_file),
            manifest_hash,
            manifest.hyperparameters_schema,
            "ACTIVE",
            sif_path,
            gpu_required,
        ),
    )
    ar_conn.commit()
    ar_conn.close()

    if build:
        from .builder import build_local_model

        build_local_model(manifest)

    if legacy_unchanged:
        return {
            "action": "noop",
            "slug": model_slug,
            "version": manifest.version,
            "docker_image": docker_image,
            "message": "Model manifest unchanged; asset registry reconciled.",
        }
    return {
        "action": "inserted_or_updated",
        "slug": model_slug,
        "version": manifest.version,
        "name": manifest.name,
        "docker_image": docker_image,
        "message": f"Model '{manifest.name}' registered from manifest.",
    }
