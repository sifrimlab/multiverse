from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import re
import yaml
from pydantic import BaseModel, Field, field_validator
from .logging_utils import get_logger
from . import registry_db

# scanpy / mudata / muon are imported lazily inside functions that load data so that
# metadata-only paths (e.g. register_from_manifest) do not trigger noisy third-party warnings.

logger = get_logger(__name__)


class DatasetManifest(BaseModel):
    name: str
    omics: List[str]
    raw_files: Dict[str, str]
    metadata_keys: Dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Manifest field 'name' must not be empty.")
        return value

    @field_validator("omics")
    @classmethod
    def validate_omics(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("Manifest field 'omics' must not be empty.")
        return value

    @field_validator("raw_files")
    @classmethod
    def validate_raw_files(cls, value: Dict[str, str]) -> Dict[str, str]:
        if not value:
            raise ValueError("Manifest field 'raw_files' must not be empty.")
        return value

def load_dataset(file_path: str) -> Any:
    """Loads a single-cell dataset from a file.

    Supported formats include `.h5ad` for AnnData and `.h5mu` for MuData.

    Args:
        file_path (str): The path to the dataset file.

    Returns:
        Union[scanpy.AnnData, mudata.MuData]: The loaded dataset object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    import scanpy as sc
    import muon as mu

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")

    if file_path.endswith(".h5ad"):
        logger.info(f"Loading AnnData from {file_path}")
        return sc.read_h5ad(file_path)
    elif file_path.endswith(".h5mu"):
        logger.info(f"Loading MuData from {file_path}")
        return mu.read_h5mu(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Use .h5ad or .h5mu.")

def validate_dataset_structure(
    data: Any,
    batch_key: str,
    cell_type_key: Optional[str] = None
) -> List[str]:
    """Verifies internal structural requirements of the dataset.

    Ensures that the specified `batch_key` and `cell_type_key` (if provided)
    exist in the dataset observations (`.obs`).

    Args:
        data (Union[sc.AnnData, md.MuData]): The dataset object to validate.
        batch_key (str): The observation key identifying experimental batches.
        cell_type_key (Optional[str]): The observation key identifying cell types.
            Defaults to None.

    Returns:
        List[str]: A list of available omics (modalities) in the dataset.

    Raises:
        ValueError: If required keys are missing from the dataset observations.
        TypeError: If the input data is not an AnnData or MuData object.
    """
    import scanpy as sc
    import mudata as md

    # Check if keys exist in observations
    if batch_key not in data.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in dataset observations.")

    if cell_type_key and cell_type_key not in data.obs.columns:
        raise ValueError(f"Cell type key '{cell_type_key}' not found in dataset observations.")

    # Extract available omics
    if isinstance(data, md.MuData):
        omics = list(data.mod.keys())
    elif isinstance(data, sc.AnnData):
        # Default to rna if it's an AnnData object
        omics = ["rna"]
    else:
        raise TypeError("Dataset must be an AnnData or MuData object.")

    logger.info(f"Dataset validated. Available omics: {omics}")
    return omics

def sanitize_slug(slug: str) -> str:
    if not slug or "/" in slug or ".." in slug:
        raise ValueError("Invalid slug.")
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", slug).strip("-").lower()
    if not safe:
        raise ValueError("Invalid slug after sanitization.")
    return safe


def resolve_manifest_path(*, manifest_path: Optional[str] = None, slug: Optional[str] = None) -> Path:
    if manifest_path:
        return Path(manifest_path).resolve()
    if not slug:
        raise ValueError("Either manifest_path or slug must be provided.")
    safe_slug = sanitize_slug(slug)
    return Path(registry_db.DATASETS_DIR).resolve() / safe_slug / "dataset.yaml"


def _compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _processed_placeholder_path(manifest_path: Path) -> str:
    dataset_dir = manifest_path.parent
    return str((dataset_dir / "data" / "processed.h5mu").resolve())


def register_from_manifest(manifest_path: str, update: Optional[bool] = None) -> Dict[str, Any]:
    """Register a dataset manifest into SQLite, metadata only and idempotent.

    STRATEGY v2 §8: every declared path goes through the registration
    hardening pipeline before any side-effect. ``raw_files`` entries that
    escape the dataset directory after symlink canonicalisation are
    rejected by :class:`multiverse.registration.PathEscapeError`.
    """
    path = Path(manifest_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Manifest is empty: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Manifest must parse to a YAML mapping.")

    dataset_dir = path.parent

    # Activate registration hardening (STRATEGY v2 §8). Path escapes are
    # refused at parse time — before SQLite is even opened.
    from .registration import validate_paths_in_mapping
    validate_paths_in_mapping(raw, root=dataset_dir)

    manifest = DatasetManifest(**raw)
    manifest_hash = _compute_file_sha256(path)

    slug = sanitize_slug(dataset_dir.name)

    # Validate referenced raw files exist.
    for _, rel in manifest.raw_files.items():
        candidate = (dataset_dir / rel).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"raw_files entry does not exist: {rel} -> {candidate}")

    batch_key = manifest.metadata_keys.get("batch")
    cell_type_key = manifest.metadata_keys.get("cell_type")
    processed_path = _processed_placeholder_path(path)

    registry_db.init_db()
    existing = registry_db.get_dataset_by_slug(slug)
    if existing:
        existing_hash = existing.get("manifest_hash")
        if existing_hash == manifest_hash:
            return {
                "action": "noop",
                "dataset_id": existing["id"],
                "slug": slug,
                "message": "Dataset already registered and manifest unchanged.",
            }
        if update is None:
            raise RuntimeError(
                "Dataset already registered with changed manifest. Re-run with update=True to replace."
            )
        if update is False:
            return {
                "action": "skipped",
                "dataset_id": existing["id"],
                "slug": slug,
                "message": "Dataset changed but update declined.",
            }

    dataset_id = registry_db.upsert_dataset_from_manifest(
        slug=slug,
        name=manifest.name,
        path=processed_path,
        omics_available=manifest.omics,
        batch_key=batch_key,
        cell_type_key=cell_type_key,
        manifest_path=str(path),
        manifest_hash=manifest_hash,
        status="READY",
    )
    return {
        "action": "updated" if existing else "inserted",
        "dataset_id": dataset_id,
        "slug": slug,
        "message": f"Dataset '{manifest.name}' registered from manifest.",
    }


def preprocess_dataset(manifest_path: str) -> str:
    """Fuse raw modality files from a dataset manifest into a single processed.h5mu.

    Reads every entry in ``raw_files``, creates a MuData object keyed by
    modality name, and writes it to the placeholder path returned by
    ``_processed_placeholder_path``.  Safe to re-run; overwrites any
    previous output.

    Returns the absolute path of the written file.
    """
    import anndata as ad
    import mudata as md

    path = Path(manifest_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest = DatasetManifest(**raw)
    dataset_dir = path.parent
    processed_path = Path(_processed_placeholder_path(path))

    # If registration created an empty directory at this path, remove it first.
    if processed_path.is_dir():
        processed_path.rmdir()
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    modalities: Dict[str, Any] = {}
    for mod_name, rel_path in manifest.raw_files.items():
        file_path = (dataset_dir / rel_path).resolve()
        logger.info(f"Loading modality '{mod_name}' from {file_path}")
        suffix = file_path.suffix.lower()
        if suffix == ".h5ad":
            modalities[mod_name] = ad.read_h5ad(str(file_path))
        elif suffix == ".h5":
            import scanpy as sc
            modalities[mod_name] = sc.read_10x_h5(str(file_path))
        else:
            raise ValueError(
                f"Unsupported format for modality '{mod_name}': {file_path}. "
                "Convert to .h5ad or .h5 (10x CellRanger) first."
            )

    logger.info(f"Building MuData with modalities: {list(modalities)}")
    mdata = md.MuData(modalities)

    # Promote declared metadata keys from modality obs to top-level mdata.obs so
    # that preflight validation and evaluation can find them without inspecting each
    # modality separately.
    declared_keys = list(manifest.metadata_keys.values())
    for key in declared_keys:
        if key and key not in mdata.obs.columns:
            for mod_name, mod_adata in mdata.mod.items():
                if key in mod_adata.obs.columns:
                    mdata.obs[key] = mod_adata.obs[key].reindex(mdata.obs.index)
                    logger.info(
                        "Promoted metadata key '%s' from modality '%s' to mdata.obs.",
                        key, mod_name,
                    )
                    break
            else:
                logger.warning(
                    "Declared metadata key '%s' not found in any modality obs; skipping promotion.",
                    key,
                )

    logger.info(f"Writing processed MuData to {processed_path}")
    mdata.write_h5mu(str(processed_path))
    return str(processed_path)
