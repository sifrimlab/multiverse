"""Simple-mode manifest schema.

The simple-mode manifest is intentionally narrower than the full
``run_manifest.yaml`` used by the existing CLI: it does not consult a SQLite
registry, so every value the runner needs must appear in the manifest
itself. It accepts a superset, so a manifest written for the daemon path
can be fed to simple-mode if every job carries a resolved ``model_image``,
``dataset_path``, and ``n_obs``.

Shape::

    schema_version: "1"
    globals:                # optional
      mv_contract_version: "1"
    jobs:
      - name: "demo_pca"
        model:
          slug: "pca"
          version: "1.0.0"
          image: "multiverse-pca:1.0.0"     # tag is fine; identity is recorded
          image_digest: "sha256:..."        # optional; promotes identity kind
          contract_version: "1"             # optional override
        dataset:
          slug: "demo"
          path: "/abs/path/to/processed.h5mu"
          n_obs: 100                         # required for embedding validation
          n_vars: 50                         # optional; recorded in fingerprint
          fingerprint:                       # optional; merged into fingerprint
            extra_key: "value"
        params: {}                          # hyperparameters (hashed)
        validators: "basic"                 # default basic; overridable
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml


class SimpleManifestError(ValueError):
    """Raised on malformed or under-specified simple-mode manifests."""


@dataclass
class SimpleJob:
    name: str
    model_slug: str
    model_version: str
    model_image: str
    contract_version: str
    dataset_slug: str
    dataset_path: Path
    batch_key: Optional[str] = None
    cell_type_key: Optional[str] = None
    dataset_n_obs: int
    dataset_n_vars: Optional[int]
    dataset_fingerprint_extra: Dict[str, Any] = field(default_factory=dict)
    image_digest: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    validators: str = "basic"
    gpu: bool = False
    preprocessing: Optional[Dict[str, Any]] = None

    def dataset_fingerprint(self) -> Dict[str, Any]:
        fp: Dict[str, Any] = {
            "slug": self.dataset_slug,
            "n_obs": int(self.dataset_n_obs),
        }
        if self.dataset_n_vars is not None:
            fp["n_vars"] = int(self.dataset_n_vars)
        if self.dataset_path.is_file():
            fp["path_sha256"] = _sha256_path(self.dataset_path)
        fp.update(self.dataset_fingerprint_extra)
        return fp


@dataclass
class SimpleManifest:
    raw_text: str
    schema_version: str
    mv_contract_version: str
    jobs: List[SimpleJob]
    path: Optional[Path] = None


def _sha256_path(path: Path) -> str:
    """Stream-hash a dataset file. Bounded RSS; used for fingerprinting."""
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise SimpleManifestError(f"{where}: missing required field '{key}'")
    return mapping[key]


def _parse_job(idx: int, raw: Mapping[str, Any], default_contract: str) -> SimpleJob:
    where = f"jobs[{idx}]"
    if not isinstance(raw, Mapping):
        raise SimpleManifestError(f"{where} must be a mapping")

    name = str(_require(raw, "name", where))

    model_block = _require(raw, "model", where)
    if not isinstance(model_block, Mapping):
        raise SimpleManifestError(f"{where}.model must be a mapping")
    model_slug = str(_require(model_block, "slug", f"{where}.model"))
    model_version = str(model_block.get("version", "0.0.0"))
    model_image = str(_require(model_block, "image", f"{where}.model"))
    contract_version = str(model_block.get("contract_version", default_contract))
    image_digest_raw = model_block.get("image_digest")
    image_digest = str(image_digest_raw) if image_digest_raw is not None else None
    # GPU is opt-in (issue #30): simple mode defaults to CPU unless the
    # manifest explicitly requests a GPU via model.gpu.
    gpu = bool(model_block.get("gpu", False))

    dataset_block = _require(raw, "dataset", where)
    if not isinstance(dataset_block, Mapping):
        raise SimpleManifestError(f"{where}.dataset must be a mapping")
    dataset_slug = str(_require(dataset_block, "slug", f"{where}.dataset"))
    dataset_path = Path(str(_require(dataset_block, "path", f"{where}.dataset")))
    try:
        dataset_n_obs = int(_require(dataset_block, "n_obs", f"{where}.dataset"))
    except (TypeError, ValueError) as exc:
        raise SimpleManifestError(f"{where}.dataset.n_obs must be an integer") from exc
    if dataset_n_obs <= 0:
        raise SimpleManifestError(
            f"{where}.dataset.n_obs must be positive (got {dataset_n_obs})"
        )
    dataset_n_vars = (
        int(dataset_block["n_vars"])
        if dataset_block.get("n_vars") is not None
        else None
    )
    fingerprint_extra = dict(dataset_block.get("fingerprint") or {})
    batch_key_raw = dataset_block.get("batch_key")
    batch_key = str(batch_key_raw) if batch_key_raw is not None else None
    cell_type_key_raw = dataset_block.get("cell_type_key")
    cell_type_key = str(cell_type_key_raw) if cell_type_key_raw is not None else None

    params_block = raw.get("params") or {}
    if not isinstance(params_block, Mapping):
        raise SimpleManifestError(f"{where}.params must be a mapping")

    validators = str(raw.get("validators", "basic")).lower()
    if validators not in {"basic", "strict", "developer"}:
        raise SimpleManifestError(
            f"{where}.validators must be one of basic/strict/developer"
        )

    preprocessing_block = raw.get("preprocessing")
    if preprocessing_block is not None and not isinstance(preprocessing_block, Mapping):
        raise SimpleManifestError(f"{where}.preprocessing must be a mapping")
    preprocessing = dict(preprocessing_block) if preprocessing_block else None

    return SimpleJob(
        name=name,
        model_slug=model_slug,
        model_version=model_version,
        model_image=model_image,
        contract_version=contract_version,
        dataset_slug=dataset_slug,
        dataset_path=dataset_path,
        batch_key=batch_key,
        cell_type_key=cell_type_key,
        dataset_n_obs=dataset_n_obs,
        dataset_n_vars=dataset_n_vars,
        dataset_fingerprint_extra=fingerprint_extra,
        image_digest=image_digest,
        params=dict(params_block),
        validators=validators,
        gpu=gpu,
        preprocessing=preprocessing,
    )


def parse_simple_manifest(source: Path | str) -> SimpleManifest:
    """Parse a YAML manifest file or string into a ``SimpleManifest``."""
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
        source_path: Optional[Path] = source
    else:
        text = source
        source_path = None

    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SimpleManifestError(f"YAML syntax error: {exc}") from exc

    if not isinstance(loaded, Mapping):
        raise SimpleManifestError("top-level document must be a mapping")

    schema_version = str(loaded.get("schema_version", "1"))
    globals_block = loaded.get("globals") or {}
    if not isinstance(globals_block, Mapping):
        raise SimpleManifestError("globals must be a mapping")
    mv_contract_version = str(globals_block.get("mv_contract_version", "1"))

    jobs_raw = loaded.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise SimpleManifestError("jobs must be a non-empty list")
    jobs = [_parse_job(i, j, mv_contract_version) for i, j in enumerate(jobs_raw)]

    return SimpleManifest(
        raw_text=text,
        schema_version=schema_version,
        mv_contract_version=mv_contract_version,
        jobs=jobs,
        path=source_path,
    )
