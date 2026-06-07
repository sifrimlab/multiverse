"""CLI entrypoints for manifest validation, submission, and run control."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import signal
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.live import Live
from rich.table import Table

from ..ingestion import (preprocess_dataset, register_from_manifest,
                         resolve_manifest_path)
# Legacy Docker runner imports are intentionally not loaded at module import.
# The production path goes through mvd; docker_runner.py is unsupported legacy code.
from ..logging_utils import get_logger, setup_logging
from ..models_ingest import (load_model_manifest, register_model_from_manifest,
                             resolve_model_manifest_path)
from ..registry_db import ARTIFACTS_DIR, get_db_connection, init_db

try:
    from ..builder import build_local_model
except ImportError as exc:
    _BUILDER_IMPORT_ERROR = exc

    def build_local_model(*args, **kwargs):
        raise RuntimeError(
            "Model image builds require the docker Python package. "
            "Install the Docker dependencies before running build commands."
        ) from _BUILDER_IMPORT_ERROR


import yaml

logger = get_logger(__name__)
console = Console()


def emit_event(event: str, **payload: Any) -> None:
    """Emit one JSON line event on stderr for machine-readable CLI progress."""
    record = {"event": event, **payload}
    print(json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)


@dataclass
class ParsedManifest:
    """Result of parsing and validating a run manifest YAML file."""

    path: Path
    data: Dict[str, Any] = field(default_factory=dict)
    plan: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
    backend: str = "docker"

    @property
    def ok(self) -> bool:
        return not self.errors


class ManifestValidationError(ValueError):
    """Raised when :attr:`ParsedManifest.errors` is non-empty."""

    def __init__(self, parsed: ParsedManifest):
        self.parsed = parsed
        message = "; ".join(f"{e['field']}: {e['message']}" for e in parsed.errors)
        super().__init__(message or "manifest validation failed")


def _manifest_error(field: str, message: str, code: str = "invalid") -> Dict[str, str]:
    """Build one structured manifest validation error record.

    Args:
        field: Dotted manifest path the error attaches to (e.g. ``jobs[0].slurm``).
        message: Human-readable explanation surfaced to the user.
        code: Machine-readable category used by callers/tests to branch on.

    Returns:
        A ``{"field", "message", "code"}`` dict appended to ``ParsedManifest.errors``.
    """
    return {"field": field, "message": message, "code": code}


# Required omics per model slug. An empty set means any modality combination
# is acceptable; a non-empty set must be a subset of the dataset's available
# omics or the job is dropped in pre-flight validation.
MODEL_REQUIRED_OMICS = {
    "multivi": {"rna", "atac"},
    "totalvi": {"rna", "adt"},
    "cobolt": {"rna", "atac"},
    "mofa": set(),
    "mowgli": set(),
    "pca": set(),
}


@dataclass
class _PeekResult:
    """Outcome of a best-effort HDF5 header peek.

    Attributes:
        value: The peeked value (column set, obs count, batch count) — a
            type-appropriate empty/zero default when the peek failed.
        error: A short structured reason code (e.g. ``"file_unreadable"``,
            ``"missing_obs"``) when the peek failed, else None.
    """

    value: Any
    error: str | None = None


def _import_h5py():
    """Import h5py for dataset content validation during run planning.

    h5py is a core dependency, so this normally succeeds; the guard only trips on
    a broken/partial install, in which case we surface a clear reinstall hint.
    """
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required for dataset content validation during run planning "
            "but could not be imported. It is a core dependency — reinstall "
            "multiverse (e.g. pip install -e .) to repair the environment."
        ) from exc
    return h5py


def _peek_obs_columns(path: str) -> _PeekResult:
    """Return obs columns plus a structured error if the HDF5 header is unreadable.

    For .h5mu files, columns from all modality-level obs groups are merged with the
    top-level obs so that keys stored only under /mod/*/obs are not falsely reported
    as absent during preflight validation.
    """
    try:
        h5py = _import_h5py()
        with h5py.File(path, "r") as f:
            cols: set = set()
            obs = f.get("obs")
            if obs is not None:
                cols |= {k for k in obs.keys() if not k.startswith("_")}
            mod = f.get("mod")
            if mod is not None:
                for mod_name in mod.keys():
                    mod_obs = mod[mod_name].get("obs")
                    if mod_obs is not None:
                        cols |= {k for k in mod_obs.keys() if not k.startswith("_")}
            if not cols:
                return _PeekResult(set(), "missing_obs")
            return _PeekResult(cols)
    except PermissionError as exc:
        logger.debug(f"Permission denied reading obs columns from {path}: {exc}")
        return _PeekResult(set(), "permission_denied")
    except (OSError, KeyError, ValueError) as exc:
        logger.debug(f"Could not read obs columns from {path}: {exc}")
        return _PeekResult(set(), "file_unreadable")


def _read_obs_columns(path: str) -> set:
    """Return obs column names from an h5ad/h5mu file, or empty set if unreadable.

    Args:
        path: Filesystem path to the ``.h5ad`` / ``.h5mu`` dataset file.

    Returns:
        The merged obs column names (top-level plus modality-level for h5mu);
        an empty set when the header cannot be read for any reason.
    """
    result = _peek_obs_columns(path)
    return result.value if isinstance(result.value, set) else set()


def _peek_obs_count(path: str) -> _PeekResult:
    """Return n_obs from obs/_index, or first obs column, without loading X."""
    try:
        h5py = _import_h5py()
        with h5py.File(path, "r") as f:
            obs = f.get("obs")
            if obs is None:
                return _PeekResult(0, "missing_obs")
            if "_index" in obs:
                return _PeekResult(int(obs["_index"].shape[0]))
            for key in obs.keys():
                node = obs[key]
                if hasattr(node, "shape"):
                    return _PeekResult(int(node.shape[0]))
                if "codes" in node:
                    return _PeekResult(int(node["codes"].shape[0]))
            return _PeekResult(0, "zero_cells")
    except PermissionError as exc:
        logger.debug(f"Permission denied reading obs count from {path}: {exc}")
        return _PeekResult(0, "permission_denied")
    except (OSError, KeyError, ValueError) as exc:
        logger.debug(f"Could not read obs count from {path}: {exc}")
        return _PeekResult(0, "file_unreadable")


def _read_obs_count(path: str) -> int:
    """Return n_obs for the dataset file, or 0 when it cannot be determined."""
    result = _peek_obs_count(path)
    return int(result.value or 0)


def _peek_batch_count(path: str, batch_key: str) -> _PeekResult:
    """Return number of unique batch values plus structured error if unreadable.

    For .h5mu files, modality-level obs groups are checked when the key is
    absent from top-level obs, consistent with _peek_obs_columns behaviour.
    """
    try:
        h5py = _import_h5py()
        with h5py.File(path, "r") as f:

            def _count_from_node(batch_node) -> int:
                if "categories" in batch_node:
                    return int(batch_node["categories"].shape[0])
                return int(len(set(batch_node[:])))

            obs = f.get("obs")
            if obs is not None and batch_key in obs:
                return _PeekResult(_count_from_node(obs[batch_key]))

            mod = f.get("mod")
            if mod is not None:
                for mod_name in mod.keys():
                    mod_obs = mod[mod_name].get("obs")
                    if mod_obs is not None and batch_key in mod_obs:
                        return _PeekResult(_count_from_node(mod_obs[batch_key]))

            return _PeekResult(0, "missing_batch_key")
    except PermissionError as exc:
        logger.debug(f"Permission denied reading batch count from {path}: {exc}")
        return _PeekResult(0, "permission_denied")
    except (OSError, KeyError, ValueError) as exc:
        logger.debug(f"Could not read batch count from {path}: {exc}")
        return _PeekResult(0, "file_unreadable")


def _read_batch_count(path: str, batch_key: str) -> int:
    """Return number of unique values of ``batch_key`` in obs; 0 if unreadable.

    Args:
        path: Filesystem path to the dataset file.
        batch_key: obs column whose distinct value count is wanted.
    """
    result = _peek_batch_count(path, batch_key)
    return int(result.value or 0)


def _record_validation_failure(job: Dict[str, Any], reason: str) -> None:
    """Log a pre-flight validation failure for a job.

    The legacy ``runs``-table write was removed in G6; this now only logs, since
    the SQLite index is a rebuildable projection and not the source of run truth.
    """
    logger.warning(
        "Validation failure for %s/%s: %s",
        job.get("dataset_name", "?"),
        job.get("model_slug", job.get("model_name", "?")),
        reason,
    )


def validate_pending_jobs(
    pending_jobs: List[Dict],
    record_failures: bool = False,
) -> Tuple[List[Dict], List[str]]:
    """Pre-flight validation gate run before any container is launched.

    Checks each job for required-omics compatibility, non-zero cell count, and
    batch-key presence, reading dataset obs headers via h5py (each file opened at
    most once via the per-dataset caches). Incompatible jobs are not dropped:
    they are marked ``_skipped`` so they stay visible in the run summary with a
    reason. Soft problems (missing cell_type_key, single batch value) become
    warnings instead of skips.

    Args:
        pending_jobs: Planned job dicts from a plan generator.
        record_failures: When True, also log each hard failure via
            ``_record_validation_failure`` (the legacy runs-table write is gone).

    Returns:
        ``(validated_jobs, warnings)`` where ``validated_jobs`` contains every
        input job (runnable ones unchanged, incompatible ones flagged
        ``_skipped``) and ``warnings`` are human-readable strings for the summary.
    """
    warnings: List[str] = []
    validated: List[Dict] = []

    # Cache per dataset_id so each file is opened at most once.
    obs_cache: Dict[int, set] = {}
    obs_error_cache: Dict[int, str | None] = {}
    obs_count_cache: Dict[int, _PeekResult] = {}
    batch_count_cache: Dict[int, int] = {}

    def skip_job(job: Dict[str, Any], reason: str, message: str) -> None:
        logger.warning(message)
        if record_failures:
            _record_validation_failure(job, reason)
        validated.append({**job, "_skipped": True, "_skip_reason": message})

    for job in pending_jobs:
        dataset_id = job["dataset_id"]
        dataset_name = job["dataset_name"]
        dataset_path = job["dataset_path"]
        model_slug = job.get("model_slug", job.get("model_name", ""))
        omics_available = set(job.get("omics_available") or [])
        batch_key = job.get("batch_key")
        cell_type_key = job.get("cell_type_key")

        required = MODEL_REQUIRED_OMICS.get(model_slug, set())
        if required and not required.issubset(omics_available):
            skip_job(
                job,
                "missing_required_omics",
                f"[SKIP] {dataset_name}/{model_slug}: model requires omics {required}, "
                f"dataset has {omics_available}",
            )
            continue

        if dataset_id not in obs_count_cache:
            obs_count_cache[dataset_id] = _peek_obs_count(dataset_path)
        obs_count = obs_count_cache[dataset_id]
        if obs_count.error in {"file_unreadable", "permission_denied"}:
            skip_job(
                job,
                obs_count.error,
                f"[SKIP] {dataset_name}/{model_slug}: dataset file unreadable ({obs_count.error})",
            )
            continue
        if int(obs_count.value or 0) == 0:
            skip_job(
                job,
                "zero_cells",
                f"[SKIP] {dataset_name}/{model_slug}: dataset has zero cells",
            )
            continue

        if batch_key:
            if dataset_id not in obs_cache:
                obs_cache[dataset_id] = _read_obs_columns(dataset_path)
                if obs_cache[dataset_id]:
                    obs_error_cache[dataset_id] = None
                else:
                    obs_error_cache[dataset_id] = _peek_obs_columns(dataset_path).error
            obs_cols = obs_cache[dataset_id]
            obs_error = obs_error_cache.get(dataset_id)
            if obs_error in {"file_unreadable", "permission_denied"}:
                skip_job(
                    job,
                    obs_error,
                    f"[SKIP] {dataset_name}/{model_slug}: dataset obs unreadable ({obs_error})",
                )
                continue
            if batch_key not in obs_cols:
                skip_job(
                    job,
                    "missing_batch_key",
                    f"[SKIP] {dataset_name}/{model_slug}: batch_key '{batch_key}' not found in dataset obs columns",
                )
                continue

        if cell_type_key:
            if dataset_id not in obs_cache:
                obs_cache[dataset_id] = _read_obs_columns(dataset_path)
                if obs_cache[dataset_id]:
                    obs_error_cache[dataset_id] = None
                else:
                    obs_error_cache[dataset_id] = _peek_obs_columns(dataset_path).error
            obs_cols = obs_cache[dataset_id]
            if cell_type_key not in obs_cols:
                msg = (
                    f"dataset '{dataset_name}': cell_type_key '{cell_type_key}' not found "
                    f"— supervised metrics will be skipped"
                )
                logger.warning(f"[WARN] {msg}")
                warnings.append(msg)

        if batch_key:
            if dataset_id not in batch_count_cache:
                batch_count_cache[dataset_id] = _read_batch_count(
                    dataset_path, batch_key
                )
            n_batches = batch_count_cache[dataset_id]
            if n_batches == 1:
                msg = (
                    f"dataset '{dataset_name}': only 1 batch value found for '{batch_key}' "
                    f"— batch-correction metrics will be skipped"
                )
                logger.warning(f"[WARN] {msg}")
                if msg not in warnings:
                    warnings.append(msg)

        validated.append(job)

    return validated, warnings


def _docker_image_status(image: str) -> tuple[bool, str | None]:
    """Probe whether a Docker image tag is usable on the local daemon.

    Args:
        image: The model's Docker image tag from the asset registry.

    Returns:
        ``(True, None)`` when the image is present locally, otherwise
        ``(False, reason)`` where ``reason`` explains the failure (no tag, no
        Docker SDK, daemon unreachable, or image absent).
    """
    if not image:
        return False, "model registry has no Docker image tag"
    try:
        import docker  # type: ignore

        client = docker.from_env()
        client.ping()
        client.images.get(image)
        return True, None
    except ImportError:
        return False, "Docker SDK is not installed in the active Python environment"
    except Exception as exc:
        return False, f"Docker image {image!r} is not available locally: {exc}"


def build_missing_images(
    plan: List[Dict[str, Any]], db_conn, *, force: bool = False
) -> List[Tuple[str, str]]:
    """Build Docker images for planned jobs that are missing locally.

    For each unique model in ``plan`` whose Docker image is not present in the
    local daemon (or always, when ``force`` is True), load the model manifest
    from the registry and build the image via :func:`build_local_model`.

    Returns a list of ``(model_slug, error_message)`` for builds that failed;
    an empty list means every required image is now present. Models with no
    build section (remote-only images) are skipped, not treated as failures.
    """
    seen: set[str] = set()
    failures: List[Tuple[str, str]] = []
    cursor = db_conn.cursor()
    for job in plan:
        if job.get("_skipped"):
            continue
        slug = str(job.get("model_slug") or job.get("model_name") or "")
        version = str(job.get("model_version") or "")
        image = str(job.get("model_image") or "")
        key = f"{slug}@{version}"
        if not image or key in seen:
            continue
        seen.add(key)

        if not force:
            ok, _ = _docker_image_status(image)
            if ok:
                continue  # already present

        cursor.execute(
            "SELECT manifest_path FROM models WHERE slug = ? AND version = ? LIMIT 1",
            (slug, version),
        )
        row = cursor.fetchone()
        manifest_path = row[0] if row else None
        if not manifest_path:
            failures.append(
                (slug, f"no manifest_path registered for {key}; cannot build")
            )
            continue
        try:
            manifest = load_model_manifest(str(manifest_path))
            logger.info("Auto-building missing image %s for model %s", image, slug)
            build_local_model(manifest)
        except (
            Exception
        ) as exc:  # noqa: BLE001 — surface any build failure to the caller
            failures.append((slug, f"build failed for {key}: {exc}"))
    return failures


def _models_metadata_conn(db_conn):
    """Return a connection whose ``models`` table exposes sif_path/gpu_required.

    SIF path and GPU metadata live in ``asset_registry.db`` (canonical per G6),
    a different database from the run-index (``registry_db``) connection that
    ``parse_manifest`` is given. If the supplied connection already has those
    columns (unit tests pass a unified in-memory DB), reuse it; otherwise open
    the canonical asset_registry connection and signal the caller to close it.

    Returns ``(connection, owned)`` where ``owned`` is True when the caller is
    responsible for closing the returned connection.
    """
    try:
        cols = {
            row[1] for row in db_conn.execute("PRAGMA table_info(models)").fetchall()
        }
        if {"sif_path", "gpu_required"} <= cols:
            return db_conn, False
    except Exception:
        pass
    from ..asset_registry import (get_asset_registry_connection,
                                  init_asset_registry)

    # Ensure the schema exists before querying — get_asset_registry_connection
    # opens (and creates an empty file for) a missing DB but does not create
    # the tables, so a first-ever Slurm run would otherwise hit
    # "no such table: models". init_asset_registry is idempotent.
    init_asset_registry()
    return get_asset_registry_connection(), True


_SLURM_INT_FIELDS = ("gpus", "time_minutes", "mem_gb", "cpus_per_task")


def _coerce_slurm_int(val):
    """Return ``val`` as an int, or None if it is not a whole-number value.

    ``bool`` is rejected (``True``/``False`` are almost certainly a mistake in
    a numeric Slurm field), as are non-integral floats and unparseable strings.
    """
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):  # reject NaN / inf before int() would raise
        return None
    return int(f) if f == int(f) else None


def _validate_slurm_numeric(
    merged_slurm: Dict[str, Any], field_prefix: str
) -> List[Dict[str, str]]:
    """Validate and normalize the integer-typed fields of a merged slurm block.

    Returns a list of manifest errors so a bad value (e.g. ``gpus: abc``)
    surfaces as a manifest validation error rather than crashing later when the
    executor option builder calls ``int(...)``. Valid fields are normalized in
    place to real ``int`` values, so a quoted whole-number float like
    ``gpus: "1.0"`` does not later crash a downstream ``int("1.0")``.
    """
    errors: List[Dict[str, str]] = []
    for key in _SLURM_INT_FIELDS:
        val = merged_slurm.get(key)
        if val is None:
            continue
        coerced = _coerce_slurm_int(val)
        if coerced is None:
            errors.append(
                _manifest_error(
                    f"{field_prefix}.slurm.{key}",
                    f"slurm.{key} must be a whole number, got {val!r}",
                    "invalid_slurm_field",
                )
            )
        else:
            merged_slurm[key] = coerced
    return errors


def parse_manifest(
    manifest_path: str,
    db_conn,
    *,
    backend_override: Optional[str] = None,
    check_images: bool = True,
) -> ParsedManifest:
    """Parse, validate, and dry-run a run manifest against the live registry.

    ``backend_override`` (e.g. from a CLI ``--backend slurm`` flag) takes
    precedence over ``globals.backend`` so the CLI override is real rather than
    router-level only.

    ``check_images`` (default True) controls the Docker image-availability
    probe. Set it False when the caller intends to build missing images itself
    (auto-build): a missing image then is not a fatal manifest error.
    """
    path = Path(manifest_path).expanduser().resolve()
    parsed = ParsedManifest(path=path)

    try:
        with open(path, "r", encoding="utf-8") as fp:
            loaded = yaml.safe_load(fp)
    except FileNotFoundError:
        parsed.errors.append(
            _manifest_error("manifest", f"file not found: {path}", "file_not_found")
        )
        return parsed
    except UnicodeDecodeError as exc:
        parsed.errors.append(
            _manifest_error(
                "manifest", f"file is not valid UTF-8: {exc}", "unicode_error"
            )
        )
        return parsed
    except yaml.YAMLError as exc:
        parsed.errors.append(
            _manifest_error("manifest", f"YAML syntax error: {exc}", "yaml_error")
        )
        return parsed

    if not isinstance(loaded, dict):
        parsed.errors.append(
            _manifest_error(
                "manifest", "top-level document must be a mapping", "schema_error"
            )
        )
        return parsed
    parsed.data = loaded

    # Extract manifest globals (backend, slurm settings)
    globals_dict = loaded.get("globals", {}) or {}
    if not isinstance(globals_dict, dict):
        parsed.errors.append(
            _manifest_error(
                "globals",
                f"globals must be a mapping, got {type(globals_dict).__name__}",
                "schema_error",
            )
        )
        return parsed
    # CLI --backend overrides the manifest's globals.backend.
    backend = backend_override or globals_dict.get("backend", "docker")
    if backend not in ("docker", "slurm"):
        parsed.errors.append(
            _manifest_error(
                "globals.backend",
                f"unknown backend {backend!r}; must be 'docker' or 'slurm'",
                "invalid_backend",
            )
        )
        return parsed
    slurm_globals = globals_dict.get("slurm", {}) or {}
    if not isinstance(slurm_globals, dict):
        parsed.errors.append(
            _manifest_error(
                "globals.slurm",
                f"globals.slurm must be a mapping, got {type(slurm_globals).__name__}",
                "invalid_slurm_block",
            )
        )
        return parsed
    parsed.data["_backend"] = backend
    parsed.data["_slurm_globals"] = slurm_globals
    parsed.backend = backend

    jobs = loaded.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        parsed.errors.append(
            _manifest_error(
                "jobs", "manifest must contain at least one job", "schema_error"
            )
        )
        return parsed

    # --- Pass 1: slug / registration checks (no Docker I/O) ---
    cursor = db_conn.cursor()
    # Collect (model_name, docker_image) pairs for the image check in pass 2.
    registered_images: list[tuple[str, str]] = []
    for idx, job in enumerate(jobs):
        field_prefix = f"jobs[{idx}]"
        if not isinstance(job, dict):
            parsed.errors.append(
                _manifest_error(field_prefix, "job must be a mapping", "schema_error")
            )
            continue

        job_slurm = job.get("slurm")
        if job_slurm is not None and not isinstance(job_slurm, dict):
            parsed.errors.append(
                _manifest_error(
                    f"{field_prefix}.slurm",
                    f"job slurm must be a mapping, got {type(job_slurm).__name__}",
                    "invalid_slurm_block",
                )
            )

        dataset_key = job.get("dataset_slug") or job.get("dataset_id")
        if not dataset_key:
            parsed.errors.append(
                _manifest_error(
                    f"{field_prefix}.dataset_slug",
                    "dataset_slug or dataset_id is required",
                    "schema_error",
                )
            )
        else:
            cursor.execute(
                "SELECT status FROM datasets WHERE (slug = ? OR name = ?) LIMIT 1",
                (dataset_key, dataset_key),
            )
            row = cursor.fetchone()
            if row is None:
                parsed.errors.append(
                    _manifest_error(
                        f"{field_prefix}.dataset_slug",
                        f"dataset '{dataset_key}' is not registered",
                        "stale_dataset_slug",
                    )
                )
            elif row[0] != "READY":
                parsed.errors.append(
                    _manifest_error(
                        f"{field_prefix}.dataset_slug",
                        f"dataset '{dataset_key}' is {row[0]}, not READY",
                        "dataset_not_ready",
                    )
                )

        if isinstance(job.get("models"), list):
            models_to_run = list(job.get("models", []))
        elif job.get("model_name"):
            models_to_run = [job.get("model_name")]
        else:
            models_to_run = []
        if not models_to_run:
            parsed.errors.append(
                _manifest_error(
                    f"{field_prefix}.models",
                    "models or model_name is required",
                    "schema_error",
                )
            )
        for model_name in models_to_run:
            model_key = str(model_name).strip().lower()
            cursor.execute(
                "SELECT status, docker_image FROM models WHERE slug = ? ORDER BY version DESC LIMIT 1",
                (model_key,),
            )
            row = cursor.fetchone()
            if row is None:
                parsed.errors.append(
                    _manifest_error(
                        f"{field_prefix}.models",
                        f"model '{model_name}' is not registered",
                        "stale_model_slug",
                    )
                )
            elif row[0] != "ACTIVE":
                parsed.errors.append(
                    _manifest_error(
                        f"{field_prefix}.models",
                        f"model '{model_name}' is {row[0]}, not ACTIVE",
                        "model_not_active",
                    )
                )
            else:
                registered_images.append((model_name, str(row[1] or "")))

    if parsed.errors:
        return parsed

    # --- Pass 2: dry-run plan (filter already-succeeded runs) ---
    # Check emptiness before paying for Docker image probes — there is nothing
    # to run if every job in the manifest already has a SUCCESS record.
    parsed.plan = generate_execution_plan_from_manifest(db_conn, parsed.data)
    if not parsed.plan:
        parsed.errors.append(
            _manifest_error(
                "jobs", "manifest dry-run produced no runnable jobs", "empty_plan"
            )
        )
        return parsed

    # --- Slurm-specific Pass: SIF resolution + GPU cross-validation ---
    # Resolution is keyed off the *plan* jobs (not the raw manifest jobs) so
    # the registry-resolved ``model_version`` is used — a manifest rarely
    # repeats the version, and defaulting to "0.0.0" would miss the registered
    # row. SIF/GPU metadata lives in asset_registry.db, which is a different
    # database from the run-index connection threaded in here.
    if backend == "slurm":
        from ..asset_registry import get_model_gpu_flag, get_model_sif_path

        # Per-(dataset, model) overrides from the raw manifest jobs: explicit
        # image_sif / image_digest plus the merged (globals + job) slurm block.
        overrides: Dict[tuple, Dict[str, Any]] = {}
        for job in jobs:
            if not isinstance(job, dict):
                continue
            ds_key = str(job.get("dataset_slug") or job.get("dataset_id") or "")
            merged_slurm = {**slurm_globals, **(dict(job.get("slurm", {}) or {}))}
            if isinstance(job.get("models"), list):
                models_to_run = list(job.get("models", []))
            elif job.get("model_name"):
                models_to_run = [job.get("model_name")]
            else:
                models_to_run = []
            for model_name in models_to_run:
                overrides[(ds_key, str(model_name).strip().lower())] = {
                    "image_sif": job.get("image_sif"),
                    "image_digest": job.get("image_digest"),
                    "slurm": merged_slurm,
                }

        models_conn, _ar_owned = _models_metadata_conn(db_conn)
        try:
            for plan_job in parsed.plan:
                if plan_job.get("_skipped"):
                    continue
                ds_slug = str(plan_job.get("dataset_slug") or "")
                m_slug = str(
                    plan_job.get("model_slug") or plan_job.get("model_name") or ""
                )
                m_version = str(plan_job.get("model_version") or "")
                ov = overrides.get((ds_slug, m_slug), {})
                merged_slurm = ov.get("slurm") or dict(slurm_globals)

                # Validate numeric slurm fields up front so a bad value becomes
                # a manifest error rather than crashing int() below / in the
                # executor option builder.
                numeric_errs = _validate_slurm_numeric(
                    merged_slurm, f"jobs.{ds_slug}.{m_slug}"
                )
                if numeric_errs:
                    parsed.errors.extend(numeric_errs)
                    continue

                # SIF resolution: explicit manifest path > registry sif_path.
                sif_path = ov.get("image_sif") or get_model_sif_path(
                    models_conn, m_slug, m_version
                )
                if not sif_path:
                    parsed.errors.append(
                        _manifest_error(
                            f"jobs.{ds_slug}.{m_slug}.image_sif",
                            f"no SIF path for model '{m_slug}' (version {m_version}); register "
                            "one with `multiverse register-model --set-sif-path` or set "
                            "`image_sif` in the manifest.",
                            "missing_sif_path",
                        )
                    )
                else:
                    plan_job["image_sif"] = sif_path

                # image_digest is optional; without it the run is unverified_local.
                if ov.get("image_digest"):
                    plan_job["image_digest"] = ov["image_digest"]

                # GPU cross-validation (hard error, no silent stripping).
                gpus_val = merged_slurm.get("gpus")
                gpus = int(gpus_val) if gpus_val is not None else 0
                gpu_required = get_model_gpu_flag(models_conn, m_slug, m_version)
                if not gpu_required and gpus > 0:
                    parsed.errors.append(
                        _manifest_error(
                            f"jobs.{ds_slug}.{m_slug}.slurm.gpus",
                            f"model '{m_slug}' declares gpu_required: false but the manifest "
                            f"requests gpus: {gpus}; remove the 'gpus' key or set "
                            "gpu_required: true in model.yaml.",
                            "gpu_conflict",
                        )
                    )
                elif gpu_required and gpus == 0:
                    print(
                        f"WARNING: model '{m_slug}' declares gpu_required: true but no gpus "
                        "are requested in the manifest slurm config.",
                        file=sys.stderr,
                    )

                plan_job["_slurm"] = merged_slurm
        finally:
            if _ar_owned:
                models_conn.close()

    # --- Pass 3: Docker image availability (only for Docker-backed jobs that will run) ---
    # Skip for Slurm backend — images are SIF files, not local Docker images.
    # Skip entirely when check_images is False — the caller will build missing
    # images (auto-build), so a missing image is not a fatal manifest error.
    # Only probe images for models that are actually in the plan: a job deduped
    # out (already succeeded with these params) must not require its image to be
    # present locally, or an unrelated already-done model would block the launch.
    if backend != "slurm" and check_images:
        planned_images: Dict[str, str] = {}
        for plan_job in parsed.plan:
            if plan_job.get("_skipped"):
                continue
            name = str(plan_job.get("model_name") or plan_job.get("model_slug") or "")
            image = str(plan_job.get("model_image") or "")
            if image:
                planned_images[name] = image
        for model_name, docker_image in planned_images.items():
            ok, reason = _docker_image_status(docker_image)
            if not ok:
                parsed.errors.append(
                    _manifest_error(
                        "jobs",
                        f"model '{model_name}' cannot run: {reason}",
                        "model_image_missing",
                    )
                )

    return parsed


def require_parsed_manifest(
    manifest_path: str,
    db_conn,
    *,
    backend_override: Optional[str] = None,
    check_images: bool = True,
) -> ParsedManifest:
    """Parse a manifest and raise unless it validated cleanly.

    Args:
        manifest_path: Path to the run manifest YAML.
        db_conn: Run-index DB connection threaded into ``parse_manifest``.
        backend_override: CLI ``--backend`` value that wins over ``globals.backend``.
        check_images: Forwarded to ``parse_manifest``; see it for image-probe semantics.

    Returns:
        The validated :class:`ParsedManifest`.

    Raises:
        ManifestValidationError: If the manifest produced any validation errors.
    """
    parsed = parse_manifest(
        manifest_path,
        db_conn,
        backend_override=backend_override,
        check_images=check_images,
    )
    if not parsed.ok:
        raise ManifestValidationError(parsed)
    return parsed


def generate_execution_plan_from_manifest(conn, manifest_data: Dict) -> List[Dict]:
    """Expand a parsed manifest into a flat list of planned job dicts.

    This is a *pure* manifest-to-plan expansion: it cross-references each
    (dataset, model) pair against the asset registry (READY datasets, ACTIVE
    models) and emits one job per resolvable pair, with its resolved image,
    version, params hash, and artifact directory name. It deliberately does not
    apply resume/dedupe — see the inline ``STRATEGY`` note and the
    ``multiverse.runner.resume`` module, which owns opt-in skip policy.

    Args:
        conn: Run-index DB connection used to resolve dataset/model rows.
        manifest_data: The already-validated manifest mapping.

    Returns:
        Planned job dicts; pairs that fail registry lookup are logged and omitted.
    """
    cursor = conn.cursor()
    pending_jobs = []
    global_metrics = manifest_data.get("globals", {}).get("metrics", {})

    for job in manifest_data.get("jobs", []):
        dataset_key = job.get("dataset_slug") or job.get("dataset_id")
        models_to_run = []
        if isinstance(job.get("models"), list):
            models_to_run = list(job.get("models", []))
        elif job.get("model_name"):
            models_to_run = [job.get("model_name")]
        mode = job.get("mode", "run")

        cursor.execute(
            "SELECT id, name, path, omics_available, batch_key, cell_type_key "
            "FROM datasets WHERE (slug = ? OR name = ?) AND status = 'READY'",
            (dataset_key, dataset_key),
        )
        dataset_row = cursor.fetchone()
        if not dataset_row:
            logger.warning(
                f"Dataset '{dataset_key}' not found in registry or not READY. Skipping."
            )
            continue

        d_id, d_name, d_path, d_omics_json, d_batch_key, d_cell_type_key = dataset_row
        d_omics = json.loads(d_omics_json) if d_omics_json else []

        for m_name in models_to_run:
            model_lookup = str(m_name).strip().lower()
            cursor.execute(
                """
                SELECT docker_image, slug, version
                FROM models
                WHERE slug = ? AND status = 'ACTIVE'
                ORDER BY version DESC
                LIMIT 1
                """,
                (model_lookup,),
            )
            model_row = cursor.fetchone()
            if not model_row:
                logger.warning(f"Model '{m_name}' not found in registry. Skipping.")
                continue

            m_image, m_slug, m_version = model_row

            model_params = job.get("model_params", {}) or {}
            # Defensive guard: a sweep spec ({"type": ...}) left in a non-sweep
            # job's model_params means the GUI emitted a run job for a swept
            # parameter (the container would receive a dict where it expects a
            # scalar and crash). Surface it here, at manifest-parse time,
            # instead of inside the container.
            if mode != "sweep":
                for _pk, _pv in model_params.items():
                    if isinstance(_pv, dict) and "type" in _pv:
                        logger.error(
                            "Job %s/%s: param '%s' is a sweep spec but mode is "
                            "'%s'. Re-generate the manifest from the Configure "
                            "tab (sweep specs must live in 'search_space' with "
                            "mode: sweep).",
                            dataset_key,
                            m_name,
                            _pk,
                            mode,
                        )
            params_hash = hashlib.sha256(
                json.dumps(model_params, sort_keys=True).encode()
            ).hexdigest()[:12]

            # NOTE (STRATEGY: MVD Manifest Resume and Dedupe): this function is
            # a *pure* manifest-to-plan expansion. It deliberately does NOT
            # consult the legacy ``runs`` table to drop already-succeeded jobs.
            # Legacy ``runs.status = 'SUCCESS'`` rows are not authoritative for
            # mvd-backed execution and could silently suppress jobs the user
            # explicitly requested. Opt-in resume (``skip_completed``) is
            # applied later by ``multiverse.runner.resume`` against durable mvd
            # state (``ARTIFACT_SUCCESS``), keeping ``params_hash`` here only for
            # artifact names, diagnostics, and older UI display code.
            job_metrics = job.get("metrics", {})
            merged_metrics = {**global_metrics, **job_metrics}

            experiment_name = str(
                manifest_data.get("globals", {}).get("experiment_name")
                or manifest_data.get("experiment_name")
                or "manifest"
            )
            experiment_slug = (
                re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment_name).strip("._-")
                or "manifest"
            )
            artifact_dir_name = f"{experiment_slug}_{d_name}_{m_slug}_{params_hash}_{uuid.uuid4().hex[:8]}"
            output_path = os.path.join(ARTIFACTS_DIR, artifact_dir_name)
            dataset_n_obs = _read_obs_count(d_path)
            if dataset_n_obs <= 0:
                logger.warning(
                    "Dataset '%s' has no readable obs count at %s. "
                    "Continuing with dataset_n_obs=0 for %s.",
                    dataset_key,
                    d_path,
                    m_slug,
                )

            job_entry = {
                "dataset_id": d_id,
                "dataset_name": d_name,
                "dataset_slug": str(dataset_key),
                "dataset_path": d_path,
                "dataset_n_obs": dataset_n_obs,
                "omics_available": d_omics,
                "batch_key": d_batch_key,
                "cell_type_key": d_cell_type_key,
                "model_name": m_slug,
                "model_slug": m_slug,
                "model_version": m_version,
                "model_image": m_image,
                "output_path": output_path,
                "artifact_dir_name": artifact_dir_name,
                "mode": mode,
                "model_params": model_params,
                "params_hash": params_hash,
                "search_space": job.get("search_space", {}),
                "optimize_metric": job.get("optimize_metric"),
                "n_trials": job.get("n_trials", 10),
                "direction": job.get("direction", "maximize"),
                "study_storage": job.get("study_storage", "sqlite:///optuna.db"),
                "metrics": merged_metrics,
            }
            if job.get("mem_limit"):
                job_entry["mem_limit"] = job["mem_limit"]
            if job.get("gpu"):
                job_entry["gpu"] = bool(job["gpu"])
            if job.get("preprocessing"):
                job_entry["preprocessing"] = dict(job["preprocessing"])
            pending_jobs.append(job_entry)

    return pending_jobs


def generate_execution_plan(conn) -> List[Dict]:
    """Build a manifest-free plan from every compatible dataset/model pairing.

    The non-manifest fallback path: cross-joins all READY datasets with all
    ACTIVE models whose supported omics are a subset of the dataset's, emitting a
    job for each pair not already recorded SUCCESS in the legacy ``runs`` table.

    Args:
        conn: Run-index DB connection.

    Returns:
        Planned job dicts for the catalog cross-product, minus already-succeeded pairs.
    """
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, name, path, omics_available, batch_key, cell_type_key "
        "FROM datasets WHERE status = 'READY'"
    )
    datasets = cursor.fetchall()

    cursor.execute(
        "SELECT slug, version, docker_image, supported_omics FROM models WHERE status = 'ACTIVE'"
    )
    models = cursor.fetchall()

    pending_jobs = []

    for d_id, d_name, d_path, d_omics_json, d_batch_key, d_cell_type_key in datasets:
        d_omics = set(json.loads(d_omics_json))
        dataset_n_obs = _read_obs_count(d_path)

        for m_slug, m_version, m_image, m_omics_json in models:
            m_omics = set(json.loads(m_omics_json))

            if m_omics.issubset(d_omics):
                cursor.execute(
                    "SELECT status FROM runs WHERE dataset_id = ? AND model_slug = ? AND model_version = ?",
                    (d_id, m_slug, m_version),
                )
                run = cursor.fetchone()

                if run is None or run[0] != "SUCCESS":
                    output_path = os.path.join(ARTIFACTS_DIR, f"{d_name}_{m_slug}")
                    pending_jobs.append(
                        {
                            "dataset_id": d_id,
                            "dataset_name": d_name,
                            "dataset_slug": d_name,
                            "dataset_path": d_path,
                            "dataset_n_obs": dataset_n_obs,
                            "omics_available": list(d_omics),
                            "batch_key": d_batch_key,
                            "cell_type_key": d_cell_type_key,
                            "model_name": m_slug,
                            "model_slug": m_slug,
                            "model_version": m_version,
                            "model_image": m_image,
                            "output_path": output_path,
                            "metrics": {},
                        }
                    )

    return pending_jobs


def generate_status_table(tasks: Dict[str, str]) -> Table:
    """Render the live dashboard table for the concurrent run workflow.

    Args:
        tasks: Mapping of task/model name to its current status string; the
            status text drives the cell colour (green/red/yellow).

    Returns:
        A Rich ``Table`` for display in the ``Live`` dashboard.
    """
    table = Table(title="Multiverse Parallel Execution Dashboard")
    table.add_column("Task/Model", justify="left", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center", style="magenta")

    for name, status in tasks.items():
        style = (
            "green"
            if status in ["Success", "Ready"]
            else "red" if "Failed" in status or "Error" in status else "yellow"
        )
        table.add_row(name, f"[{style}]{status}[/]")

    return table


def _print_run_summary(
    all_jobs: List[Dict],
    result_summary: Dict[str, str],
    warnings: List[str],
) -> None:
    """Print an end-of-run Rich table with per-job status and overall counts."""
    table = Table(title="Run Summary", show_header=True)
    table.add_column("Job", style="cyan", no_wrap=True)
    table.add_column("Dataset", style="blue")
    table.add_column("Model", style="blue")
    table.add_column("Status", justify="center")
    table.add_column("Note", style="dim")

    n_success = n_failed = n_skipped = 0

    for job in all_jobs:
        job_name = (
            job.get("name")
            or f"{job.get('dataset_name', '?')}_{job.get('model_slug') or job.get('model_name', '?')}"
        )
        dataset = job.get("dataset_name", "?")
        model = job.get("model_slug") or job.get("model_name", "?")

        if job.get("_skipped"):
            status_str = "[yellow]SKIPPED[/]"
            note = job.get("_skip_reason", "")
            n_skipped += 1
        else:
            outcome = result_summary.get(job_name, "unknown")
            if outcome == "success":
                status_str = "[green]SUCCESS[/]"
                note = ""
                n_success += 1
            else:
                status_str = "[red]FAILED[/]"
                note = "see workspace logs"
                n_failed += 1

        table.add_row(job_name, dataset, model, status_str, note)

    console.print(table)
    console.print(
        f"Completed: [green]{n_success} succeeded[/], "
        f"[red]{n_failed} failed[/], "
        f"[yellow]{n_skipped} skipped[/] of {len(all_jobs)} jobs."
    )
    if warnings:
        console.print("\n[bold yellow]Warnings:[/]")
        for w in warnings:
            console.print(f"  [yellow]•[/] {w}")


async def run_workflow_async(args: argparse.Namespace):
    """Drive the concurrent Docker-backed run workflow end to end.

    Sets up SIGTERM/SIGINT handlers that mark active runs CANCELLED and cancel
    the current task, verifies the Docker daemon, plans jobs (from a manifest or
    the catalog cross-product), runs pre-flight validation, then funnels all DB
    writes through the single writer actor while running jobs and sweeps
    concurrently. Prints the run summary and drains the writer on exit.

    Args:
        args: Parsed CLI namespace; reads ``output``, ``seed``, and optionally
            ``manifest``.
    """
    os.makedirs(args.output, exist_ok=True)
    setup_logging(args.output)

    loop = asyncio.get_running_loop()
    current_task = asyncio.current_task()
    previous_handlers: dict[int, Any] = {}
    reattach_specs: List[Tuple[Any, int, str, str]] = []

    def _request_shutdown() -> None:
        logger.warning("Shutdown requested; cancelling active workflow tasks")
        try:
            marked = mark_active_runs_failed_direct("CANCELLED")
            if marked:
                emit_event(
                    "status",
                    job="orchestrator",
                    status="marked_cancelled",
                    count=marked,
                )
        except Exception as exc:
            emit_event("error", kind="direct_write_failed", message=str(exc))
        if current_task is not None:
            current_task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handlers[sig] = signal.getsignal(sig)
            loop.add_signal_handler(sig, _request_shutdown)
        except (NotImplementedError, RuntimeError):
            previous_handlers[sig] = signal.signal(
                sig, lambda _sig, _frame: _request_shutdown()
            )

    writer_started = False
    conn = None
    try:
        docker_client = None
        docker_unavailable: Exception | None = None
        try:
            import docker  # type: ignore

            docker_client = docker.from_env()
            try:
                docker_client.ping()
            except Exception as exc:
                emit_event("error", kind="docker_daemon_offline", message=str(exc))
                raise RuntimeError(f"Docker daemon is not reachable: {exc}") from exc
        except Exception as exc:
            docker_unavailable = exc
            logger.warning("Docker client unavailable during recovery: %s", exc)
            emit_event("error", kind="docker_client_unavailable", message=str(exc))

        # recover_orphaned_runs removed in G6 (legacy docker_runner path).
        # MVD-era recovery is handled by Kernel.replay_from_journal + rebuild-index.

        conn = get_db_connection()
        manifest_data: Dict[str, Any] = {}
        manifest_run_id: str | None = None
        if hasattr(args, "manifest") and args.manifest:
            parsed_manifest = require_parsed_manifest(args.manifest, conn)
            manifest_data = parsed_manifest.data
            pending_jobs = parsed_manifest.plan
            # Stable ID for this entire manifest invocation — used to group all
            # runs from one execution together and disambiguate re-runs of the
            # same manifest. Pin via globals.manifest_run_id for reproducible CI.
            manifest_run_id = (
                manifest_data.get("globals", {}).get("manifest_run_id")
                or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
            )
            for job in pending_jobs:
                job["manifest_run_id"] = manifest_run_id
            logger.info("Manifest run ID: %s", manifest_run_id)
        else:
            pending_jobs = generate_execution_plan(conn)
        conn.close()
        conn = None

        host_ram_gb: float | None = manifest_data.get("globals", {}).get("host_ram_gb")

        if not pending_jobs and not reattach_specs:
            logger.info("No pending jobs to execute.")
            return
        if docker_unavailable is not None:
            raise RuntimeError(
                f"Docker is unavailable; refusing to queue jobs: {docker_unavailable}"
            ) from docker_unavailable

        # Start the single-writer DB actor only after planning/recovery succeeds. All DB
        # writes from parallel worker coroutines are funnelled through this task.
        start_db_writer()
        writer_started = True

        reattach_tasks = [
            asyncio.create_task(
                _supervise_container(
                    container, run_id, workspace_dir, final_artifact_dir
                ),
                name=f"reattach_run_{run_id}",
            )
            for container, run_id, workspace_dir, final_artifact_dir in reattach_specs
        ]

        if not pending_jobs:
            if reattach_tasks:
                await asyncio.gather(*reattach_tasks)
            return

        validated_jobs, pre_flight_warnings = validate_pending_jobs(
            pending_jobs, record_failures=True
        )
        runnable_jobs = [j for j in validated_jobs if not j.get("_skipped")]

        if not runnable_jobs:
            logger.info("All jobs were skipped after pre-flight validation.")
            _print_run_summary(validated_jobs, {}, pre_flight_warnings)
            return

        experiment_name = (
            manifest_data.get("globals", {}).get("experiment_name")
            or (
                os.path.basename(os.path.normpath(args.output)) if args.output else None
            )
            or "default_experiment"
        )
        models_info = []
        for job in runnable_jobs:
            job_entry = {
                "name": f"{job['dataset_name']}_{job['model_name']}",
                "image": job["model_image"],
                "dataset_path": job["dataset_path"],
                "output_path": job["output_path"],
                "dataset_id": job["dataset_id"],
                "model_name_orig": job["model_name"],
                "model_slug": job.get("model_slug", job["model_name"]),
                "model_version": job.get("model_version", "0.0.0"),
                "experiment_name": experiment_name,
                "mode": job.get("mode", "run"),
                "model_params": job.get("model_params", {}) or {},
                "search_space": job.get("search_space", {}),
                "optimize_metric": job.get("optimize_metric"),
                "n_trials": job.get("n_trials", 10),
                "direction": job.get("direction", "maximize"),
                "study_storage": job.get("study_storage", "sqlite:///optuna.db"),
                "metrics": job.get("metrics", {}),
                "manifest_run_id": job.get("manifest_run_id"),
                "params_hash": job.get("params_hash"),
            }
            if job.get("mem_limit"):
                job_entry["mem_limit"] = job["mem_limit"]
            if job.get("gpu"):
                job_entry["gpu"] = bool(job["gpu"])
            if job.get("preprocessing"):
                job_entry["preprocessing"] = dict(job["preprocessing"])
            models_info.append(job_entry)

        tasks_status = {m["name"]: "Pending" for m in models_info}
        image_tags = list(set(m["image"] for m in models_info))
        for tag in image_tags:
            tasks_status[tag] = "Queued"

        result_summary: Dict[str, str] = {}

        ensure_docker_data_root()
        from ..multiverse_config import get_docker_data_root

        console.print(f"[bold]Docker data root:[/bold] {get_docker_data_root()}")

        with Live(generate_status_table(tasks_status), refresh_per_second=4) as live:

            def update_status(name, status):
                tasks_status[name] = status
                if status in {"Starting", "Running"} or "In Progress" in str(status):
                    emit_event("job_start", job=name, status=status)
                elif status in {"Success", "Ready"} or "Success" in str(status):
                    emit_event("job_end", job=name, status="SUCCESS")
                elif "Failed" in str(status) or "Error" in str(status):
                    emit_event(
                        "job_end", job=name, status="FAILED", message=str(status)
                    )
                else:
                    emit_event("status", job=name, status=status)
                live.update(generate_status_table(tasks_status))

            update_status("Image Preparation", "In Progress")
            try:
                await build_images_concurrently(
                    image_tags, status_callback=update_status
                )
                update_status("Image Preparation", "Ready")
            except Exception as e:
                update_status("Image Preparation", f"Failed: {e}")
                _print_run_summary(validated_jobs, {}, pre_flight_warnings)
                return

            update_status("Model Execution", "In Progress")

            sweep_jobs = [m for m in models_info if m.get("mode") == "sweep"]
            run_jobs = [m for m in models_info if m.get("mode") != "sweep"]

            if run_jobs:
                result_summary.update(
                    await run_jobs_concurrently(
                        run_jobs,
                        args.seed,
                        status_callback=update_status,
                        host_ram_gb=host_ram_gb,
                    )
                )
            if sweep_jobs:
                from .tuner import run_sweep

                for sweep_job in sweep_jobs:
                    try:
                        result = run_sweep(
                            {
                                "name": sweep_job["name"],
                                "image": sweep_job["image"],
                                "dataset_id": sweep_job["dataset_id"],
                                "dataset_name": sweep_job["dataset_name"],
                                "dataset_path": sweep_job["dataset_path"],
                                "model_name_orig": sweep_job["model_name_orig"],
                                "model_slug": sweep_job.get(
                                    "model_slug", sweep_job["model_name_orig"]
                                ),
                                "model_version": sweep_job.get(
                                    "model_version", "0.0.0"
                                ),
                                "experiment_name": sweep_job.get(
                                    "experiment_name", "default_experiment"
                                ),
                                "model_params": sweep_job.get("model_params", {}) or {},
                                "search_space": sweep_job.get("search_space", {}),
                                "optimize_metric": sweep_job.get("optimize_metric"),
                                "n_trials": sweep_job.get("n_trials", 10),
                                "direction": sweep_job.get("direction", "maximize"),
                                "study_storage": sweep_job.get(
                                    "study_storage", "sqlite:///optuna.db"
                                ),
                                "seed": args.seed,
                            }
                        )
                        logger.info(
                            f"Sweep completed for {sweep_job['name']} best={result['best_value']} params={result['best_params']}"
                        )
                        result_summary[sweep_job["name"]] = "success"
                        update_status(sweep_job["name"], "Success")
                    except Exception as exc:
                        logger.error(f"Sweep failed for {sweep_job['name']}: {exc}")
                        result_summary[sweep_job["name"]] = "failed"
                        update_status(sweep_job["name"], "Failed")

            failed_models = [
                name for name, status in result_summary.items() if status == "failed"
            ]
            if failed_models:
                update_status(
                    "Model Execution", f"Completed with failures: {failed_models}"
                )
            else:
                update_status("Model Execution", "Success")

        if reattach_tasks:
            reattach_results = await asyncio.gather(
                *reattach_tasks, return_exceptions=True
            )
            for result in reattach_results:
                if isinstance(result, Exception):
                    logger.error("Recovered container supervision failed: %s", result)

        _print_run_summary(validated_jobs, result_summary, pre_flight_warnings)
    except asyncio.CancelledError:
        logger.warning("Workflow cancelled; DB writer will be drained before exit")
        raise
    finally:
        if conn is not None:
            conn.close()
        if writer_started:
            await asyncio.shield(stop_db_writer())
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.remove_signal_handler(sig)
                previous = previous_handlers.get(sig)
                if previous is not None:
                    signal.signal(sig, previous)
            except (NotImplementedError, RuntimeError, ValueError):
                pass


async def run_workflow_local_async(_args: argparse.Namespace) -> None:
    """Local (no-Docker) workflow — removed in G6.

    The canonical path is ``multiverse run`` → ``mvd_entrypoint``.
    """
    raise SystemExit(
        "The legacy local-runner path was removed. "
        "Use 'multiverse run' with the mvd entrypoint instead."
    )


def _resolve_effective_seed_for_args(args: argparse.Namespace) -> int:
    """Resolve the execution seed for the simple/local CLI paths (Gap 4).

    Precedence matches the mvd path: explicit ``--seed`` > the manifest's
    ``globals.random_seed`` > 42. Reads the manifest globals with a lightweight
    YAML load (the simple manifest schema differs from the mvd one, so this
    avoids a full ``parse_manifest``).
    """
    from .mvd_entrypoint import resolve_effective_seed

    manifest_data = None
    manifest_path = getattr(args, "manifest", None)
    if manifest_path:
        try:
            loaded = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest_data = loaded
        except Exception:
            manifest_data = None
    return resolve_effective_seed(getattr(args, "seed", None), manifest_data)


def execute_run(args: argparse.Namespace):
    """Route a ``run`` invocation to the correct execution path.

    Dispatch order: ``--simple`` (contract-only simple mode, no
    mvd/SQLite/MLflow/Optuna), then ``--local`` (legacy local path), otherwise
    the production mvd path selected by backend — ``run_via_slurm`` for the slurm
    backend, ``run_via_mvd`` for docker. Backend precedence is ``--backend`` flag,
    then ``globals.backend`` in the manifest, then docker.

    Args:
        args: Parsed CLI namespace for the ``run`` subcommand.
    """
    if getattr(args, "simple", False):
        # Simple-mode is the contract-only path: no SQLite, no MLflow, no
        # Optuna, no daemon. See STRATEGY R7.
        from ..simple.cli import main as simple_main

        argv: list[str] = [str(getattr(args, "manifest", "") or "")]
        out = getattr(args, "output", None) or getattr(args, "out", None)
        if out:
            argv += ["--out", str(out)]
        if getattr(args, "strict", False):
            argv.append("--strict")
        if getattr(args, "validators", None):
            argv += ["--validators", str(args.validators)]
        if getattr(args, "no_image_pull", False):
            argv.append("--no-image-pull")
        # Resolve the effective seed with the same precedence as the mvd path
        # (--seed > globals.random_seed > 42) so a manifest's declared seed is
        # honored in simple mode without requiring --seed (Gap 4).
        seed = _resolve_effective_seed_for_args(args)
        argv += ["--seed", str(seed)]
        raise SystemExit(simple_main(argv))

    if getattr(args, "local", False):
        # Honor globals.random_seed for the legacy local path too (Gap 4).
        args.seed = _resolve_effective_seed_for_args(args)
        try:
            asyncio.run(run_workflow_local_async(args))
        except KeyboardInterrupt:
            logger.warning("Local workflow interrupted by user")
            raise SystemExit(130)
        return

    # Production/default path: route based on backend.
    # --backend flag takes precedence; otherwise use manifest globals; fall back to docker.
    cli_backend = getattr(args, "backend", None)
    if cli_backend is not None:
        effective_backend = cli_backend
    else:
        # Try to read from manifest if available
        effective_backend = "docker"
        if getattr(args, "manifest", None):
            try:
                from ..registry_db import get_db_connection as _get_conn
                from .cli import parse_manifest

                _conn = _get_conn()
                try:
                    _parsed = parse_manifest(args.manifest, _conn)
                    effective_backend = _parsed.backend or "docker"
                finally:
                    _conn.close()
            except Exception:
                pass

    if effective_backend == "slurm":
        from .mvd_entrypoint import run_via_slurm

        raise SystemExit(run_via_slurm(args))
    else:
        from .mvd_entrypoint import run_via_mvd

        raise SystemExit(run_via_mvd(args))


def main():
    """Main entry point for the multiverse CLI."""
    parser = argparse.ArgumentParser(description="Multiverse CLI")

    def add_run_args(p):
        p.add_argument(
            "--models",
            nargs="+",
            required=False,
            help="List of models for local mode (e.g., pca mofa multivi totalvi)",
            default=[],
        )
        p.add_argument(
            "--input", required=False, help="Path to the input data directory"
        )
        p.add_argument(
            "--output", required=True, help="Path to the output results directory"
        )
        p.add_argument(
            "--seed",
            type=int,
            default=None,
            help=(
                "Random seed. Precedence: this flag, then the manifest's "
                "globals.random_seed, then 42. The resolved seed is propagated "
                "into every job_spec.json and into the resume logical-run "
                "identity."
            ),
        )
        p.add_argument(
            "--evaluate",
            required=False,
            action="store_true",
            help="Whether to run evaluation after model execution",
        )
        p.add_argument(
            "--concurrent",
            action="store_true",
            help="Deprecated flag kept for backward compatibility; registry runs are always parallel",
        )
        p.add_argument(
            "--manifest",
            required=False,
            help="Path to a run_manifest.yaml file defining jobs to execute",
        )
        p.add_argument(
            "--local",
            action="store_true",
            default=False,
            help=(
                "Run models locally without Docker by calling container/run.py directly. "
                "Requires model Python deps installed on the host. "
                "Use with --manifest for manifest-driven local dev runs."
            ),
        )
        p.add_argument(
            "--simple",
            action="store_true",
            default=False,
            help=(
                "Simple-mode contract runner. Produces a portable artifact "
                "bundle without consulting mvd, SQLite, MLflow, or Optuna. "
                "Requires --manifest pointing at a simple-mode manifest."
            ),
        )
        p.add_argument(
            "--strict",
            action="store_true",
            default=False,
            help=(
                "Publication mode. Requires a strict-acceptable image identity "
                "(registry digest or build-context hash). Rejects locally-built "
                "images that have no registry provenance. Use this when preparing "
                "a benchmark for publication."
            ),
        )
        p.add_argument(
            "--skip-completed",
            dest="skip_completed",
            action="store_true",
            default=None,
            help=(
                "Opt-in resume: skip manifest jobs whose canonical logical run "
                "already reached ARTIFACT_SUCCESS in the mvd state for the "
                "selected --output directory. Skipped jobs are shown with the "
                "completing attempt/artifact, not silently dropped. Default off; "
                "also settable via globals.skip_completed in the manifest. The "
                "legacy runs table is never consulted."
            ),
        )
        p.add_argument(
            "--no-build",
            dest="no_build",
            action="store_true",
            default=False,
            help=(
                "Docker backend: do not auto-build missing model images before "
                "running. By default, an image that is not present locally is "
                "built from its model.yaml build context. With this flag, a "
                "missing image is a hard error instead."
            ),
        )
        p.add_argument(
            "--accept-degraded",
            dest="accept_degraded",
            action="store_true",
            default=False,
            help=(
                "Slurm backend: allow a SIF with no registry provenance "
                "(image identity 'unverified_local'), e.g. one built from a "
                "Singularity.def. Required because the Slurm executor defaults "
                "to strict image identity."
            ),
        )
        p.add_argument(
            "--validators",
            choices=["basic", "strict", "developer"],
            default=None,
            help="Validator level for the simple-mode runner (default: basic).",
        )
        p.add_argument(
            "--no-image-pull",
            dest="no_image_pull",
            action="store_true",
            default=False,
            help="Simple-mode: do not pull images, use only local copies.",
        )
        p.add_argument(
            "--backend",
            choices=["docker", "slurm"],
            default=None,
            help="Execution backend (default: reads from manifest globals, falls back to docker).",
        )

    import sys

    known_commands = [
        "run",
        "register-dataset",
        "preprocess-dataset",
        "register-model",
        "init-db",
        "models",
    ]

    if (
        len(sys.argv) > 1
        and sys.argv[1] not in known_commands
        and not sys.argv[1].startswith("-h")
    ):
        add_run_args(parser)
        args = parser.parse_args()
        execute_run(args)
        return

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Run models")
    add_run_args(run_parser)

    reg_parser = subparsers.add_parser(
        "register-dataset", help="Register a dataset from dataset.yaml"
    )
    reg_parser.add_argument(
        "--slug", required=False, help="Dataset slug under store/datasets/<slug>/"
    )
    reg_parser.add_argument(
        "--manifest", required=False, help="Explicit path to dataset.yaml"
    )
    reg_parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing registry row when manifest changed.",
    )

    preproc_parser = subparsers.add_parser(
        "preprocess-dataset",
        help="Fuse raw modality files into processed.h5mu (run after register-dataset)",
    )
    preproc_parser.add_argument(
        "--slug", required=False, help="Dataset slug under store/datasets/<slug>/"
    )
    preproc_parser.add_argument(
        "--manifest", required=False, help="Explicit path to dataset.yaml"
    )

    subparsers.add_parser("init-db", help="Initialize the registry database")

    migrate_ar_parser = subparsers.add_parser(
        "migrate-asset-registry",
        help="Copy datasets/models from legacy multiverse_state.db → asset_registry.db",
    )
    migrate_ar_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report row counts without writing",
    )
    migrate_ar_parser.add_argument(
        "--state-root",
        default=None,
        help="Override state root (default: MULTIVERSE_STATE_DIR or package default)",
    )

    model_reg_parser = subparsers.add_parser(
        "register-model", help="Register a model from model.yaml"
    )
    model_reg_parser.add_argument(
        "--slug", required=False, help="Model slug under store/models/<slug>/"
    )
    model_reg_parser.add_argument(
        "--manifest", required=False, help="Explicit path to model.yaml"
    )
    model_reg_parser.add_argument(
        "--build",
        action="store_true",
        help="Build local Docker image after registration",
    )
    model_reg_parser.add_argument(
        "--set-sif-path",
        default=None,
        help="After registration, update sif_path for this model in asset_registry.",
    )

    models_parser = subparsers.add_parser(
        "models", help="Model registry/build commands"
    )
    models_subparsers = models_parser.add_subparsers(
        dest="models_command", help="Models commands"
    )

    models_register = models_subparsers.add_parser(
        "register", help="Register a model from model.yaml"
    )
    models_register.add_argument(
        "--slug", required=False, help="Model slug under store/models/<slug>/"
    )
    models_register.add_argument(
        "--manifest", required=False, help="Explicit path to model.yaml"
    )
    models_register.add_argument(
        "--build",
        action="store_true",
        help="Build local Docker image after registration",
    )

    models_build = models_subparsers.add_parser(
        "build", help="Build a model image locally from model.yaml"
    )
    models_build.add_argument(
        "--slug", required=False, help="Model slug under store/models/<slug>/"
    )
    models_build.add_argument(
        "--manifest", required=False, help="Explicit path to model.yaml"
    )

    args = parser.parse_args()

    if args.command == "init-db":
        init_db()
        print("Database and directories initialized.")
    elif args.command == "register-dataset":
        init_db()
        manifest_path = resolve_manifest_path(
            manifest_path=args.manifest, slug=args.slug
        )
        try:
            result = register_from_manifest(
                str(manifest_path), update=True if args.update else None
            )
        except RuntimeError as exc:
            prompt = f"{exc} Update now? [y/N]: "
            choice = input(prompt).strip().lower()
            if choice in {"y", "yes"}:
                result = register_from_manifest(str(manifest_path), update=True)
            else:
                print("Skipped update.")
                return
        print(result["message"])
        print(
            f"Dataset slug '{result['slug']}' registered with ID: {result['dataset_id']}"
        )
    elif args.command == "preprocess-dataset":
        manifest_path = resolve_manifest_path(
            manifest_path=args.manifest, slug=args.slug
        )
        output = preprocess_dataset(str(manifest_path))
        print(f"Processed dataset written to: {output}")
    elif args.command == "run":
        execute_run(args)
    elif args.command == "register-model":
        model_manifest = resolve_model_manifest_path(
            manifest_path=args.manifest, slug=args.slug
        )
        result = register_model_from_manifest(
            str(model_manifest),
            build=args.build,
        )
        print(result["message"])
        print(
            f"Model slug '{result['slug']}' registered at version {result['version']}."
        )
        if getattr(args, "set_sif_path", None):
            from ..asset_registry import set_model_sif_path

            updated = set_model_sif_path(
                result["slug"], result["version"], args.set_sif_path
            )
            if updated:
                print(
                    f"sif_path updated to '{args.set_sif_path}' for {result['slug']}@{result['version']}."
                )
            else:
                print(
                    f"Warning: no row found to update sif_path for {result['slug']}@{result['version']}."
                )
    elif args.command == "migrate-asset-registry":
        from pathlib import Path as _Path

        from ..asset_registry import migrate_from_legacy_db
        from ..registry_db import DB_NAME as _DB_NAME

        state_root = _Path(args.state_root) if args.state_root else None
        legacy_db = _Path(_DB_NAME)
        if not legacy_db.is_file():
            print(f"Legacy DB not found at {legacy_db}; nothing to migrate.")
            raise SystemExit(0)
        try:
            counts = migrate_from_legacy_db(
                legacy_db, state_root=state_root, dry_run=args.dry_run
            )
        except RuntimeError as exc:
            print(f"Migration refused: {exc}")
            raise SystemExit(1) from exc
        tag = "[dry-run] " if args.dry_run else ""
        print(
            f"{tag}Migrated {counts['datasets']} dataset(s) and "
            f"{counts['models']} model(s) to asset_registry.db"
        )
    elif args.command == "models":
        if args.models_command == "register":
            model_manifest = resolve_model_manifest_path(
                manifest_path=args.manifest, slug=args.slug
            )
            result = register_model_from_manifest(
                str(model_manifest),
                build=args.build,
            )
            print(result["message"])
            if args.build:
                print(f"Built local image: {result['docker_image']}")
        elif args.models_command == "build":
            model_manifest = resolve_model_manifest_path(
                manifest_path=args.manifest, slug=args.slug
            )
            manifest = load_model_manifest(str(model_manifest))
            built = build_local_model(manifest)
            if built:
                print(f"Built local image: {built}")
            else:
                print("Remote image expected, skipping local build.")
        else:
            models_parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
