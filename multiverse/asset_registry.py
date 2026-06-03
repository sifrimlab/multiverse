"""User-managed asset registry: datasets and models (STRATEGY G6 / M5).

Owns the **datasets** and **models** SQLite tables. These are
user-supplied registrations — not journal projections — and are
therefore NOT rebuildable from the journal. The data here is canonical
and lives in its own DB file (``asset_registry.db``) under the state
root, separate from the kernel's ``mvexp_state.db`` projection.

Migration path from the legacy combined DB: run
``multiverse migrate-asset-registry`` to copy the tables across.

Sole-writer invariant: only this module and
:mod:`multiverse.index_projection` (via :mod:`multiverse.index`) write
to SQLite. No other module under ``multiverse/`` should embed raw
INSERT/UPDATE/DELETE/CREATE SQL.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .state_paths import resolve_state_root as _resolve_state_root

ASSET_REGISTRY_FILENAME = "asset_registry.db"

_DEFAULT_STATE_ROOT = _resolve_state_root()


def _asset_registry_path(state_root: Optional[Path] = None) -> Path:
    """Return the path to the asset registry DB for the given state root."""
    root = Path(state_root) if state_root else _DEFAULT_STATE_ROOT
    return root / ASSET_REGISTRY_FILENAME


def get_asset_registry_connection(
    state_root: Optional[Path] = None,
) -> sqlite3.Connection:
    """Return a WAL-mode connection to the asset registry DB."""
    db_path = _asset_registry_path(state_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# ---------------------------------------------------------------------------
# Schema creation / migration
# ---------------------------------------------------------------------------


def init_asset_registry(state_root: Optional[Path] = None) -> None:
    """Create tables if absent; apply column migrations for older DBs."""
    conn = get_asset_registry_connection(state_root)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            omics_available TEXT NOT NULL,
            batch_key TEXT,
            cell_type_key TEXT,
            manifest_path TEXT,
            manifest_hash TEXT,
            status TEXT NOT NULL
        )
    """
    )
    _ensure_dataset_columns(cursor)
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_datasets_slug_unique ON datasets(slug)"
    )

    _migrate_models_table(conn)

    conn.commit()
    conn.close()


def _ensure_dataset_columns(cursor: sqlite3.Cursor) -> None:
    """Backfill new dataset columns for older DBs."""
    cursor.execute("PRAGMA table_info(datasets)")
    existing = {row[1] for row in cursor.fetchall()}
    column_defs = {
        "slug": "TEXT",
        "batch_key": "TEXT",
        "cell_type_key": "TEXT",
        "manifest_path": "TEXT",
        "manifest_hash": "TEXT",
    }
    for col, col_type in column_defs.items():
        if col not in existing:
            cursor.execute(f"ALTER TABLE datasets ADD COLUMN {col} {col_type}")


def _migrate_models_table(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    # ``docker_image`` is nullable: an Apptainer-only model (registered with a
    # SIF and no Docker image) stores NULL here. The dual-runtime model
    # (Docker + SIF) stores both.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS models (
            slug TEXT NOT NULL,
            version TEXT NOT NULL,
            name TEXT,
            docker_image TEXT,
            image_digest TEXT,
            supported_omics TEXT NOT NULL,
            manifest_path TEXT NOT NULL,
            manifest_hash TEXT NOT NULL,
            hyperparameters_schema TEXT,
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            sif_path TEXT,
            gpu_required INTEGER DEFAULT 0,
            PRIMARY KEY (slug, version)
        )
        """
    )
    cursor.execute("PRAGMA table_info(models)")
    cols = {r[1] for r in cursor.fetchall()}
    # Legacy table migration path: models(name, docker_image, supported_omics)
    if "slug" not in cols or "version" not in cols:
        cursor.execute("ALTER TABLE models RENAME TO models_legacy")
        cursor.execute(
            """
            CREATE TABLE models (
                slug TEXT NOT NULL,
                version TEXT NOT NULL,
                name TEXT,
                docker_image TEXT NOT NULL,
                image_digest TEXT,
                supported_omics TEXT NOT NULL,
                manifest_path TEXT NOT NULL,
                manifest_hash TEXT NOT NULL,
                hyperparameters_schema TEXT,
                status TEXT NOT NULL DEFAULT 'ACTIVE',
                PRIMARY KEY (slug, version)
            )
            """
        )
        legacy_rows = cursor.execute(
            "SELECT name, docker_image, supported_omics FROM models_legacy"
        ).fetchall()
        for name, image, omics in legacy_rows:
            cursor.execute(
                """
                INSERT OR REPLACE INTO models
                (slug, version, name, docker_image, image_digest, supported_omics,
                 manifest_path, manifest_hash, hyperparameters_schema, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(name),
                    "0.0.0",
                    str(name),
                    image,
                    None,
                    omics if isinstance(omics, str) else json.dumps(omics),
                    f"legacy://{name}",
                    "legacy",
                    None,
                    "LEGACY",
                ),
            )
        cursor.execute("DROP TABLE models_legacy")
    # Idempotent new-column migrations (DBs created before H1).
    try:
        cursor.execute("ALTER TABLE models ADD COLUMN sif_path TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    try:
        cursor.execute("ALTER TABLE models ADD COLUMN gpu_required INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # column already exists
    # Relax a legacy ``docker_image NOT NULL`` constraint so Apptainer-only
    # models can store NULL. SQLite cannot drop a column constraint in place,
    # so rebuild the table when the old constraint is still present.
    cursor.execute("PRAGMA table_info(models)")
    docker_col = next((r for r in cursor.fetchall() if r[1] == "docker_image"), None)
    if docker_col is not None and docker_col[3] == 1:  # notnull flag set
        cursor.execute("ALTER TABLE models RENAME TO models_nn_old")
        cursor.execute(
            """
            CREATE TABLE models (
                slug TEXT NOT NULL,
                version TEXT NOT NULL,
                name TEXT,
                docker_image TEXT,
                image_digest TEXT,
                supported_omics TEXT NOT NULL,
                manifest_path TEXT NOT NULL,
                manifest_hash TEXT NOT NULL,
                hyperparameters_schema TEXT,
                status TEXT NOT NULL DEFAULT 'ACTIVE',
                sif_path TEXT,
                gpu_required INTEGER DEFAULT 0,
                PRIMARY KEY (slug, version)
            )
            """
        )
        cursor.execute(
            "INSERT INTO models (slug, version, name, docker_image, image_digest, "
            "supported_omics, manifest_path, manifest_hash, hyperparameters_schema, "
            "status, sif_path, gpu_required) "
            "SELECT slug, version, name, docker_image, image_digest, supported_omics, "
            "manifest_path, manifest_hash, hyperparameters_schema, status, sif_path, "
            "gpu_required FROM models_nn_old"
        )
        cursor.execute("DROP TABLE models_nn_old")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_slug ON models(slug)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_status ON models(status)")


# ---------------------------------------------------------------------------
# Dataset API
# ---------------------------------------------------------------------------


def insert_dataset(
    name: str,
    path: str,
    omics_available: List[str],
    status: str = "READY",
    *,
    state_root: Optional[Path] = None,
) -> int:
    """Insert a new dataset record. Returns the new dataset id."""
    conn = get_asset_registry_connection(state_root)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO datasets (name, path, omics_available, status) VALUES (?, ?, ?, ?)",
        (name, path, json.dumps(omics_available), status),
    )
    conn.commit()
    dataset_id = cursor.lastrowid
    conn.close()
    return int(dataset_id)


def get_dataset_by_slug(
    slug: str, *, state_root: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    conn = get_asset_registry_connection(state_root)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets WHERE slug = ? LIMIT 1", (slug,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def upsert_dataset_from_manifest(
    *,
    slug: str,
    name: str,
    path: str,
    omics_available: List[str],
    batch_key: Optional[str],
    cell_type_key: Optional[str],
    manifest_path: str,
    manifest_hash: str,
    status: str = "READY",
    state_root: Optional[Path] = None,
) -> int:
    """Idempotent upsert keyed by slug."""
    conn = get_asset_registry_connection(state_root)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM datasets WHERE slug = ? LIMIT 1", (slug,))
    row = cursor.fetchone()
    payload = (
        name,
        path,
        json.dumps(omics_available),
        batch_key,
        cell_type_key,
        manifest_path,
        manifest_hash,
        status,
        slug,
    )
    if row:
        dataset_id = int(row[0])
        cursor.execute(
            """
            UPDATE datasets
            SET name = ?, path = ?, omics_available = ?, batch_key = ?,
                cell_type_key = ?, manifest_path = ?, manifest_hash = ?, status = ?
            WHERE slug = ?
            """,
            payload,
        )
    else:
        cursor.execute(
            """
            INSERT INTO datasets
            (name, path, omics_available, batch_key, cell_type_key,
             manifest_path, manifest_hash, status, slug)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        dataset_id = int(cursor.lastrowid)
    conn.commit()
    conn.close()
    return dataset_id


def get_all_datasets(*, state_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return all dataset rows."""
    conn = get_asset_registry_connection(state_root)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets")
    rows = cursor.fetchall()
    datasets = [dict(row) for row in rows]
    conn.close()
    return datasets


def mark_dataset_removed(
    slug_or_id: "str | int", *, state_root: Optional[Path] = None
) -> bool:
    """Soft-remove a dataset; preserves historical run references."""
    conn = get_asset_registry_connection(state_root)
    cursor = conn.cursor()
    if isinstance(slug_or_id, int) or str(slug_or_id).isdigit():
        cursor.execute(
            "UPDATE datasets SET status = 'REMOVED' WHERE id = ?", (int(slug_or_id),)
        )
    else:
        cursor.execute(
            "UPDATE datasets SET status = 'REMOVED' WHERE slug = ?", (str(slug_or_id),)
        )
    changed = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return changed


# ---------------------------------------------------------------------------
# Model API
# ---------------------------------------------------------------------------


def get_all_models(*, state_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Fetch latest ACTIVE model version per slug."""
    conn = get_asset_registry_connection(state_root)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT m.*
        FROM models m
        WHERE m.status = 'ACTIVE'
          AND m.version = (
              SELECT MAX(m2.version)
              FROM models m2
              WHERE m2.slug = m.slug
                AND m2.status = 'ACTIVE'
          )
        ORDER BY m.slug
        """
    )
    rows = cursor.fetchall()
    models = [dict(row) for row in rows]
    conn.close()
    return models


def mark_model_inactive(
    slug: str, version: Optional[str] = None, *, state_root: Optional[Path] = None
) -> bool:
    """Soft-remove one model version, or all versions for a slug."""
    conn = get_asset_registry_connection(state_root)
    cursor = conn.cursor()
    if version:
        cursor.execute(
            "UPDATE models SET status = 'INACTIVE' WHERE slug = ? AND version = ?",
            (slug, version),
        )
    else:
        cursor.execute("UPDATE models SET status = 'INACTIVE' WHERE slug = ?", (slug,))
    changed = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return changed


# ---------------------------------------------------------------------------
# Migration helper (legacy combined DB → asset_registry.db)
# ---------------------------------------------------------------------------


def migrate_from_legacy_db(
    legacy_db_path: Path,
    *,
    state_root: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Copy datasets and models rows from a legacy mvexp_state.db.

    Returns ``{"datasets": N, "models": N}`` counts.
    Raises ``RuntimeError`` if the target asset_registry.db already has rows
    (refuse to run twice).
    """
    src = sqlite3.connect(str(legacy_db_path))
    src.row_factory = sqlite3.Row

    # Check if legacy tables exist.
    tables = {
        r[0]
        for r in src.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    copied: Dict[str, int] = {"datasets": 0, "models": 0}
    if "datasets" not in tables and "models" not in tables:
        src.close()
        return copied

    init_asset_registry(state_root)
    dst = get_asset_registry_connection(state_root)

    # Refuse to run twice.
    existing_ds = dst.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
    existing_md = dst.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    if existing_ds > 0 or existing_md > 0:
        dst.close()
        src.close()
        raise RuntimeError(
            "asset_registry.db already contains data; refusing to migrate again. "
            "Delete the file and retry if you need a fresh migration."
        )

    if dry_run:
        dst.close()
        src.close()
        if "datasets" in tables:
            copied["datasets"] = src.execute(
                "SELECT COUNT(*) FROM datasets"
            ).fetchone()[0]
        if "models" in tables:
            copied["models"] = src.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        return copied

    if "datasets" in tables:
        for row in src.execute("SELECT * FROM datasets").fetchall():
            keys = row.keys()
            placeholders = ", ".join("?" * len(keys))
            dst.execute(
                f"INSERT OR IGNORE INTO datasets ({', '.join(keys)}) VALUES ({placeholders})",
                tuple(row),
            )
        copied["datasets"] = dst.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]

    if "models" in tables:
        for row in src.execute("SELECT * FROM models").fetchall():
            keys = row.keys()
            placeholders = ", ".join("?" * len(keys))
            dst.execute(
                f"INSERT OR IGNORE INTO models ({', '.join(keys)}) VALUES ({placeholders})",
                tuple(row),
            )
        copied["models"] = dst.execute("SELECT COUNT(*) FROM models").fetchone()[0]

    dst.commit()
    dst.close()
    src.close()
    return copied


def get_model_sif_path(
    conn: sqlite3.Connection,
    slug: str,
    version: str,
) -> Optional[str]:
    """Return sif_path for the given model slug+version, or None."""
    conn.row_factory = None
    cursor = conn.cursor()
    cursor.execute(
        "SELECT sif_path FROM models WHERE slug = ? AND version = ? LIMIT 1",
        (slug, version),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None


def get_model_gpu_flag(
    conn: sqlite3.Connection,
    slug: str,
    version: str,
) -> bool:
    """Return gpu_required flag for the given model slug+version."""
    conn.row_factory = None
    cursor = conn.cursor()
    cursor.execute(
        "SELECT gpu_required FROM models WHERE slug = ? AND version = ? LIMIT 1",
        (slug, version),
    )
    row = cursor.fetchone()
    return bool(row[0]) if row else False


def set_model_sif_path(
    slug: str,
    version: str,
    sif_path: str,
    *,
    state_root: Optional[Path] = None,
) -> bool:
    """Update sif_path for an existing model row. Returns True if a row was updated."""
    conn = get_asset_registry_connection(state_root)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE models SET sif_path = ? WHERE slug = ? AND version = ?",
        (sif_path, slug, version),
    )
    changed = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return changed


__all__ = [
    "ASSET_REGISTRY_FILENAME",
    "get_asset_registry_connection",
    "get_all_datasets",
    "get_all_models",
    "get_dataset_by_slug",
    "get_model_gpu_flag",
    "get_model_sif_path",
    "init_asset_registry",
    "insert_dataset",
    "mark_dataset_removed",
    "mark_model_inactive",
    "migrate_from_legacy_db",
    "set_model_sif_path",
    "upsert_dataset_from_manifest",
]
