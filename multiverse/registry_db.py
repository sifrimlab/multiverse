"""Legacy registry — dataset and model tables (STRATEGY G6 / M5).

This module owns the **datasets** and **models** tables in the legacy
combined ``mvexp_state.db``. The canonical NEW location for this code
is :mod:`multiverse.asset_registry`, which writes to a separate
``asset_registry.db`` and does not share a DB file with the kernel's
projection index.

**Migration path**: run ``multiverse migrate-asset-registry`` to copy
the tables from this DB into ``asset_registry.db``, then update
callers to import from :mod:`multiverse.asset_registry`.

**What was removed in G6** (previously lines 248–658):
* ``_migrate_runs_table`` — legacy runners only; kernel uses journal.
* ``_migrate_run_metrics_table`` — legacy runners only.
* ``recover_orphaned_runs`` — superseded by
  :meth:`multiverse.mvd.Kernel.replay_from_journal` and the M3
  journaled-reservation-ledger.
* ``_build_recovery_docker_client`` — only used by
  ``recover_orphaned_runs``.
"""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

from .state_paths import PACKAGE_DIR as _PACKAGE_DIR
from .state_paths import find_legacy_db as _find_legacy_db
from .state_paths import resolve_state_root as _resolve_state_root

# ``BASE_DIR`` historically meant "the package install directory" *and*
# "the state directory" — that conflation was the M1 bug. Kept so
# existing imports type-check; new code uses
# ``state_paths.resolve_state_root()`` directly.
BASE_DIR = str(_PACKAGE_DIR.parent)

# Kept as rebindable names so ``monkeypatch.setattr(registry_db, "DB_NAME",
# ...)`` in the test suite continues to work.
_DEFAULT_STATE_ROOT = str(_resolve_state_root())
DB_NAME = os.path.join(_DEFAULT_STATE_ROOT, "mvexp_state.db")
STORE_DIR = os.path.join(_DEFAULT_STATE_ROOT, "store")
DATASETS_DIR = os.path.join(STORE_DIR, "datasets")
# Legacy/backwards-compatible scaffolding (issue #23): the current dataset
# contract keeps each dataset's files under its own ``store/datasets/<slug>/``
# folder, and model runs consume the processed ``.h5mu`` only. This global
# ``datasets/raw`` directory is no longer an active run input; it is still
# created by ``init_db`` for compatibility with older state layouts and tests
# that monkeypatch it. Do not add new code that depends on it.
RAW_DATASETS_DIR = os.path.join(DATASETS_DIR, "raw")
MODELS_DIR = os.path.join(STORE_DIR, "models")
ARTIFACTS_DIR = os.path.join(STORE_DIR, "artifacts")
WORKSPACES_DIR = os.path.join(STORE_DIR, "workspaces")


class LegacyStateDirError(RuntimeError):
    """Raised when a pre-M1 ``mvexp_state.db`` exists inside the package
    directory and the caller has not explicitly opted into using it."""


def _check_legacy_db_refusal() -> None:
    if os.environ.get("MVEXP_ALLOW_LEGACY_DB") == "1":
        return
    configured_db = os.path.abspath(DB_NAME)
    current_default = os.path.abspath(
        os.path.join(str(_resolve_state_root()), "mvexp_state.db")
    )
    if configured_db != current_default:
        return
    legacy = _find_legacy_db()
    if legacy is None:
        return
    if os.path.abspath(str(legacy)) == configured_db:
        return
    raise LegacyStateDirError(
        f"Refusing to open a fresh database at {configured_db!r} while a "
        f"pre-M1 database still exists at {str(legacy)!r}.\n"
        "\n"
        "Pick one:\n"
        "  1. Move it: run `mvexp migrate-state-dir` (recommended).\n"
        "  2. Keep using the legacy location: set MVEXP_STATE_DIR to "
        f"{str(legacy.parent)!r}.\n"
        "  3. Acknowledge and proceed regardless: set "
        "MVEXP_ALLOW_LEGACY_DB=1 (NOT recommended; orphans the legacy data).\n"
    )


def get_db_connection() -> sqlite3.Connection:
    """Return a WAL-mode connection to the legacy combined SQLite registry.

    New code should use
    :func:`multiverse.asset_registry.get_asset_registry_connection` instead.
    """
    _check_legacy_db_refusal()
    os.makedirs(os.path.dirname(DB_NAME), exist_ok=True)
    conn = sqlite3.connect(DB_NAME, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    """Initialize the legacy combined DB schema.

    Creates directories, then creates the datasets, models, and legacy
    runs/run_metrics stub tables in ``mvexp_state.db``.
    """
    for directory in [RAW_DATASETS_DIR, MODELS_DIR, ARTIFACTS_DIR, WORKSPACES_DIR]:
        os.makedirs(directory, exist_ok=True)

    conn = get_db_connection()
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

    # Legacy runs/run_metrics stubs — kept so callers that rely on
    # init_db() existing and creating these tables don't crash.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            model_slug TEXT,
            model_version TEXT,
            model_name TEXT,
            status TEXT NOT NULL,
            output_path TEXT,
            container_id TEXT,
            failure_reason TEXT,
            manifest_run_id TEXT,
            params_hash TEXT
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS run_metrics (
            run_id INTEGER NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            metric_kind TEXT,
            PRIMARY KEY (run_id, metric_name)
        )
    """
    )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Dataset / model helpers — kept for backward compat; use asset_registry
# for new code.
# ---------------------------------------------------------------------------


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
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS models (
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
    cursor.execute("PRAGMA table_info(models)")
    cols = {r[1] for r in cursor.fetchall()}
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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_slug ON models(slug)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_status ON models(status)")


def insert_dataset(
    name: str, path: str, omics_available: List[str], status: str = "READY"
) -> int:
    """Insert a dataset row into the legacy SQLite registry; returns new id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO datasets (name, path, omics_available, status) VALUES (?, ?, ?, ?)",
        (name, path, json.dumps(omics_available), status),
    )
    conn.commit()
    dataset_id = cursor.lastrowid
    conn.close()
    return int(dataset_id)


def get_dataset_by_slug(slug: str) -> Optional[Dict[str, Any]]:
    """Return one dataset dict by slug, or ``None`` if not found."""
    conn = get_db_connection()
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
) -> int:
    """Insert or update a dataset from a registration manifest; returns row id."""
    conn = get_db_connection()
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


def get_all_datasets() -> List[Dict[str, Any]]:
    """Return all dataset rows from the legacy registry."""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets")
    rows = cursor.fetchall()
    datasets = [dict(row) for row in rows]
    conn.close()
    return datasets


def get_all_models() -> List[Dict[str, Any]]:
    """Return the latest ACTIVE version of each model slug."""
    conn = get_db_connection()
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


def mark_dataset_removed(slug_or_id: "str | int") -> bool:
    """Soft-delete a dataset by slug or numeric id; returns whether a row changed."""
    conn = get_db_connection()
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


def mark_model_inactive(slug: str, version: Optional[str] = None) -> bool:
    """Mark model(s) inactive in legacy DB and best-effort in asset_registry."""
    conn = get_db_connection()
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

    # Keep deletion symmetric across both registries (issue #29): the legacy
    # DB and the canonical asset_registry must not diverge, otherwise a later
    # re-registration that only reconciles one DB reports success while the
    # model stays hidden. Best-effort: a missing/empty asset registry is fine.
    try:
        from .asset_registry import mark_model_inactive as _ar_mark_inactive

        if _ar_mark_inactive(slug, version):
            changed = True
    except Exception:
        pass

    return changed


def delete_run_by_id(run_id: int) -> bool:
    """Permanently remove a run record (and its metrics) from the legacy DB.

    Kept for old installations and legacy bookkeeping. Note that
    ``generate_execution_plan_from_manifest`` no longer consults the legacy
    ``runs`` table at all (STRATEGY: MVD Manifest Resume and Dedupe), so a row
    here never suppresses a manifest job; mvd-backed resume keys on durable
    ``ARTIFACT_SUCCESS`` state instead. Artifact files on disk are NOT removed —
    only the registry row is deleted.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM run_metrics WHERE run_id = ?", (run_id,))
    cursor.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    changed = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return changed


__all__ = [
    "BASE_DIR",
    "DB_NAME",
    "LegacyStateDirError",
    "STORE_DIR",
    "delete_run_by_id",
    "get_all_datasets",
    "get_all_models",
    "get_dataset_by_slug",
    "get_db_connection",
    "init_db",
    "insert_dataset",
    "mark_dataset_removed",
    "mark_model_inactive",
    "upsert_dataset_from_manifest",
]


if __name__ == "__main__":
    init_db()
    print(f"Database and directories initialized at {BASE_DIR}.")
