import os
import sqlite3
import json
from typing import List, Optional, Dict, Any

# Calculate base directory relative to this file's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_NAME = os.path.join(BASE_DIR, "mvexp_state.db")
STORE_DIR = os.path.join(BASE_DIR, "store")
DATASETS_DIR = os.path.join(STORE_DIR, "datasets")
RAW_DATASETS_DIR = os.path.join(DATASETS_DIR, "raw")
MODELS_DIR = os.path.join(STORE_DIR, "models")
ARTIFACTS_DIR = os.path.join(STORE_DIR, "artifacts")
WORKSPACES_DIR = os.path.join(STORE_DIR, "workspaces")

def get_db_connection() -> sqlite3.Connection:
    """Return a WAL-mode connection to the SQLite registry.

    WAL mode allows concurrent readers while the single background writer holds
    the write lock.  busy_timeout retries reads for up to 10 s before raising
    OperationalError, eliminating spurious "database is locked" errors during
    parallel model runs.
    """
    conn = sqlite3.connect(DB_NAME, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA synchronous=NORMAL")   # safe with WAL; faster than FULL
    return conn

def init_db():
    """Initializes the database schema and creates necessary directories."""
    # Create directories
    for directory in [RAW_DATASETS_DIR, MODELS_DIR, ARTIFACTS_DIR, WORKSPACES_DIR]:
        os.makedirs(directory, exist_ok=True)

    # Initialize DB
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
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
    """)
    _ensure_dataset_columns(cursor)
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_datasets_slug_unique ON datasets(slug)")

    _migrate_models_table(conn)
    _migrate_runs_table(conn)
    _migrate_run_metrics_table(conn)

    conn.commit()
    conn.close()

    recover_orphaned_runs()


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
                (slug, version, name, docker_image, image_digest, supported_omics, manifest_path, manifest_hash, hyperparameters_schema, status)
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


def _migrate_runs_table(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
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
            FOREIGN KEY (dataset_id) REFERENCES datasets(id),
            FOREIGN KEY (model_slug, model_version) REFERENCES models(slug, version)
        )
        """
    )
    cursor.execute("PRAGMA table_info(runs)")
    cols = {r[1] for r in cursor.fetchall()}
    if "model_slug" not in cols or "model_version" not in cols:
        cursor.execute("ALTER TABLE runs RENAME TO runs_legacy")
        cursor.execute(
            """
            CREATE TABLE runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER,
                model_slug TEXT,
                model_version TEXT,
                model_name TEXT,
                status TEXT NOT NULL,
                output_path TEXT,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id),
                FOREIGN KEY (model_slug, model_version) REFERENCES models(slug, version)
            )
            """
        )
        legacy_rows = cursor.execute(
            "SELECT run_id, dataset_id, model_name, status, output_path FROM runs_legacy"
        ).fetchall()
        for run_id, dataset_id, model_name, status, output_path in legacy_rows:
            match = cursor.execute(
                "SELECT slug, version FROM models WHERE slug = ? ORDER BY version DESC LIMIT 1",
                (model_name,),
            ).fetchone()
            model_slug = match[0] if match else model_name
            model_version = match[1] if match else "0.0.0"
            cursor.execute(
                """
                INSERT INTO runs (run_id, dataset_id, model_slug, model_version, model_name, status, output_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, dataset_id, model_slug, model_version, model_name, status, output_path),
            )
        cursor.execute("DROP TABLE runs_legacy")
        cursor.execute("PRAGMA table_info(runs)")
        cols = {r[1] for r in cursor.fetchall()}

    column_defs = {
        "container_id": "TEXT",
        "failure_reason": "TEXT",
        "manifest_run_id": "TEXT",
        "params_hash": "TEXT",
    }
    for col, col_type in column_defs.items():
        if col not in cols:
            cursor.execute(f"ALTER TABLE runs ADD COLUMN {col} {col_type}")

def _migrate_run_metrics_table(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS run_metrics (
            run_id INTEGER NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            metric_kind TEXT,
            PRIMARY KEY (run_id, metric_name),
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_run_metrics_name ON run_metrics(metric_name)")

def insert_dataset(name: str, path: str, omics_available: List[str], status: str = "READY"):
    """Inserts a new dataset record into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO datasets (name, path, omics_available, status) VALUES (?, ?, ?, ?)",
        (name, path, json.dumps(omics_available), status)
    )
    conn.commit()
    dataset_id = cursor.lastrowid
    conn.close()
    return dataset_id


def get_dataset_by_slug(slug: str) -> Optional[Dict[str, Any]]:
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
    """Idempotent upsert keyed by slug."""
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
            SET name = ?, path = ?, omics_available = ?, batch_key = ?, cell_type_key = ?,
                manifest_path = ?, manifest_hash = ?, status = ?
            WHERE slug = ?
            """,
            payload,
        )
    else:
        cursor.execute(
            """
            INSERT INTO datasets
            (name, path, omics_available, batch_key, cell_type_key, manifest_path, manifest_hash, status, slug)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        dataset_id = int(cursor.lastrowid)
    conn.commit()
    conn.close()
    return dataset_id

def get_all_datasets() -> List[Dict]:
    """Fetches all datasets from the database."""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets")
    rows = cursor.fetchall()
    datasets = [dict(row) for row in rows]
    conn.close()
    return datasets

def get_all_models() -> List[Dict]:
    """Fetch latest ACTIVE model version per slug from the database."""
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


def mark_dataset_removed(slug_or_id: str | int) -> bool:
    """Soft-remove a dataset while preserving historical run references."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if isinstance(slug_or_id, int) or str(slug_or_id).isdigit():
        cursor.execute("UPDATE datasets SET status = 'REMOVED' WHERE id = ?", (int(slug_or_id),))
    else:
        cursor.execute("UPDATE datasets SET status = 'REMOVED' WHERE slug = ?", (str(slug_or_id),))
    changed = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return changed


def mark_model_inactive(slug: str, version: Optional[str] = None) -> bool:
    """Soft-remove one model version, or all versions for a slug."""
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
    return changed

def _build_recovery_docker_client() -> Any | None:
    """Return a live Docker client for recovery, or None if Docker is unavailable."""
    try:
        import docker  # type: ignore

        client = docker.from_env()
        client.ping()
        return client
    except Exception:
        return None


def recover_orphaned_runs(docker_client: Any = None, reattach_callback: Any = None) -> int:
    """Heal runs left in non-terminal FSM states by a crashed orchestrator.

    Per STRATEGY v2 §3 "Replace destructive recovery": this function never
    deletes result-like data. A ``PROMOTING`` row whose artifact
    directory does not carry a ``.promotion_complete`` marker is
    classified as ``FAILED`` / ``INCOMPLETE_PROMOTION`` and the half-built
    artifact directory is *moved* into ``store/quarantine/<date>/<id>/``
    via the quarantine subsystem. A ``.quarantined`` tombstone is left at
    the original path so users following the path see what happened.

    ``RUNNING`` rows are still inspected against Docker. Missing/dead
    containers are marked ``FAILED / ORPHANED``; live containers can be
    re-attached via the supplied callback. No deletion occurs in either
    branch.

    Returns the number of rows reconciled.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)

    # Lazy-import the quarantine subsystem so test environments that
    # don't have it available (or that monkey-patch DB_NAME pre-init) can
    # still call this function.
    from pathlib import Path as _Path
    try:
        from .promotion.layout import StoreLayout as _StoreLayout
        from .promotion.quarantine import quarantine_directory as _quarantine_directory
        from .promotion.errors import OwnershipMismatchError as _OwnershipMismatchError
    except ImportError:  # pragma: no cover — defensive
        _StoreLayout = None  # type: ignore[assignment]
        _quarantine_directory = None  # type: ignore[assignment]
        _OwnershipMismatchError = Exception  # type: ignore[assignment]

    if docker_client is None:
        docker_client = _build_recovery_docker_client()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(runs)")
    cols = {row[1] for row in cursor.fetchall()}
    for col, col_type in {"container_id": "TEXT", "failure_reason": "TEXT"}.items():
        if col not in cols:
            cursor.execute(f"ALTER TABLE runs ADD COLUMN {col} {col_type}")
    cursor.execute(
        """
        SELECT run_id, status, output_path, container_id
        FROM runs
        WHERE status IN ('RUNNING', 'PROMOTING', 'QUEUED')
        """
    )
    rows = cursor.fetchall()
    healed = 0

    for run_id, status, output_path, container_id in rows:
        if status == "PROMOTING":
            marker = os.path.join(output_path or "", ".promotion_complete")
            if output_path and os.path.isfile(marker):
                cursor.execute(
                    "UPDATE runs SET status = 'SUCCESS', failure_reason = NULL WHERE run_id = ?",
                    (run_id,),
                )
                _log.warning("Recovered run %s -> SUCCESS (promotion marker present)", run_id)
            else:
                # Quarantine the half-built artifact dir (STRATEGY v2 §3).
                # No more rmtree in the recovery hot path; the user owns
                # the decision about whether to adopt or gc.
                if (
                    output_path
                    and os.path.isdir(output_path)
                    and _quarantine_directory is not None
                ):
                    try:
                        layout = _StoreLayout(root=_Path(STORE_DIR)).ensure()
                        report = _quarantine_directory(
                            source=_Path(output_path),
                            layout=layout,
                            reason="recovery: PROMOTING without .promotion_complete",
                            physical_attempt_id=f"legacy_run_{run_id}",
                            extra_report=(
                                "Recovered by registry_db.recover_orphaned_runs(). "
                                "The promotion saga did not commit before the "
                                "previous orchestrator died. Adopt via the GUI's "
                                "Recovered Runs tab, or export and gc."
                            ),
                        )
                        _log.warning(
                            "Quarantined run %s artifact dir to %s",
                            run_id,
                            report.quarantine_path,
                        )
                    except _OwnershipMismatchError as exc:
                        _log.warning(
                            "Run %s: refused to quarantine %s (%s); leaving in place",
                            run_id,
                            output_path,
                            exc,
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        _log.warning(
                            "Run %s: quarantine failed (%s); leaving %s in place",
                            run_id,
                            exc,
                            output_path,
                        )
                cursor.execute(
                    """
                    UPDATE runs
                    SET status = 'FAILED', failure_reason = 'INCOMPLETE_PROMOTION'
                    WHERE run_id = ?
                    """,
                    (run_id,),
                )
                _log.warning("Recovered run %s -> FAILED (incomplete promotion)", run_id)
            healed += 1
            continue

        if status == "RUNNING":
            if container_id and docker_client is None:
                _log.warning(
                    "Leaving run %s in RUNNING during recovery; Docker client unavailable",
                    run_id,
                )
                continue
            container = None
            if docker_client is not None and container_id:
                try:
                    container = docker_client.containers.get(container_id)
                    container.reload()
                except Exception:
                    container = None
            container_status = getattr(container, "status", None)
            if container is not None and container_status in {"created", "running", "restarting"}:
                if reattach_callback is not None:
                    reattach_callback(container, run_id, output_path)
                    healed += 1
                continue
            cursor.execute(
                """
                UPDATE runs
                SET status = 'FAILED', failure_reason = 'ORPHANED'
                WHERE run_id = ?
                """,
                (run_id,),
            )
            _log.warning("Recovered run %s -> FAILED (container missing/dead)", run_id)
            healed += 1
            continue

        if status == "QUEUED":
            _log.warning("Found queued run %s during recovery; planner will re-evaluate it", run_id)
            healed += 1

    if healed:
        conn.commit()
        _log.warning("Recovered %d run(s) from previous crash.", healed)
    conn.close()
    return healed


if __name__ == "__main__":
    init_db()
    print(f"Database and directories initialized at {BASE_DIR}.")
