"""SQLite as a rebuildable projection (STRATEGY S2).

This module is the *only* component that writes to SQLite during normal
operation. Per R1 the kernel does not open SQLite at all in the hot path;
the index is updated by the kernel's index-projection plugin after every
state transition, and a full rebuild is run by ``multiverse rebuild-index``
(this module's :func:`open_index` is its only persistence dependency).

Schema is intentionally tiny — every column is reconstructable from the
journal or the artifact manifest. Index columns are denormalized for GUI
listing performance.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

INDEX_FILENAME = "multiverse_state.db"
SCHEMA_VERSION = "4"


_SCHEMA_SQL = """
-- WAL + synchronous=NORMAL: readers (GUI, CLI, doctor) never block the
-- single writer, and writers never block readers. The durability trade-off
-- (a crash can lose the last few transactions) is acceptable because the
-- index is a projection — it is fully rebuildable from the journal and the
-- artifact tree, so it is allowed to be stale or lost.
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    physical_attempt_id   TEXT PRIMARY KEY,
    logical_run_id        TEXT,
    primary_state         TEXT NOT NULL,
    failure_reason        TEXT,
    artifact_dir          TEXT,
    workspace_dir         TEXT,
    manifest_path         TEXT,
    cancel_requested      INTEGER NOT NULL DEFAULT 0,
    submitted_wall_iso    TEXT,
    last_seq              INTEGER NOT NULL DEFAULT 0,
    options_json          TEXT,
    user_id               TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_state ON runs (primary_state);
CREATE INDEX IF NOT EXISTS idx_runs_logical ON runs (logical_run_id);

CREATE TABLE IF NOT EXISTS run_projections (
    physical_attempt_id   TEXT NOT NULL,
    plugin                TEXT NOT NULL,
    status                TEXT NOT NULL,
    last_seq              INTEGER NOT NULL DEFAULT 0,
    details_json          TEXT,
    PRIMARY KEY (physical_attempt_id, plugin),
    FOREIGN KEY (physical_attempt_id) REFERENCES runs (physical_attempt_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reservation_events (
    physical_attempt_id TEXT NOT NULL,
    seq                 INTEGER NOT NULL,
    kind                TEXT NOT NULL,
    wall_iso            TEXT NOT NULL,
    ram_bytes           INTEGER,
    gpu_index           INTEGER,
    release_reason      TEXT,
    PRIMARY KEY (physical_attempt_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_resv_attempt
    ON reservation_events (physical_attempt_id);

CREATE TABLE IF NOT EXISTS rebuild_reports (
    rebuilt_at_iso        TEXT PRIMARY KEY,
    total_runs            INTEGER NOT NULL,
    artifact_success      INTEGER NOT NULL,
    recovery_pending      INTEGER NOT NULL,
    failed                INTEGER NOT NULL,
    cancelled             INTEGER NOT NULL,
    other                 INTEGER NOT NULL,
    notes_json            TEXT
);
"""


@dataclass
class SqliteIndex:
    """Thin wrapper around the SQLite connection.

    ``open_index`` creates the schema if absent and stamps the schema
    version. Callers use context managers for cursors so the WAL stays
    healthy.
    """

    path: Path
    conn: sqlite3.Connection

    # ---- lifecycle ----

    def close(self) -> None:
        """Commit any open transaction and close the connection."""
        if self.conn is not None:
            self.conn.commit()
            self.conn.close()

    def __enter__(self) -> "SqliteIndex":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ---- run upserts ----

    def upsert_run(self, record: Dict[str, Any]) -> None:
        """Insert or update one run row from a projected fact dict.

        Idempotent: re-applying an older record never regresses progress.
        ``last_seq`` only ever advances (``MAX``) and ``user_id`` is never
        cleared (``COALESCE``), so replaying journal records out of order or
        more than once converges to the same row.

        Args:
            record: Projected facts for one attempt, keyed by column name.
                ``physical_attempt_id`` and ``primary_state`` are required;
                the rest are optional and default per the schema.
        """
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO runs (
                    physical_attempt_id, logical_run_id, primary_state,
                    failure_reason, artifact_dir, workspace_dir, manifest_path,
                    cancel_requested, submitted_wall_iso, last_seq, options_json,
                    user_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(physical_attempt_id) DO UPDATE SET
                    logical_run_id     = excluded.logical_run_id,
                    primary_state      = excluded.primary_state,
                    failure_reason     = excluded.failure_reason,
                    artifact_dir       = excluded.artifact_dir,
                    workspace_dir      = excluded.workspace_dir,
                    manifest_path      = excluded.manifest_path,
                    cancel_requested   = excluded.cancel_requested,
                    submitted_wall_iso = excluded.submitted_wall_iso,
                    last_seq           = MAX(runs.last_seq, excluded.last_seq),
                    options_json       = excluded.options_json,
                    user_id            = COALESCE(excluded.user_id, runs.user_id)
                """,
                (
                    record["physical_attempt_id"],
                    record.get("logical_run_id"),
                    record["primary_state"],
                    record.get("failure_reason"),
                    record.get("artifact_dir"),
                    record.get("workspace_dir"),
                    record.get("manifest_path"),
                    int(bool(record.get("cancel_requested"))),
                    record.get("submitted_wall_iso"),
                    int(record.get("last_seq") or 0),
                    json.dumps(record.get("options") or {}, sort_keys=True),
                    record.get("user_id"),
                ),
            )

    def set_projection(
        self,
        *,
        physical_attempt_id: str,
        plugin: str,
        status: str,
        last_seq: int = 0,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record the status of one projection plugin for one attempt.

        Args:
            physical_attempt_id: The attempt this projection status belongs to.
            plugin: Projection plugin name (e.g. MLflow, Optuna, SQLite index).
            status: Plugin-reported status string for the attempt.
            last_seq: Highest journal seq this status reflects; only advances.
            details: Optional plugin-specific detail, stored as sorted JSON.
        """
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO run_projections (
                    physical_attempt_id, plugin, status, last_seq, details_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(physical_attempt_id, plugin) DO UPDATE SET
                    status       = excluded.status,
                    last_seq     = MAX(run_projections.last_seq, excluded.last_seq),
                    details_json = excluded.details_json
                """,
                (
                    physical_attempt_id,
                    plugin,
                    status,
                    int(last_seq),
                    json.dumps(details or {}, sort_keys=True),
                ),
            )

    # ---- queries ----

    def get_run(self, physical_attempt_id: str) -> Optional[Dict[str, Any]]:
        """Return the run row as a dict, or ``None`` if the attempt is unknown.

        Args:
            physical_attempt_id: The attempt to look up.

        Returns:
            Column-name-keyed dict for the run, or ``None`` if absent.
        """
        cur = self.conn.execute(
            "SELECT * FROM runs WHERE physical_attempt_id = ?",
            (physical_attempt_id,),
        )
        row = cur.fetchone()
        return _row_to_dict(row, cur.description)

    def list_runs(
        self,
        *,
        primary_state: Optional[str] = None,
        logical_run_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """List run rows, optionally filtered, in submission order.

        Args:
            primary_state: If given, only runs in this state are returned.
            logical_run_id: If given, only attempts of this logical run.
            limit: Optional maximum number of rows.

        Returns:
            Run rows as dicts, ordered by submission time then attempt id.
        """
        where: List[str] = []
        params: List[Any] = []
        if primary_state is not None:
            where.append("primary_state = ?")
            params.append(primary_state)
        if logical_run_id is not None:
            where.append("logical_run_id = ?")
            params.append(logical_run_id)
        sql = "SELECT * FROM runs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY submitted_wall_iso ASC, physical_attempt_id ASC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        cur = self.conn.execute(sql, params)
        return [_row_to_dict(row, cur.description) for row in cur.fetchall()]

    def projections_for(self, physical_attempt_id: str) -> Dict[str, str]:
        """Return a ``{plugin: status}`` map of projections for one attempt.

        Args:
            physical_attempt_id: The attempt whose projection statuses to read.

        Returns:
            Mapping from projection plugin name to its last recorded status.
        """
        cur = self.conn.execute(
            "SELECT plugin, status FROM run_projections WHERE physical_attempt_id = ?",
            (physical_attempt_id,),
        )
        return {plugin: status for plugin, status in cur.fetchall()}

    def upsert_reservation_event(
        self,
        *,
        physical_attempt_id: str,
        seq: int,
        kind: str,
        wall_iso: str,
        ram_bytes: Optional[int] = None,
        gpu_index: Optional[int] = None,
        release_reason: Optional[str] = None,
    ) -> None:
        """Record one broker reservation event (grant or release).

        Keyed by ``(physical_attempt_id, seq)`` so replaying the journal
        re-applies the same event idempotently.

        Args:
            physical_attempt_id: The attempt the lease event belongs to.
            seq: Journal seq of the event; part of the primary key.
            kind: ``"granted"`` or ``"released"``.
            wall_iso: Wall-clock timestamp of the event.
            ram_bytes: RAM held by the lease, for grants.
            gpu_index: GPU index held by the lease, for grants.
            release_reason: Why the lease was released, for releases.
        """
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO reservation_events
                    (physical_attempt_id, seq, kind, wall_iso,
                     ram_bytes, gpu_index, release_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    physical_attempt_id,
                    int(seq),
                    kind,
                    wall_iso,
                    ram_bytes,
                    gpu_index,
                    release_reason,
                ),
            )

    def list_reservation_events(self, physical_attempt_id: str) -> List[Dict[str, Any]]:
        """Return the reservation timeline for one attempt in seq order.

        Args:
            physical_attempt_id: The attempt whose lease events to read.

        Returns:
            Reservation event rows as dicts, ordered by ``seq`` ascending.
        """
        cur = self.conn.execute(
            "SELECT * FROM reservation_events WHERE physical_attempt_id = ? "
            "ORDER BY seq ASC",
            (physical_attempt_id,),
        )
        return [_row_to_dict(row, cur.description) for row in cur.fetchall()]

    def truncate_reservation_events(self) -> None:
        """Delete all reservation events, e.g. before a partial rebuild."""
        with self.conn:
            self.conn.execute("DELETE FROM reservation_events")

    # ---- rebuild bookkeeping ----

    def record_rebuild_report(self, report: Dict[str, Any]) -> None:
        """Persist a summary row from a full index rebuild.

        Args:
            report: Summary dict as produced by ``RebuildResult.summary_dict``;
                ``rebuilt_at_iso`` and ``total_runs`` are required.
        """
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO rebuild_reports (
                    rebuilt_at_iso, total_runs, artifact_success,
                    recovery_pending, failed, cancelled, other, notes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report["rebuilt_at_iso"],
                    int(report["total_runs"]),
                    int(report.get("artifact_success", 0)),
                    int(report.get("recovery_pending", 0)),
                    int(report.get("failed", 0)),
                    int(report.get("cancelled", 0)),
                    int(report.get("other", 0)),
                    json.dumps(report.get("notes") or [], sort_keys=True),
                ),
            )

    def delete_run(self, physical_attempt_id: str) -> bool:
        """Remove a single run from the index.

        The FK ``ON DELETE CASCADE`` constraint automatically removes the
        associated rows in ``run_projections`` and ``reservation_events``.
        Artifact files on disk are NOT removed. Returns True if a row was
        deleted.
        """
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM runs WHERE physical_attempt_id = ?",
                (physical_attempt_id,),
            )
        return cur.rowcount > 0

    def truncate_runs(self) -> None:
        """Used by ``rebuild_index`` before replaying the journal in
        full-rebuild mode. The kernel must be paused (R1 maintenance lock)
        before calling this."""
        with self.conn:
            self.conn.execute("DELETE FROM reservation_events")
            self.conn.execute("DELETE FROM run_projections")
            self.conn.execute("DELETE FROM runs")

    @contextmanager
    def cursor(self) -> Any:
        """Yield a cursor that is closed on exit.

        Yields:
            A SQLite cursor on this index's connection.
        """
        cur = self.conn.cursor()
        try:
            yield cur
        finally:
            cur.close()


def _row_to_dict(row, description) -> Optional[Dict[str, Any]]:
    """Map a SQLite row to a column-name-keyed dict (``None`` passes through)."""
    if row is None:
        return None
    return {col[0]: row[i] for i, col in enumerate(description)}


def open_index(
    path: Path,
    *,
    create_if_missing: bool = True,
) -> SqliteIndex:
    """Open the SQLite index, creating the schema if needed.

    Stamps ``SCHEMA_VERSION`` on first open; a mismatch with the version
    on disk raises ``RuntimeError`` (the user is expected to run
    ``multiverse rebuild-index`` after upgrading).
    """
    path = Path(path)
    if not path.exists() and not create_if_missing:
        raise FileNotFoundError(f"index does not exist: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA_SQL)
    cur = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'")
    row = cur.fetchone()
    if row is None:
        with conn:
            conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
    elif row[0] != SCHEMA_VERSION:
        actual = row[0]
        if actual in ("2", "3"):
            with conn:
                if actual == "2":
                    # G2: add user_id to runs.
                    try:
                        conn.execute("ALTER TABLE runs ADD COLUMN user_id TEXT")
                    except Exception:
                        pass
                # G3: add reservation_events table.
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS reservation_events (
                        physical_attempt_id TEXT NOT NULL,
                        seq                 INTEGER NOT NULL,
                        kind                TEXT NOT NULL,
                        wall_iso            TEXT NOT NULL,
                        ram_bytes           INTEGER,
                        gpu_index           INTEGER,
                        release_reason      TEXT,
                        PRIMARY KEY (physical_attempt_id, seq)
                    );
                    CREATE INDEX IF NOT EXISTS idx_resv_attempt
                        ON reservation_events (physical_attempt_id);
                    """
                )
                conn.execute(
                    "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
                    (SCHEMA_VERSION,),
                )
        else:
            conn.close()
            raise RuntimeError(
                f"index schema version {actual!r} != expected {SCHEMA_VERSION!r}; "
                "run multiverse rebuild-index after upgrading"
            )
    return SqliteIndex(path=path, conn=conn)
