from __future__ import annotations

import argparse
from pathlib import Path

from multiverse.registry_db import get_db_connection, init_db
from multiverse.runner.docker_runner import flatten_metric_rows
from multiverse.tracking import load_run_metrics


def rebuild_run_metrics() -> int:
    init_db()
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM run_metrics")
        rows = conn.execute(
            "SELECT run_id, output_path FROM runs WHERE status = 'SUCCESS' AND output_path IS NOT NULL"
        ).fetchall()
        inserted = 0
        for run_id, output_path in rows:
            metrics = load_run_metrics(str(output_path))
            for metric_name, metric_value, metric_kind in flatten_metric_rows(metrics):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO run_metrics
                    (run_id, metric_name, metric_value, metric_kind)
                    VALUES (?, ?, ?, ?)
                    """,
                    (run_id, metric_name, metric_value, metric_kind),
                )
                inserted += 1
        conn.commit()
        return inserted
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild run_metrics from successful run artifacts.")
    parser.parse_args()
    inserted = rebuild_run_metrics()
    print(f"Inserted {inserted} run_metrics rows.")


if __name__ == "__main__":
    main()
