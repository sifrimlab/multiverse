import sqlite3

from multiverse import registry_db
from multiverse.runner.docker_runner import flatten_metric_rows


def test_run_metrics_table_created(tmp_path):
    db_path = str(tmp_path / "registry.db")
    original_db_name = registry_db.DB_NAME
    registry_db.DB_NAME = db_path
    try:
        registry_db.init_db()
    finally:
        registry_db.DB_NAME = original_db_name

    conn = sqlite3.connect(db_path)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(run_metrics)").fetchall()}
        assert {"run_id", "metric_name", "metric_value", "metric_kind"}.issubset(cols)
    finally:
        conn.close()


def test_flatten_metric_rows_handles_nested_scalars_and_history():
    rows = dict(
        (name, (value, kind))
        for name, value, kind in flatten_metric_rows(
            {
                "loss": 0.2,
                "nested": {"score": 0.8},
                "history": {"loss": [0.9, 0.4]},
                "nan_sanitized": None,
            }
        )
    )

    assert rows["loss"] == (0.2, "scalar")
    assert rows["nested.score"] == (0.8, "scalar")
    assert rows["history.loss"] == (0.4, "history_summary")
    assert rows["nan_sanitized"] == (None, "scalar")
