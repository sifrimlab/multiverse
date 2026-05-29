#!/bin/sh
set -e

DATA_DIR="${DATA_DIR:-/data}"
DB_PATH="$DATA_DIR/mlflow.db"
ARTIFACT_ROOT="$DATA_DIR/artifacts"
mkdir -p "$ARTIFACT_ROOT"

# WAL mode is a persistent DB property: set it once here so all
# subsequent connections (MLflow's SQLAlchemy pool included) inherit it.
python3 - <<'PYEOF'
import sqlite3, os
db = os.environ.get("DATA_DIR", "/data") + "/mlflow.db"
conn = sqlite3.connect(db)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")
conn.commit()
conn.close()
PYEOF

exec mlflow server \
    --backend-store-uri "sqlite:///$DB_PATH" \
    --serve-artifacts \
    --artifacts-destination "$ARTIFACT_ROOT" \
    --host 0.0.0.0 \
    --port 5000
