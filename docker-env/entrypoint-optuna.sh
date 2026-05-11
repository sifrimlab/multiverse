#!/bin/sh
set -e

DATA_DIR="${DATA_DIR:-/data}"
DB_PATH="$DATA_DIR/optuna.db"

# Pre-create the DB with WAL mode so the dashboard starts even before
# the orchestrator creates its first study.
python3 - <<'PYEOF'
import sqlite3, os
db = os.environ.get("DATA_DIR", "/data") + "/optuna.db"
conn = sqlite3.connect(db)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")
conn.commit()
conn.close()
PYEOF

exec optuna-dashboard "sqlite:///$DB_PATH" --host 0.0.0.0 --port 8080
