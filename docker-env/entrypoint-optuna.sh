#!/bin/sh
set -e

DATA_DIR="${DATA_DIR:-/data}"
DB_PATH="$DATA_DIR/optuna.db"

# Bootstrap the DB with Optuna's own schema so the dashboard can open it
# with skip_table_creation=True even before any study has been created.
python3 - <<'PYEOF'
import os
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
db = os.environ.get("DATA_DIR", "/data") + "/optuna.db"
optuna.storages.RDBStorage(f"sqlite:///{db}")
PYEOF

exec optuna-dashboard "sqlite:///$DB_PATH" --host 0.0.0.0 --port 8080
