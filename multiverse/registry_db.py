import os
import sqlite3
import json
from typing import List, Optional

# Calculate base directory relative to this file's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_NAME = os.path.join(BASE_DIR, "mvexp_state.db")
STORE_DIR = os.path.join(BASE_DIR, "store")
DATASETS_DIR = os.path.join(STORE_DIR, "datasets")
RAW_DATASETS_DIR = os.path.join(DATASETS_DIR, "raw")
MODELS_DIR = os.path.join(STORE_DIR, "models")
ARTIFACTS_DIR = os.path.join(STORE_DIR, "artifacts")

def get_db_connection():
    """Returns a connection to the SQLite database."""
    return sqlite3.connect(DB_NAME)

def init_db():
    """Initializes the database schema and creates necessary directories."""
    # Create directories
    for directory in [RAW_DATASETS_DIR, MODELS_DIR, ARTIFACTS_DIR]:
        os.makedirs(directory, exist_ok=True)

    # Initialize DB
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            omics_available TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            name TEXT PRIMARY KEY,
            docker_image TEXT NOT NULL,
            supported_omics TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            model_name TEXT,
            status TEXT NOT NULL,
            output_path TEXT,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id),
            FOREIGN KEY (model_name) REFERENCES models(name)
        )
    """)

    conn.commit()
    conn.close()

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

if __name__ == "__main__":
    init_db()
    print(f"Database and directories initialized at {BASE_DIR}.")
