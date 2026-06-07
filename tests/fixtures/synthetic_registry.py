"""Populate a SQLite registry with synthetic data for scale testing.

Usage
-----
    # Default: 100 datasets x 50 models x 1000 runs into ./multiverse_state.db
    python -m tests.fixtures.synthetic_registry

    # Custom counts and target path
    python -m tests.fixtures.synthetic_registry \
        --datasets 100 --models 50 --runs 1000 \
        --db /tmp/scale_test.db

Reproduces the GUI Strategy's "500-runs test" (Phase 2 exit criterion):
load every Streamlit tab in <3 s at 100 datasets x 50 models x 1000 runs.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path

from multiverse import registry_db

_OMICS_POOL = ["rna", "atac", "adt"]
_STATUSES = ["SUCCESS", "FAILED", "RUNNING", "QUEUED"]
_STATUS_WEIGHTS = [0.70, 0.20, 0.05, 0.05]
_FAILURE_REASONS = [
    "VALIDATION_ERROR: missing batch_key 'batch'",
    "VALIDATION_ERROR: cell_type_key 'cell_type' not found in adata.obs",
    "RuntimeError: CUDA out of memory",
    "TimeoutError: container exceeded 3600s wall clock",
    "ImportError: scvi-tools mismatch",
]


def _omics_subset(rng: random.Random) -> list[str]:
    k = rng.randint(1, len(_OMICS_POOL))
    return rng.sample(_OMICS_POOL, k=k)


def seed_registry(
    db_path: Path,
    *,
    n_datasets: int,
    n_models: int,
    n_runs: int,
    rng_seed: int = 42,
) -> None:
    rng = random.Random(rng_seed)
    # Point registry_db at the target path before running migrations so the
    # schema is identical to what the live GUI sees.
    original_db = registry_db.DB_NAME
    registry_db.DB_NAME = str(db_path)
    try:
        registry_db.init_db()
        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            "DELETE FROM run_metrics; DELETE FROM runs; DELETE FROM models; DELETE FROM datasets;"
        )

        # Datasets
        ds_rows = []
        for i in range(1, n_datasets + 1):
            slug = f"synthetic-ds-{i:04d}"
            name = f"Synthetic Dataset {i:04d}"
            omics = _omics_subset(rng)
            ds_rows.append(
                (
                    i,
                    slug,
                    name,
                    f"store/datasets/{slug}/data.h5mu",
                    json.dumps(omics),
                    "batch",
                    "cell_type",
                    f"store/datasets/{slug}/dataset.yaml",
                    f"hash{i:08x}",
                    "READY",
                )
            )
        conn.executemany(
            "INSERT INTO datasets "
            "(id, slug, name, path, omics_available, batch_key, cell_type_key, manifest_path, manifest_hash, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ds_rows,
        )

        # Models
        md_rows = []
        for j in range(1, n_models + 1):
            slug = f"synthetic-model-{j:03d}"
            omics = _omics_subset(rng)
            md_rows.append(
                (
                    slug,
                    "1.0.0",
                    f"Synthetic Model {j:03d}",
                    f"multiverse-{slug}:1.0.0",
                    None,
                    json.dumps(omics),
                    f"store/models/{slug}/model.yaml",
                    f"hash{j:08x}",
                    None,
                    "ACTIVE",
                )
            )
        conn.executemany(
            "INSERT INTO models "
            "(slug, version, name, docker_image, image_digest, supported_omics, manifest_path, manifest_hash, hyperparameters_schema, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            md_rows,
        )

        # Runs
        artifact_root = Path("store/artifacts")
        run_rows = []
        for r in range(1, n_runs + 1):
            ds_id = rng.randint(1, n_datasets)
            mod_idx = rng.randint(1, n_models)
            mod_slug = f"synthetic-model-{mod_idx:03d}"
            mod_name = f"Synthetic Model {mod_idx:03d}"
            status = rng.choices(_STATUSES, weights=_STATUS_WEIGHTS, k=1)[0]
            output_path = (
                str(artifact_root / f"synthetic_run_{r:06d}")
                if status == "SUCCESS"
                else None
            )
            failure = rng.choice(_FAILURE_REASONS) if status == "FAILED" else None
            run_rows.append(
                (
                    r,
                    ds_id,
                    mod_slug,
                    "1.0.0",
                    mod_name,
                    status,
                    output_path,
                    None,
                    failure,
                )
            )
        conn.executemany(
            "INSERT INTO runs "
            "(run_id, dataset_id, model_slug, model_version, model_name, status, output_path, container_id, failure_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            run_rows,
        )

        # A handful of run_metrics rows so the metrics-aware UI paths exercise too.
        metric_rows = []
        for r in range(1, min(n_runs, 200) + 1):
            for metric in ("ari", "nmi", "silhouette_score"):
                metric_rows.append((r, metric, rng.uniform(0.0, 1.0), "scalar"))
        conn.executemany(
            "INSERT INTO run_metrics (run_id, metric_name, metric_value, metric_kind) VALUES (?, ?, ?, ?)",
            metric_rows,
        )

        conn.commit()
        conn.close()
    finally:
        registry_db.DB_NAME = original_db


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", type=int, default=100)
    p.add_argument("--models", type=int, default=50)
    p.add_argument("--runs", type=int, default=1000)
    p.add_argument("--db", type=Path, default=Path("multiverse_state.db"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    seed_registry(
        args.db,
        n_datasets=args.datasets,
        n_models=args.models,
        n_runs=args.runs,
        rng_seed=args.seed,
    )
    print(
        f"Seeded {args.datasets} datasets, {args.models} models, {args.runs} runs into {args.db}"
    )


if __name__ == "__main__":
    main()
