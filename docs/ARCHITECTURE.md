# Architecture Reference

Multi-verse is a **Registry-and-Store** MLOps platform. The orchestrator plans, validates,
and dispatches work; model containers are pure black boxes that read data and write results
through fixed mount paths. No orchestrator code runs inside a container.

---

## Core Concepts

| Concept | What it means |
|---|---|
| **Registry** | SQLite database (`mvexp_state.db`) tracking datasets, models, and runs |
| **Store** | Filesystem hierarchy under `store/` for raw data, workspaces, and artifacts |
| **Zero-Path Contract** | Containers read from `/input/data.h5mu`, write to `/output/`. No hardcoded paths anywhere else |
| **Workspace** | Ephemeral staging dir `store/workspaces/run_<id>/` used during execution |
| **Artifact** | Immutable promoted dir `store/artifacts/<experiment>/<dataset>/<model>/<run_id>/` |
| **Promotion** | `shutil.move(workspace → artifact)` — only happens on container exit 0 |

---

## Filesystem Hierarchy

```
mvexp_state.db                      ← SQLite registry (single file)
store/
  datasets/
    raw/                            ← registered .h5mu / .h5ad files
  models/
    pca/
      model.yaml                    ← model manifest (slug, version, image, omics)
      container/
        Dockerfile
        environment.yml
    multivi/  mofa/  mowgli/  ...
  workspaces/
    run_abc123def456/               ← live staging dir (deleted or promoted on exit)
      job_spec.json                 ← written by orchestrator before container start
      container.log
      embeddings.h5
      metrics.json
      umap.png
  artifacts/
    benchmark_run/                  ← experiment name
      pbmc10k/                      ← dataset slug
        pca/                        ← model slug
          run_abc123def456/         ← promoted workspace (immutable)
            run_manifest.yaml       ← copy of the manifest that produced this run
            job_spec.json
            embeddings.h5
            metrics.json
            umap.png
            container.log
```

---

## Job Lifecycle: Sequence Diagram

```mermaid
sequenceDiagram
    actor User
    participant GUI as Streamlit GUI
    participant CLI as Orchestrator CLI
    participant DB as SQLite Registry
    participant Docker as Docker Daemon
    participant Container as Model Container
    participant MLflow as MLflow Server

    User->>GUI: Select datasets + models + hyperparams
    GUI->>CLI: Write run_manifest.yaml, invoke `make benchmark`

    CLI->>DB: generate_execution_plan_from_manifest()
    DB-->>CLI: List of pending jobs (dataset × model pairs)

    CLI->>CLI: validate_pending_jobs()
    Note over CLI: Omics check → skip incompatible<br/>batch_key check → skip if absent<br/>cell_type / single-batch → warn only

    CLI->>Docker: build_images_concurrently(all image tags)
    Note over CLI,Docker: ALL images pulled/built before<br/>any container is dispatched

    CLI->>Docker: run_jobs_concurrently() via asyncio.gather + Semaphore
    Note over CLI: Semaphore(max_parallel) limits<br/>concurrent containers to available RAM

    loop For each runnable job (parallel)
        CLI->>CLI: _write_job_spec(workspace_dir, job, seed)
        Note over CLI: Writes seed, hyperparams, metrics<br/>config to /output/job_spec.json

        CLI->>Docker: containers.run(image, volumes, mem_limit)
        Docker->>Container: Start container
        Container->>Container: load_job_spec() → build_model_config()
        Container->>Container: set seeds (random, numpy, framework)
        Container->>Container: load_input_mudata(/input/data.h5mu)
        Container->>Container: model.train()
        Container->>Container: model.save_latent() → embeddings.h5
        Container->>Container: model.evaluate_model() → metrics.json
        Container->>Container: model.umap() → umap.png
        Container-->>Docker: exit 0 (success) or exit 1 (failure)

        Docker-->>CLI: exit_code via container.wait()
        CLI->>CLI: run_and_promote()
        Note over CLI: exit 0 → DB status=PROMOTING<br/>→ shutil.move(workspace→artifact)<br/>→ DB status=SUCCESS

        CLI->>MLflow: log_successful_run_to_mlflow() [best-effort]
        MLflow-->>CLI: Run URL (or silent skip if unavailable)

        CLI->>DB: _persist_run_status(SUCCESS/FAILED)
    end

    CLI->>User: _print_run_summary() Rich table
    Note over CLI,User: SUCCESS / FAILED / SKIPPED per job<br/>+ accumulated pre-flight warnings
```

---

## Database Schema

```sql
-- Registered datasets
CREATE TABLE datasets (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    slug           TEXT UNIQUE NOT NULL,     -- e.g. "pbmc10k"
    name           TEXT NOT NULL,
    path           TEXT NOT NULL,            -- absolute path to .h5mu
    omics_available TEXT NOT NULL,           -- JSON array: ["rna","atac"]
    batch_key      TEXT,                     -- obs column name for batch
    cell_type_key  TEXT,                     -- obs column name for cell type
    manifest_path  TEXT,
    manifest_hash  TEXT,
    status         TEXT NOT NULL             -- READY | PENDING | ERROR
);

-- Registered model containers
CREATE TABLE models (
    slug                    TEXT NOT NULL,   -- e.g. "pca"
    version                 TEXT NOT NULL,   -- semver: "1.0.0"
    name                    TEXT,
    docker_image            TEXT NOT NULL,   -- "multiverse-pca:1.0.0"
    image_digest            TEXT,
    supported_omics         TEXT NOT NULL,   -- JSON array: ["rna"]
    manifest_path           TEXT NOT NULL,
    manifest_hash           TEXT NOT NULL,
    hyperparameters_schema  TEXT,            -- path to JSON Schema file
    status                  TEXT NOT NULL,   -- ACTIVE | DEPRECATED
    PRIMARY KEY (slug, version)
);

-- Execution history
CREATE TABLE runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id    INTEGER REFERENCES datasets(id),
    model_slug    TEXT,
    model_version TEXT,
    model_name    TEXT,
    status        TEXT NOT NULL,             -- SUCCESS | FAILED | PROMOTING
    output_path   TEXT,
    FOREIGN KEY (model_slug, model_version) REFERENCES models(slug, version)
);
```

---

## Container I/O Contract

Every model container, regardless of language or framework, must honour this contract:

| Direction | Path | Format | Written by |
|---|---|---|---|
| Input | `/input/data.h5mu` | MuData HDF5 (read-only) | Orchestrator (volume mount) |
| Config | `/output/job_spec.json` | JSON (see below) | Orchestrator before container start |
| Embeddings | `/output/embeddings.h5` | HDF5 dataset `latent` shape `(n_cells, n_dims)` | Container |
| Metrics | `/output/metrics.json` | JSON flat dict of scalar floats | Container |
| UMAP | `/output/umap.png` | PNG | Container (optional) |
| Logs | `stdout/stderr` | Plain text | Container (captured by Docker) |

**`job_spec.json` schema:**

```json
{
  "seed": 42,
  "dataset_name": "pbmc10k",
  "model_name": "pca",
  "hyperparameters": {
    "n_components": 50,
    "learning_rate": 0.001
  },
  "metrics": {
    "model_metrics": ["total_variance"],
    "bio_conservation": ["silhouette_label"],
    "batch_correction": ["graph_connectivity"]
  }
}
```

The container **must** exit 0 on success and non-zero on failure. Partial output files in the workspace are discarded — only promoted artifacts are retained.

---

## Concurrency Model

```
asyncio event loop (single thread)
│
├── run_in_executor(ThreadPool) ──► ensure_image_prepared(tag)   ┐ parallel
├── run_in_executor(ThreadPool) ──► ensure_image_prepared(tag)   ┘ image pulls
│
└── asyncio.Semaphore(max_parallel)
    ├── run_in_executor ──► run_and_promote(container_A)   ┐ bounded
    ├── run_in_executor ──► run_and_promote(container_B)   │ parallel
    └── (queued) ──────────────────────────────────────────┘ container runs
```

`max_parallel` defaults to `floor(available_host_RAM / mem_limit_per_job)` and can be overridden in `run_manifest.yaml` under `globals.max_parallel_jobs`.

---

## Key Architectural Decisions

### Why SQLite and not PostgreSQL?

SQLite with WAL mode (`PRAGMA journal_mode=WAL`) supports concurrent readers and a single writer without blocking. For a single-machine benchmarking platform this is appropriate. If the platform scales to a distributed scheduler, the `registry_db.py` abstraction layer makes migration to PostgreSQL mechanical — only `get_db_connection()` changes.

### Why workspace-then-promote instead of writing directly to artifacts?

Guarantees that `store/artifacts/` contains only complete, successful runs. A failed or crashed container leaves a workspace dir (for debugging) but never corrupts the artifact store. The filesystem-level `os.rename` (same-filesystem) or `shutil.move` is the commit boundary.

### Why asyncio + ThreadPool instead of multiprocessing?

Model containers are independent OS processes managed by Docker. The orchestrator's async layer is purely I/O-bound — it submits `containers.run()`, then blocks on `container.wait()`. A ThreadPool inside `run_in_executor` handles the blocking Docker SDK calls without spawning additional Python processes.
