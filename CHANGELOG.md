# Changelog

## v2.2.0 - Planned: Architecture Hardening (not yet implemented)

Architectural risks identified in the v2.1.0 audit. Tracked here for roadmap visibility.

### Risk 1 — Container Bloat (HIGH) - addressed
Every model image ships the complete `multiverse` package (`runner/`, `gui.py`, `tracking.py`, etc.)
via `COPY multiverse/ /app/multiverse/` + `pip install -e .`. Containers only need ~4 files
(`runtime_io.py`, `base.py`, `logging_utils.py`, `data_utils.py`).
**Planned fix:** Extract `multiverse/worker/` as a minimal sub-package; update all Dockerfiles to
`COPY multiverse/worker/` only. This decouples orchestration changes from container rebuilds.

### Risk 2 — Promotion/DB Write Split (HIGH) - addressed
`shutil.move(workspace → artifact)` and `_persist_run_status(SUCCESS)` are two separate operations
with a crash window between them (`docker_runner.py:198` vs `docker_runner.py:401`). A crash between
them leaves `status=PROMOTING` in the DB while the artifact exists on disk. Additionally,
`get_db_connection()` opens a new SQLite connection per call with no WAL mode, causing
`database is locked` errors under parallel writes.
**Planned fix:** (1) Add `PRAGMA journal_mode=WAL; PRAGMA busy_timeout=10000` to `get_db_connection()`.
(2) Write `status=PROMOTING` to DB before `shutil.move`, `status=SUCCESS` after — both inside
`run_and_promote`. (3) Add `init_db()` recovery scan to heal `PROMOTING` rows on startup.

### Risk 3 — No Resource Guardrails on Parallel Dispatch (MEDIUM) - addressed
`asyncio.gather(*tasks)` starts all N containers simultaneously with no concurrency limit
(`docker_runner.py:425`). `mem_limit="16g"` is a per-container ceiling but does not prevent N×16 GB
simultaneous allocation exceeding host RAM.
**Planned fix:** Add `asyncio.Semaphore(max_parallel)` where `max_parallel = floor(available_ram / mem_limit)`,
computed via `psutil.virtual_memory()`. Expose `max_parallel_jobs` override in `run_manifest.yaml` globals.

### Risk 4 — `evaluate_model()` Not Enforced as a Contract (MEDIUM) - to be addressed
`ModelFactory.evaluate_model()` is not `@abstractmethod` — silent no-op if not overridden
(`base.py:185`). File write boilerplate and metrics gating are duplicated across all 6 models.
**Planned fix:** Rename to `_compute_model_metrics() -> dict` (abstract), seal `evaluate_model()`
in the base class as the single place that gates, writes, and handles IOError.

### Documentation
- Added `docs/ARCHITECTURE.md` with Mermaid.js job lifecycle sequence diagram, DB schema,
  container I/O contract table, and concurrency model explanation.
- Added `docs/CONTRIBUTING.md` with Diátaxis-structured model onboarding: 15-minute Python tutorial,
  R/Julia how-to, explanation of isolation rationale, and reference tables for manifest fields.
- Updated `README.md` header with language-agnostic value proposition and feature summary table.

---

## v2.1.0 - Gap Analysis Remediation

### Orchestration
- Extracted `DEFAULT_MODEL_IMAGES` as a module-level constant in `docker_runner.py`; `run_model_container()` now resolves image names via DB lookup with fallback to this constant, eliminating the duplicated inline map.
- All Docker images are now built/pulled in parallel at the start of a run (`build_images_concurrently`) before any container is dispatched — previously images were prepared per-job inline.
- The synchronous `execute_run()` path now routes all registry-based runs through `asyncio.run(run_workflow_async())`, ensuring parallel container execution everywhere; the old sequential fallback is removed.

### Pre-flight Validation
- Added `validate_pending_jobs()` in `cli.py` — called before image building — which checks each pending job against three rules:
  - **Omics compatibility**: skips jobs where the model's required omics are not a subset of the dataset's available omics (e.g. Cobolt needs RNA + ATAC).
  - **Batch key presence**: opens each dataset file once (via `h5py`, not full load) and skips jobs where the declared `batch_key` is absent from `obs`.
  - **Cell type / single batch warnings**: emits `[WARN]` (no skip) when `cell_type_key` is missing or the dataset has only one batch value.
- Dataset files are opened at most once per unique `dataset_id` across all pending jobs.

### End-of-run Summary
- Added `_print_run_summary()` in `cli.py` that prints a Rich table of all jobs (success / failed / skipped) with key metrics and accumulated pre-flight warnings after every run.

### Reproducibility
- Seeds are now enforced in all six model containers before model instantiation:
  - `pca.py`, `mofa.py`: `random.seed` + `np.random.seed`
  - `multivi.py`, `totalvi.py`: `scvi.settings.seed` (sets torch + numpy + random internally)
  - `mowgli.py`, `cobolt.py`: `random.seed` + `np.random.seed` + `torch.manual_seed`
- Seed value flows from `run_manifest.yaml` → `job_spec.json` → `build_model_config()` → each model's `main()`.

### Metrics Control
- `run_manifest.yaml` now supports a `globals.metrics` block (bio-conservation and batch-correction metric lists) and optional per-job `metrics` overrides.
- `_write_job_spec()` merges global + per-job metrics into `job_spec.json`; `build_model_config()` surfaces them as `config["metrics"]`.
- Each model's `evaluate_model()` gates metric computation against the requested list; absent or empty config keeps current defaults.
- Fixed `totalvi.py` `evaluate_model()` which previously returned an empty dict — now records `elbo_train` and `reconstruction_loss_train` from training history.

### Data Quality Warnings
- `dataloader.py`: replaced silent `np.zeros` dummy cell-type annotation with an explicit `logger.warning` and `"unknown"` string values.
- `data_utils.py` (`fuse_mudata`, `anndata_concatenate`): same — zeros replaced with `"unknown"` and logged.

### Tests
- Added `tests/unit/test_seeds.py` — 7 tests verifying each model calls the correct seed functions with the value from `job_spec`.
- Added `tests/unit/test_preflight_validation.py` — 8 tests covering omics skip, batch-key skip, cell-type warning, single-batch warning, and dataset-file deduplication.
- Added `tests/unit/test_metrics_config.py` — 9 tests covering `_write_job_spec` metrics field, `build_model_config` propagation, global/per-job merge, and per-model gating.
- Fixed `test_manifest_planner.py` and `test_planner.py`: updated in-memory SQLite schemas to match current `cli.py` queries (`slug`/`version`/`status` on models; `slug` on datasets; `model_slug`/`model_version` on runs); replaced unguarded `sys.modules['docker'] = MagicMock()` with guarded form to prevent test-ordering contamination.
- Fixed `test_docker_runner.py`: patched `_write_job_spec` and `run_and_promote` in the concurrency test to avoid hitting the real filesystem and Docker daemon.
- Fixed `test_builder.py`: added required `version` field to `ModelManifest` constructor; corrected `BuildSpec.dockerfile` path and updated `images.build` assertion to match the tar-based build API.

## v2.0.0 - Production Readiness Overhaul

### Registry & Store
- Migrated control-plane state to SQLite-backed registry tables for datasets, models, and runs.
- Standardized filesystem hierarchy under `store/`:
  - `store/datasets/` for dataset packages
  - `store/models/` for model packages
  - `store/workspaces/` for ephemeral run staging
  - `store/artifacts/` for promoted, immutable outputs
- Added workspace-to-artifact promotion semantics to guarantee only successful runs are promoted.

### Dynamic Registration
- Introduced manifest-driven onboarding for datasets (`dataset.yaml`) and models (`model.yaml`).
- Added idempotent registration using manifest hashing to skip unchanged entries.
- Added CLI + Makefile registration workflows for both data and models.

### Orchestration
- Shifted to a Zero-Path container contract:
  - input mounted as `/input/data.h5mu` (read-only)
  - output mounted as `/output/` (read-write)
  - orchestrator-written run config at `/output/job_spec.json`
- Added workspace promotion logic:
  - run in `store/workspaces/run_<id>/`
  - write logs and artifacts in workspace
  - promote to `store/artifacts/<experiment>/<dataset>/<model>/<run_id>/` only on success

### Features
- Added MLflow proxy tracking in orchestrator:
  - logs run params and metrics
  - attaches promoted artifact directory
  - degrades gracefully on MLflow connectivity/import failures
- Added Optuna sweep engine with persistent SQLite study storage and resumable studies.
- Added dynamic GUI/manifest-first planning support aligned with registry-driven execution.
- Added model-specific hyperparameter schema files under `schemas/models/` for built-in models.
- Updated Streamlit GUI to auto-render hyperparameter inputs from each model's schema.
- Added `make register-models` to batch-register all built-in models after manifest/schema updates.
