# Developer Guide

This page is for maintainers and contributors. It collects the practical knowledge needed to work on the codebase without breaking the contracts that the rest of the platform depends on. For the high-level system map, see [Architecture](ARCHITECTURE.md).

## Local Development Setup

```bash
make install            # uv sync --group dev (no ML libraries)
make init               # create mvexp_state.db + asset_registry.db
make register-models    # populate the models table
make services-up        # MLflow + Optuna for integration work
# Optional, for in-process model wrapper development:
uv sync --group dev --group ml-legacy
```

The two dependency groups are intentional. `dev` is enough for orchestrator, GUI, registry, and Slurm path work and installs in seconds. `ml-legacy` pulls scvi-tools, scanpy, muon, Mowgli, Cobolt, and torch and is needed only when working on the in-process model wrappers under `multiverse/models/`.

## Test Suite

```bash
make test                              # all tests
uv run pytest tests/unit                # unit tests only
uv run pytest tests/integration         # integration (Docker required)
uv run pytest tests/gui                 # Playwright-driven GUI tests
```

Test layout:

```text
tests/
  unit/                Pure-Python unit tests, no Docker, no network.
  integration/         End-to-end runs through the orchestrator.
  gui/                 Playwright + Streamlit assertions.
  conftest.py          Shared fixtures.
  fixtures/            Small synthetic AnnData / MuData inputs.
  simulate_dashboard.py, verify_gui.py   Manual harnesses.
```

Notable unit tests:

| Test | Covers |
|---|---|
| `test_planner.py` | Run plan generation from a manifest. |
| `test_manifest_gate.py` | Pre-flight validation. |
| `test_mvd_docker_executor.py` | mvd-backed Docker execution state flow. |
| `test_mvd_slurm_executor.py` | Slurm executor state flow and dual-digest manifest. |
| `test_docker_supervisor.py` | Container labels, leases, RealDockerEngine adapter, and cancellation. |
| `test_promotion_saga.py` | Promotion, validation, and quarantine behavior. |
| `test_accept_degraded.py` | M2 fail-closed guard: `unverified_local` images are rejected by default in both Docker and Slurm executors. |
| `test_user_id_propagation.py` | `user_id` threaded through journal records → artifact manifest. |
| `test_reservation_timeline_rebuild.py` | `rebuild_index` populates `reservation_events` from journal. |
| `test_sqlite_writer_isolation.py` | Sole-writer CI gate: no raw SQL mutations outside the designated writer modules. |
| `test_recovery_no_destruction.py` | `rebuild_index` classifies incomplete promotions as `RECOVERY_PENDING` without deleting data. |
| `test_tuner.py` | Optuna sampling and objective wiring. |
| `test_models_base.py` | `ModelFactory` lifecycle. |
| `test_worker_io.py` | `mvr-worker` SDK I/O helpers. |
| `test_metrics_config.py` | Metric gating and on-disk format. |
| `test_validator.py` | Omics / batch / cell-type validation. |
| `test_ingestion.py` | Dataset and model registration. |

## Core Boundaries

These contracts are the ones to respect when refactoring:

| Boundary | Contract |
|---|---|
| Dataset ingestion | `dataset.yaml` plus prepared `.h5ad` / `.h5mu` under `store/datasets/<slug>/`. |
| Model registration | `model.yaml` plus a JSON Schema for hyperparameters; see [Model Registration](MODEL_REGISTRATION.md). |
| Container runtime | `/input/data.h5mu`, `/output/job_spec.json`, `/output/`; see [Model Container Contract](MODEL_CONTAINER_CONTRACT.md). |
| Evaluation | Embedding row order matches `obs`; metrics are gated by `determine_valid_metrics()`. |
| Reporting | Artifact bundles are authoritative; MLflow is a projection. |
| Image identity | The Docker executor defaults open (`accept_degraded=True`) because locally-built images are the normal research workflow. Pass `--strict` to require a registry digest. The Slurm executor defaults closed (`accept_degraded=False`) because HPC runs are expected to have provenance; pass `--accept-degraded` to override. |
| Sole-writer invariant | Raw SQL mutations (`INSERT`, `UPDATE`, `DELETE`, `CREATE TABLE`) must only appear in `index/`, `index_projection`, `asset_registry`, `registry_db`, or `models_ingest`. The `test_sqlite_writer_isolation.py` CI gate enforces this; do not bypass it. |
| Asset registry split | Dataset/model catalog rows live in `asset_registry.db` (written by `asset_registry.py`). The run index lives in `mvexp_state.db` (written by `index/`). Do not merge them back. |

## Code Layout

The application package lives under `multiverse/`. The container SDK lives under `sdk/mvr-worker/` and is published into model images at build time. Schemas live under `schemas/`. Container build recipes live under `store/models/<slug>/container/` and observability Dockerfiles live under `docker-env/`.

See [Architecture — Repository Layout](ARCHITECTURE.md#repository-layout) for the annotated tree.

## Things to Get Right

1. **Promotion saga integrity.** A run is successful only after validation and atomic promotion produce a verified artifact manifest. Do not reintroduce direct workspace moves into final artifact paths.
2. **Determinism.** Every model container must apply the seed from `job_spec.json` before any stochastic call. Tests exercise this; do not regress.
3. **Control-plane ownership.** GUI and CLI execution should go through the mvd kernel/client path, not direct Docker or ad-hoc subprocess ownership.
4. **Metric gating.** `evaluate.determine_valid_metrics()` is what prevents misleading numbers when metadata is missing. New metric families should plug into the same gating function.
5. **Rebuildable state.** SQLite is an index. Artifact manifests and journals are the durable recovery inputs.

## Style Notes

- Code targets Python 3.12. Use the standard typing features.
- Prefer Pydantic models at parsing boundaries (`DatasetManifest`, `ParsedManifest`) over dictionary access.
- Kernel/client paths return structured state. Keep command output predictable; tests consume CLI and client surfaces.
- Container code should import only from `mvr_worker` and the model's own dependency stack. Importing from the host `multiverse` package inside a container will fail at runtime.

## Release Practice

Tag versions, update `CHANGELOG.md`, and ensure that the bundled images are tagged with their `model.yaml` versions. Run artifacts already record image tag, model version, and contract version; for a release to remain reproducible, those tags must continue to resolve.
