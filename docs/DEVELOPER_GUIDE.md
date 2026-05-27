# Developer Guide

This page is for maintainers and contributors. It collects the practical knowledge needed to work on the codebase without breaking the contracts that the rest of the platform depends on. For the high-level system map, see [Architecture](ARCHITECTURE.md).

## Local Development Setup

```bash
make install            # uv sync --group dev (no ML libraries)
make init               # create mvexp_state.db
make register-models    # populate the models table
make services-up        # MLflow + Optuna for integration work
# Optional, for full ML stack development:
uv sync --group dev --group ml-legacy
```

The two dependency groups are intentional. `dev` is enough for orchestrator, GUI, and registry work and installs in seconds. `ml-legacy` pulls scvi-tools, scanpy, muon, Mowgli, Cobolt, and torch and is needed only when working on the local (non-Docker) runner path or on the in-process model wrappers under `multiverse/models/`.

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
| `test_docker_runner.py` | Async supervision and promotion. |
| `test_local_runner.py` | The Python fallback path. |
| `test_tuner.py` | Optuna sampling and objective wiring. |
| `test_models_base.py` | `ModelFactory` lifecycle. |
| `test_worker_io.py` | `mvr-worker` SDK I/O helpers. |
| `test_event_stream.py` | JSON event emission to the GUI. |
| `test_metrics_config.py`, `test_run_metrics.py` | Metric gating and on-disk format. |
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
| Reporting | Each run's metrics row in MLflow ties back to the artifact bundle on disk. |

## Code Layout

The application package lives under `multiverse/`. The container SDK lives under `sdk/mvr-worker/` and is published into model images at build time. Schemas live under `schemas/`. Container build recipes live under `store/models/<slug>/container/` and observability Dockerfiles live under `docker-env/`.

See [Architecture — Repository Layout](ARCHITECTURE.md#repository-layout) for the annotated tree.

## Things to Get Right

1. **Atomicity around promotion.** The runner writes `runs.status = SUCCESS` only after `rename(2)`-ing the workspace into the artifact tree. Reorder these and you create the possibility of an orphaned successful row pointing at a missing directory.
2. **Determinism.** Every model container must apply the seed from `job_spec.json` before any stochastic call. Tests exercise this; do not regress.
3. **Concurrency.** SQLite is opened with WAL mode and a 30s busy timeout. Long write transactions in tooling will be felt by the GUI. Keep writes small.
4. **Metric gating.** `evaluate.determine_valid_metrics()` is what prevents misleading numbers when metadata is missing. New metric families should plug into the same gating function.
5. **No hidden state.** The registry is the source of truth. Functions that quietly inspect the filesystem to "figure out" what is registered are bugs in waiting.

## Style Notes

- Code targets Python 3.12. Use the standard typing features.
- Prefer Pydantic models at parsing boundaries (`DatasetManifest`, `ParsedManifest`) over dictionary access.
- The runner emits machine-readable events on stderr (one JSON object per line). Don't print free-form text alongside them; tests consume the stream.
- Container code should import only from `mvr_worker` and the model's own dependency stack. Importing from the host `multiverse` package inside a container will fail at runtime.

## Release Practice

Tag versions, update `CHANGELOG.md`, and ensure that the bundled images are tagged with their `model.yaml` versions. Run artifacts already record image tag, model version, and contract version; for a release to remain reproducible, those tags must continue to resolve.
