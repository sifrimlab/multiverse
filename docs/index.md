# multiverse

**Reproducible benchmarking for multimodal single-cell integration.**

multiverse is an MLOps platform for academic single-cell integration studies. It sits between two notebook sessions — the one in which you curate a dataset and the one in which you interpret the resulting embeddings — and replaces the brittle hand-rolled scripts in between with a registry, a containerized runner, and a Streamlit interface. The goal is to make a defensible benchmark easier to run, and the resulting Methods section easier to write.

## Two Audiences, One Platform

The documentation is organized around two distinct audiences. Some pages are useful to both; where that is the case, terminology is introduced gently.

### For Bioinformaticians

You work in `AnnData`, `MuData`, Scanpy, and Jupyter. You want to compare integration models on your data without becoming a Docker user.

Start at [Getting Started](GETTING_STARTED.md) for an end-to-end walkthrough, then consult [Data Preparation](DATA_PREPARATION.md) for recipes covering RNA, RNA+ATAC, and RNA+ADT studies. The [Models Glossary](reference/MODELS_GLOSSARY.md) and [Evaluation Metrics](reference/EVALUATION_METRICS.md) reference pages describe what each model assumes and what each metric measures.

### For MLOps and Platform Engineers

You will be running, deploying, or extending the platform. You may have no biological background, and that is fine — the system is a typed Python application built around a SQLite registry, a Docker-based runner, MLflow, Optuna, and a Streamlit front end.

Start at [Architecture](ARCHITECTURE.md) for the system map, then read [Runner & Orchestration](RUNNER.md) for the execution model and [Model Container Contract](MODEL_CONTAINER_CONTRACT.md) for the I/O boundary. [Adding a Model](ADDING_A_MODEL.md) and the [Developer Guide](DEVELOPER_GUIDE.md) cover extension work.

## What multiverse Does

| Researcher concern | Platform responsibility |
|---|---|
| Biological question and dataset curation | Dataset registration, omics-compatibility checks |
| Batch and cell-type metadata | Metric eligibility gating and clear warnings |
| Model choice and hyperparameters | Container-isolated, parallel execution with seed enforcement |
| Comparing embeddings and metrics | Results tables, artifacts, MLflow tracking, Optuna sweeps |
| Writing a reproducible Methods section | `run_manifest.yaml`, `job_spec.json`, metrics, logs, provenance |

## Quick Start

Prerequisites: Python 3.12+, [`uv`](https://docs.astral.sh/uv/), Docker with Compose v2.

```bash
make bootstrap      # install dev deps, create SQLite registry, register built-in models
make services-up    # optional: start MLflow (:25000) and Optuna Dashboard (:28080)
make setup          # optional: install GUI/local-runner extras
make gui            # launch the Streamlit GUI (:28501)
```

Open `http://localhost:28501` and follow the [Getting Started](GETTING_STARTED.md) tutorial. For headless use, run `uv run multiverse --help` and `uv run multiverse run --manifest run_manifest.yaml --output store/artifacts/run_output`.

## Documentation Map (Diátaxis)

| Type | Page | Audience |
|---|---|---|
| Tutorial | [Getting Started](GETTING_STARTED.md) | Bio |
| How-to | [Data Preparation](DATA_PREPARATION.md) | Bio |
| How-to | [Data Registration](DATA_REGISTRATION.md) | Bio |
| How-to | [Benchmarking](BENCHMARKING.md) | Bio |
| How-to | [Adding a Model](ADDING_A_MODEL.md) | Ops |
| Reference | [Models Glossary](reference/MODELS_GLOSSARY.md) | Bio |
| Reference | [Evaluation Metrics](reference/EVALUATION_METRICS.md) | Bio |
| Reference | [GUI](GUI.md) | Bio / Ops |
| Reference | [Run Manifest](RUN_MANIFEST.md) | Bio / Ops |
| Reference | [Model Container Contract](MODEL_CONTAINER_CONTRACT.md) | Ops |
| Reference | [Model Registration](MODEL_REGISTRATION.md) | Ops |
| Reference | [Runner](RUNNER.md) | Ops |
| Explanation | [Architecture](ARCHITECTURE.md) | Ops |
| Explanation | [Observability](OBSERVABILITY.md) | Ops |
| Explanation | [Developer Guide](DEVELOPER_GUIDE.md) | Ops |
| Process | [Contributing](CONTRIBUTING.md) | All |

## License

Distributed under the MIT License. See `LICENSE` for details.
