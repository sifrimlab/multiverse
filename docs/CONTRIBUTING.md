# Contributing

This guide explains how to contribute to Multiverse. Contributions should preserve the GUI-first reproducible experience.

## Code Contribution Checklist

- Keep the GUI path working for all users.
- Preserve Zero-Path execution for models.
- Write or update tests for planner, ingestion, metrics, or model contract changes.
- Update docs when user-visible behavior changes.
- Avoid adding required manual Docker or CLI steps to normal workflows.

## Documentation Checklist

- Include screenshot placeholders where a GUI action matters (e.g. `[IMAGE: Configure compatibility matrix]`).
- Include a copy-pasteable Hello World example when introducing a new contract.
- Include a Common Errors section.
- Include citation or publishing guidance when the feature affects results.
- List structured fields in tables so pages are searchable.

## How-To: Add a Built-In Model

1. Add the model package under `store/models/<slug>/` (`model.yaml`, `container/Dockerfile`, `container/environment.yml`, `container/run.py`).
2. Add or update `store/models/<slug>/hyperparameters.schema.json`.
3. Ensure the container follows [Model Container Contract](MODEL_CONTAINER_CONTRACT.md).
4. Register it with `make register-model slug=<slug>`.
5. Build the image with `make build-<slug>` and confirm it pulls cleanly.
6. Open the GUI, refresh the registry, and verify it renders in **Configure** with typed parameter controls.
7. Run a small benchmark from the test fixtures and confirm artifacts appear under `store/artifacts/`.
8. Update [Models Glossary](reference/MODELS_GLOSSARY.md).

## How-To: Add a Metric or Report Field

1. Document required metadata keys.
2. Ensure missing metadata leads to a clear warning, not a silent failure.
3. Surface it in Results or MLflow where appropriate.
4. Update [Evaluation Metrics](reference/EVALUATION_METRICS.md).

## How to Cite Contributions

If your contribution adds a scientific method, cite the original method and document how multiverse calls it. If you publish results generated with a development version, cite the commit hash and archive the run artifacts.
