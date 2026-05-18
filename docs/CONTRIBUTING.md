# Contributing

This guide explains how to contribute to mvexp without weakening the researcher-facing workflow. Contributions should preserve the GUI-first, reproducible, notebook-compatible experience.

## Documentation Principles

Use Diátaxis:

| Type | Use for |
|---|---|
| Tutorial | First successful outcome. |
| How-to | Task-oriented recipes. |
| Reference | Complete field lists and contracts. |
| Explanation | Design rationale and tradeoffs. |

Write for academic bioinformaticians first. Translate infrastructure details into benefits: reproducibility, fewer path errors, comparable model outputs, and publication-ready provenance.

## Code Contribution Checklist

- Keep the GUI path working for researchers.
- Preserve Zero-Path execution for models.
- Write or update tests for planner, ingestion, metrics, or model contract changes.
- Update docs when user-visible behavior changes.
- Avoid adding required manual Docker or CLI steps to normal workflows.

## Documentation Checklist

- Include screenshot placeholders where a GUI action matters, for example `[IMAGE: The Job Builder Matrix]`.
- Include a copy-pasteable Hello World example when introducing a new contract.
- Include a Common Errors section.
- Include citation or publishing guidance when the feature affects results.
- List structured fields in tables so pages are searchable.

## How-To: Add a Built-In Model

1. Add the model package under `store/models/<slug>/`.
2. Add or update `schemas/models/<slug>.hyperparameters.schema.json`.
3. Ensure the runtime follows [Model Container Contract](MODEL_CONTAINER_CONTRACT.md).
4. Register it through the GUI Registry tab or maintainer tooling.
5. Verify it appears in Job Builder and Parameters.
6. Run a small Hello World dataset.
7. Confirm `embeddings.h5`, `metrics.json`, and comparison report entries are produced.
8. Update [Models Glossary](reference/MODELS_GLOSSARY.md).

## How-To: Add a Metric or Report Field

1. Define whether it is bio-conservation, batch-correction, or model-specific.
2. Document required metadata keys.
3. Ensure missing metadata leads to a clear warning, not a silent failure.
4. Surface it in Results or MLflow where appropriate.
5. Update [Evaluation Metrics](reference/EVALUATION_METRICS.md).

## Common Errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Docs and GUI disagree | UI changed without doc update. | Update the relevant tutorial/how-to in the same PR. |
| New model is hard to configure | Schema fields are missing or unclear. | Add defaults, enums, minimums, and descriptions. |
| Researcher must use Docker manually | Workflow leaked implementation detail. | Route the task through GUI or maintainer-only docs. |
| Report metric is not interpretable | Metric lacks biological explanation. | Add definition, assumptions, and caveats. |

## How to Cite Contributions

If your contribution adds a scientific method, cite the original method and document how mvexp calls it. If you publish results generated with a development version, cite the commit hash and archive the run artifacts.
