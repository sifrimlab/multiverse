# Evaluation Metrics

This reference explains the biological and technical metrics used to compare model outputs. The platform evaluates successful model runs from their saved latent embeddings, then records metrics for comparison in the results tables and MLflow when tracking is enabled.

## Required Dataset Metadata

Evaluation depends on metadata registered with the dataset:

| Metadata key | Purpose |
|---|---|
| `batch_key` | Observation column identifying experimental batch, donor, technology, or another known nuisance grouping. Required for batch-correction metrics. |
| `cell_type_key` | Observation column identifying biological labels. Used by supervised bio-conservation metrics. |

**Important note:** If `cell_type` or `batch` is missing, Multiverse can still run models, but the evaluation metrics pipeline cannot run without having both available. To circumvent this limitation in the scib-metrics source code, we assign random labels for the samples for each of the missing keys (for now). **Therefore, in case a label is misisng, the results shown for that missing label might be misleading.**

We use Bio-conservation and Batch-correction metrics from [scib-metrics](https://scib-metrics.readthedocs.io/en/stable/api.html). For more information about these metrics, please consult the scib-metrics package.


## Interpretation Notes

Use bio-conservation and batch-correction metrics together. A model can mix batches well by erasing meaningful cell-type structure, or preserve biology while leaving strong technical separation. The most defensible interpretation is therefore comparative: inspect both metric groups across the same dataset, metadata keys, and selected model set.

Metric availability is part of the result. If a supervised metric is absent, first check whether the dataset was registered with a valid `cell_type_key` and contains more than one unique value. If batch-correction metrics are absent, check that the registered `batch_key` exists and contains more than one unique value.

## Model-Level Metrics

Model-level metrics come from each model wrapper and are written to `metrics.json` before cross-model evaluation. They are useful for diagnostics but should not be treated as interchangeable biological benchmark scores.

| Model | Default model-level metrics |
|---|---|
| PCA | `total_variance` |
| MOFA | `total_variance` |
| MultiVI | `silhouette_score` when labels are available |
| TotalVI | `elbo_train`, `reconstruction_loss_train` |
| Mowgli | `ot_loss` |
| Cobolt | `loss` |

Losses and ELBO values are model-specific training diagnostics. They can help compare trials of the same model, especially during Optuna sweeps, but they should not be used alone to rank different model families.

## How to Cite Metric Results

When reporting metrics, cite multiverse and the underlying metric or model method where appropriate. Archive `run_manifest.yaml`, `metrics.json`, and provenance files so readers can connect each reported value to its run recipe.
