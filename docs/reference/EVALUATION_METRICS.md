# Evaluation Metrics

This reference explains the biological and technical metrics used to compare model outputs in v2.x. The platform evaluates successful model runs from their saved latent embeddings, then records metrics for comparison in the results tables and MLflow when tracking is enabled.

## Required Dataset Metadata

Evaluation depends on metadata registered with the dataset:

| Metadata key | Purpose |
|---|---|
| `batch_key` | Observation column identifying experimental batch, donor, technology, or another known nuisance grouping. Required for batch-correction metrics. |
| `cell_type_key` | Observation column identifying biological labels. Used by supervised bio-conservation metrics. |

The platform checks these fields before or during evaluation:

| Dataset condition | Metric behavior |
|---|---|
| `cell_type_key` is missing or absent from `.obs` | Supervised label metrics such as ARI, NMI, isolated labels, cLISI, and label silhouette are skipped. |
| `batch_key` is missing from `.obs` | Batch-correction metrics are skipped. |
| Only one batch is present | Batch-correction metrics are skipped because there is no batch structure to correct. |

## Bio-Conservation Metrics

Bio-conservation metrics ask whether the integrated latent space preserves meaningful biological structure. High performance means cells with the same biological identity remain close or recoverable after integration.

| Metric | What it measures |
|---|---|
| `silhouette_label` | Separation and compactness of biological labels in the latent space. Higher values indicate that cells of the same label are closer to one another relative to other labels. |
| `nmi_ari_cluster_labels_leiden` | Agreement between Leiden clusters computed in the latent space and known biological labels, reported through NMI and ARI. |
| `nmi_ari_cluster_labels_kmeans` | Agreement between k-means clusters and known biological labels, reported through NMI and ARI. |
| `isolated_labels` | Whether rare or isolated cell labels remain distinguishable after integration. |
| `clisi_knn` | Cell-type LISI, a neighborhood mixing score for biological labels. It is used to assess whether local neighborhoods preserve label structure. |

### ARI

Adjusted Rand Index compares cluster assignments with known labels while correcting for random agreement. In this platform it is used through the scIB clustering metrics, typically after Leiden or k-means clustering on each model's latent embedding.

ARI is most useful when the registered `cell_type_key` is reliable and the expected biology is cluster-like.

### NMI

Normalized Mutual Information compares how much information cluster assignments and known labels share. It is scale-normalized, making it easier to compare across runs with different numbers of clusters.

NMI complements ARI: ARI is stricter about pairwise assignment agreement, while NMI emphasizes shared label information.

### Silhouette Score

Silhouette-based metrics compare within-label compactness to nearest other-label separation. For a biological label metric, higher values indicate that cell types are more coherently separated in latent space.

Some individual model wrappers also report a model-level `silhouette_score` when labels are available. That score is distinct from the full scIB benchmark table but has the same basic interpretation.

## Batch-Correction Metrics

Batch-correction metrics ask whether unwanted technical structure has been reduced while preserving biological signal. They require at least two observed batches.

| Metric | What it measures |
|---|---|
| `ilisi_knn` | Integration LISI, a neighborhood mixing score for batch labels. Higher values indicate better local mixing across batches. |
| `kbet_per_label` | kBET rejection behavior computed within labels. It tests whether local batch composition is consistent with global expectations. |
| `graph_connectivity` | Whether cells with the same biological label remain connected in the latent-space neighbor graph after integration. |
| `pcr_comparison` | Principal component regression comparison for residual batch-associated variance. Lower residual technical association is better after scaling by the benchmarker. |
| `bras` | Batch removal assessment score from the scIB metrics suite. |

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

## Common Errors

| Symptom | Likely cause | What to do |
|---|---|---|
| ARI or NMI is missing | No valid `cell_type_key` was registered. | Add or correct the label column and re-register. |
| Batch-correction metrics are missing | No valid multi-value `batch_key` was registered. | Confirm batch metadata contains at least two groups. |
| Training loss improves but biology worsens | Model-specific loss is not a biological conservation score. | Compare loss with bio-conservation and batch-correction metrics. |

## How to Cite Metric Results

When reporting metrics, cite mvexp and the underlying metric or model method where appropriate. Archive `run_manifest.yaml`, `metrics.json`, and provenance files so readers can connect each reported value to its run recipe.

## Interpretation Notes

Use bio-conservation and batch-correction metrics together. A model can mix batches well by erasing meaningful cell-type structure, or preserve biology while leaving strong technical separation. The most defensible interpretation is therefore comparative: inspect both metric groups across the same dataset, metadata keys, and selected model set.

Metric availability is part of the result. If a supervised metric is absent, first check whether the dataset was registered with a valid `cell_type_key`. If batch-correction metrics are absent, check that the registered `batch_key` exists and contains more than one unique value.
