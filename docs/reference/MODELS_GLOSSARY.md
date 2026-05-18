# Models Glossary

This reference summarizes the six built-in integration models available in the v2.x platform. The Streamlit GUI reads each model's registered JSON schema and renders the appropriate parameter controls; users should set these values in the GUI rather than editing legacy configuration files.

All built-in models write a latent representation to `embeddings.h5`, model-specific metrics to `metrics.json`, and a UMAP plot when the requested color key is available in dataset metadata.

## Shared Parameters

| Parameter | Meaning |
|---|---|
| `device` | Runtime target for implementations that can use CPU or CUDA. Registered choices are `cpu`, `cuda`, and `cuda:0` where supported. |
| `umap_random_state` | Seed used for UMAP neighbor graph and layout generation. This is separate from the run-level seed. |
| `umap_color_type` | Observation column used to color the UMAP plot, commonly a cell type column. If the column is absent, the plot is generated without that color annotation. |
| `latent_dimensions` | Size of the learned latent space for neural or matrix-factorization models. |
| `learning_rate` | Optimizer step size for trainable models. |

## PCA

Principal Component Analysis is the linear baseline model. It operates on concatenated AnnData features and projects cells into a low-dimensional principal-component space. If a `highly_variable` feature flag is present, PCA uses highly variable features; otherwise it uses the full feature matrix.

Supported omics: `rna`

Default model metric: `total_variance`, the sum of the PCA variance ratios retained by the selected components.

Hyperparameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `n_components` | `50` | Number of principal components to compute. |
| `device` | `cpu` | Registered for schema consistency; the current PCA wrapper is CPU-oriented. |
| `umap_random_state` | `42` | UMAP seed. |
| `umap_color_type` | `cell_type` | Observation column for UMAP coloring. |

## MOFA

MOFA uses the MOFA+ workflow through `muon` to learn latent factors across modalities. It is useful when the scientific question concerns shared and modality-specific axes of variation rather than only clustering performance.

Supported omics: `rna`, `atac`

Default model metric: `total_variance`, based on the explained variance across learned factors when available, with a fallback calculation from factor variance.

Hyperparameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `n_factors` | `20` | Number of latent factors to learn. |
| `n_iterations` | `5000` | Registered iteration budget for MOFA configuration. |
| `device` | `cpu` | CPU or CUDA target; non-CPU values enable MOFA GPU mode. |
| `umap_random_state` | `42` | UMAP seed. |
| `umap_color_type` | `cell_type` | Observation column for UMAP coloring. |

## MultiVI

MultiVI is an `scvi-tools` variational model for joint single-cell RNA and ATAC analysis. The input must contain feature annotations that distinguish gene expression features from genomic regions, because the model setup uses those feature types to configure the RNA and ATAC components.

Supported omics: `rna`, `atac`

Default model metric: `silhouette_score` when the requested label/color column is present. Training history from `scvi-tools` may also be written when available.

Hyperparameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `latent_dimensions` | `20` | Registered latent-space size for MultiVI configuration. |
| `max_epochs` | `400` | Registered training epoch budget. |
| `learning_rate` | `0.001` | Registered training learning rate. |
| `device` | `cpu` | CPU or CUDA target. |
| `umap_random_state` | `42` | UMAP seed. |
| `umap_color_type` | `cell_type` | Observation column for UMAP coloring and silhouette labels. |

## TotalVI

TotalVI is an `scvi-tools` variational model for CITE-seq style RNA plus protein data. The wrapper expects protein expression to be available in the AnnData object under `protein_expression` and uses the dataset batch annotation during setup.

Supported omics: `rna`, `adt`

Default model metrics: `elbo_train` and `reconstruction_loss_train` from the final available training history values.

Hyperparameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `latent_dimensions` | `20` | Registered latent-space size. |
| `max_epochs` | `400` | Registered training epoch budget. |
| `learning_rate` | `0.001` | Registered training learning rate. |
| `device` | `cpu` | CPU or CUDA target. |
| `umap_random_state` | `42` | UMAP seed. |
| `umap_color_type` | `cell_type` | Observation column for UMAP coloring. |

## Mowgli

Mowgli integrates multimodal data with optimal transport and non-negative matrix factorization. The learned representation is stored from the model's optimal-transport latent matrix.

Supported omics: `rna`, `atac`

Default model metric: `ot_loss`, reported from the final optimal-transport loss. The wrapper stores the loss history when available.

Hyperparameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `latent_dimensions` | `20` | Size of the learned latent representation. |
| `optimizer` | `adam` | Optimizer; registered choices are `adam`, `sgd`, and `adamw`. |
| `learning_rate` | `0.001` | Optimizer learning rate. |
| `tol_inner` | `1e-6` | Inner-loop convergence tolerance. |
| `max_iter_inner` | `500` | Maximum inner-loop iterations. |
| `device` | `cpu` | CPU or CUDA target. |
| `umap_random_state` | `42` | UMAP seed. |
| `umap_color_type` | `cell_type` | Observation column for UMAP coloring. |

## Cobolt

Cobolt integrates multimodal data with a Bayesian hierarchical model. The wrapper constructs a multiomic dataset from modality-specific AnnData objects and saves latent embeddings for cells present across all modalities.

Supported omics: `rna`, `atac`

Default model metric: `loss`, the final training loss. The wrapper stores the loss history when available.

Hyperparameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `latent_dimensions` | `20` | Size of the learned latent representation. |
| `num_epochs` | `200` | Number of training epochs. |
| `learning_rate` | `0.001` | Training learning rate. |
| `random_state` | `42` | Registered random-state parameter for the Cobolt schema; the run-level seed is also applied to Python, NumPy, and Torch. |
| `device` | `cpu` | CPU or CUDA target. |
| `umap_random_state` | `42` | UMAP seed. |
| `umap_color_type` | `cell_type` | Observation column for UMAP coloring. |

## Common Errors

| Symptom | Likely cause | What to do |
|---|---|---|
| A model is not selectable for a dataset | Required omics are not present in the registered dataset. | Check `supported_omics` and the dataset's `omics` list. |
| Parameter controls are missing | The model's hyperparameter schema is missing or invalid. | Ask the maintainer to check `hyperparameters_schema`. |
| Model-specific loss is hard to compare | Loss scales differ across model families. | Use loss for within-model diagnostics and comparison reports for cross-model ranking. |
| UMAP is uncolored | `umap_color_type` is absent from `.obs`. | Use a valid metadata column such as `cell_type`. |

## How to Cite Models

When publishing mvexp results, cite both mvexp and the original papers for the selected integration models. Archive the `model.yaml`, hyperparameter schema, `run_manifest.yaml`, and run provenance so reviewers can identify the exact model configuration used.
