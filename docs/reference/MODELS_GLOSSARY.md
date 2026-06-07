# Models Glossary

This reference summarizes the six built-in integration models available on the  platform. The Streamlit GUI reads each model's registered JSON schema and renders the appropriate parameter controls; users should set these values in the GUI rather than editing legacy configuration files.

All built-in models write a latent representation to `embeddings.h5`, model-specific metrics to `metrics.json`, and a UMAP plot when the requested color key is available in dataset metadata.

## PCA

Principal Component Analysis is the linear baseline model. It operates on concatenated AnnData features and projects cells into a low-dimensional principal-component space. If a `highly_variable` feature flag is present, PCA uses highly variable features; otherwise it uses the full feature matrix.

Supported omics: `any omics`

Default model metric: `total_variance`, the sum of the PCA variance ratios retained by the selected components.

Hyperparameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `n_components` | `50` | Number of principal components to compute. |
| `device` | `cpu` | Registered for schema consistency; the current PCA wrapper is CPU-oriented. |
| `umap_random_state` | `42` | UMAP seed. |
| `umap_color_type` | `cell_type` | Observation column for UMAP coloring. |

## MOFA

[MOFA](https://link.springer.com/article/10.15252/msb.20178124) uses the MOFA+ workflow through `muon` to learn latent factors across modalities. It is useful when the scientific question concerns shared and modality-specific axes of variation rather than only clustering performance.

Supported omics: `any omics`

Default model metric: `total_variance`, based on the explained variance across learned factors when available.

Hyperparameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `n_factors` | `20` | Number of latent factors to learn. |
| `n_iterations` | `5000` | Number of iterations for MOFA training. |
| `device` | `cpu` | CPU or CUDA target; non-CPU values enable MOFA GPU mode. |
| `umap_random_state` | `42` | UMAP seed. |
| `umap_color_type` | `cell_type` | Observation column for UMAP coloring. |

## MultiVI

[MultiVI](https://www.nature.com/articles/s41592-023-01909-9) is an `scvi-tools` variational model for joint single-cell RNA and ATAC analysis, with Protein Expression also included whenever available.

Supported omics: `rna`, `atac`, `adt` (optional)

Default model metric: `silhouette_score` when the requested label column is present. Training history from `scvi-tools` is also be written when available.

Hyperparameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `latent_dimensions` | `20` | Latent-space size for MultiVI configuration. |
| `max_epochs` | `400` | Number of training epochs. |
| `learning_rate` | `0.001` | training learning rate. |
| `device` | `cpu` | CPU or CUDA target. |
| `umap_random_state` | `42` | UMAP seed. |
| `umap_color_type` | `cell_type` | Observation column for UMAP coloring and silhouette labels. |

## TotalVI

[TotalVI](https://www.nature.com/articles/s41592-020-01050-x) is an `scvi-tools` variational model for CITE-seq style RNA and protein data.
Supported omics: `rna`, `adt`

Default model metrics: `elbo_train` and `reconstruction_loss_train` from the final available training history values.

Hyperparameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `latent_dimensions` | `20` | Latent-space size. |
| `max_epochs` | `400` | NUmber of training epochs. |
| `learning_rate` | `0.001` | Training learning rate. |
| `device` | `cpu` | CPU or CUDA target. |
| `umap_random_state` | `42` | UMAP seed. |
| `umap_color_type` | `cell_type` | Observation column for UMAP coloring. |

## Mowgli

[Mowgli](https://www.nature.com/articles/s41467-023-43019-2) integrates multimodal data with optimal transport and non-negative matrix factorization. The learned representation is stored from the model's optimal-transport latent matrix.

Supported omics: `any omics`

Default model metric: `ot_loss`, reported from the final optimal-transport loss.

Hyperparameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `latent_dimensions` | `20` | Size of the learned latent representation. |
| `optimizer` | `adam` | Optimizer for training. |
| `learning_rate` | `0.001` | Optimizer learning rate. |
| `tol_inner` | `1e-6` | Inner-loop convergence tolerance. |
| `max_iter_inner` | `500` | Number of training epochs. |
| `device` | `cpu` | CPU or CUDA target. |
| `umap_random_state` | `42` | UMAP seed. |
| `umap_color_type` | `cell_type` | Observation column for UMAP coloring. |

## Cobolt

[Cobolt](https://link.springer.com/article/10.1186/s13059-021-02556-z) integrates multimodal data with a Bayesian hierarchical model. 

Supported omics: `any omics`

Default model metric: `loss`, the final training loss. The wrapper stores the loss history when available.

Hyperparameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `latent_dimensions` | `20` | Size of the learned latent representation. |
| `num_epochs` | `200` | Number of training epochs. |
| `learning_rate` | `0.001` | Training learning rate. |
| `random_state` | `42` | Registered random-state parameter for the Cobolt model |
| `device` | `cpu` | CPU or CUDA target. |
| `umap_random_state` | `42` | UMAP seed. |
| `umap_color_type` | `cell_type` | Observation column for UMAP coloring. |
