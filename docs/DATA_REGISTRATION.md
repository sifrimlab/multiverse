# Data Registration Guide

This guide is for rookie bioinformaticians onboarding new datasets into Multiverse.

## What You Are Creating

Each dataset is a self-contained package under:

`store/datasets/<slug>/`

Required structure:

```text
store/datasets/
└── <slug>/
    ├── dataset.yaml
    └── data/
        ├── <raw files...>
        └── processed.h5mu   # created/managed by processing pipeline
```

## Step 1: Create the Dataset Folder

1. Choose a filesystem-safe slug, for example `pbmc10k`.
2. Create:
   - `store/datasets/pbmc10k/data/`
3. Place your raw source files inside `data/`.

## Step 2: Write `dataset.yaml`

Place `dataset.yaml` next to `data/`.

Example:

```yaml
name: "PBMC 10k"
omics: ["rna", "atac"]
raw_files:
  rna: "data/rna.h5ad"
  atac: "data/atac.h5ad"
metadata_keys:
  batch: "donor_id"
  cell_type: "cell_type"
```

### Schema Notes

- `name` (string): human-readable dataset name.
- `omics` (list): available modalities (`rna`, `atac`, `adt`, ...).
- `raw_files` (mapping): modality to relative path inside dataset folder.
- `metadata_keys.batch` (string): observation key used for batch-aware metrics.
- `metadata_keys.cell_type` (string, optional): observation key used for label-aware metrics.

## Step 3: Migrate Legacy Data (Optional)

If your data are still in a legacy/unstructured folder, use:

```bash
python -m multiverse.migrate_data --source /path/to/legacy --dest ./store/datasets --dry-run
python -m multiverse.migrate_data --source /path/to/legacy --dest ./store/datasets
```

The migration utility discovers compatible files, proposes metadata mappings, and materializes dataset packages.

## Step 4: Register the Dataset

Use Make:

```bash
make register slug=pbmc10k
```

Or register by explicit manifest path:

```bash
make register manifest=store/datasets/pbmc10k/dataset.yaml
```

Registration updates the SQLite registry and makes the dataset selectable in benchmark planning.

## Checklist Before Benchmarking

- Folder exists under `store/datasets/<slug>/`.
- `dataset.yaml` is valid and references existing files.
- Dataset is registered (`make register ...`) and appears as `READY` in registry-driven workflows.

