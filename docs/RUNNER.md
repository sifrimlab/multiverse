# Orchestrator Runner

This document explains how to use the central orchestrator to run the Multi-verse pipeline.

## Usage

The main entrypoint for the orchestrator is the `multiverse.runner.cli` module. You can run it from the root of the repository.

### Command

```bash
python -m multiverse.runner.cli --models <model1> <model2> ... --input /path/to/input_dir --output /path/to/output_dir
```

### Arguments

- `--models`: A space-separated list of the models you want to run (e.g., `pca multivi mowgli`). This is a required argument.
- `--input`: The path to the directory containing the input data. This directory should contain the `data.h5ad` (for PCA, MultiVI) or `data.h5mu` (for Mowgli, MOFA) file. This is a required argument.
- `--output`: The path to the directory where the results will be saved. A subdirectory will be created for each model. This is a required argument.

### Example

To run the `pca` and `multivi` models on data located in `./sample_data/input` and save the results to `./results`, you would use the following command:

```bash
python -m multiverse.runner.cli --models pca multivi --input ./sample_data/input --output ./results
```

This will create the following directory structure:
```
./results/
├── pca/
│   ├── embeddings.h5ad
│   ├── metrics.json
│   └── log.txt
└── multivi/
    ├── embeddings.h5ad
    ├── metrics.json
    └── log.txt
```
