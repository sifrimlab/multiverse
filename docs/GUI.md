# Streamlit GUI

This tutorial/reference describes the mvexp graphical workflow. The GUI is the recommended entry point for researchers who work primarily in Jupyter and want reproducible benchmarks without hand-managing Docker or runner commands.

## GUI Map

[IMAGE: mvexp Streamlit Home]

| Tab | Purpose |
|---|---|
| **Registry** | Ingestion Wizard for datasets and model registration. |
| **Job Builder** | Compatibility matrix and dataset x model selection. |
| **Parameters** | Model hyperparameters and Optuna sweep controls. |
| **Execute** | Launch and monitor runs. |
| **Results** | Inspect metrics, logs, artifacts, and comparison reports. |
| **Experiment Analysis** | MLflow-backed experiment view. |
| **Sweep Tracker** | Optuna sweep visualization. |

## Tutorial: From Data to Comparison Report

1. Open the **Registry** tab.
2. Expand **Register New Dataset**.
3. Use the **Ingestion Wizard** to create or register `dataset.yaml`.
4. Click **Register Dataset**.
5. Click **Refresh Registry**.
6. Open **Job Builder**.
7. Select compatible dataset x model pairs.
8. Click **Generate Run Manifest**.
9. Open **Parameters**.
10. Adjust fixed parameters or enable sweeps.
11. Click **Generate Run Manifest (with params)**.
12. Open **Execute**.
13. Click **Launch Run**.
14. Open **Results**.
15. Review the comparison report and artifact paths.

## Registry: Ingestion Wizard

[IMAGE: The Ingestion Wizard]

The Ingestion Wizard is the visual-first data onboarding path. It collects dataset name, modalities, file paths, `batch_key`, and `cell_type_key`, then creates the dataset record used by all downstream planning.

## Job Builder: Compatibility Matrix

[IMAGE: The Job Builder Matrix]

The matrix prevents accidental scientific mismatches. For example, an RNA+ADT model should not be selected for a dataset without ADT. This moves validation earlier, before a long run is launched.

## Parameters: Schema-Driven Controls

[IMAGE: Model Parameter Controls]

Each model's JSON schema becomes GUI controls. Integers become numeric inputs, enumerations become dropdowns, and sweepable values can become Optuna ranges.

## Execute: Safe Parallel Runs

[IMAGE: Execute Tab Status Table]

The Execute tab launches the benchmark and shows status per job. mvexp runs compatible jobs safely and records a complete artifact directory for each successful result.

## Results: Comparison Reports

[IMAGE: Comparison Report Ranking Models]

The Results tab is where scientific interpretation begins. Use it to inspect model-level metrics, bio-conservation, batch-correction metrics, logs, and artifact paths. Ranking models by biological conservation is useful, but should be interpreted together with batch-correction behavior.

## Common Errors

| Symptom | Likely cause | What to do |
|---|---|---|
| Dataset missing from Job Builder | Registry cache has not refreshed. | Return to Registry and click **Refresh Registry**. |
| Model row is incompatible | Dataset lacks required omics. | Choose another model or register the correct modality. |
| Parameter controls are missing | Model schema was not registered. | Ask the model maintainer to check `hyperparameters_schema`. |
| Launch button fails quickly | Manifest points to stale datasets or models. | Regenerate the manifest from Job Builder. |
| Metrics panel is empty | No successful runs or tracking service unavailable. | Check Results first; tracking is helpful but not required for artifacts. |

## How to Cite GUI-Generated Results

The GUI writes the same reproducibility artifacts as headless execution. Cite mvexp by version or commit and include `run_manifest.yaml`, `job_spec.json`, and provenance files with Supplementary Material.
