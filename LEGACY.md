# Legacy Local Runner — Removal Complete

The legacy JSON-config local runner cluster has been fully removed. This document
is kept as a record of what was deleted and why.

## What was removed

| File / symbol | Why removed |
|---|---|
| `runner.py` | Deprecated CLI entry point; replaced by `multiverse-cli run --local` |
| `multiverse/main.py` | Compatibility re-export shim; no active importers |
| `multiverse/config_schema.py` | Pydantic schema for old JSON config; only used by legacy runner |
| `multiverse/compat/` (entire package) | Canonical home of the legacy cluster during deprecation period |
| `tools/legacy_local_runner.py` | Legacy host-side runner; replaced by `multiverse-cli run --local` |
| `multiverse/registry.py::load_registry()` | JSON model registry loader; only used by legacy runner |
| `multiverse/registry.py::get_eligible_models()` | JSON-registry omics filter; only used by legacy runner |
| `multiverse/registry.py::ModelEntry`, `ModelRegistry` | Pydantic models for JSON registry; removed with above |
| `multiverse-cli run-legacy` subcommand | Deprecated CLI entry; replaced by `multiverse-cli run --local` |
| `tests/unit/test_registry.py` | Tests for removed JSON registry helpers |
| `tests/unit/test_config_schema.py` | Tests for removed config schema |
| `tests/integration/test_pipeline_e2e.py` | Integration test for removed legacy runner |

## What replaced them

| Old | New |
|---|---|
| `runner.py <config.json>` | `multiverse-cli run --manifest run_manifest.yaml --local` |
| `multiverse-cli run-legacy --config <file>` | `multiverse-cli run --manifest run_manifest.yaml --local` |
| JSON config schema | YAML run manifest (`run_manifest.yaml`) |
| `load_registry()` / `get_eligible_models()` | SQLite-backed `multiverse/registry_db.py` + model manifests |

## What was kept

`multiverse/registry.py::generate_compatibility_matrix()` — still used by the GUI
to render the dataset × model compatibility table.

## How the local runner works now

```bash
# Register your dataset and model first (one-time)
multiverse-cli register-dataset --slug my-dataset
multiverse-cli register-model --slug pca

# Run locally without Docker (model Python deps must be installed)
multiverse-cli run --manifest run_manifest.yaml --local --output ./results
```

The `--local` flag invokes `multiverse/runner/local_runner.py`, which:
1. Reads the YAML run manifest and queries the SQLite registry (same as Docker path).
2. Creates a workspace under `store/workspaces/run_<uuid>/`.
3. Symlinks the dataset to `input/data.h5mu` and writes `job_spec.json`.
4. Runs `store/models/<slug>/container/run.py` as a subprocess with
   `MVR_INPUT_DATA_PATH`, `MVR_OUTPUT_DIR`, and `MVR_JOB_SPEC_PATH` env vars.
5. Copies artefacts to the declared output path.

Model Python dependencies must be installed on the host. The `mvr-worker` SDK
must be installed: `pip install -e sdk/mvr-worker`.
