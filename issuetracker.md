# Issue Tracker — mvr Repository

> Generated: 2026-05-26 | Source: https://github.com/saptarshichakrabarti/mvr/issues

---

## Summary Table

| # | Title | Severity | Status | Linked Issues |
|---|-------|----------|--------|---------------|
| [#13](#issue-13--docker-data-path-default-is-user-specific) | change docker data path default | Medium (UX) | Open | — |
| [#14](#issue-14--scib-benchmark-crashes-due-to-wrong-object-type) | issue with scib benchmark | High | Open | #16 |
| [#15](#issue-15--deprecated-anndata-io-import-paths) | anndata deprecation warnings | Low (time-sensitive) | Open | — |
| [#16](#issue-16--pbmc10k-cell_type-key-not-found) | Pbmc10k dataset has cell_type key but not found somehow | Medium | Open | #14 |
| [#17](#issue-17--dead-evaluation-code--duplicate-methods) | duplicated evaluation behavior + legacy code | Low | Open | — |

---

## Root Cause Review — 2026-05-26

This review checked the tracker against the current repository state and the local PBMC10K files.

### Overall findings

- **#13 is diagnosed correctly.** The checked-in `multiverse.config.yaml` contains `docker_data_root: /mnt/data/saptarshi/docker-data`, although `multiverse/multiverse_config.py` already defines a portable fallback of `<repo>/.docker-data`.
- **#14 is only partially diagnosed.** The legacy config-driven evaluator path does convert loaded modalities to concatenated `AnnData`, but the current per-job Docker path calls `evaluate_single_run(...)` from `multiverse.evaluate`, and that function is missing from the module. That missing evaluation entry point is a more concrete root cause for per-run scIB evaluation failures than a generic `MuData -> AnnData` handoff bug.
- **#15 is not confirmed in this checkout.** A repository search found no active `read_umi_tools`, `from anndata import read_...`, or `anndata.read_...` usage outside this tracker document. The reported warning may come from a dependency, an older branch, or user-side code rather than the current repository.
- **#16 is not primarily caused by #14.** The local `store/datasets/pbmc10k/data/processed.h5mu` has top-level `/obs` keys `['cells']`, while `cell_type` exists under `/mod/rna/obs` and `/mod/atac/obs`. The preflight validator reads only top-level `/obs`, so it warns that `cell_type` is missing even though it exists in modality-level observations. The dataset manifest also has no `batch` key, only `cell_type`.
- **#17 is diagnosed incorrectly.** `run_evaluation_container` is not dead: it is imported and called by `multiverse/runner/cli.py`. The actual stale/broken evaluation surface is different: `multiverse/runner/docker_runner.py` imports `evaluate_single_run`, and tests import it too, but `multiverse/evaluate.py` does not define it.

### Highest-confidence root causes

1. PBMC10K `cell_type` warning: `multiverse/runner/cli.py::_peek_obs_columns()` only inspects top-level `/obs`, but generated `.h5mu` files may keep `cell_type` in modality-level `.obs`.
2. Per-run evaluation failure: `multiverse/runner/docker_runner.py` calls a missing `multiverse.evaluate.evaluate_single_run`.
3. Metadata loss during preprocessing: `multiverse/ingestion.py::preprocess_dataset()` builds `md.MuData(modalities)` and writes it without promoting declared metadata keys from a modality `.obs` to `mdata.obs`.

---

## Issue #13 — Docker Data Path Default is User-Specific

**Reported by:** anisdismail | **Date:** 2026-05-21

### What it is

The Docker data path defaults to a hardcoded absolute path tied to a specific developer's local folder (`saptarshi`). New users are expected to find and change this in the settings panel, which is not intuitive and leads to misconfiguration.

### Root Cause

During development a local absolute path was hardcoded as the default value for the Docker data directory. It was never replaced with a generic, user-relative, or environment-derived default, and no setup wizard or first-run prompt surfaces this setting to new users.

### Suggested Fix

- Replace the hardcoded default with a relative or environment-variable-derived path (e.g. `$HOME/mvr-data` or `./data`).
- Surface this setting prominently during initial setup or as a required field, rather than burying it in a settings panel.

### Review and Recommendations

The diagnosis is sound, but the fix should avoid creating a second source of truth. `multiverse/multiverse_config.py` already defines `DEFAULT_DOCKER_DATA_ROOT = <repo>/.docker-data`; the immediate fix should be to remove or regenerate the checked-in `multiverse.config.yaml` value so a user-specific path is not committed.

Recommended implementation:

1. Treat `multiverse.config.yaml` as local machine state. Either remove it from version control and provide `multiverse.config.example.yaml`, or keep only portable defaults in the checked-in file.
2. Keep the GUI setting, but show whether the value is inherited from the default or explicitly overridden.
3. Add a small config test that loads defaults with no config file present and asserts the path does not contain a developer username or absolute lab path.

---

## Issue #14 — scib Benchmark Crashes Due to Wrong Object Type

**Reported by:** anisdismail | **Date:** 2026-05-21

### What it is

The scib benchmarking pipeline crashes at runtime because it receives a `MuData` object (`processed.h5mu`) instead of an `AnnData` object. scib requires:
1. An `AnnData` object (not `MuData`)
2. Raw / original counts (not normalized data)

The `concatenate` method that was supposed to handle the conversion is not performing the `MuData → AnnData` cast, causing a hard crash.

### Root Cause

This is a type-mismatch bug at the interface between the preprocessing pipeline (which produces multimodal `MuData` in `.h5mu` format) and the scib benchmark runner (which expects single-modal `AnnData` in `.h5ad` format). The conversion step is missing or broken — either the wrong file/object is passed downstream, or `concatenate` does not actually perform the cast.

A secondary problem is that normalized data is being passed when raw counts are required.

### Suggested Fix

- In `concatenate` (or the handoff function feeding scib), add an explicit `MuData → AnnData` conversion before calling scib.
- Ensure raw counts are restored from `adata.raw` or the appropriate layer before passing the object to scib.
- Add a type assertion at the scib entry point to fail early with a descriptive error if the input is not `AnnData`.

> **Note:** This issue is likely the upstream cause of Issue #16.

### Review and Recommendations

The diagnosis is plausible for the old config-driven evaluator path, but it is incomplete for the current Docker runner path. `multiverse/evaluate.py::main()` uses `dataset_select(..., data_type="concatenate")`, which does return an `AnnData`. However, the active per-job path in `multiverse/runner/docker_runner.py` imports and calls `evaluate_single_run(...)`, and `multiverse/evaluate.py` does not define that function.

The suggested fix should be narrowed and made testable:

1. Implement `evaluate_single_run(output_dir, dataset_path, batch_key, label_key)` in `multiverse/evaluate.py`, or change the runner to call an existing supported evaluator. This is currently the most concrete blocker.
2. In `evaluate_single_run`, load `.h5ad` directly and convert `.h5mu` deterministically to an `AnnData` view used for scIB. Prefer a declared primary modality such as `rna` when evaluating embeddings generated from one modality, or concatenate modalities only when the embedding was trained over concatenated features.
3. Assert both object type and row alignment before benchmarking: `isinstance(adata, ad.AnnData)`, `adata.n_obs == embedding.shape[0]`, and required `.obs` keys exist when their metrics are requested.
4. Do not blindly replace `X` with raw counts for every metric. scIB metrics evaluate embeddings in `.obsm`; raw counts are relevant for specific preprocessing-sensitive comparisons. Preserve raw data in `.layers["counts"]` or `.raw`, and pass the representation expected by each metric.
5. Add tests for `.h5ad`, `.h5mu` with top-level metadata, `.h5mu` with modality-level metadata only, and embedding/cell-count mismatch.

---

## Issue #15 — Deprecated anndata IO Import Paths

**Reported by:** anisdismail | **Date:** 2026-05-21

### What it is

The codebase imports IO functions (e.g. `read_umi_tools`) from the top-level `anndata` namespace. anndata has deprecated this pattern and now requires importing from `anndata.io`:

```
FutureWarning: Importing read_umi_tools from `anndata` is deprecated.
Import anndata.io.read_umi_tools instead.
```

### Root Cause

The code was written against an older version of anndata where IO utilities were available at the top level. anndata restructured its public API but the imports in this codebase were never updated.

### Suggested Fix

Search for all affected import sites:

```bash
grep -rn "from anndata import read_\|anndata\.read_" .
```

Replace each occurrence with the `anndata.io` equivalent, e.g.:

```python
# Before
from anndata import read_umi_tools

# After
from anndata.io import read_umi_tools
```

This is low severity today but will become a hard error in a future anndata release.

### Review and Recommendations

This root cause is not confirmed in the current checkout. A repository search found no active matches for the listed deprecated import patterns outside this tracker document.

Recommended next step:

1. Reproduce the warning with the exact command that emitted it, then capture the warning traceback or run with `PYTHONWARNINGS=default` to identify the importing module.
2. If the warning originates in this repository, update that import directly and add a focused regression search/test.
3. If it originates in a dependency, track the dependency version instead of changing repository code.

---

## Issue #16 — PBMC10K `cell_type` Key Not Found

**Reported by:** anisdismail | **Date:** 2026-05-21

### What it is

When running supervised benchmark metrics on the PBMC10K dataset, the system reports:

```
dataset 'PBMC10K': cell_type_key 'cell_type' not found — supervised metrics will be skipped
```

The key does exist in the dataset; supervised metrics are silently skipped as a result.

### Root Cause

This is most likely a downstream symptom of Issue #14. When `MuData` is incorrectly passed instead of `AnnData`, `.obs` metadata — including `cell_type` — may not be propagated correctly. Other possible causes:

1. **Object level mismatch:** `cell_type` exists in a modality-specific `.obs` (e.g. `mdata['rna'].obs`) but the lookup targets the top-level `.obs`.
2. **Case sensitivity:** A mismatch between `cell_type` and the actual stored key name.
3. **Lost metadata during conversion:** The MuData → AnnData conversion (if it occurs at all) drops `.obs` columns.

### Suggested Fix

- Fix Issue #14 first; this issue may resolve automatically once the correct `AnnData` object (with `.obs` intact) is passed to the evaluation step.
- If the issue persists, add explicit debug logging to print available `.obs` columns at the point where `cell_type_key` is resolved.
- Ensure `.obs` columns are explicitly transferred during any `MuData → AnnData` conversion.

### Review and Recommendations

The reported symptom is real, but the root cause should be revised. For the local PBMC10K files:

- `store/datasets/pbmc10k/data/processed.h5mu` top-level `/obs` contains only `cells`.
- `cell_type` exists in `/mod/rna/obs` and `/mod/atac/obs`.
- `store/datasets/pbmc10k/dataset.yaml` declares `metadata_keys.cell_type: cell_type` but does not declare a batch key.
- `multiverse/runner/cli.py::_peek_obs_columns()` checks only top-level `/obs`, so preflight cannot see modality-level `cell_type`.

Recommended implementation:

1. Fix `multiverse/ingestion.py::preprocess_dataset()` to promote declared metadata keys from a chosen modality `.obs` to `mdata.obs` before writing `processed.h5mu`. This makes the runtime object and the preflight HDF5 check agree.
2. Teach `_peek_obs_columns()` to handle `.h5mu` files by checking both top-level `/obs` and `/mod/*/obs`. If a key is found only in modalities, warn that preprocessing should promote it, but do not falsely report it as absent.
3. Add a PBMC-style test fixture where `cell_type` exists only under `/mod/rna/obs` and verify `validate_pending_jobs()` does not warn incorrectly once promotion or nested lookup is implemented.
4. Consider manifest validation: if `metadata_keys.batch` is optional, metrics that require batches should be explicitly reported as unavailable because no batch key was registered, not because the file is malformed.

---

## Issue #17 — Dead Evaluation Code & Duplicate Methods

**Reported by:** anisdismail | **Date:** 2026-05-21

### What it is

Two methods — `run_evaluation_container` and `run_single_evaluation` — exist in the evaluation module but are never called anywhere in the codebase. They represent legacy/duplicate code left over from a prior architecture.

### Root Cause

During a refactor of the evaluation pipeline these methods were superseded by a new implementation but were never deleted. The current evaluation path bypasses them entirely, leaving dead code that:
- Creates confusion about which evaluation path is authoritative.
- Adds maintenance burden (any interface changes must be tracked against unused methods).

### Suggested Fix

1. Confirm neither method is called anywhere:
   ```bash
   grep -rn "run_evaluation_container\|run_single_evaluation" .
   ```
2. Verify any useful logic inside them is already covered by the current evaluation path.
3. Delete both methods and any supporting code that exists solely for their use.

### Review and Recommendations

This diagnosis is incorrect for the current checkout. `run_evaluation_container` is defined in `multiverse/runner/docker_runner.py`, imported by `multiverse/runner/cli.py`, and called from the CLI. It should not be deleted without first removing or replacing those CLI paths.

The actual issue is evaluation API drift:

1. `multiverse/runner/docker_runner.py` imports `evaluate_single_run` from `multiverse.evaluate`.
2. `tests/unit/test_evaluate.py` imports and tests `evaluate_single_run`.
3. `multiverse/evaluate.py` does not define `evaluate_single_run`.
4. `multiverse/evaluate.py` also contains unreachable code after `aggregate_results()` returns, suggesting a partial or botched refactor.

Recommended implementation:

1. Replace this issue with: "evaluation API drift: missing `evaluate_single_run` and unreachable evaluator code."
2. Implement or restore `evaluate_single_run`, then make the Docker runner and tests use that single API.
3. Only after tests pass, remove truly unreachable code in `evaluate.py` and decide whether `run_evaluation_container` is still needed as a CLI feature.

---

## Cross-Issue Relationships

```
#14 (per-run evaluation API/type handling)
  └── related to #16 only at the evaluation boundary

#13  — independent UX bug (Docker path default)
#15  — unconfirmed in current checkout; needs reproduction
#16  — H5MU metadata promotion / preflight lookup bug
#17  — evaluation API drift, not dead `run_evaluation_container`
```

Fixing #16 should start at ingestion/preflight metadata handling, not wait for #14. Fixing #14 should start by restoring the missing per-run evaluation entry point and adding explicit `.h5mu -> AnnData` behavior there.

---

## Recommended Fix Priority

| Priority | Issue | Reason |
|----------|-------|--------|
| 1 | #17/#14 | Restore the missing `evaluate_single_run` API, then make scIB input conversion explicit |
| 2 | #16 | Prevent false `cell_type` warnings by promoting/reading modality-level H5MU metadata |
| 3 | #13 | Affects every new Docker user; easy to fix |
| 4 | #15 | Reproduce before changing code; no matching import exists in the current checkout |
| 5 | Cleanup | Remove unreachable evaluation code after the working evaluation path is covered by tests |

---

## Verification Update — 2026-05-26

This pass reviewed the current working tree against the recommendations above. Several items are addressed, but the evaluation and local-runner changes still need follow-up before they should be considered complete.

### Addressed Properly

- **#13 Docker config path:** `multiverse.config.yaml` has been removed from version control, added to `.gitignore`, and replaced with `multiverse.config.example.yaml`. This resolves the user-specific checked-in path problem.
- **UMAP `.tmp` save failure:** `sdk/mvr-worker/mvr_worker/io.py::save_umap()` now writes through a temporary file with a `.png` suffix, passes `format="png"` explicitly, uses `os.replace()`, and cleans up the temp file. The focused worker regression test passes.
- **Zero-risk orphan cleanup:** the unreachable block in `multiverse/evaluate.py` was removed, several unused imports/locals were cleaned up, stale single-function orphans were removed, and the legacy JSON registry/config runner cluster was deleted or replaced.
- **Maintenance metrics rebuild:** `multiverse.tools.rebuild_run_metrics` is now registered as a console script (`multiverse-rebuild-metrics`).
- **#15 anndata warning:** no deprecated `anndata` IO import was reintroduced; still no repository-side fix is needed without a reproducible warning source.

### Partially Addressed / Still Problematic

- **#14 / #17 per-run evaluation:** `evaluate_single_run()` now exists, which closes the missing-symbol bug. However, it currently writes `evaluation_metrics.json`, while the tracking pipeline (`load_run_metrics`, `run_metrics` persistence, MLflow, and GUI metrics discovery) reads `metrics.json`. As written, scIB results can be generated but not persisted through the normal metrics path.
- **Evaluation test mismatch:** `tests/unit/test_evaluate.py` still expects `metrics.json`, while `evaluate_single_run()` writes `evaluation_metrics.json`. The test and implementation disagree.
- **Fake batch labels:** `evaluate_single_run()` assigns random `_batch` labels when no batch key is available. This can create meaningless batch-correction metrics. Missing batch metadata should disable batch-correction metrics instead of fabricating batches.
- **MuData evaluation object:** `_mudata_to_evaluation_anndata()` returns an `AnnData` with `.obs` only and no expression matrix. That is acceptable for some embedding-only metrics, but `pcr_comparison=True` may require valid matrix data. Either construct `X` from a selected modality/count layer or disable matrix-dependent metrics for embedding-only evaluation.
- **#16 metadata handling:** `preprocess_dataset()` now promotes declared metadata keys to `mdata.obs`, and `_peek_obs_columns()` checks modality-level `/mod/*/obs`. This is the right direction. Still missing: a regression test with a true `.h5mu` where `cell_type` exists only under `/mod/rna/obs`, and `_peek_batch_count()` still inspects only top-level `/obs`.
- **Local runner replacement:** `multiverse-cli run --local` and `multiverse/runner/local_runner.py` exist, replacing the deleted legacy runner. The local path runs model scripts and copies artifacts, but it does not yet insert/update `runs`, persist `run_metrics`, run per-job evaluation, or preserve stdout/stderr logs on successful runs. As a result, local runs may not appear consistently in DB/GUI-backed results.
- **Cleanup leftovers:** `pyflakes` still reports minor cleanup items, including an unused local `numpy as np` import in `multiverse/runner/cli.py::_peek_batch_count`, unused imports in `tests/unit/test_local_runner.py`, unused locals in some model `run.py` scripts, and a pointless f-string in `multiverse/gui.py`.

### Recommended Next Fixes

1. Make `evaluate_single_run()` write metrics to the artifact consumed by the platform, or update `load_run_metrics()`/GUI/MLflow persistence to intentionally include `evaluation_metrics.json`. Prefer a clear merged `metrics.json` contract if scIB scores should be first-class run metrics.
2. Stop generating random batch labels in evaluation. If no valid `batch_key` exists, disable batch-correction metrics and log a warning.
3. Decide whether per-run evaluation should build an expression-backed `AnnData` for matrix-dependent scIB metrics or disable those metrics when only embeddings are available.
4. Add tests for `.h5mu` evaluation and preflight metadata where `cell_type` exists only at modality level.
5. Extend the local runner to record DB run rows, persist metrics, preserve logs, and optionally invoke the same per-run evaluation path as Docker runs.
6. Finish the remaining mechanical cleanup reported by `pyflakes`.

### Verification Commands Run

- `python3 -m compileall multiverse sdk/mvr-worker/mvr_worker store/models tests/unit/test_local_runner.py tests/unit/test_worker_io.py` — passed.
- `pytest -q --confcutdir=tests/unit tests/unit/test_worker_io.py` — passed.
- `pytest -q --confcutdir=tests/unit tests/unit/test_evaluate.py` — could not collect in this environment because `anndata` is not installed.
- `pytest -q --confcutdir=tests/unit tests/unit/test_local_runner.py` — failed in isolated mode because package path and root fixtures are unavailable; this does not prove runtime failure, but the test file needs better isolation if it is intended to run standalone.

