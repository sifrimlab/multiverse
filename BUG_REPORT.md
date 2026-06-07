# Bug Report

**Date:** 2026-06-07
**Branch:** `arch_revamp`
**Affected runs:** 3 containers (2× PCA, 1× Cobolt), all failed
**Manifest:** `run_manifest.yaml` (experiment `benchmark_run1234`, seed 44)

---

## Overview

Three separate bugs are documented here. Bug 1 and Bug 2 were diagnosed from the
Configure tab and manifest structure. Bug 3 (container crashes) is a direct
runtime consequence of Bug 2, plus an independent preprocessing failure.

## Verification findings

I checked the report against the current branch on 2026-06-07.

**Verdict:** Bug 1, Bug 2, and Bug 3a are correctly diagnosed. Bug 3b identifies
the right failing subsystem (pre-model preprocessing/HVG), but the explanation
and proposed fix need correction: the observed `pd.cut` / `mean_bin` stack is
the Scanpy dispersion path used by `flavor="seurat"` / `"cell_ranger"`, not the
`seurat_v3` loess path. The overflow is therefore more consistent with
too-large values being treated as log-scale data and then passed through
Scanpy's `expm1` step, likely when `log_normalization=True` or when already
large/logged data reaches the `seurat` flavor.

**Strategy corrections:**

- Fix 1 is directionally correct, but the widget state key in the proposed
  helper must be `f"{job_key}::fixed::{param_name}"`, matching
  `_render_fixed_widget()`. The originally suggested `f"{job_key}::{param_name}"`
  key would not prefill the actual Streamlit input.
- Fix 2 is correct at the manifest shape level, but it must also keep GUI
  sweep specs compatible with `multiverse/runner/tuner.py`. The current tuner
  accepts `type: int`, `type: categorical`, and `type: loguniform`; it does not
  accept the GUI's `{"type": "float", "log": ...}` format unless the tuner is
  extended or the GUI translates floats into tuner-supported distribution names.
- Fix 3b should not clip values blindly. The safer short-term fix is to detect
  non-finite or implausibly large values before HVG, select the flavor based on
  actual data state as well as requested preprocessing, and fail early with a
  clear dataset/preprocessing error when the input contract is violated.

---

## Bug 1 — `random_seed` is not propagated into model hyperparameters

### Summary

The **Random Seed** field in the Configure tab writes its value to
`globals.random_seed` in the manifest. It does **not** pre-fill any model-level
seed hyperparameters (e.g. `random_state`, `umap_random_state`) in the
hyperparameter form. Those fields are independent widgets and retain whatever
default or previously-typed value they have.

### Key distinction: `random_seed` vs `random_state`

| Field | Scope | Where it lives | What it controls |
|---|---|---|---|
| `random_seed` | Platform / runner | `globals.random_seed` in manifest; resolved by `resolve_effective_seed()` | Execution-level seed passed to containers as `job_spec["seed"]`; used for UMAP in `ModelFactory` via `umap_random_state` |
| `random_state` | Per-model hyperparameter | `model_params` → `hyperparameters.<model>` in `job_spec` | Model's internal RNG (e.g. scikit-learn, Cobolt); only present if declared in the model's `hyperparameters.json` schema and filled by the user |

### Evidence

From `job_spec.json` for workspace `2ef3b4b7` (Cobolt, pbmc10k):

```json
"seed": 44,
"hyperparameters": {
  "cobolt": {
    "umap_random_state": 42,
    "random_state": { "high": 15, "low": 13, "type": "int" }
  }
}
```

The global seed is `44`, but `umap_random_state` is `42` (a hardcoded default
from the form, not the global seed). `random_state` was separately swept.

### Code location

- GUI input: `multiverse/gui.py:1009–1015` (`_render_run_configuration`)
- Manifest builder: `multiverse/gui.py:87` (`build_run_manifest`)
- Hyperparameter form: `multiverse/gui_utils.py:288–341` (`render_hyperparameters_form`)
  — no logic reads `shared_seed` to pre-fill seed-like params

### Impact

Users who set `random_seed` expecting reproducible model behaviour get
inconsistent results because model-level seed params use their defaults, not the
global seed. The coupling between the two must be done manually, which is
error-prone and non-obvious.

---

## Bug 2 — Sweep parameters in `model_params` are never routed as Optuna sweep jobs

### Summary

When the user toggles **Sweep** on for a hyperparameter in the Configure tab,
`build_run_manifest()` writes the sweep spec dict (e.g.
`{"type": "int", "low": 2, "high": 5, "log": true}`) directly into
`model_params` for that job. It does **not** set `mode: sweep` on the job, and
it does **not** generate a `search_space` block. As a result, the runner treats
these as ordinary run jobs and passes the raw sweep dicts as scalar parameter
values to the model containers, which crash.

### Key distinction: `run_gridsearch` vs sweeps

| Concept | Flag / field | Mechanism | Set by |
|---|---|---|---|
| Grid search | `globals.run_gridsearch: true` | Legacy boolean; maps to `ModelFactory.is_grid_search` (`worker/base.py:69`) | "Run Gridsearch" radio button in Configure tab |
| Optuna sweep | `mode: sweep` + `search_space` on a job | Routed to `run_sweep()` via `tuner.py`; requires `mode == "sweep"` at `cli.py:1311` | Must be hand-authored in the manifest; GUI never emits this |

`run_gridsearch: false` appearing in the manifest even when parameters are swept
is therefore not a bug in itself — it correctly reflects that the radio button
was not clicked. The real problem is that the GUI sweep toggle produces output
that has no execution path.

### Correct manifest format (from `tests/fixtures/run_manifest_sweep.yaml`)

```yaml
jobs:
  - dataset_id: dummy_dataset
    models: [pca]
    mode: sweep
    optimize_metric: total_variance
    n_trials: 3
    search_space:
      n_components:
        type: int
        low: 10
        high: 100
```

### What the GUI actually generates

```yaml
globals:
  run_gridsearch: false    # correct, radio was not clicked
jobs:
  - dataset_slug: pbmc10k
    model_name: PCA
    model_params:
      n_components:        # sweep spec written directly into model_params
        type: int
        low: 2
        high: 5
        log: true
```

No `mode: sweep`, no `search_space`. The runner places this in `run_jobs` (not
`sweep_jobs`) at `cli.py:1311–1312` and passes the dict to the container as a
literal parameter value.

### Code location

- Sweep toggle widget: `multiverse/gui_utils.py:322–331`
- Manifest builder (no `mode`/`search_space` emitted): `multiverse/gui.py:64–115`
- Runner routing: `multiverse/runner/cli.py:1311–1312`
- Correct sweep execution: `multiverse/runner/cli.py:1323–1364`, `multiverse/runner/tuner.py:199–274`

### Impact

Every job with any swept parameter is submitted as a regular run job. The
model container receives dicts instead of scalar values for swept params,
causing crashes at runtime (see Bug 3 below). No Optuna sweep is ever started.

---

## Bug 3 — Container crashes from swept params passed as scalars

Three container runs failed. All failures are downstream of Bug 2 (sweep dicts
in `model_params`) plus one independent preprocessing issue.

---

### 3a — Cobolt crash: `TypeError` from dict passed as `n_latent`

**Workspace:** `2ef3b4b7-0aba-4db7-af9a-0a6627a6a53d`
**Job:** Cobolt × pbmc10k

**Error:**
```
File "/app/run.py", line 46, in __init__
    self.model = Cobolt(
File ".../cobolt/model/cobolt.py", line 76, in __init__
    self.alpha = 50.0 / n_latent if alpha is None else alpha
TypeError: unsupported operand type(s) for /: 'float' and 'dict'
```

**Root cause:** Direct consequence of Bug 2. The `latent_dimensions` parameter
was swept in the Configure tab, so the container received:

```json
"latent_dimensions": {"type": "int", "low": 2, "high": 4, "log": false}
```

The Cobolt constructor uses this value as `n_latent` and immediately tries to
compute `50.0 / n_latent`, which fails because `n_latent` is a dict instead of
an integer. Additionally, `random_state` was also swept and arrived as a dict,
which would have caused a second crash had the first not occurred.

**Fix:** Fix Bug 2 (emit `mode: sweep` + `search_space`). The containers must
only receive scalar values for all parameters.

---

### 3b — PCA crashes: `ValueError` in `highly_variable_genes` (preprocessing overflow)

**Workspaces:**
- `199ce4ed-5e2c-4153-9665-ddb91aabb70f` — PCA × pbmc10k
- `5ee30bfc-90c2-4524-9e56-64312579e4f1` — PCA × teaseq

**Error (both identical):**
```
RuntimeWarning: overflow encountered in expm1
ValueError: cannot specify integer `bins` when input data contains infinity

  File ".../multiverse/worker/io.py", line 293, in preprocess_mudata
      sc.pp.highly_variable_genes(
  File ".../scanpy/preprocessing/_highly_variable_genes.py", line 310
      df["mean_bin"] = _get_mean_bins(df["means"], flavor, n_bins)
  File ".../pandas/core/reshape/tile.py", line 368, in _nbins_to_bins
      raise ValueError(...)
```

**Root cause:** This failure is in `preprocess_mudata` (`worker/io.py:293`),
which runs before any model code. Scanpy's `highly_variable_genes` internally
calls `np.expm1` on the data in its dispersion-based `seurat` / `cell_ranger`
path to reverse log-normalization before computing means. If the data contains
values large enough that `expm1` overflows to `+inf`, the subsequent `pd.cut`
binning step fails because infinity cannot be placed in a finite bin. The
reported `mean_bin` stack matches this dispersion path, not the `seurat_v3`
loess path.

This is an **independent bug from Bug 2**, though both PCA jobs also have
`n_components` arriving as sweep dicts (which would cause a secondary crash in
PCA itself had preprocessing passed).

Likely causes:
- The dataset(s) are being log-normalized twice, or already-large/logged values
  are reaching the `seurat` HVG flavor with `log_normalization=True`.
- Or the dataset contains genuinely extreme values after normalization, large
  enough to overflow `expm1` during HVG mean calculation.

The teaseq log also emits `UserWarning: Cannot join columns with the same name
because var_names are intersecting`, indicating overlapping feature names across
modalities in the MuData object, which may compound the numerical issue.

**Investigation needed:**
1. Verify whether the pbmc10k and teaseq datasets were pre-log-normalized before
   registration, and whether the preprocessing pipeline's `log_normalization`
   step is skipped correctly for such datasets.
2. Check whether `n_top_genes` (for HVG) is correctly set and whether the data
   has the expected count range before HVG is called.

---

## Fix Strategies

---

### Fix 1 — Propagate `random_seed` to seed-like hyperparameter fields

**Where to change:** `multiverse/gui.py` — `_render_configure_tab()`

After `_render_run_configuration()` returns the resolved `random_seed`, and
before the hyperparameter form is rendered for each dataset × model pair, inject
the seed into the widget state for any parameter whose name matches a known
seed-like pattern (`random_state`, `seed`, `umap_random_state`, or any param
ending in `_seed` / `_state`).

The hook point is immediately after the existing
`_prefill_hyperparameter_widget_state(job_key, loaded_params)` call
(`gui.py:1288–1290`). A second call can inject seed defaults **only for fields
the user has not already explicitly set**:

```python
# After the existing loaded-params prefill:
_prefill_seed_params(job_key, schema, random_seed)
```

`_prefill_seed_params` would iterate `schema["properties"]`, check if the param
name matches the seed pattern, and call
`st.session_state.setdefault(f"{job_key}::fixed::{param_name}", random_seed)` —
using `setdefault` so it never overwrites a value the user typed or loaded from
a manifest. This key shape is required because `_render_fixed_widget()` reads
fixed values from `f"{key_prefix}::fixed::{param_name}"`.

This keeps the prefill non-destructive: if a loaded manifest already has a
specific `random_state`, that value wins; only the empty/default case picks up
the global seed.

**No change needed in `build_run_manifest`** — it already passes whatever the
form collected into `model_params`. The fix is entirely in the form rendering
layer.

---

### Fix 2 — Translate GUI sweep specs into `mode: sweep` + `search_space` in the manifest

This is the most consequential fix. Two coordinated changes are required.

#### 2a — Add sweep-level configuration fields to the Configure tab

When any parameter for a job has its Sweep toggle on, the manifest needs
`n_trials`, `optimize_metric`, and `direction`. These are currently missing from
the GUI entirely. Add a collapsible section in `_render_configure_tab()` that
appears per-job (or globally) whenever at least one sweep toggle is active:

```
n_trials        int input,  default 20
optimize_metric text input, e.g. "silhouette_score"
direction       selectbox,  ["maximize", "minimize"]
study_storage   text input, default "sqlite:///optuna.db"
```

Store these in session state keyed by `(ds_name, mod_name)` so each job can
have independent sweep settings, consistent with how `pair_params` and
`pair_mem_limits` already work.

#### 2b — Rewrite `build_run_manifest()` to split scalar vs. sweep params

**File:** `multiverse/gui.py:64–115`

Change the job-building loop so that for any job whose `model_params` dict
contains at least one sweep spec (detectable as
`isinstance(v, dict) and "type" in v`), the output job entry takes the sweep
format instead of the run format:

```python
def _is_sweep_spec(v) -> bool:
    return isinstance(v, dict) and "type" in v

# Inside the job-building loop:
sweep_params = {k: v for k, v in params.items() if _is_sweep_spec(v)}
fixed_params = {k: v for k, v in params.items() if not _is_sweep_spec(v)}

if sweep_params:
    job_entry["mode"] = "sweep"
    job_entry["model_params"] = fixed_params        # scalar overrides only
    job_entry["search_space"] = sweep_params        # Optuna-format dicts
    job_entry["n_trials"] = pair_sweep_config[(ds_name, mod_name)]["n_trials"]
    job_entry["optimize_metric"] = pair_sweep_config[...]["optimize_metric"]
    job_entry["direction"] = pair_sweep_config[...]["direction"]
    job_entry["study_storage"] = pair_sweep_config[...]["study_storage"]
else:
    job_entry["model_params"] = fixed_params
```

This produces manifests that match the job routing the runner expects (as
confirmed by `tests/fixtures/run_manifest_sweep.yaml`), but there is one
compatibility caveat: the GUI's float sweep widget currently returns
`{"type": "float", "low": ..., "high": ..., "log": bool}`, while
`tuner.py:_sample_param()` currently accepts only `int`, `categorical`, and
`loguniform`. Fix 2 must include one of these two changes:

- Extend `_sample_param()` to accept `type: float` with `log: true/false` and
  call `trial.suggest_float(..., log=bool(spec.get("log")))`.
- Or translate GUI float sweep specs in `build_run_manifest()` to the
  distribution names already accepted by the tuner.

Without this additional compatibility fix, integer sweeps would route
correctly, but float sweeps such as Cobolt `learning_rate` could still fail at
Optuna sampling time.

#### 2c — Guard `generate_execution_plan_from_manifest` against residual sweep dicts

As a defensive measure, add a validation pass in
`multiverse/runner/cli.py:generate_execution_plan_from_manifest()` (around
line 915) that warns (or errors) if any value in a non-sweep job's
`model_params` is a dict with a `"type"` key. This would have surfaced the
current failure at manifest-parse time instead of inside the container:

```python
for k, v in model_params.items():
    if isinstance(v, dict) and "type" in v and mode != "sweep":
        logger.error(
            f"Job {dataset_key}/{m_name}: param '{k}' is a sweep spec but "
            f"mode is '{mode}'. Re-generate the manifest from the Configure tab."
        )
```

---

### Fix 3a — Resolved by Fix 2

Once `build_run_manifest()` correctly separates sweep specs from scalar params,
containers will only receive scalar values in `hyperparameters`. The Cobolt
`TypeError` cannot occur.

---




---

## Summary table

| # | Bug | Affected jobs | Severity |
|---|---|---|---|
| 1 | `random_seed` not propagated to model hyperparameter fields | All jobs | Medium — silently breaks reproducibility |
| 2 | GUI sweep specs written to `model_params` instead of `search_space` + `mode: sweep` | All jobs with any swept param | Critical — sweeps never execute; containers crash |
| 3a | Cobolt crash: sweep dict received as `n_latent` | Cobolt × pbmc10k | Critical — direct crash from Bug 2 |
| 3b | PCA crash: `expm1` overflow in HVG preprocessing | PCA × pbmc10k, PCA × teaseq | Critical — independent preprocessing bug |

| Fix | Touches | Resolves |
|---|---|---|
| Fix 1: prefill seed params from `shared_seed` | `gui.py` (form rendering only) | Bug 1 |
| Fix 2a: add sweep config fields to Configure tab | `gui.py` | Bug 2 |
| Fix 2b: split scalar vs sweep in `build_run_manifest` | `gui.py` | Bug 2, Bug 3a |
| Fix 2c: guard execution plan against residual sweep dicts | `runner/cli.py` | Early detection of Bug 2 regressions |
| Fix 3b (short-term): validate/detect invalid HVG input before HVG | `worker/io.py` | Bug 3b |
| Fix 3b (long-term): `is_log_normalized` in dataset registry | registry schema + `worker/io.py` | Bug 3b root cause |
