# Issue Tracker

> Last updated: 2026-05-27

---

## Issue 1 — MLflow `Permission denied: '/data'` on artifact upload
**Status:** Open  
**Severity:** High  
**Source:** `store/artifacts/run_output/multiverse.log` (run 20260526T090149)  
**Files:** `multiverse/tracking.py`, `docker-env/entrypoint-mlflow.sh`

### Description
Every `mlflow.log_artifact()` call on the host fails with `[Errno 13] Permission denied: '/data'`.
The MLflow server runs inside Docker with `DATA_DIR=/data`, so all experiments are registered in
`store/mlflow.db` with `artifact_location = /data/artifacts/<id>`. When the host process tries to
log artifacts it resolves that as a literal host path, which does not exist or is not writable.

```
sqlite> SELECT name, artifact_location FROM experiments;
testname1  ->  /data/artifacts/3
run_output ->  /data/artifacts/1
```

### Review
Root cause is correctly identified at the storage-layout level: the MLflow DB contains
container-local artifact roots (`/data/artifacts/<id>`), and host-side artifact logging is
failing when the client resolves that path locally.

The proposed preferred fix was incomplete. `multiverse/tracking.py` already defaults
`MLFLOW_TRACKING_URI` to `http://localhost:5000`; using an HTTP tracking URI alone does not
guarantee artifact proxying. With the current MLflow server command, experiments still advertise
`/data/artifacts/...` as their artifact location, so the client can still try to write to `/data`
from the host.

### Fix options
- **Preferred:** Configure the MLflow server to proxy artifacts, not just track runs over HTTP.
  Update `docker-env/entrypoint-mlflow.sh` to use served artifacts, for example with
  `--serve-artifacts`, `--artifacts-destination "$ARTIFACT_ROOT"`, and a proxy artifact root such
  as `mlflow-artifacts:/`. Then recreate affected experiments, or migrate their
  `artifact_location`, because existing experiments keep the old `/data/artifacts/<id>` roots.
- **Alternative:** Run MLflow on the host, or recreate the experiments with a `file://` artifact
  root that is valid for the process doing artifact uploads, e.g.
  `file:///mnt/data/saptarshi/testmv/mvr/store/artifacts`. This is simpler for host-only logging
  but is less portable for containers unless the same path is mounted there.
- Add a regression check that starts a temporary run and logs a small artifact from the host
  runner. The check should fail if the experiment artifact URI resolves to a non-host path like
  `/data/...`.

---

## Issue 2 — `evaluate_single_run` ImportError (first run only)
**Status:** Resolved in code, not yet verified in production  
**Severity:** Medium  
**Source:** `multiverse.log` (run 20260526T090149, lines 24/32/42/51)  
**Files:** `multiverse/evaluate.py`, `multiverse/runner/docker_runner.py`

### Description
First run (11:01 AM) failed per-job evaluation with:
```
cannot import name 'evaluate_single_run' from 'multiverse.evaluate'
```
`evaluate_single_run` now exists at `multiverse/evaluate.py:297` and is called from
`docker_runner.py:799`.

### Review
Current code contains both the function and the Docker-runner callsite, so the immediate import
surface is fixed. The log proves the runtime used by that earlier run did not contain
`evaluate_single_run`, but the stated cause ("older installed version after not reinstalling") is
only partially proven. The import error path points at the working tree
(`/mnt/data/saptarshi/testmv/mvr/multiverse/evaluate.py`), so another plausible explanation is
that the run started before the source file was updated, or from a long-lived process that had an
older module already imported.

### Fix
After any code change to the package, make sure the execution environment and long-lived processes
are refreshed:
- For command-line runs, use the project environment explicitly, e.g. `uv run python -m multiverse.runner.cli ...`.
- If using an editable install, reinstall/sync with `pip install -e .` or `uv sync` after dependency or packaging changes.
- Restart Streamlit or any runner process that may have imported `multiverse.evaluate` before the change.
- Verify with the same launcher used for jobs:
  `uv run python -c "from multiverse.evaluate import evaluate_single_run; print(evaluate_single_run)"`.

The second run no longer showed this import error, so the symptom appears resolved, but it is still
worth adding a small import smoke test for the runner environment.

---

## Issue 3 — PCA evaluation crashes with `n_components = -1`
**Status:** Open  
**Severity:** Medium  
**Source:** `multiverse.log` (run 20260526T133209, lines 79/85/91)  
**Files:** `multiverse/evaluate.py` (`evaluate_single_run`)

### Description
Per-job evaluation fails for all models in the second run:
```
Per-job evaluation failed for run 15: The 'n_components' parameter of PCA must be an int
in the range [0, inf) ... Got -1 instead.
```
The current root-cause explanation was likely incorrect or incomplete. In the same log,
evaluation first says:
```
batch_key 'batch' not found in obs; batch-correction metrics will be disabled.
```
With the current code, that makes `have_batch = False`, and therefore
`pcr_comparison=have_batch and have_matrix` is already `False`. A latent embedding with two
dimensions is therefore unlikely to be the direct reason for `n_components = -1`.

The more likely root cause is that `evaluate_single_run` can hand scib-metrics an AnnData object
with no expression features. For `.h5mu` datasets, `_mudata_to_evaluation_anndata()` constructs
`ad.AnnData(obs=obs)`, which has `n_vars == 0`; for embedding-only `.h5ad` files the same can be
true. scib-metrics/scanpy preprocessing can then derive a PCA component count from the expression
matrix as `n_vars - 1`, producing `-1` when `n_vars == 0`, before or outside the specific
`pcr_comparison` metric flag.

### Deferred future work
Do not implement this in the current MLflow/GUI cleanup pass. Track it for a later evaluation-focused
change with this exact intent:
- Avoid calling `Benchmarker` when the evaluation AnnData has fewer than 2 expression features.
- For `.h5mu`, build evaluation AnnData from a real modality matrix, preferring `rna` when present.
- Preserve model-native metrics with `evaluation: {}` when no valid matrix exists.
- Catch benchmark failures locally so optional evaluation cannot break run completion.

---

## Issue 4 — MLflow `Run is already active` on finalize
**Status:** Open  
**Severity:** Medium  
**Source:** `multiverse.log` (run 20260526T133209, lines 80/86/92)  
**Files:** `multiverse/tracking.py:250-252` (`finalize_parent_mlflow_run`)

### Description
When jobs finalize, a different active run can already be in context and MLflow throws:
```
Run with UUID <x> is already active. To start a new run, first end the current run
with mlflow.end_run(). To start a nested run, call start_run with nested=True
```

Current code at `tracking.py:250-252`:
```python
active = mlflow.active_run()
if active is None or active.info.run_id != run_id:
    mlflow.start_run(run_id=run_id)   # crashes if a different run is still open
```

### Review
The symptom is correctly identified, but the root cause is broader than Issue 3. The parent MLflow
run is intentionally opened before the container starts and left active until finalization. In the
async runner, `_open_parent_mlflow_run()` and `finalize_parent_mlflow_run()` both execute through a
thread pool. That means a finalizer can run on a worker thread that still has another job's parent
run active in MLflow's fluent run stack.

So the active-run conflict can happen even if evaluation succeeds. The evaluation crash just makes
the sequence more visible in the logs.

### Fix
Do not blindly `mlflow.end_run()` a different active run in `finalize_parent_mlflow_run()`. In
parallel execution, that "stale" run may belong to another in-flight job and ending it can corrupt
that job's tracking state.

Better fixes:
- Prefer the lower-level `MlflowClient` API for parent-run lifecycle: create or get the run by
  `run_id`, log params/metrics/artifacts with explicit `run_id`, and terminate the run with
  `set_terminated`. Avoid depending on process/thread-local fluent `active_run()` state for
  long-running concurrent jobs.
- If fluent MLflow APIs must remain, isolate each job's MLflow lifecycle so the same worker/thread
  owns open/finalize, or close the fluent run immediately after creating the parent and rely on
  explicit `run_id` attachment for later writes. Document the tradeoff if that disables host system
  metrics for the whole container runtime.
- Add a unit test where `finalize_parent_mlflow_run()` is called while a different active run is
  present. The expected behavior should be "do not end the other run and still log/terminate the
  target run" if using `MlflowClient`, or "skip with warning" for a minimal interim fix.

---

## Issue 5 — "Load Manifest Settings" button appears to do nothing
**Status:** Open  
**Severity:** Medium  
**Source:** User report  
**Files:** `multiverse/gui.py:593-609` (`_render_load_manifest_panel`)

### Description
Clicking the "Load Manifest Settings" button on the Configure tab gives no visible feedback and
the user sees no change. The underlying data loading logic is correct - the
`_pending_shared_*` -> `shared_*` pending-apply pattern (via `init_state()` in `gui_state.py` and
`_apply_pending_shared_config()` in `gui.py`) works as expected. There are two UX problems:

1. **Success message is swallowed by `st.rerun()`.**
   `st.success("Loaded manifest settings.")` is immediately cleared when `st.rerun()` fires on
   the next line, so the user sees nothing.

2. **The affected fields are far below the button.**
   The "Run Configuration" widgets (Experiment Name, Seed, Run Mode) live at the very bottom of
   the Configure tab, after the full job matrix and all hyperparameter expanders. After the rerun
   the scroll position resets to the top; the user never sees what changed.

### Review
Root cause is correctly identified. The pending-state mechanism is present in both
`gui_state.py:init_state()` and `gui.py:_apply_pending_shared_config()`, so this is mainly a
feedback/discoverability bug, not a data-loading bug.

The proposed `st.toast()` fix may still be too transient if it is emitted immediately before
`st.rerun()`. A session-state-backed notice is more deterministic.

### Fix
Store a one-shot notice before rerun, then display and clear it on the next render:
```python
st.session_state["_manifest_load_notice"] = "Manifest settings loaded."
st.rerun()
```

Near the top of `_render_configure_tab()` after `_apply_pending_shared_config()`:
```python
if "_manifest_load_notice" in st.session_state:
    st.success(st.session_state.pop("_manifest_load_notice"))
```

Also move "Run Configuration" above the job matrix, or show a compact summary directly under the
load button:
- Experiment name
- Random seed
- Run mode
- Manifest path

That makes the button's effect visible without requiring the user to scroll to the bottom of the
Configure tab.

---

## Issue 6 — `use_container_width` deprecation warnings
**Status:** Open  
**Severity:** Low  
**Source:** Streamlit stderr (2026-05-27 09:24:07)  
**Files:** `multiverse/gui.py`, `multiverse/gui_artifacts.py`, `multiverse/gui_navigation.py`

### Description
Streamlit 1.55 emits two deprecation warnings on every startup:
```
Please replace `use_container_width` with `width`.
use_container_width=True  ->  width='stretch'
use_container_width=False ->  width='content'
```
The two warnings at startup come from the registry tab's datasets and models dataframes (the
default landing tab). The current diff introduced at least part of this regression by replacing
previously correct `width="stretch"` calls with `use_container_width=True`:
```diff
- st.dataframe(pd.DataFrame(ds_rows), width="stretch")
+ st.dataframe(pd.DataFrame(ds_rows), use_container_width=True, ...)
```

### Affected locations (all need `use_container_width=True` -> `width='stretch'`)

| File | Line | Widget |
|---|---|---|
| `gui.py` | 116 | `st.dataframe` (manifest errors) |
| `gui.py` | 301 | `st.dataframe` (registry datasets table) - regression |
| `gui.py` | 328 | `st.dataframe` (registry models table) - regression |
| `gui.py` | 697 | `st.data_editor` (job matrix) |
| `gui.py` | 866 | `st.dataframe` |
| `gui.py` | 1043 | `st.dataframe` |
| `gui.py` | 1175 | `st.dataframe` (wave_df) |
| `gui.py` | 1332 | `st.dataframe` |
| `gui.py` | 1349 | `st.dataframe` (summary_df) |
| `gui.py` | 1496 | `st.dataframe` (metrics_df) |
| `gui.py` | 1589 | `st.link_button` (MLflow) |
| `gui.py` | 1591 | `st.link_button` (Optuna) |
| `gui_artifacts.py` | 110 | `st.dataframe` |
| `gui_navigation.py` | 54 | `st.button` (nav bar) |

### Review
Root cause is correct. The repository's `uv.lock` currently resolves Streamlit 1.55.0, and the
listed calls use the deprecated `use_container_width` argument. The git diff also confirms that at
least some previously correct `width="stretch"` calls were changed back to
`use_container_width=True`.

### Fix critique and improvements
Replace all listed usages in one pass and add a simple grep-based test or lint check so this does
not regress:
```bash
rg "use_container_width" multiverse tests
```

Use:
- `width="stretch"` for `use_container_width=True`
- `width="content"` for `use_container_width=False`

Because `pyproject.toml` still allows `streamlit>=1.35.0`, verify that the chosen `width` API is
available for the minimum supported Streamlit version, or raise the minimum supported version to
the version where the new API is guaranteed. Since `uv.lock` is already on 1.55.0, the practical
short-term fix is to update the call sites and keep the lockfile consistent.
