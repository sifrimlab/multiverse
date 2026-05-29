# multiverse — Gap-Closure Strategy v3

**Recheck date:** 2026-05-29
**Predecessor:** the G1–G6 implementation strategy completed on 2026-05-29.
G1–G6 closed the M0–M5 architectural skeleton gaps (accept-degraded guard,
user_id threading, reservation timeline rebuild, SIF dual-digest, real-Apptainer
integration suite, registry_db structural deletion). This document replaces
that strategy entirely, records the completed items, and introduces the next
wave of gaps (H1–H4) that address the HPC researcher workflow.

**Design correction (from G1):** the original G1 spec imposed
`accept_degraded=False` as the default for both the Docker and Slurm
executors. This was revised in v3: locally-built Docker images are the normal
development workflow for researchers who never push to a registry. The correct
defaults are:

| Executor | Default | Override |
|---|---|---|
| `MvdDockerExecutor` | open (`accept_degraded=True`) | `--strict` opts into requiring a registry digest |
| `MvdSlurmExecutor` | strict (`accept_degraded=False`) | `--accept-degraded` allows unverified SIFs |

---

## Product boundary (unchanged)

| Axis | Status |
|---|---|
| Single-user, local workstation execution | **In scope** |
| HPC execution (Slurm + Apptainer/Singularity) | **In scope** |
| Multi-user on a shared install | **Design constraint** — thread `user_id`; do not implement tenancy enforcement |
| Hosted service / multi-node kernel | **Out of scope** |
| Legacy migration of pre-mvd runs | **Out of scope** |

---

## Completed gaps (G1–G6)

| ID | Gap closed | Acceptance notes |
|---|---|---|
| **G1** | `accept_degraded` guard in Docker + Slurm executors | `test_accept_degraded.py` (8 tests). Default corrected to open for Docker; strict for Slurm — see design note above. |
| **G2** | `user_id` threaded through journal records + artifact manifest | `test_user_id_propagation.py` (5 tests). `rebuild-index` populates `user_id` column; pre-G2 journals round-trip as `NULL`. |
| **G3** | `rebuild-index` reconstructs reservation timeline | `test_reservation_timeline_rebuild.py` (3 tests). Schema v4 adds `reservation_events` table; idempotent v2→v4 and v3→v4 migrations. |
| **G4** | `RealSlurmEngine.sif_digest_for_submission` | `test_mvd_slurm_executor.py`. Dual-digest manifest fields populated end-to-end. Cache keyed by (path, mtime_ns, size). |
| **G5** | Real-Apptainer integration suite; R1 OOM classification | `tests/integration/test_mvd_real_apptainer_path.py`. Skips cleanly when user namespaces are unconfigured; OOM test fails (not skips) on regression. |
| **G6** | `registry_db.py` structural deletion; sole-writer invariant | Net ~2000 lines deleted. `asset_registry.py` is canonical. `test_sqlite_writer_isolation.py` bakes the invariant into CI. `migrate-asset-registry` CLI added. |

## Pending manual evidence

| ID | Gap | Status |
|---|---|---|
| **G7** | Real-cluster M4 evidence (`multiverse slurm-submit` on a live Slurm node) | Pending. All code prerequisites are in place (G1, G4). Run per the G7 instructions below; commit captured manifest to `docs/evidence/`. |

### G7 — Real-cluster M4 evidence

On a real login node with `sbatch`, `sacct`, `scancel`, `apptainer` on PATH,
and a partition available:

```bash
multiverse slurm-submit \
  --state-root $HOME/.mvexp \
  --model-slug pca \
  --image-sif /path/to/pca.sif \
  --image-digest sha256:... \
  --dataset-slug demo \
  --dataset-path /path/to/data.h5mu \
  --dataset-n-obs 1234 \
  --partition <p> --time-minutes 30 --mem-gb 4
```

Verify the printed snapshot reports `primary_state = ARTIFACT_SUCCESS`, the
manifest carries both `image_identity.value == sha256:...` and a populated
`runtime_image_identity.value` (sha256 of the SIF) with
`runtime_image_identity.built_from == image_identity.value`. Verify
`multiverse doctor --deep-slurm` reports `engines.slurm_deep` PASS. Commit
the captured manifest (paths redacted) to
`docs/evidence/m4-real-cluster-2026-MM-DD.md`.

---

## New gap overview

| ID | Gap | Closes acceptance for | Effort | Depends on |
|---|---|---|---|---|
| **H1** | `model.yaml` has no Apptainer runtime field; models table has no SIF path | HPC researcher workflow | 1 day | — |
| **H2** | `multiverse run --manifest` has no Slurm routing; researchers must call `slurm-submit` once per job by hand | HPC researcher workflow | 2–3 days | H1 |
| **H3** | No `multiverse build-sif` command; SIF creation from existing Dockerfiles is undocumented and unautomated | HPC researcher workflow | 2 days | H1 |
| **H4** | Researcher-authored models have no Singularity.def scaffold; the model-creation guide covers Docker only | HPC researcher workflow | 1 day | H3 |

H1 is the prerequisite for everything. H2 and H3 are independent once H1
lands. H4 is a documentation and tooling polish that can follow H3.

---

## H1 — Apptainer runtime field in `model.yaml` and models table

**Goal:** a researcher can register a model that has both a Docker image (for
local runs) and a SIF path (for HPC runs). The models table stores both.
`multiverse run` uses whichever backend is active.

**Why now:** `model.yaml` only has `runtime.image` (a Docker tag). The Slurm
executor receives `image_sif` as a CLI argument every time, disconnected from
the model registry. A researcher who builds a SIF from their own Dockerfile
has no place to record it, so `multiverse run --backend slurm` (H2) cannot
look it up automatically.

**Work items:**

1. **`model.yaml` schema** — extend `ModelManifest` in
   [`multiverse/models_ingest.py`](multiverse/models_ingest.py) with an
   optional `ApptainerSpec`:
   ```python
   class ApptainerSpec(BaseModel):
       sif_path: Optional[str] = None      # absolute or state-root-relative
       build_from: Optional[str] = None    # "dockerfile" | "def_file"
       def_file: Optional[str] = None      # path to Singularity.def
   ```
   Add `apptainer: Optional[ApptainerSpec] = None` to `ModelManifest`.
   Both `runtime.image` and `apptainer` are independently optional — a
   Docker-only model omits `apptainer`; an Apptainer-only model may omit
   `runtime.image`. Relax the current `NOT NULL` requirement on `docker_image`
   to require at least one of the two to be present (validation in
   `ModelManifest.__model_post_init__`).

2. **Models table** — add `sif_path TEXT` column to the `models` table in
   both [`multiverse/registry_db.py`](multiverse/registry_db.py) and
   [`multiverse/asset_registry.py`](multiverse/asset_registry.py). Add an
   idempotent `ALTER TABLE models ADD COLUMN sif_path TEXT` migration in both
   `init_db()` and `init_asset_registry()`. Existing rows get `NULL`; that is
   correct and expected.

3. **Ingestion** — update `register_model_from_manifest` in
   [`multiverse/models_ingest.py`](multiverse/models_ingest.py) to write
   `sif_path` from `manifest.apptainer.sif_path` when present. Add a
   `multiverse register-model --set-sif-path <path>` flag that updates only
   the `sif_path` column on an already-registered model, so
   `multiverse build-sif` (H3) can record its output without re-running full
   registration.

4. **Query surface** — add `get_model_sif_path(conn, slug, version) ->
   Optional[str]` alongside `get_all_models` in `registry_db.py`. The Slurm
   manifest runner (H2) uses this to resolve `image_sif` automatically when
   no explicit path is provided.

5. **Tests** — `tests/unit/test_model_apptainer_field.py`:
   - a `model.yaml` with `apptainer.sif_path` set round-trips through
     `ModelManifest` and registration;
   - a `model.yaml` with neither `runtime.image` nor `apptainer` is rejected
     at validation with a clear error message;
   - `get_model_sif_path` returns `None` for a Docker-only registered model
     (no regression).

**Acceptance:**
- `model.yaml` can express a SIF path alongside (or instead of) a Docker
  image.
- `multiverse register-model` stores it; `multiverse register-model
  --set-sif-path` updates it post-hoc.
- The models table migration is idempotent and does not break existing rows.

---

## H2 — Manifest-driven Slurm execution (`multiverse run --backend slurm`)

**Goal:** a researcher can write a `run_manifest.yaml` with
`globals.backend: slurm` and run `multiverse run --manifest
run_manifest.yaml` on an HPC login node to submit all jobs through
`MvdSlurmExecutor`, exactly as the Docker path submits through
`MvdDockerExecutor`. No shell loop over `slurm-submit` required.

**Why now:** `multiverse slurm-submit` handles one job per invocation and
requires the caller to supply `--image-sif`, `--dataset-path`,
`--dataset-n-obs` etc. manually. For a manifest with ten dataset × model
pairs the researcher must write a ten-iteration shell loop, defeating the
purpose of the manifest format.

**Work items:**

1. **Manifest schema** — add an optional `backend` key to manifest globals.
   Allowed values: `"docker"` (default, existing behaviour) and `"slurm"`.
   Add a `slurm` sub-dict to globals for Slurm-level defaults:
   ```yaml
   globals:
     backend: slurm
     slurm:
       partition: gpu
       account: mylab
       time_minutes: 120
       mem_gb: 32
       cpus_per_task: 8
       gpus: 1
   ```
   Per-job overrides in the same shape merge on top of globals (job-level
   wins). Update `parse_manifest` in
   [`multiverse/runner/cli.py`](multiverse/runner/cli.py) to validate these
   fields and propagate them into each job dict.

2. **SIF resolution** — when `backend == "slurm"`, each job dict must carry
   an `image_sif` path. Resolve in order:
   1. Per-job `image_sif` key in the manifest.
   2. `get_model_sif_path(conn, slug, version)` from the registry (H1).
   3. Manifest validation error: "no SIF path for model `<slug>`; register
      one with `multiverse register-model --set-sif-path` or set
      `image_sif` in the manifest."
   An `image_digest` field follows the same resolution but is optional —
   omitting it is allowed; the resulting manifest records `unverified_local`
   and the artifact is tagged as not publication-quality.

3. **Router** — add `--backend` flag to `add_run_args` in
   [`multiverse/runner/cli.py`](multiverse/runner/cli.py) (values: `docker`,
   `slurm`; default: reads from manifest globals, falls back to `docker`).
   In `execute_run`, route to a new `run_via_slurm(args)` in
   [`multiverse/runner/mvd_entrypoint.py`](multiverse/runner/mvd_entrypoint.py)
   when the backend is `slurm`.

4. **`run_via_slurm`** — mirrors `run_via_mvd` but builds a
   `MvdSlurmExecutor` instead of `MvdDockerExecutor`. Reuses a new
   `_drive_jobs_slurm` helper (Slurm-specific variant of `_drive_jobs`) that
   constructs `RealSlurmEngine` and populates Slurm-specific options from the
   job dict. `accept_degraded` defaults to `False`; `--accept-degraded` on
   the CLI overrides.

5. **GUI** — the **Run** tab should display a read-only "Slurm backend"
   indicator when a manifest with `backend: slurm` is loaded. Full GUI
   Slurm submission can be a follow-up; the CLI path is the primary
   interface for HPC runs.

6. **Tests** — `tests/unit/test_slurm_manifest_run.py`:
   - `parse_manifest` with `backend: slurm` extracts partition, account,
     time, mem, and SIF path;
   - missing SIF path raises a manifest validation error before any job is
     submitted;
   - `run_via_slurm` with `InMemorySlurmEngine` drives three jobs to
     `ARTIFACT_SUCCESS`;
   - per-job `slurm` override merges correctly over globals.

**Acceptance:**
- `multiverse run --manifest run_manifest.yaml` on a node with Slurm routes
  all jobs through `MvdSlurmExecutor` when `globals.backend: slurm`.
- Missing SIF path is caught at manifest validation, not at job launch.
- The existing Docker manifest path is unaffected (no regression).

---

## H3 — `multiverse build-sif` command

**Goal:** a researcher can build an Apptainer SIF from a model's existing
Dockerfile (or a `Singularity.def`) with one command, and have the result
automatically registered as the model's `sif_path`.

**Why now:** converting a Dockerfile to a SIF requires knowing the right
`apptainer build` invocation, dealing with the `docker-daemon://` URI scheme,
handling overlay filesystem differences on shared HPCs, and writing the output
path somewhere. None of this is documented or automated. Researchers are
expected to figure it out manually, which is a barrier to HPC adoption.

**Work items:**

1. **CLI command** — add `multiverse build-sif` as a new entry in
   [`multiverse/cli_entrypoints.py`](multiverse/cli_entrypoints.py):
   ```
   multiverse build-sif --slug <model> [--output-dir <dir>] [--method docker-daemon|def-file]
   ```
   - `--method docker-daemon` (default when `build.dockerfile` exists in
     `model.yaml`): converts the already-built local Docker image via
     `apptainer build <slug>-<version>.sif docker-daemon://<image>:<tag>`.
     Requires the Docker daemon to be running and the image to be present
     locally (`docker image inspect` as a preflight check).
   - `--method def-file`: runs
     `apptainer build <slug>-<version>.sif <def_file>` where `def_file` is
     read from `model.yaml`'s `apptainer.def_file` field (H1). Useful on
     HPC nodes where Docker is absent.
   - `--output-dir`: defaults to `<state_root>/sif/`. Created if absent.
   - After a successful build, calls the H1 `--set-sif-path` update so
     subsequent manifest runs (H2) resolve the path automatically.

2. **Preflight checks** — before invoking `apptainer build`:
   - confirm `apptainer` (or `singularity`) is on PATH;
   - for `docker-daemon`: confirm the Docker image exists locally;
   - for `def-file`: confirm the `.def` file is readable;
   - check that the output path does not already exist (or accept `--force`
     to overwrite).
   Emit a clear, actionable error for each failure; do not silently skip.

3. **Streaming output** — stream `apptainer build` stdout/stderr in real
   time so the researcher sees build progress. Exit non-zero on build
   failure; do not register a broken SIF.

4. **Makefile target** — add `make build-sif slug=<model>` that calls
   `uv run multiverse build-sif --slug <slug>`, consistent with the existing
   `make build-pca` / `make register slug=<slug>` patterns.

5. **Tests** — `tests/unit/test_build_sif_cmd.py` (no real `apptainer`
   required; mock subprocess):
   - preflight fails cleanly when `apptainer` is absent;
   - preflight fails cleanly when the Docker image is missing;
   - successful mock build calls `register_model_sif_path` with the correct
     output path;
   - `--output-dir` override is respected.
   Integration test in `tests/integration/test_build_sif_real.py` with a
   skip guard on `apptainer` presence, matching the G5 pattern.

**Acceptance:**
- `make build-sif slug=pca` on a workstation with Docker + Apptainer
  produces `<state_root>/sif/pca-1.0.0.sif` and updates the registry row.
- `make build-sif slug=pca --method def-file` on an HPC node (no Docker)
  produces the same output from a `Singularity.def`.
- A subsequent `multiverse run --manifest run_manifest.yaml --backend slurm`
  resolves the SIF path from the registry without `--image-sif` on the
  command line.

---

## H4 — `Singularity.def` scaffold for researcher-authored models

**Goal:** a researcher adding a new model to multiverse can generate a
`Singularity.def` that implements the same container contract as the existing
`Dockerfile`, so HPC runs require no Docker dependency at any point in the
workflow.

**Why now:** [`docs/ADDING_A_MODEL.md`](docs/ADDING_A_MODEL.md) and all
existing `store/models/<slug>/container/` directories have only a `Dockerfile`.
A researcher on a cluster where Docker is banned (common on shared HPC) has no
starting point. H3 covers the build automation; H4 covers the authoring
scaffold.

**Work items:**

1. **`Singularity.def` templates** — add
   `store/models/<slug>/container/Singularity.def` for each built-in model
   (pca, mofa, multivi, mowgli, cobolt, totalvi). The def files mirror the
   existing `environment.yml` conda env definition:
   ```singularity
   Bootstrap: docker
   From: mambaorg/micromamba:1.5.1

   %files
       store/models/pca/container/environment.yml /opt/environment.yml
       sdk/ /opt/sdk/

   %post
       micromamba install -y -n base -f /opt/environment.yml
       micromamba clean --all --yes
       pip install /opt/sdk/mvr-worker/ --no-deps

   %runscript
       exec python /opt/sdk/mvr-worker/run.py "$@"
   ```
   The `%runscript` must invoke the same entry point as the `Dockerfile`'s
   `CMD` so the container contract (`/input/data.h5mu`,
   `/output/job_spec.json`, `/output/`) is identical across both backends.

2. **`model.yaml` update** — for each built-in model, add:
   ```yaml
   apptainer:
     build_from: def_file
     def_file: store/models/pca/container/Singularity.def
   ```
   This lets `multiverse build-sif --method def-file` work for built-in
   models on Docker-free HPC nodes.

3. **`ADDING_A_MODEL.md` update** — add a "Building for HPC" section that:
   - explains the `Singularity.def` structure above;
   - shows how to test the SIF locally:
     ```bash
     apptainer exec \
       --bind <data>:/input:ro \
       --bind <workspace>:/output \
       <slug>.sif python /opt/sdk/mvr-worker/run.py
     ```
   - links to `multiverse build-sif` and `multiverse register-model
     --set-sif-path`;
   - documents the precondition check `multiverse doctor --deep-slurm`.

4. **Tests** — `tests/unit/test_model_def_file.py`:
   - each built-in model's `Singularity.def` passes a lightweight syntax
     check (presence of `Bootstrap:` and `%runscript` sections);
   - `model.yaml` for each built-in model passes the updated
     `ModelManifest` validation with the new `apptainer` field.

**Acceptance:**
- Every built-in model has a `Singularity.def` and a `model.yaml` with
  `apptainer.def_file` set.
- `multiverse build-sif --slug pca --method def-file` runs to completion on
  a host with `apptainer` but without Docker.
- `docs/ADDING_A_MODEL.md` covers the full HPC authoring path from writing a
  `Singularity.def` to `multiverse run --backend slurm`.

---

## Sequencing

```
H1  model.yaml SIF field + models table migration
 ├── H2  manifest-driven Slurm run (uses H1 SIF resolution)
 └── H3  build-sif command (uses H1 --set-sif-path hook)
       └── H4  Singularity.def scaffold (uses H3 --method def-file path)
```

G7 (real-cluster evidence) can be run as soon as H1 + H2 land using
`multiverse run --manifest run_manifest.yaml --backend slurm` rather than
`slurm-submit` directly, giving a more realistic evidence artifact.

---

## Risks and open decisions (carried forward)

- **R2 — `sacct` reliability.** On busy clusters, `sacct` lags `squeue` by
  seconds-to-minutes. `MvdSlurmExecutor._poll_until_terminal` treats PENDING
  as transient, but the 24 h timeout is the only guard against a stuck job.
  Surface a warning when a job has been PENDING beyond a configurable
  threshold (default 1 hr). Not a gap-closure deliverable.
- **R3 — User namespace on shared filesystems.** G2 persists `user_id` but
  does not enforce path isolation. The doctor should warn when `state_root` is
  on a shared FS (`statfs` reports NFS / GPFS / Lustre). Doctor backlog item.
- **D1 — SIF distribution channel.** H3 covers building SIFs from local
  sources; it does not cover distributing them to compute nodes. On most
  clusters a shared filesystem path (e.g. `/scratch/shared/sif/`) is
  sufficient. A pull-from-OCI path (`oras://`) is noted as a future option
  once H3 is stable.
- **D2 — Mode B (kernel inside one allocation).** Not in this plan. Revisit
  after G7 and H2 land and we have real Mode A user feedback.
- **D3 — GPU node selection.** The Slurm manifest globals (H2) accept
  `gpus: N` but do not validate against available partition resources. A
  `multiverse doctor --check-slurm-partition <name>` probe is a useful
  follow-up after H2.

---

## Verified pre-existing test failures (not gap items)

Two unit tests remain red and are tracked separately:

- `tests/unit/test_manifest_gate.py::test_empty_plan_blocks_launch`
- `tests/unit/test_planner.py::test_generate_execution_plan`

Both sit in the legacy planner code path. They were red before M0–M5 work
began and do not cover any mvd-era behavior. They will be removed when the
legacy planner is eventually deleted.
