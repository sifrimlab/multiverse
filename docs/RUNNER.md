# Runner & Orchestration

This page documents the current execution path for turning `run_manifest.yaml` into validated artifact bundles.

## Entry Points

| Surface | Command | Notes |
|---|---|---|
| Canonical CLI (Docker) | `multiverse run --manifest <path> --output <dir>` | Installed command; delegates to the mvd-backed Docker runner. |
| Canonical CLI (Slurm) | `multiverse slurm-submit --model-slug <slug> --image-sif <path> [--image-digest sha256:…]` | Submit a single job through the Slurm + Apptainer path. |
| Simple-mode (no daemon) | `multiverse run --simple <simple_manifest.yaml> --out <dir>` | Contract-only runner: no mvd, SQLite, MLflow, or Optuna required. See [Simple-Mode Runner](#simple-mode-runner). |
| Build SIF | `multiverse build-sif --slug <slug> --output-dir <dir>` | Build an Apptainer SIF from a model's Dockerfile or Singularity.def. See [Building SIF Files](#building-sif-files). |
| Source checkout | `uv run multiverse run --manifest <path> --output <dir>` | Same Docker run path without installing the package globally. |
| Compatibility CLI | `python -m multiverse.runner.cli run --manifest <path> --output <dir>` | Kept for compatibility; prefer `multiverse`. |
| GUI | Streamlit **Run** tab | Submits to the in-process mvd controller; it does not spawn the runner as a subprocess. |
| Maintenance | `multiverse doctor`, `multiverse rebuild-index`, `multiverse gc --dry-run`, `multiverse mlflow-sync`, `multiverse migrate-asset-registry`, `multiverse migrate-state-dir` | Recovery and projection commands. Use `uv run multiverse ...` from a source checkout. |

## Execution Pipelines

### Docker (default)

```mermaid
flowchart LR
    A[run_manifest.yaml] --> B[Parse and validate]
    B --> C[mvd kernel]
    C --> D[Resource broker]
    D --> E[Docker supervisor]
    E --> F[Model container]
    F --> G[Semantic validation]
    G --> H[Promotion saga]
    H --> I[Artifact bundle]
    I --> J[SQLite rebuildable index]
    I --> K[MLflow projection]
```

1. **Parse and validate.** The manifest is checked against the local registry before any run is submitted.
2. **Submit to mvd.** The CLI and GUI submit jobs through the kernel/client boundary. The kernel owns state transitions, cancellation, and execution tasks.
3. **Launch Docker through the supervisor.** The supervisor labels containers, records launch intent in the journal, and polls through the `RealDockerEngine` adapter.
4. **Write the model contract.** Each workspace receives `job_spec.json`; the container sees `/input/data.h5mu`, `/output/job_spec.json`, and `/output/`.
5. **Validate outputs.** Successful container exit is not enough. Required artifacts are opened and checked before promotion.
6. **Promote through a saga.** The workspace is staged under an owned staging directory and atomically renamed into the artifact store only after validation.
7. **Project to MLflow.** MLflow is a projection. A valid artifact bundle can be scientifically successful even if tracking sync is pending or failed.

### Slurm + Apptainer

On HPC clusters without Docker, use the Slurm path:

```bash
uv run multiverse slurm-submit \
  --model-slug pca \
  --image-sif /scratch/images/multiverse-pca-1.0.0.sif \
  --image-digest sha256:<oci-digest> \
  --params-json '{"n_components": 20}' \
  --output store/artifacts/run_output
```

The Slurm executor:

- Submits an `sbatch` script that calls `apptainer exec` against the provided SIF.
- Computes a sha256 digest of the SIF file at submission time and records both the OCI source digest (`image_digest`) and the local SIF digest as a **dual-digest pair** in the artifact manifest. This ties the scientific result to the exact binary that ran.
- Follows the same validation + promotion saga as the Docker path; the artifact bundle format is identical.

`--image-digest` is optional but strongly recommended. When supplied, the resulting manifest carries both `registry_digest` (the OCI source) and `sif_digest` (what was physically executed), satisfying the M2 dual-digest invariant. Without it, the source identity is recorded as `unverified_local` and `--accept-degraded` is required to proceed.

## Run States

| State | Meaning |
|---|---|
| `PENDING` / `ADMITTED` | The kernel accepted the run and is preparing execution. |
| `RUNNING` | The model container is active. |
| `TRAINING_SUCCEEDED` / `EVALUATING` | The container exited zero and post-run checks are in progress. |
| `PROMOTING` | Validated outputs are being promoted into the artifact store. |
| `ARTIFACT_SUCCESS` | The promoted bundle is the durable scientific result. |
| `FAILED` | Execution failed before a valid artifact was promoted. |
| `CANCELLED` | The user cancelled the run; workspace evidence is preserved. |
| `RECOVERY_PENDING` | The run needs explicit user/operator recovery or adoption. |

## Resuming Completed Work (`skip_completed`)

Launching a manifest runs **every** job in it by default. Planning is a pure
manifest → plan expansion: it never decides that prior work makes a job
unnecessary. This is deliberate — a manifest you explicitly launched should run
what it says, and a stale row in an old table must never make a requested job
silently vanish.

Skipping already-finished work is **opt-in** and **mvd-backed**:

| Surface | How to enable |
|---|---|
| CLI | `multiverse run --manifest <path> --output <dir> --skip-completed` |
| Manifest | `globals.skip_completed: true` |
| GUI | **Run** tab -> *Skip completed jobs (resume)* checkbox; defaults from `globals.skip_completed` when present, otherwise off |

Precedence is: CLI flag / explicit GUI user choice -> `globals.skip_completed`
-> default `false`. In the GUI, an untouched checkbox honors the manifest
global; changing it in the Run tab is an explicit launch override. This keeps
old manifests from skipping jobs unexpectedly while still letting you resume a
large interrupted benchmark.

When enabled, a job is skipped only when its **canonical logical run** already
reached `ARTIFACT_SUCCESS` in the mvd state for the selected `--output`
directory (read from the rebuildable index, falling back to the journal). A
skipped job is never removed from the plan: it is shown — in the CLI preflight
summary and the GUI event log — with the completing attempt id and artifact
directory.

The logical run identity folds in everything that changes the scientific result
or execution recipe: manifest hash, dataset fingerprint, image identity, params
hash, contract version, **seed, preprocessing, and model version**. Editing any
of these produces a new logical run, so it is planned and runnable again even
with `skip_completed` on.

### mvd `ARTIFACT_SUCCESS` vs legacy `SUCCESS`

mvd completion state is `ARTIFACT_SUCCESS`, recorded in the journal and projected
into the rebuildable SQLite index. The legacy `runs` table's `status = 'SUCCESS'`
is **not** authoritative for mvd-backed execution and is never consulted when
planning or resuming a manifest. The legacy table is left intact for old
installations and tooling, but it can neither suppress a manifest job nor mark
one as resumable. Only `ARTIFACT_SUCCESS` with an artifact directory still on
disk counts as completed; `FAILED`, `CANCELLED`, and `RECOVERY_PENDING` never
skip a job.

## Artifact Contract

Every successful run must contain a verified `artifact_manifest.json` and `artifact_manifest.sha256`. The manifest records logical and physical run IDs, dataset fingerprint, image identity, parameter hash, timestamps, owner token, and validated artifact entries with checksums.

SQLite is an index over this state, not the scientific source of truth. If the SQLite file is lost, `multiverse rebuild-index` reconstructs run visibility from the journal and artifact store. For the default tutorial output directory, use:

```bash
uv run multiverse rebuild-index \
  --state-root store/artifacts/run_output \
  --store-root store/artifacts/run_output/store
```

## Required Outputs

| Artifact | Description |
|---|---|
| `artifact_manifest.json` + `.sha256` | Verified bundle metadata and checksum sidecar. |
| `job_spec.json` | Exact runtime instruction passed to the model container. |
| `embeddings.h5` | Required latent matrix at HDF5 dataset `latent`. |
| `metrics.json` | Optional model diagnostics and metric summaries. |
| `umap.png` | Optional visualization. |
| `run.log` | Model SDK log written inside the container by `mvr_worker`. |
| `container.log` | Host-captured container stdout/stderr; present even when the container crashed before writing `run.log`. |
| `orchestrator.log` | Host-side per-run log: admission, launch, exit classification, promotion outcome, and failure reason. |

Logs for a run that fails before promotion remain in its workspace at `<state-root>/store/workspaces/<attempt-id>/`. Set `MVEXP_LOG_LEVEL=DEBUG` to raise verbosity across the host logs and the in-container `run.log`.

The full I/O contract is documented in [Model Container Contract](MODEL_CONTAINER_CONTRACT.md).

## Image Identity and Publication Mode

The two execution backends have different defaults because their typical workflows differ.

### Docker (local development)

Locally-built images — `make build-pca`, `docker build ...` — have no OCI registry digest. This is the normal workflow for researchers. The Docker executor accepts these images by default and records their identity as `unverified_local` in the artifact manifest.

For a publication-quality run where you want the manifest to prove exactly which registry-published image ran, pass `--strict`:

```bash
uv run multiverse run --manifest run_manifest.yaml --output store/artifacts/run_output --strict
```

`--strict` rejects any image that cannot be traced to a registry digest. Use this only when you have pushed and pulled your model images from a registry.

### Slurm + Apptainer

On HPC, the expectation is reversed: you typically pull a known image from a registry and convert it to a SIF, so the Slurm executor requires a verifiable source by default.

To run with a SIF that has no known registry provenance (e.g. a SIF built manually outside the pipeline), pass `--accept-degraded`:

```bash
uv run multiverse slurm-submit ... --accept-degraded
```

## Simple-Mode Runner

The simple-mode runner executes the model artifact contract without the mvd daemon, SQLite registry, MLflow, or Optuna. It is useful for:

- HPC environments where the daemon cannot run persistently.
- Scripted one-off runs outside the GUI workflow.
- Testing a model image against the artifact contract in isolation.

```bash
multiverse run --simple simple_manifest.yaml --out ./bundle-output
```

The simple-mode manifest is a self-contained YAML — it does not consult the asset registry, so every value the runner needs (image, dataset path, n_obs) must appear in the manifest:

```yaml
schema_version: "1"
globals:
  mv_contract_version: "1"
jobs:
  - name: "demo_pca"
    model:
      slug: "pca"
      version: "1.0.0"
      image: "multiverse-pca:1.0.0"
      image_digest: "sha256:..."   # optional; promotes identity kind
    dataset:
      slug: "demo"
      path: "/abs/path/to/processed.h5mu"
      n_obs: 100
    params:
      n_components: 20
```

Key differences from the full manifest:

| Aspect | Full manifest (`multiverse run`) | Simple-mode (`multiverse run --simple`) |
|---|---|---|
| Registry lookup | Required (SQLite) | None — all values in manifest |
| Projection | MLflow + Optuna | None |
| State tracking | mvd journal + SQLite | Per-run JSONL in output dir |
| GPU | `jobs[].gpu: true` | `jobs[].model.gpu: true` |
| Output location | `<output>/store/artifacts/<id>/` | `<out>/<job-name>/` |
| Failed run dir | `<output>/store/workspaces/<id>/` | `<out>/_failed/<job-name>/` |

Additional flags:

| Flag | Meaning |
|---|---|
| `--strict` | Publication mode: refuse images with no registry provenance. |
| `--validators basic\|strict\|developer` | Override validation level (default: `basic`). |
| `--no-image-pull` | Local-only mode; do not attempt to pull the image even if a digest is declared. |
| `--seed <int>` | Seed forwarded to the model backend. |
| `--json` | Emit a machine-readable JSON summary on stdout in addition to the human-readable output. |

## Building SIF Files

`multiverse build-sif` converts a model's Dockerfile or Singularity.def into an Apptainer SIF file and records the output path in the asset registry. Requires `apptainer` (or `singularity`) on `PATH`.

```bash
# From a Dockerfile (default when build.dockerfile is set in model.yaml)
multiverse build-sif --slug pca --output-dir /scratch/sif/

# From a Singularity.def (default when apptainer.def_file is set)
multiverse build-sif --slug my_hpc_model --method def-file --output-dir /scratch/sif/

# Force-overwrite an existing SIF
multiverse build-sif --slug pca --output-dir /scratch/sif/ --force
```

The output file is named `<slug>-<version>.sif` and placed in `<output-dir>` (default: `<state-root>/sif/`). After a successful build, the SIF path is written back into `asset_registry.db` so `multiverse slurm-submit` can find it.

| Flag | Meaning |
|---|---|
| `--slug <slug>` | Model slug (required). |
| `--output-dir <dir>` | Directory to write the `.sif` file (default: `<state-root>/sif/`). |
| `--method docker-daemon\|def-file` | Build method; auto-detected from `model.yaml` if omitted. |
| `--force` | Overwrite an existing `.sif` file. |
| `--manifest <path>` | Override slug-based manifest lookup with an explicit `model.yaml` path. |

## Asset Registry Migration

If you are upgrading from a prior version that stored dataset and model records in the combined `mvexp_state.db`, run the one-time migration:

```bash
uv run multiverse migrate-asset-registry
# or dry-run to see what would be copied:
uv run multiverse migrate-asset-registry --dry-run
```

This copies dataset and model rows from `mvexp_state.db` into the new dedicated `asset_registry.db`. The migration refuses to run twice (idempotent guard). After migration, `mvexp_state.db` continues to hold run/index state; `asset_registry.db` holds the dataset and model catalog.

## Maintenance Command Reference

### `multiverse doctor`

Probes state paths, container engines, storage, workspaces, the reservation ledger, and MLflow projection consistency.

```bash
multiverse doctor
multiverse doctor --json                          # machine-readable output
multiverse doctor --deep-slurm                    # submit a smoke job to verify Slurm end-to-end
multiverse doctor --deep-slurm --slurm-smoke-partition gpu   # specify partition for smoke job
multiverse doctor --repair-health-probes          # sweep expired entries from health-probe namespaces
```

`--deep-slurm` enumerates partitions via `sinfo` and submits a one-second `--wrap=true` smoke job. It briefly allocates a node, so only use it when you want to confirm Slurm is fully operational (not just installed).

### `multiverse rebuild-index`

Reconstructs `mvexp_state.db` from the append-only journal and artifact store.

```bash
multiverse rebuild-index --state-root <path> --store-root <path>
multiverse rebuild-index --state-root <path> --verify   # read-only drift check; exits 0 if in sync
```

`--verify` reports drift between the journal and the current SQLite projection without writing anything. Exit code 0 = in sync, 1 = drift detected.

### `multiverse gc`

Garbage-collects failed workspaces, cancelled runs, and quarantine entries.

```bash
multiverse gc --dry-run                           # show what would be deleted (default)
multiverse gc --apply                             # actually delete
multiverse gc --apply --no-export-required        # skip the EXPORTED marker requirement
multiverse gc --apply --apply-to-promoted         # also consider promoted artifacts (use with care)
multiverse gc --retention-failed 86400            # keep failed workspaces for 24 h (seconds)
multiverse gc --retention-cancelled 3600          # keep cancelled runs for 1 h
```

By default, `gc` requires an `EXPORTED` marker file before deleting promoted artifacts. Pass `--no-export-required` to delete without it. `--apply-to-promoted` includes promoted artifact bundles as candidates — this is a destructive operation; use only after exporting.

### `multiverse migrate-state-dir`

Relocates the state directory (database + store) from a pre-M1 location to the resolver's chosen location. Run this once when upgrading installations that stored state in a non-standard location.

```bash
multiverse migrate-state-dir --from /old/path --to /new/path   # preview
multiverse migrate-state-dir --from /old/path --to /new/path --apply
```

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| Launch fails before container start | Manifest references stale dataset/model rows. | Regenerate the manifest from Configure or re-register the stale object. |
| `executor crashed: unverified_local` | Running Docker path with `--strict` but image has no registry digest. | Remove `--strict` (the default is open for local builds). Slurm path: pass `--accept-degraded` if the SIF has no OCI source. |
| Run reaches `FAILED` | Container non-zero exit, Docker launch failure, or validator refusal. | Open the artifact/workspace logs and inspect `failure_reason`. |
| Run reaches `RECOVERY_PENDING` | Promotion or recovery found data that requires user decision. | Use recovery/quarantine reports before deleting anything. |
| MLflow has no successful entry | Projection sync is pending or failed. | The artifact bundle is still authoritative; run `multiverse mlflow-sync` later. |
| SQLite state looks wrong | Index drift or DB loss. | Run `multiverse rebuild-index` against the state and store roots. |
| `multiverse run` cannot import Docker SDK | The active environment was not synced from project dependencies. | Run `uv sync --group dev`, or install the package with its declared dependencies. |
| `migrate-asset-registry` says "already migrated" | Migration ran once already. | Nothing to do; `asset_registry.db` is up to date. |
