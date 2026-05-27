# Production Readiness Audit — mvexp

**Audit posture:** Staff SRE + Principal Systems Architect, scoped to the platform's actual deployment model: **single user, single workstation, local install.**
**Scope:** State machine, I/O lineage, concurrency, error surface, single-host portability.
**Audit date:** 2026-05-27 (branch `jelle-readiness`).

This document is a critique. It does not prescribe code fixes. Every claim is anchored to a `file:line` citation so engineers can confirm before remediating. Multi-user contention, network-exposed-service authentication, cluster/cloud portability, and rootless-container hardening have been intentionally de-scoped per the platform's single-user-local deployment model. Failure modes that survive that de-scoping — and there are several — are documented here.

---

## 1. Critical Failure Modes — The Blockers

Each of these fires on a single workstation. None of them require a multi-user environment to materialize.

### CFM-1. Promotion is not atomic and the recovery code can permanently destroy a successful run

The promotion sequence at `multiverse/runner/docker_runner.py:760-789` is:

1. Write `runs.status = PROMOTING` (queued async DB write).
2. `shutil.move(workspace_dir, final_artifact_dir)`.
3. Write `.promotion_complete` marker into `final_artifact_dir`.
4. Write `runs.status = SUCCESS`.

A SIGKILL, OOM kill, host reboot, or laptop-lid-close between steps 2 and 3 leaves a moved workspace with **no marker**. On the next startup of the platform, `recover_orphaned_runs()` at `multiverse/registry_db.py:411-430` does `shutil.rmtree(output_path, ignore_errors=True)` on every `PROMOTING` row that lacks the marker. The successfully promoted artifact tree — including `embeddings.h5` and `metrics.json` representing real GPU-hours of compute — is silently deleted, and the run is flipped to `FAILED` with reason `INCOMPLETE_PROMOTION`.

This is the single highest-severity issue in the system. The recovery code, intended to clean up half-done work, will instead destroy fully-done work whenever a crash or reboot lands in the marker-write window. A user who launches an overnight benchmark, sees a crash report at 3am, and reboots their machine before checking on it can lose the night's work the moment the GUI starts again. Reproduction probability scales with run count; for a researcher running dozens of benchmarks a week, this *will* happen.

### CFM-2. `shutil.move` is not atomic across filesystems and the code does not detect EXDEV

The single call at `multiverse/runner/docker_runner.py:768` is `shutil.move(workspace_dir, final_artifact_dir)`. When `store/workspaces/` and `store/artifacts/` resolve to different mount points — a common single-user setup: NVMe scratch + a slower archive disk, or a workspace on `/tmp` (tmpfs) — `shutil.move` falls back to copy-then-delete. This is:

- **Not atomic.** No reader can observe an "all-or-nothing" transition.
- **Catastrophic under ENOSPC.** A disk-full event mid-copy raises `OSError`. The catch at `docker_runner.py:773` is a bare `except Exception`; it marks the run `FAILED` and preserves the workspace, but the partially-copied destination is left in place. The next promotion that targets the same path will fail differently (destination already exists), masking the root cause.
- **Silent on permission failures.** A copy-then-delete that succeeds at copy but fails at delete leaves duplicate state on disk with no error reported.

There is no `EXDEV` check, no `os.fsync` of the parent directory, and no rollback `finally` block. On a researcher's laptop with `/tmp` on tmpfs and `~` on an NVMe, this fires every time the workspace and artifact dirs straddle the mount.

### CFM-3. The GUI does not track or kill its runner subprocess

`gui.py:165-178` calls `subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True)` inside `st.status()` and then `proc.wait()`. The PID is never stored, never offered as a cancel target, and never reaped on session termination.

Single-user behaviors that orphan the runner:

- Closing the browser tab.
- Streamlit's own auto-rerun during widget interaction.
- Browser refresh during a long run.
- Ctrl-C in the terminal that started Streamlit (kills Streamlit, not necessarily the runner subprocess and definitely not the model containers it spawned).
- Suspend/resume on a laptop.

Each of these orphans the runner process — and the entire fan-out of model containers it owns. The user sees no UI feedback path, and the only way to recover is to manually `ps`, `docker ps`, and `kill` / `docker kill` from the terminal. There is no "Cancel" affordance anywhere in the GUI. In a single-user setting, this is the most common cause of "my machine is hot and the fans are spinning but I don't know why" complaints.

### CFM-4. `evaluate_single_run` failures are swallowed; jobs are reported SUCCESS without metrics

`docker_runner.py:799-809` runs evaluation *after* the `SUCCESS` state has been committed to the DB. The evaluation call is wrapped in a bare `except` that logs a warning and continues. A job that completes the model container but fails evaluation — for any reason: malformed `metrics.json`, NaN-filled embedding, scib-metrics version skew, missing batch column at evaluation time — appears in the GUI and MLflow as a successful run with empty metrics.

This is worse than a hard failure. A `FAILED` row prompts investigation; an empty-metrics `SUCCESS` row is interpreted by downstream analysis (and by the researcher reading the Results tab) as "the model produced no signal." False scientific conclusions follow. For a single-user platform whose entire value proposition is a defensible Methods section, this is a direct hit to the platform's purpose.

### CFM-5. Pre-flight is shallow; the real validators are the model containers

`validate_pending_jobs()` at `cli.py:276-386` checks: omics subset, file-readability via HDF5 header peek, batch-key column existence, cell-type-key column existence (warning only — does not skip). It does **not** check:

- whether the batch column has more than one unique value (warns but does not skip);
- whether `obs` columns contain NaN, mixed types, or are entirely empty strings;
- whether `n_vars > 0` per modality;
- whether the requested model's hyperparameter ranges are coherent given dataset size (e.g. `n_components > n_cells`);
- whether the `cell_type` column has the expected granularity.

Every uncaught failure mode pays the full cost: image pull, container launch, dataset load, ML library initialization, and eventual stack trace inside the container. On GPU images this is minutes-to-hours of wall time per failed job, and the failure surfaces only as a non-zero exit code and a buried traceback in `container.log`. For a single user, the cost is not "system unavailable" — it is "your evening is gone."

### CFM-6. No OOM accounting; the resource pool is a static reservation, not a live observation

The `ResourcePool` at `docker_runner.py:323-378` is a committed-memory ledger sized off `psutil.virtual_memory().total` with a per-job default of 16 GiB. It admits jobs based on a static reservation. Once admitted:

- The container's actual RSS is not observed.
- The host's free memory is not re-polled between admissions.
- Page cache, browser tabs, IDE, other ML processes, and `mlflow.db` growth all consume RAM the pool believes is free.

On a workstation that is *also* a workstation — i.e. the user is running VS Code, Slack, a browser with 80 tabs, and the platform at the same time — the static reservation will admit more containers than the host can sustain. When a container exceeds its cgroup `mem_limit`, the kernel OOM-killer reaps it and Docker reports a non-zero exit code; the orchestrator marks the run `FAILED` with no signal that the cause was memory pressure. The user sees `FAILED` rows with `container.log` ending mid-tensor-allocation and has to guess.

### CFM-7. Intra-process SQLite write paths are inconsistent

Within the single-user setup, three writers still coexist *on the same machine* and routinely run concurrently:

- The Streamlit GUI process performing registry edits.
- The runner subprocess the GUI spawned, performing run-state updates.
- Ad-hoc CLI commands (`make register-model`, `make register slug=...`) that the user runs in a second terminal.

`get_db_connection()` at `registry_db.py:18-30` opens a fresh connection per call with `check_same_thread=False`. WAL mode permits concurrent readers but still serializes writers; the platform's `busy_timeout=10000` masks contention until a transaction holds the lock past that window. Within the orchestrator process, async writes are funneled through a single-writer queue (`docker_runner.py:174-213`) — a good pattern that does not extend to the other two writer classes.

The realistic single-user failure mode is: user runs `make register slug=foo` while a benchmark is mid-promotion, and one of the two operations hangs for ten seconds and then surfaces `database is locked` as a Python traceback in the terminal. The data does not corrupt — SQLite's transactions hold — but the UX is opaque and intermittent.

---

## 2. The UX Gap Audit

UX gaps that compound into "this researcher will give up before debugging." Single-user use does not soften any of these; if anything it sharpens them, because there is no one else to ask.

| Gap | Where | Effect |
|---|---|---|
| No cancel button for a running benchmark | `gui.py:165-178` | Users close the browser and assume the run died. It didn't. Fans keep spinning. |
| `PROMOTING` is invisible in the GUI | GUI Results tab | Runs that crash mid-promotion appear stuck in an undocumented state. No way to know whether to wait or rerun. |
| No live progress per job | GUI Run tab | A 40-minute MOFA run looks indistinguishable from a hung container. EpochLogger writes to MLflow, but the GUI does not poll it. |
| Pre-flight `SKIPPED` reasons are technical | `validate_pending_jobs()` log messages | "omics subset mismatch" instead of "your dataset has only RNA; MultiVI needs RNA+ATAC." |
| Empty-metrics `SUCCESS` runs (CFM-4) | Results tab | Green checkmark + empty table. Indistinguishable from a working run with no metrics defined. |
| `database is locked` surfaces as a stack trace | Multiple sites | Researcher sees a Python traceback in the terminal that started the GUI; no in-app indication of what happened. |
| No retry affordance for transient failures | None | Image pull blips, Docker daemon hiccups, MLflow timeouts all require manually re-launching the entire run. |
| Manifest editing is one-shot | Configure tab | Switching back to Configure after a run regenerates the manifest from current selections; the manifest that was actually run is not the same object as the one displayed. |
| No "this is the same as before" affordance | Configure tab | Two runs with identical manifests get distinct run IDs and look like independent experiments. No first-class concept of replay. |
| `STALE` status is documented but not actionable | Registry tab | Users see `STALE`, don't know what triggered it, and either ignore it or re-register defensively. |
| Container logs are only available after a run completes | GUI Run tab | While a job is `RUNNING`, `container.log` is in the workspace and not surfaced. The user is staring at a status badge with no detail. |
| No surfaced disk-usage warnings | Anywhere | `store/artifacts/` grows linearly. Nothing warns the user that the disk is at 90% capacity until promotion fails — and CFM-2 turns that failure into corruption. |
| MLflow + Optuna iframes are blank when services are down | Analysis tab | Silent. Users assume the feature is broken rather than the service. |
| Run cost (wall time, peak RSS, GPU minutes) not surfaced | Results tab | Users cannot compare models on cost as well as accuracy. |
| Evaluation errors produce blank metric rows | Results + MLflow | See CFM-4. |
| Launching `make services-up` and `make setup` is a two-terminal dance | Makefile | First-time users routinely run only one of them and wonder why the Analysis tab is blank or why nothing launches. |

---

## 3. Local-Install Dead-Ends

Single-user, single-workstation deployment removes most of the multi-host / cloud / multi-tenant concerns. The assumptions below still bite even at that scope.

| Assumption | Location | Breaks when |
|---|---|---|
| `store/` resolves relative to `BASE_DIR` derived from `os.path.dirname(__file__)` | `registry_db.py:8-11` | The repo is moved, the package is reinstalled into a venv, or the user runs the CLI from a different working directory than they ran the GUI. There is no env var or config knob to relocate state. |
| Docker daemon is accessed via `docker.from_env()` | `docker_runner.py:49`, `cli.py:709`, `builder.py:73` | A user on a workstation without Docker (or with Podman, or with Docker Desktop in a non-default state) gets opaque errors. |
| `host.docker.internal` is how containers reach host services | `docker_runner.py:554, 1049, 1165` | Linux hosts without explicit `--add-host` for the gateway. The code does inject this, but the assumption is fragile across Docker versions and rootless setups. |
| MLflow URI defaults to `http://localhost:5000` | `tracking.py:210`, `docker_runner.py:552` fallback | Users who already have MLflow running on 5000 for another project — common in single-user research setups — get silent collisions. |
| Optuna Dashboard defaults to 8080 | `gui.py:233` | Same collision risk; 8080 is a popular default. |
| Streamlit defaults to 8501 | `Makefile:20` | Same. |
| All Dockerfiles set `USER root` | `store/models/*/container/Dockerfile` | Container output files in `store/artifacts/` are owned by root on the host. Subsequent host-side reads from Jupyter need either elevation or a `chown`. This is the *single-user* impact of running as root — host file ownership, not privilege escalation. |
| UID/GID of the host user is not propagated to containers | All Dockerfiles | Same root cause as above. |
| `psutil.virtual_memory().total` is the resource ceiling | `docker_runner.py:1115` | The platform thinks it has the whole machine. On a workstation it does not — the user's browser and IDE are real consumers. |
| GUI is launched as a foreground Streamlit process under `make setup` | `Makefile:16-20` | A user who closes the terminal kills the GUI. No background-launch mode is documented. |
| MLflow / Optuna services live in Docker Compose, but their backing SQLite is on `./store/` mounted from the host | `docker-compose.yml` | If the user moves `store/` to follow data to a new disk, the services need restarting; this is not surfaced anywhere. |
| Logging is to stderr and per-container files | Throughout | Users debugging a failed run have to know which file in which subdirectory to read. No unified `mvexp logs <run-id>` command. |
| First-run discovery: user must run `make bootstrap`, then `make services-up`, then `make setup` | `Makefile` | Three steps in the right order; missing the middle one produces a working-looking GUI with a blank Analysis tab. |

---

## 4. Hardening Roadmap (Prioritized)

P0 — must address before the platform stops costing researchers their evenings. P1 — required before the platform feels reliable rather than usable. P2 — durability and ergonomics polish.

### P0 — Data integrity and the off switch

1. **Make promotion crash-safe.** Replace the `move → marker → status` sequence with a single atomic state transition. The marker file is a workaround for not having atomicity; it is itself a source of the data loss it was added to prevent (CFM-1). The recovery code must never `rmtree` an artifact directory it did not create in the current process lifetime.
2. **Detect cross-filesystem promotion (EXDEV) and refuse to start.** A platform that silently degrades from `rename(2)` to copy-then-delete cannot be made reliable post-hoc. Either require a single filesystem for `store/`, or implement copy-then-fsync-then-swap with explicit rollback. (CFM-2)
3. **PID tracking and a real cancel button.** Every runner subprocess must be tracked (in a per-experiment file or in the DB) with PID and start time. The GUI must surface a Cancel button that sends SIGTERM, waits, then SIGKILL, and propagates `docker kill` to the fan-out. (CFM-3)
4. **Evaluation failures must produce `FAILED`, not `SUCCESS`.** A job that completes the container but fails evaluation must not be presented as a successful run. The current behavior produces false scientific conclusions on a platform whose entire value proposition is reproducible benchmarking. (CFM-4)

### P1 — Operational durability

5. **Schema-validate the run manifest at the parsing boundary.** Use the existing Pydantic models to reject malformed manifests before any job pre-flight runs.
6. **Deepen pre-flight.** Open each dataset, check `n_cells > 0` per modality, `n_vars > 0`, `batch` column has ≥ 2 unique non-null values, `cell_type` (when requested) has ≥ 2 non-null values, and per-model preconditions (count-likeness for VAEs, modality presence). Cache the result in the registry keyed by manifest hash so repeated runs do not re-validate. (CFM-5)
7. **OOM observability.** Subscribe to Docker events for `OOMKilled` exit reasons and surface them as a distinct failure category, not as generic `FAILED`. Re-poll host memory between admissions; do not trust the static reservation under workstation contention. (CFM-6)
8. **Job retries with idempotent state.** Transient failures (image pull timeout, MLflow 503, Docker daemon hiccup) should auto-retry with exponential backoff up to a bounded count. Today the user re-launches manually.
9. **Configurable storage root.** A single env var (`MVEXP_STORE_ROOT`) that overrides `BASE_DIR/store/`. One-line change; outsized portability value when a user wants `store/` on a faster or larger disk than the repo.
10. **Disk-pressure circuit breaker.** Refuse admission to new jobs when `store/` is above 90% capacity. Surface the threshold in the GUI before the user launches a run that cannot promote — closing the door on the CFM-2 failure mode.
11. **Cancel sub-states in the DB.** A `CANCELLED` terminal state distinct from `FAILED`, populated by the cancel path from P0/#3.
12. **Drop root in containers for output ownership reasons.** Even without a multi-user threat model, root-owned files in `store/artifacts/` are an ergonomic regression for the researcher's Jupyter workflow. Add a non-root `USER` and pass the host UID at run time.

### P2 — Durability, ergonomics, and operator quality of life

13. **Per-run cost accounting.** Wall time, peak RSS, GPU seconds. Logged to MLflow tags. Surfaced in the Results table. Researchers benchmark by accuracy *and* by cost.
14. **Live progress in the GUI.** Poll the run's MLflow metric stream or the `metrics.jsonl` sidecar; render per-job progress without waiting for completion.
15. **In-app log viewer.** A "Logs" tab per run that surfaces `container.log` and `model.log` while the run is in flight, not only after it completes.
16. **Unified `mvexp logs <run-id>` CLI command.** A single entry point for log retrieval that does not require the user to know the artifact path.
17. **Replay-by-manifest.** A first-class "rerun this manifest" action that surfaces a diff against the prior run's metrics, instead of treating identical recipes as independent experiments.
18. **Operator runbook.** A documented procedure for what to do when promotion fails, when `database is locked` persists, when an OOM-storm hits, when MLflow goes down mid-run. Today this knowledge is in the code.
19. **Single-command bootstrap.** Collapse `make bootstrap && make services-up && make setup` into a single `make start` that does the right thing on a fresh clone and on subsequent runs. The current three-step dance is the most common failure point for new users.
20. **Port-collision detection.** On startup, check whether 5000, 8080, 8501 are free and degrade gracefully (auto-pick + report) rather than failing opaquely.

---

## 5. Subsystems Judged Structurally Unreliable

Per the audit constraint, the following components are called out as "too complex / wrong-shape to be reliable" — even in a single-user, local-only deployment.

- **The promotion path.** The marker-file workaround compounds the original atomicity problem. The recovery code's reflex to `rmtree` on missing-marker means the system's own fault-recovery logic is its largest single source of data loss. This needs a design redo, not a patch.
- **The evaluation-as-side-effect coupling.** Evaluation runs *after* `SUCCESS` is recorded and its failures are logged-and-ignored. Either evaluation is part of the run (and failures matter) or it is not (and `SUCCESS` should not depend on it). The current half-way design produces the worst of both: false `SUCCESS` and silently absent metrics. For a benchmarking platform, this is a structural defect.
- **The GUI → subprocess boundary.** A Streamlit `Popen` without PID tracking is not a control plane; it is a launcher with no off switch. The whole boundary should be replaced with a job-server actor that the GUI calls into and that survives a Streamlit rerun. In a single-user setting this matters more, not less — there is no operator to clean up after a stranded process.

---