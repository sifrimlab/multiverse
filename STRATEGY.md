# multiverse Hardening Strategy

**Inputs:** [`PRODUCTION_READINESS_AUDIT.md`](PRODUCTION_READINESS_AUDIT.md), [`CRITIQUE_OF_AUDIT.md`](CRITIQUE_OF_AUDIT.md).
**Binding ADR:** [`docs/adr/0001-mvd-platform-assumptions.md`](docs/adr/0001-mvd-platform-assumptions.md).
**Posture:** This document converts the audit's symptom list and the critique's structural reframing into a single, sequenced design strategy. It starts at the architectural level and now includes binding implementation specifications where ambiguity would create new failure modes.
**Constraint:** Single-user, single-workstation deployment model is preserved. The strategies below do not introduce multi-tenant infrastructure; they introduce *durability* and *idempotency*, which a single user needs as much as any operator.

---

## North Star

The audit named the right symptoms. The critique named the right cause: there is no authoritative owner of job intent, leases, cancellation, recovery, idempotency, or reconciliation. Streamlit, a subprocess, an in-process SQLite writer actor, Docker containers, the filesystem, evaluation, and MLflow are all loose-coupled peers, and partial failure is the dominant failure mode of any such system.

The strategy is to introduce **one durable local owner** — a small local job daemon, hereafter called **mvd** — and reshape every fragile boundary into a contract with that daemon. Most of the strategies below are either *moves into the daemon*, *contracts the daemon enforces*, or *invariants the daemon makes recoverable*.

This is not "add more guardrails." It is *fewer, better-owned* guardrails.

---

## Strategy Map

| # | Strategy | Addresses (audit / critique) |
|---|---|---|
| S1 | The mvd local job daemon | CFM-3, CFM-7, critique rows on actor scope, GUI/runner boundary, final verdict |
| S2 | Artifact store as source of truth; SQLite as index | Critique row on SQLite-as-truth; rebuild-index |
| S3 | Promotion saga with ownership tokens | CFM-1, CFM-2; critique rows on promotion, EXDEV, ownership |
| S4 | Quarantine-only recovery | CFM-1; critique row on never-delete-unknown-data |
| S5 | Richer state machine with sub-terminal states | CFM-4; critique row on partial successes |
| S6 | Tiered, versioned validation | CFM-5; critique rows on pre-flight and over-validation |
| S7 | Docker supervisor with labels and leases | Critique row on Docker daemon reconciliation |
| S8 | Cancellation as a saga | CFM-3; critique row on cancel saga |
| S9 | Resource broker (RAM, VRAM, disk, inodes, FDs) | CFM-6; critique row on broader resource accounting |
| S10 | First-class GC with ownership tokens | Critique row on GC |
| S11 | Storage capability probes + storage backend abstraction | Critique rows on storage config validation and read-only mounts |
| S12 | Idempotent retries with deterministic keys | Audit P1/#8; critique row on retry-without-idempotency |
| S13 | MLflow as declared cache | Critique row on MLflow cache-vs-contract |
| S14 | Clock and timestamp discipline | Critique row on clock skew |
| S15 | Output semantic validation before promotion | Critique row on file-existence ≠ validation |
| S16 | Content-addressed run identity and image digest pinning | Critique rows on run identity and image tags |
| S17 | `multiverse doctor` and `multiverse rebuild-index` | Critique rows on DB corruption and in-GUI diagnosis |
| S18 | Portable export/import bundles | Critique row on backup/export |
| S19 | Defensive surface against untrusted model registrations | Critique row on local-Docker security |
| S20 | Minimal reliable core / simple-mode | Critique row on complexity bloat |

The remainder of this document describes each strategy in enough detail to begin design work without prejudging implementation language, libraries, or APIs.

---

## Red-Team Critique of This Strategy

The strategy is directionally right, but it still has several failure modes of its own. The most important one: it risks replacing today's loose coupling with a large daemon that becomes a new single point of failure, a new installation burden, and a new source of configuration drift. For a single-user local tool, that tradeoff is only acceptable if the daemon is kept small and built on stable artifact, label, and recovery contracts from the start.

| Strategy Assumption | Failure Mode Introduced | Required Improvement |
|---|---|---|
| `mvd` becomes the sole owner of DB, Docker, promotion, GC, MLflow, and Optuna. | The daemon becomes a "god process." If it cannot start, the entire platform becomes unusable. Debugging also becomes harder because all failures collapse into "mvd unhealthy." | Define a minimal daemon kernel: job journal, leases, Docker supervision, state transitions, and promotion. Keep MLflow sync, GC, export, and doctor as restartable plugins/commands that can run outside the hot path. |
| "GUI and CLI do not touch DB/filesystem/Docker directly." | This is architecturally clean but creates a large blast radius if implemented before the daemon contract is narrow and testable. Every command becomes unavailable when `mvd` is down. | Define a strict API boundary first: submit job, cancel job, stream logs, query state, run doctor. Do not put export, MLflow sync, GC, or model registration behind the daemon until the kernel is stable. |
| SQLite is rebuildable from artifacts. | In-flight runs are not reconstructible from promoted artifacts because they have no final manifest yet. A DB loss during `RUNNING` still strands containers and workspaces. | Add a separate append-only `store/journal/` for in-flight intent records. Rebuild must combine journal, Docker labels, workspaces, and artifact manifests. Artifact manifests alone are insufficient. |
| `artifact_manifest.json` carries complete truth. | A single mutable JSON file can be edited, partially written, or inconsistent with the files it describes. If it is the truth, corruption of that file is catastrophic. | Write manifests atomically as `artifact_manifest.json.tmp` then rename. Include a detached checksum sidecar such as `artifact_manifest.sha256`. Keep a compact transition log in `store/journal/` so the manifest is reproducible. |
| Promotion saga removes half-built artifact dirs if owner token matches. | This contradicts quarantine-only recovery. A bug in token assignment could still delete valuable data. | Prefer quarantine over deletion even for owned failed promotion directories, unless running explicit GC. The hot path should not delete result-like data. |
| Run success and tracking sync are conflated. | If MLflow is cache, a run with a valid artifact bundle should not be blocked from scientific success by dashboard sync. | Keep primary run state focused on artifact correctness: `ARTIFACT_SUCCESS` for validated/promoted bundles, with `TRACKING_SYNCED` or `TRACKING_SYNC_FAILED` as projection status. The GUI can show "Success, MLflow out of sync" without making MLflow part of scientific correctness. |
| Simple-mode does not require `mvd`, SQLite, MLflow, Optuna, or GUI. | Good idea, but it is sequenced last. That makes the anti-complexity escape hatch arrive after the complex daemon architecture. | Move simple-mode to the first implementation phase. It should define the artifact contract, validator contract, export bundle shape, and promotion semantics before the daemon consumes them. |
| Storage probes reject Dropbox/OneDrive/NFS heuristically. | Overzealous rejection will frustrate rookies and create platform-specific false positives. Some users have no choice but institutional synced folders. | Use graded storage health: `supported`, `degraded`, `dangerous`, `blocked`. Allow degraded mode with explicit warnings and disabled crash-safety claims. Block only missing required semantics like no write, no rename, or no Docker visibility. |
| `doctor` creates temp MLflow/Optuna records and deletes them. | Health checks can mutate user-visible dashboards or fail cleanup, creating noise. | Use dedicated hidden health-check experiment/study names and TTL cleanup. The doctor report must distinguish "probe failed" from "cleanup failed." |
| Content-addressed identity includes image digest. | Digest availability is inconsistent for locally built images. Strict refusal can block development and offline use. | Represent image identity as one of: registry digest, local image ID, build context hash, or explicit `unverified-local`. Strict publication mode requires digest or build context hash; development mode can proceed with warnings. |
| Resource broker observes host resources at admission. | Local workstation resources change after admission: browser tabs, IDEs, Docker Desktop, and OS updates can steal RAM/VRAM during a run. | Add continuous monitoring and soft preemption policy. For local mode, do not kill jobs automatically; surface "resource pressure" and stop admitting new jobs. Record peak RSS/VRAM and OOM evidence in the artifact manifest. |
| GC can run as a daemon subsystem. | Background deletion is scary in a research tool. Even dry-run-by-default can become dangerous if config flips later. | Make GC manually triggered by default. Automatic GC may only clean clearly temporary daemon-owned scratch files, never promoted artifacts, quarantine, cancelled workspaces, or failed workspaces. |
| Symlink canonicalization during promotion is enough. | Race conditions remain: a symlink can change between validation and use. | Use descriptor-based traversal where possible, reject symlinks in managed store paths by policy, and re-check canonical paths immediately before destructive or promotion operations. |
| State machine adds many terminal and sub-terminal states. | The GUI may become harder to understand than the current green/red model. Users need action-oriented states, not internal FSM vocabulary. | Separate internal states from user-facing status. Internally keep precise states; in GUI show concise groups: Running, Needs attention, Succeeded, Cancelled, Failed, Recoverable. |
| The strategy is "silent on implementation detail." | Several recommendations depend on OS-specific behavior: `fsync` directories, Unix sockets, systemd user units, Docker Desktop differences, NVML availability. | Add an implementation decision record before coding: supported OS matrix, filesystem assumptions, daemon startup model, socket/API auth, and Docker Desktop behavior. Without that, the daemon will drift per platform. |

**Revised priority.** Because this is active development with no legacy users to migrate, the plan can be more aggressive. Still, it should not start by building a broad daemon. It should start with the portable artifact contract and simple-mode runner, because those are the stable core that every later daemon feature depends on.

1. **Artifact contract first:** `artifact_manifest.json`, atomic manifest write, checksums, output semantic validation, logical/physical run IDs, and export/import bundle shape.
2. **Core execution hardening:** Docker labels, ownership tokens, quarantine-only recovery, storage probes, and promotion saga in the core execution path before daemon feature expansion.
3. **Simple-mode:** one command that produces the same artifact bundle without GUI, MLflow, Optuna, or daemon.
4. **Daemon kernel:** only job journal, leases, Docker supervision, state transitions, cancellation, and promotion ownership.
5. **Projections and maintenance:** MLflow sync, doctor, rebuild-index, GC, resource broker, and GUI diagnostics.

This order reduces data-loss risk while keeping `mvd` small enough to test exhaustively. Since there is no installed user base to preserve, old direct-control paths can be deleted once the new core path exists; they do not need a long coexistence window.

---

## S1 — The `mvd` Local Job Daemon

**Problem.** The Streamlit `Popen`+`wait` pattern is not a control plane (CFM-3). The in-process async DB writer is process-local and dies with its host process (critique row on actor scope). Browser lifecycle and job lifecycle are different timescales, and the current code couples them.

**Strategy.** A single long-lived local daemon — `mvd` — becomes the sole owner of:

- the `runs` state machine;
- the Docker control plane (container creation, supervision, kill, reconciliation by label);
- the workspace → artifact promotion saga (S3);
- the write-ahead intent journal (S2);
- the resource broker (S9);
- the GC scheduler (S10);
- MLflow and Optuna interactions (S13).

The daemon exposes a small local API (Unix socket; HTTP only over loopback). The Streamlit GUI and the `multiverse` CLI become **clients** of `mvd`. They submit jobs, poll state, request cancellation, and stream logs — they do not touch the DB, the filesystem, or the Docker daemon directly.

The daemon launches under the user's session (systemd user unit on Linux, `launchd` on macOS). It restarts on crash, replays its journal at startup, reconciles Docker state by label query, and only then accepts new requests.

**Acceptance criteria.**

- Closing the browser, refreshing the tab, suspending the laptop, or `Ctrl-C`-ing Streamlit leaves jobs running unaffected.
- A user can `multiverse jobs cancel <id>` from any terminal and the cancellation saga executes.
- `mvd` crash + restart loses no committed state; uncommitted intents replay from the journal.
- The GUI has no DB connection and no Docker client. Grep confirms.

**Dependencies.** S2 (journal), S3 (saga), S7 (Docker supervision), S8 (cancel saga). These are designed assuming the daemon exists.

---

## S2 — Artifact Store as Source of Truth; SQLite as Index

**Problem.** Today `mvexp_state.db` is the only authoritative record. Corruption, deletion, antivirus locking, network-FS flakiness, or schema-migration interruption renders past runs invisible. The artifact tree carries enough data to be the source of truth, but the code does not treat it that way.

**Strategy.** Every promoted artifact directory carries a self-describing `artifact_manifest.json`:

```text
{
  "schema_version": "1",
  "logical_run_id": "<content hash>",         # see S16
  "physical_attempt_id": "<uuid>",
  "manifest_hash": "<sha256 of run_manifest.yaml>",
  "dataset_fingerprint": {...},               # slug, file hashes, obs/var shape
  "model_image_digest": "<sha256:...>",       # see S16
  "params_hash": "<sha256 of resolved hyperparameters>",
  "produced_at": {"wall": "...", "monotonic_ns": ..., "tz": "..."},
  "produced_by": {"mvd_version": "...", "git_commit": "..."},
  "artifacts": [
    {"name": "embeddings.h5", "sha256": "...", "size": 12345, "produced": true, "validated": true},
    ...
  ],
  "state_transitions": [{"from": "RUNNING", "to": "TRAINING_SUCCEEDED", "at": "...", "reason": "..."}],
  "owner_token": "<token written by mvd at promote-prepare>"
}
```

SQLite remains the *index* — fast queries for the GUI, foreign-key relationships, lockable rows for in-flight state — but is **rebuildable** from the artifact store. A `multiverse rebuild-index` command (S17) walks `store/artifacts/`, parses every `artifact_manifest.json`, and reconstructs `datasets`, `models`, and `runs` rows with a conflict report.

The DB's `runs.status` for a row that no longer matches its artifact manifest is treated as stale, not authoritative. Reconciliation prefers the manifest.

**Acceptance criteria.**

- Deleting `mvexp_state.db` is non-destructive: `multiverse rebuild-index` restores the registry from disk.
- Every promoted artifact directory contains `artifact_manifest.json`. No exceptions.
- The GUI reads metrics from `artifact_manifest.json` artifact entries (via checksum-verified loads), not from a DB column.

---

## S3 — Promotion as a Saga, Not a `shutil.move`

**Problem.** CFM-1 and CFM-2: the current `move → marker → status` sequence is non-atomic across all three boundaries (DB, FS, marker), and EXDEV silently degrades to copy-then-delete with no checks.

**Strategy.** Promotion becomes an explicit ordered saga with idempotent steps, each writing its intent to the journal (S2) before performing the side effect. Each step is replayable; no step destroys a destination it does not own (S4):

1. **PREPARE.** mvd writes a journal entry naming `workspace_dir`, `final_artifact_dir`, and a unique `owner_token`. The artifact dir is created with the owner token written into its `.mvd_owner` file.
2. **VALIDATE.** Output semantic checks run against the workspace (S15). On failure, transition to `EVALUATION_FAILED` or `PROMOTION_FAILED` (S5), quarantine the owned failed-promotion directory with a report, and stop. The hot path does not delete result-like data.
3. **STAGE.** Detect filesystem boundary explicitly. Same-FS: `os.rename` direct to the final path (atomic). Cross-FS: copy into a sibling temp directory, `fsync` files and the directory inode, then atomic `rename` of the temp directory into place. Compute and verify checksums during the copy.
4. **COMMIT MANIFEST.** Write the final `artifact_manifest.json` (S2), `fsync` it.
5. **COMMIT INDEX.** Update the SQLite `runs` row to a terminal state in a single transaction whose `WHERE` clause includes the prior expected state (optimistic concurrency).
6. **COMMIT TRACKING PROJECTION.** Push to MLflow (S13). Failure here updates projection status to `TRACKING_SYNC_FAILED` but does not change the primary artifact-success state.

The owner token survives. On crash mid-saga, replay reads the journal, sees the last committed step, and resumes from the next step. The owner token is what makes "is this artifact dir mine to delete on failure?" answerable.

**Acceptance criteria.**

- A SIGKILL between any two steps is survivable: replay completes the saga or marks the explicit failure state.
- Cross-filesystem promotion either runs the staged-copy variant or refuses with a clear message. Never a silent fallback.
- `os.fsync` on the artifact directory inode is verified.
- No code path deletes a directory without first reading and matching `.mvd_owner`.

---

## S4 — Quarantine-Only Recovery

**Problem.** CFM-1's root cause is that recovery code deletes data it does not understand. `recover_orphaned_runs()` currently runs `shutil.rmtree(output_path, ignore_errors=True)` on any `PROMOTING` row missing a marker.

**Strategy.** Recovery is the most conservative subsystem in mvd. Its only mutation power is:

- transitioning DB rows to `RECOVERY_PENDING` or to an explicit failure state;
- moving suspicious directories from `store/artifacts/` and `store/workspaces/` to `store/quarantine/<date>/` with a recovery report alongside;
- writing a `RECOVERY_REPORT.md` per quarantined directory listing what was found, what was inferred, and what action the user can take.

Deletion *never* happens during recovery. Deletion happens via GC (S10), under an explicit retention policy, against directories with verifiable ownership tokens and checksum reports.

The GUI surfaces quarantine as a first-class concept: a "Recovered Runs" tab where the user can **adopt** (move back to `artifacts/` and reconcile DB), **delete** (with confirmation), or **export** the recovered bundle.

**Acceptance criteria.**

- The phrase `rmtree` does not appear in any recovery code path. (Grep gate in CI.)
- Every quarantined directory has a sibling `RECOVERY_REPORT.md` describing why it was quarantined.
- The user can recover from any past audit-window failure without losing data.

---

## S5 — Richer State Machine with Sub-Terminal States

**Problem.** CFM-4 collapses two different facts — training succeeded, evaluation failed — into one `SUCCESS` (false positive) or one `FAILED` (loses partial value, triggers wasteful reruns).

**Strategy.** The `runs` table grows a more honest terminal vocabulary:

| State | Meaning |
|---|---|
| `PENDING` | In the plan; not yet admitted by the resource broker. |
| `ADMITTED` | Resources reserved, image being pulled. |
| `RUNNING` | Container active. |
| `TRAINING_SUCCEEDED` | Container exited 0; embeddings present and *semantically* valid (S15). |
| `EVALUATING` | Evaluation is running against a training-successful workspace. |
| `EVALUATION_FAILED` | Training succeeded but `evaluate_single_run` raised. Embedding is preserved. |
| `PROMOTING` | Saga in flight. |
| `PROMOTION_FAILED` | Saga aborted; workspace preserved, artifact dir quarantined. |
| `ARTIFACT_SUCCESS` | Validated artifact bundle promoted. This is the scientific success state. |
| `CANCEL_REQUESTED` | User requested cancel; saga in flight. |
| `CANCELLED` | Cancel saga completed; workspace preserved per policy. |
| `FAILED` | Container exited non-zero or pre-flight rejected. |

Importantly: `TRAINING_SUCCEEDED` and `EVALUATION_FAILED` are **sub-terminal**. The GUI shows them as "Trained, evaluation failed — retry evaluation" with an explicit action. Rerunning evaluation does not retrain.

MLflow and Optuna sync status is not part of the primary run state. It is a projection status, for example `TRACKING_PENDING`, `TRACKING_SYNCED`, or `TRACKING_SYNC_FAILED`. A run can be `ARTIFACT_SUCCESS` and still show "MLflow out of sync" in the GUI.

Every non-success terminal outcome writes a `run_attempt_manifest.json` into its final workspace, cancellation directory, or quarantine directory. That attempt manifest records the same identity fields, resource observations, failure reason, log checksums, and recovery hints that a successful `artifact_manifest.json` would have carried. Failed runs therefore remain diagnosable and exportable even though they are not promoted as successful artifacts.

The artifact manifest (S2) records the full transition log, so the artifact bundle is honest about what completed.

**Acceptance criteria.**

- No `ARTIFACT_SUCCESS` row exists without all of: validated embedding, validated metrics, committed manifest, and verified checksums.
- An `EVALUATION_FAILED` run can be re-evaluated without retraining the model container.
- The GUI displays a green success state only for `ARTIFACT_SUCCESS`; projection failures are warnings attached to that state, not replacements for it.

---

## S6 — Tiered, Versioned Validation

**Problem.** CFM-5 says pre-flight is too shallow. The critique counters: indiscriminately deepening it turns multiverse into a data-curation platform and walls rookies away from PCA on a small dataset.

**Strategy.** Validation runs at three explicit levels, each opt-in per run or per environment:

| Level | When | Checks |
|---|---|---|
| `basic` | Default. Every run. | File-readability, omics-presence, `n_cells > 0`, batch/cell-type columns exist if requested, hyperparameters validate against schema. Cheap; should reject obvious waste in <1 s. |
| `strict` | Opt-in for publication runs. | Adds: batch has ≥2 unique values, cell_type has ≥2 unique values, no NaN-dominated obs columns, dataset shape coherent with hyperparameters (e.g. `n_components ≤ n_cells`), modality-presence per model contract, image digest pinned. |
| `developer` | Opt-in when adding a new model. | Adds: round-trip a tiny synthetic dataset through the candidate model image, verify it honours the container contract end-to-end. |

Each validator is **versioned** and **tied to a model manifest version**, not scattered through the runner. A model manifest declares the validators it requires; mvd resolves and runs them. Drift between validator and container reality becomes a manifest-version bump, not a hidden change.

Validators are warnings *by default* and hard fails *when the level requires*. A user running `basic` sees "your batch column has 1 unique value — batch metrics will be skipped" and proceeds. A user running `strict` sees the same condition as a refusal.

Pre-flight results are cached keyed by `(manifest_hash, dataset_fingerprint, validator_version)` so repeated runs do not re-validate.

**Acceptance criteria.**

- `multiverse run --strict` is the documented publication path.
- A new model adds its validators in `model.yaml`; no edits to the runner.
- The pre-flight log distinguishes warning from refusal.

---

## S7 — Docker Supervisor with Labels and Leases

**Problem.** The orchestrator tracks containers by Python-held container IDs. Docker daemon restart, Docker Desktop sleep, host reboot mid-run, image prune, or container name collision can split the runner's belief from actual container state.

**Strategy.** Every container mvd launches carries a fixed set of Docker labels:

```text
multiverse.run_id=<physical_attempt_id>
multiverse.logical_run_id=<content-addressed id>
multiverse.manifest_hash=<sha256>
multiverse.workspace=<absolute path>
multiverse.owner_token=<token>
multiverse.mvd_version=<...>
multiverse.host_pid=<mvd pid at launch>
```

Reconciliation is then a query against the Docker daemon: `docker ps -a --filter "label=multiverse.run_id=..."`. The daemon's state is authoritative for "is this container alive?"; the journal is authoritative for "what did we intend?". The supervisor cross-references on startup and on every reconciliation tick.

The supervisor also holds a **lease** per running container, refreshed periodically. If `mvd` dies, the lease expires; the next mvd boot reads the journal, queries Docker by label, and either reattaches (container still alive) or transitions to a terminal failure (container exited while we were dead).

**Acceptance criteria.**

- `docker restart` mid-run does not orphan tracking: mvd reconciles on next tick.
- `docker rm` of a known container is detected and recorded.
- A user can answer "what multiverse containers are running right now?" with one labelled `docker ps` query.

---

## S8 — Cancellation as a Saga

**Problem.** Killing the Python runner does not guarantee child containers stop, MLflow finalizes, workspace is marked cancelled, or partial outputs are preserved.

**Strategy.** Cancel is a saga with the same shape as promotion:

1. **CANCEL_REQUESTED** journal entry committed.
2. `docker stop --time=<grace>` issued to every container with `multiverse.run_id=<id>`.
3. After grace period, `docker kill`.
4. Snapshot `container.log`, `model.log`, partial metrics into the workspace.
5. Move workspace to `store/cancelled/<run_id>/` (separate root from `quarantine/` so users can find their own cancellations).
6. Final transition to `CANCELLED`.
7. MLflow run closed with status `KILLED`.

Every step idempotent. Replay-safe. The GUI's Cancel button submits the intent and returns immediately; mvd drives the saga.

**Acceptance criteria.**

- Cancelling a job leaves the user with a recoverable workspace and a labeled MLflow run, not a directory full of orphan files.
- `CANCELLED` is a distinct GUI state with its own colour/icon.

---

## S9 — Resource Broker

**Problem.** CFM-6: memory accounting is a static reservation off `psutil.virtual_memory().total`. The critique adds: memory is one resource among many; CPU oversubscription, GPU VRAM, file descriptors, inode exhaustion, and Docker log growth also kill runs.

**Strategy.** Replace the `ResourcePool` with a **resource broker** inside mvd that:

- Re-polls host metrics at admission time (not just at startup): RAM free, VRAM free per GPU, disk free for `store/`, disk free for Docker data root, inode count for both, mvd's own open-FD count.
- Maintains a *live* reservation ledger that subtracts admitted-but-unrealized usage from observed free.
- Subscribes to Docker events to detect `OOMKilled` exits and feeds them back as broker signals (lower headroom estimate).
- Refuses admission with a clear, surface-able reason: "Cannot admit `multivi` job: only 4 GiB VRAM free on cuda:0, model requests 8 GiB."
- Surfaces *why* a job is `PENDING` in the GUI ("waiting for 6 GiB RAM").

GPU VRAM accounting uses `nvidia-smi` JSON output or NVML when available, and degrades to "GPU jobs serialize" when not.

**Acceptance criteria.**

- The GUI shows "PENDING — waiting for resources" with the specific shortage.
- OOM-killed runs surface a distinct failure category, not a generic `FAILED`.
- A job that needs more VRAM than the host has is rejected at admission, not at container runtime.

---

## S10 — First-Class Garbage Collection with Ownership Tokens

**Problem.** Workspaces, failed runs, Docker containers, dangling images, `metrics.jsonl` sidecars, MLflow artifacts, Optuna studies, and quarantine directories all grow until a local disk fills. Today, nothing reclaims them.

**Strategy.** GC is a first-class mvd subsystem with:

- **Retention policies** declared in config: `failed_workspaces > 7 days`, `cancelled_workspaces > 30 days`, `quarantine > 90 days`, `dangling_docker_objects with multiverse.* labels > 1 day`.
- **Ownership tokens** (S3) are the deletion gate. mvd refuses to delete anything that does not carry a token it issued, even if a retention window has expired. (Manual deletion is always available via the GUI's "Recovered Runs" tab.)
- **Dry-run mode** by default. GC produces a report the user approves in the GUI before any deletion runs.
- **Checksum verification** for "archive successful artifacts": before tarring an artifact bundle for archival, verify every checksum in `artifact_manifest.json`.
- **Docker reclamation** queries Docker by `multiverse.*` labels and reclaims only objects we created. Never `docker system prune`.

**Acceptance criteria.**

- A user can run for a year with disk growth visible through `doctor` and `gc --dry-run`; promoted artifacts are never silently deleted.
- GC never deletes a directory without an owner token match.
- The dry-run report is the default; explicit confirm is required to delete.

---

## S11 — Storage Capability Probes and a Storage Backend Abstraction

**Problem.** Audit P1/#9 (`MVEXP_STORE_ROOT`) is necessary but, as the critique notes, dangerous if naive. Users will point storage at Dropbox, OneDrive, NFS, exFAT, or read-only institutional mounts — each with subtly different POSIX semantics.

**Strategy.** mvd treats the storage root as a **backend**, not a path string. At startup and before each run it executes capability probes:

- Create a tempfile, fsync, `rename` it into place, verify content survives, delete it.
- Open with `O_DIRECT` or check `fallocate` availability; record what the FS supports.
- Probe case sensitivity, symlink behaviour, hardlink behaviour, atime semantics.
- Verify free space ≥ a configurable run reservation.
- Verify the path is visible from inside Docker (mount probe via a one-shot probe container).
- Reject Dropbox/OneDrive paths heuristically (the presence of `.dropbox`, `.tmp.driveupload`, etc.) and require an `--i-know-what-im-doing` flag to proceed.

Probe results are recorded in `mvd`'s health snapshot and surfaced by `multiverse doctor` (S17).

Failed probes refuse to start the daemon with a clear message naming the missing capability.

**Acceptance criteria.**

- Pointing the store at a read-only mount fails at startup with a clear reason, not at promotion time.
- Pointing the store at a Dropbox-synced folder warns explicitly.
- Docker-mount visibility is verified at startup.

---

## S12 — Idempotent Retries with Deterministic Keys

**Problem.** Audit P1/#8 recommends retries for transient failures. The critique correctly warns that retrying promotion, MLflow logging, DB writes, or evaluation without idempotency duplicates metrics, overwrites artifacts, or marks the wrong attempt successful.

**Strategy.** Every retryable side effect carries a deterministic idempotency key. The key is derived from immutable inputs of the operation:

| Operation | Idempotency key |
|---|---|
| Image pull | `image:digest` |
| Container launch | `multiverse.run_id` (Docker rejects duplicates by name; we use a derived name) |
| MLflow run create | `manifest_hash + dataset_fingerprint + attempt_id` |
| MLflow metric log | `(run_id, metric_name, step)` deduped client-side |
| Optuna trial record | `study_name + trial_number` |
| Artifact promotion | `owner_token` |
| DB state transition | optimistic `WHERE status = <expected>` |

Only operations with a key are retryable. Operations without a key are *not* retried — they propagate.

Retries use bounded exponential backoff with jitter and a maximum attempt count surfaced in the journal.

**Acceptance criteria.**

- Two retries of the same MLflow metric log produce one MLflow point, not two.
- A retried promotion never overwrites a different attempt's data.
- A retry log is browsable per run in the GUI.

---

## S13 — Declare MLflow's Posture: Cache, Not Contract

**Problem.** Today MLflow is treated as semi-canonical: the GUI's Analysis tab assumes it. But its failures are logged-and-ignored. This is the "split-brain" pattern the critique flags.

**Strategy.** Declare MLflow a **cache** of the artifact store. The artifact store (S2) is the contract; MLflow is a denormalized projection optimized for cross-run comparison.

Concrete consequences:

- MLflow failures set projection status to `TRACKING_SYNC_FAILED` but do not block `ARTIFACT_SUCCESS` for the underlying artifact bundle.
- A `multiverse mlflow-sync` command rebuilds MLflow entries from artifact manifests. The reverse direction does not exist.
- Documentation says, in plain language: "Cite the artifact bundle, not the MLflow link."
- The GUI's Analysis tab shows a banner when MLflow is out of sync with the artifact store, with a "Sync now" button.

**Acceptance criteria.**

- A user with a corrupted MLflow DB can `multiverse mlflow-sync` and restore the dashboard.
- The artifact bundle is sufficient for publication; nothing critical lives only in MLflow.

---

## S14 — Clock and Timestamp Discipline

**Problem.** Host and container clocks can disagree (VM time sync, Docker Desktop, suspend/resume). Today the system writes wall-clock timestamps everywhere and sorts by them. Across a suspend/resume, ordering is wrong.

**Strategy.** mvd records, for every state transition and every artifact:

- `monotonic_ns` — monotonic counter from the mvd process start.
- `mvd_boot_id` — UUID generated at daemon start so monotonic values from different daemon lifetimes are never compared without an epoch.
- `wall_iso` — wall-clock ISO 8601 with explicit timezone.
- `clock_source` — `"host"` or `"container"` and, for containers, an offset measured at launch.

Ordering uses monotonic. Display uses wall-clock. Forensic reconstruction has all three.

The artifact manifest's `produced_at` field is a struct, not a string.

**Acceptance criteria.**

- A laptop suspend during a run produces correctly-ordered transitions on resume.
- Per-epoch metric streams from inside containers carry both the container's wall-clock and an offset to host-monotonic.

---

## S15 — Output Semantic Validation Before Promotion

**Problem.** Today, file existence is mistaken for validation. A model can write malformed HDF5, wrong-shape latent, NaN embeddings, mismatched cell order, or metrics for a different dataset, and the platform shrugs and writes `SUCCESS`.

**Strategy.** Between container exit and promotion, mvd runs a **post-flight validator** that asserts:

- `embeddings.h5` exists, opens, contains exactly one top-level dataset named `latent`.
- `latent.shape[0]` equals the input cell count recorded in the dataset fingerprint.
- `latent.dtype` is floating; finite-value fraction is above a threshold (configurable; default 100%).
- `metrics.json` parses and matches the metrics schema for the model contract version.
- `umap.png` exists and is a non-trivial PNG (header check, minimum size).
- All declared artifacts in the model contract are present and pass per-artifact checksum.

Validation failure transitions to `EVALUATION_FAILED` or `PROMOTION_FAILED` per the model contract's classification. Validation success is the precondition for promoting to the artifact store, and the per-artifact checksums become entries in `artifact_manifest.json`.

**Acceptance criteria.**

- A model that writes a NaN-filled embedding cannot reach `SUCCESS`.
- An embedding with the wrong row count is rejected before promotion.
- The artifact manifest's checksum entries are verified on every GUI display.

---

## S16 — Content-Addressed Run Identity and Image Digest Pinning

**Problem.** Run IDs are random UUIDs and image tags are mutable. Two identical manifests get unrelated run IDs; a quietly-rebuilt `multiverse-pca:1.0.0` silently changes behaviour.

**Strategy.** Separate **logical run identity** from **physical attempt identity**:

- **Logical run ID** = `sha256(manifest_hash || dataset_fingerprint || image_digest || params_hash || mv_contract_version)`. The same recipe always hashes to the same logical ID.
- **Physical attempt ID** = a UUID per concrete execution.

The artifact store is indexed by logical ID; multiple attempts cluster under the same logical run and surface as a replay group in the GUI.

Image identity uses the immutable `sha256:` digest pulled at admission time, not the human-readable tag. The digest is written to the journal at admission, into the container's environment, and into the artifact manifest. Strict-mode runs (S6) refuse to start if the digest is missing.

**Acceptance criteria.**

- Rerunning an identical manifest groups under the same logical ID in the GUI.
- A rebuilt model image at the same tag is detected as a different digest and treated as a different logical run.
- Publication-mode runs cannot be launched without a pinned digest.

---

## S17 — `multiverse doctor` and `multiverse rebuild-index`

**Problem.** A researcher with a hot laptop, full disk, and a `database is locked` traceback will not read a runbook. The platform must diagnose itself in-app.

**Strategy.** Two top-level commands, both surfaced as a "Diagnostics" panel in the GUI:

- **`multiverse doctor`** runs a battery of probes and prints a colour-coded report:
  - DB health: `PRAGMA integrity_check`, WAL file size, schema version vs expected, last successful checkpoint timestamp.
  - Storage health: every probe from S11, current free space, inode count.
  - Docker health: daemon version, data-root path, free space on data-root, count of multiverse-labelled containers and images, any unreconciled labels.
  - mvd health: PID, uptime, lease holders, pending journal entries, queue depths.
  - MLflow / Optuna health: end-to-end probe (create temp run, log temp metric, read it back, delete it).
- **`multiverse rebuild-index`** walks `store/artifacts/`, reads every `artifact_manifest.json`, and rebuilds `mvexp_state.db` with a per-row conflict report. Quarantines anything it cannot adopt. Does not delete.

The GUI Diagnostics panel runs `doctor` on demand and offers one-click "safe repair" actions: checkpoint WAL, restart mvd, run a Docker reconciliation tick, re-probe storage.

**Acceptance criteria.**

- A user who deletes `mvexp_state.db` can run `multiverse rebuild-index` and recover their history.
- `multiverse doctor` exits non-zero if any blocking probe fails, so it composes with shell scripts.
- The GUI shows a red banner with a "Run doctor" button when probes fail in the background.

---

## S18 — Portable Export and Import Bundles

**Problem.** A local workstation is more exposed to accidental deletion, OS reinstall, laptop loss, and cloud-sync conflicts than a managed server. The audit and critique both flag the absence of any backup story.

**Strategy.** Two commands:

- **`multiverse export-run <id>`** produces a portable archive containing:
  - The full artifact directory.
  - The `artifact_manifest.json` (already there).
  - A copy of the model manifest and the run manifest.
  - The image digest and a `manifest.txt` containing every input checksum.
  - An environment record: mvd version, mv-worker version, Python version, host OS, GPU model.
  - A `README.md` describing how to reproduce.
- **`multiverse export-study <name>`** bundles every run in a logical study (S16 grouping) plus the Optuna study export.
- **`multiverse import-run <archive>`** verifies checksums, adopts the run into the local store, and reindexes.

Export bundles are the canonical *publication-supplementary-material* artifact. Documentation directs users to attach them, not to attach `mvexp_state.db`.

**Acceptance criteria.**

- A user can hand a `multiverse export-run` output to a colleague and have them reproduce the result from a fresh clone.
- The export verifies its own checksums on creation and on import.

---

## S19 — Defensive Surface Against Untrusted Model Registrations

**Problem.** The critique reminds us that "single user" does not mean "no untrusted input." A model registration is essentially `Dockerfile + run.py`; a malicious or buggy model.yaml with `..` in paths, a symlink under `store/`, or a privileged Docker run flag can destroy a user's machine without any multi-user threat model.

**Strategy.** Model registration runs through a defensive pipeline:

- **Path normalization**: every path in `model.yaml` and `dataset.yaml` is resolved with `realpath` and checked to stay under the configured store root. Paths escaping the root are rejected.
- **Symlink discipline**: during promotion and GC, symlinks under `store/` are followed-with-canonicalization and any link target outside the store is treated as quarantine-worthy.
- **Privilege labels**: any model.yaml that requests `privileged`, host network, host PID namespace, or `--volume` outside `/input` and `/output` is flagged as "elevated" and requires explicit user confirmation at registration.
- **Trusted vs imported**: built-in models registered by `make register-models` are trusted; user-imported model directories are marked `imported` and surface a banner in the GUI.

**Acceptance criteria.**

- A `model.yaml` with `raw_files: rna: ../../../etc/passwd` is rejected at parse time.
- Promotion never follows a symlink out of the configured store root.
- The GUI distinguishes built-in from imported models visually.

---

## S20 — Minimal Reliable Core / Simple-Mode

**Problem.** The critique's final warning: multiverse risks becoming a distributed systems platform when the user's task is "compare a few biological models." Every strategy above adds machinery. The platform needs a counter-strategy.

**Strategy.** Define and protect a **minimal reliable core**:

```text
multiverse run --simple path/to/run_manifest.yaml --out path/to/bundle/
```

This single command:

- Reads the manifest.
- Pulls images by digest (or uses local).
- Runs each container with the standard contract.
- Validates outputs (S15).
- Writes a portable bundle (S18) — a single directory the user can tar.
- Does **not** require mvd, MLflow, Optuna, the GUI, or the SQLite registry.

The simple-mode bundle is the same shape as the export-run output. A user who wants comparison dashboards opts in by running with mvd. A user who wants reproducibility on a colleague's laptop opts for simple mode.

This is the strategy that protects the platform's purpose against its own machinery.

**Acceptance criteria.**

- The simple-mode command works on a fresh clone with only Docker installed.
- The simple-mode bundle round-trips through `multiverse import-run` cleanly when mvd is available.
- Artifact-contract and validator tests run on every commit; Docker end-to-end simple-mode runs execute in the integration CI lane or nightly job.

---

## Sequencing

These strategies are not independent. Because there are no legacy users to migrate, the delivery order should optimize for correctness and contract clarity rather than backward compatibility:

1. **Portable artifact contract:** S2 (artifact manifests), S14 (timestamp discipline), S15 (output validation), S16 (logical run identity), S18 (export/import). This defines what a correct result is before any daemon owns it.
2. **Simple-mode core:** S20 (simple-mode protection), using the same artifact contract. This proves the platform can produce a publication-grade bundle without GUI, MLflow, Optuna, SQLite, or `mvd`.
3. **Execution safety:** S3 (promotion saga), S4 (quarantine recovery), S7 (Docker labels), S8 (cancel saga), S12 (idempotency keys). This closes the direct data-loss paths.
4. **Daemon kernel:** S1 (`mvd`) with only journal, leases, Docker supervision, state transitions, cancellation, and promotion ownership. Delete old direct-control paths once this path is working; do not preserve them as parallel architecture.
5. **Operational durability:** S9 (resource broker), S10 (manual-first GC), S11 (storage probes), S6 (tiered validation), S13 (MLflow as cache/projection).
6. **Self-service and surface hardening:** S17 (`doctor` / `rebuild-index`) and S19 (untrusted model defence).

Steps 1-4 are the irreducible minimum to call multiverse production-grade for a single local user. Steps 5-6 are important, but they should not expand the daemon kernel until the artifact contract and execution safety path are stable.

---

## Implementation Order

This is the build order. It is intentionally stricter than the strategy map: later milestones should not start until the exit gates of earlier milestones are met, except for design-only work.

| Order | Milestone | Implement | Do not implement yet | Exit gate |
|---|---|---|---|---|
| 0 | Platform ADR | `docs/adr/0001-mvd-platform-assumptions.md`; supported OS/filesystem matrix; daemon launch model; socket auth; Docker Desktop assumptions; Python and dependency minimums. | Daemon code, journal code, or promotion rewrites. | ADR merged and linked from this document; CI has a placeholder check that fails if the ADR is missing. |
| 1 | Artifact Contract Library | Shared library for `artifact_manifest.json`, `artifact_manifest.sha256`, `run_attempt_manifest.json`, logical/physical IDs, image identity union, timestamp fields including `mvd_boot_id`, checksum helpers, normalized bundle comparison. | Docker execution, daemon, MLflow, Optuna, GUI work. | Unit tests cover atomic manifest write/read, checksum mismatch detection without mutation, logical ID stability, and normalized bundle equivalence. |
| 2 | Validators and Bundle Writer | Output semantic validators, validation levels (`basic`, `strict`, `developer`), export bundle layout, failure attempt manifest writer. | Resource broker, GC, daemon API, MLflow sync. | Fixture artifacts pass/fail deterministically; failed/cancelled/quarantined attempts produce `run_attempt_manifest.json`; no DB required. |
| 3 | Simple-Mode Runner | `multiverse run --simple`; Docker contract execution; image identity resolution; bundle output using the shared writer; strict-mode refusal rules. | Streamlit integration, daemon supervision, SQLite rebuild, background services. | Fresh-clone simple-mode run produces a contract-valid bundle; fast tests run every commit; Docker E2E runs in integration/nightly. |
| 4 | Journal Core | Append-only `store/journal/`; segment rotation; blob spill for large manifest payloads; fsync durability boundary before returning durable IDs; replay reader. | Long-running daemon, GUI API, MLflow, GC. | Crash tests prove acknowledged journal records survive process kill; replay reconstructs run intent from current and rotated segments. |
| 5 | Promotion and Recovery Core | Ownership tokens, promotion saga, cross-filesystem staged copy, quarantine-only recovery, tombstones, descriptor-relative filesystem helpers at destructive/promotion boundaries. | Daemon API exposure, resource broker, automatic GC. | Fault-injection tests kill between every promotion step; no hot-path deletion of result-like data; recovery quarantines rather than deletes. |
| 6 | Docker Supervision Core | Docker labels, container lease records, reattach/reconcile by label, cancellation saga, `EVALUATING` and primary-state transitions. | Streamlit client rewrite, MLflow projection, doctor GUI. | Host process kill/restart reconciles running/exited containers; cancel leaves a recoverable workspace and attempt manifest. |
| 7 | Minimal `mvd` Kernel | Unix-socket API with the seven v1 verbs, journal-backed state machine, submit/cancel/query/list/stream/health/projection-report, paused-kernel maintenance lock. | GC, resource broker, MLflow sync as kernel code, broad registration APIs. | GUI/CLI can run a job through `mvd`; kernel import graph excludes MLflow, Optuna, GC, exporter; API surface test enforces seven verbs. |
| 8 | Index and Rebuild | SQLite as rebuildable index; `multiverse rebuild-index` as paused-kernel maintenance; three-source merge from journal, Docker labels, and artifacts. | Advanced GUI diagnostics, retention cleanup. | Deleting `mvexp_state.db` and rebuilding restores promoted runs and classifies in-flight/failed attempts without deletion. |
| 9 | GUI and CLI Client Cutover | GUI/CLI use the daemon API for run submission, cancellation, state, and logs; remove old direct-control paths instead of maintaining parallel architecture. | New product features. | Grep confirms GUI has no Docker client and no direct run-state DB writes; old subprocess runner path is gone. |
| 10 | Projections | `mvd-mlflow-sync`; projection status reporting; Analysis tab out-of-sync banner; Optuna projection if needed. | Treating MLflow/Optuna as primary truth. | MLflow outage still yields `ARTIFACT_SUCCESS`; later sync changes only projection status. |
| 11 | Diagnostics and Storage Health | `multiverse doctor`; storage probes with supported/degraded/dangerous/blocked levels; hidden health probe namespaces; `mvd-health-sweeper`. | User-visible artifact deletion. | Doctor is useful on a broken local install; health probe leaks are reported separately from probe failures and repairable. |
| 12 | Manual-First Maintenance | `multiverse gc --dry-run`; explicit `--apply`; owner-token deletion gates; export-required deletion policy; GC reports. | Automatic deletion of user-visible artifacts. | Default GC cannot delete promoted, failed, cancelled, or quarantined user artifacts; auto-clean is limited to enumerated daemon scratch. |
| 13 | Resource Broker | Admission-time and continuous resource observation; pressure modes; OOM classification; resource observations in manifests. | Automatic killing of running jobs. | Stress tests pause admission under pressure, never kill user jobs, and classify OOM as `OOM_KILLED`. |
| 14 | Registration Hardening | `mvd-register`; path normalization; symlink policy; trusted/imported model distinction; elevated Docker flag warnings. | Multi-user auth or remote registry. | Malicious path escapes are rejected; GUI distinguishes built-in and imported models. |

**Blocking rules.**

- No daemon implementation before Milestones 1-3 define and test the artifact contract through simple-mode.
- No GUI cutover before Milestone 7 has a passing seven-verb API contract test.
- No MLflow/Optuna work before `ARTIFACT_SUCCESS` is independent of projection status.
- No GC deletion before export, owner-token, retention, and dry-run reports are implemented.
- No publication/strict-mode claim until storage health, image identity, validators, and artifact manifests all stamp their evidence into the bundle.

**First production-grade checkpoint.** Milestones 0-9 are the minimum bar for "production-grade local single-user execution." Milestones 10-14 improve operations and ergonomics, but they should not enlarge the daemon kernel or weaken the artifact contract.

---

## Explicit Out-of-Scope

To prevent gold-plating, the following are declared *not* on the roadmap and require a product-line decision to add:

- Multi-tenant access control, network-exposed services with auth, RBAC.
- Distributed scheduler, multi-host execution, cluster Kubernetes deployment.
- Replacing SQLite with Postgres for the local install.
- A managed cloud offering.

If any of these become product goals, this strategy document is no longer sufficient and a separate Platform Architecture revision is required. Until then, the single-user-local product boundary is enforced *as a feature*: it is what makes the system small enough to reason about.

---

## Red-Team Response: Implementation Specifications

The red-team critique at the top of this document flags fifteen failure modes the strategy itself can introduce. The S1-S20 bodies have been edited to absorb those critiques at the strategy level; this section binds them down to concrete data structures, file layouts, decision trees, and contract surfaces.

**Precedence rule.** Where an R-spec below disagrees with an S-strategy body, the R-spec wins. The S bodies are the high-level shape; the R-specs are the binding details.

The order below mirrors the rows of the red-team table for traceability.

### R1 — Daemon kernel vs plugins

**Refines.** S1, S10, S13, S17.

**Specification.** Split `mvd` into a tightly-scoped **kernel** and a set of **plugins** that run in distinct processes and never share writable state with the kernel.

Kernel surface — the only code allowed in the hot path:

- Append-only journal writer and reader (see R3 / R4).
- Lease manager (acquires and renews container leases, fails over on missed renewals).
- Docker supervisor (launch by label, reconcile by label, kill by label).
- State machine: validated transitions over the journal-backed `runs` table.
- Promotion driver: executes the saga in S3 under explicit ownership tokens.
- Cancellation driver: executes the saga in S8.
- Local API: a minimum surface, see R2.

Plugins — run as short-lived subprocesses or as separate long-running processes the kernel does not depend on:

| Plugin | Process model | Trigger | Failure isolation |
|---|---|---|---|
| `mvd-mlflow-sync` | Worker process, restartable | Kernel post-promotion event or manual `multiverse mlflow-sync` | Kernel marks projection status `TRACKING_SYNC_FAILED`; never blocks artifact success. |
| `mvd-gc` | One-shot CLI invocation | Manual `multiverse gc` (R12) | Cannot run while kernel holds an active lease on the candidate dir. |
| `mvd-doctor` | One-shot CLI invocation | Manual `multiverse doctor` | Read-only against kernel; cannot mutate. |
| `mvd-health-sweeper` | One-shot CLI invocation | Manual `multiverse doctor --repair-health-probes` or idle maintenance | Mutates only reserved health-probe namespaces; never user artifacts or user experiments. |
| `mvd-export` | One-shot CLI invocation | Manual `multiverse export-run` | Read-only against artifact store; takes an advisory read lease. |
| `mvd-register` | One-shot CLI invocation | GUI or CLI registration flow | Validates and writes only under `store/models/` or `store/datasets/`; no Docker or run-state mutation. |
| `mvd-rebuild-index` | Kernel maintenance command, not a normal plugin | Manual `multiverse rebuild-index` | Kernel refuses new submissions while rebuild is in progress; this is the only non-hot-path component allowed to rewrite SQLite. |

Plugins talk to the kernel only through the local API (R2) and the artifact filesystem; normal plugins do not open SQLite directly. `mvd-rebuild-index` is explicitly a paused-kernel maintenance mode, not a normal plugin. If a plugin crashes, the kernel does not.

**Acceptance.**

- Killing `mvd-mlflow-sync` mid-sync does not delay any kernel transition.
- `multiverse gc` failing does not prevent further job submissions.
- The kernel binary's import graph is grep-checked: no MLflow, no Optuna, no GC scheduler, no exporter.

---

### R2 — Strict, narrow daemon API

**Refines.** S1.

**Specification.** The kernel exposes a single Unix-domain-socket API at `${MULTIVERSE_STATE_ROOT}/mvd.sock` (default `~/.local/state/multiverse/mvd.sock`). HTTP-over-loopback is permitted only when explicitly configured.

The API is intentionally tiny:

| Verb | Semantics | Idempotency key |
|---|---|---|
| `submit_run(manifest_path, options)` | Validate manifest, write `JobIntent` to journal, return `physical_attempt_id`. | client-supplied or derived from `manifest_hash + nonce` |
| `cancel_run(physical_attempt_id)` | Append `CANCEL_REQUESTED` intent. | `physical_attempt_id` |
| `query_run(physical_attempt_id)` | Read-only state snapshot. | n/a |
| `list_runs(filter)` | Read-only summary. | n/a |
| `stream_events(physical_attempt_id)` | Server-sent events: state transitions + log tail. | n/a |
| `health()` | Kernel self-check; no probes. | n/a |
| `report_projection_status(plugin, physical_attempt_id, status, details)` | Projection plugin reports `TRACKING_SYNCED`, `TRACKING_SYNC_FAILED`, or equivalent projection-only status. Kernel validates the projection name and allowed transition. | `plugin + physical_attempt_id + projection_generation` |

That is the complete kernel API for v1. Notable omissions, deferred to either plugins or later versions:

- No `register_model`, `register_dataset`. Registration is handled by the `mvd-register` plugin invoked by GUI or CLI. The plugin validates paths and writes under `store/models/<slug>/model.yaml` or `store/datasets/<slug>/dataset.yaml`. The kernel discovers changes at the next reconciliation tick and emits a `RegistryStale` event for the GUI.
- No `delete_run`, no `gc_run`. Deletion is a `multiverse gc` plugin operation that the kernel only authorizes (by checking ownership tokens) but does not perform.
- No `export_run`. Export is `mvd-export` reading the artifact store directly.
- No `sync_mlflow`. MLflow sync is an event-triggered plugin.

API auth is filesystem-permission-based in Unix-socket mode: the socket is mode `0600` and owned by the user running `mvd`. If HTTP-over-loopback is enabled, it additionally requires a random per-session bearer token and strict browser-origin checks; localhost alone is not an auth boundary.

**Acceptance.**

- The kernel's API handler module is under 500 lines and exposes exactly seven verbs.
- GUI and CLI bindings depend only on the seven verbs.
- A test enumerates `dir(api)` and fails if a new verb appears without a roadmap entry.

---

### R3 — Append-only journal and rebuild from multiple sources

**Refines.** S2, S17.

**Specification.** A separate persistence surface, `store/journal/`, is the durable record of *intent*. It is not the same file as the SQLite index and is not the same file as the per-run `artifact_manifest.json`. It is the **only** thing the kernel writes synchronously before any side effect.

Journal layout:

```text
store/journal/
  current.log                 active append-only segment
  rotated/
    <iso-timestamp>.log.zst   rotated segments (gzip or zstd)
  checkpoint.json             last segment offset known fully reconciled
```

Record format (one JSON object per line, newline-delimited, fsync on group commit):

```text
{
  "seq": 42,
  "monotonic_ns": 12345678901234,
  "wall_iso": "2026-05-27T11:00:00+02:00",
  "mvd_boot_id": "<uuid>",
  "kind": "PROMOTE_PREPARE",
  "physical_attempt_id": "...",
  "payload": {...},
  "prev_state": "TRAINING_SUCCEEDED",
  "next_state": "PROMOTING"
}
```

An API call that returns a durable identifier, such as `submit_run`, may return only after the journal record containing that identifier has reached the configured durability boundary. Default boundary: file data fsynced and parent segment directory fsynced when a new segment is created. Group commit may batch multiple API calls, but it may not acknowledge any one of them before the batch fsync succeeds.

Rebuild is a three-source merge, not a manifest-only walk:

1. **Journal** is authoritative for *intent*: what the kernel meant to do, in order.
2. **Docker labels** are authoritative for *runtime existence*: what containers are alive right now.
3. **Workspace and artifact directories** are authoritative for *outcome*: what files exist on disk.

`multiverse rebuild-index` performs the merge:

- Replay the journal forward from the last checkpoint.
- For each `RUNNING` record without a terminal transition, query Docker by label. If alive, reattach. If exited, classify by exit code and journal-resume.
- For each `PROMOTE_PREPARE` without `PROMOTE_COMMIT_MANIFEST`, quarantine the candidate artifact dir (R5) and mark the run `RECOVERY_PENDING`.
- For each `PROMOTE_COMMIT_MANIFEST` with a valid `artifact_manifest.json` on disk, mark the run `ARTIFACT_SUCCESS` even if the SQLite index never got the commit.

The artifact manifest alone is not sufficient: in-flight runs have no manifest yet. The journal supplies the missing fact.

**Acceptance.**

- A run killed during `RUNNING` reconciles correctly with `mvexp_state.db` deleted and only `journal/` + Docker labels present.
- A test that deletes both the index and `current.log` (forcing replay from rotated segments only) still reconciles the last 100 promoted runs.
- The journal segment size is bounded; rotation triggers at a configurable threshold.

---

### R4 — Atomic manifest writes with detached checksum

**Refines.** S2, S3.

**Specification.** The artifact manifest is not a single file written in place. It is a three-step content-addressed commit:

1. Write `artifact_manifest.json.tmp` in the artifact directory. `fsync` the file.
2. Compute `sha256(artifact_manifest.json.tmp)` and write `artifact_manifest.sha256` (detached sidecar). `fsync`.
3. `rename` `artifact_manifest.json.tmp` → `artifact_manifest.json`. `fsync` the artifact directory inode.

Every reader of `artifact_manifest.json` MUST verify it against `artifact_manifest.sha256`. A mismatch is treated as corruption: ordinary readers return a corruption result and report the directory as `RECOVERY_PENDING`, but they do not rename, rewrite, or delete anything. Only explicit repair commands such as `multiverse doctor --repair` or `multiverse rebuild-index` may move the bad manifest to `artifact_manifest.corrupt.<timestamp>.json` and attempt journal-driven reconstruction.

A compact transition log is duplicated into the journal so that the manifest is reconstructable from journal records alone if both the manifest and its sidecar are lost:

```text
journal record kind: ARTIFACT_MANIFEST_COMMIT
payload: {
  "artifact_dir": "...",
  "manifest_sha256": "...",
  "manifest_body": {...},   # embedded only when <= configured inline limit
  "manifest_blob": "store/journal/blobs/<sha256>.json"  # otherwise
}
```

Large manifest bodies spill to content-addressed blobs under `store/journal/blobs/<sha256>.json`; the journal record stores the blob hash and path. The inline limit defaults to 256 KiB. Blob writes follow the same temp-write, fsync, rename, fsync-parent protocol.

This makes the artifact manifest a *cache* of the journal record, not the other way around. If the file on disk is corrupt and the sidecar disagrees, ordinary readers report corruption but do not mutate the artifact directory. `multiverse doctor --repair` or `multiverse rebuild-index` may re-emit the manifest from the journal.

**Acceptance.**

- `truncate -s 0 artifact_manifest.json` after promotion is detected on next access and reconstructed from the journal only by an explicit repair command.
- Manual editing of `artifact_manifest.json` (e.g. an over-eager user) is caught by the checksum check at next display.
- `fsync` of the parent directory after the rename is exercised by a test using `strace` or equivalent.

---

### R5 — Quarantine even for owned failed promotion directories

**Refines.** S3 step 2, S4, S10.

**Specification.** Owner-token ownership authorizes *adoption*, not *deletion*. In the hot path the kernel may move an owned-failed directory only to quarantine; it may not unlink it. Deletion is a GC operation gated by retention policy, executed only by the `multiverse gc` plugin (R12).

S3 step 2 is amended:

> 2. **VALIDATE.** Run output semantic checks (S15) against the workspace. On failure, transition to `EVALUATION_FAILED` or `PROMOTION_FAILED` (S5), write a `PROMOTION_QUARANTINE` journal entry, and *move* the half-built artifact dir (which we own by token) to `store/quarantine/<date>/<physical_attempt_id>/` with a `QUARANTINE_REPORT.md`. Stop.

The grep gate for "no `rmtree` in recovery code" extends to the entire promotion saga and cancellation saga. The only code path in the entire kernel that calls `os.unlink`, `os.rmdir`, or `shutil.rmtree` is the `multiverse gc` plugin, and that plugin checks owner token, retention age, and a content-checksum manifest before each deletion.

A "tombstone" record is left at the original artifact path when a directory is quarantined:

```text
store/artifacts/<orig-path>.quarantined
  {"quarantined_to": "store/quarantine/<date>/<id>/", "reason": "...", "at": ...}
```

So a user (or a manifest cross-reference) following the original path discovers what happened.

**Acceptance.**

- No `unlink`/`rmdir`/`rmtree` call in `mvd/kernel/**`, enforced in CI.
- A botched promotion produces a quarantine entry; the original artifact path contains a `.quarantined` tombstone.
- The user can `multiverse runs adopt-quarantine <id>` to roll a quarantined dir back into the artifact tree if the false-positive case occurs.

---

### R6 — Primary state focused on artifact correctness

**Refines.** S5, S13.

**Specification.** The kernel exposes two orthogonal state surfaces, never collapsed:

1. **Primary run state** — the scientific outcome. Values: `PENDING`, `ADMITTED`, `RUNNING`, `TRAINING_SUCCEEDED`, `EVALUATION_FAILED`, `PROMOTING`, `PROMOTION_FAILED`, `ARTIFACT_SUCCESS`, `CANCEL_REQUESTED`, `CANCELLED`, `FAILED`. Stored in `runs.primary_state`.
2. **Projection statuses** — denormalized caches of the artifact bundle. Stored in `runs.projections` as a JSON object, e.g. `{"mlflow": "TRACKING_SYNCED", "optuna": "TRACKING_NOT_APPLICABLE"}`. Each projection has its own valid value set.

The GUI's status rollup follows R14's grouping rules; it must surface projection sync failures as advisory chips next to the primary state, never as a replacement for it.

The kernel never delays a `PROMOTING → ARTIFACT_SUCCESS` transition on a projection commit. Projection commits are dispatched as plugin events after `ARTIFACT_SUCCESS` is committed; their failure path mutates only `runs.projections`.

**Acceptance.**

- A run with MLflow disabled reaches `ARTIFACT_SUCCESS` with projection status `TRACKING_NOT_CONFIGURED`.
- A run that completes with MLflow unreachable reaches `ARTIFACT_SUCCESS` with projection status `TRACKING_SYNC_FAILED`; `multiverse mlflow-sync` later transitions only the projection.
- The publication-mode `multiverse run --strict` exits non-zero on `EVALUATION_FAILED` or `PROMOTION_FAILED` but exits zero on `ARTIFACT_SUCCESS` with `TRACKING_SYNC_FAILED`.

---

### R7 — Simple-mode first

**Refines.** S20, Sequencing.

**Specification.** Simple-mode is the **first** implementation milestone, not the last. It defines the contracts that every later phase consumes:

| Contract defined by simple-mode | Consumed by |
|---|---|
| `artifact_manifest.json` schema and write protocol (R4) | S2, S3, S17, S18 |
| Output semantic validator suite (S15) | S3, S15 |
| Image identity union type (R10) | S16 |
| Export bundle layout (S18) | S17, S18 |
| Logical run ID derivation (S16) | S2, S5, S16 |

Until simple-mode ships, no daemon code is written. This is a hard sequencing rule, not a guideline.

Simple-mode CLI shape:

```text
multiverse run --simple <manifest.yaml> \
  --out <bundle-dir> \
  [--strict] \
  [--validators basic|strict|developer] \
  [--no-image-pull]
```

It produces a directory that is contract-equivalent to a `multiverse export-run` archive produced by the daemon path against the same manifest. Equivalence is checked after normalizing timestamps, generated attempt IDs, compression metadata, absolute paths, and log ordering.

The daemon's promotion saga (S3 step 4) imports the simple-mode bundle writer verbatim: there is one writer of artifact manifests in the codebase. The daemon is a *user* of the simple-mode core, not a parallel implementation.

**Acceptance.**

- A CI job builds the simple-mode binary with only the artifact-contract module imported (no kernel, no plugins). Fast fixture tests run on every commit; Docker-backed end-to-end fixtures run in the integration lane or nightly.
- A normalized diff between a daemon-produced bundle and a simple-mode-produced bundle (against the same manifest, same seed, same image identity) is empty modulo approved nondeterministic fields.

---

### R8 — Graded storage health

**Refines.** S11.

**Specification.** Storage capability probes report one of four levels per probe; the daemon's startup decision is a function of the *set* of levels, not a single heuristic.

| Level | Meaning | Crash-safety claim |
|---|---|---|
| `supported` | The capability is present and behaves as expected. | Full. |
| `degraded` | The capability is present but with caveats (e.g. case-insensitive FS, no `O_DIRECT`). | Degraded; specific guarantees disabled and surfaced. |
| `dangerous` | The capability is suspect (cloud-sync path, network FS without locking). | None claimed; user must pass `--accept-degraded`. |
| `blocked` | Required capability missing (no write, no rename, no Docker visibility). | Cannot start. |

Probe matrix (non-exhaustive):

| Probe | Required level for `mvd` start |
|---|---|
| `write_then_read` | `supported` (else `blocked`) |
| `atomic_rename` | `supported` (else `blocked`) |
| `fsync_file` | `supported` (else `dangerous` warned) |
| `fsync_dir` | `supported` (else `degraded` — promotion still works but durability guarantee reduced) |
| `docker_mount_visibility` | `supported` (else `blocked`) |
| `cloud_sync_heuristic` | `supported` if no sync markers; `dangerous` if `.dropbox` / `.tmp.driveupload` / OneDrive markers present |
| `case_sensitivity` | `supported` on case-sensitive FS; `degraded` on case-insensitive (renaming `Run` vs `run` collides) |
| `free_space_reservation` | `supported` if ≥ configured threshold; `degraded` if 10-30% headroom; `dangerous` if < 10% |

Behavior:

- `blocked` on any probe → daemon refuses to start, prints which probe failed and what the user can do.
- `dangerous` on any probe → daemon refuses to start unless `--accept-degraded` was passed on the command line *and* the same flag is recorded in the journal of every run admitted in this session.
- `degraded` on any probe → daemon starts; the affected guarantees are listed in `multiverse doctor` and in the artifact manifest's `produced_by.degraded_capabilities` field.

The artifact manifest's degraded-capabilities list is what stops a user from later claiming a publication-mode result was produced under "supported" conditions when it was not.

**Acceptance.**

- Pointing the store at a Dropbox folder starts the daemon only after an explicit acknowledgement and stamps every run with `degraded_capabilities: ["cloud_sync"]`.
- A read-only mount produces `blocked` with the exact probe name in the error message.
- `multiverse doctor` distinguishes the four levels per probe.

---

### R9 — Doctor probes are hidden-namespaced with TTL

**Refines.** S17.

**Specification.** `multiverse doctor` never touches user dashboards or user artifacts. It may create short-lived probe records only inside reserved hidden namespaces, and it attempts best-effort cleanup of records it created during the current invocation. Cleanup of older leaked probe records belongs to `mvd-health-sweeper`, invoked explicitly through `multiverse doctor --repair-health-probes` or by idle maintenance.

External-service probe contract:

| Service | Reserved name | TTL |
|---|---|---|
| MLflow experiment | `__mvd_health_probe__` | Records have `mvd_probe_expires_at` tag; `mvd-health-sweeper` deletes records older than 1 h. |
| Optuna study | `__mvd_health_probe__<rfc3339>` | Study name embeds creation time; `mvd-health-sweeper` deletes studies older than 1 h. |
| Docker probe container | `multiverse_health_probe_<rfc3339>` | Container removed in `finally`; `mvd-health-sweeper` removes containers with label `multiverse.health_probe=true` older than 1 h. |
| Workspace probe directory | `store/workspaces/__mvd_health_probe__/` | `mvd-health-sweeper` removes entries older than 1 h. |

Each probe produces three outputs:

1. **Probe result** — what the probe was testing. `pass` / `fail` / `skipped` (service not configured).
2. **Cleanup result** — whether the probe's own teardown succeeded. `clean` / `leaked` / `cleanup_failed`.
3. **Leak inventory** — whether older leaked probes from previous runs exist. `leaks_<n>` / `none` / `inventory_failed`. The report suggests `multiverse doctor --repair-health-probes` when leaks exist.

The doctor report shows all three columns. A probe that passed but failed to clean up is not the same as a probe that failed; the user sees both honestly.

Doctor never runs against user experiments. The reserved names are recognized everywhere in the codebase (GUI hides them; export refuses to include them; rebuild-index ignores them).

**Acceptance.**

- Running `multiverse doctor` twice in a row never produces a duplicate entry in the user's MLflow experiment list.
- A probe whose cleanup fails surfaces as `pass / leaked / leaks_3` rather than `fail`, with a repair hint.
- `mvd-health-sweeper` is covered by a unit test that asserts it leaves no health-probe entry older than the TTL.

---

### R10 — Image identity as a union type

**Refines.** S16.

**Specification.** Image identity is one of four variants, declared at admission time and recorded in the journal and artifact manifest:

```text
ImageIdentity = OneOf:
  - {kind: "registry_digest", value: "sha256:..."}
  - {kind: "local_image_id",   value: "sha256:...", note: "no registry digest available"}
  - {kind: "build_context_hash", value: "sha256:...", dockerfile_path: "...", context_root: "..."}
  - {kind: "unverified_local", value: "<image-id-or-tag>", warning: "not pinnable"}
```

Resolution rules at admission:

1. If the image is pulled from a registry → `registry_digest`.
2. Else if locally built and we can recompute the build-context hash deterministically → `build_context_hash`.
3. Else if the image exists locally with a stable image ID → `local_image_id`.
4. Else → `unverified_local`.

Per-mode strict-ness:

| Mode | Acceptable variants |
|---|---|
| `multiverse run --strict` (publication) | `registry_digest` or `build_context_hash` only. |
| Daemon default | Any of the four; warns on `unverified_local`. |
| Simple-mode default | Any of the four; bundle records the variant. |

Logical run ID (S16) is computed over the variant *value*, not the kind, so rebuilds at the same tag produce a different logical ID iff the underlying bytes differ.

**Acceptance.**

- A locally-built image with no registry push produces a logical run ID and reaches `ARTIFACT_SUCCESS` outside strict mode.
- The same recipe under `--strict` is refused with a clear message: "image variant is `local_image_id`; strict mode requires `registry_digest` or `build_context_hash`."
- An artifact manifest produced in `unverified_local` mode is flagged in the GUI with a yellow chip.

---

### R11 — Continuous resource observation and soft preemption

**Refines.** S9.

**Specification.** The resource broker observes host state at three rates:

| Rate | Operation |
|---|---|
| At admission | Re-poll RAM, VRAM, disk, inodes, FDs. Refuse admission if the live total minus the reservation ledger is below the job's request. |
| Continuous (5 s tick during `RUNNING`) | Re-poll the same metrics. Update a rolling pressure score per resource. |
| On Docker events | Update the broker on container exit, OOMKilled, paused, etc. |

When pressure exceeds a configured threshold, the broker enters one of three modes:

| Mode | Behavior |
|---|---|
| `normal` | Admit, run, no intervention. |
| `pressure` | Stop admitting new jobs. Running jobs continue untouched. Surface "resource pressure — admission paused" in the GUI. |
| `critical` | Stop admitting; emit a warning event. Do **not** auto-kill running jobs. The user retains authority over running work. |

`critical` does *not* preempt running containers in local mode. The audit and critique both prioritize user trust over automatic enforcement. If the user wants a container killed, the user kills it.

The artifact manifest records, for each completed run:

```text
"resource_observations": {
  "peak_rss_bytes": ...,
  "peak_vram_bytes": ...,
  "oom_killed": false,
  "broker_pressure_events": [{"at": "...", "level": "pressure", "resource": "ram"}]
}
```

So a user reviewing a failed result can see whether OOM-or-pressure was the proximate cause.

**Acceptance.**

- During a deliberate stress test the broker pauses admission and resumes when pressure subsides; no running job is killed.
- OOMKilled exits propagate to a distinct `FAILED` reason `OOM_KILLED` and the artifact manifest records `oom_killed: true`.
- The GUI surfaces pressure level on the Run tab.

---

### R12 — Manual-first GC; auto-GC limited to daemon scratch

**Refines.** S10.

**Specification.** GC has two tiers, with hard boundaries between them:

**Tier 1: kernel-internal scratch.** Auto-cleaned by the kernel at startup and at idle ticks, with no user prompt:

- `store/journal/rotated/*.zst` older than the journal-retention window (default 90 days).
- `store/workspaces/__mvd_health_probe__/` entries older than 1 h.
- Docker objects with `multiverse.health_probe=true` older than 1 h.
- Orphan `__mvd_*` MLflow experiments and Optuna studies older than 1 h.

This tier may not touch any user-visible directory. The set of paths it can touch is enumerated in code as a closed list, and CI grep-checks that no other path appears in auto-GC code.

**Tier 2: user-visible artifacts.** Only `multiverse gc` plugin invocations. Defaults:

- Dry-run is the default. The plugin prints what it *would* delete and exits zero without modifying anything unless `--apply` is passed.
- Retention policies are read from `multiverse_config.yaml`; they default to "infinite" (no retention) until the user opts in.
- Every candidate deletion is gated by:
  1. Owner token match.
  2. Retention age exceeded.
  3. Either a checksummed export exists, OR the user passed `--no-export-required`.
- A deletion report is written to `store/gc_reports/<rfc3339>.md` listing every deleted path and its manifest summary.

The GUI's "Recovered Runs" tab never auto-deletes. It can only invoke the `multiverse gc` plugin with `--apply` after explicit user confirmation, and shows the dry-run output in a panel for review.

**Acceptance.**

- A user can run for a year without any artifact ever being auto-deleted.
- `multiverse gc --apply` without retention policy configured is a no-op.
- Every Tier-1 cleanup path is enumerated in a single test, and adding a new path requires updating that test.

---

### R13 — Descriptor-based traversal and re-check at use

**Refines.** S3 step 3, S19.

**Specification.** Symlink canonicalization at validation time is insufficient because a symlink can be retargeted between the check and the use (TOCTOU). The kernel uses three reinforcing rules:

1. **Descriptor-based traversal where supported.** Where the OS exposes `openat`, `renameat`, `*at`-family syscalls (Linux, modern macOS) the kernel opens the artifact root once and performs every subsequent operation as `*at(rootfd, "relative/path", ...)`. This prevents a symlink swap from redirecting an operation outside the root.
2. **Reject symlinks within managed store paths as policy.** Inside `store/artifacts/`, `store/workspaces/`, `store/quarantine/`, and `store/journal/`, symlinks are not allowed. Discovery of a symlink during traversal halts the operation, quarantines the containing directory (R5), and reports it via `multiverse doctor`.
3. **Re-check immediately before destructive operations.** Promotion's atomic-rename step (S3 step 3) re-canonicalizes the source and destination paths *as the final pre-rename act* and aborts if the canonical paths drifted from the prepared values.

For paths *outside* the managed roots (user-supplied dataset paths, model build contexts), symlinks are allowed but every component of the resolved path is recorded in the journal so an after-the-fact audit can detect a swap.

**Acceptance.**

- A test introduces a symlink swap between PREPARE and STAGE and confirms the saga aborts cleanly with a journal entry naming the drift.
- `find store/{artifacts,workspaces,quarantine,journal} -type l` is empty under normal operation; presence of a symlink triggers an alert from the next reconciliation tick.
- Destructive and promotion-boundary filesystem operations under the kernel use descriptor-relative helpers on Linux. CI checks those boundary modules for raw path traversal or direct destructive calls outside an approved-list; harmless path formatting is allowed.

---

### R14 — Internal precise state, user-facing grouped state

**Refines.** S5, GUI.

**Specification.** The state vocabulary splits into two layers:

**Internal (precise) state** — what the kernel writes and what every test asserts on:

`PENDING`, `ADMITTED`, `RUNNING`, `TRAINING_SUCCEEDED`, `EVALUATING`, `EVALUATION_FAILED`, `PROMOTING`, `PROMOTION_FAILED`, `ARTIFACT_SUCCESS`, `CANCEL_REQUESTED`, `CANCELLED`, `FAILED`, `RECOVERY_PENDING`.

**User-facing (grouped) state** — what the GUI renders:

| GUI group | Internal states mapped | Default action surfaced |
|---|---|---|
| Running | `PENDING`, `ADMITTED`, `RUNNING`, `TRAINING_SUCCEEDED`, `EVALUATING`, `PROMOTING`, `CANCEL_REQUESTED` | Cancel |
| Succeeded | `ARTIFACT_SUCCESS` | Open in Results |
| Needs attention | `EVALUATION_FAILED`, `PROMOTION_FAILED`, `RECOVERY_PENDING` | "Re-evaluate" / "Adopt from quarantine" |
| Cancelled | `CANCELLED` | "Open workspace" |
| Failed | `FAILED` | "Open log" |

The mapping is a single table in the GUI's view layer; the API surface returns the precise state, and the GUI translates. The CLI's `multiverse list` accepts either precise or grouped names as filters.

A run with a projection failure (R6) is in its primary group with an advisory chip; it is not moved to a different group. The "MLflow sync failed" condition shows as a yellow chip on a "Succeeded" row, not as a "Needs attention" entry.

**Acceptance.**

- The mapping table is the only place GUI code reads internal state strings; introducing a new internal state requires editing the table or CI fails.
- A user survey question — "what does each chip on the Run tab mean?" — has unambiguous answers for every chip from the action-oriented vocabulary, not the FSM vocabulary.

---

### R15 — Architecture Decision Record before coding

**Refines.** all.

**Specification.** Before the first kernel commit lands, an ADR-style document `docs/adr/0001-mvd-platform-assumptions.md` is written and reviewed. It is the binding implementation reference; subsequent strategy edits go in front of it, not into the S/R bodies.

ADR sections:

1. **Supported OS matrix.**
   - Linux: glibc ≥ 2.31, kernel ≥ 5.10, systemd user-units required for daemon launch, `openat`/`renameat` available.
   - macOS: 13+ with launchd user agent; `openat` available; Docker Desktop assumed; `host.docker.internal` available.
   - Windows: deferred to a later ADR. WSL2 is treated as Linux.
2. **Filesystem assumptions.** Per supported OS, the expected `supported` level for each probe in R8. NFS is `dangerous`. exFAT/FAT32 is `blocked`. SMB is `dangerous`. ZFS/btrfs/ext4/APFS are `supported`.
3. **Daemon launch model.** Linux: systemd user unit at `~/.config/systemd/user/mvd.service`. macOS: launchd agent at `~/Library/LaunchAgents/io.multiverse.mvd.plist`. Manual `multiverse daemon start` for setups without these.
4. **Socket and API auth.** Filesystem-permission only. Socket mode `0600`. No tokens.
5. **Docker Desktop behavior.** Specific notes on Docker Desktop sleep/resume and `host.docker.internal` reliability. The kernel re-resolves the host gateway on every reconciliation tick.
6. **NVML availability.** Where present, used for VRAM accounting. Where absent, GPU jobs serialize (one at a time per GPU).
7. **Python version.** 3.12 minimum; the kernel does not vendor a runtime.
8. **Process model.** Kernel is a single process. Plugins are subprocesses. No threading inside the kernel for I/O; asyncio only.
9. **External versions pinned.** Docker SDK, mlflow, optuna, psutil — versions declared in the ADR with upgrade-trigger criteria.

The ADR is **append-only**. Changes are recorded as new ADRs (`0002-...`, `0003-...`) that supersede sections of earlier ones. Strategy edits that conflict with the ADR are blocked at review.

**Acceptance.**

- `docs/adr/0001-mvd-platform-assumptions.md` exists and is linked from this strategy document.
- A test verifies the Python version requirement and the presence of `openat` on Linux at daemon startup.
- A second ADR (`0002-...`) is required before any platform assumption can change.

---

*End of strategy.*
