# ADR 0001 — mvd Platform Assumptions

**Status:** Accepted (initial version)
**Date:** 2026-05-27
**Supersedes:** none
**Related strategy:** [STRATEGY.md §R15](../../STRATEGY.md)

This ADR is the binding implementation reference for the `mvd` local-job-daemon
work. The strategy document describes the *intent* of the platform; this ADR
freezes the platform assumptions that the kernel, plugins, and artifact
contract may rely on. If a strategy edit conflicts with this ADR, the ADR
wins until superseded by a later `0002-…` ADR.

The ADR is **append-only**. Mutations are recorded as new ADRs that supersede
specific sections; the file you are reading is not edited in place after
acceptance.

---

## 1. Supported OS matrix

| OS | Minimum | Notes |
|---|---|---|
| Linux | glibc ≥ 2.31, kernel ≥ 5.10 | Primary target. `openat`/`renameat` family required (used by R13). systemd user units required for daemon launch (R15 §3). |
| macOS | 13 (Ventura) | `openat` available; Docker Desktop assumed; `host.docker.internal` available. launchd user agent for daemon launch. |
| Windows | deferred | WSL2 is treated as Linux. Native Win32 platform is out of scope for ADR 0001. |

The kernel performs an OS check at startup and refuses to run on unsupported
platforms.

## 2. Filesystem assumptions

The strategy's storage probes (R8) classify per-probe behaviour as
`supported`, `degraded`, `dangerous`, or `blocked`. This ADR fixes the
expected baseline level per filesystem family.

| Filesystem | Baseline | Reason |
|---|---|---|
| ext4 | `supported` | POSIX rename, fsync, fsync-dir, openat all work as expected. |
| xfs | `supported` | Same as ext4. |
| btrfs | `supported` | Same as ext4. |
| zfs (ZoL/OpenZFS) | `supported` | Same as ext4. |
| APFS | `supported` | macOS default. Case sensitivity is configurable; the case-sensitivity probe is authoritative. |
| HFS+ | `degraded` | Discouraged; case-insensitivity collisions possible on rename. |
| NFSv4 | `dangerous` | Distributed locking, fsync semantics, and rename atomicity vary across servers. Requires `--accept-degraded`. |
| SMB / CIFS | `dangerous` | Same as NFS plus Windows-style locking. |
| Dropbox / OneDrive / Google Drive synced paths | `dangerous` | Cloud-sync markers (`.dropbox`, `.tmp.driveupload`) are heuristically detected and produce `dangerous`. |
| exFAT / FAT32 | `blocked` | No symlink support, no POSIX permissions, no atomic rename across all OS kernels. |

`blocked` causes daemon startup to fail. `dangerous` causes startup failure
unless `--accept-degraded` is passed on the command line; that flag is then
stamped into the journal of every run admitted in the session.

## 3. Daemon launch model

| OS | Launch unit | Path |
|---|---|---|
| Linux | systemd user unit | `~/.config/systemd/user/mvd.service` |
| macOS | launchd user agent | `~/Library/LaunchAgents/io.multiverse.mvd.plist` |
| Manual (fallback) | foreground/background command | `multiverse daemon start` |

The user installs the unit/agent via `multiverse daemon install`. The kernel
never installs itself as a system-wide service; single-user, single-session is
the supported deployment model.

## 4. Socket and API auth

- Transport: Unix domain socket. Default path:
  `${MULTIVERSE_STATE_ROOT}/mvd.sock` (default
  `~/.local/state/multiverse/mvd.sock`).
- Socket mode: `0600`. Owner: the user running `mvd`.
- Filesystem permissions are the only auth boundary; no tokens, no TLS.
- HTTP-over-loopback is permitted only when explicitly enabled in config. In
  that mode a random per-session bearer token is required plus strict
  browser-origin checks. Localhost alone is not an auth boundary.

This is the *single-user* threat model. Multi-user access control is
explicitly out of scope; see STRATEGY.md "Explicit Out-of-Scope".

## 5. Docker Desktop behaviour

The kernel runs against either a native Docker engine (Linux) or Docker
Desktop (macOS, WSL2). Notes specific to Docker Desktop:

- The Docker daemon may sleep when the host suspends. On wake, the kernel
  re-resolves the host gateway (`host.docker.internal`) on the next
  reconciliation tick rather than caching it across the suspend.
- Docker Desktop's VM-backed storage means a path on the host is not
  automatically visible inside containers; the `docker_mount_visibility`
  probe (R8) is required at startup.
- The kernel does not depend on Docker Compose features.

## 6. NVML availability

GPU VRAM accounting uses NVML (via `pynvml`) when available. If NVML is not
installed or no NVIDIA GPU is present:

- The resource broker (R11) does not admit GPU jobs in parallel; GPU jobs
  serialize one-at-a-time per device index.
- `multiverse doctor` reports `GPU accounting: degraded (no NVML)`.

NVML is not a hard requirement for the kernel; it is a hard requirement for
parallel GPU admission.

## 7. Python version and runtime

- Python 3.12 minimum (matches `pyproject.toml`).
- The kernel does not vendor a Python runtime. Distribution model is a normal
  PyPI package installed into the user's environment (`pip install
  multi-verse`).
- The kernel imports a strict subset of the existing `multiverse` package; the
  import graph is checked in CI (see §8 and STRATEGY R1).

## 8. Process model

- The kernel is a single OS process.
- The kernel uses `asyncio` exclusively for concurrency. No threading inside
  the kernel for I/O. The only permitted thread is the database writer
  thread that fronts SQLite for the *rebuildable index* (Milestone 8).
- Plugins run as separate processes (subprocess or long-running daemon) and
  communicate with the kernel through the Unix socket and the artifact
  filesystem. Plugins never share writable state with the kernel.
- Hot-path modules (`mvd/kernel/**`) are forbidden from importing MLflow,
  Optuna, Streamlit, the GC scheduler, or the exporter. A CI grep gate
  enforces this.

## 9. External versions pinned

Production minimums (upgrade requires a follow-up ADR):

| Dependency | Minimum | Upgrade trigger |
|---|---|---|
| `docker` (SDK) | 6.1.3 | Docker Engine API ≥ 1.43 features used. |
| `mlflow` | 2.16.0 | Tracking server backend compatibility. |
| `optuna` | 3.x | RDB storage compatibility. |
| `psutil` | 5.9.0 | Cross-platform RSS/VRAM polling. |
| `pydantic` | 2.x | Used for manifest and contract validation. |
| `pyyaml` | any 6.x | Manifest parsing. |

When a dependency's pinned floor must change, a new ADR is required.

## 10. Journal and storage layout

The kernel writes durability records under `${MULTIVERSE_STATE_ROOT}` (see
STRATEGY R3). The default root is `~/.local/state/multiverse/` on Linux and
`~/Library/Application Support/multiverse/` on macOS. The store root
(`store/artifacts/`, `store/workspaces/`, …) is independent of the state
root and is configured via `multiverse.config.yaml`.

Within the state root:

```
mvd.sock                       Unix domain socket
journal/current.log            active append-only segment
journal/rotated/*.zst          rotated segments
journal/blobs/<sha256>.json    content-addressed spill for large manifests
journal/checkpoint.json        last fully reconciled offset
```

## 11. Time, identity, and durability

These three are surfaces every later milestone consumes. They are fixed here.

- **Logical run ID** = `sha256(manifest_hash || dataset_fingerprint ||
  image_identity_value || params_hash || mv_contract_version)`. Stable across
  attempts of the same recipe.
- **Physical attempt ID** = UUIDv4 per concrete execution.
- **`mvd_boot_id`** = UUID generated at daemon start. Stamped on every
  journal record and every artifact manifest. Used to disambiguate monotonic
  counters between daemon lifetimes.
- **`monotonic_ns`** = `time.monotonic_ns()` from the kernel process. Used
  for ordering within a boot.
- **`wall_iso`** = ISO 8601 with explicit timezone (`datetime.now(tz=…)`
  with the host's IANA tz; UTC fallback if the local tz cannot be resolved).
- **Durability boundary** = file data `fsync` plus parent directory `fsync`
  when creating a new segment or completing a rename. API calls that return
  a durable identifier may not acknowledge until their journal record has
  passed this boundary (R3).

## 12. Verification at startup

The kernel performs the following at boot before accepting any API request:

1. OS and Python version check.
2. `openat`/`renameat` availability check on Linux.
3. Storage capability probes (R8). Refuse to start on `blocked`.
4. `mvd_boot_id` generation.
5. Journal replay (when journal exists).
6. Docker daemon reachability check (`docker.ping()` with a short timeout).
7. SQLite index open (read-only at first; writes only after replay completes).

A failure in any of steps 1–3 prints a single-line cause and exits non-zero.
Steps 4–7 are logged structured.

## 13. Versioning of the ADR itself

This document is `ADR 0001`. Future ADRs:

- `0002` — *required* before any of §1, §2, §3, §4, §5, §8, or §9 change.
- `0003+` — additional decisions (logging, error taxonomy, etc.).

A CI check verifies that this file exists and contains the required section
headers; the gate fails if either is missing.

---

*End of ADR 0001.*
