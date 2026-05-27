# multiverse Current Gaps and Next Moves

**Recheck date:** 2026-05-27  
**Scope:** Remaining work before calling the local single-user product production-grade.  
**Product boundary:** Single-user, local workstation execution. No hosted service, cluster, multi-user, or legacy-migration requirement.

Completed strategy work is intentionally excluded. This file now tracks only gaps that still need evidence or implementation.

## Current Required Implementations

### 1. Make the real-Docker mvd integration suite non-skipping in the target environment

The code now includes optional real-Docker end-to-end coverage in `tests/integration/test_mvd_real_path.py`, but it skips unless the machine already has a local shell base image. This is correct for normal developer runs because tests must not pull from the network implicitly, but it means production-grade evidence still requires a prepared Docker test environment.

Required work:

- Preload `busybox:latest` or `alpine:latest`, or set `MVD_REAL_DOCKER_BASE_IMAGE` to a local shell image with `sh` and `cp`.
- Run `pytest -q tests/integration/test_mvd_real_path.py tests/integration/test_mvd_real_docker_engine.py` on that prepared machine.
- Add or provide `MVD_REAL_DOCKER_OOM_IMAGE` for deterministic OOM classification coverage, or replace the explicit image requirement with a reliable locally built OOM fixture.
- Keep these tests optional for default local runs, but mandatory in the release/acceptance checklist.

Done when: the real path, not only fake-engine unit tests or adapter smoke tests, passes happy path, non-zero exit, validation failure, cancellation, crash-after-staging recovery, SQLite rebuild, and OOM/OOM-like classification in a prepared Docker environment.

## Production-Grade Exit Criteria

| Criterion | Required evidence |
|---|---|
| GUI results persist through restart. | A GUI-launched mvd run appears in Results from the rebuildable index with artifact browsing intact. |
| SQLite is rebuildable. | Deleting `mvexp_state.db` and running `rebuild-index` restores promoted runs and classifies incomplete attempts without deleting data. |
| Real-path fault tests pass. | The optional real-Docker suite is run non-skipped in a prepared local Docker environment. |

## Immediate Next Move

Prepare the local/CI Docker environment with the required base image and run the real-Docker suite non-skipped. The implementation hooks are now present; the remaining gap is hard evidence from the actual Docker-backed path.
