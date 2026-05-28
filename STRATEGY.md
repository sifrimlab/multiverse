# multiverse Current Gaps and Next Moves

**Recheck date:** 2026-05-28  
**Scope:** Remaining work before calling the local single-user product production-grade.  
**Product boundary:** Single-user, local workstation execution. No hosted service, cluster, multi-user, or legacy-migration requirement.

Completed strategy work is intentionally excluded. This file now tracks only gaps that still need evidence or implementation.

## Current Required Implementations

### 1. Provide deterministic real-Docker OOM evidence

The real-Docker mvd suite now runs against local images without requiring Docker image builds or network pulls. On this machine, the suite produced `7 passed, 1 skipped`: happy path, non-zero exit, validation failure, cancellation, crash-after-staging recovery, SQLite rebuild, and adapter smoke coverage all ran through real Docker. The only remaining skip is OOM classification because the available local Python image exits `1` under memory pressure instead of Docker reporting `OOMKilled=true`.

Required work:

- Provide `MVD_REAL_DOCKER_OOM_IMAGE` as a local image that reliably triggers Docker `OOMKilled=true` under the test's memory limit, or add an equivalent deterministic local fixture.
- Run `pytest -q -rs tests/integration/test_mvd_real_path.py tests/integration/test_mvd_real_docker_engine.py` and confirm the OOM test no longer skips.
- Keep the OOM test optional for ordinary developer machines; make it mandatory only for release/acceptance evidence where the required local image exists.

Done when: the real-Docker suite reports all tests passing non-skipped in a prepared Docker environment, including OOM classification.

## Production-Grade Exit Criteria

| Criterion | Required evidence |
|---|---|
| GUI results persist through restart. | A GUI-launched mvd run appears in Results from the rebuildable index with artifact browsing intact. |
| SQLite is rebuildable. | Deleting `mvexp_state.db` and running `rebuild-index` restores promoted runs and classifies incomplete attempts without deleting data. |
| Real-path fault tests pass. | The optional real-Docker suite runs non-skipped in a prepared local Docker environment, including deterministic OOM evidence. |

## Immediate Next Move

Create or pre-load a deterministic OOM fixture image and set `MVD_REAL_DOCKER_OOM_IMAGE` for acceptance runs. The rest of the real Docker-backed path now has passing evidence on this workstation.
