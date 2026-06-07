"""Production ``MvdDockerExecutor`` (STRATEGY v2 §4).

Composes the safer architecture into a single ``RunExecutor`` that the
kernel can drive:

* admission through :class:`multiverse.broker.ResourceBroker`;
* container launch through :class:`multiverse.docker_supervisor.DockerSupervisor`
  with labels and a lease;
* state transitions through the kernel only — every step goes via
  :meth:`Kernel.transition`;
* evaluation represented with ``EVALUATING`` and explicit
  ``EVALUATION_FAILED``;
* promotion through :class:`multiverse.promotion.PromotionSaga`;
* success as ``ARTIFACT_SUCCESS``;
* projection sync emitted *after* the artifact-success commit so an
  MLflow outage cannot block the scientific outcome (R6).

The executor is hot-path-clean: it imports the docker_supervisor protocol
types but never the Docker SDK. Production wires a real
``RealDockerEngine``; tests wire ``InMemoryContainerEngine`` plus a
``producer`` callable that drops outputs into the workspace before the
container "exits".
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from ..contract import job_spec_payload, write_job_spec
from ..artifact import (ArtifactManifest, BootContext, ImageIdentity,
                        ModelOutputContract, ProducedAt, ProducedBy,
                        ValidationLevel, compute_logical_run_id,
                        compute_manifest_hash, compute_params_hash,
                        produced_at_now, runtime_identity_fingerprint,
                        verify_runtime_identity_matches_source)
from ..broker import ResourceBroker, ResourceRequest
from ..docker_supervisor import CancelSaga, ContainerState, DockerSupervisor
from ..journal import JournalWriter
from ..logging_utils import resolve_log_level
from ..promotion import PromotionOutcome, PromotionSaga, StoreLayout
from .runs import RunRecord
from .state import PrimaryState

ProducerHook = Callable[[Path, Mapping[str, Any]], None]
"""Optional pre-exit hook called after ``supervisor.launch`` returns. In
production this is unused (the real container drops files into the
mounted workspace by itself); tests use it to synthesize outputs."""


@dataclass
class MvdDockerExecutor:
    """``RunExecutor`` that drives one run end-to-end through the safer
    architecture.

    The executor expects each ``RunRecord``'s ``options`` to carry the
    job-specific data the kernel itself does not model — image, dataset
    path, resource request, etc. See :func:`build_executor_options` for
    the canonical shape.
    """

    journal: JournalWriter
    boot: BootContext
    store: StoreLayout
    supervisor: DockerSupervisor
    broker: ResourceBroker
    state_root: Path
    mvd_version: str = "0.1.0-mvd"
    git_commit: Optional[str] = None
    poll_interval_seconds: float = 1.0
    max_poll_iterations: int = 86400
    producer_hook: Optional[ProducerHook] = None
    """Tests pass a callable that synthesizes container outputs into the
    workspace right after launch. Production leaves this None."""

    accept_degraded: bool = False
    """Mirror of ``KernelConfig.accept_degraded``. When False (default),
    ``execute`` refuses to launch a run whose image identity is not
    strict-acceptable, matching the M2 default-fail-closed guarantee."""

    user_id: Optional[str] = None
    """Owner of the runs produced by this executor (G2). Stamped into the
    artifact manifest's ProducedBy and onto every journal record."""

    name: str = "mvd-docker"

    # ------------------------------------------------------------------
    # RunExecutor surface
    # ------------------------------------------------------------------

    async def execute(self, *, record: RunRecord, kernel) -> None:
        options = dict(record.options or {})
        try:
            spec = _ExecutorJobSpec.from_options(options, record.physical_attempt_id)
        except _BadOptions as exc:
            await kernel.transition(
                record.physical_attempt_id,
                to_state=PrimaryState.FAILED,
                reason=f"MvdDockerExecutor options error: {exc}",
            )
            return

        # Compute the logical run id and stamp it on the record so later
        # queries can group attempts of the same recipe.
        identity = _resolve_identity_from_options(spec)
        logical_run_id = logical_run_id_for_spec(spec, identity)
        record.logical_run_id = logical_run_id

        # ---- 1. ADMISSION ----
        admission = self.broker.admit(
            physical_attempt_id=record.physical_attempt_id,
            request=spec.resource_request,
        )
        if not admission.admitted:
            await kernel.transition(
                record.physical_attempt_id,
                to_state=PrimaryState.FAILED,
                reason=f"broker refused admission: {admission.detail}",
            )
            return
        run_logger: Optional[logging.Logger] = None
        orch_handler: Optional[logging.Handler] = None
        try:
            await kernel.transition(
                record.physical_attempt_id, to_state=PrimaryState.ADMITTED
            )
            if record.cancel_requested:
                await self._cancel_terminate(record, kernel, reason="pre-launch cancel")
                return

            # ---- 1a. STRICT-IMAGE GUARD (G1) ----
            if not self.accept_degraded and not identity.is_strict_acceptable:
                await kernel.transition(
                    record.physical_attempt_id,
                    to_state=PrimaryState.FAILED,
                    reason=(
                        f"refused to launch with non-strict-acceptable image "
                        f"identity {identity.kind.value!r}; set accept_degraded=True "
                        "to override"
                    ),
                )
                return

            # ---- 2. WORKSPACE ----
            workspace = self.store.workspaces / record.physical_attempt_id
            workspace.mkdir(parents=True, exist_ok=True)
            # Model images may run as container-specific UIDs. This directory
            # is per-attempt scratch owned by mvd, so make the bind mount
            # writable before launch instead of requiring every image to run as
            # the host user.
            workspace.chmod(0o777)
            record.workspace_dir = str(workspace)
            run_logger, orch_handler = self._open_run_logger(
                workspace, record.physical_attempt_id
            )
            run_logger.info(
                "admitted attempt=%s logical_run_id=%s model=%s image=%s "
                "image_identity=%s dataset=%s",
                record.physical_attempt_id,
                logical_run_id,
                spec.model_slug,
                spec.model_image,
                identity.kind.value,
                spec.dataset_slug,
            )
            self._write_job_spec(spec, workspace)

            # ---- 3. CONTAINER LAUNCH ----
            launch = self.supervisor.launch(
                physical_attempt_id=record.physical_attempt_id,
                logical_run_id=logical_run_id,
                manifest_hash=spec.manifest_hash,
                workspace=workspace,
                owner_token=record.physical_attempt_id,  # provisional; saga issues final token
                image=spec.model_image,
                command=spec.container_command,
                env=spec.container_env,
                volumes=spec.container_volumes(workspace),
                mem_limit=spec.mem_limit,
                name=f"mvd-{record.physical_attempt_id[:8]}",
                entrypoint=spec.container_entrypoint,
                gpu_requested=spec.gpu_requested,
            )
            run_logger.info(
                "container launched id=%s name=mvd-%s",
                launch.container_id,
                record.physical_attempt_id[:8],
            )
            await kernel.transition(
                record.physical_attempt_id, to_state=PrimaryState.RUNNING
            )

            # ---- 3a. (test only) producer hook synthesizes outputs ----
            if self.producer_hook is not None:
                self.producer_hook(workspace, spec.params)

            # ---- 4. WAIT FOR EXIT ----
            exit_info = await self._wait_for_exit(launch.lease, record, kernel)
            if exit_info is _CANCELLED:
                return
            # Capture the container's stdout/stderr before any cleanup so a
            # failed or non-SDK run still leaves host-side evidence.
            self._capture_container_log(launch.container_id, workspace, run_logger)
            oom = bool(exit_info.get("oom_killed", False))
            exit_code = int(exit_info.get("exit_code", 0))
            self.broker.classify_exit(
                physical_attempt_id=record.physical_attempt_id,
                exit_code=exit_code,
                oom_killed=oom,
            )
            if exit_code != 0:
                reason = (
                    "container OOM_KILLED" if oom else f"container exited {exit_code}"
                )
                run_logger.error("run failed: %s", reason)
                await kernel.transition(
                    record.physical_attempt_id,
                    to_state=PrimaryState.FAILED,
                    reason=reason,
                )
                return

            run_logger.info("container exited cleanly (exit_code=0)")
            await kernel.transition(
                record.physical_attempt_id, to_state=PrimaryState.TRAINING_SUCCEEDED
            )

            # ---- 5. EVALUATION (explicit state per R6) ----
            await kernel.transition(
                record.physical_attempt_id, to_state=PrimaryState.EVALUATING
            )

            # ---- 6. PROMOTION SAGA ----
            await kernel.transition(
                record.physical_attempt_id, to_state=PrimaryState.PROMOTING
            )
            runtime_identity = self._compose_runtime_identity(
                source=identity, container_id=launch.container_id
            )
            try:
                manifest = self._compose_manifest(
                    spec,
                    identity,
                    logical_run_id,
                    record,
                    runtime_identity=runtime_identity,
                )
            except ValueError as exc:
                await kernel.transition(
                    record.physical_attempt_id,
                    to_state=PrimaryState.FAILED,
                    reason=f"dual-digest invariant violation: {exc}",
                )
                return
            target_artifact_dir = self.store.artifacts / spec.artifact_dir_name
            saga = PromotionSaga(
                journal=self.journal,
                layout=self.store,
                physical_attempt_id=record.physical_attempt_id,
                logical_run_id=logical_run_id,
                workspace_dir=workspace,
                target_artifact_dir=target_artifact_dir,
                manifest=manifest,
                contract=ModelOutputContract.default(
                    expected_n_obs=spec.dataset_n_obs,
                    mv_contract_version=spec.contract_version,
                ),
                validators=spec.validators,
            )
            result = saga.run()
            if result.outcome is PromotionOutcome.PROMOTED:
                record.artifact_dir = str(result.artifact_dir)
                run_logger.info(
                    "promoted to artifact_dir=%s (ARTIFACT_SUCCESS)",
                    result.artifact_dir,
                )
                await kernel.transition(
                    record.physical_attempt_id, to_state=PrimaryState.ARTIFACT_SUCCESS
                )
                # ---- 7. PROJECTION STATUS (after artifact-success) ----
                await kernel.report_projection_status(
                    plugin="mlflow",
                    physical_attempt_id=record.physical_attempt_id,
                    status="TRACKING_PENDING",
                    details={"sync_required": True},
                )
                return
            if result.outcome is PromotionOutcome.QUARANTINED:
                run_logger.error(
                    "promotion quarantined: %s",
                    result.failure_reason or "promotion quarantined",
                )
                await kernel.transition(
                    record.physical_attempt_id,
                    to_state=PrimaryState.PROMOTION_FAILED,
                    reason=result.failure_reason or "promotion quarantined",
                )
                await kernel.transition(
                    record.physical_attempt_id,
                    to_state=PrimaryState.RECOVERY_PENDING,
                    reason="quarantined; user adoption required",
                )
                return
            run_logger.error(
                "promotion failed: %s",
                result.failure_reason or "promotion pre-prepare failure",
            )
            await kernel.transition(
                record.physical_attempt_id,
                to_state=PrimaryState.FAILED,
                reason=result.failure_reason or "promotion pre-prepare failure",
            )
        finally:
            self._close_run_logger(orch_handler)
            self.broker.release(record.physical_attempt_id)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _open_run_logger(
        self, workspace: Path, attempt_id: str
    ) -> "tuple[logging.Logger, logging.Handler]":
        """Open a per-attempt host-side log at ``workspace/orchestrator.log``.

        The handler is attached to a dedicated, non-propagating logger keyed
        on the attempt id so concurrent runs never bleed into each other's
        files. The level honours ``$MULTIVERSE_LOG_LEVEL``.
        """
        level = resolve_log_level()
        handler = logging.FileHandler(
            workspace / "orchestrator.log", mode="a", encoding="utf-8"
        )
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        run_logger = logging.getLogger(f"multiverse.mvd.run.{attempt_id}")
        run_logger.setLevel(level)
        run_logger.propagate = False
        # Replace any stale handler from a reattached attempt of the same id.
        for stale in run_logger.handlers[:]:
            run_logger.removeHandler(stale)
            try:
                stale.close()
            except Exception:
                pass
        run_logger.addHandler(handler)
        return run_logger, handler

    @staticmethod
    def _close_run_logger(handler: Optional[logging.Handler]) -> None:
        if handler is None:
            return
        logging.getLogger().removeHandler(handler)
        for logger in logging.Logger.manager.loggerDict.values():
            if isinstance(logger, logging.Logger) and handler in logger.handlers:
                logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    def _capture_container_log(
        self,
        container_id: str,
        workspace: Path,
        run_logger: Optional[logging.Logger] = None,
    ) -> None:
        """Best-effort host capture of container stdout/stderr.

        Never raises: a logging failure must not fail an otherwise-successful
        run. Engines that do not implement ``logs`` are silently skipped.
        """
        engine = getattr(self.supervisor, "engine", None)
        logs_fn = getattr(engine, "logs", None)
        if not callable(logs_fn):
            return
        try:
            raw = logs_fn(container_id) or b""
        except Exception as exc:
            if run_logger is not None:
                run_logger.warning("could not capture container logs: %s", exc)
            return
        if isinstance(raw, str):
            raw = raw.encode("utf-8", errors="replace")
        try:
            (workspace / "container.log").write_bytes(raw)
        except OSError as exc:
            if run_logger is not None:
                run_logger.warning("could not write container.log: %s", exc)

    async def _wait_for_exit(self, lease, record: RunRecord, kernel) -> Any:
        """Poll the supervisor until the container exits or cancellation
        is requested. Returns a dict with ``exit_code`` + ``oom_killed``
        on natural exit, or the sentinel :data:`_CANCELLED` on cancel."""
        for _ in range(self.max_poll_iterations):
            if record.cancel_requested:
                await self._cancel_terminate(
                    record, kernel, lease=lease, reason="user cancel"
                )
                return _CANCELLED
            entry = self.supervisor.reconcile_one(lease)
            if entry.state is ContainerState.RUNNING:
                await asyncio.sleep(self.poll_interval_seconds)
                continue
            return {
                "exit_code": entry.exit_code if entry.exit_code is not None else 1,
                "oom_killed": bool(entry.oom_killed),
            }
        await kernel.transition(
            record.physical_attempt_id,
            to_state=PrimaryState.FAILED,
            reason=f"container did not exit within "
            f"{self.max_poll_iterations * self.poll_interval_seconds}s",
        )
        return _CANCELLED

    async def _cancel_terminate(
        self,
        record: RunRecord,
        kernel,
        *,
        lease: Optional[Any] = None,
        reason: str = "cancel",
    ) -> None:
        """Drive the cancellation saga and the kernel state transitions."""
        await kernel.transition(
            record.physical_attempt_id, to_state=PrimaryState.CANCEL_REQUESTED
        )
        if lease is not None:
            saga = CancelSaga(
                engine=self.supervisor.engine,
                journal=self.journal,
                layout=self.store,
                boot=self.boot,
                physical_attempt_id=record.physical_attempt_id,
                logical_run_id=record.logical_run_id or "",
                lease=lease,
            )
            saga.run()
        await kernel.transition(
            record.physical_attempt_id,
            to_state=PrimaryState.CANCELLED,
            reason=reason,
        )

    def _write_job_spec(self, spec: "_ExecutorJobSpec", workspace: Path) -> None:
        payload = job_spec_payload(
            model_name=spec.model_slug,
            model_version=spec.model_version,
            dataset_slug=spec.dataset_slug,
            hyperparameters={spec.model_slug: dict(spec.params)},
            seed=spec.seed,
            batch_key=spec.batch_key,
            cell_type_key=spec.cell_type_key,
            preprocessing=dict(spec.preprocessing) if spec.preprocessing else None,
        )
        write_job_spec(workspace / "job_spec.json", payload)

    def _compose_manifest(
        self,
        spec: "_ExecutorJobSpec",
        identity: ImageIdentity,
        logical_run_id: str,
        record: RunRecord,
        runtime_identity: Optional[ImageIdentity] = None,
    ) -> ArtifactManifest:
        # Dual-digest invariant (STRATEGY M2). When the engine reported a
        # runtime SIF, the SIF's built_from MUST equal image_identity.value.
        # Raises ValueError on mismatch; the caller catches and FAILs the
        # run rather than promoting a non-comparable artifact.
        verify_runtime_identity_matches_source(identity, runtime_identity)
        return ArtifactManifest(
            logical_run_id=logical_run_id,
            physical_attempt_id=record.physical_attempt_id,
            manifest_hash=spec.manifest_hash,
            dataset_fingerprint=spec.dataset_fingerprint,
            image_identity=identity,
            params_hash=compute_params_hash(spec.params),
            mv_contract_version=spec.contract_version,
            produced_at=ProducedAt.from_dict(produced_at_now(self.boot)),
            produced_by=ProducedBy(
                mvd_version=self.mvd_version,
                git_commit=self.git_commit,
                user_id=self.user_id,
            ),
            artifacts=[],
            owner_token=record.physical_attempt_id,
            runtime_image_identity=runtime_identity,
        )

    def _compose_runtime_identity(
        self,
        *,
        source: ImageIdentity,
        container_id: str,
    ) -> Optional[ImageIdentity]:
        """Ask the engine whether it produced a runtime-derived identity
        (e.g. an Apptainer SIF) for the just-launched container. Returns
        ``None`` for engines that run the source image directly (Docker).
        """
        engine = getattr(self.supervisor, "engine", None)
        if engine is None:
            return None
        sif_lookup = getattr(engine, "sif_digest_for", None)
        if not callable(sif_lookup):
            return None
        sif = sif_lookup(container_id)
        if not sif:
            return None
        # Use the source digest as built_from when the source is itself a
        # content-addressed identity; otherwise built_from stays None and
        # `is_strict_acceptable` will reject the result under strict mode.
        built_from: Optional[str] = None
        if source.kind.value in {"registry_digest", "build_context_hash"}:
            built_from = source.value
        built_by = (
            "apptainer-pull-runtime"
            if getattr(engine, "name", "").startswith("apptainer")
            else None
        )
        return ImageIdentity.sif_digest(sif, built_from=built_from, built_by=built_by)


# Sentinel returned by ``_wait_for_exit`` when execution was diverted to
# the cancel branch — distinct from any natural-exit dict.
_CANCELLED = object()


# ---------------------------------------------------------------------------
# options shape
# ---------------------------------------------------------------------------


class _BadOptions(ValueError):
    pass


@dataclass(frozen=True)
class _ExecutorJobSpec:
    physical_attempt_id: str
    model_slug: str
    model_version: str
    model_image: str
    image_digest: Optional[str]
    contract_version: str
    dataset_slug: str
    dataset_path: str
    dataset_n_obs: int
    dataset_n_vars: Optional[int]
    dataset_fingerprint: Dict[str, Any]
    params: Dict[str, Any]
    manifest_hash: str
    resource_request: ResourceRequest
    artifact_dir_name: str
    mem_limit: Optional[str]
    gpu_requested: bool
    preprocessing: Optional[Dict[str, Any]]
    container_command: Optional[List[str]]
    container_entrypoint: Optional[str]
    batch_key: Optional[str]
    cell_type_key: Optional[str]
    validators: ValidationLevel
    seed: Optional[int]
    env_extra: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_options(
        cls, options: Mapping[str, Any], attempt_id: str
    ) -> "_ExecutorJobSpec":
        def _req(key: str) -> Any:
            if key not in options:
                raise _BadOptions(f"missing option {key!r}")
            return options[key]

        params = dict(options.get("params") or {})
        fingerprint = {
            "slug": str(_req("dataset_slug")),
            "n_obs": int(_req("dataset_n_obs")),
        }
        if options.get("dataset_n_vars") is not None:
            fingerprint["n_vars"] = int(options["dataset_n_vars"])
        if options.get("dataset_fingerprint_extra"):
            fingerprint.update(dict(options["dataset_fingerprint_extra"]))

        manifest_hash = str(
            options.get("manifest_hash")
            or compute_manifest_hash(str(options.get("manifest_text") or ""))
        )
        ram_bytes = int(options.get("ram_request_bytes") or 256 * 1024 * 1024)
        request = ResourceRequest(ram_bytes=ram_bytes)

        validators_raw = str(options.get("validators", "basic")).lower()
        try:
            validators = ValidationLevel(validators_raw)
        except ValueError as exc:
            raise _BadOptions(
                f"validators must be one of basic/strict/developer, got {validators_raw!r}"
            ) from exc

        artifact_dir_name = str(
            options.get("artifact_dir_name")
            or f"{fingerprint['slug']}_{str(_req('model_slug'))}_{attempt_id[:8]}"
        )

        env_extra_raw = options.get("container_env_extra") or {}
        env_extra = {str(k): str(v) for k, v in dict(env_extra_raw).items()}

        return cls(
            physical_attempt_id=attempt_id,
            model_slug=str(_req("model_slug")),
            model_version=str(options.get("model_version", "0.0.0")),
            model_image=str(_req("model_image")),
            image_digest=options.get("image_digest"),
            contract_version=str(options.get("contract_version", "1")),
            dataset_slug=str(options["dataset_slug"]),
            batch_key=(str(options["batch_key"]) if options.get("batch_key") else None),
            cell_type_key=(
                str(options["cell_type_key"]) if options.get("cell_type_key") else None
            ),
            dataset_path=str(_req("dataset_path")),
            dataset_n_obs=int(_req("dataset_n_obs")),
            dataset_n_vars=(
                int(options["dataset_n_vars"])
                if options.get("dataset_n_vars") is not None
                else None
            ),
            dataset_fingerprint=fingerprint,
            params=params,
            manifest_hash=manifest_hash,
            resource_request=request,
            artifact_dir_name=artifact_dir_name,
            mem_limit=options.get("mem_limit"),
            gpu_requested=bool(options.get("gpu_requested", False)),
            preprocessing=(
                dict(options["preprocessing"]) if options.get("preprocessing") else None
            ),
            container_command=(
                [str(part) for part in options["container_command"]]
                if options.get("container_command") is not None
                else None
            ),
            container_entrypoint=(
                str(options["container_entrypoint"])
                if options.get("container_entrypoint") is not None
                else None
            ),
            validators=validators,
            seed=(int(options["seed"]) if options.get("seed") is not None else None),
            env_extra=env_extra,
        )

    @property
    def container_env(self) -> Dict[str, str]:
        base = {
            "MVR_INPUT_DATA_PATH": "/input/data.h5mu",
            "MVR_OUTPUT_DIR": "/output",
            "MVR_JOB_SPEC_PATH": "/output/job_spec.json",
        }
        # Forward the host log level so the worker SDK (run.log) matches the
        # orchestrator's verbosity without a separate knob.
        host_level = os.environ.get("MULTIVERSE_LOG_LEVEL")
        if host_level:
            base["MULTIVERSE_LOG_LEVEL"] = str(host_level)
        base.update(self.env_extra)
        return base

    def container_volumes(self, workspace: Path) -> Dict[str, str]:
        return {
            str(Path(self.dataset_path).expanduser().resolve()): "/input/data.h5mu",
            str(workspace.resolve()): "/output",
        }


def _resolve_identity_from_options(spec: _ExecutorJobSpec) -> ImageIdentity:
    if spec.image_digest:
        return ImageIdentity.registry_digest(spec.image_digest)
    return ImageIdentity.unverified_local(spec.model_image)


def logical_run_id_for_spec(spec: _ExecutorJobSpec, identity: ImageIdentity) -> str:
    """Canonical logical-run-ID for a docker job spec (STRATEGY: one identity).

    Folds the behaviour-affecting runtime fields that live outside
    ``model_params`` — seed, preprocessing, model version — into the identity
    via ``runtime_fingerprint`` so that editing any of them produces a new
    runnable logical run. The resume planner reuses this exact function so a
    planned job and the executed attempt that completed it hash identically.
    """
    return compute_logical_run_id(
        manifest_hash=spec.manifest_hash,
        dataset_fingerprint=spec.dataset_fingerprint,
        image_identity=identity,
        params_hash=compute_params_hash(spec.params),
        mv_contract_version=spec.contract_version,
        runtime_fingerprint=runtime_identity_fingerprint(
            seed=spec.seed,
            preprocessing=spec.preprocessing,
            model_version=spec.model_version,
        ),
    )


# ---------------------------------------------------------------------------
# Convenience for callers building ``submit_run`` options
# ---------------------------------------------------------------------------


def build_executor_options(
    *,
    model_slug: str,
    model_image: str,
    dataset_slug: str,
    dataset_path: str,
    dataset_n_obs: int,
    params: Optional[Mapping[str, Any]] = None,
    image_digest: Optional[str] = None,
    model_version: str = "0.0.0",
    contract_version: str = "1",
    dataset_n_vars: Optional[int] = None,
    manifest_text: str = "",
    ram_request_bytes: int = 256 * 1024 * 1024,
    mem_limit: Optional[str] = None,
    gpu_requested: bool = False,
    preprocessing: Optional[Mapping[str, Any]] = None,
    container_command: Optional[List[str]] = None,
    container_entrypoint: Optional[str] = None,
    validators: str = "basic",
    batch_key: Optional[str] = None,
    cell_type_key: Optional[str] = None,
    artifact_dir_name: Optional[str] = None,
    seed: Optional[int] = None,
    container_env_extra: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Build a canonical ``options`` dict for ``Kernel.submit_run``.

    Production callers (the CLI cutover in Move 6) build this from a YAML
    manifest; tests build it inline.
    """
    out: Dict[str, Any] = {
        "model_slug": model_slug,
        "model_image": model_image,
        "model_version": model_version,
        "contract_version": contract_version,
        "dataset_slug": dataset_slug,
        "dataset_path": dataset_path,
        "dataset_n_obs": int(dataset_n_obs),
        "params": dict(params or {}),
        "manifest_text": manifest_text,
        "ram_request_bytes": int(ram_request_bytes),
        "validators": validators,
    }
    if seed is not None:
        out["seed"] = int(seed)
    if image_digest:
        out["image_digest"] = image_digest
    if dataset_n_vars is not None:
        out["dataset_n_vars"] = int(dataset_n_vars)
    if mem_limit:
        out["mem_limit"] = mem_limit
    if gpu_requested:
        out["gpu_requested"] = True
    if preprocessing:
        out["preprocessing"] = dict(preprocessing)
    if container_command is not None:
        out["container_command"] = [str(part) for part in container_command]
    if container_entrypoint is not None:
        out["container_entrypoint"] = str(container_entrypoint)
    if artifact_dir_name:
        out["artifact_dir_name"] = artifact_dir_name
    if container_env_extra:
        out["container_env_extra"] = {
            str(k): str(v) for k, v in dict(container_env_extra).items()
        }
    if batch_key:
        out["batch_key"] = str(batch_key)
    if cell_type_key:
        out["cell_type_key"] = str(cell_type_key)
        
    return out
