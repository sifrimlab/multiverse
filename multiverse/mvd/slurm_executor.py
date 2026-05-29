"""``SlurmRunExecutor`` (STRATEGY M4 Mode A).

Sibling of :class:`MvdDockerExecutor`. Each run is one ``sbatch``
invocation; the executor polls ``sacct`` for completion, classifies the
terminal state, then drives the same promotion saga the Docker path
uses.

Design notes
============

* **No Docker types in the import graph.** The Slurm executor depends
  on the journal, broker, promotion, artifact, and slurm packages —
  nothing in :mod:`multiverse.docker_supervisor`.

* **Image acquisition is the engine's problem.** The executor expects
  the manifest to either point at a pre-built SIF (preferred) or at an
  OCI reference that the engine resolves at submit time. The
  ``runtime_image_identity`` plumbed into the artifact manifest follows
  the M2 dual-digest invariant.

* **Broker is a dispatch rate-limiter, not a RAM gate.** Wire the
  broker with ``max_inflight_dispatches=...``. The executor still calls
  ``broker.admit`` / ``broker.release`` so the journaled-admission-
  ledger contract (M3) stays intact under Slurm.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..artifact import (
    ArtifactManifest,
    BootContext,
    ImageIdentity,
    ModelOutputContract,
    ProducedAt,
    ProducedBy,
    ValidationLevel,
    compute_logical_run_id,
    compute_manifest_hash,
    compute_params_hash,
    produced_at_now,
    verify_runtime_identity_matches_source,
)
from ..broker import ResourceBroker, ResourceRequest
from ..journal import JournalKind, JournalWriter
from ..promotion import (
    PromotionOutcome,
    PromotionSaga,
    StoreLayout,
)
from ..slurm import (
    SlurmEngine,
    SlurmEngineError,
    SlurmJobSpec,
    SlurmJobState,
)
from .runs import RunRecord
from .state import PrimaryState


@dataclass
class MvdSlurmExecutor:
    """``RunExecutor`` that drives one run end-to-end via Slurm.

    Production wires :class:`~multiverse.slurm.RealSlurmEngine`; tests
    wire :class:`~multiverse.slurm.InMemorySlurmEngine`.
    """

    journal: JournalWriter
    boot: BootContext
    store: StoreLayout
    engine: SlurmEngine
    broker: ResourceBroker
    state_root: Path
    mvd_version: str = "0.1.0-mvd"
    git_commit: Optional[str] = None
    poll_interval_seconds: float = 5.0
    max_poll_iterations: int = 17280  # 24h @ 5s
    accept_degraded: bool = False
    """Mirror of ``KernelConfig.accept_degraded``. When False (default),
    ``execute`` refuses to launch a run whose image identity is not
    strict-acceptable, matching the M2 default-fail-closed guarantee."""

    user_id: Optional[str] = None
    """Owner of the runs produced by this executor (G2). Stamped into the
    artifact manifest's ProducedBy and onto every journal record."""

    name: str = "mvd-slurm"
    job_script_dir_name: str = "slurm-scripts"

    # ------------------------------------------------------------------
    # RunExecutor surface
    # ------------------------------------------------------------------

    async def execute(self, *, record: RunRecord, kernel) -> None:
        options = dict(record.options or {})
        try:
            spec = _SlurmJobSpec.from_options(options, record.physical_attempt_id)
        except _BadOptions as exc:
            await kernel.transition(
                record.physical_attempt_id,
                to_state=PrimaryState.FAILED,
                reason=f"MvdSlurmExecutor options error: {exc}",
            )
            return

        identity = _resolve_identity_from_options(spec)
        logical_run_id = compute_logical_run_id(
            manifest_hash=spec.manifest_hash,
            dataset_fingerprint=spec.dataset_fingerprint,
            image_identity=identity,
            params_hash=compute_params_hash(spec.params),
            mv_contract_version=spec.contract_version,
        )
        record.logical_run_id = logical_run_id

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

        submission = None
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

            workspace = self.store.workspaces / record.physical_attempt_id
            workspace.mkdir(parents=True, exist_ok=True)
            workspace.chmod(0o777)
            record.workspace_dir = str(workspace)

            job_spec = SlurmJobSpec(
                job_name=f"mvd-{record.physical_attempt_id[:8]}",
                image_sif=spec.image_sif,
                workspace=workspace,
                dataset_path=Path(spec.dataset_path).expanduser().resolve(),
                command=spec.container_command or [],
                env=spec.container_env,
                partition=spec.partition,
                account=spec.account,
                qos=spec.qos,
                time_minutes=spec.time_minutes,
                mem_gb=spec.mem_gb,
                cpus_per_task=spec.cpus_per_task,
                gpus=spec.gpus,
                extra_directives=spec.extra_directives,
                output_log=workspace / "slurm.out",
                error_log=workspace / "slurm.err",
            )

            try:
                submission = self.engine.submit(
                    job_spec,
                    script_dir=self.state_root / self.job_script_dir_name,
                )
            except SlurmEngineError as exc:
                await kernel.transition(
                    record.physical_attempt_id,
                    to_state=PrimaryState.FAILED,
                    reason=f"sbatch failed: {exc}",
                )
                return

            self._record_dispatch(record.physical_attempt_id, submission.job_id)
            await kernel.transition(
                record.physical_attempt_id, to_state=PrimaryState.RUNNING
            )

            terminal = await self._poll_until_terminal(
                submission.job_id, record, kernel
            )
            if terminal is None:
                return  # cancelled or timed-out polling

            if terminal.state is not SlurmJobState.COMPLETED:
                await kernel.transition(
                    record.physical_attempt_id,
                    to_state=PrimaryState.FAILED,
                    reason=_failure_reason(terminal),
                )
                return

            await kernel.transition(
                record.physical_attempt_id, to_state=PrimaryState.TRAINING_SUCCEEDED
            )
            await kernel.transition(
                record.physical_attempt_id, to_state=PrimaryState.EVALUATING
            )
            await kernel.transition(
                record.physical_attempt_id, to_state=PrimaryState.PROMOTING
            )

            runtime_identity = self._compose_runtime_identity(
                source=identity, job_spec=job_spec
            )
            try:
                manifest = self._compose_manifest(
                    spec, identity, logical_run_id, record, runtime_identity
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
                await kernel.transition(
                    record.physical_attempt_id,
                    to_state=PrimaryState.ARTIFACT_SUCCESS,
                )
                await kernel.report_projection_status(
                    plugin="mlflow",
                    physical_attempt_id=record.physical_attempt_id,
                    status="TRACKING_PENDING",
                    details={"sync_required": True},
                )
                return
            if result.outcome is PromotionOutcome.QUARANTINED:
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
            await kernel.transition(
                record.physical_attempt_id,
                to_state=PrimaryState.FAILED,
                reason=result.failure_reason or "promotion pre-prepare failure",
            )
        finally:
            self.broker.release(record.physical_attempt_id, reason="terminal")

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _poll_until_terminal(
        self,
        job_id: str,
        record: RunRecord,
        kernel,
    ):
        for _ in range(self.max_poll_iterations):
            if record.cancel_requested:
                await self._cancel_terminate(
                    record, kernel, job_id=job_id, reason="user cancel"
                )
                return None
            try:
                info = self.engine.query(job_id)
            except SlurmEngineError as exc:
                await kernel.transition(
                    record.physical_attempt_id,
                    to_state=PrimaryState.FAILED,
                    reason=f"sacct query failed: {exc}",
                )
                return None
            if info.is_terminal:
                return info
            await asyncio.sleep(self.poll_interval_seconds)
        await kernel.transition(
            record.physical_attempt_id,
            to_state=PrimaryState.FAILED,
            reason=(
                f"slurm job {job_id} did not reach a terminal state within "
                f"{self.max_poll_iterations * self.poll_interval_seconds}s"
            ),
        )
        return None

    async def _cancel_terminate(
        self,
        record: RunRecord,
        kernel,
        *,
        job_id: Optional[str] = None,
        reason: str = "cancel",
    ) -> None:
        await kernel.transition(
            record.physical_attempt_id, to_state=PrimaryState.CANCEL_REQUESTED
        )
        if job_id is not None:
            try:
                self.engine.cancel(job_id)
            except SlurmEngineError:
                pass
        await kernel.transition(
            record.physical_attempt_id,
            to_state=PrimaryState.CANCELLED,
            reason=reason,
        )

    def _record_dispatch(self, attempt_id: str, job_id: str) -> None:
        """Pin the Slurm job id into the journal so crash recovery can
        rediscover an in-flight ``sbatch`` without scraping a side
        index. Re-uses ``CONTAINER_LAUNCH`` since the journal kind is a
        generic 'we just dispatched a unit of work' record."""
        self.journal.append(
            JournalKind.CONTAINER_LAUNCH,
            payload={
                "engine": self.engine.name,
                "slurm_job_id": job_id,
            },
            physical_attempt_id=attempt_id,
        )
        self.journal.commit()

    def _compose_manifest(
        self,
        spec: "_SlurmJobSpec",
        identity: ImageIdentity,
        logical_run_id: str,
        record: RunRecord,
        runtime_identity: Optional[ImageIdentity],
    ) -> ArtifactManifest:
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
        self, *, source: ImageIdentity, job_spec: SlurmJobSpec
    ) -> Optional[ImageIdentity]:
        """Surface a SIF identity when the engine can compute one.

        Requires a strict-acceptable source kind (registry_digest or
        build_context_hash) so the dual-digest invariant can be anchored.
        Returns None for unverified_local sources — no valid built_from
        would be available, and verify_runtime_identity_matches_source
        would raise.
        """
        if source.kind.value not in {"registry_digest", "build_context_hash"}:
            return None
        sif = self.engine.sif_digest_for_submission(job_spec)
        if not sif:
            return None
        return ImageIdentity.sif_digest(
            sif, built_from=source.value, built_by="slurm-apptainer"
        )


# ---------------------------------------------------------------------------
# job spec
# ---------------------------------------------------------------------------


class _BadOptions(ValueError):
    pass


@dataclass(frozen=True)
class _SlurmJobSpec:
    physical_attempt_id: str
    model_slug: str
    model_version: str
    image_sif: Path
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
    container_command: Optional[List[str]]
    validators: ValidationLevel
    seed: Optional[int]
    env_extra: Dict[str, str] = field(default_factory=dict)
    # Slurm-specific knobs.
    partition: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    time_minutes: Optional[int] = None
    mem_gb: Optional[int] = None
    cpus_per_task: int = 1
    gpus: Optional[int] = None
    extra_directives: List[str] = field(default_factory=list)

    @classmethod
    def from_options(
        cls, options: Mapping[str, Any], attempt_id: str
    ) -> "_SlurmJobSpec":
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

        # M4 §2: broker is a rate-limiter, not a RAM gate. Resource
        # math is Slurm's job. Reserve only the conceptual slot so the
        # ledger counts dispatches; the actual mem/cpu directives go
        # into the sbatch script.
        request = ResourceRequest(ram_bytes=1)

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

        slurm_block = dict(options.get("slurm") or {})
        extra_directives = [str(x) for x in slurm_block.get("extra_directives") or []]

        image_sif = options.get("image_sif")
        if not image_sif:
            raise _BadOptions("missing option 'image_sif' (path to a SIF)")

        return cls(
            physical_attempt_id=attempt_id,
            model_slug=str(_req("model_slug")),
            model_version=str(options.get("model_version", "0.0.0")),
            image_sif=Path(str(image_sif)),
            image_digest=options.get("image_digest"),
            contract_version=str(options.get("contract_version", "1")),
            dataset_slug=str(options["dataset_slug"]),
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
            container_command=(
                [str(part) for part in options["container_command"]]
                if options.get("container_command") is not None
                else None
            ),
            validators=validators,
            seed=(int(options["seed"]) if options.get("seed") is not None else None),
            env_extra=env_extra,
            partition=slurm_block.get("partition"),
            account=slurm_block.get("account"),
            qos=slurm_block.get("qos"),
            time_minutes=(
                int(slurm_block["time_minutes"])
                if slurm_block.get("time_minutes") is not None
                else None
            ),
            mem_gb=(
                int(slurm_block["mem_gb"])
                if slurm_block.get("mem_gb") is not None
                else None
            ),
            cpus_per_task=int(slurm_block.get("cpus_per_task", 1)),
            gpus=(
                int(slurm_block["gpus"])
                if slurm_block.get("gpus") is not None
                else None
            ),
            extra_directives=extra_directives,
        )

    @property
    def container_env(self) -> Dict[str, str]:
        base = {
            "MVR_INPUT_DATA_PATH": "/input/data.h5mu",
            "MVR_OUTPUT_DIR": "/output",
            "MVR_JOB_SPEC_PATH": "/output/job_spec.json",
        }
        base.update(self.env_extra)
        return base


def _resolve_identity_from_options(spec: _SlurmJobSpec) -> ImageIdentity:
    if spec.image_digest:
        return ImageIdentity.registry_digest(spec.image_digest)
    return ImageIdentity.unverified_local(str(spec.image_sif))


def _failure_reason(info) -> str:
    state = info.state.value
    if info.reason:
        return f"slurm job ended in {state} ({info.reason})"
    return f"slurm job ended in {state}"


# ---------------------------------------------------------------------------
# options builder
# ---------------------------------------------------------------------------


def build_slurm_executor_options(
    *,
    model_slug: str,
    image_sif: str,
    dataset_slug: str,
    dataset_path: str,
    dataset_n_obs: int,
    params: Optional[Mapping[str, Any]] = None,
    image_digest: Optional[str] = None,
    model_version: str = "0.0.0",
    contract_version: str = "1",
    dataset_n_vars: Optional[int] = None,
    manifest_text: str = "",
    container_command: Optional[List[str]] = None,
    validators: str = "basic",
    artifact_dir_name: Optional[str] = None,
    seed: Optional[int] = None,
    container_env_extra: Optional[Mapping[str, str]] = None,
    partition: Optional[str] = None,
    account: Optional[str] = None,
    qos: Optional[str] = None,
    time_minutes: Optional[int] = None,
    mem_gb: Optional[int] = None,
    cpus_per_task: int = 1,
    gpus: Optional[int] = None,
    extra_directives: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Canonical ``options`` dict for the Slurm executor.

    Mirrors :func:`multiverse.mvd.docker_executor.build_executor_options`
    in shape; the Slurm-specific knobs live under the nested
    ``"slurm"`` key so the docker options layer stays untouched.
    """
    out: Dict[str, Any] = {
        "model_slug": model_slug,
        "model_version": model_version,
        "image_sif": image_sif,
        "contract_version": contract_version,
        "dataset_slug": dataset_slug,
        "dataset_path": dataset_path,
        "dataset_n_obs": int(dataset_n_obs),
        "params": dict(params or {}),
        "manifest_text": manifest_text,
        "validators": validators,
    }
    if seed is not None:
        out["seed"] = int(seed)
    if image_digest:
        out["image_digest"] = image_digest
    if dataset_n_vars is not None:
        out["dataset_n_vars"] = int(dataset_n_vars)
    if container_command is not None:
        out["container_command"] = [str(part) for part in container_command]
    if artifact_dir_name:
        out["artifact_dir_name"] = artifact_dir_name
    if container_env_extra:
        out["container_env_extra"] = {
            str(k): str(v) for k, v in dict(container_env_extra).items()
        }
    slurm_block: Dict[str, Any] = {"cpus_per_task": int(cpus_per_task)}
    if partition is not None:
        slurm_block["partition"] = partition
    if account is not None:
        slurm_block["account"] = account
    if qos is not None:
        slurm_block["qos"] = qos
    if time_minutes is not None:
        slurm_block["time_minutes"] = int(time_minutes)
    if mem_gb is not None:
        slurm_block["mem_gb"] = int(mem_gb)
    if gpus is not None:
        slurm_block["gpus"] = int(gpus)
    if extra_directives:
        slurm_block["extra_directives"] = list(extra_directives)
    out["slurm"] = slurm_block
    return out
