"""The simple-mode runner — STRATEGY R7.

Pipeline per job:
    1. Resolve image identity via the backend.
    2. Compute the dataset fingerprint and recipe hashes.
    3. Spin up a workspace under ``<out>/_workspaces/<job-name>/``.
    4. Invoke the backend to populate the workspace.
    5. Run the post-flight validators against the workspace.
    6. On success: assemble the artifact manifest and write a portable bundle
       under ``<out>/<job-name>/``.
    7. On refusal: write a ``run_attempt_manifest.json`` under
       ``<out>/_failed/<job-name>/`` and move the workspace there as
       ``<out>/_failed/<job-name>/workspace/`` for diagnosis (the original
       ``_workspaces/<job-name>`` is removed once the copy succeeds).

Strict mode (``--strict``):
    * The image identity must be strict-acceptable (R10).
    * Every validator runs at ``ValidationLevel.STRICT``.
    * Any validator refusal — including those that would be warnings under
      basic — exits non-zero.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..artifact import (ArtifactManifest, BootContext, BundleInputs,
                        ExpectedArtifact, ImageIdentity, ImageIdentityKind,
                        ModelOutputContract, ProducedAt, ProducedBy,
                        RunAttemptManifest, StateTransition, ValidationLevel,
                        ValidationReport, compute_logical_run_id,
                        compute_manifest_hash, compute_params_hash,
                        new_physical_attempt_id, produced_at_now,
                        validate_output_bundle, write_bundle,
                        write_run_attempt_manifest)
from ..logging_utils import get_logger
from .backends.base import ExecutionBackend
from .manifest import SimpleJob, SimpleManifest

logger = get_logger(__name__)

_MVD_VERSION = "0.1.0-simple"


class StrictModeViolation(Exception):
    """Raised when a strict-mode run is asked to do something its mode forbids.

    For instance, the image identity is ``unverified_local`` and the runner
    was invoked with ``--strict``. The runner refuses to start the job
    rather than producing a bundle that would later be stripped of its
    strict-mode claim.
    """


class JobStatus(str, Enum):
    """Terminal state of a single simple-mode job.

    Attributes:
        ARTIFACT_SUCCESS: Bundle written; the only success state.
        EVALUATION_FAILED: Backend ran but post-flight validators refused.
        FAILED: Backend or strict-mode image gate failed before evaluation.
    """

    ARTIFACT_SUCCESS = "ARTIFACT_SUCCESS"
    EVALUATION_FAILED = "EVALUATION_FAILED"
    FAILED = "FAILED"


@dataclass
class JobOutcome:
    """Per-job result returned by ``SimpleModeRunner.run``."""

    job_name: str
    status: JobStatus
    bundle_path: Optional[Path] = None
    failure_dir: Optional[Path] = None
    validation_report: Optional[ValidationReport] = None
    logical_run_id: Optional[str] = None
    physical_attempt_id: Optional[str] = None
    failure_reason: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.status is JobStatus.ARTIFACT_SUCCESS


@dataclass
class SimpleModeResult:
    """Aggregate of one ``SimpleModeRunner.run`` invocation."""

    outcomes: List[JobOutcome] = field(default_factory=list)
    boot_id: Optional[str] = None

    @property
    def all_succeeded(self) -> bool:
        return bool(self.outcomes) and all(o.succeeded for o in self.outcomes)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class SimpleModeRunner:
    """Orchestrates one or more jobs end-to-end against the artifact contract.

    ``strict=True`` corresponds to ``multiverse run --simple --strict``. It is
    the publication-grade switch: image identity must be strict-acceptable,
    validators run at ``STRICT`` level, and validator warnings become hard
    refusals.
    """

    backend: ExecutionBackend
    output_root: Path
    strict: bool = False
    validators: ValidationLevel = ValidationLevel.BASIC
    seed: Optional[int] = None
    mvd_version: str = _MVD_VERSION
    git_commit: Optional[str] = None

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        if self.strict:
            self.validators = ValidationLevel.STRICT

    # ---- public API ----

    def run(self, manifest: SimpleManifest) -> SimpleModeResult:
        """Run every job in the manifest and collect their outcomes.

        Creates the output root, the shared ``_workspaces/`` scratch root,
        and a single ``BootContext`` whose ``boot_id`` ties together every
        timestamp in this invocation. Jobs run sequentially; one job's
        failure does not abort the rest.

        Args:
            manifest: Parsed manifest whose ``jobs`` are executed in order.

        Returns:
            Aggregate result carrying the boot id and one ``JobOutcome`` per
            job in manifest order.
        """
        boot = BootContext.new(mvd_version=self.mvd_version, git_commit=self.git_commit)
        self.output_root.mkdir(parents=True, exist_ok=True)
        workspaces_root = self.output_root / "_workspaces"
        failed_root = self.output_root / "_failed"
        workspaces_root.mkdir(parents=True, exist_ok=True)

        result = SimpleModeResult(boot_id=boot.boot_id)
        for job in manifest.jobs:
            outcome = self._run_job(
                job=job,
                manifest=manifest,
                boot=boot,
                workspaces_root=workspaces_root,
                failed_root=failed_root,
            )
            result.outcomes.append(outcome)
        return result

    # ---- internal ----

    def _run_job(
        self,
        *,
        job: SimpleJob,
        manifest: SimpleManifest,
        boot: BootContext,
        workspaces_root: Path,
        failed_root: Path,
    ) -> JobOutcome:
        """Execute one job through the full simple-mode pipeline.

        Runs the backend, applies the strict-mode image-identity gate, runs
        the post-flight validators, and on success composes the artifact
        manifest and writes the bundle under ``<out>/<job-name>/``. Any
        failure short-circuits to ``_record_failure`` which preserves the
        workspace and writes a ``run_attempt_manifest.json``.

        Args:
            job: The job to run.
            manifest: The owning manifest; its raw text feeds the manifest
                hash and its path is bundled as an input when present.
            boot: Boot context whose ``boot_id`` stamps every transition.
            workspaces_root: Scratch root; this job's workspace is recreated
                under ``<workspaces_root>/<job-name>/``.
            failed_root: Root under which failures are recorded.

        Returns:
            The job outcome (success bundle path or failure directory).
        """
        physical_attempt_id = new_physical_attempt_id()
        workspace = workspaces_root / job.name
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)

        transitions: List[StateTransition] = []

        def record_transition(
            from_state: str, to_state: str, reason: Optional[str] = None
        ) -> None:
            transitions.append(
                StateTransition(
                    from_state=from_state,
                    to_state=to_state,
                    at=produced_at_now(boot),
                    reason=reason,
                )
            )

        record_transition("PENDING", "RUNNING")

        # 1. Execute via backend.
        try:
            exec_result = self.backend.execute(
                job=job,
                workspace_dir=workspace,
                seed=self.seed,
            )
        except Exception as exc:
            record_transition("RUNNING", "FAILED", reason=f"backend: {exc}")
            return self._record_failure(
                job=job,
                manifest=manifest,
                boot=boot,
                physical_attempt_id=physical_attempt_id,
                image_identity=ImageIdentity.unverified_local(job.model_image),
                failed_root=failed_root,
                workspace=workspace,
                final_state=JobStatus.FAILED,
                failure_reason=f"backend: {type(exc).__name__}: {exc}",
                transitions=transitions,
                validation_report=None,
            )

        image_identity = exec_result.image_identity
        record_transition("RUNNING", "TRAINING_SUCCEEDED")

        # 2. Strict-mode image-identity gate.
        if self.strict and not image_identity.is_strict_acceptable:
            record_transition(
                "TRAINING_SUCCEEDED",
                "FAILED",
                reason=(
                    f"strict refusal: image variant is {image_identity.kind.value}; "
                    "strict mode requires registry_digest or build_context_hash"
                ),
            )
            return self._record_failure(
                job=job,
                manifest=manifest,
                boot=boot,
                physical_attempt_id=physical_attempt_id,
                image_identity=image_identity,
                failed_root=failed_root,
                workspace=workspace,
                final_state=JobStatus.FAILED,
                failure_reason=(
                    "strict mode refused image identity variant "
                    f"{image_identity.kind.value!r}; need registry_digest or build_context_hash"
                ),
                transitions=transitions,
                validation_report=None,
            )

        # 3. Post-flight validators.
        contract = ModelOutputContract(
            mv_contract_version=job.contract_version,
            artifacts=[
                ExpectedArtifact.embedding(expected_n_obs=job.dataset_n_obs),
                ExpectedArtifact.metrics(required=False),
                ExpectedArtifact.umap(required=False),
            ],
        )
        record_transition("TRAINING_SUCCEEDED", "EVALUATING")
        report = validate_output_bundle(workspace, contract, level=self.validators)

        if not report.passed:
            record_transition(
                "EVALUATING",
                "EVALUATION_FAILED",
                reason="; ".join(i.code for i in report.refusals),
            )
            return self._record_failure(
                job=job,
                manifest=manifest,
                boot=boot,
                physical_attempt_id=physical_attempt_id,
                image_identity=image_identity,
                failed_root=failed_root,
                workspace=workspace,
                final_state=JobStatus.EVALUATION_FAILED,
                failure_reason="post-flight validator refused: "
                + ", ".join(f"{i.code}: {i.message}" for i in report.refusals),
                transitions=transitions,
                validation_report=report,
            )

        # 4. Compose the artifact manifest and write the bundle.
        params_hash = compute_params_hash(job.params)
        manifest_hash = compute_manifest_hash(manifest.raw_text)
        fingerprint = job.dataset_fingerprint()
        logical_run_id = compute_logical_run_id(
            manifest_hash=manifest_hash,
            dataset_fingerprint=fingerprint,
            image_identity=image_identity,
            params_hash=params_hash,
            mv_contract_version=job.contract_version,
        )
        record_transition("EVALUATING", "ARTIFACT_SUCCESS")

        artifact_manifest = ArtifactManifest(
            logical_run_id=logical_run_id,
            physical_attempt_id=physical_attempt_id,
            manifest_hash=manifest_hash,
            dataset_fingerprint=fingerprint,
            image_identity=image_identity,
            params_hash=params_hash,
            mv_contract_version=job.contract_version,
            produced_at=ProducedAt.from_dict(produced_at_now(boot)),
            produced_by=ProducedBy(
                mvd_version=self.mvd_version,
                git_commit=self.git_commit,
            ),
            artifacts=list(report.artifact_entries),
            state_transitions=transitions,
            owner_token=None,
        )

        bundle_dir = self.output_root / job.name
        bundle_outputs: Dict[str, Path] = {}
        for entry in report.artifact_entries:
            src = workspace / entry.name
            if src.is_file():
                bundle_outputs[entry.name] = src
        bundle_logs: Dict[str, Path] = {}
        if exec_result.container_log_path and exec_result.container_log_path.is_file():
            bundle_logs["container.log"] = exec_result.container_log_path
        if exec_result.model_log_path and exec_result.model_log_path.is_file():
            bundle_logs["model.log"] = exec_result.model_log_path
        bundle_inputs: Dict[str, Path] = {}
        if manifest.path and manifest.path.is_file():
            bundle_inputs["run_manifest.yaml"] = manifest.path

        environment: Dict[str, Any] = {
            "mvd_version": self.mvd_version,
            "boot_id": boot.boot_id,
            "validators": self.validators.value,
            "strict": self.strict,
            "backend": getattr(self.backend, "name", type(self.backend).__name__),
        }
        if self.git_commit:
            environment["git_commit"] = self.git_commit

        write_bundle(
            bundle_dir,
            BundleInputs(
                artifact_manifest=artifact_manifest,
                outputs=bundle_outputs,
                inputs=bundle_inputs,
                logs=bundle_logs,
                environment=environment,
                validation_report=report.to_dict(),
            ),
        )

        return JobOutcome(
            job_name=job.name,
            status=JobStatus.ARTIFACT_SUCCESS,
            bundle_path=bundle_dir,
            validation_report=report,
            logical_run_id=logical_run_id,
            physical_attempt_id=physical_attempt_id,
        )

    def _record_failure(
        self,
        *,
        job: SimpleJob,
        manifest: SimpleManifest,
        boot: BootContext,
        physical_attempt_id: str,
        image_identity: ImageIdentity,
        failed_root: Path,
        workspace: Path,
        final_state: JobStatus,
        failure_reason: str,
        transitions: List[StateTransition],
        validation_report: Optional[ValidationReport],
    ) -> JobOutcome:
        """Record a failed job and preserve its workspace for diagnosis.

        Writes a ``run_attempt_manifest.json`` under
        ``<out>/_failed/<job-name>/`` and moves the workspace there as
        ``workspace/``. The copy-then-remove is guarded so a copy failure
        never destroys the only surviving copy (S5: workspace preserved).

        Args:
            job: The failed job.
            manifest: Owning manifest (for the manifest hash).
            boot: Boot context for transition timestamps.
            physical_attempt_id: Id of this execution attempt.
            image_identity: Resolved (or fallback) image identity to record.
            failed_root: Root under which the failure directory is created.
            workspace: The job's workspace to preserve.
            final_state: Terminal status to record (FAILED / EVALUATION_FAILED).
            failure_reason: Human-readable cause stored in the attempt record.
            transitions: State transitions accumulated so far.
            validation_report: Validator report when evaluation ran, else None.

        Returns:
            The failure outcome pointing at the recorded failure directory.
        """
        failed_root.mkdir(parents=True, exist_ok=True)
        failure_dir = failed_root / job.name
        failure_dir.mkdir(parents=True, exist_ok=True)

        manifest_hash = compute_manifest_hash(manifest.raw_text)
        params_hash = compute_params_hash(job.params)
        fingerprint = job.dataset_fingerprint()
        logical_run_id = compute_logical_run_id(
            manifest_hash=manifest_hash,
            dataset_fingerprint=fingerprint,
            image_identity=image_identity,
            params_hash=params_hash,
            mv_contract_version=job.contract_version,
        )
        recovery_hint = _recovery_hint_for(
            final_state, image_identity, validation_report
        )
        attempt = RunAttemptManifest(
            physical_attempt_id=physical_attempt_id,
            logical_run_id=logical_run_id,
            manifest_hash=manifest_hash,
            params_hash=params_hash,
            image_identity=image_identity.to_dict(),
            mv_contract_version=job.contract_version,
            final_state=final_state.value,
            failure_reason=failure_reason,
            produced_at=produced_at_now(boot),
            produced_by=ProducedBy(
                mvd_version=self.mvd_version,
                git_commit=self.git_commit,
            ).to_dict(),
            state_transitions=[t.to_dict() for t in transitions],
            recovery_hint=recovery_hint,
            validation_report=(
                validation_report.to_dict() if validation_report else None
            ),
        )
        write_run_attempt_manifest(failure_dir, attempt)

        # Preserve the workspace contents alongside the attempt record so the
        # user can inspect partial outputs (S5: "workspace preserved").
        # ``_failed/<job>/workspace`` is the canonical failed artifact, so once
        # the copy succeeds we remove the original ``_workspaces/<job>`` rather
        # than leaving a duplicate (issue #24). Both operations are guarded so a
        # copy failure never destroys the only surviving copy.
        preserved = failure_dir / "workspace"
        try:
            if preserved.exists():
                shutil.rmtree(preserved)
            shutil.copytree(workspace, preserved)
        except Exception:
            # Copy failed: keep the original workspace intact for diagnosis and
            # do not attempt the removal below.
            logger.warning(
                "failed to copy workspace for job %r into %s; leaving original "
                "workspace in place for inspection",
                job.name,
                preserved,
            )
        else:
            shutil.rmtree(workspace, ignore_errors=True)

        return JobOutcome(
            job_name=job.name,
            status=final_state,
            failure_dir=failure_dir,
            validation_report=validation_report,
            logical_run_id=logical_run_id,
            physical_attempt_id=physical_attempt_id,
            failure_reason=failure_reason,
        )


def _recovery_hint_for(
    status: JobStatus,
    identity: ImageIdentity,
    report: Optional[ValidationReport],
) -> str:
    """Compose an operator-facing recovery hint for a failed attempt.

    Returns:
        A sentence telling the user where to look and what to fix, tailored
        to validator refusals or an ``unverified_local`` image identity.
    """
    if status is JobStatus.EVALUATION_FAILED and report:
        return (
            "Inspect the saved workspace under workspace/. Re-run "
            "--validators basic to surface warnings, then fix the model "
            "outputs. Failed checks: " + ", ".join(i.code for i in report.refusals)
        )
    if identity.kind is ImageIdentityKind.UNVERIFIED_LOCAL:
        return (
            "Image identity is unverified_local; for strict mode push to a "
            "registry or record a build_context_hash."
        )
    return "Inspect workspace/ for partial outputs and container logs."
