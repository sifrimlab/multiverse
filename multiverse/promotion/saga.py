"""The promotion saga (STRATEGY S3 / v2 §5).

Six steps, each idempotent, each journaled *before* the side-effect:

    1. PREPARE         — write owner token into a *staging* directory at
                         ``<artifact_dir>.staging.<attempt_id>``.
    2. VALIDATE        — run output semantic checks against the workspace.
    3. STAGE           — populate the staging dir with workspace contents
                         (same-FS per-file rename or cross-FS staged copy).
                         The final artifact path is NOT touched.
    4. COMMIT_MANIFEST — write the artifact manifest into staging, then
                         atomically rename the *entire* staging tree onto
                         the final artifact path. The final path appears
                         only after this single atomic operation, so the
                         workspace/artifact truth is never split across a
                         crash boundary.
    5. COMMIT_INDEX    — Milestone 8 (SQLite).
    6. COMMIT_TRACKING — Milestone 10 (MLflow projection).

On failure at step 2 the saga quarantines the *staging* dir (which it
owns by token) into ``store/quarantine/<date>/<attempt>/`` with a
tombstone at the planned-but-not-promoted artifact path. Per R5 the hot
path never deletes.

On crash between any two steps, replay reads the journal, sees the last
committed step, and resumes from the next one. The owner token in
``<staging>/.mvd_owner`` is how "is this staging dir mine to continue?"
is answered.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..artifact import (ArtifactManifest, ModelOutputContract, ValidationLevel,
                        ValidationReport, validate_output_bundle,
                        write_manifest)
from ..artifact.checksums import fsync_path
from ..journal import JournalKind, JournalWriter
from .errors import OwnershipMismatchError
from .fsutil import is_same_filesystem, staged_copy_directory
from .layout import StoreLayout
from .quarantine import QuarantineReport, quarantine_directory
from .tokens import (OWNER_TOKEN_FILENAME, new_owner_token, read_owner_token,
                     write_owner_token)


class PromotionStep(str, Enum):
    PREPARE = "PROMOTE_PREPARE"
    VALIDATE = "PROMOTE_VALIDATE"
    STAGE = "PROMOTE_STAGE"
    COMMIT_MANIFEST = "PROMOTE_COMMIT_MANIFEST"


class PromotionOutcome(str, Enum):
    PROMOTED = "PROMOTED"
    QUARANTINED = "QUARANTINED"
    FAILED_PRE_PREPARE = "FAILED_PRE_PREPARE"


# Friendly alias re-exported through ``__init__``.
OwnerToken = str


@dataclass
class PromotionResult:
    outcome: PromotionOutcome
    artifact_dir: Optional[Path] = None
    owner_token: Optional[OwnerToken] = None
    validation_report: Optional[ValidationReport] = None
    quarantine_report: Optional[QuarantineReport] = None
    committed_steps: List[PromotionStep] = field(default_factory=list)
    failure_reason: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.outcome is PromotionOutcome.PROMOTED


@dataclass
class PromotionSaga:
    """Drives one workspace through PREPARE → ... → COMMIT_MANIFEST.

    The saga reads from ``workspace_dir`` (the output of a model container)
    and writes into ``target_artifact_dir`` (under ``store/artifacts/``).
    All journaling is done through the supplied ``journal`` writer; the
    caller is responsible for calling ``journal.commit()`` cadence-wise
    (the saga does so at every step).
    """

    journal: JournalWriter
    layout: StoreLayout
    physical_attempt_id: str
    logical_run_id: str
    workspace_dir: Path
    target_artifact_dir: Path
    manifest: ArtifactManifest
    contract: ModelOutputContract
    validators: ValidationLevel = ValidationLevel.BASIC
    fsync_enabled: bool = True

    # Optional fault-injection hook used by tests to simulate crash between
    # steps. Not called in production. The callable is invoked AFTER the
    # journal commit for each step and BEFORE the next step starts.
    after_step_hook: Optional[Callable[[PromotionStep], None]] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> PromotionResult:
        result = PromotionResult(outcome=PromotionOutcome.FAILED_PRE_PREPARE)

        # Step 1: PREPARE — owner token written into the *staging* dir
        # (the final artifact path is not touched until step 4).
        try:
            owner_token = self._step_prepare()
        except FileExistsError as exc:
            result.failure_reason = str(exc)
            return result

        result.owner_token = owner_token
        # ``artifact_dir`` is the *planned* final location; it doesn't yet
        # exist on disk until COMMIT_MANIFEST atomic-swaps the staging dir.
        result.artifact_dir = self.target_artifact_dir
        result.committed_steps.append(PromotionStep.PREPARE)
        self._maybe_fault(PromotionStep.PREPARE)

        # Step 2: VALIDATE.
        report = self._step_validate()
        result.validation_report = report
        if not report.passed:
            result.committed_steps.append(PromotionStep.VALIDATE)
            quarantine_report = self._quarantine_after_validation_failure(
                owner_token=owner_token,
                refusal_codes=[i.code for i in report.refusals],
            )
            result.quarantine_report = quarantine_report
            result.outcome = PromotionOutcome.QUARANTINED
            result.failure_reason = "; ".join(
                f"{i.code}: {i.message}" for i in report.refusals
            )
            return result

        self.manifest.artifacts = list(report.artifact_entries)
        result.committed_steps.append(PromotionStep.VALIDATE)
        self._maybe_fault(PromotionStep.VALIDATE)

        # Step 3: STAGE — actually move workspace contents into the prepared
        # artifact dir (which already contains .mvd_owner).
        staged_checksums = self._step_stage(owner_token=owner_token)
        result.committed_steps.append(PromotionStep.STAGE)
        self._maybe_fault(PromotionStep.STAGE)

        # Step 4: COMMIT MANIFEST — atomic write + sidecar.
        self._step_commit_manifest(staged_checksums=staged_checksums)
        result.committed_steps.append(PromotionStep.COMMIT_MANIFEST)
        self._maybe_fault(PromotionStep.COMMIT_MANIFEST)

        result.outcome = PromotionOutcome.PROMOTED
        return result

    # ------------------------------------------------------------------
    # Resume entry point (used by replay)
    # ------------------------------------------------------------------

    def resume(self, *, last_committed_step: PromotionStep) -> PromotionResult:
        """Resume a saga that crashed after ``last_committed_step``.

        Looks for ``<artifact>.staging.<attempt_id>/.mvd_owner`` to prove
        ownership of the in-flight staging directory. If the staging dir
        is gone (e.g. the crash was after step 4's atomic swap), we treat
        the run as already PROMOTED and return.
        """
        # Already-promoted case: if the final artifact dir exists with a
        # token under this attempt, the atomic swap already happened.
        if self.target_artifact_dir.exists():
            final_token = read_owner_token(self.target_artifact_dir)
            if (
                final_token is not None
                and final_token.physical_attempt_id == self.physical_attempt_id
            ):
                return PromotionResult(
                    outcome=PromotionOutcome.PROMOTED,
                    owner_token=final_token.owner_token,
                    artifact_dir=self.target_artifact_dir,
                    committed_steps=[
                        PromotionStep.PREPARE,
                        PromotionStep.VALIDATE,
                        PromotionStep.STAGE,
                        PromotionStep.COMMIT_MANIFEST,
                    ],
                )

        staging_dir = self._staging_dir()
        token_file = read_owner_token(staging_dir)
        if token_file is None:
            # The staging dir vanished — re-run from PREPARE.
            return self.run()
        if token_file.physical_attempt_id != self.physical_attempt_id:
            raise OwnershipMismatchError(
                f"refusing to resume saga on {staging_dir}: "
                f"directory belongs to attempt "
                f"{token_file.physical_attempt_id!r}, not {self.physical_attempt_id!r}"
            )

        owner_token = token_file.owner_token
        result = PromotionResult(
            outcome=PromotionOutcome.FAILED_PRE_PREPARE,
            owner_token=owner_token,
            artifact_dir=self.target_artifact_dir,
            committed_steps=[PromotionStep.PREPARE],
        )

        order = [
            PromotionStep.PREPARE,
            PromotionStep.VALIDATE,
            PromotionStep.STAGE,
            PromotionStep.COMMIT_MANIFEST,
        ]
        idx_last = order.index(last_committed_step)
        if idx_last < order.index(PromotionStep.VALIDATE):
            # Resume from VALIDATE.
            report = self._step_validate()
            result.validation_report = report
            if not report.passed:
                quarantine_report = self._quarantine_after_validation_failure(
                    owner_token=owner_token,
                    refusal_codes=[i.code for i in report.refusals],
                )
                result.quarantine_report = quarantine_report
                result.outcome = PromotionOutcome.QUARANTINED
                result.committed_steps.append(PromotionStep.VALIDATE)
                return result
            self.manifest.artifacts = list(report.artifact_entries)
            result.committed_steps.append(PromotionStep.VALIDATE)
            self._maybe_fault(PromotionStep.VALIDATE)

        staged_checksums: Dict[str, str] = {}
        if idx_last < order.index(PromotionStep.STAGE):
            staged_checksums = self._step_stage(owner_token=owner_token)
            result.committed_steps.append(PromotionStep.STAGE)
            self._maybe_fault(PromotionStep.STAGE)

        if idx_last < order.index(PromotionStep.COMMIT_MANIFEST):
            self._step_commit_manifest(staged_checksums=staged_checksums)
            result.committed_steps.append(PromotionStep.COMMIT_MANIFEST)
            self._maybe_fault(PromotionStep.COMMIT_MANIFEST)

        result.outcome = PromotionOutcome.PROMOTED
        return result

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _staging_dir(self) -> Path:
        """Sibling staging path the saga owns until step 4 atomic-swaps it
        into ``target_artifact_dir``.

        Embedding ``physical_attempt_id`` in the name guarantees uniqueness
        across retries — two concurrent attempts on the same target name
        do not collide.
        """
        return (
            self.target_artifact_dir.parent
            / f".{self.target_artifact_dir.name}.staging.{self.physical_attempt_id}"
        )

    def _step_prepare(self) -> OwnerToken:
        owner_token = new_owner_token()
        if self.target_artifact_dir.exists():
            raise FileExistsError(
                f"refusing PREPARE on existing artifact dir {self.target_artifact_dir}; "
                "the prior owner (if any) decides what to do with it via quarantine."
            )
        staging_dir = self._staging_dir()
        if staging_dir.exists():
            raise FileExistsError(
                f"refusing PREPARE: staging dir already exists at {staging_dir}; "
                "a prior attempt with the same id left scratch on disk — "
                "Tier-1 GC will reclaim it."
            )

        self.journal.append(
            JournalKind.PROMOTE_PREPARE,
            payload={
                "workspace_dir": str(self.workspace_dir),
                "final_artifact_dir": str(self.target_artifact_dir),
                "staging_dir": str(staging_dir),
                "owner_token": owner_token,
            },
            physical_attempt_id=self.physical_attempt_id,
            logical_run_id=self.logical_run_id,
            prev_state="TRAINING_SUCCEEDED",
            next_state="PROMOTING",
        )
        self.journal.commit()

        write_owner_token(
            staging_dir,
            owner_token=owner_token,
            physical_attempt_id=self.physical_attempt_id,
            mvd_boot_id=self.journal.boot_id,
            purpose="promotion-prepare",
        )
        if self.fsync_enabled:
            fsync_path(staging_dir)
        return owner_token

    def _step_validate(self) -> ValidationReport:
        report = validate_output_bundle(
            self.workspace_dir, self.contract, level=self.validators
        )
        self.journal.append(
            JournalKind.PROMOTE_VALIDATE,
            payload={
                "passed": report.passed,
                "level": report.level.value,
                "issues": [i.to_dict() for i in report.issues],
            },
            physical_attempt_id=self.physical_attempt_id,
            logical_run_id=self.logical_run_id,
        )
        self.journal.commit()
        return report

    def _step_stage(self, *, owner_token: OwnerToken) -> Dict[str, str]:
        """Populate the staging directory with workspace contents.

        The final artifact path is NOT touched. We stage into
        ``<artifact>.staging.<attempt_id>/`` which already contains the
        ``.mvd_owner`` token from PREPARE. Same-FS uses per-file rename
        into staging; cross-FS uses ``staged_copy_directory`` into a
        substaging that we then absorb file-by-file. Both branches end
        with every file present under staging — never under the final
        artifact path.
        """
        staging_dir = self._staging_dir()
        same_fs = is_same_filesystem(self.workspace_dir, staging_dir)
        checksums: Dict[str, str] = {}

        from ..artifact.checksums import sha256_file

        if same_fs:
            for src in sorted(self.workspace_dir.rglob("*")):
                if src.is_dir():
                    continue
                rel = src.relative_to(self.workspace_dir)
                # Never overwrite the staging-owned token file.
                if rel.parts and rel.parts[0] == OWNER_TOKEN_FILENAME:
                    continue
                dst = staging_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(str(src), str(dst))
                checksums[str(rel)] = sha256_file(dst)
            if self.fsync_enabled:
                fsync_path(staging_dir)
        else:
            # Cross-FS: copy workspace contents into a substaging that we
            # then atomic-rename a file at a time into the saga's
            # staging dir. Both halves live under ``staging_dir``.
            substaging = staged_copy_directory(
                self.workspace_dir,
                staging_dir / "_substage",
                staging_token=self.physical_attempt_id,
                fsync_enabled=self.fsync_enabled,
            )
            for rel_str, digest in substaging.items():
                src = staging_dir / "_substage" / rel_str
                dst = staging_dir / rel_str
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(str(src), str(dst))
                checksums[rel_str] = digest
            substage_root = staging_dir / "_substage"
            # Walk substage bottom-up and remove empty dirs (substage is
            # our own per-attempt scratch, see R12 Tier-1 enumeration).
            if substage_root.exists():
                for entry in sorted(substage_root.rglob("*"), reverse=True):
                    try:
                        entry.rmdir() if entry.is_dir() else entry.unlink()
                    except OSError:
                        pass
                try:
                    substage_root.rmdir()
                except OSError:
                    pass
            if self.fsync_enabled:
                fsync_path(staging_dir)

        self.journal.append(
            JournalKind.PROMOTE_STAGE,
            payload={
                "same_filesystem": same_fs,
                "file_count": len(checksums),
                "owner_token": owner_token,
                "staging_dir": str(staging_dir),
            },
            physical_attempt_id=self.physical_attempt_id,
            logical_run_id=self.logical_run_id,
        )
        self.journal.commit()
        return checksums

    def _step_commit_manifest(self, *, staged_checksums: Dict[str, str]) -> None:
        """Write the manifest into staging, then atomic-swap staging onto
        the final artifact path.

        This is the only step that touches ``target_artifact_dir``. Before
        this rename the final path does not exist on disk; after this
        rename it carries the full manifest + every staged artifact in a
        single atomic operation. A crash between any prior steps leaves a
        recognisable staging dir and no final-path debris — there is no
        split-truth window.
        """
        staging_dir = self._staging_dir()
        if not self.manifest.artifacts:
            report = validate_output_bundle(
                staging_dir, self.contract, level=self.validators
            )
            if not report.passed:
                reasons = "; ".join(f"{i.code}: {i.message}" for i in report.refusals)
                raise ValueError(
                    "refusing COMMIT_MANIFEST with invalid staged artifacts: " + reasons
                )
            self.manifest.artifacts = list(report.artifact_entries)
        body_sha = write_manifest(
            staging_dir,
            self.manifest,
            fsync=self.fsync_enabled,
        )
        if self.target_artifact_dir.exists():
            raise FileExistsError(
                f"refusing COMMIT_MANIFEST: target {self.target_artifact_dir} "
                "appeared after PREPARE; quarantine staging via the recovery path."
            )
        os.replace(str(staging_dir), str(self.target_artifact_dir))
        if self.fsync_enabled:
            fsync_path(self.target_artifact_dir.parent)
        self.journal.append(
            JournalKind.PROMOTE_COMMIT_MANIFEST,
            payload={
                "artifact_dir": str(self.target_artifact_dir),
                "manifest_sha256": body_sha,
                "staged_file_count": len(staged_checksums),
            },
            physical_attempt_id=self.physical_attempt_id,
            logical_run_id=self.logical_run_id,
            prev_state="PROMOTING",
            next_state="ARTIFACT_SUCCESS",
        )
        self.journal.commit()

    # ------------------------------------------------------------------
    # Quarantine branch
    # ------------------------------------------------------------------

    def _quarantine_after_validation_failure(
        self,
        *,
        owner_token: OwnerToken,
        refusal_codes: List[str],
    ) -> QuarantineReport:
        staging_dir = self._staging_dir()
        self.journal.append(
            JournalKind.PROMOTION_QUARANTINE,
            payload={
                "source": str(staging_dir),
                "planned_artifact_dir": str(self.target_artifact_dir),
                "reason": "validator_refused",
                "refusal_codes": refusal_codes,
                "owner_token": owner_token,
            },
            physical_attempt_id=self.physical_attempt_id,
            logical_run_id=self.logical_run_id,
            prev_state="PROMOTING",
            next_state="PROMOTION_FAILED",
        )
        self.journal.commit()

        return quarantine_directory(
            source=staging_dir,
            layout=self.layout,
            reason="validator refused: " + ", ".join(refusal_codes),
            expected_owner_token=owner_token,
            physical_attempt_id=self.physical_attempt_id,
            extra_report=(
                "The promotion saga prepared this directory and the post-flight "
                "validator refused it. Workspace contents remain available for "
                "re-evaluation."
            ),
        )

    # ------------------------------------------------------------------
    # Test hooks
    # ------------------------------------------------------------------

    def _maybe_fault(self, step: PromotionStep) -> None:
        if self.after_step_hook is not None:
            self.after_step_hook(step)
