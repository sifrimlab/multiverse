"""Run executor protocol.

The kernel delegates "actually run a model and drive its run record to a
terminal state" to a ``RunExecutor`` injected at construction. Production
wires the Docker supervisor + promotion saga; tests use
``SyntheticRunExecutor`` which drives a run through the state machine
deterministically.

Keeping execution behind a protocol means:

* the kernel kernel imports neither Docker nor model code;
* the seven-verb surface can be tested without Docker or h5py.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from .runs import RunRecord
from .state import PrimaryState


@runtime_checkable
class RunExecutor(Protocol):
    """Backend that runs a model and drives its run record to a terminal state.

    Attributes:
        name: Short executor identity reported by ``Kernel.health``.
    """

    name: str

    async def execute(self, *, record: RunRecord, kernel: "Kernel") -> None:  # type: ignore[name-defined]
        """Drive ``record`` from PENDING to a terminal state.

        The executor must use ``kernel.transition()`` for every state
        change so the journal stays authoritative. It must also check
        ``record.cancel_requested`` between steps and honour it by
        transitioning through CANCEL_REQUESTED → CANCELLED.

        Args:
            record: The run record to execute and mutate via the kernel.
            kernel: The owning kernel, used for journaled state transitions.
        """
        ...


class NullRunExecutor:
    """Executor that immediately marks every run FAILED.

    Used as a placeholder when the kernel is constructed for API-surface
    tests that don't care about run lifecycle.
    """

    name = "null"

    async def execute(self, *, record: RunRecord, kernel) -> None:
        """Transition ``record`` straight to FAILED.

        Args:
            record: The run record to fail.
            kernel: The owning kernel used for the journaled transition.
        """
        await kernel.transition(
            record.physical_attempt_id,
            to_state=PrimaryState.FAILED,
            reason="NullRunExecutor: no execution backend configured",
        )


class SyntheticRunExecutor:
    """Executor that walks a run deterministically through the state machine.

    Used by tests and by the in-memory simple-mode pipeline. ``outcome``
    selects which terminal path is taken:

    * ``"success"`` → PENDING → ADMITTED → RUNNING → TRAINING_SUCCEEDED
                      → EVALUATING → PROMOTING → ARTIFACT_SUCCESS
    * ``"eval_fail"`` → ... → EVALUATING → EVALUATION_FAILED → RECOVERY_PENDING
    * ``"container_fail"`` → ... → RUNNING → FAILED
    """

    def __init__(self, outcome: str = "success") -> None:
        """Construct a synthetic executor for one of the supported paths.

        Args:
            outcome: One of ``"success"``, ``"eval_fail"``, or
                ``"container_fail"``; selects the terminal path.

        Raises:
            ValueError: If ``outcome`` is not a recognized path.
        """
        if outcome not in {"success", "eval_fail", "container_fail"}:
            raise ValueError(outcome)
        self.outcome = outcome
        self.name = f"synthetic-{outcome}"

    async def execute(self, *, record: RunRecord, kernel) -> None:
        """Drive ``record`` through the path selected by ``outcome``.

        Honours ``record.cancel_requested`` between steps, transitioning to
        CANCELLED if a cancel is observed.

        Args:
            record: The run record to drive through the state machine.
            kernel: The owning kernel used for journaled transitions.
        """
        async def _check_cancel() -> bool:
            if record.cancel_requested:
                await kernel.transition(
                    record.physical_attempt_id,
                    to_state=PrimaryState.CANCELLED,
                    reason="cancel_requested honoured by SyntheticRunExecutor",
                )
                return True
            return False

        await kernel.transition(
            record.physical_attempt_id, to_state=PrimaryState.ADMITTED
        )
        if await _check_cancel():
            return
        await kernel.transition(
            record.physical_attempt_id, to_state=PrimaryState.RUNNING
        )
        if await _check_cancel():
            return

        if self.outcome == "container_fail":
            await kernel.transition(
                record.physical_attempt_id,
                to_state=PrimaryState.FAILED,
                reason="synthetic: container exited non-zero",
            )
            return

        await kernel.transition(
            record.physical_attempt_id, to_state=PrimaryState.TRAINING_SUCCEEDED
        )
        if await _check_cancel():
            return
        await kernel.transition(
            record.physical_attempt_id, to_state=PrimaryState.EVALUATING
        )

        if self.outcome == "eval_fail":
            await kernel.transition(
                record.physical_attempt_id,
                to_state=PrimaryState.EVALUATION_FAILED,
                reason="synthetic: validator refused",
            )
            await kernel.transition(
                record.physical_attempt_id,
                to_state=PrimaryState.RECOVERY_PENDING,
                reason="awaiting user adopt/recover",
            )
            return

        await kernel.transition(
            record.physical_attempt_id, to_state=PrimaryState.PROMOTING
        )
        await kernel.transition(
            record.physical_attempt_id, to_state=PrimaryState.ARTIFACT_SUCCESS
        )
