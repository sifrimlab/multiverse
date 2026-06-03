"""Primary state machine (STRATEGY S5 / R6 / R14).

Primary states are the *scientific* outcome surface. The GUI mapping (R14)
lives in the GUI module, not here; the kernel's API returns the precise
internal state. Projection statuses (MLflow, Optuna) live in
``RunRecord.projections``, never in this enum.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet


class PrimaryState(str, Enum):
    """Canonical run lifecycle states (journal authority, GUI reads via API).

    Valid edges are defined in :data:`STATE_TRANSITIONS`. Terminal states end
    the run; sub-terminal states (training/eval/promotion branch outcomes)
    may still transition before a terminal state.
    """

    PENDING = "PENDING"
    ADMITTED = "ADMITTED"
    RUNNING = "RUNNING"
    TRAINING_SUCCEEDED = "TRAINING_SUCCEEDED"
    EVALUATING = "EVALUATING"
    EVALUATION_FAILED = "EVALUATION_FAILED"
    PROMOTING = "PROMOTING"
    PROMOTION_FAILED = "PROMOTION_FAILED"
    ARTIFACT_SUCCESS = "ARTIFACT_SUCCESS"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    RECOVERY_PENDING = "RECOVERY_PENDING"

    @property
    def is_terminal(self) -> bool:
        """True when no further primary-state transitions are allowed."""
        return self in _TERMINAL

    @property
    def is_sub_terminal(self) -> bool:
        """True for branch outcomes that may still transition (e.g. re-evaluate)."""
        return self in _SUB_TERMINAL


_TERMINAL: FrozenSet[PrimaryState] = frozenset(
    {
        PrimaryState.ARTIFACT_SUCCESS,
        PrimaryState.CANCELLED,
        PrimaryState.FAILED,
        PrimaryState.RECOVERY_PENDING,
    }
)
_SUB_TERMINAL: FrozenSet[PrimaryState] = frozenset(
    {
        PrimaryState.TRAINING_SUCCEEDED,
        PrimaryState.EVALUATION_FAILED,
        PrimaryState.PROMOTION_FAILED,
    }
)


# Canonical transitions. Cancellation can interrupt the active branch from
# any non-terminal state and is enumerated explicitly.
STATE_TRANSITIONS: Dict[PrimaryState, FrozenSet[PrimaryState]] = {
    PrimaryState.PENDING: frozenset(
        {PrimaryState.ADMITTED, PrimaryState.CANCEL_REQUESTED, PrimaryState.FAILED}
    ),
    PrimaryState.ADMITTED: frozenset(
        {PrimaryState.RUNNING, PrimaryState.CANCEL_REQUESTED, PrimaryState.FAILED}
    ),
    PrimaryState.RUNNING: frozenset(
        {
            PrimaryState.TRAINING_SUCCEEDED,
            PrimaryState.FAILED,
            PrimaryState.CANCEL_REQUESTED,
        }
    ),
    PrimaryState.TRAINING_SUCCEEDED: frozenset(
        {
            PrimaryState.EVALUATING,
            PrimaryState.PROMOTING,  # if no eval phase
            PrimaryState.CANCEL_REQUESTED,
        }
    ),
    PrimaryState.EVALUATING: frozenset(
        {
            PrimaryState.EVALUATION_FAILED,
            PrimaryState.PROMOTING,
            PrimaryState.CANCEL_REQUESTED,
        }
    ),
    PrimaryState.EVALUATION_FAILED: frozenset(
        {
            # Re-evaluate without retraining (S5 acceptance).
            PrimaryState.EVALUATING,
            PrimaryState.RECOVERY_PENDING,
        }
    ),
    PrimaryState.PROMOTING: frozenset(
        {
            PrimaryState.ARTIFACT_SUCCESS,
            PrimaryState.PROMOTION_FAILED,
            PrimaryState.CANCEL_REQUESTED,
        }
    ),
    PrimaryState.PROMOTION_FAILED: frozenset({PrimaryState.RECOVERY_PENDING}),
    PrimaryState.CANCEL_REQUESTED: frozenset(
        {PrimaryState.CANCELLED, PrimaryState.FAILED}
    ),
    # Terminal states have no successors.
    PrimaryState.ARTIFACT_SUCCESS: frozenset(),
    PrimaryState.CANCELLED: frozenset(),
    PrimaryState.FAILED: frozenset(),
    PrimaryState.RECOVERY_PENDING: frozenset(),
}


PROJECTION_STATUSES: Dict[str, FrozenSet[str]] = {
    # MLflow projection (S13 / R6).
    "mlflow": frozenset(
        {
            "TRACKING_PENDING",
            "TRACKING_SYNCED",
            "TRACKING_SYNC_FAILED",
            "TRACKING_NOT_CONFIGURED",
            "TRACKING_NOT_APPLICABLE",
        }
    ),
    "optuna": frozenset(
        {
            "TRACKING_PENDING",
            "TRACKING_SYNCED",
            "TRACKING_SYNC_FAILED",
            "TRACKING_NOT_APPLICABLE",
        }
    ),
}


def assert_valid_transition(from_state: PrimaryState, to_state: PrimaryState) -> None:
    """Raise ``ValueError`` if the transition is not in the canonical set.

    Used by every state-mutating kernel verb; the kernel's invariants make
    illegal moves impossible from the API surface — illegal moves come from
    bugs in the executor or in replay.
    """
    if to_state == from_state:
        return  # idempotent self-loop (rare; the kernel accepts).
    valid = STATE_TRANSITIONS.get(from_state, frozenset())
    if to_state not in valid:
        raise ValueError(
            f"illegal state transition {from_state.value} -> {to_state.value}"
        )
