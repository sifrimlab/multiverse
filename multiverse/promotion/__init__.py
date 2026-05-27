"""Promotion saga and quarantine-only recovery (STRATEGY S3 / S4 / R5 / R13).

Hot-path module. Per ADR §8 the kernel must not import MLflow, Optuna, GC,
or the exporter; this module imports only the artifact contract and the
journal. SQLite and MLflow projection commits live in higher milestones.

The promotion saga is the single owner of the move from a workspace into a
final artifact directory. Every step is idempotent, every step writes its
intent to the journal *before* the side effect, and every step refuses to
delete a directory it does not hold an ownership token for. The only
filesystem mutation paths in this package are:

* writing the ``.mvd_owner`` token file in step 1;
* the staged cross-filesystem copy (or same-FS rename) in step 3;
* moving an owned-failed directory to ``store/quarantine/`` in the failure
  branch.

There is no ``os.unlink`` / ``os.rmdir`` / ``shutil.rmtree`` anywhere in this
package — enforced in CI by ``tests/unit/test_promotion_no_rmtree.py``.
"""

from .errors import (
    OwnershipMismatchError,
    PromotionError,
    SymlinkPolicyError,
)
from .fsutil import (
    StoreRoot,
    is_same_filesystem,
    open_dir_fd,
    safe_relative_path,
)
from .layout import StoreLayout
from .quarantine import (
    QUARANTINE_REPORT_FILENAME,
    TOMBSTONE_SUFFIX,
    QuarantineReport,
    quarantine_directory,
)
from .saga import (
    PromotionOutcome,
    PromotionResult,
    PromotionSaga,
    PromotionStep,
    OwnerToken,
)
from .tokens import (
    OWNER_TOKEN_FILENAME,
    read_owner_token,
    write_owner_token,
)

__all__ = [
    "OWNER_TOKEN_FILENAME",
    "OwnerToken",
    "OwnershipMismatchError",
    "PromotionError",
    "PromotionOutcome",
    "PromotionResult",
    "PromotionSaga",
    "PromotionStep",
    "QUARANTINE_REPORT_FILENAME",
    "QuarantineReport",
    "StoreLayout",
    "StoreRoot",
    "SymlinkPolicyError",
    "TOMBSTONE_SUFFIX",
    "is_same_filesystem",
    "open_dir_fd",
    "quarantine_directory",
    "read_owner_token",
    "safe_relative_path",
    "write_owner_token",
]
