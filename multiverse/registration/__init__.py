"""Defensive registration pipeline (STRATEGY S19 / Milestone 14).

"Single user" does not mean "no untrusted input". A model registration is a
``Dockerfile + run.py``; a malicious or buggy ``model.yaml`` with ``..`` in
paths, a symlink under ``store/``, or a privileged Docker run flag can
destroy a user's machine without any multi-user threat model.

This package validates model and dataset registrations:

* **Path normalisation** — every declared path is resolved with
  ``realpath`` and refused if it escapes the configured store root.
* **Symlink policy** — see :mod:`multiverse.promotion.fsutil`. Symlinks
  inside managed roots are policy-rejected.
* **Privilege flags** — any model.yaml that requests ``privileged``,
  ``--network host``, ``--pid host``, or a volume mount outside the
  ``/input`` and ``/output`` contract is marked ``elevated`` and the
  caller must confirm before registration proceeds.
* **Trust level** — built-in models (registered via ``make
  register-models``) are ``BUILTIN``; user-supplied directories are
  ``IMPORTED`` and the GUI surfaces a banner.
"""

from .errors import (PathEscapeError, PrivilegedRegistrationError,
                     RegistrationError)
from .paths import safe_under_root, validate_paths_in_mapping
from .privileges import PRIVILEGE_FLAGS, PrivilegeAudit, audit_docker_flags
from .trust import TrustLevel, classify_trust
from .validator import (DatasetRegistrationReport, ModelRegistrationReport,
                        validate_dataset_manifest, validate_model_manifest)

__all__ = [
    "DatasetRegistrationReport",
    "ModelRegistrationReport",
    "PRIVILEGE_FLAGS",
    "PathEscapeError",
    "PrivilegeAudit",
    "PrivilegedRegistrationError",
    "RegistrationError",
    "TrustLevel",
    "audit_docker_flags",
    "classify_trust",
    "safe_under_root",
    "validate_dataset_manifest",
    "validate_model_manifest",
    "validate_paths_in_mapping",
]
