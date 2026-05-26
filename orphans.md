# Orphaned Code Report

Generated: 2026-05-26 (verification pass)

## Scope and Method

Original scan used AST/name-reference analysis, `rg` reference checks, and `pyflakes`. This document adds a second-pass live verification with exhaustive `rg` call-site searches across the full repository (excluding `.venv`, caches, and generated metadata) and direct source inspection of each candidate.

Framework-called code such as pytest tests, Pydantic validators, Streamlit fragments, CLI subcommand callbacks, and `if __name__ == "__main__"` entry points was not counted as orphaned solely because it has few textual references.

Confidence legend:

- **High:** no call sites or unreachable by construction — verified by two independent search passes.
- **Medium:** reachable only through a legacy/island path or unregistered manual script.
- **Low:** dead-code smell, but could be intentionally kept for manual use or compatibility.

---

## Verification Summary

All orphan and legacy-island claims from the initial report were confirmed. One nuance: the initial report's claim about `registry.py` lines 18 and 50 needed clarification — both `load_registry()` and `get_eligible_models()` are called, but *only* from `tools/legacy_local_runner.py` (making them part of the same legacy island, not independent orphans). `generate_compatibility_matrix()` in the same file remains active (used by the GUI). The functional separation of these three functions is confirmed.

---

## Architecture Context (for removal strategy)

The repository has two distinct runtime paths that should not be confused:

| | **Modern path** | **Legacy path** |
|---|---|---|
| Entry point | `multiverse-cli` → `multiverse/runner/cli.py` | `runner.py` → `tools/legacy_local_runner.py` |
| Config format | YAML run manifest + SQLite registry | JSON config file |
| Data loading | Single `/input/data.h5mu` via Docker mount | `multiverse/data_utils.py` host-local |
| Model dispatch | Docker containers, per-model manifests | Direct Python class instantiation |
| Evaluation | `run_evaluation_container()` | Placeholder `evaluation_metrics.json` |
| Registration | `register_from_manifest()` → SQLite | Not present / legacy JSON registry |

The modern path is the production system. The legacy path is a development convenience that has not been maintained. Nothing in `setup.py`'s `console_scripts` points to the legacy path; `multiverse-cli` is the only installed entry point.

**Modularity constraint:** The platform is designed so contributors can register models and datasets with minimal operational setup. The manifest-based registration interface (`register_from_manifest`, `register_model_from_manifest`) and YAML run manifests are the stable extension points. Any removal strategy must preserve or improve this interface, not regress to the JSON-config model.

---

## Confirmed Orphans

### `multiverse/evaluate.py:131-145` — unreachable metrics-write block

**Verification:** Confirmed. Line 128 is an unconditional `return final_results` inside `aggregate_results()`. Lines 131–145 are indented one level *past the function boundary* — they are floating module-level statements, not part of any function. The names `bm` (a `Benchmarker` instance) and `requested_metrics` are undefined at module scope. An identical, correctly scoped version of this logic already exists in `Evaluator.evaluate_models()` (lines 245–286), confirming this is a copy-paste artifact from a refactor.

**Removal strategy:**

- **Delete lines 131–145** unconditionally. They cannot execute; removing them is zero-risk.
- No tests exercise this path (it is unreachable). No test changes needed.
- Do this as a standalone commit with the message `remove: unreachable metrics block in aggregate_results (copy-paste artifact)`.

---

### `multiverse/dataloader.py:138` — `DataLoader.read_mudata()`

**Verification:** Confirmed. Zero call sites found across the entire codebase. `multiverse/data_utils.py` constructs `DataLoader` and calls only `.preprocessing()`. Modern container I/O uses `multiverse/models/runtime_io.py::load_input_mudata()` instead.

**Removal strategy:**

- **Delete the method**. It is dead API surface that creates a false impression of a supported MuData-direct loading path.
- If MuData-direct loading is ever needed again, it belongs in `runtime_io.py` where the modern contract lives, not as a method on the legacy `DataLoader`.
- If deleting now is premature, mark it with a deprecation warning and a `# TODO: remove` comment at minimum, but prefer deletion.

---

### `multiverse/ingestion.py:125` — `register_dataset()`

**Verification:** Confirmed. Zero production call sites. The only reference outside the definition is in `tests/unit/test_ingestion.py`. The active registration path is `register_from_manifest()`, which additionally populates `slug`, `manifest_path`, `manifest_hash`, and `cell_type_key` — fields `register_dataset()` does not handle.

**Removal strategy:**

- **Delete the function and its corresponding unit tests.**
- `register_from_manifest()` is the public API. Keeping `register_dataset()` creates confusion about which function a contributor should call.
- If a single-file "quick register" CLI shortcut is wanted, add it as a new `multiverse-cli register-file` subcommand that calls `register_from_manifest()` under the hood with a synthesized in-memory manifest — do not resurrect the old function.

---

### `multiverse/models/base.py:87` — `ModelFactory.update_parameters()`

**Verification:** Confirmed. Zero call sites across the entire codebase. Model parameter flow now goes through job specs and per-container `run.py` scripts; `ModelFactory` instances are not mutated post-construction anywhere.

**Removal strategy:**

- **Delete the method.** It accepts arbitrary kwargs and logs unknown names, which is an unvalidated mutation surface. Removing it enforces the current design where parameters are set at construction time via manifests.
- If runtime parameter override becomes a requirement in future, implement it with schema validation against the model manifest — not open-ended attribute mutation.

---

### `multiverse/runner/cli.py:318` — `load_manifest()`

**Verification:** Confirmed (line 321 in current source). The function is defined but never called within `cli.py` or anywhere else. `parse_manifest()` (line 328) and `require_parsed_manifest()` (line 406) are the actively used functions that provide structured validation.

**Removal strategy:**

- **Delete the function.** `parse_manifest()` already wraps file I/O with validation. If YAML loading is ever needed without validation, use `yaml.safe_load` directly rather than a wrapper that bypasses the validated path.

---

### `multiverse/gui_navigation.py:38` — `current_tab_index()`

**Verification:** Confirmed. Zero call sites. `render_top_nav()` computes the index inline with `TABS.index(current)`.

**Removal strategy:**

- **Delete the function.** It is a one-liner helper that is redundant with an already-inline expression. If `render_top_nav()` grows complex enough to warrant extracting the index lookup, add it back then.

---

## Legacy Islands

These modules are not unreachable in the strict sense, but they form a self-contained cluster that is isolated from the modern Docker/registry workflow.

### Cluster: `runner.py`, `multiverse/main.py`, `tools/legacy_local_runner.py`, `multiverse/config_schema.py`, `multiverse/registry.py:18+50`

**What they do collectively:** `runner.py` calls `tools.legacy_local_runner.main_workflow()`, which loads datasets with `multiverse.data_utils`, filters models using the JSON registry (`load_registry`, `get_eligible_models`), calls model classes directly in host Python, and writes placeholder `evaluation_metrics.json` files. `multiverse/main.py` re-exports `main_workflow` and two helpers for legacy import compatibility. `multiverse/config_schema.py` provides the Pydantic schema for the old JSON config. None of these are reachable from `multiverse-cli`.

**Verification:** All confirmed. `runner.py` and `tools/legacy_local_runner.py` exist. `setup.py` entry points only list `multiverse-cli=multiverse.runner.cli:main`. `multiverse/main.py` has zero importers in active code.

**Removal strategy:**

The legacy runner provides one thing the modern path does not: **the ability to run models locally without Docker**. This is genuinely useful for contributors developing new models who do not yet have a container, and for CI environments that cannot run Docker. The right response is not to silently delete this capability but to **migrate it intentionally**.

**Recommended approach — two-phase migration:**

**Phase 1 (short-term, ~1 sprint): Isolate and document.**

1. Move the entire legacy cluster into `multiverse/compat/` (a dedicated sub-package):
   - `multiverse/compat/local_runner.py` (was `tools/legacy_local_runner.py`)
   - `multiverse/compat/config_schema.py` (was `multiverse/config_schema.py`)
   - `multiverse/compat/__init__.py` with a module-level `DeprecationWarning`
2. Update `runner.py` and `multiverse/main.py` to import from the new location, add deprecation notices, and update their module docstrings.
3. Register a `multiverse-cli run-legacy` subcommand that invokes `main_workflow()` so the capability is discoverable but clearly labelled as legacy.
4. Add a `LEGACY.md` in the repo root documenting: what it does, when to use it (local dev without Docker), its limitations (no manifest fields, no evaluation), and the roadmap for removal.

**Phase 2 (medium-term): Replace, not delete.**

5. Implement `multiverse-cli run --local` as a first-class subcommand that uses the modern manifest schema but skips Docker: it reads the YAML run manifest, resolves the model manifest, and calls the model's Python entry point directly. This gives contributors a local-dev path without the JSON config.
6. Once `multiverse-cli run --local` is stable and tested, remove the `compat/` package, `runner.py`, and `multiverse/main.py` in a single PR.

**Why this matters for modularity:** The manifest-based workflow (`register_model_from_manifest`, `register_from_manifest`) is the stable extension point that lets contributors add models and datasets with minimal overhead. A `--local` flag on the existing CLI keeps that interface intact while removing the need for Docker during development. This avoids a situation where contributors fall back to the legacy JSON-config path because it is easier to run locally.

---

## Cleanup-Only Dead Fragments

### `multiverse/runner/cli.py:18`, `:20`, `:21` — unused imports

**Verification:** Confirmed. `run_models_concurrently`, `run_job_container_sync`, and `ensure_image_prepared` are imported but never referenced in the file body. The functions themselves are live (used by the tuner, image preparation code, and tests).

**Removal strategy:** Remove the three import names from the `from .docker_runner import (...)` block in a single mechanical commit. No logic changes needed.

---

### Miscellaneous unused imports and locals

`pyflakes` and manual inspection confirmed the following dead imports and locals:

| Location | Item | Action |
|---|---|---|
| `multiverse/registry.py:3` | `Union` | remove import |
| `multiverse/dataloader.py:11` | `numpy as np` | remove import |
| `multiverse/ingestion.py:7` | `json` | remove import |
| `multiverse/ingestion.py:9` | `Union` | remove import |
| `multiverse/migrate_data.py:14` | `sys` | remove import |
| `multiverse/models/{pca,mofa,mowgli,multivi,cobolt,totalvi}.py:1` | `json` | remove import (6 files) |
| `multiverse/tools/rebuild_run_metrics.py:4` | `Path` | remove import |
| `multiverse/runner/tuner.py:3-4` | `json`, `os` | remove imports |
| `multiverse/runner/docker_runner.py:1025`, `:1119` | `loop` | remove assignments |
| `multiverse/runner/cli.py:242` | `missing` | remove assignment |

**Removal strategy:** Do these in one mechanical commit, separate from any logic-changing removals, so they are trivially reviewable.

---

## Manual or Unregistered Utilities

### `multiverse/tools/rebuild_run_metrics.py`

**Verification:** Confirmed. Has `if __name__ == "__main__": main()` at lines 45–46. Not registered in `setup.py`. Zero callers in the codebase.

**Removal strategy:** This is a maintenance utility, not dead code. **Register it** as a console script:

```python
# setup.py entry_points
"multiverse-cli-rebuild-metrics=multiverse.tools.rebuild_run_metrics:main",
```

or as a `multiverse-cli rebuild-metrics` subcommand. Add a one-paragraph doc entry under a "Maintenance Commands" section. Do not remove it — an unregistered maintenance script that is not documented will be re-written from scratch the next time it is needed.

---

## Suggested Cleanup Order

Execute these as separate, reviewable PRs:

1. **PR 1 — Zero-risk deletions (no behavior change):**
   - Delete unreachable block `multiverse/evaluate.py:131-145`.
   - Remove three unused imports from `multiverse/runner/cli.py`.
   - Remove all miscellaneous unused imports and locals listed above.

2. **PR 2 — Confirmed single-function orphan removals:**
   - Delete `DataLoader.read_mudata()` and its import scaffolding.
   - Delete `ingestion.register_dataset()` and its unit tests.
   - Delete `ModelFactory.update_parameters()`.
   - Delete `cli.load_manifest()`.
   - Delete `gui_navigation.current_tab_index()`.

3. **PR 3 — Maintenance utility registration:**
   - Register `rebuild_run_metrics` as a CLI subcommand or console script.
   - Add documentation.

4. **PR 4 — Legacy cluster isolation (Phase 1):**
   - Move legacy runner cluster to `multiverse/compat/`.
   - Add deprecation warnings and `LEGACY.md`.
   - Register `multiverse-cli run-legacy` subcommand.

5. **PR 5 — Local-dev run mode (Phase 2 prerequisite):**
   - Implement `multiverse-cli run --local` using manifest schema without Docker.
   - Cover with integration tests that do not require Docker.

6. **PR 6 — Legacy cluster removal (Phase 2):**
   - Remove `multiverse/compat/`, `runner.py`, `multiverse/main.py`.
   - Remove `multiverse/config_schema.py`.
   - Remove `load_registry()` and `get_eligible_models()` from `multiverse/registry.py`.
   - Update any remaining tests that import from these paths.

---

## Not Counted as Orphaned

- Pydantic validators with no direct call sites: called by model construction.
- pytest test functions and fixtures: called by pytest.
- Streamlit fragment-decorated functions and local GUI render helpers: called within the GUI module or by Streamlit.
- Nested helper functions in async runners and tuners: called by enclosing functions.
- `multiverse/runner/docker_runner.py::run_evaluation_container()`: called by the CLI.
- Model container `store/models/*/container/run.py` entry points: invoked by Docker container entrypoints/manifests.
- `multiverse/registry.py::generate_compatibility_matrix()`: actively used by `multiverse/gui.py`.
