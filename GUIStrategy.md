# GUI Strategy — Production-Grade UX Overhaul
**Document type:** Full UX audit + implementation strategy
**Author role:** Principal UI/UX Architect
**Date:** 2026-05-26
**Codebase audited:** `multiverse/gui.py` (1,715 lines) + `multiverse/gui_navigation.py`
**Scope:** Every tab, every widget, every navigation interaction — from first load to artifact download.

---

## 0. Executive Summary

The GUI implements a sound technical foundation (fragment-based live streaming, query-param navigation, typed session state, artifact helpers) but has accumulated **structural UX debt** that will actively confuse new users and frustrate daily users. The three biggest problems are:

1. **Two navigation systems doing the same job.** A horizontal radio-button top bar *and* a sidebar workflow stepper both let you switch tabs. The redundancy signals uncertainty about where navigation lives and wastes ~25% of the visible sidebar height on something the top bar already handles.
2. **One logical workflow fragmented across too many tabs.** The five-step core workflow (Register → Build Jobs → Set Params → Execute → View Results) is split across 5 of the 8 tabs, but "Parameters" is so thin it shouldn't be a top-level tab — it's a sub-section of job configuration. The remaining 3 tabs (Experiment Analysis, Sweep Tracker, Settings) have wildly different weights: Experiment Analysis embeds a full tool, Sweep Tracker is a dead-end (CSP blocks the iframe), and Settings has exactly one control.
3. **Shared configuration drifts silently between tabs.** Experiment name, random seed, and manifest path each appear in 2–3 separate places with separate session-state keys. A user who sets the seed in Job Builder gets a different default in Execute. A user who changes the experiment name in Parameters does not see it change in Job Builder.

Fixing these three structural problems is the entire strategy. Everything else in this document is a consequence.

---

## 1. Navigation & Information Architecture

### 1.1 The Dual Navigation Problem

**Current state:** `render_top_nav()` renders a horizontal `st.radio` with 8 options. `render_workflow_stepper()` renders 8 numbered buttons in the sidebar. Both navigate via `go_to()`. They are identical in function and near-identical in visual weight.

**Problems:**
- A user seeing both for the first time doesn't know which is canonical.
- The sidebar stepper occupies ~200px of sidebar height that should belong to observability status — the one thing the sidebar is uniquely suited to show because it's always visible.
- `st.radio(horizontal=True)` with 8 long labels ("Experiment Analysis", "Sweep Tracker") wraps or clips on any display narrower than 1400px wide.
- The stepper labels show step numbers (1-8) implying a mandatory sequence. But tabs 6, 7, 8 (MLflow, Optuna, Settings) are not steps in the core workflow. Numbering them as steps 6-8 is misleading.

**Fix:** Remove `render_workflow_stepper()` from the sidebar entirely. The top nav is the single navigation control. Replace the `st.radio` implementation with a pill-style nav rendered via `st.columns` + `st.button` (no horizontal radio) — this gives full control over styling, active state, and wrapping behaviour. The sidebar becomes observability-only + a settings shortcut.

### 1.2 Tab Count and Grouping

8 tabs is too many for a horizontal nav bar, especially with long labels. The solution is to collapse the tab count to 5 by:

| Before | After | Rationale |
|--------|-------|-----------|
| Registry | Registry | Unchanged |
| Job Builder | Configure | Merges Job Builder + Parameters into a two-section page with an internal sub-nav (radio or expanders) |
| Parameters | *(merged into Configure)* | Parameters is ≤40 lines of content per job; it's a sub-section not a top-level page |
| Execute | Run | Shorter label; same content |
| Results | Results | Unchanged |
| Experiment Analysis | Analysis | Shorter; hosts MLflow iframe + future in-app charts |
| Sweep Tracker | *(removed as tab)* | Dead-end tab (CSP blocks embed). Move Optuna link to sidebar observability section. |
| Settings | *(moved to sidebar)* | One control doesn't justify a tab. Move to a sidebar `st.expander("Settings")` |

Final tab structure: **Registry → Configure → Run → Results → Analysis** — five tabs, short labels, all meaningful.

### 1.3 The Sidebar Redesign

With navigation and settings removed, the sidebar has one job: **ambient system status**.

```
┌─────────────────────────┐
│  Services               │
│  ● MLflow  [Open ↗]     │
│  ● Optuna  [Open ↗]     │
│                         │
│  ▸ Settings             │  ← st.expander
│    Docker data root     │
│    [Save & Apply]       │
└─────────────────────────┘
```

This is always-visible, not navigational, and immediately useful. Collapsing settings here eliminates the Settings tab.

---

## 2. Tab-by-Tab Audit

### 2.1 Registry Tab

**Good:** Welcome empty-state with example dataset button is well-designed. Two-column datasets/models layout is clean. `fetch_registry_data()` with `@st.cache_data` is correct.

**Issues:**

| ID | Severity | Issue | Fix |
|----|----------|-------|-----|
| R-01 | Medium | Refresh button is always visible, but the `registry_dirty` warning banner appears at the top of `main()` above the nav bar — users may not correlate them. | Move the dirty banner inline into the Registry tab header, immediately above the data tables. Remove the standalone "Refresh Registry" button; make the banner itself the clickable refresh CTA. |
| R-02 | Medium | No way to deregister/delete a dataset or model. Users who register a test dataset are stuck with it. | Add a "Remove" button per row in the datasets/models dataframes, with a confirmation step (`st.warning` + second "Confirm remove" button). |
| R-03 | Low | Model registration has no field-builder mode. Dataset registration has a toggle "Build manifest from fields". The asymmetry is confusing. | Add a parallel toggle for model registration. Minimum fields: name, image tag, supported omics. |
| R-04 | Low | The two "Register Dataset" buttons (field-mode and manifest-mode) have different keys but the same label. On screen they look identical until you expand the expander. | Label them "Register from fields" and "Register from manifest" respectively. |
| R-05 | Low | After successful registration, the cache is cleared but there is no auto-rerun in the fields path (the manifest path has it). Line 400: the field-registration path clears the cache but doesn't call `st.rerun()`. | Add `st.rerun()` after the `fetch_registry_data.clear()` call in the fields registration path. |

### 2.2 Configure Tab (merged Job Builder + Parameters)

This is the highest-complexity tab and the source of the most UX problems.

**Current fragmentation issues:**

| ID | Severity | Issue | Fix |
|----|----------|-------|-----|
| C-01 | **Critical** | Experiment name, random seed, and run mode appear in *both* Job Builder and Parameters tabs with overlapping but non-identical session state keys. Specifically: `jb_seed` (Job Builder) and `exec_seed` (Execute) are *different keys*, so changing the seed in one place does not update the other. A user who sets seed=1234 in Job Builder will see seed=42 in Execute. | Establish a single source of truth for each: `shared_experiment_name`, `shared_seed`, `shared_run_mode`, `shared_manifest_path`. All tabs read from these keys. Remove the duplicates. |
| C-02 | **Critical** | "Generate Run Manifest" button exists in *both* Job Builder and Parameters. They produce different outputs (one includes `pair_params`, one doesn't). Users don't know which one to use, and using the wrong one silently produces a manifest without hyperparameter overrides. | One manifest generation UI only. Put it at the bottom of the merged Configure tab as a final step after both job selection and parameter configuration. |
| C-03 | High | After generating a manifest, the tab shows the YAML in `st.code` and a Makefile snippet but provides no "Next: go to Run tab" CTA. The user has to know to navigate to Execute next. | Add `st.button("Proceed to Run →", ...)` that calls `go_to("run")` immediately after the success message. |
| C-04 | High | The compatibility matrix and the job editor are two separate sections. The display-only `st.dataframe` (color-coded) sits above the `data_editor`. Scrolling between them on a large matrix is tedious. | Merge them: add a `Selected` column directly to the color-coded display dataframe styled with `st.data_editor`, eliminating the need for a separate display-only table above it. |
| C-05 | Medium | Resource summary metrics (Total Jobs, Unique Datasets, Unique Models, Committed RAM, Available RAM) appear in both Job Builder and Execute. | Show this summary only once, in the Run tab, where it is decision-relevant (about to launch). Remove from Configure. |
| C-06 | Medium | "Load manifest settings" is buried in an `st.expander` below the manifest generation controls. Most users won't find it. | Move it to the top of the Configure tab as a visible "Load existing manifest" section with a distinctive style (e.g., `st.info` background). |
| C-07 | Low | Incompatible pairs are deselected *after* the user has already checked them, with a `st.warning`. This is surprising and feels like an error. | Disable (grey out) incompatible rows in the data_editor rather than letting users check them and then unchecking them programmatically. Use `column_config` with a disabled condition if Streamlit supports it; otherwise use a tooltip-bearing read-only checkbox styled red. |
| C-08 | Low | Parameters tab shows "No schema found for this model. Falling back to JSON override input." This is not informative about *why* no schema was found or how to add one. | Show the expected schema path in the fallback message: "No hyperparameter schema found at `{expected_path}`. Add a `hyperparameters.json` to enable form fields." |

### 2.3 Run Tab (Execute)

**Issues:**

| ID | Severity | Issue | Fix |
|----|----------|-------|-----|
| E-01 | High | `exec_seed` and `jb_seed` are different session state keys. If you set seed=99 in Configure, Execute still shows 42. | Use the single shared seed key (C-01 fix). |
| E-02 | High | `exec_manifest_path` and `jb_manifest_path` are different keys. After generating a manifest at a custom path in Configure, Execute still points to `run_manifest.yaml`. | Use a single shared manifest path key. |
| E-03 | High | If manifest validation fails, `st.stop()` is called on line 1132. This prevents the "Live MLflow Metrics" panel below from rendering — the very panel a user might want to check after a failed run to diagnose what went wrong from a previous run. | Replace `st.stop()` with `return` after rendering the errors, allowing the rest of the page to continue rendering. |
| E-04 | Medium | The run monitor (`_run_monitor_fragment`) is wrapped in `st.status("Pipeline run", expanded=True)`. The `st.status` container shows a spinner while running and a checkmark when done, which is correct — but the cancel button, log output, and download button are all *inside* this container. On a tall log output, the container expands to fill the page, making it hard to scroll to the "Pipeline completed" banner. | Move the cancel button and completion banner *outside* the `st.status` container. `st.status` should contain only the log stream. |
| E-05 | Medium | `st.code(...)` for log display doesn't auto-scroll. The last 40 lines are shown but the user always sees the *top* of the code block on rerender. | Use `st.empty()` with a `components.html` auto-scrolling `<pre>` block, or simply render a `<div>` with `overflow-y: scroll; max-height: 400px` via `st.markdown` + unsafe_allow_html. |
| E-06 | Low | "Cancel Run" terminates the process with `proc.terminate()` but shows only a warning. No confirmation guard. The button is rendered inside the 1-second fragment, so accidental clicks are easy. | Add a `st.session_state["cancel_requested"]` flag. First click sets it and shows "Click again to confirm cancellation". Second click actually terminates. Reset on fragment re-render if not confirmed. |
| E-07 | Low | Live MLflow Metrics section shows "MLflow is offline" inline but the sidebar already shows this. The panel renders the service check again (`_check_service`) on every fragment rerun. | Read the service status from session state (populated by the sidebar check on page load) rather than re-issuing the HTTP probe in the fragment. |

### 2.4 Results Tab

**Issues:**

| ID | Severity | Issue | Fix |
|----|----------|-------|-----|
| RS-01 | High | Drill-down uses a `st.selectbox` to pick a run. With 50+ runs on a page this is a long dropdown with cryptic labels ("Run 42 — SUCCESS — pca"). | Replace with row-click selection: use `st.dataframe` with `selection_mode="single-row"` (Streamlit 1.35+). The selected row drives the drill-down below. No separate selectbox needed. |
| RS-02 | High | The results summary table shows "Run ID", "Model", "Status", "Output Path", "Failure Reason" — but not the **dataset**. This makes it impossible to distinguish two runs of the same model on different datasets from the table alone. | Add "Dataset" column. The `runs` table has `dataset_id`; join to dataset name in `_fetch_runs()`. |
| RS-03 | Medium | "Validation Retries" section appears between the summary table and the drill-down selector. Its placement interrupts the visual hierarchy. If there are no retryable runs it renders nothing, leaving a dead zone. | Move validation retries into the drill-down panel — show a "Retry" button only when the selected run has a `VALIDATION_ERROR` failure reason. Remove the standalone section. |
| RS-04 | Medium | MLflow deep-link section is at the bottom of the Results tab, buried below artifact tree + provenance + expanders. It requires scrolling past all content to find. | Move the "Open in Experiment Analysis →" CTA to the top of the drill-down section, immediately after the run header. Make it a prominent `st.link_button` or `st.button` with `go_to("analysis")`. |
| RS-05 | Low | `st.write("")` used as a spacer before the Refresh button (line 1243). This is a layout hack that adds an invisible text node to the DOM. | Remove the `st.write("")`. Use `st.columns([4, 1])` and rely on natural column alignment; or add vertical margin via a zero-height `st.markdown('<div style="margin-top:1.6rem"></div>', unsafe_allow_html=True)`. |
| RS-06 | Low | The Refresh button calls `st.cache_data.clear()` which clears *all* cached data app-wide, including registry data. | Call `fetch_registry_data.clear()` specifically, and clear only the runs-related cache. If using a dedicated `@st.cache_data` function for runs, clear that. |

### 2.5 Analysis Tab (MLflow)

**Issues:**

| ID | Severity | Issue | Fix |
|----|----------|-------|-----|
| A-01 | Medium | The iframe height slider (`st.slider("Height (px)", 400, 1200, 860, ...)`) is an unusual control. Users shouldn't need to manually resize an iframe; the app should fill available space. | Remove the slider. Set height to a fixed `900` or use `window.innerHeight` via `components.html` to set height dynamically. |
| A-02 | Low | The mixed-content warning ("your browser may be blocking mixed content") appears on every load, even when not applicable (HTTP-only deployments). It creates false alarm. | Show this warning only when `window.location.protocol == "https:"` — detectable via a one-time JS probe injected with `components.html`. |
| A-03 | Low | "Show all" button clears the active experiment. Its placement in a `[4,1]` column next to the experiment info banner is easy to accidentally click. | Rename to "Clear filter" and add a subtle style (secondary button type). Consider moving it into the expander for manual experiment selection. |

### 2.6 Sweep Tracker Tab (Optuna)

**Issue:** The entire tab body is:
> "Optuna Dashboard cannot be embedded due to its Content-Security-Policy. Open it in a new tab using the button below."

This is a dead-end tab. A tab that exists solely to say "this doesn't work here" is a UX antipattern. It trains users to distrust tabs.

**Fix:** Remove the Optuna tab entirely. Add an "Optuna" link to the sidebar observability section alongside the existing MLflow link. This is what users actually need: a quick way to open Optuna, not an empty tab.

### 2.7 Settings Tab

**Issue:** One setting (Docker data root). A whole tab for one setting is disproportionate and adds cognitive overhead to the navigation bar.

**Fix:** Move settings to a collapsed `st.expander("⚙ Settings")` at the bottom of the sidebar. This is always accessible without consuming a tab slot.

**Additional settings issue — Docker restart without confirmation:**
Line 1646 calls `subprocess.run(["systemctl", "--user", "restart", "docker"], ...)` immediately when the user clicks "Save & Apply". Restarting Docker interrupts any running containers (including active benchmark jobs). There is no confirmation step.

**Fix:** Add a `st.warning("Saving will restart the Docker daemon. Any running benchmark jobs will be interrupted.")` and a separate `st.button("Confirm and restart Docker")` before executing the `systemctl` call.

---

## 3. Cross-Cutting Issues

These span multiple tabs or affect the whole app.

| ID | Severity | Issue | Fix |
|----|----------|-------|-----|
| X-01 | High | **Shared state keys drift.** `jb_seed` ≠ `exec_seed`. `jb_manifest_path` ≠ `exec_manifest_path`. Silent divergence means the Execute tab doesn't pick up what the user set in Configure. | Rename to `shared_seed`, `shared_manifest_path`. All tabs read/write the same keys. |
| X-02 | High | **Experiment name widget collision.** Job Builder and Parameters both render `st.text_input(key="experiment_name")`. Since only one tab renders per page load, no Streamlit error is thrown — but it's a latent bug if the tab structure ever changes. | See C-01: one definition, one widget, read as a badge elsewhere. |
| X-03 | Medium | **Button label inconsistency.** "Go to Registry ->", "Go to Job Builder ->", "Proceed to Execute →" — inconsistent arrow styles and casing. Some use `->` (ASCII), some use `→` (Unicode). | Standardize to Unicode `→` and a consistent casing pattern. Or use Streamlit's icon param. |
| X-04 | Medium | **`st.dataframe` without explicit column widths** renders column widths determined by content. On narrow screens or sparse data, columns are either too wide or too narrow. | Specify `column_config` with explicit `width=` for every dataframe that is user-facing. |
| X-05 | Medium | **Service health probes run on every rerender.** `_check_service()` is called in the sidebar, in the Execute tab, and in the Analysis tab — potentially 3× per second during live runs (fragment reruns every 1 second). Each probe is a 1.5s-timeout HTTP request. | Cache service health in `st.session_state` with a 10-second TTL. Only re-probe when the cached result is stale. |
| X-06 | Low | **Emoji inconsistency.** Some buttons use emoji ("🔄 Refresh Registry") and some don't ("Refresh", "Launch Run"). Some headers use emoji (implied by tab labels), some don't. | Remove emojis from all button labels and tab labels. Reserve emojis for status indicators (● MLflow online / ○ offline). |
| X-07 | Low | **No page-level loading skeleton.** When `fetch_registry_data()` runs for the first time (cold cache), the page renders empty for 1-3 seconds. No spinner or skeleton is shown. | Wrap the initial data fetch in `with st.spinner("Loading registry…"):` in the Registry tab. |
| X-08 | Low | **`st.cache_data.clear()` (global) called from Results.** This clears registry data, defeating the purpose of caching it. | Use function-specific cache clearing: `fetch_registry_data.clear()` only when the registry changes, and a separate `@st.cache_data` for run queries. |

---

## 4. Proposed Information Architecture

### 4.1 Revised Tab Structure

```
[ Registry ] [ Configure ] [ Run ] [ Results ] [ Analysis ]
```

Tab mapping:
- **Registry** — unchanged content; incorporates dirty-banner inline.
- **Configure** — Job Builder + Parameters merged. Two sections: "1. Select Jobs" and "2. Set Parameters". Manifest generation is a single action at the bottom.
- **Run** — Execute tab content; resource ledger + launch + live monitor.
- **Results** — unchanged content; with RS-01 through RS-06 fixes.
- **Analysis** — MLflow iframe tab; Optuna link moved to sidebar.

### 4.2 Revised Sidebar

```
┌──────────────────────────────┐
│  Services                    │
│  ● MLflow online   [Open ↗]  │
│  ○ Optuna offline  [Open ↗]  │
│                              │
│  ▾ Settings                  │
│    Docker data root          │
│    /var/lib/docker           │
│    [Save & Apply]            │
└──────────────────────────────┘
```

No workflow stepper. No navigation. Sidebar is purely ambient status + settings.

### 4.3 Configure Tab Internal Layout

```
Configure
├── [Load existing manifest ↑]        ← prominent banner at top (was hidden in expander)
│
├── Section 1: Select Jobs
│   ├── multiselect: Datasets
│   ├── multiselect: Models
│   └── data_editor: compatibility + Selected checkbox (merged, not split)
│
├── Section 2: Hyperparameter Overrides     ← collapsed by default, one expander per pair
│   └── (one expander per selected pair)
│
├── Run Configuration                        ← single block, feeds both sections
│   ├── Experiment Name   [shared key]
│   ├── Random Seed       [shared key]
│   ├── Run Mode
│   └── Manifest save path [shared key]
│
└── [Generate & Save Manifest]  →  [Proceed to Run →]
```

---

## 5. Implementation Roadmap

Items are sequenced by dependency and blast radius, not just impact.

### Phase 1 — Structural Fixes (2 days, ~10 hrs)
Exit criterion: a new user can navigate the app without confusion about which nav control to use, and the 5-tab structure is in place.

| Item | Effort | Description |
|------|--------|-------------|
| Remove Optuna tab + move to sidebar | 30 min | Delete `_render_optuna_tab()`. Add Optuna link to `_render_observability_sidebar()`. |
| Remove sidebar workflow stepper | 15 min | Delete `render_workflow_stepper()` call from sidebar; keep function for potential future reuse. |
| Move Settings to sidebar expander | 1 hr | Extract `_render_settings_tab()` content into `st.expander("⚙ Settings")` in sidebar. Remove Settings tab from `TAB_LABELS`. |
| Merge Parameters into Configure tab | 2 hr | Combine `_render_job_builder_tab()` and `_render_parameters_tab()` into a single `_render_configure_tab()` with two labelled sections. |
| Unify shared state keys (X-01, C-01) | 1 hr | Rename `jb_seed`→`shared_seed`, `jb_manifest_path`→`shared_manifest_path`. Update all read/write sites. |
| Single manifest generation (C-02) | 30 min | Remove the "Generate Run Manifest (with params)" button from Parameters. Single generation button at the bottom of Configure. |
| Docker restart confirmation (Settings) | 30 min | Add confirmation warning + second button before `systemctl restart`. |
| Cache service health (X-05) | 45 min | Add a 10-second TTL session-state cache for MLflow/Optuna health status. |

### Phase 2 — Workflow Polish (1.5 days, ~8 hrs)
Exit criterion: the core 5-step workflow (register → configure → run → results → analysis) can be completed without the user needing to scroll to find the next action.

| Item | Effort | Description |
|------|--------|-------------|
| Inline registry-dirty banner (R-01) | 30 min | Remove global banner from `main()`; show inline at top of Registry tab header. |
| Load manifest at top of Configure (C-06) | 45 min | Move the "Load manifest settings" expander from below the generation section to the very top of Configure, styled as a distinct "Import settings" panel. |
| Post-manifest "Proceed to Run →" CTA (C-03) | 15 min | Add navigation button after manifest generation success. |
| Fix `st.stop()` in Execute (E-03) | 15 min | Replace `st.stop()` with `return`. |
| Dataset column in Results table (RS-02) | 1 hr | Join `dataset_id` → dataset name in `_fetch_runs()`. Add column to summary table. |
| Row-click drill-down in Results (RS-01) | 1.5 hr | Replace selectbox with `st.dataframe` row selection (`selection_mode="single-row"`). Handle Streamlit version fallback (selectbox if <1.35). |
| MLflow CTA in Results drill-down (RS-04) | 30 min | Move "Open in Analysis" button to top of drill-down, above artifact tree. |
| Remove Validation Retries section (RS-03) | 30 min | Integrate retry button into drill-down panel. |
| Fix `st.write("")` spacer (RS-05) | 5 min | Remove the spacer hack. |
| Add deregister/delete to Registry (R-02) | 1.5 hr | Add per-row Remove button with `st.warning` + confirmation in datasets and models tables. |

### Phase 3 — Hardening & Polish (1 day, ~5 hrs)
Exit criterion: the app handles edge cases gracefully and passes a basic usability test with an unfamiliar colleague.

| Item | Effort | Description |
|------|--------|-------------|
| Auto-scroll log viewer (E-05) | 1 hr | Replace `st.code` log display with a scrollable auto-scrolling HTML block. |
| Cancel run guard (E-06) | 30 min | Two-click cancel confirmation. |
| Disable incompatible rows (C-07) | 1 hr | Style incompatible rows as visually disabled in the job editor. |
| MLflow iframe height (A-01) | 15 min | Remove height slider, fix to 900px. |
| Mixed-content conditional warning (A-02) | 30 min | Show warning only on HTTPS deployments. |
| Model registration field builder (R-03) | 1 hr | Add toggle parity with dataset registration. |
| Standardise button labels and arrows (X-03, X-06) | 30 min | Global search-replace: `->` → `→`, remove emoji from buttons. |
| Function-scoped cache clearing (X-08, RS-06) | 30 min | Replace global `st.cache_data.clear()` with targeted clears. |
| Add dataset name to runs query (RS-02 follow-through) | Already in Phase 2 | — |

### Deferred to v1.x
The following are valuable but not blocking production quality:

- **Cross-run comparison** — side-by-side metrics for multiple runs (needs charting layer)
- **In-app metric charts** — replace the MLflow iframe with native Streamlit charts for key metrics
- **Run history timeline** — scrollable run history with filtering (N-07 from previous audit)
- **Full manifest import** — reconstruct job selection + param widgets from a saved manifest
- **Log filtering** — search/grep within the log viewer
- **Dark mode** — Streamlit default theming

---

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `st.dataframe` row selection requires Streamlit ≥1.35; repo may be pinned lower | Medium | Medium | Detect `st.__version__` at runtime; fall back to selectbox if <1.35. Pin `streamlit>=1.35` in `pyproject.toml`. |
| Merging Job Builder + Parameters into one tab increases tab render time for large job lists | Low | Low | The Parameters section renders only for `planned_jobs`; if empty, it renders nothing. No perf regression expected. |
| Removing the sidebar stepper breaks any bookmarked or scripted navigation that relies on it | Low | Low | `go_to()` and query-params remain unchanged; the stepper was purely a visual shortcut. |
| Shared seed key (`shared_seed`) changes default for existing users who relied on per-tab isolation | Low | Medium | Migration: on first load after the change, if `jb_seed` or `exec_seed` exists in session state, copy the value to `shared_seed` and delete the old keys. |
| Docker restart confirmation step may confuse users who expect "Save" to just save | Low | Low | Label the button "Save configuration" and the second button "Save and restart Docker". Make clear in the warning that restart is required for the change to take effect. |

---

## 7. Definition of Done

### Per Item
An item is done when **all four** are true:
1. Code change landed and the affected path is manually smoke-tested.
2. The corresponding audit row in this document is struck through with the commit SHA.
3. No regression in adjacent tabs (verified by navigating the full 5-tab flow end-to-end).
4. If the change introduces or removes a user-visible action, `gui_telemetry.track()` is updated accordingly.

### Per Phase
- **Phase 1:** Five-tab navigation is live; sidebar has only observability + settings; no dual navigation; shared state keys are unified.
- **Phase 2:** Core 5-step workflow completes with zero backtracking required; every tab has a visible next-step CTA where appropriate.
- **Phase 3:** App handles the failure path (failed run, offline MLflow, CSP warning) gracefully; a colleague unfamiliar with the tool reaches a successful run in ≤15 minutes without verbal prompting.

---

## 8. What This Strategy Does Not Do

- **No new analysis features.** No new charts, no cross-run comparison, no embedding previews. The scope is structural UX, not new functionality.
- **No CSS overrides or custom theming.** Streamlit defaults only. Custom styling is a v1.x item.
- **No backend changes.** The only backend-adjacent change is adding `dataset_id` → name resolution in `_fetch_runs()` (RS-02), which is a read-only query join.
- **No breaking API changes to `gui_navigation.py`.** The `go_to()`, `current_tab_slug()`, and `render_top_nav()` interfaces are preserved or extended, not replaced.
