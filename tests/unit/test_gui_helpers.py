from pathlib import Path

from multiverse.gui import build_run_manifest, slugify_experiment_name


def test_build_run_manifest_ignores_stale_pair_params():
    manifest = build_run_manifest(
        experiment_name="My Run",
        random_seed=7,
        run_mode="Use User Params",
        planned_jobs=[{"Dataset": "Dataset A", "Model": "pca"}],
        dataset_name_to_slug={"Dataset A": "dataset-a", "Dataset B": "dataset-b"},
        pair_params={
            ("Dataset A", "pca"): {"n_components": 20},
            ("Dataset B", "mofa"): {"stale": True},
        },
    )

    assert manifest["jobs"] == [
        {
            "dataset_slug": "dataset-a",
            "model_name": "pca",
            "model_params": {"n_components": 20},
        }
    ]
    assert "stale" not in str(manifest)


def test_slugify_experiment_name_rejects_empty_value():
    try:
        slugify_experiment_name(" !!! ")
    except ValueError as exc:
        assert "Experiment Name" in str(exc)
    else:
        raise AssertionError("Expected ValueError")



def test_shared_state_migrates_old_gui_keys(monkeypatch):
    from multiverse import gui_state

    fake_state = {
        "experiment_name": "legacy-exp",
        "jb_seed": 99,
        "exec_manifest_path": "legacy_manifest.yaml",
        "run_mode": "Run Gridsearch",
    }
    monkeypatch.setattr(gui_state.st, "session_state", fake_state)

    state = gui_state.init_state()

    assert state.shared_experiment_name == "legacy-exp"
    assert state.shared_seed == 99
    assert state.shared_manifest_path == "legacy_manifest.yaml"
    assert state.shared_run_mode == "Run Gridsearch"
    assert fake_state["experiment_name"] == "legacy-exp"
    assert fake_state["run_mode"] == "Run Gridsearch"


def test_fetch_runs_includes_dataset_name(tmp_path, monkeypatch):
    from multiverse import gui
    from multiverse import registry_db

    db_path = tmp_path / "state.db"
    for attr, path in {
        "DB_NAME": db_path,
        "STORE_DIR": tmp_path / "store",
        "DATASETS_DIR": tmp_path / "store" / "datasets",
        "RAW_DATASETS_DIR": tmp_path / "store" / "datasets" / "raw",
        "MODELS_DIR": tmp_path / "store" / "models",
        "ARTIFACTS_DIR": tmp_path / "store" / "artifacts",
        "WORKSPACES_DIR": tmp_path / "store" / "workspaces",
    }.items():
        monkeypatch.setattr(registry_db, attr, str(path))

    registry_db.init_db()
    conn = registry_db.get_db_connection()
    try:
        conn.execute(
            "INSERT INTO datasets (id, slug, name, path, omics_available, status) VALUES (1, 'ds', 'Dataset One', '/tmp/ds', '[\"rna\"]', 'READY')"
        )
        conn.execute(
            "INSERT INTO runs (dataset_id, model_slug, model_version, model_name, status, output_path) VALUES (1, 'pca', '1.0.0', 'PCA', 'SUCCESS', '/tmp/out')"
        )
        conn.commit()
    finally:
        conn.close()

    rows = gui._fetch_runs()

    assert rows[0]["dataset_name"] == "Dataset One"


def test_arrow_safe_summary_df_coerces_display_columns():
    from multiverse import gui

    df = gui.pd.DataFrame(
        [
            {"Run ID": 27, "Dataset": "PBMC10K", "Model": "pca", "Status": "FAILED"},
            {"Run ID": "3801ed44-a927", "Dataset": 3, "Model": None, "Status": "ARTIFACT_SUCCESS"},
        ]
    )

    safe = gui._arrow_safe_summary_df(df)

    assert safe["Run ID"].tolist() == ["27", "3801ed44-a927"]
    assert safe["Dataset"].tolist() == ["PBMC10K", "3"]
    assert safe["Model"].tolist() == ["pca", ""]


def test_selected_run_defaults_to_first_row(monkeypatch):
    from multiverse import gui

    class Event:
        class selection:
            rows = []

    monkeypatch.setattr(gui.st, "dataframe", lambda *args, **kwargs: Event())
    rows = [
        {"run_id": 2, "status": "SUCCESS"},
        {"run_id": 1, "status": "FAILED"},
    ]

    selected = gui._selected_run_from_summary(gui.pd.DataFrame([{"Run ID": 2}, {"Run ID": 1}]), rows)

    assert selected["run_id"] == 2


def test_manifest_load_notice_survives_rerun_once(monkeypatch):
    from multiverse import gui

    fake_state = {"_manifest_load_notice": "Manifest settings loaded."}
    messages = []
    monkeypatch.setattr(gui.st, "session_state", fake_state)
    monkeypatch.setattr(gui.st, "success", lambda message: messages.append(message))

    gui._render_manifest_load_notice()

    assert messages == ["Manifest settings loaded."]
    assert "_manifest_load_notice" not in fake_state


def test_run_configuration_renders_visible_shared_values(monkeypatch):
    from multiverse import gui

    fake_state = {
        "shared_experiment_name": "exp",
        "shared_seed": 11,
        "shared_run_mode": "Run Gridsearch",
        "shared_manifest_path": "manifest.yaml",
    }
    subheaders = []
    monkeypatch.setattr(gui.st, "session_state", fake_state)
    monkeypatch.setattr(gui.st, "subheader", lambda label: subheaders.append(label))
    monkeypatch.setattr(gui.st, "text_input", lambda _label, value, **_kwargs: value)
    monkeypatch.setattr(gui.st, "number_input", lambda _label, value, **_kwargs: value)
    monkeypatch.setattr(gui.st, "radio", lambda _label, options, index, **_kwargs: options[index])

    config = gui._render_run_configuration()

    assert subheaders == ["Run Configuration"]
    assert config == ("exp", 11, "Run Gridsearch", "manifest.yaml")
    assert fake_state["experiment_name"] == "exp"
    assert fake_state["run_mode"] == "Run Gridsearch"


def test_streamlit_use_container_width_is_not_used_in_app_code():
    offenders = []
    needle = "use_container" + "_width"
    for path in Path("multiverse").rglob("*.py"):
        if needle in path.read_text(encoding="utf-8"):
            offenders.append(str(path))
    assert offenders == []


def test_get_state_does_not_rewrite_widget_backed_shared_keys(monkeypatch):
    from multiverse import gui_state

    class GuardedState(dict):
        def __setitem__(self, key, value):
            if key in {"shared_run_mode", "shared_experiment_name"}:
                raise AssertionError(f"unexpected rewrite of {key}")
            super().__setitem__(key, value)

    fake_state = GuardedState({
        "shared_run_mode": "Run Gridsearch",
        "shared_experiment_name": "exp",
        "shared_seed": 3,
        "shared_manifest_path": "manifest.yaml",
    })
    monkeypatch.setattr(gui_state.st, "session_state", fake_state)

    state = gui_state.get_state()

    assert state.shared_run_mode == "Run Gridsearch"
    assert state.shared_experiment_name == "exp"


def test_configure_job_matrix_key_does_not_call_get_state_after_widgets(monkeypatch):
    from multiverse import gui

    monkeypatch.setattr(gui, "fetch_registry_data", lambda: ([{"name": "Dataset A", "slug": "dataset-a"}], [{"name": "pca"}]))
    monkeypatch.setattr(gui, "generate_compatibility_matrix", lambda datasets, models: gui.pd.DataFrame({"pca": ["Compatible"]}, index=["Dataset A"]))
    monkeypatch.setattr(gui, "_render_load_manifest_panel", lambda: None)
    monkeypatch.setattr(gui, "_render_manifest_load_notice", lambda: None)
    monkeypatch.setattr(gui, "_render_run_configuration", lambda: ("exp", 42, "Use User Params", "run_manifest.yaml"))
    monkeypatch.setattr(gui, "get_state", lambda: (_ for _ in ()).throw(AssertionError("late get_state call")))

    fake_state = {
        "selected_datasets": ["Dataset A"],
        "selected_models": ["pca"],
        "editor_version": 4,
    }
    monkeypatch.setattr(gui.st, "session_state", fake_state)
    monkeypatch.setattr(gui.st, "header", lambda *args, **kwargs: None)
    monkeypatch.setattr(gui.st, "divider", lambda *args, **kwargs: None)
    monkeypatch.setattr(gui.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(gui.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(gui.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(gui.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(gui.st, "multiselect", lambda label, options, default, key: default)

    def fake_data_editor(_df, **kwargs):
        assert kwargs["key"] == "job_matrix_editor_v5"
        return gui.pd.DataFrame([{
            "Selected": False,
            "Dataset": "Dataset A",
            "Model": "pca",
            "Compatibility": "Compatible",
        }])

    monkeypatch.setattr(gui.st, "data_editor", fake_data_editor)

    gui._render_configure_tab()


def test_stage_loaded_manifest_imports_jobs_and_params(monkeypatch):
    from multiverse import gui

    fake_state = {}
    monkeypatch.setattr(gui.st, "session_state", fake_state)

    gui._stage_loaded_manifest(
        {
            "globals": {"experiment_name": "exp", "random_seed": 9, "run_user_params": True},
            "jobs": [
                {"dataset_slug": "pbmc10k", "model_name": "PCA", "model_params": {"n_components": 2}},
                {"dataset_slug": "pbmc10k", "model_name": "MOFA", "model_params": {"n_factors": 3}},
            ],
        },
        "run_manifest.yaml",
    )

    assert fake_state["_pending_shared_experiment_name"] == "exp"
    assert fake_state["_pending_shared_seed"] == 9
    assert fake_state["_pending_shared_run_mode"] == "Use User Params"
    assert fake_state["_pending_manifest_jobs"] == [
        {"dataset_slug": "pbmc10k", "model_name": "PCA"},
        {"dataset_slug": "pbmc10k", "model_name": "MOFA"},
    ]
    assert fake_state["_pending_manifest_pair_params"][("pbmc10k", "PCA")] == {"n_components": 2}
    assert fake_state["_manifest_load_notice"] == "Manifest settings loaded (2 jobs)."


def test_apply_pending_manifest_jobs_maps_registry_names_and_params(monkeypatch):
    from multiverse import gui

    fake_state = {
        "_pending_manifest_jobs": [
            {"dataset_slug": "pbmc10k", "model_name": "pca"},
            {"dataset_slug": "missing", "model_name": "PCA"},
        ],
        "_pending_manifest_pair_params": {("pbmc10k", "pca"): {"n_components": 2}},
        "editor_version": 0,
    }
    monkeypatch.setattr(gui.st, "session_state", fake_state)

    loaded = gui._apply_pending_manifest_jobs(
        datasets=[{"name": "PBMC10K", "slug": "pbmc10k"}],
        models=[{"name": "PCA", "slug": "pca"}],
    )

    assert fake_state["selected_datasets"] == ["PBMC10K"]
    assert fake_state["selected_models"] == ["PCA"]
    assert loaded == {("PBMC10K", "PCA"): {"n_components": 2}}
    assert fake_state["_loaded_manifest_pair_params"] == loaded
    assert fake_state["editor_version"] == 1
    assert "_pending_manifest_jobs" not in fake_state


def test_prefill_hyperparameter_widget_state_handles_fixed_and_sweep(monkeypatch):
    from multiverse import gui

    fake_state = {}
    monkeypatch.setattr(gui.st, "session_state", fake_state)

    gui._prefill_hyperparameter_widget_state(
        "PBMC10K::PCA",
        {
            "n_components": 2,
            "latent_dimensions": {"type": "int", "low": 2, "high": 8, "log": False},
            "solver": {"type": "categorical", "choices": ["a", "b"]},
        },
    )

    assert fake_state["PBMC10K::PCA::fixed::n_components"] == 2
    assert fake_state["PBMC10K::PCA::sweep_toggle::n_components"] is False
    assert fake_state["PBMC10K::PCA::sweep_toggle::latent_dimensions"] is True
    assert fake_state["PBMC10K::PCA::sweep::latent_dimensions::low"] == 2
    assert fake_state["PBMC10K::PCA::sweep::latent_dimensions::high"] == 8
    assert fake_state["PBMC10K::PCA::sweep::latent_dimensions::dist"] == "int_uniform"
    assert fake_state["PBMC10K::PCA::sweep::solver::choices"] == ["a", "b"]


def test_find_umap_images_returns_only_supported_umap_images(tmp_path):
    from multiverse.gui_artifacts import find_umap_images

    (tmp_path / "umap.png").write_bytes(b"png")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "rna_umap.jpeg").write_bytes(b"jpeg")
    (tmp_path / "plot.png").write_bytes(b"png")
    (tmp_path / "umap.txt").write_text("not an image")

    found = [path.relative_to(tmp_path).as_posix() for path in find_umap_images(tmp_path)]

    assert found == ["nested/rna_umap.jpeg", "umap.png"]


def test_render_artifact_tree_expands_umap_image_with_preview_and_download(tmp_path, monkeypatch):
    from multiverse import gui_artifacts

    image = tmp_path / "umap.png"
    image.write_bytes(b"png")
    calls = {"expander": [], "image": [], "download": [], "dataframe": []}

    class Expander:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_expander(label, expanded=False):
        calls["expander"].append((label, expanded))
        return Expander()

    monkeypatch.setattr(gui_artifacts.st, "expander", fake_expander)
    monkeypatch.setattr(gui_artifacts.st, "image", lambda *args, **kwargs: calls["image"].append((args, kwargs)))
    monkeypatch.setattr(gui_artifacts.st, "download_button", lambda *args, **kwargs: calls["download"].append((args, kwargs)) or False)
    monkeypatch.setattr(gui_artifacts.st, "dataframe", lambda *args, **kwargs: calls["dataframe"].append((args, kwargs)))

    gui_artifacts.render_artifact_tree(tmp_path)

    assert calls["expander"] == [("umap.png (0.00 MB)", True)]
    assert calls["image"]
    assert calls["image"][0][0][0] == str(image)
    assert calls["download"]
    assert calls["download"][0][1]["file_name"] == "umap.png"
