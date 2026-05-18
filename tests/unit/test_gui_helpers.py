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
