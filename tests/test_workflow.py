import main


def test_run_pipeline_basic(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "plot_dir", str(tmp_path))
    monkeypatch.setattr(main.settings, "dataset_name", "iris")
    monkeypatch.setattr(main.settings, "engines", ["sklearn"])
    monkeypatch.setattr(main.settings, "cv_folds", 2)
    monkeypatch.setattr(
        main.settings,
        "sklearn_models",
        [
            {
                "tag": "LR",
                "class": "sklearn.linear_model.LogisticRegression",
                "params": {"max_iter": 100}
            }
        ],
    )

    metrics = main.run_pipeline()
    assert not metrics.empty
    assert "model" in metrics.columns
    assert any(metrics["model"] == "LR")

