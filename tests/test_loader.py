import pandas as pd
from config import settings
from data.loader import load_data


def test_load_sklearn_iris(monkeypatch):
    monkeypatch.setattr(settings, "dataset_name", "iris")
    X, y = load_data()
    assert not X.empty
    assert len(X) == len(y)


def test_load_url_csv(monkeypatch, tmp_path):
    df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    monkeypatch.setattr(settings, "dataset_name", "url_csv")
    monkeypatch.setattr(settings, "url", str(csv_path))
    monkeypatch.setattr(settings, "target_col", "target")
    X, y = load_data()
    assert list(X.columns) == ["a"]
    assert y.tolist() == [0, 1]


def test_load_url_parquet(monkeypatch, tmp_path):
    df = pd.DataFrame({"b": [3, 4], "target": [1, 0]})
    pq_path = tmp_path / "data.parquet"
    df.to_parquet(pq_path)
    monkeypatch.setattr(settings, "dataset_name", "url_parquet")
    monkeypatch.setattr(settings, "url", str(pq_path))
    monkeypatch.setattr(settings, "target_col", "target")
    X, y = load_data()
    assert list(X.columns) == ["b"]
    assert y.tolist() == [1, 0]
