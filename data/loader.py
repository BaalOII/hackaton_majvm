"""data/loader.py – centralized dataset access.

Supports several modes controlled by ``settings.dataset_name``:

* names in ``SKLEARN_LOADERS`` (e.g. ``"breast_cancer"``, ``"iris"``, ``"wine"``)
  which load scikit‑learn built‑in datasets (default requires no external files)
* ``"csv"`` and ``"parquet"`` – load a local file from ``settings.csv_path`` /
  ``settings.parquet_path`` with the target column given by ``settings.target_col``
* ``"url_csv"`` and ``"url_parquet"`` – as above but download from
  ``settings.url``

Returns ``X, y`` as pandas objects so the rest of the pipeline remains
unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Callable, Dict, Any
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    load_digits,
)

from config import settings

# Mapping of available sklearn datasets
SKLEARN_LOADERS: Dict[str, Callable[..., Any]] = {
    "breast_cancer": load_breast_cancer,
    "iris": load_iris,
    "wine": load_wine,
    "digits": load_digits,
}

__all__ = ["load_data"]


def _load_sklearn(name: str) -> Tuple[pd.DataFrame, pd.Series]:
    loader = SKLEARN_LOADERS[name]
    data = loader(as_frame=True)
    df = data.frame
    X = df.drop(columns="target")
    y = df["target"]
    return X, y


def _load_csv(path: Path, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df[target_col]
    X = df.drop(columns=target_col)
    return X, y


def _load_parquet(path: Path, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(path)
    y = df[target_col]
    X = df.drop(columns=target_col)
    return X, y


def _load_url_csv(url: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(url)
    y = df[target_col]
    X = df.drop(columns=target_col)
    return X, y


def _load_url_parquet(url: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(url)
    y = df[target_col]
    X = df.drop(columns=target_col)
    return X, y


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Return ``X, y`` based on the dataset specified in ``settings``."""

    name = settings.dataset_name.lower()
    if name in SKLEARN_LOADERS:
        return _load_sklearn(name)

    if name == "csv":
        if settings.csv_path is None or settings.target_col is None:
            raise ValueError("csv_path and target_col must be set in Settings when dataset_name='csv'.")
        return _load_csv(Path(settings.csv_path), settings.target_col)

    if name == "parquet":
        if settings.parquet_path is None or settings.target_col is None:
            raise ValueError("parquet_path and target_col must be set in Settings when dataset_name='parquet'.")
        return _load_parquet(Path(settings.parquet_path), settings.target_col)

    if name == "url_csv":
        if settings.url is None or settings.target_col is None:
            raise ValueError("url and target_col must be set in Settings when dataset_name='url_csv'.")
        return _load_url_csv(settings.url, settings.target_col)

    if name == "url_parquet":
        if settings.url is None or settings.target_col is None:
            raise ValueError("url and target_col must be set in Settings when dataset_name='url_parquet'.")
        return _load_url_parquet(settings.url, settings.target_col)

    raise ValueError(f"Unknown dataset_name '{settings.dataset_name}'.")
