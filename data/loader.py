"""data/loader.py – centralized dataset access.

Supports three modes controlled by ``settings.dataset_name``:

* ``"breast_cancer"`` – scikit‑learn built‑in (default, no external files).
* ``"csv"`` – load a user‑supplied CSV from ``settings.csv_path`` with
  the target column given by ``settings.target_col``.
* ``"parquet"`` – same as CSV but for Parquet files (auto‑detects dtypes).

Returns ``X, y`` as pandas objects so the rest of the pipeline remains
unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.datasets import load_breast_cancer

from config import settings

__all__ = ["load_data"]


def _load_breast_cancer() -> Tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer(as_frame=True)
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


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Return ``X, y`` based on the dataset specified in ``settings``."""

    name = settings.dataset_name.lower()
    if name == "breast_cancer":
        return _load_breast_cancer()

    if name == "csv":
        if settings.csv_path is None or settings.target_col is None:
            raise ValueError("csv_path and target_col must be set in Settings when dataset_name='csv'.")
        return _load_csv(Path(settings.csv_path), settings.target_col)

    if name == "parquet":
        if settings.parquet_path is None or settings.target_col is None:
            raise ValueError("parquet_path and target_col must be set in Settings when dataset_name='parquet'.")
        return _load_parquet(Path(settings.parquet_path), settings.target_col)

    raise ValueError(f"Unknown dataset_name '{settings.dataset_name}'.")
