"""engines/sklearn_engine.py – generates records that include the fitted estimator
so the reporting layer can plot confusion matrices and ROC curves.
"""
from __future__ import annotations

import inspect
from typing import Dict, Tuple, Any, List

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from config import settings
import importlib

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _build_registry():
    registry = {}
    for entry in settings.sklearn_models:
        module, cls_name = entry["class"].rsplit(".", 1)
        cls_ = getattr(importlib.import_module(module), cls_name)
        registry[entry["tag"]] = (cls_, entry.get("params", {}))
    return registry

MODEL_REGISTRY = _build_registry()

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _instantiate_model(model_cls, params):
    """Inject random_state only if the estimator supports it."""
    if "random_state" not in params and "random_state" in inspect.signature(model_cls).parameters:
        params = {**params, "random_state": settings.random_state}
    return model_cls(**params)


def _make_pipeline(model_cls, params) -> Pipeline:
    steps = [("imputer", SimpleImputer()), ("scaler", StandardScaler())]
    if model_cls.__name__.lower() in [m.lower() for m in settings.models_with_pca]:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=settings.pca_variance,
                    svd_solver="full",
                    random_state=settings.random_state,
                ),
            )
        )
    model = _instantiate_model(model_cls, params)
    steps.append(("clf", model))
    return Pipeline(steps)


def _evaluate_cv(name: str, pipeline: Pipeline, X_tr, y_tr):
    cv = StratifiedKFold(settings.cv_folds, shuffle=True, random_state=settings.random_state)
    scoring = list(settings.scoring)
    n_classes = len(np.unique(y_tr))
    if n_classes > 2:
        scoring = [s if s != "roc_auc" else "roc_auc_ovr" for s in scoring]
    else:
        scoring = [s if s != "roc_auc_ovr" else "roc_auc" for s in scoring]
    scores = cross_validate(
        pipeline,
        X_tr,
        y_tr,
        scoring=scoring,
        cv=cv,
        return_estimator=True,
        n_jobs=-1,
    )
    row = {f"cv_{k}": np.mean(v) for k, v in scores.items() if k.startswith("test_")}
    if "cv_test_roc_auc_ovr" in row:
        row["cv_test_roc_auc"] = row.pop("cv_test_roc_auc_ovr")
    if "cv_test_roc_auc_ovo" in row:
        row["cv_test_roc_auc"] = row.pop("cv_test_roc_auc_ovo")
    row["model"] = name
    # Keep last fitted estimator (already trained)
    row["estimator"] = scores["estimator"][-1]
    return row


def _evaluate_holdout(est, X_te, y_te):
    preds = est.predict(X_te)
    res = {
        "test_accuracy": accuracy_score(y_te, preds),
        "test_f1_weighted": f1_score(y_te, preds, average="weighted"),
    }
    if hasattr(est, "predict_proba"):
        y_score = est.predict_proba(X_te)
    elif hasattr(est, "decision_function"):
        y_score = est.decision_function(X_te)
    else:
        y_score = None
    if y_score is not None:
        if np.ndim(y_score) > 1 and y_score.shape[1] > 1:
            res["test_roc_auc"] = roc_auc_score(y_te, y_score, multi_class="ovr")
        else:
            if np.ndim(y_score) > 1:
                y_score = y_score[:, 1]
            res["test_roc_auc"] = roc_auc_score(y_te, y_score)
    return res

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(X_train, X_test, y_train, y_test) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for tag, (cls_, params) in MODEL_REGISTRY.items():
        pipeline = _make_pipeline(cls_, params)
        cv_row = _evaluate_cv(tag, pipeline, X_train, y_train)
        est = cv_row["estimator"]  # keep estimator for plotting
        ho_row = _evaluate_holdout(est, X_test, y_test)
        records.append({**cv_row, **ho_row})

    return records
