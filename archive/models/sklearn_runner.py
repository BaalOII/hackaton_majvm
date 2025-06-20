"""Sklearn engine (refined).

* Adds safe model instantiation: only passes ``random_state`` if the model
  accepts it and if not already set in ``params``.
* Otherwise identical behaviour: returns list[dict] with CV + hold‑out
  metrics, no side‑effects.
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

# ---------------------------------------------------------------------------
# 1. Registry of models and default params
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Tuple[Any, Dict[str, Any]]] = {
    "logreg": (LogisticRegression, dict(max_iter=1000, class_weight="balanced")),
    "knn": (KNeighborsClassifier, {}),
    "rf": (RandomForestClassifier, dict(n_estimators=200, class_weight="balanced")),
    "lda": (LinearDiscriminantAnalysis, dict(solver="lsqr", shrinkage="auto")),
    "qda": (QuadraticDiscriminantAnalysis, dict(reg_param=0.1)),
    "gnb": (GaussianNB, {}),
    "svc": (SVC, dict(kernel="rbf", probability=True)),
}


# ---------------------------------------------------------------------------
# 2. Helper: instantiate model safely
# ---------------------------------------------------------------------------

def _instantiate_model(model_cls, params):
    """Create a model, injecting ``random_state`` only when supported/needed."""
    if "random_state" not in params:
        if "random_state" in inspect.signature(model_cls).parameters:
            params = {**params, "random_state": settings.random_state}
    return model_cls(**params)


# ---------------------------------------------------------------------------
# 3. Build preprocessing + model pipeline
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 4. CV and hold‑out evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_cv(name: str, pipeline: Pipeline, X_tr, y_tr):
    cv = StratifiedKFold(
        n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_state
    )
    scores = cross_validate(
        pipeline,
        X_tr,
        y_tr,
        scoring=settings.scoring,
        cv=cv,
        return_estimator=True,
        n_jobs=-1,
    )
    row = {f"cv_{k}": np.mean(v) for k, v in scores.items() if k.startswith("test_")}
    row.update({"model": name, "estimator": scores["estimator"][-1]})
    return row


def _evaluate_holdout(est, X_te, y_te):
    preds = est.predict(X_te)
    record = {
        "test_accuracy": accuracy_score(y_te, preds),
        "test_f1_weighted": f1_score(y_te, preds, average="weighted"),
    }
    if hasattr(est, "predict_proba"):
        y_score = est.predict_proba(X_te)[:, 1]
    elif hasattr(est, "decision_function"):
        y_score = est.decision_function(X_te)
    else:
        y_score = None
    if y_score is not None:
        record["test_roc_auc"] = roc_auc_score(y_te, y_score)
    return record


# ---------------------------------------------------------------------------
# 5. Public API
# ---------------------------------------------------------------------------

def run(X_train, X_test, y_train, y_test) -> List[Dict[str, Any]]:  # noqa: N802
    records: List[Dict[str, Any]] = []

    for tag, (cls_, params) in MODEL_REGISTRY.items():
        pipeline = _make_pipeline(cls_, params)
        cv_row = _evaluate_cv(tag, pipeline, X_train, y_train)
        est = cv_row["estimator"]  # keep estimator for plotting
        ho_row = _evaluate_holdout(est, X_test, y_test)
        records.append({**cv_row, **ho_row})

    return records
