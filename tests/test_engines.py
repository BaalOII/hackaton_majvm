import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from engines.sklearn_engine import _evaluate_holdout


def _check_basic_keys(res):
    assert "test_accuracy" in res
    assert "test_f1_weighted" in res
    assert "test_roc_auc" in res


def test_evaluate_holdout_binary():
    X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    est = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    res = _evaluate_holdout(est, X_te, y_te)
    _check_basic_keys(res)
    assert isinstance(res["test_roc_auc"], float) or np.isnan(res["test_roc_auc"])


def test_evaluate_holdout_multiclass():
    X, y = make_classification(
        n_samples=60,
        n_features=5,
        n_classes=3,
        n_informative=4,
        n_clusters_per_class=1,
        random_state=0,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    est = LogisticRegression(max_iter=1000, multi_class="multinomial").fit(X_tr, y_tr)
    res = _evaluate_holdout(est, X_te, y_te)
    _check_basic_keys(res)
    assert isinstance(res["test_roc_auc"], float) or np.isnan(res["test_roc_auc"])

