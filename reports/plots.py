"""reports/plots.py

Generate diagnostic figures (confusion matrix, ROC curve, loss curve) for
both scikit‑learn and Torch engines.  All plots are written to
``settings.plot_dir`` and a nested dict of filepaths is returned so the
report layer can embed them.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import config 

__all__ = ["make_all_figures"]


# ───────────────────────────── helper plots ───────────────────────────────

def _plot_confusion(y_true, y_pred, title: str, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_roc(y_true, y_score, title: str, out_path: Path):
    y_t = np.asarray(y_true)
    y_s = np.asarray(y_score)

    if y_s.ndim == 1:
        fpr, tpr, _ = roc_curve(y_t, y_s)
    elif y_s.ndim == 2 and y_s.shape[1] == 1:
        fpr, tpr, _ = roc_curve(y_t, y_s.ravel())
    elif y_s.ndim == 2 and y_s.shape[1] == 2 and len(np.unique(y_t)) <= 2:
        fpr, tpr, _ = roc_curve(y_t, y_s[:, 1])
    else:
        classes = np.unique(y_t)
        y_bin = label_binarize(y_t, classes=classes)
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_s.ravel())
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_loss_curve(train, val, title: str, out_path: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(train, label="Train")
    plt.plot(val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_pca_variance(X, out_path: Path):
    """Plot explained and cumulative variance for PCA."""
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(random_state=config.settings.random_state).fit(X_scaled)
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    n_components = np.argmax(cumulative_var >= config.settings.pca_variance) + 1

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(explained_var) + 1), explained_var, marker="o", label="Explained Variance")
    plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker="s", label="Cumulative Variance")
    plt.axhline(y=config.settings.pca_variance, color="red", linestyle="--")
    plt.axvline(x=n_components, color="green", linestyle="--")
    plt.legend()
    plt.title("PCA Variance")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



# ───────────────────────────── public API ─────────────────────────────────

def make_all_figures(records: List[Dict[str, Any]], X_test, y_test) -> Dict[str, Dict[str, str]]:
    """Create plots for every model record and return their paths."""
    plot_dir = Path(config.settings.plot_dir)   # always fresh
    plot_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Dict[str, str]] = {}

    # PCA variance plot using the provided feature matrix
    pca_path = plot_dir / "pca_variance.png"
    _plot_pca_variance(X_test, pca_path)
    out["PCA variance"] = {"variance": str(pca_path)}

    for rec in records:
        name = rec.get("model", "unknown")
        out[name] = {}

        # 1) confusion + ROC from estimator -------------------------
        est = rec.get("estimator")
        if est is not None:
            y_pred = est.predict(X_test)
            cm_path = plot_dir / f"confusion_{name}.png"
            _plot_confusion(y_test, y_pred, f"Confusion: {name}", cm_path)
            out[name]["confusion"] = str(cm_path)

            # ROC only if scores available
            if hasattr(est, "predict_proba"):
                y_score = est.predict_proba(X_test)
            elif hasattr(est, "decision_function"):
                y_score = est.decision_function(X_test)
            else:
                y_score = None
            if y_score is not None:
                roc_path = plot_dir / f"roc_{name}.png"
                _plot_roc(y_test, y_score, f"ROC: {name}", roc_path)
                out[name]["roc"] = str(roc_path)

        # 2) confusion + ROC from raw arrays ------------------------
        if "probs" in rec and "preds" in rec:
            # ensure numpy arrays
            probs = np.asarray(rec["probs"])
            preds = np.asarray(rec["preds"])
            cm_path = plot_dir / f"confusion_{name}.png"
            _plot_confusion(y_test, preds, f"Confusion: {name}", cm_path)
            out[name]["confusion"] = str(cm_path)

            roc_path = plot_dir / f"roc_{name}.png"
            _plot_roc(y_test, probs, f"ROC: {name}", roc_path)
            out[name]["roc"] = str(roc_path)

        # 3) loss curve ---------------------------------------------
        if "train_losses" in rec and "val_losses" in rec:
            loss_path = plot_dir / f"loss_{name}.png"
            _plot_loss_curve(rec["train_losses"], rec["val_losses"], f"Loss Curve: {name}", loss_path)
            out[name]["loss_curve"] = str(loss_path)

    return out
