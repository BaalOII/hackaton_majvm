"""engines/torch_engine.py – now includes cross‑validation.

* ``settings.torch_cv_folds`` controls the number of CV folds (≥2).  If it is
  ``1`` the engine behaves like before (single train/val split).
* For each fold we train from scratch, evaluate on the validation fold, and
  collect accuracy, F1‑weighted, and ROC‑AUC.  The mean of these scores is
  returned with keys ``cv_test_*`` – matching the Sk‑learn engine naming.
* After CV we retrain on the *entire* training set (with early stopping) and
  evaluate on the external test set – keeping behaviour unchanged.
"""
from __future__ import annotations

from typing import List, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import settings


# ---------------------------------------------------------------------------
# Simple MLP
# ---------------------------------------------------------------------------
class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.d1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.d2 = nn.Dropout(dropout)
        self.out = nn.Linear(hidden2, 1)

    def forward(self, x):  # type: ignore[override]
        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = self.d2(x)
        return torch.sigmoid(self.out(x))


# ---------------------------------------------------------------------------
# Training routine (with early stopping)
# ---------------------------------------------------------------------------

def _fit_single_model(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = _MLP(
        input_dim=X_train.shape[1],
        hidden1=settings.torch_hidden1,
        hidden2=settings.torch_hidden2,
        dropout=settings.torch_dropout,
    ).to(device)

    X_t = torch.from_numpy(X_train.astype(np.float32))
    y_t = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
    ds = TensorDataset(X_t, y_t)

    val_size = int(len(ds) * settings.torch_val_split)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(settings.random_state))

    loader_tr = DataLoader(train_ds, batch_size=settings.torch_batch_size, shuffle=True)
    loader_val = DataLoader(val_ds, batch_size=settings.torch_batch_size, shuffle=False)

    optim = torch.optim.Adam(net.parameters(), lr=settings.torch_lr)
    crit = nn.BCELoss()

    best_val = float("inf")
    best_state = None
    patience_ctr = 0
    tloss, vloss = [], []

    for epoch in range(settings.torch_epochs):
        # --- train ---
        net.train()
        run = 0.0
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            preds = net(xb)
            loss = crit(preds, yb)
            loss.backward()
            optim.step()
            run += loss.item() * xb.size(0)
        tloss.append(run / train_size)

        # --- val ---
        net.eval()
        run = 0.0
        with torch.no_grad():
            for xb, yb in loader_val:
                xb, yb = xb.to(device), yb.to(device)
                run += crit(net(xb), yb).item() * xb.size(0)
        vloss.append(run / val_size)

        # early stopping
        if vloss[-1] < best_val - 1e-6:
            best_val = vloss[-1]
            best_state = net.state_dict()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= settings.torch_patience:
                break

    net.load_state_dict(best_state)
    net.eval()

    return dict(model=net, device=device, train_losses=tloss, val_losses=vloss)


# ---------------------------------------------------------------------------
# Cross‑validation helper
# ---------------------------------------------------------------------------

def _cross_validate_torch(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=settings.torch_cv_folds, shuffle=True, random_state=settings.random_state)
    accs, f1s, rocs = [], [], []

    for train_idx, val_idx in cv.split(X, y):
        artifacts = _fit_single_model(X[train_idx], y[train_idx])
        net, device = artifacts["model"], artifacts["device"]
        with torch.no_grad():
            probs = net(torch.from_numpy(X[val_idx].astype(np.float32)).to(device)).cpu().numpy().ravel()
        preds = (probs >= 0.5).astype(int)
        accs.append(accuracy_score(y[val_idx], preds))
        f1s.append(f1_score(y[val_idx], preds, average="weighted"))
        rocs.append(roc_auc_score(y[val_idx], probs))

    return {
        "cv_test_accuracy": float(np.mean(accs)),
        "cv_test_f1_weighted": float(np.mean(f1s)),
        "cv_test_roc_auc": float(np.mean(rocs)),
    }


# ---------------------------------------------------------------------------
# Public engine API
# ---------------------------------------------------------------------------

def run_torch(X_train, X_test, y_train, y_test) -> List[Dict[str, Any]]:  # noqa: N802
    """Train Torch MLP with CV and return a record list (one element)."""
    X_tr = np.asarray(X_train) if not isinstance(X_train, np.ndarray) else X_train
    X_te = np.asarray(X_test) if not isinstance(X_test, np.ndarray) else X_test
    y_tr = np.asarray(y_train).astype(np.float32)
    y_te = np.asarray(y_test).astype(np.float32)

    record: Dict[str, Any] = {"model": "2-Layer Torch MLP"}

    # 1) cross‑validation ------------------------------------------
    if settings.torch_cv_folds > 1:
        record.update(_cross_validate_torch(X_tr, y_tr))

    # 2) retrain on full training set ------------------------------
    artifacts = _fit_single_model(X_tr, y_tr)
    net, device = artifacts["model"], artifacts["device"]

    with torch.no_grad():
        probs = net(torch.from_numpy(X_te.astype(np.float32)).to(device)).cpu().numpy().ravel()
    preds = (probs >= 0.5).astype(int)

    record.update(
        {
            "test_accuracy": accuracy_score(y_te, preds),
            "test_f1_weighted": f1_score(y_te, preds, average="weighted"),
            "test_roc_auc": roc_auc_score(y_te, probs),
            "train_losses": artifacts["train_losses"],
            "val_losses": artifacts["val_losses"],
            "probs": probs.tolist(),
            "preds": preds.tolist(),
        }
    )

    return [record]
