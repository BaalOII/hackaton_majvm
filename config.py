"""
config.py – ultra-light JSON loader

• Reads *config.json* (or the path in env var ML_CFG) at import-time
• If the file doesn’t exist, writes one pre-filled with default values
• Exposes a global `settings` SimpleNamespace that the rest of the
  pipeline imports:  `from config import settings`
"""

from __future__ import annotations

import json, os, types
from pathlib import Path

# ────────────────────────────────────────────────────────────────────
# 1. Which file to load?
#    Override with:  ML_CFG=other.json python main.py
# ────────────────────────────────────────────────────────────────────
CFG_FILE = Path(os.environ.get("ML_CFG", "config.json"))

# ────────────────────────────────────────────────────────────────────
# 2. All default parameters (same as the old dataclass/Pydantic)
# ────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    # — generic —
    "random_state": 42,
    "test_size": 0.20,
    "plot_dir": "plots",

    # — preprocessing / CV —
    "pca_variance": 0.95,
    "models_with_pca": ["logreg", "knn", "lda", "qda", "gnb", "svc"],
    "cv_folds": 5,
    "scoring": ["accuracy", "roc_auc", "f1_weighted"],

    # — engines to run —
    "engines": ["sklearn", "torch"],

    # — torch defaults —
    "torch_epochs": 50,
    "torch_lr": 0.001,
    "torch_batch_size": 32,
    "torch_dropout": 0.3,
    "torch_hidden1": 64,
    "torch_hidden2": 32,
    "torch_patience": 7,
    "torch_val_split": 0.20,
    "torch_cv_folds": 5,

    # — dataset options —
    #   breast_cancer | csv | parquet
    "dataset_name": "breast_cancer",
    "csv_path": None,
    "parquet_path": None,
    "url": None,
    "target_col": None,
}

# ────────────────────────────────────────────────────────────────────
# 3. Load JSON (or create it on first run) and merge with defaults
# ────────────────────────────────────────────────────────────────────
if CFG_FILE.exists():
    cfg_dict = json.loads(CFG_FILE.read_text())
else:
    # first run – write a template file the user can edit
    CFG_FILE.write_text(json.dumps(_DEFAULTS, indent=4))
    cfg_dict = {}
    print(f"[config] Wrote default config to {CFG_FILE.relative_to(Path.cwd())}")

final_cfg = {**_DEFAULTS, **cfg_dict}          # file overrides defaults
settings = types.SimpleNamespace(**final_cfg)  # import this everywhere
