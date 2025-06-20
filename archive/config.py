# config.py
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path

_CFG_FILE = Path("config.json")        # change if you want a different name

@dataclass
class Config:
    # — generic —
    random_state: int = 42
    test_size: float = 0.20
    plot_dir: str = "plots"

    # — preprocessing / CV —
    pca_variance: float = 0.95
    cv_folds: int = 5
    scoring: tuple[str, ...] = ("accuracy", "roc_auc", "f1_weighted")
    models_with_pca: tuple[str, ...] = (
        "logreg", "knn", "lda", "qda", "gnb", "svc"
    )

    # — engines to run —
    engines: tuple[str, ...] = ("sklearn", "torch")

    # — torch defaults —
    torch_epochs: int = 50
    torch_lr: float = 1e-3
    torch_batch_size: int = 32
    torch_dropout: float = 0.3
    torch_hidden1: int = 64
    torch_hidden2: int = 32
    torch_patience: int = 7
    torch_val_split: float = 0.2
    torch_cv_folds: int = 5

    # — dataset options —
    dataset_name: str = "breast_cancer"          # breast_cancer | csv | parquet | url_csv | url_parquet
    csv_path: str | None = None
    parquet_path: str | None = None
    url: str | None = None
    target_col: str | None = None

    # -----------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path = _CFG_FILE) -> "Config":
        if Path(path).exists():
            return cls(**json.loads(Path(path).read_text()))
        # no file → write defaults for the user, then load
        defaults = cls()
        Path(path).write_text(json.dumps(asdict(defaults), indent=4))
        return defaults

# singleton – import this everywhere
settings = Config.load()
