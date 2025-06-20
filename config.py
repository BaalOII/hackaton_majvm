import json
import os
from dataclasses import dataclass, asdict

@dataclass
class Config:
    random_state: int = 42
    test_size: float = 0.2
    pca_variance: float = 0.95
    cv_folds: int = 5
    plot_dir: str = "plots"
    models_with_pca: tuple = (
        'logisticregression',
        'kneighborsclassifier',
        'lineardiscriminantanalysis',
        'quadraticdiscriminantanalysis',
        'gaussiannb',
        'svc'
    )

    def save(self, path="config.json"):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @staticmethod
    def load(path="config.json"):
        if not os.path.exists(path):
            # Create default config if file doesn't exist
            config = Config()
            config.save(path)
            return config
        with open(path, "r") as f:
            data = json.load(f)
        return Config(**data)
