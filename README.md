# Hackaton Majvm ML Pipeline

This repository contains a small machine learning workflow implemented in Python. The code was designed to be easily configurable and to produce ready‑to‑share reports. The main components are placed under `data/`, `engines/`, and `reports/` and are orchestrated by `main.py`.

## Configuration

Parameters are stored in **`config.json`**. At import time `config.py` reads this JSON file (or the path given in the `ML_CFG` environment variable) and exposes a `settings` object used throughout the project. If no file is found a template with default values is written. The options include random seeds, test split size, which engines to run, scikit‑learn model definitions, and PyTorch hyper‑parameters.

```python
from config import settings
```

Other modules simply import `settings` to access the values.

## Data utilities

- **`data/loader.py`** – Centralised dataset loader. Depending on `settings.dataset_name` it loads one of the built‑in scikit‑learn datasets or reads a local/remote CSV or Parquet file. The function always returns `X, y` as pandas objects so the rest of the pipeline stays the same.
- **`data/exploring.py`** – Helper functions used for exploratory data analysis (histograms, correlation matrix, box plots, etc.). Outputs are written to the plot directory.
- **`data/eda_runner.py`** – Convenience script that loads the data, calls the exploring helpers and finally builds an EDA HTML file using `data/eda_report_generator.py`.
- **`data/eda_report_generator.py`** – Reads the saved EDA plots/csv files and composes a simple HTML report with inline images.

## Engines

The project ships with two model engines selectable through the configuration file.

- **`engines/sklearn_engine.py`** – Builds a registry of scikit‑learn models from `settings.sklearn_models`. For each model a preprocessing pipeline is created (imputation, scaling and optional PCA). Models are evaluated with `StratifiedKFold` cross‑validation and the final estimator is kept for generating plots. Hold‑out metrics are computed on the external test set.
- **`engines/torch_engine.py`** – Implements a small two‑layer MLP in PyTorch. Training uses early stopping and can perform additional cross‑validation (`settings.torch_cv_folds`). After CV the network is retrained on the full training data and evaluated on the test set. Loss curves and prediction probabilities are returned for reporting.

Both engines produce a list of record dictionaries with metrics and (optionally) fitted estimators.

## Reporting

- **`reports/plots.py`** – Uses the records to produce standard figures: confusion matrices, ROC curves, loss curves and a PCA variance plot. Paths to the generated PNG files are returned for inclusion in the report.
- **`reports/report_generator.py`** – Converts the metrics dataframe and figures into a self‑contained HTML page. Verbose columns like raw estimators or prediction arrays are omitted to keep the table compact. Images are embedded as base64 so the report can be shared standalone.

## Workflow

`main.py` ties everything together:

1. Ensure the plot directory exists.
2. Load the dataset via `data.loader.load_data()`.
3. Split into train and test sets.
4. Run the selected engines (scikit‑learn and/or PyTorch).
5. Generate all figures and save a combined metrics CSV.
6. Build the final HTML report.

Running the script returns the metrics dataframe.

```bash
python main.py
```

The EDA utilities can be executed separately:

```bash
python data/eda_runner.py
```

## Tests

Basic tests covering the loader and parts of the engines are located in `tests/`. After installing the dependencies, run

```bash
pytest -q
```

to make sure everything works as expected.

