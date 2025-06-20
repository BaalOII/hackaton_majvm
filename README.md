# Hackaton ML Pipeline

This project contains a simple machine learning workflow with optional data exploration tools.

## Setup

1. Create a Python environment (Python 3.10+ recommended).
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. On the first run a `config.json` file is created with default settings. You can
   point to a different configuration with the `ML_CFG` environment variable:
   ```bash
   ML_CFG=path/to/other_config.json python main.py
   ```

## Optional data exploration

The `data/exploring.py` module provides helper functions for inspecting a
dataset (missing values, categorical normalisation, etc.). Running it will
produce plots and a short text summary. The resulting HTML file is placed in the
`plots/eda` directory under the dataset-specific folder.

Trigger the exploration report with:
```bash
python data/exploring.py
```

## Running the pipeline

Execute the main script to run the full training workflow and generate all
metrics:
```bash
python main.py
```
Results are written under `plots/<dataset>/models` along with an HTML report
summarising scores and figures.

## Generating reports

After the pipeline finishes, an HTML report is created in
`plots/<dataset>/models/report.html`. This report contains the metrics table and
diagnostic figures for all models. If the exploration step was run beforehand,
its report appears in `plots/<dataset>/eda/`.

