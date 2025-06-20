# Hackaton ML Pipeline

This project contains a small machine learning pipeline. Configuration values
are stored in `config.json` and loaded via `config.settings`.

## Running the full pipeline

Run the training workflow which trains the selected models and produces plots
and a metrics report:

```bash
python main.py
```

Outputs are written to the directory specified by `plot_dir` in `config.json`.

## Exploratory data analysis

To only explore the dataset and generate an HTML data report without training
any models use the `--explore-only` flag:

```bash
python main.py --explore-only
```

The resulting report is saved as `data_report.html` in the plot directory.

