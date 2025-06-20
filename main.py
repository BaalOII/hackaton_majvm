"""
main.py – thin orchestrator
• Reads global parameters from config.json via `config.settings`
• Loads data with data.loader.load_data()
• Runs the selected engines
• Generates plots and an HTML report
• Saves combined metrics CSV
"""


from __future__ import annotations

from pathlib import Path
import pandas as pd
import importlib
import argparse

# reload config (snippet above) …
import config as _cfg
_cfg = importlib.reload(_cfg)
settings = _cfg.settings
import config
config.settings = settings

# — project imports —              # simple namespace from config.json
from data.loader import load_data
from data.exploring import run_data_exploration
from engines.sklearn_engine import run as run_sklearn
from engines.torch_engine import run_torch
from reports.plots import make_all_figures
from reports.report_generator import generate_report, generate_data_report


def run_pipeline():
    # 1) guarantee plot directory exists ----------------------------
    Path(settings.plot_dir).mkdir(parents=True, exist_ok=True)
    
    # 2) load data --------------------------------------------------
    X, y = load_data()                       # returns pandas objects

    # 3) train/test split ------------------------------------------
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        stratify=y,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )

    # 4) run engines -----------------------------------------------
    all_records: list[dict] = []

    if "sklearn" in settings.engines:
        print("→ Running Sklearn engine")
        all_records.extend(run_sklearn(X_tr, X_te, y_tr, y_te))

    if "torch" in settings.engines:
        print("→ Running Torch engine")
        all_records.extend(run_torch(X_tr, X_te, y_tr, y_te))

    if not all_records:
        raise ValueError("No engines selected — update engines in config.json")

    
    fig_paths = make_all_figures(all_records, X_te, y_te)
    print("✓ Plots written:", fig_paths)

    metrics_df = pd.DataFrame(all_records)
    csv_path = Path(settings.plot_dir) / "combined_model_results.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"✓ Metrics saved to {csv_path.resolve()}")

    generate_report(metrics_df, fig_paths, report_path=Path(settings.plot_dir) / "report.html")

    return metrics_df


def run_exploration():
    """Run only the data exploration workflow."""
    Path(settings.plot_dir).mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    target = settings.target_col or "target"
    df = X.copy()
    df[target] = y
    stats = run_data_exploration(df, target)
    generate_data_report(stats, report_path=Path(settings.plot_dir) / "data_report.html")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine learning pipeline")
    parser.add_argument("--explore-only", action="store_true", help="Run data exploration without training models")
    args = parser.parse_args()

    if args.explore_only:
        run_exploration()
    else:
        run_pipeline()
