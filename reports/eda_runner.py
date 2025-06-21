from pathlib import Path

import pandas as pd
from config import settings
from data.loader import load_data

from data.exploring import explore_the_df, eksplore_regression
from .eda_report_generator import generate_eda_report



def run_eda(eda_dir: str | Path | None = None) -> Path:
    """Run exploratory data analysis and save outputs."""
    if eda_dir is None:
        eda_dir = Path(settings.plot_dir) / "eda"
    else:
        eda_dir = Path(eda_dir)

    X, y = load_data()
    df = pd.concat([X, y.rename("target")], axis=1)
    explore_the_df(df, "target", eda_dir)

    # optional regression-style plots if target is numeric
    if pd.api.types.is_numeric_dtype(df["target"]):
        eksplore_regression(df, "target", eda_dir)

    generate_eda_report(eda_dir)

    return eda_dir


if __name__ == "__main__":
    run_eda()
