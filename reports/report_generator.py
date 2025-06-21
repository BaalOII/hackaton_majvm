"""reports/report_generator.py – build a compact HTML report.

* **Simplified metrics table**: drops verbose columns such as `estimator`,
  `train_losses`, `val_losses`, `probs`, `preds` to avoid clutter.
* Figures grouped by model under each row so DecisionTree printout or other
  large objects never appear in the table.
"""
from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from config import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inline_png(path: Path, width: str = "420px") -> str:
    """Return an <img> tag with the PNG inlined as base64."""
    with open(path, "rb") as fh:
        enc = base64.b64encode(fh.read()).decode()
    return f'<img src="data:image/png;base64,{enc}" width="{width}" />'


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def generate_report(
    metrics_df: pd.DataFrame,
    fig_paths: Dict[str, Dict[str, str]],
    *,
    report_path: str | Path = None,
    drop_cols: List[str] | None = None,
) -> Path:
    """Generate a self‑contained HTML report.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics returned by all engines.
    fig_paths : dict
        Nested mapping from `make_all_figures()` → {model: {plot_type: path}}.
    report_path : str or Path
        Where to write the HTML file.  Defaults to ``plots/report.html``.
    drop_cols : list[str], optional
        Extra columns to drop from the metrics table.
    """
    if report_path is None:
        report_path = Path(settings.plot_dir) / "report.html"
    else:
        report_path = Path(report_path)

    # ---- tidy metrics table ---------------------------------------
    non_scalar_cols = {
        "estimator",
        "train_losses",
        "val_losses",
        "probs",
        "preds",
    }
    if drop_cols:
        non_scalar_cols.update(drop_cols)

    table_df = metrics_df.drop(columns=[c for c in non_scalar_cols if c in metrics_df.columns])

    # human‑friendly order
    preferred = [
        "model",
        "cv_test_accuracy",
        "cv_test_roc_auc",
        "cv_test_f1_weighted",
        "test_accuracy",
        "test_roc_auc",
        "test_f1_weighted",
    ]
    cols = [c for c in preferred if c in table_df.columns] + [c for c in table_df.columns if c not in preferred]
    table_df = table_df[cols]

    metrics_html = table_df.to_html(
        index=False,
        float_format="{:.3f}".format,
        border=0,
        classes="metrics-table",
    )

    # ---- figure rows ----------------------------------------------
    fig_rows = []
    for model, paths in fig_paths.items():
        imgs = "<br/>".join(_inline_png(Path(p)) for p in paths.values())
        fig_rows.append(f"<tr><td><b>{model}</b></td><td>{imgs}</td></tr>")

    # ---- assemble HTML --------------------------------------------
    html = f"""
    <html><head>
        <style>
            body  {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1    {{ color: #2d3e50; }}
            table.metrics-table {{ border-collapse: collapse; width: 100%; margin-bottom: 40px; }}
            table.metrics-table th, table.metrics-table td {{ border: 1px solid #ddd; padding: 6px; text-align: center; }}
            table.metrics-table th {{ background-color: #f2f2f2; }}
        </style>
    </head><body>
        <h1>ML Pipeline Report</h1>
        <p>Generated: {datetime.now():%Y-%m-%d %H:%M}</p>
        <h2>Metrics Summary</h2>
        {metrics_html}
        <h2>Figures</h2>
        <table border="0">{''.join(fig_rows)}</table>
    </body></html>
    """

    report_path.write_text(html, encoding="utf-8")
    print(f"✓ HTML report written to {report_path.resolve()}")
    return report_path