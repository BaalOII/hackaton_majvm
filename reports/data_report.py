from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from config import settings

__all__ = ["generate_data_report"]


def _inline_png(path: Path, width: str = "420px") -> str:
    """Return an <img> tag with the PNG inlined as base64."""
    with open(path, "rb") as fh:
        enc = base64.b64encode(fh.read()).decode()
    return f'<img src="data:image/png;base64,{enc}" width="{width}" />'


def _to_html(value: Any) -> str:
    """Convert statistics object to HTML."""
    if isinstance(value, pd.DataFrame):
        return value.to_html(border=0, classes="stats-table")
    if isinstance(value, pd.Series):
        return value.to_frame().to_html(border=0, classes="stats-table")
    return f"<pre>{str(value)}</pre>"


def generate_data_report(
    stats: Dict[str, Any],
    plot_paths: Dict[str, str],
    *,
    report_path: str | Path = None,
) -> Path:
    """Generate an HTML report summarizing EDA results.

    Parameters
    ----------
    stats : dict
        Mapping of statistic name to value (DataFrame, Series, or text).
    plot_paths : dict
        Mapping of plot titles to saved image paths.
    report_path : str or Path, optional
        Where to write the report. Defaults to
        ``plots/<dataset>/eda/eda_report.html``.
    """
    if report_path is None:
        report_path = Path(settings.plot_dir) / "eda" / "eda_report.html"
    else:
        report_path = Path(report_path)

    # ---- statistics sections --------------------------------------
    stats_sections = []
    for title, value in stats.items():
        stats_sections.append(f"<h2>{title}</h2>{_to_html(value)}")

    # ---- figure sections -----------------------------------------
    figure_sections = []
    for title, p in plot_paths.items():
        figure_sections.append(f"<h2>{title}</h2>{_inline_png(Path(p))}")

    # ---- assemble HTML --------------------------------------------
    html = f"""
    <html><head>
        <style>
            body  {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1    {{ color: #2d3e50; }}
            table.stats-table {{ border-collapse: collapse; margin-bottom: 20px; }}
            table.stats-table th, table.stats-table td {{ border: 1px solid #ddd; padding: 6px; text-align: center; }}
            table.stats-table th {{ background-color: #f2f2f2; }}
        </style>
    </head><body>
        <h1>Data Exploration Report</h1>
        <p>Generated: {datetime.now():%Y-%m-%d %H:%M}</p>
        {''.join(stats_sections)}
        {''.join(figure_sections)}
    </body></html>
    """

    report_path.write_text(html, encoding="utf-8")
    print(f"\u2713 EDA report written to {report_path.resolve()}")
    return report_path
