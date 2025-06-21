from __future__ import annotations

import base64
from pathlib import Path

from typing import List, Tuple


import pandas as pd
from config import settings


__all__ = ["gather_eda_assets", "generate_eda_report"]


def gather_eda_assets(eda_dir: str | Path | None = None) -> Tuple[List[str], List[str]]:
    """Return lists of inline PNG tags and HTML tables stored in ``eda_dir``."""
    if eda_dir is None:
        eda_dir = Path(settings.plot_dir) / "eda"
    else:
        eda_dir = Path(eda_dir)

    imgs: List[str] = []
    tables: List[str] = []
    if eda_dir.exists():
        for p in sorted(eda_dir.glob("*.png")):
            imgs.append(_inline_png(p))
        for csv in sorted(eda_dir.glob("*.csv")):
            df_csv = pd.read_csv(csv)
            table_html = df_csv.to_html(index=False, float_format="{:.3f}".format, border=0)
            tables.append(f"<h3>{csv.name}</h3>{table_html}")

    return imgs, tables



def _inline_png(path: Path, width: str = "420px") -> str:
    """Return an <img> tag embedding the PNG as base64."""
    with open(path, "rb") as fh:
        enc = base64.b64encode(fh.read()).decode()
    return f'<img src="data:image/png;base64,{enc}" width="{width}" />'


def generate_eda_report(
    eda_dir: str | Path | None = None,
    report_path: str | Path | None = None,
) -> Path:
    """Generate an HTML report from saved EDA outputs."""
    if eda_dir is None:
        eda_dir = Path(settings.plot_dir) / "eda"
    else:
        eda_dir = Path(eda_dir)

    if report_path is None:
        report_path = eda_dir / "eda_report.html"
    else:
        report_path = Path(report_path)


    imgs, tables = gather_eda_assets(eda_dir)


    html = f"""
    <html><head>
        <style>
            body  {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; margin-bottom: 20px; }}
            table, th, td {{ border: 1px solid #ccc; padding: 4px; text-align: center; }}
        </style>
    </head><body>
        <h1>EDA Report</h1>
        {''.join(imgs)}
        {''.join(tables)}
    </body></html>
    """

    report_path.write_text(html, encoding="utf-8")
    print(f"✓ EDA report written to {report_path.resolve()}")
    return report_path


if __name__ == "__main__":
    generate_eda_report()
