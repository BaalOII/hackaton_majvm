import os

def generate_report(summary: dict, plot_dir="plots", report_path="reports/report.html"):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    html = [
        "<html><head><title>Model Evaluation Report</title></head><body>",
        "<h1>Model Evaluation Summary</h1>",
        "<table border='1' cellpadding='5'>",
        "<tr><th>Model</th><th>Accuracy</th><th>ROC AUC</th><th>F1 Score</th></tr>"
    ]

    for model, metrics in summary.items():
        html.append(
            f"<tr><td>{model}</td>"
            f"<td>{metrics['Accuracy Mean']:.4f}</td>"
            f"<td>{metrics['ROC AUC Mean']:.4f}</td>"
            f"<td>{metrics['F1 Mean']:.4f}</td></tr>"
        )

    html.append("</table>")

    html.append("<h2>Visualizations</h2>")

    for model in summary:
        for suffix in ["confusion_matrix", "roc_curve"]:
            fname = f"{suffix}_{model}.png"
            fpath = os.path.join(plot_dir, fname)
            if os.path.exists(fpath):
                html.append(f"<h3>{model} - {suffix.replace('_', ' ').title()}</h3>")
                # Use relative path from report to plots folder
                rel_path = os.path.relpath(fpath, os.path.dirname(report_path))
                html.append(f"<img src='{rel_path}' width='400'>")

    html.append("</body></html>")

    with open(report_path, "w") as f:
        f.write("\n".join(html))

    print(f"✅ Report generated at: {report_path}")
