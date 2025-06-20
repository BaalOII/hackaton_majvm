import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from models.sklearn_runner import run_sklearn_models
from models.torch_runner import TorchModelRunner
from reports.report_generator import generate_report
from config import Config

def main():
    # Load config
    config = Config.load()
    os.makedirs(config.plot_dir, exist_ok=True)

    # Load dataset
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    X = df.drop(columns='target')
    y = df['target']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=config.random_state
    )

    # Run sklearn models
    print("\nRunning Scikit-learn models...")
    sklearn_results = run_sklearn_models(X_train, X_test, y_train, y_test, config)

    # Run PyTorch model
    print("\nRunning PyTorch model...")
    torch_runner = TorchModelRunner(X_train, X_test, y_train, y_test, plot_dir=config.plot_dir)
    torch_results = torch_runner.run()

    # Combine all results
    all_results = sklearn_results.copy()
    all_results.update(torch_results)

    # Save summary CSV
    results_df = pd.DataFrame(all_results).T.sort_values(by="ROC AUC Mean", ascending=False)
    results_csv_path = os.path.join(config.plot_dir, "combined_model_results.csv")
    results_df.to_csv(results_csv_path)
    print(f"\nResults saved to {results_csv_path}")

    # Generate report
    report_path = os.path.join("reports", "report.html")
    generate_report(all_results, plot_dir=config.plot_dir, report_path=report_path)

if __name__ == "__main__":
    main()
