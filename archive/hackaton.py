# Enhanced Machine Learning Pipeline with Config & Improvements

import os
import json
import warnings
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import torch_runner
# ------------------ CONFIGURATION ------------------ #
@dataclass
class Config:
    random_state: int = 42
    test_size: float = 0.2
    pca_variance: float = 0.95
    cv_folds: int = 5
    plot_dir: str = "plots"
    models_with_pca: list = (
        'logisticregression',
        'kneighborsclassifier',
        'lineardiscriminantanalysis',
        'quadraticdiscriminantanalysis',
        'gaussiannb',
        'svc'
    )

    def save(self, path="config.json"):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @staticmethod
    def load(path="config.json"):
        with open(path, "r") as f:
            data = json.load(f)
        return Config(**data)

try:
    config = Config.load()
except FileNotFoundError:
    config = Config()
    config.save()

os.makedirs(config.plot_dir, exist_ok=True)

# ------------------ LOGGING & WARNINGS ------------------ #
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------ DATA PROCESSING ------------------ #
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Processing data (imputation and basic cleaning)")
    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)
    return df

def data_split(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=config.test_size, random_state=config.random_state, stratify=y)

# ------------------ EDA & VALIDATION ------------------ #
def data_validation_report(df: pd.DataFrame, target_col: str):
    logging.info("Running data validation report.")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(df[target_col].value_counts(normalize=True))

    numeric_cols = df.select_dtypes(include=np.number).columns.drop(target_col, errors='ignore')
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)].shape[0]
        print(f"{col}: {outliers} outliers")

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), cmap='coolwarm', annot=False)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(config.plot_dir, 'correlation_matrix.png'))
    plt.close()

# ------------------ PCA ------------------ #
def plot_pca_variance(X: pd.DataFrame):
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(random_state=config.random_state).fit(X_scaled)
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    n_components_needed = np.argmax(cumulative_var >= config.pca_variance) + 1

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_var)+1), explained_var, marker='o', label='Explained Variance')
    plt.plot(range(1, len(cumulative_var)+1), cumulative_var, marker='s', label='Cumulative Variance')
    plt.axhline(y=config.pca_variance, color='red', linestyle='--')
    plt.axvline(x=n_components_needed, color='green', linestyle='--')
    plt.legend()
    plt.title('PCA Variance')
    plt.tight_layout()
    plt.savefig(os.path.join(config.plot_dir, 'pca_variance.png'))
    plt.close()

    logging.info(f"PCA suggests {n_components_needed} components for {config.pca_variance*100:.0f}% variance.")
    return n_components_needed

# ------------------ MODEL BUILDING ------------------ #
def build_pipeline(model: Any, n_components: int) -> Pipeline:
    steps = [('imputer', SimpleImputer()), ('scaler', StandardScaler())]
    model_name = model.__class__.__name__.lower()
    if model_name in config.models_with_pca:
        steps.append(('pca', PCA(n_components=n_components, random_state=config.random_state)))
    steps.append(('classifier', model))
    return Pipeline(steps)

# ------------------ MODEL EVALUATION ------------------ #
def evaluate_model_cv(X, y, model_pipeline):
    scoring = ['accuracy', 'roc_auc', 'f1_weighted', 'precision_weighted', 'recall_weighted']
    scores = cross_validate(model_pipeline, X, y, 
                            cv=StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state), 
                            scoring=scoring, return_train_score=True)
    return scores

# ------------------ REPORT GENERATION ------------------ #
def generate_html_report(summary: Dict[str, Dict[str, float]], report_path: str):
    html = ["<html><head><title>Model Evaluation Report</title></head><body>"]
    html.append("<h1>Model Evaluation Summary</h1>")
    html.append("<table border='1' cellpadding='5'><tr><th>Model</th><th>Accuracy Mean</th><th>Accuracy Std</th><th>ROC AUC Mean</th><th>F1 Mean</th></tr>")
    for model, scores in summary.items():
        html.append(f"<tr><td>{model}</td><td>{scores['Accuracy Mean']:.4f}</td><td>{scores['Accuracy Std']:.4f}</td><td>{scores['ROC AUC Mean']:.4f}</td><td>{scores['F1 Mean']:.4f}</td></tr>")
    html.append("</table>")

    html.append("<h2>Model Visualizations</h2>")
    model_names = summary.keys()
    for model in model_names:
        html.append(f"<h3>{model}</h3>")
        for suffix in ["confusion_matrix", "roc_curve"]:
            filename = f"{suffix}_{model}.png"
            file_path = os.path.join(config.plot_dir, filename)
            if os.path.exists(file_path):
                html.append(f"<p><img src='{filename}' width='400'><br>{filename}</p>")

    html.append("</body></html>")

    with open(report_path, "w") as f:
        f.write("".join(html))


# ------------------ MAIN EXECUTION ------------------ #
if __name__ == "__main__":
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame

    data_validation_report(df, target_col='target')
    df_clean = process_data(df)
    X_train, X_test, y_train, y_test = data_split(df_clean, target_col='target')

    n_components = plot_pca_variance(X_train)

    models = [
        LogisticRegression(max_iter=1000, random_state=config.random_state, class_weight='balanced'),
        KNeighborsClassifier(),
        RandomForestClassifier(random_state=config.random_state, class_weight='balanced'),
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
        QuadraticDiscriminantAnalysis(reg_param=0.1),
        GaussianNB(),
        SVC(kernel='rbf', probability=True, random_state=config.random_state)
    ]

    summary = {}
    for model in models:
        name = model.__class__.__name__
        pipeline = build_pipeline(model, n_components)
        scores = evaluate_model_cv(X_train, y_train, pipeline)

        summary[name] = {
            'Accuracy Mean': np.mean(scores['test_accuracy']),
            'Accuracy Std': np.std(scores['test_accuracy']),
            'ROC AUC Mean': np.mean(scores['test_roc_auc']),
            'F1 Mean': np.mean(scores['test_f1_weighted'])
        }

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(config.plot_dir, f'confusion_matrix_{name}.png'))
        plt.close()

        if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.title(f'ROC Curve: {name}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(config.plot_dir, f'roc_curve_{name}.png'))
            plt.close()

    results_df = pd.DataFrame(summary).T.sort_values(by='ROC AUC Mean', ascending=False)
    print("\nModel Evaluation Summary (5-Fold CV):")
    print(results_df.round(4))
    results_df.to_csv(os.path.join(config.plot_dir, "model_comparison_summary.csv"))
    generate_html_report(summary, os.path.join(config.plot_dir, "report.html"))
    print("Report generated at:", os.path.join(config.plot_dir, "report.html"))
