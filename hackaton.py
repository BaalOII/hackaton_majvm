import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, confusion_matrix,
    classification_report
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# 1. Data Processing
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Data Info:")
    print(df.info())
    print("\nMissing Values Proportion:")
    print(df.isna().mean())
    print("\nData Types:")
    print(df.dtypes)
    return df.dropna()  # You could replace this with more advanced imputation

# 2. Linear Regression
def perform_linear_regression(df: pd.DataFrame, target_col: str) -> RegressorMixin:
    X: pd.DataFrame = df.drop(columns=[target_col])
    y: pd.Series = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred: np.ndarray = model.predict(X_test)

    print("\nLinear Regression:")
    print(f"R2 Score: {r2_score(y_test, y_pred):.3f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")
    return model

# 3. K-Nearest Neighbors Classifier
def perform_knn(df: pd.DataFrame, target_col: str, k: int = 3) -> ClassifierMixin:
    X: pd.DataFrame = df.drop(columns=[target_col])
    y: pd.Series = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled: np.ndarray = scaler.fit_transform(X_train)
    X_test_scaled: np.ndarray = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    y_pred: np.ndarray = model.predict(X_test_scaled)

    print("\nKNN Classification:")
    print(classification_report(y_test, y_pred))
    return model

# 4. Logistic Regression
def perform_logistic_regression(df: pd.DataFrame, target_col: str) -> ClassifierMixin:
    X: pd.DataFrame = df.drop(columns=[target_col])
    y: pd.Series = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled: np.ndarray = scaler.fit_transform(X_train)
    X_test_scaled: np.ndarray = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_pred: np.ndarray = model.predict(X_test_scaled)

    print("\nLogistic Regression:")
    print(classification_report(y_test, y_pred))
    return model

# 5. LDA and QDA
def perform_lda_qda(df: pd.DataFrame, target_col: str) -> Tuple[ClassifierMixin, ClassifierMixin]:
    X: pd.DataFrame = df.drop(columns=[target_col])
    y: pd.Series = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nLDA:")
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    lda_pred: np.ndarray = lda.predict(X_test)
    print(classification_report(y_test, lda_pred))

    print("\nQDA:")
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    qda_pred: np.ndarray = qda.predict(X_test)
    print(classification_report(y_test, qda_pred))

    return lda, qda

# 6. Simple ML Model: Random Forest
def perform_random_forest(df: pd.DataFrame, target_col: str) -> ClassifierMixin:
    X: pd.DataFrame = df.drop(columns=[target_col])
    y: pd.Series = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred: np.ndarray = model.predict(X_test)

    print("\nRandom Forest Classifier:")
    print(classification_report(y_test, y_pred))
    return model

# 7. Plot Results
def plot_results(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray], task_type: str = 'classification') -> None:
    if task_type == 'classification':
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()
    elif task_type == 'regression':
        plt.scatter(y_true, y_pred)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Regression Results")
        plt.show()


cancer = load_breast_cancer(as_frame=True)
df_cancer = cancer.frame

df_clean = process_data(df_cancer)  # Drops NAs, prints types and missing data

target = 'target'

regression = perform_logistic_regression(df_clean, target_col=target)
knn = perform_knn(df_clean, target_col=target, k=5)
random_forest = perform_random_forest(df_clean, target_col=target)
logistic_regression = perform_logistic_regression(df_clean, target_col=target)
lda =