import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def build_pipeline(model, config, use_pca=True, n_components=None):
    steps = [('imputer', SimpleImputer()), ('scaler', StandardScaler())]
    if use_pca and n_components:
        steps.append(('pca', PCA(n_components=n_components, random_state=config.random_state)))
    steps.append(('classifier', model))
    return Pipeline(steps)

def evaluate_model(name, model, X_train, X_test, y_train, y_test, config, n_components=None):
    use_pca = name.lower() in config.models_with_pca
    pipeline = build_pipeline(model, config, use_pca, n_components)

    scoring = ['accuracy', 'roc_auc', 'f1_weighted']
    scores = cross_validate(
        pipeline, X_train, y_train,
        scoring=scoring,
        cv=StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state),
        return_train_score=False
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = np.mean(scores['test_accuracy'])
    roc = np.mean(scores['test_roc_auc'])
    f1 = np.mean(scores['test_f1_weighted'])

    print(f"\nClassification Report: {name}")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(config.plot_dir, f'confusion_matrix_{name}.png'))
    plt.close()

    # ROC Curve
    if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title(f'ROC Curve: {name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config.plot_dir, f'roc_curve_{name}.png'))
        plt.close()

    return {
        name: {
            "Accuracy Mean": acc,
            "ROC AUC Mean": roc,
            "F1 Mean": f1
        }
    }

def train_logistic(X_train, X_test, y_train, y_test, config, n_components):
    model = LogisticRegression(max_iter=1000, random_state=config.random_state, class_weight='balanced')
    return evaluate_model("LogisticRegression", model, X_train, X_test, y_train, y_test, config, n_components)

def train_knn(X_train, X_test, y_train, y_test, config, n_components):
    model = KNeighborsClassifier()
    return evaluate_model("KNeighborsClassifier", model, X_train, X_test, y_train, y_test, config, n_components)

def train_random_forest(X_train, X_test, y_train, y_test, config, n_components):
    model = RandomForestClassifier(random_state=config.random_state, class_weight='balanced')
    return evaluate_model("RandomForestClassifier", model, X_train, X_test, y_train, y_test, config, n_components)

def train_lda(X_train, X_test, y_train, y_test, config, n_components):
    model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    return evaluate_model("LinearDiscriminantAnalysis", model, X_train, X_test, y_train, y_test, config, n_components)

def train_qda(X_train, X_test, y_train, y_test, config, n_components):
    model = QuadraticDiscriminantAnalysis(reg_param=0.1)
    return evaluate_model("QuadraticDiscriminantAnalysis", model, X_train, X_test, y_train, y_test, config, n_components)

def train_nb(X_train, X_test, y_train, y_test, config, n_components):
    model = GaussianNB()
    return evaluate_model("GaussianNB", model, X_train, X_test, y_train, y_test, config, n_components)

def train_svc(X_train, X_test, y_train, y_test, config, n_components):
    model = SVC(kernel='rbf', probability=True, random_state=config.random_state)
    return evaluate_model("SVC", model, X_train, X_test, y_train, y_test, config, n_components)

def run_sklearn_models(X_train, X_test, y_train, y_test, config):
    preproc = Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler())])
    X_proc = preproc.fit_transform(X_train)
    pca = PCA(random_state=config.random_state).fit(X_proc)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative >= config.pca_variance) + 1

    print(f"\nPCA Components for {config.pca_variance*100:.0f}% variance: {n_components}")

    model_funcs = [
        train_logistic,
        train_knn,
        train_random_forest,
        train_lda,
        train_qda,
        train_nb,
        train_svc
    ]

    summary = {}
    for model_func in model_funcs:
        results = model_func(X_train, X_test, y_train, y_test, config, n_components)
        summary.update(results)

    return summary
