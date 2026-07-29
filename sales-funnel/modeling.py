"""
Predictive modeling utilities for the sales-funnel conversion analysis.

Implements class-imbalance handling (SMOTE) and three classifiers
(Decision Tree, Random Forest, LDA), returning standardized evaluation
metrics (accuracy, sensitivity, specificity, PPV/NPV, ROC AUC) so the
Streamlit app can compare them side by side.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

MODELS = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "LDA": LinearDiscriminantAnalysis(),
}


@dataclass
class ModelResult:
    name: str
    accuracy: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    roc_auc: float
    fpr: np.ndarray
    tpr: np.ndarray
    confusion: np.ndarray
    feature_importance: pd.Series | None


def prepare_features(df: pd.DataFrame, feature_cols: list[str], target_col: str):
    X = pd.get_dummies(df[feature_cols], drop_first=True)
    y = df[target_col].values
    return X, y


def run_model(name: str, X: pd.DataFrame, y: np.ndarray, use_smote: bool = True, test_size: float = 0.3) -> ModelResult:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    if use_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    model = MODELS[name]
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    auc = roc_auc_score(y_test, y_score)
    fpr, tpr, _ = roc_curve(y_test, y_score)

    importance = None
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    elif hasattr(model, "coef_"):
        importance = pd.Series(np.abs(model.coef_[0]), index=X.columns).sort_values(ascending=False)

    return ModelResult(
        name=name,
        accuracy=accuracy,
        sensitivity=sensitivity,
        specificity=specificity,
        ppv=ppv,
        npv=npv,
        roc_auc=auc,
        fpr=fpr,
        tpr=tpr,
        confusion=np.array([[tn, fp], [fn, tp]]),
        feature_importance=importance,
    )


def run_all_models(df: pd.DataFrame, feature_cols: list[str], target_col: str, use_smote: bool = True) -> dict[str, ModelResult]:
    X, y = prepare_features(df, feature_cols, target_col)
    return {name: run_model(name, X, y, use_smote=use_smote) for name in MODELS}
