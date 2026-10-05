"""Train and compare fraud classifiers on the processed data.

Run from the project root with ``python -m src.train``. The data module
creates a stratified train/val/test split and scales Time and Amount using
only the training partition before returning the processed features.

The validation set can be used for threshold tuning or early stopping;
final metrics reported here are always on the **test set**.
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.config import (
    CM_PLOT_PATH,
    ALL_FEATURES,
    METRICS_PATH,
    PR_CURVE_PATH,
    XGB_MODEL_PATH,
    XGB_PARAMS,
    ensure_dirs,
)
from src.data import get_splits


def evaluate_model(name, model, X_test, y_test):
    """Print and return test metrics, confusion matrix, and fraud scores."""
    # Test frames carry unscaled reporting columns; pass only trained features.
    model_features = X_test.loc[:, ALL_FEATURES]
    probabilities = model.predict_proba(model_features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    metrics = {
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": matrix.tolist(),
    }

    print(f"\n--- {name} (test set, threshold=0.5) ---")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(matrix)
    return metrics, probabilities, matrix


def main():
    ensure_dirs()

    # Unpack all six return values; val set reserved for later use
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits()

    # Accuracy can look excellent when almost every transaction is legitimate,
    # even if the model misses most fraud; precision, recall, F1, and AUC are
    # more informative for this highly imbalanced dataset.
    logistic = LogisticRegression(class_weight="balanced", max_iter=1000)
    logistic.fit(X_train, y_train)

    positive_count = int(np.count_nonzero(y_train == 1))
    negative_count = int(np.count_nonzero(y_train == 0))
    if positive_count == 0:
        raise ValueError("Training data must contain at least one fraud case.")

    xgb_params = dict(XGB_PARAMS)
    xgb_params["scale_pos_weight"] = negative_count / positive_count
    # This parameter was used before SMOTE, so remove the obsolete option for
    # compatibility with current XGBoost releases.
    xgb_params.pop("use_label_encoder", None)
    xgboost = XGBClassifier(**xgb_params)
    xgboost.fit(X_train, y_train)

    lr_metrics, lr_scores, _ = evaluate_model(
        "Logistic Regression", logistic, X_test, y_test
    )
    xgb_metrics, xgb_scores, xgb_matrix = evaluate_model(
        "XGBoost", xgboost, X_test, y_test
    )

    # Compare both models on one precision-recall plot.
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, scores, metrics in (
        ("Logistic Regression", lr_scores, lr_metrics),
        ("XGBoost", xgb_scores, xgb_metrics),
    ):
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ax.plot(recall, precision, label=f"{label} (PR-AUC={metrics['pr_auc']:.3f})")
    ax.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PR_CURVE_PATH, dpi=150)
    plt.close(fig)

    # Save the confusion matrix for XGBoost only, with counts shown in cells.
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=xgb_matrix,
        display_labels=["Legitimate", "Fraud"],
    ).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("XGBoost Confusion Matrix (threshold=0.5)")
    fig.tight_layout()
    fig.savefig(CM_PLOT_PATH, dpi=150)
    plt.close(fig)

    joblib.dump(xgboost, XGB_MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(
            {
                "split": "60/20/20 (train/val/test)",
                "evaluated_on": "test",
                "threshold": 0.5,
                "xgboost_scale_pos_weight": xgb_params["scale_pos_weight"],
                "train_rows": int(len(y_train)),
                "val_rows": int(len(y_val)),
                "test_rows": int(len(y_test)),
                "models": {
                    "logistic_regression": lr_metrics,
                    "xgboost": xgb_metrics,
                },
            },
            metrics_file,
            indent=2,
        )
        metrics_file.write("\n")

    print(f"\nXGBoost model saved to: {XGB_MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Precision-recall plot saved to: {PR_CURVE_PATH}")
    print(f"XGBoost confusion matrix plot saved to: {CM_PLOT_PATH}")


if __name__ == "__main__":
    main()
