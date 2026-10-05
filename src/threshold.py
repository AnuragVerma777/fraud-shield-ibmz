"""Inspect XGBoost precision and recall at fraud decision thresholds.

Run from the project root with ``python -m src.threshold`` after training
the model with ``python -m src.train``.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from src.config import ALL_FEATURES, THRESHOLD_PLOT_PATH, XGB_MODEL_PATH, ensure_dirs
from src.data import get_splits


def main():
    ensure_dirs()
    try:
        model = joblib.load(XGB_MODEL_PATH)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Trained XGBoost model not found at {XGB_MODEL_PATH}. "
            "Run 'python -m src.train' first."
        ) from error

    _, _, X_test, _, _, y_test = get_splits()
    probabilities = model.predict_proba(X_test.loc[:, ALL_FEATURES])[:, 1]
    thresholds = np.round(np.arange(0.1, 1.0, 0.1), 1)
    results = []

    print("\nXGBoost test-set results by threshold")
    print("Threshold  Precision  Recall  Genuine blocked (FP)  Fraud missed (FN)")
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        result = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "true_negatives": int(tn),
        }
        results.append(result)
        print(
            f"{threshold:9.1f}  {precision:9.3f}  {recall:6.3f}"
            f"  {fp:20,d}  {fn:16,d}"
        )

    # Among tested thresholds, BLOCK favors precision (and uses recall as a
    # tie-breaker); FLAG favors recall (and uses precision as a tie-breaker).
    block = max(results, key=lambda row: (row["precision"], row["recall"]))
    flag = max(results, key=lambda row: (row["recall"], row["precision"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, [row["precision"] for row in results], marker="o", label="Precision")
    ax.plot(thresholds, [row["recall"] for row in results], marker="o", label="Recall")
    ax.axvline(block["threshold"], color="tab:red", linestyle="--", alpha=0.7,
               label=f"Suggested BLOCK: {block['threshold']:.1f}")
    ax.axvline(flag["threshold"], color="tab:green", linestyle=":", alpha=0.8,
               label=f"Suggested FLAG: {flag['threshold']:.1f}")
    ax.set(
        title="XGBoost Precision and Recall by Decision Threshold",
        xlabel="Fraud probability threshold",
        ylabel="Score",
        xlim=(0.08, 0.92),
        ylim=(0, 1.02),
    )
    ax.set_xticks(thresholds)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(THRESHOLD_PLOT_PATH, dpi=150)
    plt.close(fig)

    print("\nSuggested thresholds (chosen from this test-set sweep):")
    print(
        f"BLOCK at {block['threshold']:.1f}: precision {block['precision']:.3f}, "
        f"recall {block['recall']:.3f}. Of {block['false_positives']:,} "
        "legitimate payments, the model would incorrectly block them; "
        f"it would miss {block['false_negatives']:,} fraud cases."
    )
    print(
        f"FLAG at {flag['threshold']:.1f}: precision {flag['precision']:.3f}, "
        f"recall {flag['recall']:.3f}. This catches more fraud for review, "
        f"but sends {flag['false_positives']:,} legitimate payments to review "
        "as false alarms and misses "
        f"{flag['false_negatives']:,} fraud cases."
    )
    print(
        "Business trade-off: raising the BLOCK threshold means fewer genuine "
        "customers are wrongly blocked, but more fraud gets through. Lowering "
        "the FLAG threshold catches more fraud for human review, at the cost "
        "of reviewing more genuine payments."
    )
    print(f"Threshold trade-off plot saved to: {THRESHOLD_PLOT_PATH}")


if __name__ == "__main__":
    main()
