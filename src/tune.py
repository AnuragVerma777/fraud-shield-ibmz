"""Tune risk-score weights and decision thresholds on the validation set.

Run after training the XGBoost model and Isolation Forest:
``python -m src.tune``.
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.anomaly import anomaly_score
from src.config import (
    IFOREST_PATH,
    RISK_CONFIG_PATH,
    RISK_THRESHOLD_PLOT_PATH,
    XGB_MODEL_PATH,
    ensure_dirs,
)
from src.data import get_splits
from src.risk import compute_risk


WEIGHT_OPTIONS = (1.0, 0.8, 0.7, 0.6)
BLOCK_PRECISION_TARGET = 0.90
FLAG_RECALL_TARGET = 0.85


def main():
    ensure_dirs()
    xgb_model = joblib.load(XGB_MODEL_PATH)
    iforest_bundle = joblib.load(IFOREST_PATH)

    _, X_val, _, _, y_val, _ = get_splits()
    labels = np.asarray(y_val)
    xgb_probabilities = xgb_model.predict_proba(X_val)[:, 1]
    anomaly_scores = anomaly_score(iforest_bundle, X_val)

    best = None
    print("Validation PR-AUC by risk-score weights:")
    for w_xgb in WEIGHT_OPTIONS:
        w_anom = 1.0 - w_xgb
        risk_scores = np.asarray(
            [
                compute_risk(float(prob), float(anom), w_xgb, w_anom)
                for prob, anom in zip(xgb_probabilities, anomaly_scores)
            ]
        )
        pr_auc = float(average_precision_score(labels, risk_scores))
        print(f"  w_xgb={w_xgb:.1f}, w_anom={w_anom:.1f}: PR-AUC={pr_auc:.6f}")
        if best is None or pr_auc > best["pr_auc"]:
            best = {
                "w_xgb": w_xgb,
                "w_anom": w_anom,
                "pr_auc": pr_auc,
                "scores": risk_scores,
            }

    # precision_recall_curve returns thresholds in ascending score order.
    # The first threshold meeting each requirement is therefore the lowest
    # qualifying score, as requested.
    precision, recall, thresholds = precision_recall_curve(labels, best["scores"])
    block_candidates = np.flatnonzero(precision[:-1] >= BLOCK_PRECISION_TARGET)
    flag_candidates = np.flatnonzero(recall[:-1] >= FLAG_RECALL_TARGET)
    if block_candidates.size == 0:
        raise ValueError(
            "No validation score threshold reaches the required 0.90 precision "
            "for BLOCK; inspect the model or revise the target."
        )
    if flag_candidates.size == 0:
        raise ValueError(
            "No validation score threshold reaches the required 0.85 recall "
            "for FLAG; inspect the model or revise the target."
        )

    block_index = int(block_candidates[0])
    flag_index = int(flag_candidates[0])
    block_threshold = float(thresholds[block_index])
    flag_threshold = float(thresholds[flag_index])
    block_precision = float(precision[block_index])
    block_recall = float(recall[block_index])
    flag_precision = float(precision[flag_index])
    flag_recall = float(recall[flag_index])

    risk_config = {
        "w_xgb": float(best["w_xgb"]),
        "w_anom": float(best["w_anom"]),
        "flag_threshold": flag_threshold,
        "block_threshold": block_threshold,
        "validation_pr_auc": float(best["pr_auc"]),
        "threshold_targets": {
            "block_precision": BLOCK_PRECISION_TARGET,
            "flag_recall": FLAG_RECALL_TARGET,
        },
    }
    with open(RISK_CONFIG_PATH, "w", encoding="utf-8") as config_file:
        json.dump(risk_config, config_file, indent=2)
        config_file.write("\n")

    # Plot the complete validation precision/recall curves over score cutoff.
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, precision[:-1], label="Precision", color="tab:blue")
    ax.plot(thresholds, recall[:-1], label="Recall", color="tab:orange")
    ax.axhline(BLOCK_PRECISION_TARGET, color="tab:blue", linestyle=":", alpha=0.7,
               label="BLOCK precision target (0.90)")
    ax.axhline(FLAG_RECALL_TARGET, color="tab:orange", linestyle=":", alpha=0.7,
               label="FLAG recall target (0.85)")
    ax.axvline(block_threshold, color="tab:red", linestyle="--", alpha=0.8,
               label=f"BLOCK threshold ({block_threshold:.2f})")
    ax.axvline(flag_threshold, color="tab:green", linestyle="--", alpha=0.8,
               label=f"FLAG threshold ({flag_threshold:.2f})")
    ax.set(
        title="Validation Precision and Recall by Risk Score Threshold",
        xlabel="Risk score threshold (0–100)",
        ylabel="Precision / Recall",
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(RISK_THRESHOLD_PLOT_PATH, dpi=150)
    plt.close(fig)

    # At the blocking cutoff, the share of blocked transactions that are
    # genuine is one minus precision. Below the FLAG cutoff, fraud is missed.
    print(f"\nSelected weights: XGBoost={best['w_xgb']:.1f}, anomaly={best['w_anom']:.1f}")
    print(f"Best validation PR-AUC: {best['pr_auc']:.6f}")
    print(
        f"BLOCK threshold: {block_threshold:.2f} "
        f"(precision={block_precision:.3f}, recall={block_recall:.3f})"
    )
    print(
        f"FLAG threshold: {flag_threshold:.2f} "
        f"(precision={flag_precision:.3f}, recall={flag_recall:.3f})"
    )
    print(
        "Trade-off: a high-precision BLOCK cutoff limits genuine customers "
        "being rejected, but its lower recall means more fraud can escape "
        "blocking. The lower FLAG cutoff catches at least 85% of fraud for "
        "review, while creating more review work for genuine transactions."
    )
    print(f"Risk configuration saved to: {RISK_CONFIG_PATH}")
    print(f"Threshold plot saved to: {RISK_THRESHOLD_PLOT_PATH}")


if __name__ == "__main__":
    main()
