"""Evaluate XGBoost and the configured combined risk engine on test data.

Run from the project root with ``python -m src.evaluate`` after training
the models and tuning ``models/risk_config.json``.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from src.anomaly import anomaly_score
from src.config import (
    ALL_FEATURES,
    DAY2_DECISION_PLOT_PATH,
    DAY2_PERFORMANCE_PLOT_PATH,
    METRICS_DAY2_PATH,
    ORIGINAL_RAW_FEATURES,
    RAW_FEATURES,
    RISK_CONFIG_PATH,
    ensure_dirs,
)
from src.data import get_splits
from src.risk import RiskEngine


DECISIONS = ("APPROVE", "FLAG", "BLOCK")
ACTUAL_LABELS = ("Genuine", "Fraud")
LATENCY_SAMPLE_SIZE = 1_000


def main():
    ensure_dirs()
    if not os.path.isfile(RISK_CONFIG_PATH):
        raise FileNotFoundError(
            f"Tuned risk config not found at {RISK_CONFIG_PATH}. "
            "Run 'python -m src.tune' first."
        )
    _, _, X_test, _, _, y_test = get_splits()
    labels = np.asarray(y_test, dtype=int)

    # Test rows include scaled model features and separate original Time and
    # Amount columns for reports. Keep the model's trained columns explicit.
    engine = RiskEngine()
    # Avoid spawning all available workers for every single-row prediction;
    # this keeps per-transaction latency measurements representative and
    # prevents parallel-startup overhead from dominating the benchmark.
    if hasattr(engine.xgb_model, "set_params"):
        engine.xgb_model.set_params(n_jobs=1)
    if hasattr(engine.if_bundle.get("model"), "set_params"):
        engine.if_bundle["model"].set_params(n_jobs=1)
    model_test = X_test.loc[:, ALL_FEATURES]
    raw_test = model_test.copy()
    raw_test.loc[:, RAW_FEATURES] = X_test.loc[:, ORIGINAL_RAW_FEATURES].to_numpy()

    # Score the full test set in batches through the same model, anomaly
    # scorer, weights, and decision cutoffs held by RiskEngine. Batch scoring
    # keeps evaluation practical; individual-request latency is measured below
    # on a reproducible sample using score_transaction itself.
    with threadpool_limits(limits=1):
        xgb_probabilities = engine.xgb_model.predict_proba(model_test)[:, 1]
        anomaly_scores = np.asarray(anomaly_score(engine.if_bundle, model_test))
    risk_scores = np.clip(
        engine.w_xgb * xgb_probabilities + engine.w_anom * anomaly_scores,
        0.0,
        1.0,
    ) * 100.0
    decisions = np.asarray(
        [
            "BLOCK" if score >= engine.block_threshold
            else "FLAG" if score >= engine.flag_threshold
            else "APPROVE"
            for score in risk_scores
        ]
    )

    sample_size = min(LATENCY_SAMPLE_SIZE, len(raw_test))
    sample_indices = np.linspace(0, len(raw_test) - 1, sample_size, dtype=int)
    with threadpool_limits(limits=1):
        latency_results = [
            engine.score_transaction(raw_test.iloc[index].to_dict())
            for index in sample_indices
        ]
    latencies_ms = np.asarray([result["latency_ms"] for result in latency_results])

    xgb_metrics = {
        "pr_auc": float(average_precision_score(labels, xgb_probabilities)),
        "roc_auc": float(roc_auc_score(labels, xgb_probabilities)),
    }
    combined_metrics = {
        "pr_auc": float(average_precision_score(labels, risk_scores)),
        "roc_auc": float(roc_auc_score(labels, risk_scores)),
    }

    # Rows are actual labels and columns are the three operational actions.
    decision_matrix = np.zeros((2, 3), dtype=int)
    for actual, decision in zip(labels, decisions):
        decision_matrix[int(actual), DECISIONS.index(decision)] += 1

    fraud_mask = labels == 1
    genuine_mask = labels == 0
    fraud_blocked = int(np.count_nonzero(fraud_mask & (decisions == "BLOCK")))
    fraud_flagged = int(np.count_nonzero(fraud_mask & (decisions == "FLAG")))
    fraud_approved = int(np.count_nonzero(fraud_mask & (decisions == "APPROVE")))
    genuine_blocked = int(np.count_nonzero(genuine_mask & (decisions == "BLOCK")))
    genuine_flagged = int(np.count_nonzero(genuine_mask & (decisions == "FLAG")))
    genuine_approved = int(np.count_nonzero(genuine_mask & (decisions == "APPROVE")))

    comparison = {
        "pr_auc_delta_combined_minus_xgboost": combined_metrics["pr_auc"] - xgb_metrics["pr_auc"],
        "roc_auc_delta_combined_minus_xgboost": combined_metrics["roc_auc"] - xgb_metrics["roc_auc"],
    }
    combined_better_on_both = (
        comparison["pr_auc_delta_combined_minus_xgboost"] >= 0
        and comparison["roc_auc_delta_combined_minus_xgboost"] >= 0
        and (
            comparison["pr_auc_delta_combined_minus_xgboost"] > 0
            or comparison["roc_auc_delta_combined_minus_xgboost"] > 0
        )
    )

    with open(RISK_CONFIG_PATH, "r", encoding="utf-8") as config_file:
        risk_config = json.load(config_file)

    metrics = {
        "evaluated_on": "held-out test set",
        "test_rows": int(len(labels)),
        "risk_config": {
            "w_xgb": float(engine.w_xgb),
            "w_anom": float(engine.w_anom),
            "flag_threshold": float(engine.flag_threshold),
            "block_threshold": float(engine.block_threshold),
            "validation_pr_auc": risk_config.get("validation_pr_auc"),
            "source": RISK_CONFIG_PATH,
        },
        "ranking_metrics": {
            "xgboost_alone": xgb_metrics,
            "combined_risk": combined_metrics,
            "combined_minus_xgboost": comparison,
            "combined_better_on_both_auc_metrics": bool(combined_better_on_both),
        },
        "three_decision_confusion_matrix": {
            "actual_labels": list(ACTUAL_LABELS),
            "predicted_decisions": list(DECISIONS),
            "counts": decision_matrix.tolist(),
        },
        "outcomes": {
            "fraud_caught_block": fraud_blocked,
            "fraud_caught_flag": fraud_flagged,
            "fraud_caught_block_plus_flag": fraud_blocked + fraud_flagged,
            "fraud_missed_approved": fraud_approved,
            "genuine_customers_blocked": genuine_blocked,
            "genuine_customers_flagged": genuine_flagged,
            "genuine_customers_approved": genuine_approved,
        },
        "latency_ms_per_transaction": {
            "average": float(np.mean(latencies_ms)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "measurements": int(len(latencies_ms)),
            "measurement_method": "individual RiskEngine calls on an evenly spaced test-set sample",
        },
        "anomaly_value_note": (
            "Anomaly detection can surface transactions that differ from known "
            "legitimate behavior, including patterns not represented in the "
            "supervised model's fraud examples; this is a complementary signal, "
            "not proof that unseen fraud was caught in this test."
        ),
    }

    with open(METRICS_DAY2_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")

    # Plot the ranking metrics side by side for a quick model comparison.
    metric_names = ("PR-AUC", "ROC-AUC")
    xgb_values = [xgb_metrics["pr_auc"], xgb_metrics["roc_auc"]]
    combined_values = [combined_metrics["pr_auc"], combined_metrics["roc_auc"]]
    positions = np.arange(len(metric_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(positions - width / 2, xgb_values, width, label="XGBoost alone")
    ax.bar(positions + width / 2, combined_values, width, label="Combined risk")
    ax.set_xticks(positions, metric_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Held-Out Test Ranking Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DAY2_PERFORMANCE_PLOT_PATH, dpi=150)
    plt.close(fig)

    # Plot the requested two-by-three decision confusion matrix.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(decision_matrix, cmap="Blues")
    ax.set_xticks(range(len(DECISIONS)), DECISIONS)
    ax.set_yticks(range(len(ACTUAL_LABELS)), ACTUAL_LABELS)
    ax.set_xlabel("System decision")
    ax.set_ylabel("Actual class")
    ax.set_title("Test Set: Actual Class vs Risk Decision")
    for row in range(decision_matrix.shape[0]):
        for col in range(decision_matrix.shape[1]):
            ax.text(col, row, f"{decision_matrix[row, col]:,}",
                    ha="center", va="center",
                    color="white" if decision_matrix[row, col] > decision_matrix.max() / 2 else "black")
    fig.colorbar(image, ax=ax, label="Transactions")
    fig.tight_layout()
    fig.savefig(DAY2_DECISION_PLOT_PATH, dpi=150)
    plt.close(fig)

    print("\nHeld-out test ranking metrics")
    print("Model              PR-AUC    ROC-AUC")
    print(f"XGBoost alone      {xgb_metrics['pr_auc']:.4f}    {xgb_metrics['roc_auc']:.4f}")
    print(f"Combined risk      {combined_metrics['pr_auc']:.4f}    {combined_metrics['roc_auc']:.4f}")
    print("\nThree-decision confusion matrix (rows=actual, columns=decision):")
    print(f"                   {DECISIONS}")
    print(f"Genuine            {decision_matrix[0].tolist()}")
    print(f"Fraud               {decision_matrix[1].tolist()}")
    print(
        f"Fraud caught (BLOCK + FLAG): {fraud_blocked + fraud_flagged:,}; "
        f"fraud missed (APPROVE): {fraud_approved:,}."
    )
    print(
        f"Genuine customers blocked: {genuine_blocked:,}; "
        f"flagged for review: {genuine_flagged:,}."
    )
    print(
        f"Latency: average {metrics['latency_ms_per_transaction']['average']:.3f} ms, "
        f"p95 {metrics['latency_ms_per_transaction']['p95']:.3f} ms per transaction."
    )
    if combined_better_on_both:
        print("Combined risk is better on both PR-AUC and ROC-AUC for this test set.")
    else:
        print(
            "Combined risk is not better on both PR-AUC and ROC-AUC than XGBoost "
            "alone on this test set. Anomaly detection can still add value by "
            "surfacing unusual behavior that differs from known legitimate "
            "patterns, including potential new fraud patterns; this evaluation "
            "does not prove that it caught previously unseen fraud."
        )
    print(f"Metrics saved to: {METRICS_DAY2_PATH}")
    print(f"Charts saved to: {DAY2_PERFORMANCE_PLOT_PATH}, {DAY2_DECISION_PLOT_PATH}")


if __name__ == "__main__":
    main()
