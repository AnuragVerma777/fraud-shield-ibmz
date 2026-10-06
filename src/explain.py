"""SHAP explanations for individual transactions and global feature impact.

Run ``python -m src.explain`` from the repository root to save a summary plot
using up to 2,000 validation rows.
"""

from functools import lru_cache
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    ALL_FEATURES,
    RANDOM_STATE,
    SHAP_SUMMARY_PLOT_PATH,
    XGB_MODEL_PATH,
    ensure_dirs,
)


# ULB's V1–V28 columns are anonymized PCA components. A production bank would
# explain named fields such as "new payee" or "device change" instead.
FEATURE_LABELS = {
    **{f"V{i}": f"Behavior signal V{i}" for i in range(1, 29)},
    "Amount": "Transaction amount",
    "Time": "Transaction time",
}


@lru_cache(maxsize=1)
def load_explainer():
    """Load the trained XGBoost model and cache its TreeExplainer once."""
    import joblib
    import shap

    model = joblib.load(XGB_MODEL_PATH)
    return shap.TreeExplainer(model)


def _as_feature_frame(feature_row: Any) -> pd.DataFrame:
    """Convert one transaction to a one-row frame in model feature order."""
    if isinstance(feature_row, pd.DataFrame):
        if len(feature_row) != 1:
            raise ValueError("feature_row must contain exactly one transaction")
        row = feature_row
    elif isinstance(feature_row, pd.Series):
        row = feature_row.to_frame().T
    elif isinstance(feature_row, dict):
        row = pd.DataFrame([feature_row])
    else:
        values = np.asarray(feature_row)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[0] != 1:
            raise ValueError("feature_row must contain exactly one transaction")
        if values.shape[1] != len(ALL_FEATURES):
            raise ValueError(
                f"Expected {len(ALL_FEATURES)} model features, received {values.shape[1]}"
            )
        row = pd.DataFrame(values, columns=ALL_FEATURES)

    missing = [feature for feature in ALL_FEATURES if feature not in row.columns]
    if missing:
        raise ValueError(f"feature_row is missing model features: {missing}")
    return row.loc[:, ALL_FEATURES]


def _positive_class_values(shap_result: Any) -> np.ndarray:
    """Normalize common SHAP return formats to (rows, features) values."""
    if isinstance(shap_result, list):
        # Older SHAP versions return one array per class for classifiers.
        shap_result = shap_result[1] if len(shap_result) > 1 else shap_result[0]
    elif hasattr(shap_result, "values"):
        # Newer SHAP Explanation objects expose contributions on ``values``.
        shap_result = shap_result.values

    values = np.asarray(shap_result)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    elif values.ndim == 3:
        # Some SHAP releases put the class axis last; others put it first.
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        elif values.shape[0] == 2:
            values = values[1]
        else:
            raise ValueError(f"Cannot identify the positive-class SHAP values: {values.shape}")
    if values.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values, received shape {values.shape}")
    return values


def explain_transaction(feature_row, top_k: int = 5) -> list[dict]:
    """Return the strongest positive- and negative-risk feature contributions."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    features = _as_feature_frame(feature_row)
    contributions = _positive_class_values(
        load_explainer().shap_values(features)
    )[0]
    if len(contributions) != len(ALL_FEATURES):
        raise ValueError(
            f"Expected {len(ALL_FEATURES)} SHAP values, received {len(contributions)}"
        )

    selected = np.argsort(np.abs(contributions))[::-1][: min(top_k, len(ALL_FEATURES))]
    explanation = []
    for index in selected:
        feature_name = ALL_FEATURES[int(index)]
        shap_value = float(contributions[index])
        explanation.append(
            {
                "feature": FEATURE_LABELS[feature_name],
                "value": float(features.iloc[0][feature_name]),
                "shap_value": shap_value,
                "direction": "pushes risk UP" if shap_value >= 0 else "pushes risk DOWN",
            }
        )
    return explanation


def _join_feature_names(names: list[str]) -> str:
    """Format feature labels naturally for a short explanation sentence."""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def to_plain_language(explanation: list[dict]) -> str:
    """Turn ranked SHAP contributions into two short, plain-language sentences."""
    if not explanation:
        return "There are no feature contributions to explain."

    upward = [item["feature"] for item in explanation if item["shap_value"] > 0][:3]
    downward = [item["feature"] for item in explanation if item["shap_value"] < 0][:3]

    if upward:
        first = f"Risk is pushed up mainly by {_join_feature_names(upward)}."
    else:
        first = "The strongest listed features do not raise the risk score."

    if downward:
        second = f"Signals pushing risk down include {_join_feature_names(downward)}."
    else:
        second = "The listed features do not show a strong signal lowering risk."

    return f"{first} {second}"


def save_summary_plot(sample_size: int = 2_000) -> str:
    """Save a global SHAP summary plot for a reproducible validation sample."""
    from src.data import get_splits

    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")

    _, X_val, _, _, _, _ = get_splits()
    sample = X_val.loc[:, ALL_FEATURES].sample(
        n=min(sample_size, len(X_val)), random_state=RANDOM_STATE
    )
    shap_values = _positive_class_values(load_explainer().shap_values(sample))
    if shap_values.shape != sample.shape:
        raise ValueError(
            f"SHAP values shape {shap_values.shape} does not match sample {sample.shape}"
        )

    import shap

    ensure_dirs()
    shap.summary_plot(shap_values, sample, max_display=15, show=False)
    plt.gcf().tight_layout()
    plt.savefig(SHAP_SUMMARY_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    return SHAP_SUMMARY_PLOT_PATH


def main() -> None:
    """Generate the project-wide SHAP plot from validation transactions."""
    path = save_summary_plot(sample_size=2_000)
    print(f"SHAP summary plot saved to: {path}")


if __name__ == "__main__":
    main()
