"""Tests for transaction-level SHAP explanation formatting."""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import ALL_FEATURES
from src import explain


class FakeExplainer:
    """Return predictable contributions without loading model artifacts."""

    def __init__(self, values):
        self.values = np.asarray(values)

    def shap_values(self, _features):
        return self.values


def test_explanation_has_expected_structure_and_direction(monkeypatch):
    contributions = np.zeros(len(ALL_FEATURES))
    contributions[ALL_FEATURES.index("V14")] = 0.8
    contributions[ALL_FEATURES.index("Amount")] = -0.4
    monkeypatch.setattr(
        explain, "load_explainer", lambda: FakeExplainer(contributions)
    )
    row = {feature: float(index) for index, feature in enumerate(ALL_FEATURES)}

    result = explain.explain_transaction(row, top_k=2)

    assert len(result) == 2
    assert all(
        set(item) == {"feature", "value", "shap_value", "direction"}
        for item in result
    )
    assert result[0]["feature"] == "Behavior signal V14"
    assert result[0]["direction"] == "pushes risk UP"
    assert result[1]["feature"] == "Transaction amount"
    assert result[1]["direction"] == "pushes risk DOWN"


def test_top_k_items_are_sorted_by_absolute_shap_value(monkeypatch):
    contributions = np.zeros(len(ALL_FEATURES))
    contributions[ALL_FEATURES.index("V1")] = 0.1
    contributions[ALL_FEATURES.index("V4")] = -0.9
    contributions[ALL_FEATURES.index("V14")] = 0.5
    monkeypatch.setattr(
        explain, "load_explainer", lambda: FakeExplainer(contributions)
    )
    row = {feature: 0.0 for feature in ALL_FEATURES}

    result = explain.explain_transaction(row, top_k=3)

    magnitudes = [abs(item["shap_value"]) for item in result]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert [item["feature"] for item in result] == [
        "Behavior signal V4",
        "Behavior signal V14",
        "Behavior signal V1",
    ]
