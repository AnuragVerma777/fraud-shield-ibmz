"""Tests for risk-score bounds, action cutoffs, and transaction results."""

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import ALL_FEATURES
from src import risk


@pytest.mark.parametrize(
    ("xgb_prob", "anomaly_norm"),
    [(0.0, 0.0), (1.0, 1.0), (0.5, 0.7), (0.0, 1.0), (1.0, 0.0)],
)
def test_compute_risk_stays_between_zero_and_100(xgb_prob, anomaly_norm):
    score = risk.compute_risk(xgb_prob, anomaly_norm)

    assert 0.0 <= score <= 100.0


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (29.99, "APPROVE"),
        (30.0, "FLAG"),
        (69.99, "FLAG"),
        (70.0, "BLOCK"),
    ],
)
def test_decide_threshold_boundaries(score, expected):
    assert risk.decide(score, flag_threshold=30, block_threshold=70) == expected


def test_score_transaction_returns_expected_keys_and_positive_latency(
    monkeypatch, tmp_path
):
    class FakeScaler:
        def transform(self, values):
            return np.asarray(values)

    class FakeXGBoost:
        def predict_proba(self, features):
            return np.array([[0.8, 0.2]])

    fake_bundle = {"model": object(), "raw_min": -1.0, "raw_max": 1.0}
    fake_artifacts = iter([FakeXGBoost(), FakeScaler(), fake_bundle])
    monkeypatch.setattr(risk.joblib, "load", lambda _path: next(fake_artifacts))
    monkeypatch.setattr(
        risk,
        "_anomaly_score",
        lambda _bundle, features: np.array([0.4] * len(features)),
    )

    engine = risk.RiskEngine(risk_config_path=str(tmp_path / "missing.json"))
    row = {feature: 0.0 for feature in ALL_FEATURES}
    result = engine.score_transaction(row)

    assert set(result) == {
        "risk_score",
        "decision",
        "xgb_prob",
        "anomaly_score",
        "latency_ms",
    }
    assert result["latency_ms"] > 0


def test_numpy_path_matches_dataframe_path_on_500_test_rows():
    """
    Score 500 test transactions using both the legacy DataFrame path
    and the refactored pre-allocated NumPy array path.
    Assert that risk_scores match within 1e-6 and print latency comparison.
    """
    import time
    from src.data import get_splits

    _, _, X_test, _, _, _ = get_splits()
    engine = risk.RiskEngine()

    test_sample = X_test.iloc[:500].copy()
    test_sample["Time"] = test_sample["Time_original"]
    test_sample["Amount"] = test_sample["Amount_original"]
    rows = test_sample[ALL_FEATURES].to_dict(orient="records")

    # Warmup
    for r in rows[:10]:
        engine._score_transaction_dataframe(r)
        engine.score_transaction(r)

    # Benchmark legacy DataFrame path
    t0 = time.perf_counter()
    legacy_results = [engine._score_transaction_dataframe(r) for r in rows]
    t1 = time.perf_counter()
    avg_legacy_ms = (t1 - t0) * 1000 / len(rows)

    # Benchmark refactored NumPy path
    t2 = time.perf_counter()
    numpy_results = [engine.score_transaction(r) for r in rows]
    t3 = time.perf_counter()
    avg_numpy_ms = (t3 - t2) * 1000 / len(rows)

    print(
        f"\n[Latency Report] 500 Test Rows:\n"
        f"  - Legacy DataFrame path: {avg_legacy_ms:.3f} ms / transaction\n"
        f"  - Refactored NumPy path:  {avg_numpy_ms:.3f} ms / transaction\n"
        f"  - Latency Reduction:     {(avg_legacy_ms - avg_numpy_ms) / avg_legacy_ms * 100:.1f}%\n"
        f"  - Speedup:               {avg_legacy_ms / max(avg_numpy_ms, 1e-6):.2f}x"
    )

    for i in range(len(rows)):
        diff = abs(legacy_results[i]["risk_score"] - numpy_results[i]["risk_score"])
        assert diff <= 1e-6, (
            f"Row {i} risk score mismatch: legacy={legacy_results[i]['risk_score']}, "
            f"numpy={numpy_results[i]['risk_score']}, diff={diff}"
        )
        assert legacy_results[i]["decision"] == numpy_results[i]["decision"]
        assert abs(legacy_results[i]["xgb_prob"] - numpy_results[i]["xgb_prob"]) <= 1e-5
        assert (
            abs(legacy_results[i]["anomaly_score"] - numpy_results[i]["anomaly_score"])
            <= 1e-5
        )
