"""
tests/test_stream.py — Unit and integration tests for the Kafka-style event stream.
"""

import os
import queue
import shutil
import tempfile
import threading
import time
import numpy as np
import pandas as pd
import pytest

from src.config import ALL_FEATURES
from src.stream import (
    CSV_FIELDNAMES,
    StreamSimulator,
    TransactionConsumer,
    TransactionProducer,
    _SENTINEL,
)


class MockRiskEngine:
    """Mock risk engine to isolate stream testing from model loading."""

    def __init__(self, decisions=None):
        self.scored_features = []
        self.decision_pattern = tuple(decisions or ("FLAG",))

    def score_transaction(self, row):
        # Record input to verify true_label is kept hidden
        self.scored_features.append(dict(row))
        decision = self.decision_pattern[(len(self.scored_features) - 1) % len(self.decision_pattern)]
        return {
            "risk_score": {"APPROVE": 10.0, "FLAG": 42.5, "BLOCK": 85.0}[decision],
            "decision": decision,
            "xgb_prob": 0.45,
            "anomaly_score": 0.35,
            "latency_ms": 1.25,
        }


@pytest.fixture
def temp_dir():
    """Create a temporary directory inside the local workspace."""
    d = tempfile.mkdtemp(prefix="test_stream_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_data():
    """Create a small deterministic test dataset with 5 transactions."""
    num_rows = 5
    data = {feat: [float(i + 1) for i in range(num_rows)] for feat in ALL_FEATURES}
    data["Time"] = [10.0, 20.0, 30.0, 40.0, 50.0]
    data["Amount"] = [15.5, 25.0, 100.0, 5.0, 99.99]
    df = pd.DataFrame(data, index=[101, 102, 103, 104, 105])
    labels = pd.Series([0, 0, 1, 0, 1], index=df.index)
    return df, labels


def test_producer_event_structure(sample_data):
    """Verify that Producer produces event dicts with the required fields."""
    features_df, labels = sample_data
    q = queue.Queue()

    producer = TransactionProducer(
        event_queue=q,
        features_df=features_df,
        labels_series=labels,
        transactions_per_second=1000.0,  # Fast for unit test
        max_events=3,
    )
    producer.start()
    producer.join(timeout=2.0)

    # First event
    event1 = q.get_nowait()
    assert isinstance(event1, dict)
    assert event1["txn_id"] == "tx_101"
    assert event1["event_time"] == 10.0
    assert event1["raw_amount"] == 15.5
    assert event1["true_label"] == 0
    assert "features" in event1
    assert set(ALL_FEATURES).issubset(set(event1["features"].keys()))

    # Second event
    event2 = q.get_nowait()
    assert event2["txn_id"] == "tx_102"
    assert event2["event_time"] == 20.0

    # Third event
    event3 = q.get_nowait()
    assert event3["txn_id"] == "tx_103"
    assert event3["true_label"] == 1

    # End sentinel
    sentinel = q.get_nowait()
    assert sentinel is _SENTINEL


def test_consumer_scores_and_logs(temp_dir, sample_data):
    """Verify Consumer scores events and writes all 9 columns to CSV."""
    features_df, labels = sample_data
    q = queue.Queue()
    log_file = os.path.join(temp_dir, "stream_log.csv")
    alerts_file = os.path.join(temp_dir, "alerts.csv")
    engine = MockRiskEngine()

    consumer = TransactionConsumer(
        event_queue=q,
        risk_engine=engine,
        log_path=log_file,
        alerts_path=alerts_file,
        progress_interval=2,
    )
    consumer.start()

    # Feed an event into the queue
    row_feat = {col: 1.0 for col in ALL_FEATURES}
    q.put(
        {
            "txn_id": "tx_999",
            "event_time": 12.34,
            "raw_amount": 99.5,
            "true_label": 1,
            "features": row_feat,
        }
    )
    q.put(_SENTINEL)
    consumer.join(timeout=2.0)

    assert consumer.events_processed == 1
    assert consumer.decisions["FLAG"] == 1

    # Verify true_label was NOT leaked to the scoring engine
    assert len(engine.scored_features) == 1
    assert "true_label" not in engine.scored_features[0]

    # Verify CSV file contents
    assert os.path.exists(log_file)
    logged_df = pd.read_csv(log_file)
    assert list(logged_df.columns) == CSV_FIELDNAMES
    assert len(logged_df) == 1
    row = logged_df.iloc[0]
    assert row["txn_id"] == "tx_999"
    assert row["risk_score"] == 42.5
    assert row["decision"] == "FLAG"
    assert row["xgb_prob"] == 0.45
    assert row["anomaly_score"] == 0.35
    assert row["latency_ms"] == 1.25
    assert row["raw_amount"] == 99.5
    assert row["true_label"] == 1
    assert pd.notna(row["processed_at"])
    alerts_df = pd.read_csv(alerts_file)
    assert len(alerts_df) == 1
    assert alerts_df.iloc[0]["txn_id"] == "tx_999"


def test_stream_simulator_start_stop(temp_dir, sample_data):
    """Test full start, execution, and stop lifecycle of StreamSimulator."""
    features_df, labels = sample_data
    log_file = os.path.join(temp_dir, "stream_log.csv")
    alerts_file = os.path.join(temp_dir, "alerts.csv")
    engine = MockRiskEngine()

    simulator = StreamSimulator(
        features_df=features_df,
        labels_series=labels,
        transactions_per_second=200.0,
        max_events=4,
        log_path=log_file,
        alerts_path=alerts_file,
        risk_engine=engine,
        progress_interval=2,
    )

    simulator.start()
    simulator.join(timeout=3.0)

    assert not simulator.is_alive()
    logged_df = pd.read_csv(log_file)
    assert len(logged_df) == 4


def test_early_stop(temp_dir, sample_data):
    """Verify that calling stop() terminates the stream early and cleanly."""
    features_df, labels = sample_data
    log_file = os.path.join(temp_dir, "stream_log.csv")
    alerts_file = os.path.join(temp_dir, "alerts.csv")
    engine = MockRiskEngine()

    simulator = StreamSimulator(
        features_df=features_df,
        labels_series=labels,
        transactions_per_second=1.0,  # Slow: 1 sec per transaction
        max_events=5,
        log_path=log_file,
        alerts_path=alerts_file,
        risk_engine=engine,
    )

    simulator.start()
    time.sleep(0.3)
    simulator.stop(timeout=1.0)

    assert not simulator.is_alive()


def test_high_speed_stream_logs_300_events_and_matching_alerts(temp_dir):
    """A 300-event fast stream logs unique rows and alerts only for FLAG/BLOCK."""
    event_count = 300
    index = np.arange(10_000, 10_000 + event_count)
    data = {
        feature: np.arange(event_count, dtype=float)
        for feature in ALL_FEATURES
    }
    data["Time"] = np.arange(event_count, dtype=float)
    data["Amount"] = np.arange(event_count, dtype=float) + 1.0
    features_df = pd.DataFrame(data, index=index)
    labels = pd.Series(np.zeros(event_count, dtype=int), index=index)

    log_file = os.path.join(temp_dir, "stream_300.csv")
    alerts_file = os.path.join(temp_dir, "alerts_300.csv")
    simulator = StreamSimulator(
        features_df=features_df,
        labels_series=labels,
        transactions_per_second=100_000,
        max_events=event_count,
        log_path=log_file,
        alerts_path=alerts_file,
        risk_engine=MockRiskEngine(decisions=("APPROVE", "FLAG", "BLOCK")),
        progress_interval=0,
    )

    simulator.start()
    simulator.join(timeout=10.0)

    assert not simulator.is_alive()
    results = pd.read_csv(log_file)
    assert len(results) == event_count
    assert results["txn_id"].is_unique
    assert set(results["decision"]).issubset({"APPROVE", "FLAG", "BLOCK"})
    assert (results["latency_ms"] > 0).all()

    expected_alert_txns = set(
        results.loc[results["decision"] != "APPROVE", "txn_id"]
    )
    alerts = pd.read_csv(alerts_file)
    assert set(alerts["txn_id"]) == expected_alert_txns
    assert len(alerts) == len(expected_alert_txns)
