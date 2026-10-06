"""Tests for dashboard data readers and KPI calculations."""

import csv

import pandas as pd
import pytest

from app.data_access import (
    ALERT_COLUMNS,
    STREAM_LOG_COLUMNS,
    compute_kpis,
    load_alerts,
    load_stream_log,
)
from app import data_access


def _write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(columns)
        writer.writerows(rows)


def test_load_stream_log_missing_and_empty_files(tmp_path):
    missing = tmp_path / "missing.csv"
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")

    assert list(load_stream_log(str(missing)).columns) == STREAM_LOG_COLUMNS
    assert load_stream_log(str(missing)).empty
    assert list(load_stream_log(str(empty)).columns) == STREAM_LOG_COLUMNS
    assert load_stream_log(str(empty)).empty


def test_load_stream_log_ignores_partially_written_last_line(tmp_path):
    log_path = tmp_path / "stream_log.csv"
    complete_record = [
        "tx_1", "72.0", "BLOCK", "0.95", "0.2", "2.0", "50.0", "1", "2026-10-06T10:00:00Z"
    ]
    _write_csv(log_path, STREAM_LOG_COLUMNS, [complete_record])
    with log_path.open("a", encoding="utf-8", newline="") as file_obj:
        file_obj.write('tx_2,31.0,FLAG,0.4,0.1,')

    result = load_stream_log(str(log_path))

    assert len(result) == 1
    assert result.loc[0, "txn_id"] == "tx_1"


def test_load_alerts_ignores_partially_written_last_line(tmp_path):
    alert_path = tmp_path / "alerts.csv"
    complete_alert = [
        "alert_1", "tx_1", "HIGH", "72.0", "50.0", "risk score crossed BLOCK", "2026-10-06T10:00:00Z"
    ]
    _write_csv(alert_path, ALERT_COLUMNS, [complete_alert])
    with alert_path.open("a", encoding="utf-8", newline="") as file_obj:
        file_obj.write("alert_2,tx_2,MEDIUM,31.0")

    result = load_alerts(str(alert_path))

    assert len(result) == 1
    assert result.loc[0, "alert_id"] == "alert_1"


def test_compute_kpis_from_fake_stream_log():
    fake_log = pd.DataFrame(
        [
            {"decision": "APPROVE", "true_label": 0, "latency_ms": 2, "raw_amount": 10},
            {"decision": "FLAG", "true_label": 1, "latency_ms": 4, "raw_amount": 25},
            {"decision": "BLOCK", "true_label": 0, "latency_ms": 6, "raw_amount": 50},
        ]
    )

    result = compute_kpis(fake_log)

    assert result == {
        "total_processed": 3,
        "approved": 1,
        "flagged": 1,
        "blocked": 1,
        "fraud_caught": 1,
        "average_latency": 4.0,
        "p95_latency": 5.8,
        "money_blocked": 50.0,
    }


def test_get_features_for_txn_returns_a_copy_and_none_for_unknown(monkeypatch):
    cached = {"tx_17": {"V1": 0.25, "Amount": 12.0}}
    monkeypatch.setattr(data_access, "_get_test_feature_lookup", lambda: cached)

    result = data_access.get_features_for_txn("tx_17")
    result["V1"] = 999

    assert cached["tx_17"]["V1"] == 0.25
    assert data_access.get_features_for_txn("tx_404") is None


@pytest.mark.parametrize("loader,columns", [(load_stream_log, STREAM_LOG_COLUMNS), (load_alerts, ALERT_COLUMNS)])
def test_loaders_return_schema_for_missing_file(loader, columns, tmp_path):
    result = loader(str(tmp_path / "does_not_exist.csv"))
    assert list(result.columns) == columns
    assert result.empty
