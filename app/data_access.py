"""Data access helpers shared by the Fraud Shield dashboard.

This module contains no Streamlit widgets or rendering logic. Streamlit is used
only to cache the immutable test-set lookup across dashboard reruns.
"""

import csv
from functools import lru_cache
import io

import numpy as np
import pandas as pd

from src.config import (
    ALERTS_PATH,
    ALL_FEATURES,
    ORIGINAL_RAW_FEATURES,
    RAW_FEATURES,
    STREAM_LOG_PATH,
)

try:
    import streamlit as st
except ImportError:  # Keep these helpers testable without the dashboard package.
    st = None


STREAM_LOG_COLUMNS = [
    "txn_id",
    "risk_score",
    "decision",
    "xgb_prob",
    "anomaly_score",
    "latency_ms",
    "raw_amount",
    "true_label",
    "processed_at",
]
ALERT_COLUMNS = [
    "alert_id",
    "txn_id",
    "severity",
    "risk_score",
    "raw_amount",
    "reason",
    "timestamp",
]


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    """Return an empty frame with the known log schema."""
    return pd.DataFrame(columns=columns)


def _read_csv_safely(path: str, empty_columns: list[str]) -> pd.DataFrame:
    """Read complete CSV records, ignoring an incomplete final write."""
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as csv_file:
            contents = csv_file.read()
    except FileNotFoundError:
        return _empty_frame(empty_columns)

    if not contents.strip():
        return _empty_frame(empty_columns)

    # The stream writer ends each flushed CSV record with a newline. If a
    # process stopped in the middle of writing, discard that unterminated tail.
    if not contents.endswith(("\n", "\r")):
        last_newline = max(contents.rfind("\n"), contents.rfind("\r"))
        if last_newline < 0:
            return _empty_frame(empty_columns)
        contents = contents[: last_newline + 1]

    records = []
    header = None
    reader = csv.reader(io.StringIO(contents, newline=""), strict=True)
    try:
        header = next(reader, None)
        if not header:
            return _empty_frame(empty_columns)

        for row in reader:
            if not row or all(not value.strip() for value in row):
                continue
            # csv.writer emits one complete physical row per transaction. A
            # wrong-width record is malformed, so skip it rather than failing
            # the whole dashboard refresh.
            if len(row) != len(header):
                continue
            records.append(row)
    except csv.Error:
        # A malformed or half-written quoted record should not hide rows that
        # were parsed successfully before it.
        pass

    if not header:
        return _empty_frame(empty_columns)
    return pd.DataFrame(records, columns=header)


def load_stream_log(path: str = STREAM_LOG_PATH) -> pd.DataFrame:
    """Load complete stream records, or an empty frame if the log is absent."""
    return _read_csv_safely(path, STREAM_LOG_COLUMNS)


def load_alerts(path: str = ALERTS_PATH) -> pd.DataFrame:
    """Load complete alert records, tolerating absent or partially written CSVs."""
    return _read_csv_safely(path, ALERT_COLUMNS)


def _build_test_feature_lookup() -> dict[str, dict]:
    """Build raw RiskEngine input rows keyed like stream transaction IDs."""
    from src.data import get_splits

    _, _, X_test, _, _, _ = get_splits()
    test_rows = X_test.loc[:, ALL_FEATURES].copy()

    # RiskEngine expects raw Time and Amount; the split keeps originals for
    # reporting, separate from the scaled model input columns.
    for raw_name, original_name in zip(RAW_FEATURES, ORIGINAL_RAW_FEATURES):
        if original_name in X_test.columns:
            test_rows[raw_name] = X_test[original_name].to_numpy()

    return {
        f"tx_{txn_index}": row
        for txn_index, row in zip(
            X_test.index,
            test_rows.to_dict(orient="records"),
        )
    }


if st is not None:
    # This mapping is built once and reused between Streamlit reruns.
    _get_test_feature_lookup = st.cache_resource(_build_test_feature_lookup)
else:
    # A tiny fallback keeps non-dashboard unit tests independent of Streamlit.
    _get_test_feature_lookup = lru_cache(maxsize=1)(_build_test_feature_lookup)


def get_features_for_txn(txn_id: str) -> dict | None:
    """Return a copy of the raw model-feature row for a test transaction ID."""
    row = _get_test_feature_lookup().get(str(txn_id))
    return row.copy() if row is not None else None


def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute dashboard totals from stream records without rendering UI."""
    if df is None:
        df = pd.DataFrame()

    decisions = (
        df["decision"].fillna("").astype(str).str.strip().str.upper()
        if "decision" in df.columns
        else pd.Series("", index=df.index, dtype="object")
    )
    labels = (
        pd.to_numeric(df["true_label"], errors="coerce").fillna(0)
        if "true_label" in df.columns
        else pd.Series(0, index=df.index, dtype="int64")
    )
    latency = (
        pd.to_numeric(df["latency_ms"], errors="coerce").dropna().to_numpy()
        if "latency_ms" in df.columns
        else np.asarray([], dtype=float)
    )
    amounts = (
        pd.to_numeric(df["raw_amount"], errors="coerce").fillna(0)
        if "raw_amount" in df.columns
        else pd.Series(0.0, index=df.index, dtype=float)
    )

    return {
        "total_processed": int(len(df)),
        "approved": int((decisions == "APPROVE").sum()),
        "flagged": int((decisions == "FLAG").sum()),
        "blocked": int((decisions == "BLOCK").sum()),
        "fraud_caught": int(((labels == 1) & decisions.isin(["FLAG", "BLOCK"])).sum()),
        "average_latency": float(np.mean(latency)) if latency.size else 0.0,
        "p95_latency": float(np.percentile(latency, 95)) if latency.size else 0.0,
        "money_blocked": float(amounts[decisions == "BLOCK"].sum()),
    }
