"""
test_data.py — Unit tests for src/data.py

Checks:
    1. Train/val/test split sizes match the 60/20/20 config.
    2. Fraud ratio is preserved (similar) across all three sets.
    3. Scaled columns (Time, Amount) have near-zero mean on the train set.
    4. Scaler file is saved to disk.

Run:
    python -m pytest tests/test_data.py -v
"""

import os
import sys
import pytest
import numpy as np

# ── Make sure project root is on sys.path ────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    TEST_SIZE,
    VAL_SIZE,
    SCALER_PATH,
    RAW_FEATURES,
    ORIGINAL_RAW_FEATURES,
    ALL_FEATURES,
)
from src.data import load_data, get_splits


# ── Load once for all tests (expensive I/O) ─────────────────────
@pytest.fixture(scope="module")
def splits():
    """Run the full pipeline once and share across tests."""
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits()
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────
# TEST 1 — Split sizes (60 / 20 / 20)
# ─────────────────────────────────────────────────────────────────
def test_split_sizes(splits):
    """
    The test set should be ~20%, the validation set ~20%, and the
    train set ~60% of the total rows (config: TEST_SIZE=0.2,
    VAL_SIZE=0.25 of the remaining 80%).
    """
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    total = len(X_train) + len(X_val) + len(X_test)

    actual_test_ratio  = len(X_test) / total
    actual_val_ratio   = len(X_val)  / total
    actual_train_ratio = len(X_train) / total

    # Allow a tiny margin because integer rounding can't be exact
    assert abs(actual_test_ratio - 0.20) < 0.01, (
        f"Test ratio {actual_test_ratio:.4f} is too far from 0.20"
    )
    assert abs(actual_val_ratio - 0.20) < 0.01, (
        f"Val ratio {actual_val_ratio:.4f} is too far from 0.20"
    )
    assert abs(actual_train_ratio - 0.60) < 0.01, (
        f"Train ratio {actual_train_ratio:.4f} is too far from 0.60"
    )

    # Also check that X and y lengths match
    assert len(X_train) == len(y_train), "X_train / y_train length mismatch"
    assert len(X_val)   == len(y_val),   "X_val / y_val length mismatch"
    assert len(X_test)  == len(y_test),  "X_test / y_test length mismatch"


# ─────────────────────────────────────────────────────────────────
# TEST 2 — Fraud ratio preserved across all three sets
# ─────────────────────────────────────────────────────────────────
def test_fraud_ratio_preserved(splits):
    """
    Stratified splitting should keep the fraud percentage nearly
    identical in train, val, and test (within 0.05 percentage points).
    """
    _, _, _, y_train, y_val, y_test = splits

    train_fraud_pct = y_train.mean() * 100
    val_fraud_pct   = y_val.mean()   * 100
    test_fraud_pct  = y_test.mean()  * 100

    for name, pct in [("val", val_fraud_pct), ("test", test_fraud_pct)]:
        diff = abs(train_fraud_pct - pct)
        assert diff < 0.05, (
            f"Fraud ratio differs by {diff:.4f} pp — "
            f"train={train_fraud_pct:.4f}%, {name}={pct:.4f}%"
        )


# ─────────────────────────────────────────────────────────────────
# TEST 3 — Scaling correctness
# ─────────────────────────────────────────────────────────────────
def test_scaled_columns_near_zero_mean(splits):
    """
    After StandardScaler (fit on train), the train set's Time and
    Amount columns should have mean very close to 0.
    """
    X_train = splits[0]

    for col in RAW_FEATURES:
        col_mean = X_train[col].mean()
        assert abs(col_mean) < 1e-6, (
            f"Train column '{col}' has mean {col_mean:.6f}, expected ~0"
        )


# ─────────────────────────────────────────────────────────────────
# TEST 4 — Scaler persisted
# ─────────────────────────────────────────────────────────────────
def test_scaler_saved(splits):
    """The fitted scaler should be saved to models/scaler.joblib."""
    # splits fixture ensures get_splits() has run
    assert os.path.isfile(SCALER_PATH), (
        f"Scaler file not found at {SCALER_PATH}"
    )


def test_test_split_keeps_original_values_for_reporting(splits):
    """Original raw values are retained in report columns, outside model inputs."""
    X_test = splits[2]
    original_data = load_data()

    assert set(ALL_FEATURES).issubset(X_test.columns)
    assert set(ORIGINAL_RAW_FEATURES).issubset(X_test.columns)
    assert not set(ORIGINAL_RAW_FEATURES).intersection(ALL_FEATURES)

    for raw_column, original_column in zip(RAW_FEATURES, ORIGINAL_RAW_FEATURES):
        np.testing.assert_allclose(
            X_test[original_column].to_numpy(),
            original_data.loc[X_test.index, raw_column].to_numpy(),
        )
