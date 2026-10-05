"""
data.py — Load, preprocess, and split the credit card fraud dataset.

Functions:
    load_data()    → pd.DataFrame       Load the raw CSV.
    preprocess()   → tuple[pd.DataFrame, ...]  Scale Time & Amount (fit on train only).
    get_splits()   → (X_train, X_val, X_test, y_train, y_val, y_test)
                     Stratified 60/20/20 split.

Usage:
    python -m src.data          # quick smoke test from the command line
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    SCALER_PATH,
    MODEL_DIR,
    TARGET_COL,
    ALL_FEATURES,
    RAW_FEATURES,
    ORIGINAL_RAW_FEATURES,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    ensure_dirs,
)


# ─────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Read the raw CSV and return a DataFrame.

    Parameters
    ----------
    path : str
        Full path to creditcard.csv (defaults to config.DATA_PATH).

    Returns
    -------
    pd.DataFrame with all 31 original columns.
    """
    try:
        df = pd.read_csv(path)
        print(f"[load_data] Loaded {len(df):,} rows from {path}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place creditcard.csv inside data/"
        )


# ─────────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────────
def preprocess(
    df_train: pd.DataFrame,
    *other_dfs: pd.DataFrame,
    save_scaler: bool = True,
) -> tuple[pd.DataFrame, ...]:
    """
    Scale the 'Time' and 'Amount' columns with StandardScaler.

    IMPORTANT — the scaler is **fit on training data only**, then
    applied (transform) to every set passed in.  This prevents
    data leakage from the validation / test sets into the scaler.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training split (will be used to fit the scaler).
    *other_dfs : pd.DataFrame
        Any additional splits (validation, test, …) — transform only.
    save_scaler : bool
        If True, persist the fitted scaler to models/scaler.joblib.

    Returns
    -------
    Tuple of DataFrames in the same order they were passed in, with
    Time & Amount replaced by their scaled versions.
    """
    # Work on copies so we never mutate the caller's data
    train = df_train.copy()

    # Fit scaler on training set only
    scaler = StandardScaler()
    scaler.fit(train[RAW_FEATURES])

    # Transform training set
    train[RAW_FEATURES] = scaler.transform(train[RAW_FEATURES])

    # Transform every other set
    transformed = [train]
    for df in other_dfs:
        copy = df.copy()
        copy[RAW_FEATURES] = scaler.transform(copy[RAW_FEATURES])
        transformed.append(copy)

    # Persist the scaler so it can be reloaded at inference time
    if save_scaler:
        ensure_dirs()
        joblib.dump(scaler, SCALER_PATH)
        print(f"[preprocess] Scaler saved to {SCALER_PATH}")

    print(f"[preprocess] Scaled columns: {RAW_FEATURES}")
    return tuple(transformed)


# ─────────────────────────────────────────────────
# 3. GET SPLITS
# ─────────────────────────────────────────────────
def get_splits(
    df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    End-to-end pipeline: load → split → preprocess → return arrays.

    Steps:
        1. Load data (if df not provided).
        2. Stratified split into train (60%), validation (20%), test (20%).
           - First split: 80% train+val vs 20% test.
           - Second split: 75% train vs 25% val (of the 80%).
        3. Scale Time & Amount (fit on train only).

    Returns
    -------
    X_train, X_val : pd.DataFrame — scaled model features (30 columns).
    X_test : pd.DataFrame — 30 scaled model features plus reporting-only
        Time_original and Amount_original columns.
    y_train, y_val, y_test : pd.Series      — target labels (0 or 1)
    """
    # Step 1 — load
    if df is None:
        df = load_data()

    # Step 2 — stratified splits
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]

    # First split: train+val (80%) vs test (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Keep raw test values for human-readable reports. These are appended only
    # after preprocessing and are excluded from model inputs via ALL_FEATURES.
    original_test_values = X_test[RAW_FEATURES].copy()
    original_test_values.columns = ORIGINAL_RAW_FEATURES

    # Second split: train (60% total) vs val (20% total)
    # VAL_SIZE = 0.25 means 25% of the 80% → 20% of total
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )

    print(
        f"[get_splits] Train: {len(X_train):,}  |  "
        f"Val: {len(X_val):,}  |  Test: {len(X_test):,}"
    )

    # Step 3 — preprocess (scale Time & Amount, fit on train only)
    X_train, X_val, X_test = preprocess(
        X_train, X_val, X_test, save_scaler=True,
    )
    X_test = X_test.join(original_test_values)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────
# CLI — quick smoke test
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits()

    print("\n--- Quick Checks ---")
    print(f"X_train shape : {X_train.shape}")
    print(f"X_val   shape : {X_val.shape}")
    print(f"X_test  shape : {X_test.shape}")

    train_fraud = y_train.mean() * 100
    val_fraud   = y_val.mean()   * 100
    test_fraud  = y_test.mean()  * 100
    print(f"Train fraud % : {train_fraud:.4f}%")
    print(f"Val   fraud % : {val_fraud:.4f}%")
    print(f"Test  fraud % : {test_fraud:.4f}%")
    print(f"Max diff      : {max(abs(train_fraud - val_fraud), abs(train_fraud - test_fraud)):.4f} pp")

    print(f"\nTime  mean (train): {X_train['Time'].mean():.4f}  (should be ~0)")
    print(f"Amount mean (train): {X_train['Amount'].mean():.4f}  (should be ~0)")
    print("\nDone.")
