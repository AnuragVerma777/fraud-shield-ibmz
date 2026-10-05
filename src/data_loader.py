"""
data_loader.py — Load, validate, and summarise the credit card fraud dataset.

Functions:
    load_raw_data()      → pd.DataFrame
    validate_data(df)    → dict of checks
    get_summary(df)      → dict of summary stats
    print_summary(df)    → pretty-print to console
"""

import pandas as pd
import numpy as np
from src.config import DATA_PATH, TARGET_COL, PCA_FEATURES, RAW_FEATURES


# ─────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────
def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Read the raw CSV into a DataFrame.
    Raises FileNotFoundError with a helpful message if file is missing.
    """
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded {len(df):,} rows × {df.shape[1]} cols from {path}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(
            f"❌ Dataset not found at:\n   {path}\n"
            "   Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "   and place 'creditcard.csv' inside the data/ folder."
        )


# ─────────────────────────────────────────────────
# VALIDATE
# ─────────────────────────────────────────────────
def validate_data(df: pd.DataFrame) -> dict:
    """
    Run basic quality checks on the dataset.
    Returns a dict with check results.
    """
    checks = {}

    # 1. Expected columns present?
    expected_cols = PCA_FEATURES + RAW_FEATURES + [TARGET_COL]
    missing_cols  = [c for c in expected_cols if c not in df.columns]
    checks["missing_columns"] = missing_cols
    checks["columns_ok"] = len(missing_cols) == 0

    # 2. Any nulls?
    null_counts = df.isnull().sum()
    checks["total_nulls"] = int(null_counts.sum())
    checks["nulls_ok"] = checks["total_nulls"] == 0

    # 3. Target values valid? (should be only 0 and 1)
    unique_targets = sorted(df[TARGET_COL].unique().tolist())
    checks["target_values"] = unique_targets
    checks["target_ok"] = unique_targets == [0, 1]

    # 4. Row count sanity
    checks["n_rows"] = len(df)
    checks["n_cols"] = df.shape[1]

    return checks


# ─────────────────────────────────────────────────
# SUMMARISE
# ─────────────────────────────────────────────────
def get_summary(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics relevant to fraud detection.
    """
    n_total   = len(df)
    n_fraud   = int(df[TARGET_COL].sum())
    n_legit   = n_total - n_fraud
    fraud_pct = n_fraud / n_total * 100

    return {
        "n_total":   n_total,
        "n_fraud":   n_fraud,
        "n_legit":   n_legit,
        "fraud_pct": round(fraud_pct, 4),
        "amount_stats": {
            "mean":   round(df["Amount"].mean(), 2),
            "median": round(df["Amount"].median(), 2),
            "max":    round(df["Amount"].max(), 2),
            "min":    round(df["Amount"].min(), 2),
        },
        "time_range_hours": round(df["Time"].max() / 3600, 2),
    }


def print_summary(df: pd.DataFrame) -> None:
    """
    Pretty-print dataset summary to console.
    """
    s = get_summary(df)
    v = validate_data(df)

    print("\n" + "=" * 55)
    print("   FRAUD SHIELD — Dataset Summary")
    print("=" * 55)
    print(f"  Rows            : {s['n_total']:>10,}")
    print(f"  Columns         : {v['n_cols']:>10}")
    print(f"  Legit (Class=0) : {s['n_legit']:>10,}")
    print(f"  Fraud (Class=1) : {s['n_fraud']:>10,}")
    print(f"  Fraud %         : {s['fraud_pct']:>10.4f}%")
    print(f"  Time span       : {s['time_range_hours']:>10.1f} hrs")
    print("-" * 55)
    a = s["amount_stats"]
    print(f"  Amount  mean    : {a['mean']:>10.2f}")
    print(f"  Amount  median  : {a['median']:>10.2f}")
    print(f"  Amount  min     : {a['min']:>10.2f}")
    print(f"  Amount  max     : {a['max']:>10.2f}")
    print("-" * 55)
    print(f"  Nulls           : {v['total_nulls']}")
    print(f"  Columns OK      : {v['columns_ok']}")
    print(f"  Target OK       : {v['target_ok']}")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_raw_data()
    print_summary(df)
