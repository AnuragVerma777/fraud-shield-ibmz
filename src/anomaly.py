"""
anomaly.py — Unsupervised anomaly detection for Fraud Shield on IBM Z.

An Isolation Forest is trained on **legitimate transactions only** so it
learns what "normal" looks like.  At inference time, transactions that
deviate from normal receive a high anomaly score (0 = perfectly normal,
1 = most anomalous).

Functions:
    train_isolation_forest(X_train, y_train)  → dict (bundle)
    anomaly_score(bundle, X)                  → np.ndarray of [0, 1]

Usage:
    python -m src.anomaly
"""

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config import IFOREST_PATH, RANDOM_STATE, ensure_dirs
from src.data import get_splits


# ─────────────────────────────────────────────────
# 1. TRAIN
# ─────────────────────────────────────────────────
def train_isolation_forest(
    X_train,
    y_train,
    save: bool = True,
) -> dict:
    """
    Train an Isolation Forest on NON-FRAUD rows so it learns "normal".

    How Isolation Forest works (simplified):
        • It builds many random trees that recursively split features at
          random values.
        • Normal points sit deep inside dense regions and need many splits
          to be isolated → long path length.
        • Anomalies are sparse and get isolated quickly → short path length.
        • decision_function() returns a signed score where more negative
          values indicate stronger anomalies.

    Parameters
    ----------
    X_train : array-like
        Feature matrix (all rows — fraud + legit).
    y_train : array-like
        Target labels (0 = legit, 1 = fraud).  Used only to *filter* the
        training data; the model itself is fully unsupervised.
    save : bool
        If True, persist the model + normalization params to disk.

    Returns
    -------
    dict  — a "bundle" with keys:
        "model"     : the fitted IsolationForest
        "raw_min"   : float, minimum raw score seen on the training set
        "raw_max"   : float, maximum raw score seen on the training set
    """
    # ── Step 1: Compute the fraud ratio so the model knows roughly
    #    how much contamination to expect.
    fraud_ratio = float(np.mean(y_train == 1))
    print(f"[anomaly] Fraud ratio in training set: {fraud_ratio:.6f}")

    # ── Step 2: Keep only legitimate (Class=0) rows for training.
    #    This teaches the forest what *normal* transactions look like;
    #    it has never seen fraud during fitting.
    legit_mask = (np.asarray(y_train) == 0)
    X_legit = X_train[legit_mask] if hasattr(X_train, "iloc") else X_train[legit_mask]
    print(f"[anomaly] Training Isolation Forest on {len(X_legit):,} legit rows")

    # ── Step 3: Fit the Isolation Forest.
    #    • n_estimators=200   — more trees → more stable scores.
    #    • contamination      — set to the observed fraud ratio so the
    #                           internal threshold aligns with our data.
    #    • random_state       — reproducibility.
    model = IsolationForest(
        n_estimators=200,
        contamination=fraud_ratio,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_legit)

    # ── Step 4: Compute min / max of the raw decision_function on the
    #    *full* training set (legit + fraud) so we can min-max normalise
    #    later.  We negate the raw scores first because sklearn's
    #    decision_function returns  positive = normal, negative = anomaly
    #    and we want higher = more anomalous.
    raw_scores = -model.decision_function(X_train)   # negate
    raw_min = float(raw_scores.min())
    raw_max = float(raw_scores.max())
    print(f"[anomaly] Raw score range (negated): [{raw_min:.4f}, {raw_max:.4f}]")

    # ── Step 5: Bundle everything we need at inference time.
    bundle = {
        "model":   model,
        "raw_min": raw_min,
        "raw_max": raw_max,
    }

    # ── Step 6: Persist to disk.
    if save:
        ensure_dirs()
        joblib.dump(bundle, IFOREST_PATH)
        print(f"[anomaly] Saved IsolationForest bundle to {IFOREST_PATH}")

    return bundle


# ─────────────────────────────────────────────────
# 2. SCORE
# ─────────────────────────────────────────────────
def anomaly_score(bundle: dict, X) -> np.ndarray:
    """
    Return a 0-to-1 anomaly score for each row in X.

    Steps:
        1. Get raw decision_function output from the Isolation Forest.
        2. Negate it (so higher = more anomalous).
        3. Min-max normalise using the min/max saved from training data
           to keep the scale consistent across train / val / test.
        4. Clip to [0, 1] in case new data falls outside the training range.

    Parameters
    ----------
    bundle : dict
        The bundle returned by train_isolation_forest() (or loaded from
        models/iforest.joblib).
    X : array-like
        Feature matrix to score.

    Returns
    -------
    np.ndarray of shape (n_samples,), values in [0, 1].
    """
    model   = bundle["model"]
    raw_min = bundle["raw_min"]
    raw_max = bundle["raw_max"]

    # Negate so that more-anomalous → higher value
    raw = -model.decision_function(X)

    # Min-max normalise using *training* statistics.
    # Guard against the edge case where raw_min == raw_max (would cause
    # division by zero); in practice this never happens with real data.
    denom = raw_max - raw_min
    if denom == 0:
        return np.zeros(len(X))

    scores = (raw - raw_min) / denom

    # Clip to [0, 1] — unseen data may exceed training range
    scores = np.clip(scores, 0.0, 1.0)

    return scores


# ─────────────────────────────────────────────────
# 3. LOAD HELPER
# ─────────────────────────────────────────────────
def load_bundle(path: str = IFOREST_PATH) -> dict:
    """Load a previously saved IsolationForest bundle from disk."""
    try:
        bundle = joblib.load(path)
        print(f"[anomaly] Loaded IsolationForest bundle from {path}")
        return bundle
    except FileNotFoundError:
        raise FileNotFoundError(
            f"IsolationForest bundle not found at {path}. "
            "Run 'python -m src.anomaly' first."
        )


# ─────────────────────────────────────────────────
# CLI — train + evaluate on the validation set
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Load the three-way splits ────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits()

    # ── Train the Isolation Forest ───────────────
    bundle = train_isolation_forest(X_train, y_train, save=True)

    # ── Score the validation set ─────────────────
    val_scores = anomaly_score(bundle, X_val)

    # ── Separate scores by class for reporting ───
    val_labels  = np.asarray(y_val)
    legit_mask  = (val_labels == 0)
    fraud_mask  = (val_labels == 1)

    avg_legit = float(val_scores[legit_mask].mean())
    avg_fraud = float(val_scores[fraud_mask].mean())

    print("\n--- Anomaly Scores on Validation Set ---")
    print(f"  Avg score (legit) : {avg_legit:.4f}")
    print(f"  Avg score (fraud) : {avg_fraud:.4f}")
    print(f"  Separation        : {avg_fraud - avg_legit:.4f}")

    # ── Standalone discrimination metrics ────────
    #    ROC-AUC and PR-AUC treat the anomaly score as a "fraud probability"
    #    to measure how well the unsupervised model separates fraud from legit
    #    *on its own*, before we combine it with XGBoost.
    roc = roc_auc_score(val_labels, val_scores)
    pr  = average_precision_score(val_labels, val_scores)

    print(f"\n  ROC-AUC (standalone) : {roc:.4f}")
    print(f"  PR-AUC  (standalone) : {pr:.4f}")
    print("\nDone.")
