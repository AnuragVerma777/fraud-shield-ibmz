"""
risk.py — Combined risk scoring and decision engine for Fraud Shield.

Fuses the XGBoost fraud probability (supervised) with the Isolation Forest
anomaly score (unsupervised) into a single 0–100 risk score, then maps it
to a three-way decision: APPROVE / FLAG / BLOCK.

Functions:
    compute_risk(xgb_prob, anomaly_norm, w_xgb, w_anom)  → float  (0–100)
    decide(score, flag_threshold, block_threshold)        → str

Class:
    RiskEngine  — loads all saved artifacts and scores a raw transaction
                  end-to-end, returning a result dict with latency.

Usage:
    python -m src.risk          # demo on 5 random test transactions
"""

import json
import os
import time
import numpy as np
import pandas as pd
import joblib

from src.config import (
    XGB_MODEL_PATH,
    SCALER_PATH,
    IFOREST_PATH,
    RISK_CONFIG_PATH,
    FLAG_THRESHOLD,
    BLOCK_THRESHOLD,
    ALL_FEATURES,
    RAW_FEATURES,
    PCA_FEATURES,
)
import warnings
from src.anomaly import anomaly_score as _anomaly_score


# ─────────────────────────────────────────────────
# 1. COMPUTE RISK SCORE
# ─────────────────────────────────────────────────
def compute_risk(
    xgb_prob: float,
    anomaly_norm: float,
    w_xgb: float = 0.7,
    w_anom: float = 0.3,
) -> float:
    """
    Weighted combination of two signals, scaled to 0–100.

    Parameters
    ----------
    xgb_prob : float
        XGBoost fraud probability, in [0, 1].
    anomaly_norm : float
        Normalised anomaly score from the Isolation Forest, in [0, 1].
    w_xgb : float
        Weight for the supervised signal (default 0.7).
    w_anom : float
        Weight for the unsupervised signal (default 0.3).

    Returns
    -------
    float in [0, 100].

    Notes
    -----
    The weighted average is itself in [0, 1] so we just multiply by 100.
    70/30 weighting gives the supervised model — which has higher precision —
    the dominant voice, while still letting the anomaly detector boost the
    score for transactions that *look* unusual even if XGBoost is unsure.
    """
    blended = w_xgb * xgb_prob + w_anom * anomaly_norm
    # Clip defensively, then scale to 0–100
    return float(np.clip(blended, 0.0, 1.0) * 100)


# ─────────────────────────────────────────────────
# 2. DECISION FUNCTION
# ─────────────────────────────────────────────────
def decide(
    score: float,
    flag_threshold: float = FLAG_THRESHOLD,
    block_threshold: float = BLOCK_THRESHOLD,
) -> str:
    """
    Map a 0–100 risk score to a three-way action.

    Rules (using config defaults of 30 / 70):
        score < 30   →  "APPROVE"   — low risk, let it through
        30 ≤ score < 70  →  "FLAG"  — medium risk, queue for review
        score ≥ 70   →  "BLOCK"     — high risk, reject immediately

    Parameters
    ----------
    score : float
        Combined risk score in [0, 100].
    flag_threshold : float
        Scores at or above this are flagged (default from config).
    block_threshold : float
        Scores at or above this are blocked (default from config).

    Returns
    -------
    One of "APPROVE", "FLAG", or "BLOCK".
    """
    if score >= block_threshold:
        return "BLOCK"
    if score >= flag_threshold:
        return "FLAG"
    return "APPROVE"


# ─────────────────────────────────────────────────
# 3. RISK ENGINE (end-to-end scorer)
# ─────────────────────────────────────────────────
class RiskEngine:
    """
    Loads XGBoost, the scaler, and the Isolation Forest from disk,
    then scores individual transactions end-to-end.

    Usage
    -----
    >>> engine = RiskEngine()
    >>> result = engine.score_transaction(row)
    >>> result["decision"]   # "APPROVE", "FLAG", or "BLOCK"
    """

    def __init__(
        self,
        xgb_path: str = XGB_MODEL_PATH,
        scaler_path: str = SCALER_PATH,
        iforest_path: str = IFOREST_PATH,
        risk_config_path: str = RISK_CONFIG_PATH,
        flag_threshold: float | None = None,
        block_threshold: float | None = None,
        w_xgb: float | None = None,
        w_anom: float | None = None,
    ):
        # ── Load all three artifacts from models/ ──
        self.xgb_model = joblib.load(xgb_path)
        self.scaler    = joblib.load(scaler_path)
        self.if_bundle = joblib.load(iforest_path)

        # Load tuned values when available; explicit constructor arguments
        # take precedence, followed by project defaults before the first tune.
        saved_config = {}
        if os.path.exists(risk_config_path):
            with open(risk_config_path, "r", encoding="utf-8") as config_file:
                saved_config = json.load(config_file)

        self.flag_threshold = float(
            flag_threshold
            if flag_threshold is not None
            else saved_config.get("flag_threshold", FLAG_THRESHOLD)
        )
        self.block_threshold = float(
            block_threshold
            if block_threshold is not None
            else saved_config.get("block_threshold", BLOCK_THRESHOLD)
        )
        self.w_xgb = float(
            w_xgb if w_xgb is not None else saved_config.get("w_xgb", 0.7)
        )
        self.w_anom = float(
            w_anom if w_anom is not None else saved_config.get("w_anom", 0.3)
        )

        # ── Fast-path pre-allocated NumPy buffer and cached parameters ──
        self._feature_buffer = np.zeros((1, len(ALL_FEATURES)), dtype=np.float64)
        self._pca_features = tuple(PCA_FEATURES)
        self._raw_features = tuple(RAW_FEATURES)
        self._all_features = tuple(ALL_FEATURES)

        # Cache StandardScaler parameters for direct mathematical scaling:
        # z = (x - mean_) / scale_
        self._has_scaler_params = (
            hasattr(self.scaler, "mean_")
            and hasattr(self.scaler, "scale_")
            and self.scaler.mean_ is not None
            and self.scaler.scale_ is not None
        )
        if self._has_scaler_params:
            self._time_mean = float(self.scaler.mean_[0])
            self._time_scale = float(self.scaler.scale_[0])
            self._amount_mean = float(self.scaler.mean_[1])
            self._amount_scale = float(self.scaler.scale_[1])

        # Avoid spawning worker pools for single-row evaluations
        if hasattr(self.xgb_model, "set_params"):
            try:
                self.xgb_model.set_params(n_jobs=1)
            except Exception:
                pass
        if hasattr(self.if_bundle, "get") and hasattr(self.if_bundle.get("model"), "set_params"):
            try:
                self.if_bundle["model"].set_params(n_jobs=1)
            except Exception:
                pass

        # Suppress scikit-learn feature-name UserWarnings when scoring raw NumPy arrays
        warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")

        print(
            f"[RiskEngine] Loaded XGBoost, scaler, IsolationForest  |  "
            f"FLAG >= {self.flag_threshold}  |  BLOCK >= {self.block_threshold}"
        )

    # ──────────────────────────────────────────
    def score_transaction(self, row: pd.Series | dict | np.ndarray) -> dict:
        """
        Score a single raw transaction using a pre-allocated NumPy array.

        Bypasses per-row pandas Series and DataFrame allocations to minimize
        single-transaction latency for high-throughput streaming.

        Parameters
        ----------
        row : pd.Series, dict, or np.ndarray
            Must contain the 30 feature columns (V1–V28, Time, Amount).
            Values should be *raw* (unscaled) — the engine applies the
            saved scaler internally.

        Returns
        -------
        dict with keys:
            risk_score    : float   (0–100)
            decision      : str     ("APPROVE" / "FLAG" / "BLOCK")
            xgb_prob      : float   (0–1, XGBoost fraud probability)
            anomaly_score : float   (0–1, normalised Isolation Forest score)
            latency_ms    : float   (wall-clock scoring time in milliseconds)
        """
        t0 = time.perf_counter()

        buf = self._feature_buffer

        # ── Step 1 & 2: Populate pre-allocated buffer & scale Time/Amount ──
        if isinstance(row, dict):
            for i, feat in enumerate(self._pca_features):
                buf[0, i] = row[feat]
            raw_time = float(row["Time"])
            raw_amount = float(row["Amount"])
            if self._has_scaler_params:
                buf[0, 28] = (raw_time - self._time_mean) / self._time_scale
                buf[0, 29] = (raw_amount - self._amount_mean) / self._amount_scale
            else:
                scaled = self.scaler.transform([[raw_time, raw_amount]])
                buf[0, 28] = scaled[0, 0]
                buf[0, 29] = scaled[0, 1]
        elif isinstance(row, pd.Series):
            for i, feat in enumerate(self._pca_features):
                buf[0, i] = row[feat]
            raw_time = float(row["Time"])
            raw_amount = float(row["Amount"])
            if self._has_scaler_params:
                buf[0, 28] = (raw_time - self._time_mean) / self._time_scale
                buf[0, 29] = (raw_amount - self._amount_mean) / self._amount_scale
            else:
                scaled = self.scaler.transform([[raw_time, raw_amount]])
                buf[0, 28] = scaled[0, 0]
                buf[0, 29] = scaled[0, 1]
        elif isinstance(row, np.ndarray):
            buf[0, :] = row
        else:
            raise TypeError(f"Unsupported row type: {type(row)}")

        # ── Step 3: XGBoost fraud probability ──
        xgb_prob = float(self.xgb_model.predict_proba(buf)[:, 1][0])

        # ── Step 4: Isolation Forest anomaly score ──
        anom = float(_anomaly_score(self.if_bundle, buf)[0])

        # ── Step 5: Combined risk score (0–100) ──
        risk = compute_risk(xgb_prob, anom, self.w_xgb, self.w_anom)

        # ── Step 6: Three-way decision ──
        decision = decide(risk, self.flag_threshold, self.block_threshold)

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000

        return {
            "risk_score":    round(risk, 2),
            "decision":      decision,
            "xgb_prob":      round(xgb_prob, 6),
            "anomaly_score": round(anom, 6),
            "latency_ms":    round(latency_ms, 3),
        }

    # ──────────────────────────────────────────
    def _score_transaction_dataframe(self, row: pd.Series | dict) -> dict:
        """
        Legacy DataFrame-based scoring path, kept for regression testing and benchmarking.
        """
        t0 = time.perf_counter()

        # ── Step 1: Build a single-row DataFrame in feature order ──
        if isinstance(row, dict):
            row = pd.Series(row)
        features = row[ALL_FEATURES].to_frame().T

        # ── Step 2: Scale Time & Amount using the saved scaler ──
        features[RAW_FEATURES] = self.scaler.transform(features[RAW_FEATURES])

        # ── Step 3: XGBoost fraud probability ──
        xgb_prob = float(self.xgb_model.predict_proba(features)[:, 1][0])

        # ── Step 4: Isolation Forest anomaly score ──
        anom = float(_anomaly_score(self.if_bundle, features)[0])

        # ── Step 5: Combined risk score (0–100) ──
        risk = compute_risk(xgb_prob, anom, self.w_xgb, self.w_anom)

        # ── Step 6: Three-way decision ──
        decision = decide(risk, self.flag_threshold, self.block_threshold)

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000

        return {
            "risk_score":    round(risk, 2),
            "decision":      decision,
            "xgb_prob":      round(xgb_prob, 6),
            "anomaly_score": round(anom, 6),
            "latency_ms":    round(latency_ms, 3),
        }


# ─────────────────────────────────────────────────
# CLI — demo on a few test transactions
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data import get_splits

    # Load the data (we only need the test set here)
    _, _, X_test, _, _, y_test = get_splits()

    # Initialise the engine (loads all three models from disk)
    engine = RiskEngine()

    # Pick 5 random transactions: a mix of fraud and legit
    np.random.seed(42)
    # Grab up to 3 fraud + 2 legit so the demo is interesting
    fraud_idx = y_test[y_test == 1].index.tolist()
    legit_idx = y_test[y_test == 0].index.tolist()
    sample_idx = (
        list(np.random.choice(fraud_idx, size=min(3, len(fraud_idx)), replace=False))
        + list(np.random.choice(legit_idx, size=2, replace=False))
    )

    print("\n--- Risk Engine Demo (5 test transactions) ---")
    print(f"{'#':<3} {'Actual':<8} {'Decision':<9} {'Risk':>6} "
          f"{'XGB':>8} {'Anomaly':>9} {'Latency':>10}")
    print("-" * 62)

    for i, idx in enumerate(sample_idx, 1):
        # Pass the *raw* (already-scaled in our pipeline) features.
        # In a real deployment the scaler would run on truly raw input;
        # here X_test is already scaled, so we skip re-scaling by
        # calling the individual model components directly instead.
        row = X_test.loc[idx, ALL_FEATURES]
        actual = "FRAUD" if y_test.loc[idx] == 1 else "LEGIT"

        # For the demo we call the internal pieces directly on
        # already-scaled data to avoid double-scaling.
        xgb_prob = float(engine.xgb_model.predict_proba(row.to_frame().T)[:, 1][0])
        anom     = float(_anomaly_score(engine.if_bundle, row.to_frame().T)[0])
        risk     = compute_risk(xgb_prob, anom, engine.w_xgb, engine.w_anom)
        decision = decide(risk, engine.flag_threshold, engine.block_threshold)

        print(f"{i:<3} {actual:<8} {decision:<9} {risk:>6.2f} "
              f"{xgb_prob:>8.4f} {anom:>9.4f} {'-':>10}")

    print("\nDone.")
