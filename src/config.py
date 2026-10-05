"""
config.py — Central configuration for Fraud Shield on IBM Z.

Every path, constant, and hyperparameter the project needs
is defined here. NO other file should hard-code paths.
"""

import os

# ──────────────────────────────────────────────
# 1. PROJECT ROOT  (auto-detected)
# ──────────────────────────────────────────────
# This file lives at  <project_root>/src/config.py
# so the project root is one directory up.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────
# 2. DIRECTORY PATHS
# ──────────────────────────────────────────────
DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR  = os.path.join(PROJECT_ROOT, "models")
DOCS_DIR   = os.path.join(PROJECT_ROOT, "docs")
APP_DIR    = os.path.join(PROJECT_ROOT, "app")
TESTS_DIR  = os.path.join(PROJECT_ROOT, "tests")

# ──────────────────────────────────────────────
# 3. FILE PATHS
# ──────────────────────────────────────────────
DATA_PATH           = os.path.join(DATA_DIR, "creditcard.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "creditcard_processed.csv")
MODEL_PATH          = os.path.join(MODEL_DIR, "xgb_fraud_model.joblib")
XGB_MODEL_PATH      = os.path.join(MODEL_DIR, "xgb_model.joblib")
ONNX_MODEL_PATH     = os.path.join(MODEL_DIR, "xgb_fraud_model.onnx")
SCALER_PATH         = os.path.join(MODEL_DIR, "scaler.joblib")
THRESHOLD_PATH      = os.path.join(MODEL_DIR, "thresholds.json")
IFOREST_PATH        = os.path.join(MODEL_DIR, "iforest.joblib")
RISK_CONFIG_PATH    = os.path.join(MODEL_DIR, "risk_config.json")
STREAM_LOG_PATH     = os.path.join(DATA_DIR, "stream_log.csv")
ALERTS_PATH         = os.path.join(DATA_DIR, "alerts.csv")

# Training outputs (saved to docs/)
PR_CURVE_PATH       = os.path.join(DOCS_DIR, "pr_curve.png")
CM_PLOT_PATH        = os.path.join(DOCS_DIR, "confusion_matrix.png")
METRICS_PATH        = os.path.join(DOCS_DIR, "metrics_day1.json")
METRICS_DAY2_PATH   = os.path.join(DOCS_DIR, "metrics_day2.json")
THRESHOLD_PLOT_PATH = os.path.join(DOCS_DIR, "threshold_tradeoff.png")
RISK_THRESHOLD_PLOT_PATH = os.path.join(DOCS_DIR, "risk_threshold_tradeoff.png")
DAY2_PERFORMANCE_PLOT_PATH = os.path.join(DOCS_DIR, "day2_model_comparison.png")
DAY2_DECISION_PLOT_PATH = os.path.join(DOCS_DIR, "day2_decision_matrix.png")

# ──────────────────────────────────────────────
# 4. REPRODUCIBILITY
# ──────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2
# After removing 20% for test, 25% of the remaining 80% → 20% overall
VAL_SIZE     = 0.25   # results in a 60 / 20 / 20 split

# ──────────────────────────────────────────────
# 5. DATASET CONSTANTS
# ──────────────────────────────────────────────
TARGET_COL   = "Class"                           # 0 = legit, 1 = fraud
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]   # V1 … V28
RAW_FEATURES = ["Time", "Amount"]                 # non-PCA columns
ORIGINAL_RAW_FEATURES = [f"{column}_original" for column in RAW_FEATURES]
ALL_FEATURES = PCA_FEATURES + RAW_FEATURES        # used during training

# Rule-based alert reasons (initial values; tune as the product evolves).
ALERT_LARGE_AMOUNT_THRESHOLD = 1000.0
ALERT_HIGH_ANOMALY_THRESHOLD = 0.8
ALERT_HIGH_XGB_PROB_THRESHOLD = 0.9

# ──────────────────────────────────────────────
# 6. MODEL HYPERPARAMETERS  (sensible defaults)
# ──────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":     200,
    "max_depth":        6,
    "learning_rate":    0.1,
    "scale_pos_weight": 1,          # adjusted after SMOTE
    "eval_metric":      "aucpr",
    "use_label_encoder": False,
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
}

# ──────────────────────────────────────────────
# 7. RISK-SCORE DECISION THRESHOLDS  (0–100 scale)
# ──────────────────────────────────────────────
# score < FLAG_THRESHOLD   →  APPROVE
# FLAG  ≤ score < BLOCK    →  FLAG for manual review
# score ≥ BLOCK_THRESHOLD  →  BLOCK immediately
# These are *placeholders*; tune on the validation set in Task 4.
FLAG_THRESHOLD  = 30
BLOCK_THRESHOLD = 70

# ──────────────────────────────────────────────
# 8. HELPER — create output dirs if missing
# ──────────────────────────────────────────────
def ensure_dirs():
    """Create every output directory that doesn't exist yet."""
    for d in [DATA_DIR, MODEL_DIR, DOCS_DIR]:
        os.makedirs(d, exist_ok=True)
