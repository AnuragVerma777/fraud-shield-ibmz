"""Create simple, rule-based alerts for flagged or blocked transactions.

SHAP explanations are planned for a later project stage. For now, alert
reasons describe which visible rule(s) caused the transaction to stand out.
"""

from datetime import datetime, timezone
import uuid

from src.config import (
    ALERT_HIGH_ANOMALY_THRESHOLD,
    ALERT_HIGH_XGB_PROB_THRESHOLD,
    ALERT_LARGE_AMOUNT_THRESHOLD,
)


ALERT_FIELDNAMES = [
    "alert_id",
    "txn_id",
    "severity",
    "risk_score",
    "raw_amount",
    "reason",
    "timestamp",
]


def generate_alert(result: dict) -> dict | None:
    """Build an alert for BLOCK/FLAG results; return ``None`` for APPROVE.

    ``result`` should include ``txn_id``, ``decision``, ``risk_score``,
    ``raw_amount``, ``xgb_prob`` and ``anomaly_score``. The stream consumer
    adds the raw amount and transaction ID before calling this function.
    """
    decision = str(result["decision"]).upper()
    if decision == "APPROVE":
        return None
    if decision not in {"FLAG", "BLOCK"}:
        raise ValueError(f"Unsupported transaction decision: {decision!r}")

    reasons = []
    raw_amount = float(result["raw_amount"])
    anomaly = float(result["anomaly_score"])
    xgb_prob = float(result["xgb_prob"])

    if raw_amount >= ALERT_LARGE_AMOUNT_THRESHOLD:
        reasons.append("unusually large amount")
    if anomaly >= ALERT_HIGH_ANOMALY_THRESHOLD:
        reasons.append("high anomaly score")
    if xgb_prob >= ALERT_HIGH_XGB_PROB_THRESHOLD:
        reasons.append("model fraud probability above 0.9")
    if not reasons:
        reasons.append(f"risk score crossed the {decision} decision threshold")

    return {
        "alert_id": f"alert_{uuid.uuid4().hex}",
        "txn_id": str(result["txn_id"]),
        "severity": "HIGH" if decision == "BLOCK" else "MEDIUM",
        "risk_score": float(result["risk_score"]),
        "raw_amount": raw_amount,
        "reason": "; ".join(reasons),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
