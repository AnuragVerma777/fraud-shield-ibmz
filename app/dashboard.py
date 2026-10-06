"""Live Streamlit dashboard for Fraud Shield's transaction simulation.

Run from the repository root with ``streamlit run app/dashboard.py``.
"""

import json
import math
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_access import (
    compute_kpis,
    get_features_for_txn,
    load_alerts,
    load_stream_log,
)
from app.stream_control import is_running, start_stream, stop_stream
from src.config import (
    ALERTS_PATH,
    BATCH_VS_REALTIME_PATH,
    BLOCK_THRESHOLD,
    BENCHMARK_PATH,
    METRICS_DAY2_PATH,
    RISK_CONFIG_PATH,
    STREAM_LOG_PATH,
)


st.set_page_config(
    page_title="Fraud Shield on IBM Z",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.25rem; padding-bottom: 1.5rem; }
      .status-badge {
        display: inline-block; padding: 0.35rem 0.8rem; border-radius: 999px;
        font-weight: 700; font-size: 0.82rem; letter-spacing: 0.03em;
      }
      .status-running { background: #123D25; color: #2ECC71; border: 1px solid #2ECC71; }
      .status-stopped { background: #292E38; color: #B9C1CF; border: 1px solid #454D5A; }
      [data-testid="stVerticalBlockBorderWrapper"] {
        background: #171B24; border: 1px solid #2A303B; border-radius: 14px;
      }
      div[data-testid="stMetric"] {
        background: transparent; border: 0; padding: 0.25rem 0.2rem;
      }
      [data-testid="stMetricLabel"] { color: #AAB3C2; }
      @media (max-width: 768px) {
        .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
        [data-testid="stHorizontalBlock"] {
          flex-direction: column !important; gap: 0.65rem !important;
        }
        [data-testid="stColumn"] {
          width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important;
        }
        h1 { font-size: 1.7rem !important; }
        [data-testid="stMetricValue"] { font-size: 1.35rem !important; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_risk_thresholds() -> tuple[float, float]:
    """Read tuned decision cutoffs, falling back to project placeholders."""
    try:
        with open(RISK_CONFIG_PATH, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        flag = float(config.get("flag_threshold", 30))
        block = float(config.get("block_threshold", BLOCK_THRESHOLD))
        return flag, block
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 30.0, float(BLOCK_THRESHOLD)


def _reset_logs() -> None:
    """Remove the stream and alert CSVs so the next view starts empty."""
    for path in (STREAM_LOG_PATH, ALERTS_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _number(value, digits: int = 2) -> str:
    """Format numeric CSV values safely for dashboard text."""
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _metric_card(column, label: str, value: str) -> None:
    """Place one Streamlit metric inside the dashboard's rounded card style."""
    with column:
        with st.container(border=True):
            st.metric(label, value)


def _risk_score_figure(log_df: pd.DataFrame, flag_threshold: float, block_threshold: float):
    """Build the recent risk-score chart and tuned action threshold lines."""
    recent = log_df.tail(200).copy()
    if not recent.empty:
        recent["risk_score"] = pd.to_numeric(recent["risk_score"], errors="coerce")
        recent["processed_at"] = pd.to_datetime(recent["processed_at"], errors="coerce")
        recent = recent.dropna(subset=["risk_score"])

    fig = go.Figure()
    if not recent.empty:
        x_values = recent["processed_at"]
        if x_values.isna().all():
            x_values = list(range(1, len(recent) + 1))
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=recent["risk_score"],
                mode="lines+markers",
                name="Risk score",
                line={"color": "#2ECC71", "width": 2},
                marker={"size": 5},
                customdata=recent[["txn_id", "decision"]].to_numpy(),
                hovertemplate=(
                    "Risk: %{y:.2f}<br>Transaction: %{customdata[0]}"
                    "<br>Decision: %{customdata[1]}<extra></extra>"
                ),
            )
        )
    else:
        fig.add_annotation(
            text="Waiting for transactions…",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": "#AAB3C2", "size": 15},
        )

    fig.add_hline(
        y=flag_threshold,
        line_dash="dash",
        line_color="#F39C12",
        annotation_text=f"FLAG {flag_threshold:.2f}",
        annotation_position="top left",
    )
    fig.add_hline(
        y=block_threshold,
        line_dash="dash",
        line_color="#E74C3C",
        annotation_text=f"BLOCK {block_threshold:.2f}",
        annotation_position="top left",
    )
    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin={"l": 10, "r": 20, "t": 15, "b": 10},
        xaxis_title="Processed time",
        yaxis_title="Risk score (0–100)",
        yaxis={"range": [0, 100], "title": "Risk score (0–100)"},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return fig


def _decision_donut(log_df: pd.DataFrame):
    """Build a donut chart with stable colors for the three decisions."""
    decisions = (
        log_df.get("decision", pd.Series(dtype="object"))
        .fillna("")
        .astype(str)
        .str.upper()
    )
    counts = decisions.value_counts().reindex(["APPROVE", "FLAG", "BLOCK"], fill_value=0)
    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.66,
            sort=False,
            marker={"colors": ["#2ECC71", "#F39C12", "#E74C3C"]},
            textinfo="label+value",
            hovertemplate="%{label}: %{value}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=280,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        showlegend=False,
        annotations=[
            {
                "text": f"{len(log_df):,}<br>transactions",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 15, "color": "#F1F5F9"},
            }
        ],
    )
    return fig


def _style_decision_rows(row):
    """Apply green/orange/red row colors to the transaction table."""
    colors = {
        "APPROVE": "#173B27",
        "FLAG": "#493514",
        "BLOCK": "#491E1E",
    }
    background = colors.get(str(row.get("decision", "")).upper(), "transparent")
    return [f"background-color: {background}; color: #F1F5F9;"] * len(row)


def _render_alerts(alert_df: pd.DataFrame) -> None:
    """Show the most recent high and medium severity alerts."""
    if alert_df.empty:
        st.caption("No alerts yet. FLAG and BLOCK decisions will appear here.")
        return

    recent = alert_df.copy()
    if "severity" in recent.columns:
        recent = recent[recent["severity"].astype(str).str.upper().isin(["HIGH", "MEDIUM"])]
    if "timestamp" in recent.columns:
        recent["_sort_time"] = pd.to_datetime(recent["timestamp"], errors="coerce")
        recent = recent.sort_values("_sort_time", na_position="first")

    recent = recent.tail(10).iloc[::-1]
    if recent.empty:
        st.caption("No HIGH or MEDIUM alerts yet.")
        return

    for _, alert in recent.iterrows():
        severity = str(alert.get("severity", "MEDIUM")).upper()
        with st.container(border=True):
            st.write(
                f"**:{'red' if severity == 'HIGH' else 'orange'}[{severity}]** · "
                f"**{alert.get('txn_id', 'Unknown transaction')}**"
                f" · Risk {_number(alert.get('risk_score'))}"
            )
            st.caption(str(alert.get("reason", "No reason recorded")))
            st.caption(
                f"Amount: {_number(alert.get('raw_amount'))} · "
                f"{alert.get('timestamp', '')}"
            )


@st.cache_data(max_entries=500, show_spinner=False)
def _cached_transaction_explanation(txn_id: str):
    """Compute SHAP once per selected transaction, not on every fragment tick."""
    from src.explain import explain_transaction, to_plain_language

    feature_row = get_features_for_txn(txn_id)
    if feature_row is None:
        return None, None
    explanation = explain_transaction(feature_row, top_k=5)
    return explanation, to_plain_language(explanation)


def _alerted_transaction_options(alert_df: pd.DataFrame, log_df: pd.DataFrame) -> list[str]:
    """Return logged FLAG/BLOCK transaction IDs which also have saved alerts."""
    if alert_df.empty or log_df.empty:
        return []
    if not {"txn_id", "severity"}.issubset(alert_df.columns):
        return []
    if not {"txn_id", "decision"}.issubset(log_df.columns):
        return []

    alert_rows = alert_df[
        alert_df["severity"].astype(str).str.upper().isin(["HIGH", "MEDIUM"])
    ]
    alert_ids = set(alert_rows["txn_id"].dropna().astype(str))
    flagged_rows = log_df[
        log_df["decision"].astype(str).str.upper().isin(["FLAG", "BLOCK"])
    ]
    transaction_ids = flagged_rows["txn_id"].dropna().astype(str)
    options = list(dict.fromkeys(txn_id for txn_id in transaction_ids if txn_id in alert_ids))

    if "timestamp" in alert_rows.columns:
        timestamp_lookup = (
            alert_rows.assign(
                _timestamp=pd.to_datetime(alert_rows["timestamp"], errors="coerce")
            )
            .groupby("txn_id")["_timestamp"]
            .max()
        )
        options.sort(
            key=lambda txn_id: timestamp_lookup.get(txn_id, pd.NaT),
            reverse=True,
        )
    return options


def _feature_impact_figure(explanation: list[dict]):
    """Plot selected transaction's SHAP contributions as a red/green bar chart."""
    ranked = pd.DataFrame(explanation).copy()
    ranked["bar_color"] = ranked["shap_value"].map(
        lambda value: "#E74C3C" if value >= 0 else "#2ECC71"
    )
    fig = go.Figure(
        go.Bar(
            x=ranked["shap_value"],
            y=ranked["feature"],
            orientation="h",
            marker_color=ranked["bar_color"],
            customdata=ranked["direction"],
            hovertemplate=(
                "%{y}<br>SHAP value: %{x:.4f}<br>%{customdata}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_color="#AAB3C2", line_width=1)
    fig.update_layout(
        template="plotly_dark",
        height=270,
        margin={"l": 10, "r": 10, "t": 15, "b": 10},
        xaxis_title="Contribution to fraud risk",
        yaxis_title="",
        yaxis={"autorange": "reversed"},
        showlegend=False,
    )
    return fig


def _render_why_flagged(alert_df: pd.DataFrame, log_df: pd.DataFrame) -> None:
    """Explain one user-selected FLAG/BLOCK transaction on demand."""
    st.subheader("Why was this flagged?")
    options = _alerted_transaction_options(alert_df, log_df)
    if not options:
        st.info("No FLAG or BLOCK alerts are available to explain yet.")
        return

    log_rows = log_df.drop_duplicates(subset=["txn_id"], keep="last").set_index("txn_id")
    selected_txn = st.selectbox(
        "Choose a flagged or blocked transaction",
        options=options,
        index=None,
        placeholder="Select an alert to explain…",
        format_func=lambda txn_id: (
            "Select an alert to explain…"
            if txn_id is None
            else f"{txn_id} · {str(log_rows.loc[txn_id, 'decision']).upper()} · "
            f"risk {_number(log_rows.loc[txn_id, 'risk_score'])}"
        ),
        key="why_flagged_transaction",
    )
    if selected_txn is None:
        st.caption("Choose an alert to calculate its SHAP feature contributions.")
        return
    selected = log_rows.loc[selected_txn]

    details = st.columns(4)
    _metric_card(details[0], "Risk score", _number(selected.get("risk_score")))
    _metric_card(details[1], "Decision", str(selected.get("decision", "—")).upper())
    _metric_card(details[2], "XGBoost probability", _number(selected.get("xgb_prob"), 4))
    _metric_card(details[3], "Anomaly score", _number(selected.get("anomaly_score"), 4))

    latency = _number(selected.get("latency_ms"), 3)
    st.caption(f"Latency for this decision: {latency} ms")

    # This cached call is keyed by the selected transaction ID. The fragment
    # refreshes every second, but SHAP is calculated only the first time a
    # transaction is selected (or after its cached explanation is cleared).
    with st.spinner("Explaining the selected transaction…"):
        explanation, plain_language = _cached_transaction_explanation(selected_txn)
    if explanation is None:
        st.warning("The selected transaction was not found in the held-out test features.")
        return

    chart_column, text_column = st.columns([1.3, 1], gap="large")
    with chart_column:
        st.plotly_chart(
            _feature_impact_figure(explanation),
            use_container_width=True,
            key=f"shap_explanation_{selected_txn}",
        )
        st.caption("Red bars raise the model's fraud risk; green bars lower it.")
    with text_column:
        st.markdown("**Plain-language explanation**")
        st.write(plain_language)


def _render_live_content() -> None:
    """Refresh all live dashboard content without rerunning the full page."""
    running = is_running()
    title_column, status_column = st.columns([8, 2], vertical_alignment="center")
    with title_column:
        st.title("Fraud Shield on IBM Z")
        st.caption("Real-time AI for critical decisions")
    with status_column:
        badge_style = "status-running" if running else "status-stopped"
        badge_text = "Running" if running else "Stopped"
        st.markdown(
            f'<div style="text-align:right"><span class="status-badge {badge_style}">'
            f"{badge_text}</span></div>",
            unsafe_allow_html=True,
        )

    log_df = load_stream_log()
    alert_df = load_alerts()
    kpis = compute_kpis(log_df)

    st.subheader("Live overview")
    kpi_columns = st.columns(6)
    kpi_values = [
        ("Processed", f"{kpis['total_processed']:,}"),
        ("Approved", f"{kpis['approved']:,}"),
        ("Flagged", f"{kpis['flagged']:,}"),
        ("Blocked", f"{kpis['blocked']:,}"),
        ("Avg latency (ms)", _number(kpis["average_latency"])),
        ("Money blocked", _number(kpis["money_blocked"])),
    ]
    for column, (label, value) in zip(kpi_columns, kpi_values):
        _metric_card(column, label, value)

    if log_df.empty:
        st.info("Press Start to begin the simulation.")

    flag_threshold, block_threshold = _load_risk_thresholds()
    left_column, right_column = st.columns([1.7, 1], gap="large")
    with left_column:
        st.subheader("Live risk score · last 200 transactions")
        st.plotly_chart(
            _risk_score_figure(log_df, flag_threshold, block_threshold),
            use_container_width=True,
            key="live_risk_score_chart",
        )
    with right_column:
        st.subheader("Decision breakdown")
        st.plotly_chart(
            _decision_donut(log_df),
            use_container_width=True,
            key="decision_donut_chart",
        )

    table_column, alerts_column = st.columns([1.4, 1], gap="large")
    with table_column:
        st.subheader("Latest transactions")
        display_columns = [
            "txn_id",
            "risk_score",
            "decision",
            "xgb_prob",
            "anomaly_score",
            "raw_amount",
            "latency_ms",
            "processed_at",
        ]
        if log_df.empty:
            st.caption("Transactions will appear here as they are scored.")
        else:
            available = [name for name in display_columns if name in log_df.columns]
            recent = log_df.tail(15).iloc[::-1].loc[:, available].copy()
            if "risk_score" in recent:
                recent["risk_score"] = pd.to_numeric(recent["risk_score"], errors="coerce").round(2)
            if "xgb_prob" in recent:
                recent["xgb_prob"] = pd.to_numeric(recent["xgb_prob"], errors="coerce").round(4)
            if "anomaly_score" in recent:
                recent["anomaly_score"] = pd.to_numeric(recent["anomaly_score"], errors="coerce").round(4)
            if "raw_amount" in recent:
                recent["raw_amount"] = pd.to_numeric(recent["raw_amount"], errors="coerce").round(2)
            styled = recent.style.apply(_style_decision_rows, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True, height=360)

    with alerts_column:
        st.subheader("Latest alerts")
        _render_alerts(alert_df)

    st.divider()
    _render_why_flagged(alert_df, log_df)


@st.fragment(run_every=1)
def _live_fragment() -> None:
    """Refresh charts, tables, KPIs, alerts, and status every second."""
    _render_live_content()


def _load_json_file(path: str, label: str) -> dict | None:
    """Load a documentation JSON file, surfacing missing/corrupt files in UI."""
    try:
        with open(path, "r", encoding="utf-8") as json_file:
            value = json.load(json_file)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, json.JSONDecodeError, ValueError) as error:
        st.warning(f"Could not load {label} from `{path}`: {error}")
        return None


def _first_number(data: dict, keys: tuple[str, ...]):
    """Find the first finite numeric field among likely amount-key names."""
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


_SAVED_AMOUNT_KEYS = (
    "money_saved",
    "amount_saved",
    "saved_amount",
    "fraud_amount_blocked",
    "fraud_blocked_amount",
    "fraud_prevented_amount",
)
_LOST_AMOUNT_KEYS = (
    "money_lost",
    "amount_lost",
    "lost_amount",
    "fraud_amount_missed",
    "fraud_missed_amount",
    "fraud_approved_amount",
)


def _impact_figure(impact_data: dict | None):
    """Build an interactive saved-versus-lost bar chart from impact JSON."""
    fig = go.Figure()
    modes = []
    if impact_data:
        for name, aliases in (
            ("Batch", ("batch", "batch_evaluation")),
            ("Real-time", ("real_time", "realtime", "stream", "real_time_stream")),
        ):
            section = next(
                (impact_data[key] for key in aliases if isinstance(impact_data.get(key), dict)),
                None,
            )
            if section is not None:
                saved = _first_number(section, _SAVED_AMOUNT_KEYS)
                lost = _first_number(section, _LOST_AMOUNT_KEYS)
                if saved is not None or lost is not None:
                    modes.append((name, saved or 0.0, lost or 0.0))

        # Also accept one overall comparison object with root-level totals.
        if not modes:
            saved = _first_number(impact_data, _SAVED_AMOUNT_KEYS)
            lost = _first_number(impact_data, _LOST_AMOUNT_KEYS)
            if saved is not None or lost is not None:
                modes.append(("Test-set policy", saved or 0.0, lost or 0.0))

    colors = {"saved": "#2ECC71", "lost": "#E74C3C"}
    if modes:
        fig.add_trace(
            go.Bar(
                name="Money saved",
                x=[mode[0] for mode in modes],
                y=[mode[1] for mode in modes],
                marker_color=colors["saved"],
                hovertemplate="%{x}<br>Money saved: %{y:,.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Money lost",
                x=[mode[0] for mode in modes],
                y=[mode[2] for mode in modes],
                marker_color=colors["lost"],
                hovertemplate="%{x}<br>Money lost: %{y:,.2f}<extra></extra>",
            )
        )
    else:
        fig.add_annotation(
            text="No saved/lost transaction amounts are recorded in this file yet.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": "#AAB3C2", "size": 14},
        )

    fig.update_layout(
        title="Money saved vs. fraud loss",
        template="plotly_dark",
        barmode="group",
        height=320,
        margin={"l": 15, "r": 15, "t": 55, "b": 15},
        yaxis_title="Amount (dataset units)",
        xaxis_title="Evaluation mode",
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return fig


def _render_impact_tab() -> None:
    """Display model/stream performance and any recorded monetary impact."""
    st.header("Impact")
    st.caption("Test-set model results and the local stream benchmark.")
    impact_data = _load_json_file(BATCH_VS_REALTIME_PATH, "money impact results")
    benchmark = _load_json_file(BENCHMARK_PATH, "benchmark results")
    metrics = _load_json_file(METRICS_DAY2_PATH, "Day 2 evaluation metrics")

    st.subheader("Money saved vs. fraud loss")
    st.plotly_chart(
        _impact_figure(impact_data),
        use_container_width=True,
        key="impact_money_chart",
    )

    if impact_data:
        currency = impact_data.get("currency", "dataset units")
        st.caption(f"Amount values are shown in {currency}.")

    if benchmark:
        st.subheader("Batch and stream benchmark")
        batch = benchmark.get("batch", {})
        stream = benchmark.get("real_time_stream", {})
        cards = st.columns(6)
        benchmark_cards = [
            ("Batch rows", f"{int(batch.get('transactions_scored', 0)):,}"),
            ("Stream events", f"{int(stream.get('transactions_scored', 0)):,}"),
            ("Batch avg latency", f"{float(batch.get('average_latency_ms', 0)):.2f} ms"),
            ("Stream avg latency", f"{float(stream.get('average_latency_ms', 0)):.2f} ms"),
            ("Batch p95 latency", f"{float(batch.get('p95_latency_ms', 0)):.2f} ms"),
            ("Stream p95 latency", f"{float(stream.get('p95_latency_ms', 0)):.2f} ms"),
        ]
        for column, (label, value) in zip(cards, benchmark_cards):
            _metric_card(column, label, value)
        caveat = benchmark.get("caveat")
        if caveat:
            st.caption(caveat)

    if metrics:
        st.subheader("Day 2 fraud outcomes")
        ranking = metrics.get("ranking_metrics", {})
        xgb = ranking.get("xgboost_alone", {})
        outcomes = metrics.get("outcomes", {})
        cards = st.columns(5)
        day2_cards = [
            ("XGBoost PR-AUC", f"{float(xgb.get('pr_auc', 0)):.4f}"),
            ("XGBoost ROC-AUC", f"{float(xgb.get('roc_auc', 0)):.4f}"),
            ("Fraud caught", f"{int(outcomes.get('fraud_caught_block_plus_flag', 0)):,}"),
            ("Fraud missed", f"{int(outcomes.get('fraud_missed_approved', 0)):,}"),
            ("Genuine blocked", f"{int(outcomes.get('genuine_customers_blocked', 0)):,}"),
        ]
        for column, (label, value) in zip(cards, day2_cards):
            _metric_card(column, label, value)
        st.caption(
            "The combined score had the same PR-AUC and ROC-AUC as XGBoost alone; "
            "the tuned validation weight for anomaly detection was zero."
        )

    with st.expander("Assumptions"):
        st.markdown(
            "- Saved and lost money are shown only when amount totals are present "
            "in `docs/batch_vs_realtime.json`; detection counts alone do not establish "
            "transaction value.\n"
            "- A blocked fraud amount can be treated as prevented exposure; a flagged "
            "transaction is not counted as saved unless a review actually stops it.\n"
            "- Fraud approved is treated as missed exposure. These are historical "
            "held-out test results, not a forecast of future losses.\n"
            "- The stream benchmark is a small local replay, so its laptop latency "
            "does not predict IBM Z production performance."
        )


def _render_ibm_z_tab() -> None:
    """Explain the intended transaction-scoring deployment on IBM Z."""
    st.header("IBM Z Deployment")
    st.markdown(
        "**Target deployment architecture. This demo simulates the scoring layer locally in Python.**"
    )
    architecture = r"""
    digraph FraudShield {
      graph [rankdir=LR, bgcolor="#0E1117", pad="0.25", nodesep="0.45", ranksep="0.65"];
      node [shape=box, style="rounded,filled", color="#2ECC71", fillcolor="#171B24", fontcolor="#F1F5F9", fontname="Arial", margin="0.18,0.12"];
      edge [color="#2ECC71", penwidth=1.6, arrowsize=0.8];
      payment [label="Payment app\non IBM Z"];
      stream [label="Transaction\nevent stream"];
      features [label="Feature\nengineering"];
      scoring [label="ML scoring\nIBM Machine Learning for z/OS\nTransactional AI"];
      decision [label="Decision engine\nAPPROVE / FLAG / BLOCK"];
      analysts [label="Analyst\ndashboard", color="#F39C12"];
      payment -> stream -> features -> scoring -> decision -> analysts;
    }
    """
    st.graphviz_chart(architecture, use_container_width=True)

    st.markdown("### Why score inside the IBM Z transaction?")
    st.markdown(
        "- **Low latency:** score and decide in the transaction flow, where milliseconds matter.\n"
        "- **Data stays secure:** sensitive payment data can remain within the bank's controlled IBM Z environment.\n"
        "- **Scale:** use the transaction platform's capacity to process high volumes consistently.\n"
        "- **No extra network hop:** avoid sending each transaction to a remote scoring service and waiting for a round trip."
    )


def main() -> None:
    """Render static sidebar controls and the independently refreshed view."""
    with st.sidebar:
        st.header("Stream controls")
        speed_tps = st.slider(
            "Speed (transactions/sec)", min_value=5, max_value=100, value=20, step=5
        )
        max_events = st.number_input(
            "Maximum events", min_value=1, max_value=56_962, value=2_000, step=100
        )

        running = is_running()
        if st.button("Start", type="primary", use_container_width=True, disabled=running):
            try:
                start_stream(speed_tps=speed_tps, max_events=max_events)
                st.success("Stream started. Logs were reset.")
            except (OSError, RuntimeError, ValueError) as error:
                st.error(f"Could not start the stream: {error}")

        if st.button("Stop", use_container_width=True, disabled=not running):
            stop_stream()
            st.info("Stream stopped.")

        if st.button("Reset logs", use_container_width=True):
            if is_running():
                stop_stream()
            _reset_logs()
            st.success("Stream and alert logs cleared.")

        st.divider()
        st.caption("The stream replays held-out test transactions in chronological order.")

    live_tab, impact_tab, deployment_tab = st.tabs(
        ["Live Monitor", "Impact", "IBM Z Deployment"]
    )
    with live_tab:
        _live_fragment()
    with impact_tab:
        _render_impact_tab()
    with deployment_tab:
        _render_ibm_z_tab()

    st.divider()
    st.caption("Built for the IBM Z Datathon")


if __name__ == "__main__":
    main()
