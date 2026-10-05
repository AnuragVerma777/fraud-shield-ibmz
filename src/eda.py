"""
eda.py — Exploratory Data Analysis for Fraud Shield on IBM Z.

Generates and saves charts + a text report into reports/.
Run:  python -m src.eda

Charts produced:
  1. class_distribution.png     — bar chart of fraud vs legit
  2. amount_distribution.png    — histogram of Amount by class
  3. time_distribution.png      — transaction density over time
  4. correlation_heatmap.png    — top correlated features with Class
  5. eda_report.txt             — text summary
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import DOCS_DIR, TARGET_COL, PCA_FEATURES, ensure_dirs
from src.data_loader import load_raw_data, get_summary, validate_data


def plot_class_distribution(df: pd.DataFrame, save_dir: str) -> None:
    """Bar chart showing class imbalance (fraud vs legit)."""
    counts = df[TARGET_COL].value_counts()
    labels = ["Legit (0)", "Fraud (1)"]
    colors = ["#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, counts.values, color=colors, edgecolor="black")

    # annotate counts on bars
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1000, f"{v:,}", ha="center", fontweight="bold")

    ax.set_title("Class Distribution — Extreme Imbalance", fontsize=13)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "class_distribution.png"), dpi=150)
    plt.close(fig)
    print("  📊 Saved class_distribution.png")


def plot_amount_distribution(df: pd.DataFrame, save_dir: str) -> None:
    """Overlapping histograms of transaction Amount by class."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # left: full range
    for cls, color, label in [(0, "#2ecc71", "Legit"), (1, "#e74c3c", "Fraud")]:
        axes[0].hist(
            df[df[TARGET_COL] == cls]["Amount"],
            bins=80, alpha=0.6, color=color, label=label,
        )
    axes[0].set_title("Amount Distribution (full range)")
    axes[0].set_xlabel("Amount")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # right: zoom in (Amount < 500) — where most fraud lives
    for cls, color, label in [(0, "#2ecc71", "Legit"), (1, "#e74c3c", "Fraud")]:
        axes[1].hist(
            df[(df[TARGET_COL] == cls) & (df["Amount"] < 500)]["Amount"],
            bins=80, alpha=0.6, color=color, label=label,
        )
    axes[1].set_title("Amount Distribution (< $500)")
    axes[1].set_xlabel("Amount")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "amount_distribution.png"), dpi=150)
    plt.close(fig)
    print("  📊 Saved amount_distribution.png")


def plot_time_distribution(df: pd.DataFrame, save_dir: str) -> None:
    """Line/area plot of transaction density over hours."""
    df_plot = df.copy()
    df_plot["Hour"] = df_plot["Time"] / 3600  # seconds → hours

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(
        [df_plot[df_plot[TARGET_COL] == 0]["Hour"],
         df_plot[df_plot[TARGET_COL] == 1]["Hour"]],
        bins=48, stacked=True, color=["#2ecc71", "#e74c3c"],
        label=["Legit", "Fraud"], edgecolor="black", linewidth=0.3,
    )
    ax.set_title("Transactions Over Time (hours)")
    ax.set_xlabel("Time (hours since first txn)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "time_distribution.png"), dpi=150)
    plt.close(fig)
    print("  📊 Saved time_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str) -> None:
    """Heatmap of top-10 features most correlated with Class."""
    # correlations with target
    corr = df[PCA_FEATURES + ["Amount", TARGET_COL]].corr()[TARGET_COL]
    top_features = corr.abs().sort_values(ascending=False).head(11).index.tolist()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df[top_features].corr(),
        annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Top Features Correlated with Fraud (Class)")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  📊 Saved correlation_heatmap.png")


def write_eda_report(df: pd.DataFrame, save_dir: str) -> None:
    """Write a plain-text EDA summary report."""
    summary = get_summary(df)
    checks  = validate_data(df)

    lines = [
        "=" * 60,
        "  FRAUD SHIELD on IBM Z — EDA Report",
        "=" * 60,
        "",
        f"Total transactions : {summary['n_total']:,}",
        f"Legitimate (0)     : {summary['n_legit']:,}",
        f"Fraudulent (1)     : {summary['n_fraud']:,}",
        f"Fraud percentage   : {summary['fraud_pct']:.4f}%",
        f"Time span          : {summary['time_range_hours']:.1f} hours",
        "",
        "--- Amount Statistics ---",
        f"  Mean   : {summary['amount_stats']['mean']:.2f}",
        f"  Median : {summary['amount_stats']['median']:.2f}",
        f"  Min    : {summary['amount_stats']['min']:.2f}",
        f"  Max    : {summary['amount_stats']['max']:.2f}",
        "",
        "--- Data Quality ---",
        f"  Null values   : {checks['total_nulls']}",
        f"  Columns OK    : {checks['columns_ok']}",
        f"  Target values : {checks['target_values']}",
        "",
        "--- Key Observations ---",
        "  • Dataset is highly imbalanced (~0.17% fraud) — SMOTE needed.",
        "  • V1–V28 are PCA-transformed (pre-anonymised by ULB).",
        "  • Amount & Time are the only raw features — need scaling.",
        "  • No missing values — clean dataset.",
        "  • Fraud transactions tend to have lower amounts.",
        "",
        "=" * 60,
    ]

    report_path = os.path.join(save_dir, "eda_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  📝 Saved eda_report.txt")


def run_eda() -> None:
    """Run the full EDA pipeline."""
    ensure_dirs()
    print("\n🔍 Starting Exploratory Data Analysis …\n")

    df = load_raw_data()

    print("\nGenerating charts …")
    plot_class_distribution(df, DOCS_DIR)
    plot_amount_distribution(df, DOCS_DIR)
    plot_time_distribution(df, DOCS_DIR)
    plot_correlation_heatmap(df, DOCS_DIR)
    write_eda_report(df, DOCS_DIR)

    print(f"\n✅ EDA complete! All outputs saved to: {DOCS_DIR}\n")


# ─────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    run_eda()
