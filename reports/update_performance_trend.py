"""Generate performance trend chart from history CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    history_path = Path("reports/performance_history.csv")
    df = pd.read_csv(history_path, parse_dates=["date"])
    df.sort_values("date", inplace=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["date"], df["accuracy"], marker="o", label="Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.58, 0.66)

    ax2 = ax.twinx()
    ax2.plot(df["date"], df["roc_auc"], marker="s", color="#1f77b4", linestyle="--", label="ROC-AUC")
    ax2.plot(df["date"], df["log_loss"], marker="^", color="#d62728", linestyle=":", label="Log Loss")
    ax2.set_ylabel("ROC-AUC / Log Loss")

    ax.set_title("Model Performance Over Time")
    ax.grid(True, linestyle="--", alpha=0.3)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower right")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig("reports/performance_trend.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
