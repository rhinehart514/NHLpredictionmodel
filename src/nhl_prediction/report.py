"""Generate shareable reports and visualisations for the NHL model."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # ensure plots render in headless environments.
import matplotlib.pyplot as plt
import pandas as pd
import typer
from rich.console import Console
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, roc_curve

from .model import (
    blend_with_elo,
    calibrate_threshold,
    compute_feature_effects,
    compute_metrics,
    create_baseline_model,
    fit_model,
    predict_probabilities,
    tune_logreg_c,
)
from .pipeline import Dataset, build_dataset

app = typer.Typer(add_completion=False)
console = Console()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_seasons(train_seasons: List[str] | None, test_season: str | None) -> tuple[List[str], str]:
    default_train = ["20212022", "20222023"]
    default_test = "20232024"
    train = train_seasons or default_train
    test = test_season or default_test
    if test in train:
        raise typer.BadParameter("Test season must be distinct from training seasons.")
    return train, test


def _plot_roc(y_true, y_prob, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {compute_metrics(y_true, y_prob)['roc_auc']:0.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Home Win Classifier")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_calibration(y_true, y_prob, path: Path) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("Predicted Probability (Home Win)")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve – Reliability of Probabilities")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_confusion(y_true, y_prob, threshold: float, path: Path) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["Away Win", "Home Win"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix @ 0.5 Threshold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_predictions(
    dataset: Dataset, probs: pd.Series, mask: pd.Series, threshold: float, output_path: Path
) -> None:
    games = dataset.games.loc[mask].copy()
    games["home_win_probability"] = probs
    games["predicted_home_win"] = (probs >= threshold).astype(int)
    games["correct"] = games["predicted_home_win"] == games["home_win"]

    columns = [
        "gameDate",
        "gameId",
        "teamFullName_home",
        "teamFullName_away",
        "home_score",
        "away_score",
        "home_win_probability",
        "predicted_home_win",
        "home_win",
        "correct",
        "games_played_prior_home",
        "games_played_prior_away",
        "rolling_win_pct_5_diff",
        "rolling_goal_diff_5_diff",
        "special_teams_matchup",
    ]
    available_columns = [col for col in columns if col in games.columns]
    games.loc[:, available_columns].sort_values("gameDate").to_csv(output_path, index=False)


def _save_feature_effects(model, feature_names, output_path: Path) -> None:
    feature_df = compute_feature_effects(model, feature_names)
    feature_df.to_csv(output_path, index=False)


@app.command()
def report(
    train_seasons: List[str] = typer.Option(None, help="Season IDs for training data."),
    test_season: str = typer.Option(None, help="Hold-out season ID for evaluation."),
    output_dir: Path = typer.Option(Path("reports"), help="Directory to store CSVs and plots."),
    logreg_c: Optional[float] = typer.Option(
        None, help="Override logistic regression C (skip tuning and force this value)."
    ),
) -> None:
    """Generate CSV outputs and plots for stakeholders."""
    train_ids, test_id = _resolve_seasons(train_seasons, test_season)
    combined_seasons = sorted(set(train_ids + [test_id]))

    console.log(f"Preparing dataset for seasons: {', '.join(combined_seasons)}")
    dataset: Dataset = build_dataset(combined_seasons)

    features = dataset.features
    target = dataset.target
    games = dataset.games

    train_mask = games["seasonId"].isin(train_ids)
    test_mask = games["seasonId"] == test_id

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise typer.BadParameter("Insufficient games for the provided seasons.")

    candidate_cs = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0]
    best_c = logreg_c if logreg_c is not None else tune_logreg_c(candidate_cs, features, target, games, train_ids)
    threshold, val_acc, calibrator = calibrate_threshold(best_c, features, target, games, train_ids)
    decision_threshold = 0.5

    if logreg_c is not None:
        console.log(f"Using logistic regression C={best_c} (override)")
    else:
        console.log(f"Selected logistic regression C={best_c}")
    if val_acc is not None:
        console.log(
            f"Validation suggested threshold {threshold:.3f} (validation accuracy {val_acc:.3f}); "
            "reports retain 0.500 for out-of-sample comparability."
        )
    else:
        console.log("Using default 0.500 decision threshold")

    model = create_baseline_model(C=best_c)
    model = fit_model(model, features, target, train_mask)

    train_probs_raw = predict_probabilities(model, features, train_mask)
    test_probs_raw = predict_probabilities(model, features, test_mask)
    train_probs = blend_with_elo(train_probs_raw, games.loc[train_mask, "elo_expectation_home"].to_numpy())
    test_probs = blend_with_elo(test_probs_raw, games.loc[test_mask, "elo_expectation_home"].to_numpy())

    train_metrics = compute_metrics(target.loc[train_mask], train_probs)
    test_metrics = compute_metrics(target.loc[test_mask], test_probs)

    train_metrics["accuracy"] = accuracy_score(
        target.loc[train_mask], (train_probs >= decision_threshold).astype(int)
    )
    test_metrics["accuracy"] = accuracy_score(
        target.loc[test_mask], (test_probs >= decision_threshold).astype(int)
    )

    console.print("[bold]Metrics[/bold]")
    console.print(f"Train – Accuracy {train_metrics['accuracy']:0.3f}, LogLoss {train_metrics['log_loss']:0.3f}, "
                  f"ROC-AUC {train_metrics['roc_auc']:0.3f}")
    console.print(f"Test  – Accuracy {test_metrics['accuracy']:0.3f}, LogLoss {test_metrics['log_loss']:0.3f}, "
                  f"ROC-AUC {test_metrics['roc_auc']:0.3f}")

    out_dir = _ensure_dir(output_dir)
    console.log(f"Saving outputs to {out_dir}")

    _save_predictions(
        dataset,
        pd.Series(test_probs, index=target.loc[test_mask].index),
        test_mask,
        decision_threshold,
        out_dir / f"predictions_{test_id}.csv",
    )
    _save_feature_effects(model, features.columns, out_dir / "feature_importance.csv")

    _plot_roc(target.loc[test_mask], test_probs, out_dir / "roc_curve.png")
    _plot_calibration(target.loc[test_mask], test_probs, out_dir / "calibration_curve.png")
    _plot_confusion(target.loc[test_mask], test_probs, decision_threshold, out_dir / "confusion_matrix.png")

    console.log("Report generation complete.")


if __name__ == "__main__":
    app()
