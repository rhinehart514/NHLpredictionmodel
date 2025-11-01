"""Command-line interface to train and evaluate NHL game prediction model."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score

from .model import (
    compute_metrics,
    create_baseline_model,
    create_histgb_model,
    fit_model,
    format_metrics,
    predict_probabilities,
    tune_histgb_params,
    tune_logreg_c,
    find_optimal_threshold,
)
from .pipeline import Dataset, build_dataset

app = typer.Typer(add_completion=False)
console = Console()


def _resolve_seasons(train_seasons: List[str] | None, test_season: str | None) -> tuple[List[str], str]:
    default_train = ["20212022", "20222023"]
    default_test = "20232024"
    train = train_seasons or default_train
    test = test_season or default_test
    if test in train:
        raise typer.BadParameter("Test season must be distinct from training seasons.")
    return train, test


def _is_better(candidate: Dict[str, Any], incumbent: Dict[str, Any]) -> bool:
    cand_val = candidate["val_metrics"]
    inc_val = incumbent["val_metrics"]
    if cand_val is not None and inc_val is not None:
        acc_diff = cand_val["accuracy"] - inc_val["accuracy"]
        if acc_diff > 0.01:
            return True
        if acc_diff < -0.01:
            return False

        loss_diff = cand_val["log_loss"] - inc_val["log_loss"]
        if loss_diff < -0.01:
            return True
        if loss_diff > 0.01:
            return False

        auc_diff = cand_val["roc_auc"] - inc_val["roc_auc"]
        if abs(auc_diff) > 0.01:
            return auc_diff > 0

        return candidate["train_metrics"]["accuracy"] > incumbent["train_metrics"]["accuracy"]

    if cand_val is not None:
        return True
    if inc_val is not None:
        return False

    acc_diff = candidate["test_metrics"]["accuracy"] - incumbent["test_metrics"]["accuracy"]
    if abs(acc_diff) > 0.01:
        return acc_diff > 0
    return candidate["test_metrics"]["log_loss"] < incumbent["test_metrics"]["log_loss"]


def compare_models(
    dataset: Dataset,
    train_ids: List[str],
    test_id: str,
    logreg_c_override: Optional[float] = None,
) -> Dict[str, Any]:
    games = dataset.games
    features = dataset.features
    target = dataset.target

    train_mask = games["seasonId"].isin(train_ids)
    test_mask = games["seasonId"] == test_id

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Insufficient games for the provided seasons.")

    sorted_train = sorted(train_ids)
    core_mask: Optional[pd.Series] = None
    val_mask: Optional[pd.Series] = None
    if len(sorted_train) >= 2:
        val_season = sorted_train[-1]
        core_seasons = sorted_train[:-1]
        core_mask = games["seasonId"].isin(core_seasons)
        val_mask = games["seasonId"] == val_season

    candidates: List[Dict[str, Any]] = []

    candidate_cs = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0]
    best_c = logreg_c_override if logreg_c_override is not None else tune_logreg_c(
        candidate_cs, features, target, games, train_ids
    )
    log_result = evaluate_candidate(
        name="Logistic Regression",
        hyperparams={"C": best_c},
        model_factory=lambda: create_baseline_model(C=best_c),
        features=features,
        target=target,
        train_mask=train_mask,
        test_mask=test_mask,
        core_mask=core_mask,
        val_mask=val_mask,
    )
    candidates.append(log_result)

    if logreg_c_override is None:
        gb_param_grid: Sequence[Dict[str, Any]] = [
            {"learning_rate": 0.05, "max_depth": 3, "max_leaf_nodes": 31, "min_samples_leaf": 20},
            {"learning_rate": 0.05, "max_depth": 4, "max_leaf_nodes": 63, "min_samples_leaf": 25},
            {"learning_rate": 0.08, "max_depth": 3, "max_leaf_nodes": 63, "min_samples_leaf": 30, "l2_regularization": 0.01},
            {"learning_rate": 0.1, "max_depth": 3, "max_leaf_nodes": 31, "min_samples_leaf": 35, "l2_regularization": 0.02},
        ]
        best_gb_params = tune_histgb_params(gb_param_grid, features, target, games, train_ids)
        gb_result = evaluate_candidate(
            name="HistGradientBoosting",
            hyperparams=best_gb_params,
            model_factory=lambda: create_histgb_model(params=best_gb_params),
            features=features,
            target=target,
            train_mask=train_mask,
            test_mask=test_mask,
            core_mask=core_mask,
            val_mask=val_mask,
        )
        candidates.append(gb_result)

    best_result = candidates[0]
    for candidate in candidates[1:]:
        if _is_better(candidate, best_result):
            best_result = candidate

    return {
        "best_result": best_result,
        "candidates": candidates,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "core_mask": core_mask,
        "val_mask": val_mask,
    }

def evaluate_candidate(
    name: str,
    hyperparams: Dict[str, Any],
    model_factory: Callable[[], Any],
    features: pd.DataFrame,
    target: pd.Series,
    train_mask: pd.Series,
    test_mask: pd.Series,
    core_mask: Optional[pd.Series],
    val_mask: Optional[pd.Series],
) -> Dict[str, Any]:
    """Fit, calibrate, and score a single model candidate."""
    calibrator: IsotonicRegression | None = None
    decision_threshold = 0.5
    val_metrics: Optional[Dict[str, float]] = None
    val_accuracy: Optional[float] = None
    recommended_threshold = 0.5

    if core_mask is not None and val_mask is not None and core_mask.sum() > 0 and val_mask.sum() > 0:
        val_model = model_factory()
        val_model = fit_model(val_model, features, target, core_mask)
        val_probs = predict_probabilities(val_model, features, val_mask)

        base_acc = accuracy_score(target.loc[val_mask], (val_probs >= 0.5).astype(int))
        threshold, threshold_acc = find_optimal_threshold(val_probs, target.loc[val_mask])

        if np.unique(val_probs).size > 1:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(val_probs, target.loc[val_mask])

        if threshold_acc > base_acc + 0.01:
            recommended_threshold = threshold
            val_accuracy = threshold_acc
        else:
            recommended_threshold = 0.5
            val_accuracy = base_acc

        val_metrics = compute_metrics(target.loc[val_mask], val_probs)
        val_metrics["accuracy"] = val_accuracy
        val_metrics["recommended_threshold"] = recommended_threshold
    else:
        calibrator = None
        val_accuracy = None
        val_metrics = None

    decision_threshold = 0.5

    model = model_factory()
    model = fit_model(model, features, target, train_mask)

    train_probs_raw = predict_probabilities(model, features, train_mask)
    test_probs_raw = predict_probabilities(model, features, test_mask)

    train_metrics = compute_metrics(target.loc[train_mask], train_probs_raw)
    test_metrics = compute_metrics(target.loc[test_mask], test_probs_raw)

    train_metrics["accuracy"] = accuracy_score(
        target.loc[train_mask], (train_probs_raw >= decision_threshold).astype(int)
    )
    test_metrics["accuracy"] = accuracy_score(
        target.loc[test_mask], (test_probs_raw >= decision_threshold).astype(int)
    )

    return {
        "name": name,
        "model": model,
        "hyperparams": hyperparams,
        "calibrator": calibrator,
        "decision_threshold": decision_threshold,
        "recommended_threshold": recommended_threshold,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "val_accuracy": val_accuracy,
    }

@app.command()
def train(
    train_seasons: List[str] = typer.Option(None, help="Season IDs for training data."),
    test_season: str = typer.Option(None, help="Hold-out season ID for evaluation."),
    logreg_c: Optional[float] = typer.Option(
        None,
        help="Override logistic regression C (skip tuning and force this value).",
    ),
) -> None:
    """Train the model suite and print evaluation metrics."""
    train_ids, test_id = _resolve_seasons(train_seasons, test_season)
    combined_seasons = sorted(set(train_ids + [test_id]))

    console.log(f"Fetching data for seasons: {', '.join(combined_seasons)}")
    dataset: Dataset = build_dataset(combined_seasons)
    target = dataset.target

    comparison = compare_models(dataset, train_ids, test_id, logreg_c_override=logreg_c)
    candidates = comparison["candidates"]
    best_result = comparison["best_result"]
    train_mask = comparison["train_mask"]
    test_mask = comparison["test_mask"]
    val_mask = comparison["val_mask"]

    for result in candidates:
        if result["name"] == "Logistic Regression":
            if logreg_c is not None:
                console.log(f"Logistic regression C set to {logreg_c}")
            else:
                console.log(f"Tuned logistic regression C={result['hyperparams']['C']}")
        elif result["name"] == "HistGradientBoosting":
            console.log(f"Tuned gradient boosting params={result['hyperparams']}")

    console.print("\n[bold]Validation Comparison[/bold]")
    if val_mask is not None and val_mask.sum() > 0:
        for result in candidates:
            val_metrics = result["val_metrics"]
            if val_metrics is None:
                console.print(f"- {result['name']}: validation split unavailable")
            else:
                console.print(
                    f"- {result['name']}: accuracy {val_metrics['accuracy']:.3f} | "
                    f"ROC-AUC {val_metrics['roc_auc']:.3f} | Log Loss {val_metrics['log_loss']:.3f} | "
                    f"recommended threshold {result['recommended_threshold']:.3f}"
                )
    else:
        console.print("- No separate validation season; selecting by test accuracy.")

    console.print(
        f"\n[bold]Selected Model:[/bold] {best_result['name']} "
        f"(decision threshold {best_result['decision_threshold']:.3f}, "
        f"recommended validation threshold {best_result['recommended_threshold']:.3f})"
    )
    console.print(f"Hyperparameters: {best_result['hyperparams']}")

    train_metrics = best_result["train_metrics"]
    test_metrics = best_result["test_metrics"]

    console.print("\n[bold]Training Metrics[/bold]")
    console.print(format_metrics("Train", train_metrics))

    console.print("\n[bold]Test Metrics[/bold]")
    console.print(format_metrics("Test", test_metrics))

    baseline = max(target.loc[test_mask].mean(), 1 - target.loc[test_mask].mean())
    console.print(f"\nBaseline Accuracy (Most Frequent Class): {baseline:0.4f}")


if __name__ == "__main__":
    app()
