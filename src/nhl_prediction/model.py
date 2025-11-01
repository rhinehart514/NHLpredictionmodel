"""Shared modelling utilities."""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_baseline_model(C: float = 1.0, random_state: int | None = 42) -> Pipeline:
    """Return the logistic regression baseline pipeline."""
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=random_state, C=C)),
        ]
    )


def fit_model(model: Pipeline, features: pd.DataFrame, target: pd.Series, mask: pd.Series) -> Pipeline:
    """Fit the provided model on masked rows."""
    model.fit(features.loc[mask], target.loc[mask])
    return model


def predict_probabilities(model: Pipeline, features: pd.DataFrame, mask: pd.Series) -> np.ndarray:
    """Return probability of home win for masked rows."""
    return model.predict_proba(features.loc[mask])[:, 1]


def compute_metrics(y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute standard evaluation metrics."""
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
    return metrics


def format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    """Format metrics into a printable string."""
    parts = [f"{prefix}"]
    parts.extend(f"{key.replace('_', ' ').title()}: {value:0.4f}" for key, value in metrics.items())
    return " | ".join(parts)


def compute_feature_effects(model: Pipeline, feature_names: pd.Index) -> pd.DataFrame:
    """Return coefficient impacts in original feature scale."""
    scaler: StandardScaler = model.named_steps["scale"]
    clf: LogisticRegression = model.named_steps["clf"]

    coef = clf.coef_[0]
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    adjusted_coef = coef / scale

    importance = pd.DataFrame(
        {"feature": feature_names, "coefficient": adjusted_coef, "absolute_importance": np.abs(adjusted_coef)}
    ).sort_values("absolute_importance", ascending=False)
    return importance


def create_histgb_model(params: Dict[str, Any] | None = None, random_state: int | None = 42) -> Pipeline:
    """Return a histogram-based gradient boosting classifier wrapped in a pipeline."""
    default_params: Dict[str, Any] = {
        "learning_rate": 0.05,
        "max_depth": 3,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "early_stopping": False,
        "random_state": random_state,
    }
    if params:
        default_params.update(params)
    clf = HistGradientBoostingClassifier(**default_params)
    return Pipeline(steps=[("clf", clf)])


def predict_calibrated_probabilities(
    model: Pipeline, features: pd.DataFrame, mask: pd.Series, calibrator: IsotonicRegression | None = None
) -> np.ndarray:
    """Predict calibrated probabilities for masked rows."""
    probs = predict_probabilities(model, features, mask)
    if calibrator is not None:
        probs = calibrator.predict(probs)
    return np.clip(probs, 0.0, 1.0)


def blend_with_elo(logistic_probs: np.ndarray, elo_probs: np.ndarray, weight: float = 0.6) -> np.ndarray:
    """Blend logistic model probabilities with Elo expectations."""
    blended = weight * logistic_probs + (1.0 - weight) * elo_probs
    return np.clip(blended, 0.0, 1.0)


def tune_logreg_c(
    candidate_cs: Sequence[float],
    features: pd.DataFrame,
    target: pd.Series,
    games: pd.DataFrame,
    train_seasons: Sequence[str],
) -> float:
    """Select regularisation strength using the final training season as validation."""
    if len(train_seasons) < 2:
        return 1.0

    sorted_train = sorted(train_seasons)
    val_season = sorted_train[-1]
    core_seasons = sorted_train[:-1]

    core_mask = games["seasonId"].isin(core_seasons)
    val_mask = games["seasonId"] == val_season

    if core_mask.sum() == 0 or val_mask.sum() == 0:
        return 1.0

    best_c = candidate_cs[0]
    best_acc = float("-inf")
    best_loss = float("inf")

    for c in candidate_cs:
        model = create_baseline_model(C=c)
        model = fit_model(model, features, target, core_mask)
        probs = predict_probabilities(model, features, val_mask)
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(target.loc[val_mask], preds)
        loss = log_loss(target.loc[val_mask], probs)
        if acc > best_acc or (np.isclose(acc, best_acc) and loss < best_loss):
            best_acc = acc
            best_loss = loss
            best_c = c

    return best_c


def tune_histgb_params(
    param_grid: Sequence[Dict[str, Any]],
    features: pd.DataFrame,
    target: pd.Series,
    games: pd.DataFrame,
    train_seasons: Sequence[str],
) -> Dict[str, Any]:
    """Select gradient boosting hyperparameters using final training season as validation."""
    if not param_grid:
        return {}
    if len(train_seasons) < 2:
        return dict(param_grid[0])

    sorted_train = sorted(train_seasons)
    val_season = sorted_train[-1]
    core_seasons = sorted_train[:-1]

    core_mask = games["seasonId"].isin(core_seasons)
    val_mask = games["seasonId"] == val_season

    if core_mask.sum() == 0 or val_mask.sum() == 0:
        return dict(param_grid[0])

    best_params = dict(param_grid[0])
    best_loss = float("inf")

    for candidate in param_grid:
        params = dict(candidate)
        model = create_histgb_model(params=params)
        model = fit_model(model, features, target, core_mask)
        probs = predict_probabilities(model, features, val_mask)
        loss = log_loss(target.loc[val_mask], probs)
        if loss < best_loss:
            best_loss = loss
            best_params = params

    return best_params


def find_optimal_threshold(probs: np.ndarray, y_true: pd.Series) -> tuple[float, float]:
    """Return probability threshold that maximises accuracy."""
    if len(probs) == 0:
        return 0.5, 0.0

    sorted_probs = np.unique(np.round(probs, 4))
    candidates = np.clip(
        np.concatenate(([0.0], (sorted_probs[:-1] + sorted_probs[1:]) / 2, [1.0])), 0.0, 1.0
    )

    best_threshold = 0.5
    best_accuracy = -np.inf
    for threshold in candidates:
        preds = (probs >= threshold).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold

    return float(best_threshold), float(best_accuracy)


def calibrate_model_threshold(
    model_factory: Callable[[], Pipeline],
    features: pd.DataFrame,
    target: pd.Series,
    games: pd.DataFrame,
    train_seasons: Sequence[str],
) -> tuple[float, float | None, IsotonicRegression | None]:
    """Determine decision threshold using validation season."""
    if len(train_seasons) < 2:
        return 0.5, None, None

    sorted_train = sorted(train_seasons)
    val_season = sorted_train[-1]
    core_seasons = sorted_train[:-1]

    core_mask = games["seasonId"].isin(core_seasons)
    val_mask = games["seasonId"] == val_season

    if core_mask.sum() == 0 or val_mask.sum() == 0:
        return 0.5, None, None

    model = model_factory()
    model = fit_model(model, features, target, core_mask)
    val_probs = predict_probabilities(model, features, val_mask)
    base_acc = accuracy_score(target.loc[val_mask], (val_probs >= 0.5).astype(int))
    threshold, val_acc = find_optimal_threshold(val_probs, target.loc[val_mask])
    calibrator: IsotonicRegression | None = None
    unique_probs = np.unique(val_probs)
    if unique_probs.size > 1:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_probs, target.loc[val_mask])

    if val_acc is None or val_acc <= base_acc + 0.01:
        return 0.5, base_acc, calibrator
    return threshold, val_acc, calibrator


def calibrate_threshold(
    C: float,
    features: pd.DataFrame,
    target: pd.Series,
    games: pd.DataFrame,
    train_seasons: Sequence[str],
) -> tuple[float, float | None, IsotonicRegression | None]:
    """Logistic regression-specific threshold calibration (backwards compatible wrapper)."""
    factory = lambda: create_baseline_model(C=C)
    return calibrate_model_threshold(factory, features, target, games, train_seasons)


__all__ = [
    "create_baseline_model",
    "create_histgb_model",
    "fit_model",
    "predict_probabilities",
    "predict_calibrated_probabilities",
    "compute_metrics",
    "format_metrics",
    "compute_feature_effects",
    "tune_logreg_c",
    "tune_histgb_params",
    "find_optimal_threshold",
    "calibrate_model_threshold",
    "calibrate_threshold",
    "blend_with_elo",
]
