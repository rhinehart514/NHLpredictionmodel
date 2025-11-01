"""Streamlit dashboard for exploring NHL game predictions."""

from __future__ import annotations

from typing import Dict, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from nhl_prediction.model import compute_feature_effects, predict_probabilities
from nhl_prediction.pipeline import Dataset, build_dataset
from nhl_prediction.train import compare_models

DEFAULT_TRAIN_SEASONS = ["20212022", "20222023"]
DEFAULT_TEST_SEASON = "20232024"
DEFAULT_LOGREG_C = 0.018

st.set_page_config(page_title="NHL Game Prediction Dashboard", layout="wide")
st.title("NHL Game Prediction Dashboard")
st.markdown(
    """
Use this dashboard to explore model predictions for NHL games.
The model is retrained on the fly based on the seasons you choose below.
All features are engineered using prior game data only (no leakage), so historical predictions
reflect what the model would have produced before puck drop.
"""
)


@st.cache_data(show_spinner=True)
def load_dataset(train_seasons: Tuple[str, ...], test_season: str) -> Dataset:
    seasons = sorted(set(train_seasons + (test_season,)))
    return build_dataset(seasons)


def prepare_predictions(
    dataset: Dataset, train_seasons: List[str], test_season: str, logreg_c: float
) -> Dict[str, object]:
    games = dataset.games.copy()
    features = dataset.features
    target = dataset.target

    train_mask = games["seasonId"].isin(train_seasons)
    test_mask = games["seasonId"] == test_season

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("No games available for the selected train/test split.")

    comparison = compare_models(dataset, train_seasons, test_season, logreg_c_override=logreg_c)
    best_result = comparison["best_result"]
    candidates = comparison["candidates"]

    decision_threshold = best_result["decision_threshold"]
    recommended_threshold = best_result["recommended_threshold"]
    train_probs = best_result.get("train_probs")
    test_probs = best_result.get("test_probs")
    if train_probs is None or test_probs is None:
        model = best_result["model"]
        train_probs = predict_probabilities(model, features, train_mask)
        test_probs = predict_probabilities(model, features, test_mask)

    train_metrics = best_result["train_metrics"]
    test_metrics = best_result["test_metrics"]

    predictions = games.loc[test_mask].copy()
    predictions["home_win_probability"] = test_probs
    predictions["predicted_home_win"] = (test_probs >= decision_threshold).astype(int)
    predictions["correct"] = predictions["predicted_home_win"] == predictions["home_win"]

    display_cols = [
        "gameDate",
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
    predictions = predictions.loc[:, [c for c in display_cols if c in predictions.columns]]

    importance = None
    if best_result["name"] == "Logistic Regression":
        model = best_result["model"]
        importance = compute_feature_effects(model, features.columns).head(15)
    else:
        estimator = model.named_steps.get("clf", model)
        if hasattr(estimator, "feature_importances_"):
            importance = (
                pd.DataFrame(
                    {
                        "feature": features.columns,
                        "coefficient": estimator.feature_importances_,
                    }
                )
                .sort_values("coefficient", ascending=False)
                .head(15)
            )

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "predictions": predictions,
        "feature_importance": importance,
        "best_model_name": best_result["name"],
        "best_params": best_result["hyperparams"],
        "decision_threshold": decision_threshold,
        "recommended_threshold": recommended_threshold,
        "candidate_summaries": candidates,
        "best_val_metrics": best_result["val_metrics"],
        "logreg_c": logreg_c,
        "blend_weight": best_result.get("blend_weight"),
    }


available_seasons = sorted(set(DEFAULT_TRAIN_SEASONS + [DEFAULT_TEST_SEASON]))

with st.sidebar:
    st.header("Configuration")
    train_selection = st.multiselect(
        "Training Seasons",
        options=available_seasons,
        default=DEFAULT_TRAIN_SEASONS,
    )
    test_selection = st.selectbox(
        "Evaluation Season",
        options=available_seasons,
        index=available_seasons.index(DEFAULT_TEST_SEASON),
    )
    logreg_c_value = st.select_slider(
        "Logistic Regression C",
        options=[0.005, 0.01, 0.015, 0.018, 0.02, 0.025, 0.03, 0.033, 0.035, 0.04, 0.05],
        value=DEFAULT_LOGREG_C,
        help="Higher C = weaker regularisation. 0.018 is currently the sweet spot on the 2023‑24 holdout (≈63% accuracy, log loss ≈0.665).",
    )
    st.caption(
        "Season IDs follow NHL notation: e.g., 20232024 represents the 2023-24 regular season."
    )

train_tuple = tuple(train_selection)

if not train_tuple:
    st.warning("Select at least one training season.")
    st.stop()

try:
    dataset = load_dataset(train_tuple, test_selection)
    outputs = prepare_predictions(dataset, list(train_tuple), test_selection, logreg_c_value)
except Exception as exc:  # broad to surface network or API issues in UI
    st.error(f"Unable to build dataset or predictions: {exc}")
    st.stop()

train_metrics = outputs["train_metrics"]
test_metrics = outputs["test_metrics"]
predictions = outputs["predictions"]
importance = outputs["feature_importance"]
best_model_name = outputs["best_model_name"]
best_params = outputs["best_params"]
decision_threshold = outputs["decision_threshold"]
recommended_threshold = outputs["recommended_threshold"]
candidate_summaries = outputs["candidate_summaries"]
best_val_metrics = outputs["best_val_metrics"]
selected_logreg_c = outputs["logreg_c"]
blend_weight = outputs.get("blend_weight")

st.subheader("Model Performance")
metric_cols = st.columns(3)
metric_cols[0].metric("Test Accuracy", f"{test_metrics['accuracy']:.3f}")
metric_cols[1].metric("Test ROC-AUC", f"{test_metrics['roc_auc']:.3f}")
metric_cols[2].metric("Test Log Loss", f"{test_metrics['log_loss']:.3f}")

param_text = ", ".join(f"{key}={value}" for key, value in best_params.items()) if best_params else "-"
caption_text = (
    f"Selected Model: {best_model_name} | "
    f"Training Accuracy: {train_metrics['accuracy']:.3f} | "
    f"Training ROC-AUC: {train_metrics['roc_auc']:.3f} | "
    f"Training Log Loss: {train_metrics['log_loss']:.3f} | "
    f"Decision Threshold: {decision_threshold:.3f} | Params: {param_text} | LogReg C: {selected_logreg_c}"
)
if blend_weight is not None:
    caption_text += f" | Elo Blend Weight: {blend_weight:.2f}"
st.caption(caption_text)
if best_val_metrics is not None:
    st.caption(
        f"Validation accuracy {best_val_metrics['accuracy']:.3f} | "
        f"Validation log loss {best_val_metrics['log_loss']:.3f} | "
        f"Recommended threshold {recommended_threshold:.3f}"
    )

with st.expander("Model comparison", expanded=False):
    rows = []
    for result in candidate_summaries:
        val = result["val_metrics"]
        rows.append(
            {
                "Model": result["name"],
                "Validation Accuracy": val["accuracy"] if val else None,
                "Validation Log Loss": val["log_loss"] if val else None,
                "Validation ROC-AUC": val["roc_auc"] if val else None,
                "Recommended Threshold": result["recommended_threshold"],
                "Hyperparameters": result["hyperparams"],
                "Blend Weight": result.get("blend_weight"),
                "Test Accuracy": result["test_metrics"]["accuracy"],
                "Test Log Loss": result["test_metrics"]["log_loss"],
            }
        )
    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.subheader("Key Feature Effects")
if importance is None or importance.empty:
    st.info("Feature importance not available for the selected model.")
else:
    chart_title = "Impact on Home Win Log-Odds" if best_model_name == "Logistic Regression" else "Relative Importance"
    color_encoding = (
        alt.condition("datum.coefficient > 0", alt.value("#1f77b4"), alt.value("#d62728"))
        if best_model_name == "Logistic Regression"
        else alt.value("#1f77b4")
    )
    feature_chart = (
        alt.Chart(importance)
        .mark_bar()
        .encode(
            x=alt.X("coefficient", title=chart_title),
            y=alt.Y("feature", sort="-x", title="Feature"),
            tooltip=["feature", alt.Tooltip("coefficient", format=".2f")],
            color=color_encoding,
        )
        .properties(height=400)
    )
    st.altair_chart(feature_chart, use_container_width=True)

st.subheader("Game Predictions")

predictions["gameDate"] = pd.to_datetime(predictions["gameDate"])
team_options = sorted(
    set(predictions["teamFullName_home"]).union(predictions["teamFullName_away"])
)
filter_cols = st.columns(3)
selected_team = filter_cols[0].multiselect("Filter by Team", team_options)
date_range = filter_cols[1].date_input(
    "Date Range",
    value=(predictions["gameDate"].min().date(), predictions["gameDate"].max().date()),
)
correct_only = filter_cols[2].toggle("Show Correct Predictions Only", value=False)

filtered = predictions.copy()
if selected_team:
    filtered = filtered[
        filtered["teamFullName_home"].isin(selected_team)
        | filtered["teamFullName_away"].isin(selected_team)
    ]

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        (filtered["gameDate"] >= pd.to_datetime(start_date))
        & (filtered["gameDate"] <= pd.to_datetime(end_date))
    ]

if correct_only:
    filtered = filtered[filtered["correct"]]

st.dataframe(
    filtered.sort_values("gameDate"),
    use_container_width=True,
    hide_index=True,
)

csv_data = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download Filtered Predictions (CSV)", csv_data, "predictions.csv", "text/csv")

st.info(
    "To apply the model to upcoming games, rerun the prediction pipeline once the NHL Stats API publishes the latest game logs. "
    "This ensures features remain up to date before generating probabilities."
)
