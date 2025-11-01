"""High-level dataset preparation for NHL prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict

import numpy as np
import pandas as pd

from .data_ingest import build_game_dataframe, fetch_multi_season_logs, get_team_reference
from .features import ROLL_WINDOWS, engineer_team_features


@dataclass(frozen=True)
class Dataset:
    games: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series


def build_dataset(seasons: Iterable[str]) -> Dataset:
    """Fetch data, engineer features, and prepare modelling matrix."""
    raw_logs = fetch_multi_season_logs(seasons)
    enriched_logs = engineer_team_features(raw_logs)
    games = build_game_dataframe(enriched_logs)
    games = _add_elo_features(games)
    games = _add_head_to_head_features(games)
    games = _add_static_metadata(games)

    rolling_windows = ROLL_WINDOWS
    feature_bases: List[str] = [
        "season_win_pct",
        "season_goal_diff_avg",
        "season_shot_margin",
        "shot_margin_last_game",
        "momentum_win_pct",
        "momentum_goal_diff",
        "momentum_shot_margin",
        "win_streak",
        "wins_prior",
        "losses_prior",
        "ot_losses_prior",
        "reg_ot_wins_prior",
        "wins_reg_prior",
        "wins_so_prior",
        "points_prior",
        "point_pct_prior",
        "points_per_game_prior",
        "pp_pct_prior",
        "pk_pct_prior",
        "special_teams_net_prior",
        "rest_days",
        "is_b2b",
        "games_last_3d",
        "games_last_6d",
    ]

    for window in rolling_windows:
        feature_bases.extend(
            [
                f"rolling_win_pct_{window}",
                f"rolling_goal_diff_{window}",
                f"rolling_pp_pct_{window}",
                f"rolling_pk_pct_{window}",
                f"rolling_faceoff_{window}",
                f"shotsFor_roll_{window}",
                f"shotsAgainst_roll_{window}",
            ]
        )

    feature_columns: List[str] = []
    for base in feature_bases:
        home_col = f"{base}_home"
        away_col = f"{base}_away"
        if home_col in games.columns and away_col in games.columns:
            if np.issubdtype(games[home_col].dtype, np.number) and np.issubdtype(games[away_col].dtype, np.number):
                diff_col = f"{base}_diff"
                games[diff_col] = games[home_col] - games[away_col]
                feature_columns.append(diff_col)

    additional_features = [
        "games_played_prior_home",
        "games_played_prior_away",
        "rest_days_home",
        "rest_days_away",
        "games_last_3d_home",
        "games_last_3d_away",
        "games_last_6d_home",
        "games_last_6d_away",
        "is_b2b_home",
        "is_b2b_away",
        "elo_diff_pre",
        "elo_expectation_home",
        "last_meeting_goal_diff_oriented",
        "last_meeting_home_win",
        "last_meeting_days",
        "same_conference",
        "same_division",
        "season_win_pct_home",
        "season_win_pct_away",
        "win_streak_home",
        "win_streak_away",
        "rolling_win_pct_5_home",
        "rolling_win_pct_5_away",
        "elo_home_pre",
        "elo_away_pre",
    ]
    for feat in additional_features:
        if feat in games.columns:
            feature_columns.append(feat)

    # Situational features that mix home vs away metrics.
    game_columns = set(games.columns)
    if {"rolling_pp_pct_5_home", "rolling_pk_pct_5_away"} <= game_columns:
        games["special_teams_matchup"] = (
            games["rolling_pp_pct_5_home"] - games["rolling_pk_pct_5_away"]
        )
        feature_columns.append("special_teams_matchup")
        games["special_teams_ratio"] = np.divide(
            games["rolling_pp_pct_5_home"],
            games["rolling_pk_pct_5_away"].replace(0, np.nan),
        ).fillna(0.0)
        feature_columns.append("special_teams_ratio")
    if {"rolling_pp_pct_5_away", "rolling_pk_pct_5_home"} <= game_columns:
        games["special_teams_matchup_inverse"] = (
            games["rolling_pk_pct_5_home"] - games["rolling_pp_pct_5_away"]
        )
        feature_columns.append("special_teams_matchup_inverse")
        games["special_teams_ratio_inverse"] = np.divide(
            games["rolling_pp_pct_5_away"],
            games["rolling_pk_pct_5_home"].replace(0, np.nan),
        ).fillna(0.0)
        feature_columns.append("special_teams_ratio_inverse")

    if {"shotsFor_roll_5_home", "shotsAgainst_roll_5_away"} <= game_columns:
        games["shot_pressure_ratio"] = np.divide(
            games["shotsFor_roll_5_home"],
            games["shotsAgainst_roll_5_away"].replace(0, np.nan),
        ).fillna(0.0)
        feature_columns.append("shot_pressure_ratio")
    if {"shotsFor_roll_5_away", "shotsAgainst_roll_5_home"} <= game_columns:
        games["shot_pressure_ratio_inverse"] = np.divide(
            games["shotsFor_roll_5_away"],
            games["shotsAgainst_roll_5_home"].replace(0, np.nan),
        ).fillna(0.0)
        feature_columns.append("shot_pressure_ratio_inverse")

    # Rest-based features.
    def _rest_bucket(days: float) -> str:
        if pd.isna(days):
            return "no_prev"
        if days <= 1:
            return "b2b"
        if days == 2:
            return "one_day"
        if days == 3:
            return "two_days"
        return "three_plus"

    games["rest_bucket_home"] = games["rest_days_home"].apply(_rest_bucket)
    games["rest_bucket_away"] = games["rest_days_away"].apply(_rest_bucket)
    games["rest_diff"] = games["rest_days_home"] - games["rest_days_away"]
    games["home_b2b"] = (games["rest_bucket_home"] == "b2b").astype(int)
    games["away_b2b"] = (games["rest_bucket_away"] == "b2b").astype(int)
    feature_columns.extend(["rest_diff", "home_b2b", "away_b2b"])

    games = pd.get_dummies(
        games,
        columns=["rest_bucket_home", "rest_bucket_away"],
        prefix=["rest_home", "rest_away"],
        dtype=int,
    )
    rest_dummy_cols = [
        col for col in games.columns if col.startswith("rest_home_") or col.startswith("rest_away_")
    ]
    feature_columns.extend(rest_dummy_cols)

    home_team_dummies = pd.get_dummies(games["teamId_home"], prefix="home_team", dtype=int)
    away_team_dummies = pd.get_dummies(games["teamId_away"], prefix="away_team", dtype=int)
    games = pd.concat([games, home_team_dummies, away_team_dummies], axis=1)
    feature_columns.extend(home_team_dummies.columns.tolist())
    feature_columns.extend(away_team_dummies.columns.tolist())

    conf_home = pd.get_dummies(games["home_conference"], prefix="home_conf", dtype=int)
    conf_away = pd.get_dummies(games["away_conference"], prefix="away_conf", dtype=int)
    div_home = pd.get_dummies(games["home_division"], prefix="home_div", dtype=int)
    div_away = pd.get_dummies(games["away_division"], prefix="away_div", dtype=int)
    games = pd.concat([games, conf_home, conf_away, div_home, div_away], axis=1)
    feature_columns.extend(conf_home.columns.tolist())
    feature_columns.extend(conf_away.columns.tolist())
    feature_columns.extend(div_home.columns.tolist())
    feature_columns.extend(div_away.columns.tolist())

    features = games[feature_columns].fillna(0.0)
    target = games["home_win"]
    return Dataset(games=games, features=features, target=target)


def _add_elo_features(
    games: pd.DataFrame,
    base_rating: float = 1500.0,
    k_factor: float = 10.0,
    home_advantage: float = 30.0,
) -> pd.DataFrame:
    """Compute pre-game Elo ratings per team per season."""
    games = games.sort_values("gameDate").copy()
    elo_home: List[float] = []
    elo_away: List[float] = []
    expected_home_probs: List[float] = []

    current_season: str | None = None
    ratings: Dict[int, float] = {}

    for _, row in games.iterrows():
        season = row["seasonId"]
        if season != current_season:
            current_season = season
            ratings = {}

        home_id = int(row["teamId_home"])
        away_id = int(row["teamId_away"])

        home_rating = ratings.get(home_id, base_rating)
        away_rating = ratings.get(away_id, base_rating)

        elo_home.append(home_rating)
        elo_away.append(away_rating)

        expected_home = 1.0 / (1.0 + 10 ** ((away_rating - (home_rating + home_advantage)) / 400))
        expected_home_probs.append(expected_home)

        outcome_home = 1.0 if row["home_win"] == 1 else 0.0
        goal_diff = row["home_score"] - row["away_score"]
        margin = max(abs(goal_diff), 1)
        multiplier = np.log(margin + 1) * (2.2 / ((abs(home_rating - away_rating) * 0.001) + 2.2))
        delta = k_factor * multiplier * (outcome_home - expected_home)

        ratings[home_id] = home_rating + delta
        ratings[away_id] = away_rating - delta

    games["elo_home_pre"] = elo_home
    games["elo_away_pre"] = elo_away
    games["elo_diff_pre"] = games["elo_home_pre"] - games["elo_away_pre"]
    games["elo_expectation_home"] = expected_home_probs
    return games


def _add_head_to_head_features(games: pd.DataFrame) -> pd.DataFrame:
    """Attach info from the previous meeting between the two teams in the same season."""
    games = games.sort_values("gameDate").copy()
    last_matchups: Dict[tuple[str, int, int], tuple[int, int, pd.Timestamp]] = {}
    oriented_diffs: list[float] = []
    home_win_flags: list[int] = []
    days_since_last: list[float] = []

    for _, row in games.iterrows():
        season = row["seasonId"]
        home_id = int(row["teamId_home"])
        away_id = int(row["teamId_away"])
        key = (season, min(home_id, away_id), max(home_id, away_id))
        prev = last_matchups.get(key)

        if prev is None:
            oriented_diffs.append(0.0)
            home_win_flags.append(0)
            days_since_last.append(0.0)
        else:
            prev_home_id, prev_goal_diff, prev_date = prev
            oriented_diff = prev_goal_diff if prev_home_id == home_id else -prev_goal_diff
            oriented_diffs.append(float(oriented_diff))
            home_win_flags.append(1 if oriented_diff > 0 else 0)
            days_since_last.append((row["gameDate"] - prev_date).days if pd.notna(prev_date) else 0.0)

        current_goal_diff = int(row["home_score"] - row["away_score"])
        last_matchups[key] = (home_id, current_goal_diff, row["gameDate"])

    games["last_meeting_goal_diff_oriented"] = oriented_diffs
    games["last_meeting_home_win"] = home_win_flags
    games["last_meeting_days"] = days_since_last
    return games


def _add_static_metadata(games: pd.DataFrame) -> pd.DataFrame:
    """Add conference/division and matchup flags."""
    teams = get_team_reference()[["teamId", "divisionId", "conference"]].copy()
    teams.rename(columns={"divisionId": "division"}, inplace=True)
    games = games.merge(
        teams.rename(columns={"teamId": "teamId_home", "division": "home_division", "conference": "home_conference"}),
        on="teamId_home",
        how="left",
    )
    games = games.merge(
        teams.rename(columns={"teamId": "teamId_away", "division": "away_division", "conference": "away_conference"}),
        on="teamId_away",
        how="left",
    )
    games["same_conference"] = (games["home_conference"] == games["away_conference"]).astype(int)
    games["same_division"] = (games["home_division"] == games["away_division"]).astype(int)
    return games
