"""Feature engineering helpers for NHL game prediction."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ROLL_WINDOWS: Sequence[int] = (3, 5, 10)


def _lagged_rolling(group: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    return group.shift(1).rolling(window, min_periods=min_periods).mean()


def _streak(series: pd.Series) -> pd.Series:
    """Return length of current win streak entering each game."""
    streak = np.zeros(len(series), dtype=int)
    count = 0
    for idx, value in enumerate(series):
        if value == 1:
            count += 1
        else:
            count = 0
        streak[idx] = count
    return pd.Series(streak, index=series.index).shift(1).fillna(0).astype(int)


def engineer_team_features(logs: pd.DataFrame, rolling_windows: Iterable[int] = ROLL_WINDOWS) -> pd.DataFrame:
    """Create lagged features using only information available prior to each game."""
    logs = logs.copy()

    numeric_columns = [
        "goalsFor",
        "goalsAgainst",
        "powerPlayPct",
        "penaltyKillPct",
        "shotsForPerGame",
        "shotsAgainstPerGame",
        "faceoffWinPct",
        "wins",
        "losses",
        "otLosses",
        "regulationAndOtWins",
        "winsInRegulation",
        "winsInShootout",
        "points",
        "pointPct",
        "powerPlayNetPct",
        "penaltyKillNetPct",
    ]
    for column in numeric_columns:
        logs[column] = pd.to_numeric(logs[column], errors="coerce")

    logs["goal_diff"] = logs["goalsFor"] - logs["goalsAgainst"]
    logs["win"] = (logs["goal_diff"] > 0).astype(int)

    logs.sort_values(["teamId", "seasonId", "gameDate", "gameId"], inplace=True)

    group = logs.groupby(["teamId", "seasonId"], sort=False)
    logs["games_played_prior"] = group.cumcount()

    denom = logs["games_played_prior"].replace(0, np.nan)
    logs["season_win_pct"] = group["win"].cumsum().shift(1) / denom
    logs["season_goal_diff_avg"] = group["goal_diff"].cumsum().shift(1) / denom

    # Rest metrics
    logs["rest_days"] = group["gameDate"].diff().dt.days
    logs["is_b2b"] = logs["rest_days"].fillna(10).le(1).astype(int)

    # Rolling statistics
    roll_features: dict[str, pd.Series] = {}
    helper_cols = ["_home_flag", "_away_flag", "_home_win_flag", "_away_win_flag", "_home_goal_flag", "_away_goal_flag"]
    logs.drop(columns=helper_cols, inplace=True, errors="ignore")

    for window in rolling_windows:
        roll_features[f"rolling_win_pct_{window}"] = group["win"].transform(
            lambda s, w=window: _lagged_rolling(s, w)
        )
        roll_features[f"rolling_goal_diff_{window}"] = group["goal_diff"].transform(
            lambda s, w=window: _lagged_rolling(s, w)
        )
        roll_features[f"rolling_pp_pct_{window}"] = group["powerPlayPct"].transform(
            lambda s, w=window: _lagged_rolling(s, w)
        ) / 100.0
        roll_features[f"rolling_pk_pct_{window}"] = group["penaltyKillPct"].transform(
            lambda s, w=window: _lagged_rolling(s, w)
        ) / 100.0
        roll_features[f"rolling_faceoff_{window}"] = group["faceoffWinPct"].transform(
            lambda s, w=window: _lagged_rolling(s, w)
        ) / 100.0
        roll_features[f"shotsFor_roll_{window}"] = group["shotsForPerGame"].transform(
            lambda s, w=window: _lagged_rolling(s, w)
        )
        roll_features[f"shotsAgainst_roll_{window}"] = group["shotsAgainstPerGame"].transform(
            lambda s, w=window: _lagged_rolling(s, w)
        )

    logs = logs.assign(**roll_features)

    # Shot margin trends
    logs["shot_margin"] = logs["shotsForPerGame"] - logs["shotsAgainstPerGame"]
    team_group = logs.groupby(["teamId", "seasonId"], sort=False)
    logs["shot_margin_last_game"] = team_group["shot_margin"].shift(1)
    logs["season_shot_margin"] = team_group["shot_margin"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    logs["momentum_win_pct"] = logs["rolling_win_pct_5"] - logs["season_win_pct"]
    logs["momentum_goal_diff"] = logs["rolling_goal_diff_5"] - logs["season_goal_diff_avg"]
    logs["momentum_shot_margin"] = logs["shot_margin_last_game"] - logs["season_shot_margin"]

    # Team strength prior to the current game
    logs["win_streak"] = group["win"].transform(_streak)
    logs["wins_prior"] = group["wins"].shift(1)
    logs["losses_prior"] = group["losses"].shift(1)
    logs["ot_losses_prior"] = group["otLosses"].shift(1)
    logs["reg_ot_wins_prior"] = group["regulationAndOtWins"].shift(1)
    logs["wins_reg_prior"] = group["winsInRegulation"].shift(1)
    logs["wins_so_prior"] = group["winsInShootout"].shift(1)
    logs["points_prior"] = group["points"].shift(1)
    logs["point_pct_prior"] = group["pointPct"].shift(1)
    logs["points_per_game_prior"] = logs["points_prior"] / (logs["games_played_prior"].replace(0, np.nan) * 2)
    logs["pp_pct_prior"] = group["powerPlayPct"].shift(1) / 100.0
    logs["pk_pct_prior"] = group["penaltyKillPct"].shift(1) / 100.0
    logs["special_teams_net_prior"] = (group["powerPlayNetPct"].shift(1) - group["penaltyKillNetPct"].shift(1)) / 100.0

    logs["_home_flag"] = (logs["homeRoad"] == "H").astype(int)
    logs["_away_flag"] = 1 - logs["_home_flag"]
    logs["_home_win_flag"] = logs["win"] * logs["_home_flag"]
    logs["_away_win_flag"] = logs["win"] * logs["_away_flag"]
    logs["_home_goal_flag"] = logs["goal_diff"] * logs["_home_flag"]
    logs["_away_goal_flag"] = logs["goal_diff"] * logs["_away_flag"]

    group_latest = logs.groupby(["teamId", "seasonId"], sort=False)

    logs["home_games_prior"] = group_latest["_home_flag"].cumsum() - logs["_home_flag"]
    logs["away_games_prior"] = group_latest["_away_flag"].cumsum() - logs["_away_flag"]

    home_wins_prior = group_latest["_home_win_flag"].cumsum() - logs["_home_win_flag"]
    away_wins_prior = group_latest["_away_win_flag"].cumsum() - logs["_away_win_flag"]
    home_goal_diff_prior = group_latest["_home_goal_flag"].cumsum() - logs["_home_goal_flag"]
    away_goal_diff_prior = group_latest["_away_goal_flag"].cumsum() - logs["_away_goal_flag"]

    logs["home_win_pct_prior"] = np.divide(
        home_wins_prior,
        logs["home_games_prior"],
        out=np.zeros(len(logs)),
        where=logs["home_games_prior"] != 0,
    )
    logs["home_goal_diff_avg_prior"] = np.divide(
        home_goal_diff_prior,
        logs["home_games_prior"],
        out=np.zeros(len(logs)),
        where=logs["home_games_prior"] != 0,
    )
    logs["away_win_pct_prior"] = np.divide(
        away_wins_prior,
        logs["away_games_prior"],
        out=np.zeros(len(logs)),
        where=logs["away_games_prior"] != 0,
    )
    logs["away_goal_diff_avg_prior"] = np.divide(
        away_goal_diff_prior,
        logs["away_games_prior"],
        out=np.zeros(len(logs)),
        where=logs["away_games_prior"] != 0,
    )

    # Schedule congestion indicators
    gap = logs.groupby("teamId", sort=False)["rest_days"]
    recent_one_day = gap.transform(lambda s: s.fillna(10).le(1).astype(int))
    logs["games_last_3d"] = (recent_one_day + recent_one_day.shift(1).fillna(0)).clip(0, 3)
    recent_two_day = gap.transform(lambda s: s.fillna(10).le(2).astype(int))
    logs["games_last_6d"] = (
        recent_two_day
        + recent_two_day.shift(1).fillna(0)
        + recent_two_day.shift(2).fillna(0)
        + recent_two_day.shift(3).fillna(0)
    ).clip(0, 4)

    feature_cols = [
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
        "home_win_pct_prior",
        "home_goal_diff_avg_prior",
        "away_win_pct_prior",
        "away_goal_diff_avg_prior",
        "home_games_prior",
        "away_games_prior",
        "rest_days",
        "is_b2b",
        "games_last_3d",
        "games_last_6d",
    ]

    for window in rolling_windows:
        feature_cols.extend(
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

    logs[feature_cols] = logs[feature_cols].fillna(0.0)
    return logs
