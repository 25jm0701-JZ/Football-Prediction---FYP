"""
Player-level feature engineering for World Cup prediction.

Aggregates per-tournament player statistics to team-level quality features,
which are then used as differential match features (Home − Away) in the
prediction model.

Available across ALL years (2014–2026) with 100 % coverage:
  goals_per_90_overall, assists_per_90_overall,
  goals_involved_per_90_overall (goals + assists), conceded_per_90_overall,
  minutes_played_overall, age, appearances_overall, clean_sheets_overall

Newer years (2022–2026) additionally supply average rating, xG, tackles,
interceptions, clearances, etc., but those cannot be used for leave-one-year-out
backtesting since 2014 and 2018 lack them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .team_names import canonical_team

# ---------------------------------------------------------------------------
# Position groups
# ---------------------------------------------------------------------------
FORWARD = {"Forward"}
MIDFIELDER = {"Midfielder"}
DEFENDER = {"Defender"}
GOALKEEPER = {"Goalkeeper"}
OUTFIELD = {"Forward", "Midfielder", "Defender"}
NON_PLAYER = {"Coach"}

# ---------------------------------------------------------------------------
# Player-derived features added to the model.
# Each is a "home − away" differential computed from team-level aggregates.
# ---------------------------------------------------------------------------
PLAYER_FEATURES = [
    "fwd_goals_involved_per_90_diff",
    "mid_assists_per_90_diff",
    "def_conceded_per_90_diff",
    "squad_avg_age_diff",
]

# Minutes-weighted variants (better handles rotation / fringe players)
PLAYER_FEATURES_WEIGHTED = [
    "fwd_goals_involved_per_90_w_diff",
    "mid_assists_per_90_w_diff",
    "def_conceded_per_90_w_diff",
    "squad_avg_age_w_diff",
]

# Number of player features (used for slicing in the model)
N_PLAYER_FEATURES = len(PLAYER_FEATURES)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def player_feature_names() -> list[str]:
    """Return the list of player-derived feature column names."""
    return list(PLAYER_FEATURES)


def player_feature_names_weighted() -> list[str]:
    """Return the list of minutes-weighted player-derived feature names."""
    return list(PLAYER_FEATURES_WEIGHTED)


# ---------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------


def _load_single(path: Path) -> pd.DataFrame:
    """Load a single player CSV and standardise team names."""
    df = pd.read_csv(path)
    df["nationality"] = df["nationality"].map(canonical_team)
    return df


def load_team_features(year_to_path: dict[int, Path]) -> pd.DataFrame:
    """
    Load player data from multiple tournaments and aggregate to team level.

    Parameters
    ----------
    year_to_path : dict[int, Path]
        Mapping of tournament year → CSV path, e.g.
        ``{2014: p, 2018: p, 2022: p, 2026: p}``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Year, nationality, {base_feature_cols}``.
        One row per (year, team) combination.
    """
    frames: list[pd.DataFrame] = []
    for year, path in year_to_path.items():
        df = _load_single(path)
        df["_year"] = year
        frames.append(df)

    all_players = pd.concat(frames, ignore_index=True)

    # Drop non-player personnel (coaches etc.)
    all_players = all_players[~all_players["position"].isin(NON_PLAYER)].copy()

    # Filter out fringe players with very few minutes (< 45) to avoid
    # extreme per-90 rates from small-sample outliers (e.g. 2 goals in
    # 31 min → 5.81 goals_per_90).
    all_players = all_players[all_players["minutes_played_overall"] >= 45].copy()

    # -- Position-group aggregations ---------------------------------------
    fwd = all_players[all_players["position"].isin(FORWARD)]
    fwd_agg = (
        fwd.groupby(["_year", "nationality"])
        .agg(fwd_goals_involved_per_90=("goals_involved_per_90_overall", "mean"),
             fwd_count=("goals_involved_per_90_overall", "count"))
        .reset_index()
    )

    mid = all_players[all_players["position"].isin(MIDFIELDER)]
    mid_agg = (
        mid.groupby(["_year", "nationality"])
        .agg(mid_assists_per_90=("assists_per_90_overall", "mean"),
             mid_count=("assists_per_90_overall", "count"))
        .reset_index()
    )

    deff = all_players[all_players["position"].isin(DEFENDER)]
    def_agg = (
        deff.groupby(["_year", "nationality"])
        .agg(def_conceded_per_90=("conceded_per_90_overall", "mean"),
             def_count=("conceded_per_90_overall", "count"))
        .reset_index()
    )

    # -- Team-level overall stats -------------------------------------------
    team_agg = (
        all_players.groupby(["_year", "nationality"])
        .agg(team_goals_per_90=("goals_per_90_overall", "mean"),
             squad_avg_age=("age", "mean"),
             squad_avg_minutes=("minutes_played_overall", "mean"),
             squad_size=("minutes_played_overall", "count"))
        .reset_index()
    )

    # Merge
    result = team_agg \
        .merge(fwd_agg, on=["_year", "nationality"], how="left") \
        .merge(mid_agg, on=["_year", "nationality"], how="left") \
        .merge(def_agg, on=["_year", "nationality"], how="left")

    result = result.rename(columns={"_year": "Year"})
    return result


def load_team_features_weighted(year_to_path: dict[int, Path]) -> pd.DataFrame:
    """
    Minutes-weighted version of :func:`load_team_features`.

    Instead of ``mean(per_90_rate)`` across players in a position group, uses
    ``sum(stat) / sum(minutes) * 90``, which is the team's actual per-90 rate
    for that position group.  This reduces noise from fringe players who
    happened to score in limited minutes.
    """
    frames: list[pd.DataFrame] = []
    for year, path in year_to_path.items():
        df = _load_single(path)
        df["_year"] = year
        frames.append(df)

    all_players = pd.concat(frames, ignore_index=True)
    all_players = all_players[~all_players["position"].isin(NON_PLAYER)].copy()
    all_players = all_players[all_players["minutes_played_overall"] >= 45].copy()

    # Helper: weighted mean of a per-90 column using minutes as weights.
    # Equivalent to sum(raw_stat) / sum(minutes) * 90.
    def _weighted_per_90(group: pd.DataFrame, col: str) -> float:
        num = (group[col] * group["minutes_played_overall"]).sum()
        den = group["minutes_played_overall"].sum()
        return float(num / den) if den > 0 else 0.0

    # --- Position-group weighted aggregations -------------------------------
    fwd = all_players[all_players["position"].isin(FORWARD)]
    fwd_agg = (
        fwd.groupby(["_year", "nationality"])
        .apply(lambda g: pd.Series({
            "fwd_goals_involved_per_90_w": _weighted_per_90(g, "goals_involved_per_90_overall"),
            "fwd_count_w": int(g["minutes_played_overall"].sum() > 0),
        }), include_groups=False)
        .reset_index()
    )

    mid = all_players[all_players["position"].isin(MIDFIELDER)]
    mid_agg = (
        mid.groupby(["_year", "nationality"])
        .apply(lambda g: pd.Series({
            "mid_assists_per_90_w": _weighted_per_90(g, "assists_per_90_overall"),
            "mid_count_w": int(g["minutes_played_overall"].sum() > 0),
        }), include_groups=False)
        .reset_index()
    )

    deff = all_players[all_players["position"].isin(DEFENDER)]
    def_agg = (
        deff.groupby(["_year", "nationality"])
        .apply(lambda g: pd.Series({
            "def_conceded_per_90_w": _weighted_per_90(g, "conceded_per_90_overall"),
            "def_count_w": int(g["minutes_played_overall"].sum() > 0),
        }), include_groups=False)
        .reset_index()
    )

    # Team-level overall stats (minutes-weighted where sensible)
    team_agg = (
        all_players.groupby(["_year", "nationality"])
        .apply(lambda g: pd.Series({
            "team_goals_per_90_w": _weighted_per_90(g, "goals_per_90_overall"),
            "squad_avg_age_w": float(np.average(g["age"], weights=g["minutes_played_overall"])),
            "squad_avg_minutes_w": float(g["minutes_played_overall"].mean()),
            "squad_size_w": int(len(g)),
        }), include_groups=False)
        .reset_index()
    )

    result = team_agg \
        .merge(fwd_agg, on=["_year", "nationality"], how="left") \
        .merge(mid_agg, on=["_year", "nationality"], how="left") \
        .merge(def_agg, on=["_year", "nationality"], how="left")

    result = result.rename(columns={"_year": "Year"})
    return result


# ---------------------------------------------------------------------------
# Attach features to match / fixture DataFrames
# ---------------------------------------------------------------------------


def add_player_features_to_matches(
    matches: pd.DataFrame,
    team_features: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add player-derived differential columns to a match DataFrame.

    For each match::
        feature = home_team_value − away_team_value

    Parameters
    ----------
    matches : pd.DataFrame
        Must contain columns ``Year``, ``Home``, ``Away``.
    team_features : pd.DataFrame
        Output of :func:`load_team_features` or :func:`load_team_features_weighted`.
    feature_names : list[str] | None
        Which features (from ``PLAYER_FEATURES`` / ``PLAYER_FEATURES_WEIGHTED``)
        to compute.  Default ``PLAYER_FEATURES``.

    Missing values are filled with 0 (no advantage).
    """
    output = matches.copy()
    if feature_names is None:
        feature_names = PLAYER_FEATURES

    # Build a quick lookup: (Year, nationality) → feature values
    tf = team_features.set_index(["Year", "nationality"])

    # Align home / away via MultiIndex reindex (vectorised)
    home_idx = pd.MultiIndex.from_arrays(
        [output["Year"], output["Home"]], names=["Year", "nationality"]
    )
    away_idx = pd.MultiIndex.from_arrays(
        [output["Year"], output["Away"]], names=["Year", "nationality"]
    )

    home_vals = tf.reindex(home_idx)
    away_vals = tf.reindex(away_idx)

    for pf in feature_names:
        base = pf.replace("_diff", "")
        if base not in tf.columns:
            output[pf] = 0.0
            continue
        h = home_vals[base].to_numpy()
        a = away_vals[base].to_numpy()
        diff = np.where(pd.notna(h) & pd.notna(a), h - a, 0.0)
        output[pf] = diff

    return output
