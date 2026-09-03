"""
Feature engineering for football match prediction.

Implements rolling-window features inspired by Atta Mills et al. (2024):
  - Team form / recent performance
  - Attack & defense strength
  - Goal metrics
  - Shot & efficiency metrics
  - Betting odds implied probabilities
  - Head-to-head features
  - FootyStats extra features (PPG, xG, possession) — PL only
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# ── Rolling window defaults ──────────────────────────────────────
DEFAULT_WINDOW = 5       # short-term form
MEDIUM_WINDOW = 10       # medium-term
LONG_WINDOW = 20         # long-term (used in Paper 1)

# ── FootyStats integration ────────────────────────────────────────
# Path to downloaded footystats data (relative to project root)
FOOTYSTATS_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "footystats" / "raw"

# Team name mapping: football-data.co.uk → footystats
FOOTYSTATS_TEAM_MAP = {
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa",
    "Bournemouth": "AFC Bournemouth", "Brentford": "Brentford",
    "Brighton": "Brighton & Hove Albion", "Burnley": "Burnley",
    "Chelsea": "Chelsea", "Crystal Palace": "Crystal Palace",
    "Everton": "Everton", "Fulham": "Fulham",
    "Ipswich": "Ipswich Town", "Leeds": "Leeds United",
    "Leicester": "Leicester City", "Liverpool": "Liverpool",
    "Luton": "Luton Town", "Man City": "Manchester City",
    "Man United": "Manchester United", "Newcastle": "Newcastle United",
    "Norwich": "Norwich City", "Nott'm Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield United", "Southampton": "Southampton",
    "Sunderland": "Sunderland", "Tottenham": "Tottenham Hotspur",
    "Watford": "Watford", "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United", "Wolves": "Wolverhampton Wanderers",
}

# Season ID mapping: season label → footystats season_id
FOOTYSTATS_SEASONS = {
    "2019-20": 2012, "2020-21": 4759, "2021-22": 6135,
    "2022-23": 7704, "2023-24": 9660, "2024-25": 12325,
}

# Cache for loaded footystats data (load once per session)
_footystats_cache: dict[int, pd.DataFrame] = {}


def _load_footystats_season(season_label: str) -> pd.DataFrame | None:
    """Load one season's footystats match data from cached JSON."""
    sid = FOOTYSTATS_SEASONS.get(season_label)
    if sid is None:
        return None
    if sid in _footystats_cache:
        return _footystats_cache[sid]

    path = FOOTYSTATS_RAW_DIR / f"matches_{sid}.json"
    if not path.exists():
        return None

    matches = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(matches)
    df["date"] = pd.to_datetime(df["date_unix"], unit="s")
    df["date_key"] = df["date"].dt.date
    _footystats_cache[sid] = df
    return df


def _footystats_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge footystats extra features into the match DataFrame.

    Adds for PL matches (matched by team+date):
      - home_ppg, away_ppg, ppg_diff       — cross-season points per game
      - pre_match_xg_home/away/diff         — pre-match expected goals
      - home_possession, away_possession,
        possession_diff                     — match possession

    Unmatched matches (non-PL or missing) get NaN — handle downstream.
    """
    features = pd.DataFrame(index=df.index)
    features["home_ppg"] = np.nan
    features["away_ppg"] = np.nan
    features["ppg_diff"] = np.nan
    features["pre_match_xg_home"] = np.nan
    features["pre_match_xg_away"] = np.nan
    features["pre_match_xg_diff"] = np.nan
    features["home_possession"] = np.nan
    features["away_possession"] = np.nan
    features["possession_diff"] = np.nan

    # Determine season from the data (use first available season label)
    # If df has a 'season' column, use it directly
    if "season" not in df.columns:
        return features  # can't merge without season info

    # Map team names to footystats format
    df["_home_ft"] = df["home_team"].map(FOOTYSTATS_TEAM_MAP)
    df["_away_ft"] = df["away_team"].map(FOOTYSTATS_TEAM_MAP)
    df["_date_key"] = df["date"].dt.date

    for season_label in df["season"].unique():
        ft_df = _load_footystats_season(season_label)
        if ft_df is None:
            continue

        mask = df["season"] == season_label
        idx = df.index[mask]

        # Prepare footystats extra columns
        ft_extra = ft_df[["date_key", "home_name", "away_name",
                          "home_ppg", "away_ppg",
                          "team_a_xg_prematch", "team_b_xg_prematch",
                          "team_a_possession", "team_b_possession"]].copy()
        ft_extra.columns = ["_dk", "_hm", "_aw",
                            "pp_h", "pp_a", "xg_h", "xg_a",
                            "pos_h", "pos_a"]

        # Merge on (date_key, home_ft, away_ft)
        merge_df = df.loc[idx, ["_date_key", "_home_ft", "_away_ft"]].copy()
        merge_df = merge_df.merge(
            ft_extra, left_on=["_date_key", "_home_ft", "_away_ft"],
            right_on=["_dk", "_hm", "_aw"], how="left", suffixes=("", "_y")
        )

        # Write back
        features.loc[idx, "home_ppg"] = merge_df["pp_h"].values
        features.loc[idx, "away_ppg"] = merge_df["pp_a"].values
        features.loc[idx, "pre_match_xg_home"] = merge_df["xg_h"].values
        features.loc[idx, "pre_match_xg_away"] = merge_df["xg_a"].values
        features.loc[idx, "home_possession"] = merge_df["pos_h"].values
        features.loc[idx, "away_possession"] = merge_df["pos_a"].values

    # Derived differentials
    features["ppg_diff"] = features["home_ppg"] - features["away_ppg"]
    features["pre_match_xg_diff"] = features["pre_match_xg_home"] - features["pre_match_xg_away"]
    features["possession_diff"] = features["home_possession"] - features["away_possession"]

    # Log match rate
    n_total = len(features)
    n_matched = features["home_ppg"].notna().sum()
    if n_matched < n_total:
        print(f"  [FootyStats] Matched {n_matched}/{n_total} ({n_matched/n_total*100:.0f}%) — "
              f"{n_total - n_matched} unmatched (non-PL or missing season)")
    else:
        print(f"  [FootyStats] All {n_total} matches enriched with PPG/xG/possession")

    # Clean up temporary columns
    df.drop(columns=["_home_ft", "_away_ft", "_date_key"], inplace=True, errors="ignore")

    return features


def _team_rolling_stats(
    df: pd.DataFrame,
    team_col: str,
    goal_for_col: str,
    goal_against_col: str,
    is_home: bool,
    window: int,
) -> pd.DataFrame:
    """Compute rolling stats for a team (home or away perspective).

    Args:
        df: Full match DataFrame, sorted by date
        team_col: Column name for the team (home_team or away_team)
        goal_for_col: Goals scored by this team
        goal_against_col: Goals conceded by this team
        is_home: True if this is home perspective
        window: Rolling window size

    Returns:
        DataFrame of features indexed same as input
    """
    prefix = "home_" if is_home else "away_"

    # Build per-team chronological stats
    team_games = df[[team_col, goal_for_col, goal_against_col, "date"]].copy()
    team_games = team_games.rename(
        columns={team_col: "team", goal_for_col: "gf", goal_against_col: "ga"}
    )

    # Sort by team then date
    team_games = team_games.sort_values(["team", "date"])

    # Rolling features per team
    features = pd.DataFrame(index=team_games.index)

    # Shift by 1 so we don't leak the current match
    shifted_gf = team_games.groupby("team")["gf"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    features[f"{prefix}avg_goals_for_{window}"] = shifted_gf

    shifted_ga = team_games.groupby("team")["ga"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    features[f"{prefix}avg_goals_against_{window}"] = shifted_ga

    # Goal differential
    features[f"{prefix}avg_goal_diff_{window}"] = shifted_gf - shifted_ga

    # Recent form: points from last `window` matches (win=3, draw=1, loss=0)
    def _points_from_row(row):
        if row["gf"] > row["ga"]:
            return 3
        elif row["gf"] == row["ga"]:
            return 1
        return 0

    team_games["points"] = team_games.apply(_points_from_row, axis=1)
    shifted_points = team_games.groupby("team")["points"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    features[f"{prefix}avg_points_{window}"] = shifted_points

    # Win / draw / loss rates
    def _win_rate(series):
        return (series == 3).astype(float).rolling(window, min_periods=1).mean()
    def _draw_rate(series):
        return (series == 1).astype(float).rolling(window, min_periods=1).mean()
    def _loss_rate(series):
        return (series == 0).astype(float).rolling(window, min_periods=1).mean()

    shifted_pts_sorted = team_games.sort_index().groupby("team")["points"].transform(
        lambda x: x.shift(1)
    )
    # Recompute on the shifted values
    team_games["shifted_points"] = shifted_pts_sorted
    temp = team_games.groupby("team")["shifted_points"]

    features[f"{prefix}win_rate_{window}"] = temp.transform(
        lambda x: (x == 3).rolling(window, min_periods=1).mean()
    )
    features[f"{prefix}draw_rate_{window}"] = temp.transform(
        lambda x: (x == 1).rolling(window, min_periods=1).mean()
    )
    features[f"{prefix}loss_rate_{window}"] = temp.transform(
        lambda x: (x == 0).rolling(window, min_periods=1).mean()
    )
    return features


def _shot_features(
    df: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Compute rolling shot-related features for both teams."""
    features = pd.DataFrame(index=df.index)

    for prefix, shot_col, sht_col, gf_col in [
        ("home_", "home_shots", "home_shots_target", "home_goals_full"),
        ("away_", "away_shots", "away_shots_target", "away_goals_full"),
    ]:
        team_col = f"{prefix}team"
        # Average shots per game
        shifted_shots = df.groupby(team_col)[shot_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        features[f"{prefix}avg_shots_{window}"] = shifted_shots

        # Shots on target
        shifted_sot = df.groupby(team_col)[sht_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        features[f"{prefix}avg_sot_{window}"] = shifted_sot

        # Shot accuracy (SOT / Shots)
        acc = shifted_sot / shifted_shots.replace(0, np.nan)
        features[f"{prefix}shot_accuracy_{window}"] = acc

        # Conversion rate (goals / shots)
        shifted_gf = df.groupby(team_col)[gf_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).sum()
        )
        shifted_shots_sum = df.groupby(team_col)[shot_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).sum()
        )
        conv = shifted_gf / shifted_shots_sum.replace(0, np.nan)
        features[f"{prefix}conversion_rate_{window}"] = conv

    return features


def _odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute implied probabilities from betting odds.

    Odds -> implied probability: 1 / odds.
    Also computes overround (margin) adjustment.
    """
    features = pd.DataFrame(index=df.index)

    # Opening odds implied probabilities
    if all(c in df.columns for c in ["odds_home", "odds_draw", "odds_away"]):
        impl_h = 1.0 / df["odds_home"]
        impl_d = 1.0 / df["odds_draw"]
        impl_a = 1.0 / df["odds_away"]
        total = impl_h + impl_d + impl_a
        features["impl_home_prob"] = impl_h / total
        features["impl_draw_prob"] = impl_d / total
        features["impl_away_prob"] = impl_a / total
        features["overround"] = total - 1.0

    # Closing odds (market final)
    if all(c in df.columns for c in ["odds_home_close", "odds_draw_close", "odds_away_close"]):
        impl_h_c = 1.0 / df["odds_home_close"]
        impl_d_c = 1.0 / df["odds_draw_close"]
        impl_a_c = 1.0 / df["odds_away_close"]
        total_c = impl_h_c + impl_d_c + impl_a_c
        features["impl_home_prob_close"] = impl_h_c / total_c
        features["impl_draw_prob_close"] = impl_d_c / total_c
        features["impl_away_prob_close"] = impl_a_c / total_c

    # Over/Under 2.5 implied
    if all(c in df.columns for c in ["over_2.5", "under_2.5"]):
        impl_over = 1.0 / df["over_2.5"]
        impl_under = 1.0 / df["under_2.5"]
        total_ou = impl_over + impl_under
        features["impl_over_prob"] = impl_over / total_ou
        features["impl_under_prob"] = impl_under / total_ou

    return features


def _h2h_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute head-to-head features for each team pair.

    For each match, look at previous encounters between the same two teams.
    """
    features = pd.DataFrame(index=df.index)
    features["h2h_home_wins"] = 0
    features["h2h_away_wins"] = 0
    features["h2h_draws"] = 0
    features["h2h_home_avg_goals"] = 0.0
    features["h2h_away_avg_goals"] = 0.0

    # Group by team pair
    pair_key = df.apply(
        lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1
    )

    for pair in pair_key.unique():
        mask = pair_key == pair
        indices = df[mask].sort_values("date").index

        home_wins = []
        away_wins = []
        draws = []
        home_goals = []
        away_goals = []

        for idx in indices:
            features.at[idx, "h2h_home_wins"] = sum(home_wins[-window:])
            features.at[idx, "h2h_away_wins"] = sum(away_wins[-window:])
            features.at[idx, "h2h_draws"] = sum(draws[-window:])
            features.at[idx, "h2h_home_avg_goals"] = (
                np.mean(home_goals[-window:]) if home_goals else 0
            )
            features.at[idx, "h2h_away_avg_goals"] = (
                np.mean(away_goals[-window:]) if away_goals else 0
            )

            r = df.loc[idx]
            if r["home_goals_full"] > r["away_goals_full"]:
                if r["home_team"] == pair[0]:
                    home_wins.append(1)
                    away_wins.append(0)
                else:
                    home_wins.append(0)
                    away_wins.append(1)
                draws.append(0)
            elif r["home_goals_full"] == r["away_goals_full"]:
                draws.append(1)
                home_wins.append(0)
                away_wins.append(0)
            else:
                if r["home_team"] == pair[0]:
                    home_wins.append(0)
                    away_wins.append(1)
                else:
                    home_wins.append(1)
                    away_wins.append(0)
                draws.append(0)

            if r["home_team"] == pair[0]:
                home_goals.append(r["home_goals_full"])
                away_goals.append(r["away_goals_full"])
            else:
                home_goals.append(r["away_goals_full"])
                away_goals.append(r["home_goals_full"])

    return features


# ── League-aware features (Stage 2-3) ───────────────────────────

def _get_rolling_feature_cols(df: pd.DataFrame) -> list[str]:
    """Identify all rolling-window feature columns.

    These are the columns that get league-normalized:
      avg_goals_for/against, avg_goal_diff, avg_points,
      win/draw/loss rate, avg_shots, avg_sot, shot_accuracy, conversion_rate
    """
    patterns = [
        "avg_goals_for_", "avg_goals_against_", "avg_goal_diff_",
        "avg_points_", "win_rate_", "draw_rate_", "loss_rate_",
        "avg_shots_", "avg_sot_", "shot_accuracy_", "conversion_rate_",
    ]
    cols = []
    for c in df.columns:
        if not (c.startswith("home_") or c.startswith("away_")):
            continue
        for p in patterns:
            if p in c:
                cols.append(c)
                break
    return cols


def _league_onehot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create one-hot encoded league features from the 'division' column.

    Stage-2: tells the model which league each match belongs to.
    """
    features = pd.DataFrame(index=df.index)
    if "division" not in df.columns:
        return features
    divisions = sorted(df["division"].dropna().unique())
    for div in divisions:
        features[f"league_{div}"] = (df["division"] == div).astype(float)
    return features


def _league_normalize_features(
    df: pd.DataFrame,
    rolling_cols: list[str],
) -> pd.DataFrame:
    """Create league-relative features by subtracting per-division means.

    Stage-3: instead of 'this team scores 2.0 goals/game' which means
    different things in different leagues, produce
    'this team scores 0.3 goals/game ABOVE league average'.
    """
    features = pd.DataFrame(index=df.index)
    if "division" not in df.columns:
        return features

    for col in rolling_cols:
        league_mean = df.groupby("division")[col].transform("mean")
        # home_avg_goals_for_5 → rel_home_avg_goals_for_5
        new_col = f"rel_{col}"
        features[new_col] = df[col] - league_mean

    return features


# ── Public API ───────────────────────────────────────────────────

def _pi_rating_features(
    df: pd.DataFrame, pi: float | None = None
) -> pd.DataFrame:
    """Compute pi-rating features for each match.

    Runs the PiRating model chronologically over all matches,
    generating pre-match team strength estimates.

    Features added:
      home_pi_mu, away_pi_mu           — rating (mean)
      home_pi_sigma, away_pi_sigma     — uncertainty
      pi_mu_diff                        — rating difference
      pi_home_win_prob, pi_draw_prob,
        pi_away_win_prob                — model's expected probabilities
    """
    from src.ratings import PiRating

    pr = PiRating(pi=pi or 0.015)
    pr.fit(df)
    return pr.add_features(df)


def build_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    include_shots: bool = True,
    include_odds: bool = False,
    include_h2h: bool = True,
    league_features: bool = False,    # Stage-2: add league one-hot
    league_normalize: bool = False,   # Stage-3: add league-relative features
    include_pi_rating: bool = False,  # Paper 1: pi-rating dynamic strength
    pi_rating_pi: float | None = None,  # pi parameter (None=default 0.015)
    include_footystats: bool = False, # FootyStats extra features (PPG, xG, poss)
    footystats_season_col: str = "season",  # column containing season labels
) -> pd.DataFrame:
    """Build all features for match prediction.

    Args:
        df: DataFrame from data_loader.load_data(), must be sorted by date
        windows: List of rolling window sizes (default: [5, 10, 20])
        include_shots: Include shot-based features
        include_odds: Include betting odds features
        include_h2h: Include head-to-head features
        include_footystats: Merge FootyStats extras (PPG, xG, possession)
                           Only works for PL matches with available data.
        footystats_season_col: Column with season labels like "2020-21".
                              Required when include_footystats=True.

    Returns:
        DataFrame with additional feature columns
    """
    if windows is None:
        windows = [DEFAULT_WINDOW, MEDIUM_WINDOW, LONG_WINDOW]

    result = df.copy()
    result = result.sort_values("date").reset_index(drop=True)

    all_features = []

    # Rolling team stats (for each window)
    for w in windows:
        # Home team features
        home_feats = _team_rolling_stats(
            result, "home_team", "home_goals_full", "away_goals_full",
            is_home=True, window=w,
        )
        all_features.append(home_feats)

        # Away team features
        away_feats = _team_rolling_stats(
            result, "away_team", "away_goals_full", "home_goals_full",
            is_home=False, window=w,
        )
        all_features.append(away_feats)

    # Shot features (optional)
    if include_shots:
        for w in windows[:2]:  # only short & medium for shots
            all_features.append(_shot_features(result, window=w))

    # Odds features
    if include_odds:
        all_features.append(_odds_features(result))

    # H2H features (optional, can be slow for large datasets)
    if include_h2h:
        all_features.append(_h2h_features(result))

    # League-aware features (Stage 2-3)
    if league_features:
        all_features.append(_league_onehot_features(result))
    if league_normalize:
        rolling_cols = _get_rolling_feature_cols(result)
        all_features.append(_league_normalize_features(result, rolling_cols))

    # Pi-rating dynamic strength features (Paper 1)
    if include_pi_rating:
        pi_feats = _pi_rating_features(result, pi=pi_rating_pi)
        pi_cols = [c for c in pi_feats.columns if c.startswith(("home_pi_", "away_pi_", "pi_"))]
        all_features.append(pi_feats[pi_cols])

    # FootyStats extra features (PPG, xG, possession) — PL only
    if include_footystats:
        if footystats_season_col in result.columns:
            ft_feats = _footystats_extra_features(result)
            all_features.append(ft_feats)
        else:
            import warnings
            warnings.warn(
                f"include_footystats=True but column '{footystats_season_col}' "
                f"not found. Add a 'season' column (e.g. '2020-21')."
            )

    # Concatenate all features
    for feat_df in all_features:
        for col in feat_df.columns:
            result[col] = feat_df[col]

    return result


# Current-match raw stat columns (home_shots, away_corners, etc. from the CSV).
# These are the match's OWN statistics — using them as features leaks the very
# result being predicted. The shift(1)-lagged rolling shot/form features already
# capture this information legitimately (see _shot_features / _team_rolling_stats).
LEAKED_MATCH_STATS = {
    "home_shots", "away_shots",
    "home_shots_target", "away_shots_target",
    "home_fouls", "away_fouls",
    "home_corners", "away_corners",
    "home_yellow", "away_yellow",
    "home_red", "away_red",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of feature column names (excludes ID/label columns)."""
    exclude = {
        "division", "date", "time", "home_team", "away_team",
        "home_goals_full", "away_goals_full", "result_full",
        "home_goals_half", "away_goals_half", "result_half",
        "referee", "total_goals",
        "target_result", "target_over_2.5", "home_win",
        "source", "season", "asian_handicap",
    }
    exclude |= LEAKED_MATCH_STATS
    # Also exclude raw odds columns (we use implied probabilities)
    odds_patterns = {
        "odds_", "over_2.5", "under_2.5", "asian_handicap",
        "ah_home", "ah_away",
    }

    features = []
    for col in df.columns:
        if col in exclude:
            continue
        if any(col.startswith(p) for p in ["odds_", "over_", "under_", "ah_"]):
            if col not in [
                "impl_home_prob", "impl_draw_prob", "impl_away_prob",
                "impl_home_prob_close", "impl_draw_prob_close", "impl_away_prob_close",
                "impl_over_prob", "impl_under_prob", "overround",
            ]:
                continue
        features.append(col)

    return features
