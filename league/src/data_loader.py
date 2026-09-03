"""
Data loader for football-data.co.uk format — Premier League focused.

Handles:
  - Loading single CSV files
  - Merging multiple seasons
  - Standard column renaming
  - Basic validation

Data source: https://www.football-data.co.uk/
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# ── Column definitions ───────────────────────────────────────────
COLUMN_MAP = {
    "Div": "division", "Date": "date", "Time": "time",
    "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "home_goals_full", "FTAG": "away_goals_full", "FTR": "result_full",
    "HTHG": "home_goals_half", "HTAG": "away_goals_half", "HTR": "result_half",
    "Referee": "referee",
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_shots_target", "AST": "away_shots_target",
    "HF": "home_fouls", "AF": "away_fouls",
    "HC": "home_corners", "AC": "away_corners",
    "HY": "home_yellow", "AY": "away_yellow",
    "HR": "home_red", "AR": "away_red",
    "B365H": "odds_home", "B365D": "odds_draw", "B365A": "odds_away",
    "BWH": "odds_home_bw", "BWD": "odds_draw_bw", "BWA": "odds_away_bw",
    "PSH": "odds_home_ps", "PSD": "odds_draw_ps", "PSA": "odds_away_ps",
    "MaxH": "odds_home_max", "MaxD": "odds_draw_max", "MaxA": "odds_away_max",
    "AvgH": "odds_home_avg", "AvgD": "odds_draw_avg", "AvgA": "odds_away_avg",
    "B365CH": "odds_home_close", "B365CD": "odds_draw_close", "B365CA": "odds_away_close",
    "BWCH": "odds_home_bw_close", "BWCD": "odds_draw_bw_close", "BWCA": "odds_away_bw_close",
    "PSCH": "odds_home_ps_close", "PSCD": "odds_draw_ps_close", "PSCA": "odds_away_ps_close",
    "MaxCH": "odds_home_max_close", "MaxCD": "odds_draw_max_close", "MaxCA": "odds_away_max_close",
    "AvgCH": "odds_home_avg_close", "AvgCD": "odds_draw_avg_close", "AvgCA": "odds_away_avg_close",
    "B365>2.5": "over_2.5", "B365<2.5": "under_2.5",
    "Avg>2.5": "over_2.5_avg", "Avg<2.5": "under_2.5_avg",
    "B365C>2.5": "over_2.5_close", "B365C<2.5": "under_2.5_close",
    "AvgC>2.5": "over_2.5_avg_close", "AvgC<2.5": "under_2.5_avg_close",
    "AHh": "asian_handicap", "B365AHH": "ah_home", "B365AHA": "ah_away",
}

NUMERIC_COLS = [
    "home_goals_full", "away_goals_full",
    "home_goals_half", "away_goals_half",
    "home_shots", "away_shots",
    "home_shots_target", "away_shots_target",
    "home_fouls", "away_fouls",
    "home_corners", "away_corners",
    "home_yellow", "away_yellow",
    "home_red", "away_red",
]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known columns to canonical names, drop unknown ones."""
    df = df.rename(columns=COLUMN_MAP)
    keep = [c for c in COLUMN_MAP.values() if c in df.columns]
    extra = [c for c in df.columns if c not in COLUMN_MAP and c not in COLUMN_MAP.values()]
    return df[[c for c in keep if c in df.columns]]


def parse_dates(df: pd.DataFrame, dayfirst: bool = True) -> pd.DataFrame:
    """Parse the 'date' column to datetime."""
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], dayfirst=dayfirst, errors="coerce")
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Force known numeric columns to float."""
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_result_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived target columns for prediction.

    Adds:
      - target_result: encoded result (H=2, D=1, A=0)
      - target_over_2.5: 1 if total goals > 2.5 else 0
      - home_win: 1 if home wins, 0 otherwise
      - total_goals: sum of home + away goals
    """
    df = df.copy()
    if "result_full" in df.columns:
        df["target_result"] = df["result_full"].map({"H": 2, "D": 1, "A": 0})
    if "home_goals_full" in df.columns and "away_goals_full" in df.columns:
        df["total_goals"] = df["home_goals_full"] + df["away_goals_full"]
        df["target_over_2.5"] = (df["total_goals"] > 2.5).astype(int)
        df["home_win"] = (df["home_goals_full"] > df["away_goals_full"]).astype(int)
    return df


def load_data(
    path: Path | str,
    standardize: bool = True,
    parse_dates_flag: bool = True,
    add_targets: bool = True,
) -> pd.DataFrame:
    """Load a football-data.co.uk CSV file and process it.

    Args:
        path: Path to CSV file
        standardize: Rename columns to canonical names
        parse_dates_flag: Parse date column
        add_targets: Add derived target columns

    Returns:
        Processed DataFrame
    """
    df = pd.read_csv(path)
    if standardize:
        df = standardize_columns(df)
    df = coerce_numeric(df)
    if parse_dates_flag:
        df = parse_dates(df)
    if add_targets:
        df = add_result_targets(df)
    return df


def load_multiple(
    paths: list[Path | str],
    tags: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load and concatenate multiple CSV files.

    Args:
        paths: List of CSV file paths
        tags: Optional tags to add as a 'source' column

    Returns:
        Concatenated DataFrame
    """
    frames = []
    for i, path in enumerate(paths):
        df = load_data(path)
        if tags:
            df["source"] = tags[i] if i < len(tags) else str(path)
        else:
            df["source"] = str(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
