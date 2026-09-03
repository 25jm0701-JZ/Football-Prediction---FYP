#!/usr/bin/env python3
"""
FootyStats Backtest: Premier League 2018-2025
=============================================

Complete backtesting pipeline:
  1. Load 7 seasons of PL match + player data from footystats API
  2. Rolling team stats + squad features (odds are NEVER model features)
  3. Logistic Regression → match result prediction
  4. Model probabilities vs Bet365 CLOSING market implied probabilities (B365C)
  5. Statistical tests: bootstrap CI, Diebold-Mariano

Requires: data/footystats/raw/*.json (from download_footystats_data.py)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

# Add project root
PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

# ── Paths ──────────────────────────────────────────────────────────
RAW_DIR = PROJ_ROOT / "data" / "footystats" / "raw"
OUTPUT_DIR = PROJ_ROOT / "outputs" / "footystats_backtest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEASON_IDS = {
    1625: "2018-19", 2012: "2019-20", 4759: "2020-21",
    6135: "2021-22", 7704: "2022-23", 9660: "2023-24", 12325: "2024-25",
}

SEASON_ORDER = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]


# ═══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════

def load_matches() -> pd.DataFrame:
    """Load all matches from raw JSON files into a single DataFrame."""
    frames = []
    for sid, label in SEASON_IDS.items():
        path = RAW_DIR / f"matches_{sid}.json"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        matches = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(matches)

        # Parse date
        df["date"] = pd.to_datetime(df["date_unix"], unit="s")
        df["season"] = label
        df["season_id"] = sid

        # Add derived targets
        df["target_result"] = df.apply(
            lambda r: 2 if r["homeGoalCount"] > r["awayGoalCount"]
            else 1 if r["homeGoalCount"] == r["awayGoalCount"]
            else 0, axis=1
        )
        df["target_home_win"] = (df["homeGoalCount"] > df["awayGoalCount"]).astype(int)
        df["total_goals"] = df["homeGoalCount"] + df["awayGoalCount"]
        df["target_over_2.5"] = (df["total_goals"] > 2.5).astype(int)

        # Map team_a / team_b → home / away
        # In footystats data: team_a = home, team_b = away
        # (confirmed: possession_home=46, possession_away=54 in first match)
        df["home_team"] = df["home_name"]
        df["away_team"] = df["away_name"]
        df["home_goals"] = df["homeGoalCount"]
        df["away_goals"] = df["awayGoalCount"]

        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["season_id", "game_week", "date"]).reset_index(drop=True)
    print(f"  Total matches: {len(result)}")
    print(f"  Seasons: {result['season'].unique().tolist()}")
    return result


def load_players() -> pd.DataFrame:
    """Load all player data from raw JSON files."""
    frames = []
    for sid, label in SEASON_IDS.items():
        path = RAW_DIR / f"players_{sid}.json"
        if not path.exists():
            continue
        players = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(players)
        df["season"] = label
        df["season_id"] = sid
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    print(f"  Total player records: {len(result)}")
    return result


# ═══════════════════════════════════════════════════════════════════
# 1b. CLOSING-ODDS MARKET BENCHMARK (Bet365 B365C)
# ═══════════════════════════════════════════════════════════════════

TEAM_MAP_FD = {
    'Arsenal':'Arsenal','Aston Villa':'Aston Villa',
    'Bournemouth':'AFC Bournemouth','Brentford':'Brentford',
    'Brighton':'Brighton & Hove Albion','Burnley':'Burnley',
    'Chelsea':'Chelsea','Crystal Palace':'Crystal Palace',
    'Everton':'Everton','Fulham':'Fulham',
    'Ipswich':'Ipswich Town','Leeds':'Leeds United','Leicester':'Leicester City',
    'Liverpool':'Liverpool','Luton':'Luton Town',
    'Man City':'Manchester City','Man United':'Manchester United',
    'Newcastle':'Newcastle United','Norwich':'Norwich City',
    "Nott'm Forest":'Nottingham Forest',
    'Sheffield United':'Sheffield United','Southampton':'Southampton',
    'Tottenham':'Tottenham Hotspur','Watford':'Watford',
    'West Brom':'West Bromwich Albion','West Ham':'West Ham United',
    'Wolves':'Wolverhampton Wanderers',
}


def load_closing_odds() -> pd.DataFrame:
    """Load Bet365 CLOSING odds (B365C) from football-data.co.uk.

    The unified market baseline is the pre-match closing line. The footystats
    API exposes a single odds snapshot (odds_ft_*) that is neither documented
    as opening nor closing, so it is not used as the benchmark.

    Merge key is (season, home_name, away_name): a fixture is unique within a
    season, so this is robust to rescheduled matches whose recorded dates
    differ between the two sources.
    """
    SEASON_CSV = {"2019-20": "PL1920.csv", "2020-21": "PL2021.csv",
                  "2021-22": "PL2122.csv", "2022-23": "PL2223.csv",
                  "2023-24": "PL2324.csv", "2024-25": "PL2425.csv"}
    frames = []
    for season, fname in SEASON_CSV.items():
        p = PROJ_ROOT / "league" / "data" / "raw" / fname
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d.rename(columns={"Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
                              "B365CH": "odds_home_close", "B365CD": "odds_draw_close",
                              "B365CA": "odds_away_close"})
        if not {"odds_home_close", "odds_draw_close", "odds_away_close"}.issubset(d.columns):
            continue
        d["season"] = season
        d["home_name"] = d["home_team"].map(TEAM_MAP_FD).fillna(d["home_team"])
        d["away_name"] = d["away_team"].map(TEAM_MAP_FD).fillna(d["away_team"])
        frames.append(d[["season", "home_name", "away_name",
                         "odds_home_close", "odds_draw_close", "odds_away_close"]])
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # Guard against any duplicate fixture rows
    result = result.drop_duplicates(subset=["season", "home_name", "away_name"])
    print(f"  Closing-odds rows loaded: {len(result)}")
    return result


# ═══════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

def compute_rolling_features(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    """Compute rolling-window team stats per season.

    For each team, computes rolling averages of:
      - goals for / against
      - goal difference
      - points per game
      - shots for / against
      - possession
    """
    if windows is None:
        windows = [5, 10]

    result = df.copy()

    for team_prefix, team_col, gf_col, ga_col in [
        ("home_", "home_team", "home_goals", "away_goals"),
        ("away_", "away_team", "away_goals", "home_goals"),
    ]:
        # Cross-season rolling: keep continuity across seasons
        # Shift by 1 to avoid lookahead, then rolling window
        for w in windows:
            # Goals
            gf_roll = result.groupby(team_col)[gf_col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            result[f"{team_prefix}avg_goals_for_{w}"] = gf_roll

            ga_roll = result.groupby(team_col)[ga_col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            result[f"{team_prefix}avg_goals_against_{w}"] = ga_roll

            result[f"{team_prefix}avg_goal_diff_{w}"] = gf_roll - ga_roll

            # Points (win=3, draw=1, loss=0)
            def pts(gf, ga):
                if gf > ga:
                    return 3
                elif gf == ga:
                    return 1
                return 0

            result["_pts"] = result.apply(lambda r: pts(r[gf_col], r[ga_col]), axis=1)
            pts_roll = result.groupby(team_col)["_pts"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            result[f"{team_prefix}avg_points_{w}"] = pts_roll
            result.drop(columns=["_pts"], inplace=True)

            # Win rate
            result["_win"] = (result[gf_col] > result[ga_col]).astype(float)
            win_roll = result.groupby(team_col)["_win"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            result[f"{team_prefix}win_rate_{w}"] = win_roll
            result.drop(columns=["_win"], inplace=True)

            # Shots (if available)
            a_shots = f"team_{'a' if team_prefix == 'home_' else 'b'}_shots"
            if a_shots in result.columns:
                s_roll = result.groupby(team_col)[a_shots].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean()
                )
                result[f"{team_prefix}avg_shots_{w}"] = s_roll

            # Shots on target
            a_sot = f"team_{'a' if team_prefix == 'home_' else 'b'}_shotsOnTarget"
            if a_sot in result.columns:
                sot_roll = result.groupby(team_col)[a_sot].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean()
                )
                result[f"{team_prefix}avg_sot_{w}"] = sot_roll

            # Possession
            a_poss = f"team_{'a' if team_prefix == 'home_' else 'b'}_possession"
            if a_poss in result.columns:
                poss_roll = result.groupby(team_col)[a_poss].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean()
                )
                result[f"{team_prefix}avg_possession_{w}"] = poss_roll

    return result


def compute_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Bet365 CLOSING odds (B365C) to implied probabilities.

    The market baseline is the pre-match closing line merged in from
    football-data.co.uk (see load_closing_odds). Odds are ONLY an evaluation
    benchmark here — they are never used as model features.
    """
    result = df.copy()

    # Check if closing odds columns exist (merged from football-data)
    if not all(c in result.columns for c in ["odds_home_close", "odds_draw_close", "odds_away_close"]):
        print("  WARNING: closing odds columns not found!")
        return result

    # Implied probabilities (with overround correction)
    impl_h = 1.0 / result["odds_home_close"].replace(0, np.nan)
    impl_d = 1.0 / result["odds_draw_close"].replace(0, np.nan)
    impl_a = 1.0 / result["odds_away_close"].replace(0, np.nan)
    total = impl_h + impl_d + impl_a

    result["market_home_prob"] = impl_h / total
    result["market_draw_prob"] = impl_d / total
    result["market_away_prob"] = impl_a / total
    result["market_overround"] = total - 1.0

    # Market's predicted winner (as class index: 2=home, 1=draw, 0=away)
    # Home-first tie-break: symmetric matches (home==away odds) pick home,
    # matching the convention in trackb_clean_eval / fair_comparison.
    home_first = np.argmax(
        np.column_stack([result["market_home_prob"], result["market_draw_prob"], result["market_away_prob"]]),
        axis=1  # 0=home, 1=draw, 2=away
    )
    result["market_pred"] = np.where(home_first == 0, 2, np.where(home_first == 2, 0, 1))

    return result


def compute_prematch_features(df: pd.DataFrame) -> pd.DataFrame:
    """Use pre-match PPG and xG features.

    Note: pre_match_home_ppg is 0 for GW1-GW2 (tournament-internal).
    But home_ppg (cross-season rolling) is always available.
    """
    result = df.copy()

    if "pre_match_home_ppg" in result.columns:
        result["pre_match_home_ppg_flag"] = (result["pre_match_home_ppg"] > 0).astype(float)

    if "team_a_xg_prematch" in result.columns:
        result["pre_match_xg_home"] = result["team_a_xg_prematch"]
        result["pre_match_xg_away"] = result["team_b_xg_prematch"]
        result["pre_match_xg_diff"] = result["pre_match_xg_home"] - result["pre_match_xg_away"]
        # Flag for availability (useful since GW1-2 = 0)
        result["pre_match_xg_available"] = (result["pre_match_xg_home"] > 0).astype(float)

    # home_ppg / away_ppg (cross-season rolling, always available from GW1)
    if "home_ppg" in result.columns:
        result["home_ppg_cross"] = result["home_ppg"]
        result["away_ppg_cross"] = result["away_ppg"]
        result["ppg_diff_cross"] = result["home_ppg"] - result["away_ppg"]

    return result


# ═══════════════════════════════════════════════════════════════════
# 3. SQUAD FEATURES FROM PLAYER DATA
# ═══════════════════════════════════════════════════════════════════

def build_squad_features(player_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player data into team-level squad descriptors per season.

    Uses pre-season known features only to avoid lookahead bias:
      - Age distribution
      - Position mix (counts per position)
      - Squad depth (minutes concentration)
      - Height/weight averages
    """
    records = []

    for (season, season_id), squad in player_df.groupby(["season", "season_id"]):
        for team_id, team_squad in squad.groupby("club_team_id"):
            n_players = len(team_squad)
            avg_age = team_squad["age"].mean()
            avg_height = team_squad["height"].mean() if "height" in team_squad.columns else 0
            avg_weight = team_squad["weight"].mean() if "weight" in team_squad.columns else 0

            # Position counts
            pos_counts = team_squad["position"].value_counts()
            n_gk = pos_counts.get("Goalkeeper", 0)
            n_def = pos_counts.get("Defender", 0)
            n_mid = pos_counts.get("Midfielder", 0)
            n_fwd = pos_counts.get("Forward", 0)

            # Squad depth: players with > 900 min (half season equivalent)
            regulars = team_squad[team_squad["minutes_played_overall"] >= 900]
            n_regulars = len(regulars)
            regular_ratio = n_regulars / n_players if n_players > 0 else 0

            # Minutes concentration (Herfindahl index)
            total_min = team_squad["minutes_played_overall"].sum()
            if total_min > 0:
                min_shares = team_squad["minutes_played_overall"] / total_min
                min_concentration = (min_shares ** 2).sum()
            else:
                min_concentration = 0

            # Experience proxy: average min_per_match
            avg_min_per_match = team_squad["min_per_match"].mean()

            # Team name from the first player's club
            team_name = team_squad.iloc[0].get("club_name", f"team_{team_id}")

            records.append({
                "season": season,
                "season_id": season_id,
                "team_id": team_id,
                # Squad composition
                "n_players": n_players,
                "n_gk": n_gk,
                "n_def": n_def,
                "n_mid": n_mid,
                "n_fwd": n_fwd,
                "def_ratio": n_def / n_players if n_players else 0,
                "mid_ratio": n_mid / n_players if n_players else 0,
                "fwd_ratio": n_fwd / n_players if n_players else 0,
                # Physical
                "avg_age": avg_age,
                "avg_height": avg_height,
                "avg_weight": avg_weight,
                # Depth
                "n_regulars": n_regulars,
                "regular_ratio": regular_ratio,
                "min_concentration": min_concentration,
                "avg_min_per_match": avg_min_per_match,
                # Squad size-adjusted features
                "def_per_10players": n_def / n_players * 10 if n_players else 0,
                "mid_per_10players": n_mid / n_players * 10 if n_players else 0,
                "fwd_per_10players": n_fwd / n_players * 10 if n_players else 0,
            })

    squad_df = pd.DataFrame(records)
    squad_df = squad_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"  Squad features: {len(squad_df)} team-seasons, {len(squad_df.columns)} columns")
    return squad_df


def merge_squad_features(match_df: pd.DataFrame, squad_df: pd.DataFrame) -> pd.DataFrame:
    """Map team_id to home/away squad features via club_team_id.

    In footystats API, team IDs are in columns like homeID/awayID.
    """
    result = match_df.copy()

    # We need a mapping from team_id to team name for each season
    # The match data has homeID and awayID which are the same as club_team_id in player data

    # Merge home team squad features
    home_squad = squad_df.copy()
    home_squad.columns = [f"home_{c}" if c not in ["season", "season_id", "team_id"]
                          else c for c in home_squad.columns]
    result = result.merge(
        home_squad, left_on=["season_id", "homeID"],
        right_on=["season_id", "team_id"], how="left", suffixes=("", "_home_dup")
    )
    # Drop duplicate columns from merge
    for c in result.columns:
        if c.endswith("_home_dup"):
            result.drop(columns=[c], inplace=True)

    # Merge away team squad features
    away_squad = squad_df.copy()
    away_squad.columns = [f"away_{c}" if c not in ["season", "season_id", "team_id"]
                          else c for c in away_squad.columns]
    result = result.merge(
        away_squad, left_on=["season_id", "awayID"],
        right_on=["season_id", "team_id"], how="left", suffixes=("", "_away_dup")
    )
    for c in result.columns:
        if c.endswith("_away_dup"):
            result.drop(columns=[c], inplace=True)

    # Squad differential features
    squad_diff_cols = [c for c in squad_df.columns
                       if c not in ["season", "season_id", "team_id"]]
    for col in squad_diff_cols:
        h_col = f"home_{col}"
        a_col = f"away_{col}"
        if h_col in result.columns and a_col in result.columns:
            result[f"squad_diff_{col}"] = result[h_col] - result[a_col]

    return result


# ═══════════════════════════════════════════════════════════════════
# 4. FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════

def get_feature_cols(df: pd.DataFrame) -> list:
    """Identify feature columns for training."""
    rolling_prefixes = [
        "home_avg_goals_for_", "home_avg_goals_against_", "home_avg_goal_diff_",
        "away_avg_goals_for_", "away_avg_goals_against_", "away_avg_goal_diff_",
        "home_avg_points_", "away_avg_points_",
        "home_win_rate_", "away_win_rate_",
        "home_avg_shots_", "away_avg_shots_",
        "home_avg_sot_", "away_avg_sot_",
        "home_avg_possession_", "away_avg_possession_",
    ]

    squad_prefixes = [
        "home_n_players", "away_n_players",
        "home_n_gk", "away_n_gk", "home_n_def", "away_n_def",
        "home_n_mid", "away_n_mid", "home_n_fwd", "away_n_fwd",
        "home_avg_age", "away_avg_age",
        "home_regular_ratio", "away_regular_ratio",
        "home_min_concentration", "away_min_concentration",
        "home_avg_min_per_match", "away_avg_min_per_match",
        "home_def_per_10players", "away_def_per_10players",
        "home_mid_per_10players", "away_mid_per_10players",
        "home_fwd_per_10players", "away_fwd_per_10players",
        "squad_diff_",
    ]

    # NOTE: market (odds-derived) columns are deliberately NOT features —
    # the model is built without any betting odds input.
    prematch_cols = ["home_ppg_cross", "away_ppg_cross", "ppg_diff_cross",
                     "pre_match_xg_diff"]

    features = []
    for c in df.columns:
        for prefix in rolling_prefixes + squad_prefixes + prematch_cols:
            if c.startswith(prefix) or c == prefix:
                features.append(c)
                break

    # Deduplicate
    return sorted(set(features))


# ═══════════════════════════════════════════════════════════════════
# 5. TRAIN & EVALUATE
# ═══════════════════════════════════════════════════════════════════

def fill_na(X: np.ndarray) -> np.ndarray:
    X = X.copy().astype(float)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X


def temporal_split_by_season(df: pd.DataFrame, test_seasons: list):
    """Split by season: train on earlier seasons, test on later ones."""
    train = df[~df["season"].isin(test_seasons)].copy()
    test = df[df["season"].isin(test_seasons)].copy()
    print(f"  Train: {len(train)} matches ({train['season'].nunique()} seasons)")
    print(f"  Test:  {len(test)} matches ({test['season'].nunique()} seasons)")
    return train, test


def train_lr(X_train, y_train, X_test, y_test, label=""):
    """Train Logistic Regression and return metrics."""
    X_train_f = fill_na(X_train.values if hasattr(X_train, 'values') else X_train)
    X_test_f = fill_na(X_test.values if hasattr(X_test, 'values') else X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_f)
    X_test_s = scaler.transform(X_test_f)

    # Use lbfgs which supports multinomial natively (avoids liblinear warning)
    lr = LogisticRegression(
        penalty="l2", solver="lbfgs", C=1.0, max_iter=1000,
        random_state=42, multi_class="multinomial"
    )
    lr.fit(X_train_s, y_train)

    y_pred = lr.predict(X_test_s)
    y_prob = lr.predict_proba(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average="weighted")
    ll = log_loss(y_test, y_prob)

    # Per-class Brier score
    brier_scores = []
    for i in range(3):
        y_bin = (y_test == i).astype(int)
        bs = brier_score_loss(y_bin, y_prob[:, i])
        brier_scores.append(bs)
    brier_mean = np.mean(brier_scores)

    print(f"\n  [{label}] Results:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    F1 (weighted): {f1_w:.4f}")
    print(f"    Log-loss: {ll:.4f}")
    print(f"    Brier (mean): {brier_mean:.4f}")
    print(f"    Brier per class (A/D/H): {[f'{b:.4f}' for b in brier_scores]}")

    return {
        "accuracy": acc, "f1_weighted": f1_w, "log_loss": ll,
        "brier_mean": brier_mean, "brier_per_class": brier_scores,
        "y_pred": y_pred, "y_prob": y_prob, "model": lr, "scaler": scaler,
    }


def evaluate_market(df_test: pd.DataFrame, y_test):
    """Evaluate market odds as a predictor on test set."""
    market_prob = df_test[["market_away_prob", "market_draw_prob", "market_home_prob"]].values
    market_pred = df_test["market_pred"].values

    acc = accuracy_score(y_test, market_pred)
    ll = log_loss(y_test, market_prob)
    brier_scores = []
    for i in range(3):
        y_bin = (y_test == i).astype(int)
        bs = brier_score_loss(y_bin, market_prob[:, i])
        brier_scores.append(bs)
    brier_mean = np.mean(brier_scores)

    print(f"\n  [Market Odds] Results:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Log-loss: {ll:.4f}")
    print(f"    Brier (mean): {brier_mean:.4f}")
    print(f"    Brier per class (A/D/H): {[f'{b:.4f}' for b in brier_scores]}")

    return {
        "accuracy": acc, "log_loss": ll,
        "brier_mean": brier_mean, "brier_per_class": brier_scores,
        "y_prob": market_prob, "y_pred": market_pred,
    }


# ═══════════════════════════════════════════════════════════════════
# 6. STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════

def bootstrap_brier_diff(y_test, model_prob, market_prob, n_bootstrap=5000):
    """Bootstrap test for Brier score difference between model and market.

    Returns: (mean_diff, ci_lower, ci_upper, pct_model_wins)
      where diff = model_brier - market_brier (negative = model better)
    """
    n = len(y_test)
    model_brier = np.zeros(n)
    market_brier = np.zeros(n)

    for i in range(3):
        y_bin = (y_test == i).astype(float)
        model_brier += (model_prob[:, i] - y_bin) ** 2
        market_brier += (market_prob[:, i] - y_bin) ** 2

    model_brier /= 3
    market_brier /= 3

    diffs = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        diffs.append(model_brier[idx].mean() - market_brier[idx].mean())

    diffs = np.sort(diffs)
    ci_lower = diffs[int(0.025 * n_bootstrap)]
    ci_upper = diffs[int(0.975 * n_bootstrap)]
    mean_diff = np.mean(diffs)
    pct_model_wins = (np.array(diffs) < 0).mean() * 100

    print(f"\n  [Bootstrap Test] Model Brier - Market Brier:")
    print(f"    Mean diff: {mean_diff:.4f}")
    print(f"    95% CI:    [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"    Model 'wins' (diff<0): {pct_model_wins:.1f}% of bootstraps")
    if ci_upper < 0:
        print(f"    >>> Model significantly BETTER than market (p<0.05)")
    elif ci_lower > 0:
        print(f"    >>> Market significantly BETTER than model (p<0.05)")
    else:
        print(f"    >>> No significant difference (model ≈ market)")

    return {"mean_diff": mean_diff, "ci_lower": ci_lower, "ci_upper": ci_upper,
            "pct_model_wins": pct_model_wins}


def diebold_mariano_test(y_test, model_prob, market_prob):
    """Diebold-Mariano test for equal predictive accuracy.

    Uses squared error loss for multi-class probabilities.
    """
    n = len(y_test)
    # Multi-class squared error loss per observation
    model_loss = np.zeros(n)
    market_loss = np.zeros(n)
    for i in range(3):
        y_bin = (y_test == i).astype(float)
        model_loss += (model_prob[:, i] - y_bin) ** 2
        market_loss += (market_prob[:, i] - y_bin) ** 2

    model_loss /= 3
    market_loss /= 3

    d = model_loss - market_loss  # loss differential
    d_bar = d.mean()

    # Newey-West-type variance (just use lag-1 autocorrelation)
    gamma_0 = np.var(d, ddof=1)
    gamma_1 = np.cov(d[:-1], d[1:])[0, 1] if n > 1 else 0
    var_d = (gamma_0 + 2 * gamma_1) / n

    if var_d <= 0:
        return {"DM_stat": 0, "p_value": 1.0, "conclusion": "Variance too small"}

    DM_stat = d_bar / np.sqrt(var_d)

    # Two-sided p-value from normal distribution
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(DM_stat)))

    print(f"\n  [Diebold-Mariano Test]")
    print(f"    DM statistic: {DM_stat:.4f}")
    print(f"    p-value:      {p_value:.4f}")
    if p_value < 0.05:
        if DM_stat < 0:
            print(f"    >>> Model significantly BETTER than market (DM<0, p<0.05)")
        else:
            print(f"    >>> Market significantly BETTER than model (DM>0, p<0.05)")
    else:
        print(f"    >>> No significant difference (p={p_value:.3f})")

    return {"DM_stat": DM_stat, "p_value": p_value}


# ═══════════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  FootyStats Backtest — Premier League 2018-2025")
    print("=" * 65)

    # ── 1. Load data ──
    print("\n[1/6] Loading match data...")
    df = load_matches()

    print("\n[2/6] Loading player data...")
    players = load_players()

    print("  Loading Bet365 CLOSING odds (market benchmark, not features)...")
    closing = load_closing_odds()
    if len(closing):
        df = df.merge(closing, on=["season", "home_name", "away_name"], how="left")
        print(f"  Closing-odds merge rate: {df['odds_home_close'].notna().mean()*100:.1f}%")

    # ── 2. Feature engineering ──
    print("\n[3/6] Computing features...")

    print("  Rolling team stats...")
    df = compute_rolling_features(df, windows=[5, 10])

    print("  Odds → implied probabilities...")
    df = compute_odds_features(df)

    print("  Pre-match features...")
    df = compute_prematch_features(df)

    print("  Building squad features from player data...")
    squad = build_squad_features(players)
    df = merge_squad_features(df, squad)

    # Identify feature sets
    all_features = get_feature_cols(df)
    rolling_cols = [c for c in all_features if c.startswith(("home_avg_", "away_avg_", "home_win_", "away_win_"))]
    squad_cols = [c for c in all_features if c.startswith(("home_n_", "away_n_", "squad_diff_",
                                                            "home_avg_age", "away_avg_age",
                                                            "home_regular", "away_regular",
                                                            "home_min_", "away_min_",
                                                            "home_def_", "away_def_",
                                                            "home_mid_", "away_mid_",
                                                            "home_fwd_", "away_fwd_"))]
    prematch_cols = [c for c in all_features if c.startswith(("home_ppg", "away_ppg", "ppg_", "pre_match_"))]

    print(f"\n  Feature summary:")
    print(f"    Rolling ({len(rolling_cols)}):   {rolling_cols[:3]}...")
    print(f"    Squad ({len(squad_cols)}):     {squad_cols[:3]}...")
    print(f"    Market: benchmark only (closing odds B365C), NEVER model features")
    print(f"    Pre-match ({len(prematch_cols)}): {prematch_cols}")

    # ── 3. Temporal split ──
    print("\n[4/6] Temporal train/test split...")
    test_seasons = ["2023-24", "2024-25"]  # last 2 seasons for testing
    train_df, test_df = temporal_split_by_season(df, test_seasons)
    y_train = train_df["target_result"].values
    y_test = test_df["target_result"].values

    # ── 4. Train models ──
    print("\n[5/6] Training & evaluation...")

    # Baseline: rolling stats only
    base_cols = [c for c in rolling_cols if c in train_df.columns]
    print(f"\n  Baseline features: {len(base_cols)}")
    base_results = train_lr(
        train_df[base_cols], y_train, test_df[base_cols], y_test,
        label="BASELINE (rolling stats)"
    )

    # Rolling + squad features
    rs_cols = [c for c in list(set(rolling_cols + squad_cols)) if c in train_df.columns]
    print(f"\n  Rolling+Squad features: {len(rs_cols)}")
    rs_results = train_lr(
        train_df[rs_cols], y_train, test_df[rs_cols], y_test,
        label="ROLLING + SQUAD"
    )

    # Rolling + pre-match xG/PPG
    rp_cols = [c for c in list(set(rolling_cols + prematch_cols)) if c in train_df.columns]
    print(f"\n  Rolling+PreMatch features: {len(rp_cols)}")
    rp_results = train_lr(
        train_df[rp_cols], y_train, test_df[rp_cols], y_test,
        label="ROLLING + PRE-MATCH"
    )

    # All features
    all_train_cols = [c for c in all_features if c in train_df.columns]
    print(f"\n  ALL features: {len(all_train_cols)}")
    all_results = train_lr(
        train_df[all_train_cols], y_train, test_df[all_train_cols], y_test,
        label="ALL FEATURES"
    )

    # Market odds as benchmark
    print(f"\n  Market odds benchmark:")
    # Market odds don't need training - they're already there
    market_results = evaluate_market(test_df, y_test)

    # ── 5. Statistical tests ──
    print("\n[6/6] Statistical tests...")

    # Compare best model vs market
    best_model_prob = all_results["y_prob"]
    market_prob = test_df[["market_away_prob", "market_draw_prob", "market_home_prob"]].values

    print("\n  ── Model vs Market ──")
    bootstrap_brier_diff(y_test, best_model_prob, market_prob)

    try:
        diebold_mariano_test(y_test, best_model_prob, market_prob)
    except Exception as e:
        print(f"  DM test skipped: {e}")

    # ── Summary table ──
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    print(f"  {'Model':<25} {'Acc':<8} {'F1':<8} {'Brier':<8} {'LogLoss':<8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, res in [
        ("Baseline (rolling)", base_results),
        ("Rolling + Squad", rs_results),
        ("Rolling + PreMatch", rp_results),
        ("All features", all_results),
        ("Market odds (closing)", market_results),
    ]:
        if isinstance(res, dict) and res.get("accuracy"):
            f1 = res.get("f1_weighted", 0)
            print(f"  {name:<25} {res['accuracy']:<8.4f} {f1:<8.4f} "
                  f"{res['brier_mean']:<8.4f} {res['log_loss']:<8.4f}")

    # ── Save predictions ──
    pred_path = OUTPUT_DIR / "predictions.csv"
    test_out = test_df[["date", "season", "game_week", "home_team", "away_team",
                        "home_goals", "away_goals"]].copy()
    test_out["actual"] = y_test
    test_out["model_pred"] = all_results["y_pred"]
    test_out["market_pred"] = test_df["market_pred"].values
    for i, label in enumerate(["prob_away", "prob_draw", "prob_home"]):
        test_out[f"model_{label}"] = all_results["y_prob"][:, i]
        test_out[f"market_{label}"] = market_prob[:, i]
    test_out.to_csv(pred_path, index=False)
    print(f"\n  Predictions saved: {pred_path}")

    # ── Feature importance ──
    print("\n\n  Top 20 feature coefficients (All Features model):")
    coef = all_results["model"].coef_[0]
    feat_names = all_train_cols
    imp = pd.DataFrame({"feature": feat_names, "coef": np.abs(coef)})
    imp["direction"] = ["+" if coef[list(feat_names).index(f)] > 0 else "-"
                        for f in feat_names]
    imp = imp.sort_values("coef", ascending=False).head(20)
    for _, row in imp.iterrows():
        print(f"    {row['direction']} {row['feature']:<40} {row['coef']:.4f}")

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
