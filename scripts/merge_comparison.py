#!/usr/bin/env python3
"""
Merge footystats extra features into the original pipeline and compare.

Compares:
  1. Original pipeline (rolling stats only) on PL data
  2. Original pipeline + footystats extras (PPG, xG, possession)
  3. Market odds benchmark

Seasons: 2020/21 → 2024/25 (5 seasons overlapping between both data sources)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss, brier_score_loss

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

OUTPUT_DIR = PROJ_ROOT / "outputs" / "merge_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Team name mapping (football-data.co.uk → footystats) ──────────
TEAM_MAP = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "AFC Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton & Hove Albion",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich": "Ipswich Town",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Liverpool": "Liverpool",
    "Luton": "Luton Town",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Norwich": "Norwich City",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield United",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Tottenham": "Tottenham Hotspur",
    "Watford": "Watford",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
}

# Reverse mapping (footystats → original)
REV_MAP = {v: k for k, v in TEAM_MAP.items()}

# Season ID mapping (footystats season_id → label)
FT_SEASONS = {
    4759: "2020-21", 6135: "2021-22", 7704: "2022-23",
    9660: "2023-24", 12325: "2024-25",
}
# football-data.co.uk files per season
ORIG_FILES = {
    "2020-21": "PL2021.csv", "2021-22": "PL2122.csv", "2022-23": "PL2223.csv",
    "2023-24": "PL2324.csv", "2024-25": "PL2425.csv",
}


def load_original_pl(data_dir: Path) -> pd.DataFrame:
    """Load PL data from football-data.co.uk CSV files directly."""
    COL_MAP = {
        "Div": "division", "Date": "date", "HomeTeam": "home_team",
        "AwayTeam": "away_team", "FTHG": "home_goals", "FTAG": "away_goals",
        "FTR": "result", "B365H": "odds_home", "B365D": "odds_draw",
        "B365A": "odds_away",
    }

    frames = []
    for season, fname in ORIG_FILES.items():
        path = data_dir / fname
        if not path.exists():
            print(f"  File not found: {path}")
            continue
        df = pd.read_csv(path)
        df.rename(columns=COL_MAP, inplace=True)
        # Keep only mapped columns
        df = df[[c for c in COL_MAP.values() if c in df.columns]]
        df["season"] = season
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        # Targets
        df["target_result"] = df["result"].map({"H": 2, "D": 1, "A": 0})
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["season", "date"]).reset_index(drop=True)
    print(f"  Original PL: {len(result)} matches, {result['season'].nunique()} seasons")
    return result


def load_footystats_pl(raw_dir: Path) -> pd.DataFrame:
    """Load PL data from footystats API JSON files."""
    frames = []
    for sid, season in FT_SEASONS.items():
        path = raw_dir / f"matches_{sid}.json"
        if not path.exists():
            continue
        matches = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(matches)
        df["date"] = pd.to_datetime(df["date_unix"], unit="s")
        df["season"] = season
        df["home_team"] = df["home_name"]
        df["away_team"] = df["away_name"]
        df["home_goals"] = df["homeGoalCount"]
        df["away_goals"] = df["awayGoalCount"]
        df["target_result"] = df.apply(
            lambda r: 2 if r["homeGoalCount"] > r["awayGoalCount"]
            else 1 if r["homeGoalCount"] == r["awayGoalCount"]
            else 0, axis=1
        )
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["season", "date"]).reset_index(drop=True)
    print(f"  Footystats PL: {len(result)} matches, {result['season'].nunique()} seasons")
    return result


def merge_extra_features(orig_df: pd.DataFrame, ft_df: pd.DataFrame) -> pd.DataFrame:
    """Merge footystats extra features into original data by matching team+date."""
    result = orig_df.copy()

    # Map team names to footystats format
    result["home_team_ft"] = result["home_team"].map(TEAM_MAP)
    result["away_team_ft"] = result["away_team"].map(TEAM_MAP)

    # Normalize dates to date-only for matching
    result["date_key"] = result["date"].dt.date
    ft_df["date_key"] = ft_df["date"].dt.date

    # Extra features to borrow from footystats
    extra_cols = [
        "home_ppg", "away_ppg",                              # cross-season PPG
        "team_a_xg_prematch", "team_b_xg_prematch",          # pre-match xG
        "pre_match_home_ppg", "pre_match_away_ppg",          # tournament-internal PPG
        "team_a_possession", "team_b_possession",            # match possession
    ]

    # Also keep game_week from footystats for alignment verification
    ft_extra = ft_df[["date_key", "home_team", "away_team",
                      "game_week"] + extra_cols].copy()
    ft_extra.rename(columns={
        "home_team": "home_team_ft_match",
        "away_team": "away_team_ft_match",
        "team_a_xg_prematch": "pre_match_xg_home",
        "team_b_xg_prematch": "pre_match_xg_away",
        "pre_match_home_ppg": "pre_match_ppg_home",
        "pre_match_away_ppg": "pre_match_ppg_away",
        "team_a_possession": "home_possession",
        "team_b_possession": "away_possession",
    }, inplace=True)

    # Merge on (date_key, home_team_ft, away_team_ft)
    merged = result.merge(
        ft_extra,
        left_on=["date_key", "home_team_ft", "away_team_ft"],
        right_on=["date_key", "home_team_ft_match", "away_team_ft_match"],
        how="left", suffixes=("", "_ft")
    )

    n_matched = merged["home_ppg"].notna().sum()
    print(f"  Matched: {n_matched}/{len(merged)} matches ({n_matched/len(merged)*100:.0f}%)")

    # Create differential and derived features
    merged["ppg_diff"] = merged["home_ppg"] - merged["away_ppg"]
    merged["pre_match_xg_diff"] = merged["pre_match_xg_home"] - merged["pre_match_xg_away"]
    merged["possession_diff"] = merged["home_possession"] - merged["away_possession"]

    # Flag for pre-match xG availability (0 for GW1-2)
    merged["pre_match_xg_avail"] = (merged["pre_match_xg_home"] > 0).astype(float)

    # Clean up temporary columns
    drop_cols = ["home_team_ft", "away_team_ft", "date_key",
                 "home_team_ft_match", "away_team_ft_match", "game_week"]
    for c in drop_cols:
        if c in merged.columns:
            merged.drop(columns=[c], inplace=True)

    return merged


def compute_rolling_features(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    """Cross-season rolling features (same as original pipeline)."""
    if windows is None:
        windows = [5, 10]
    result = df.copy()

    for team_prefix, team_col, gf_col, ga_col in [
        ("home_", "home_team", "home_goals", "away_goals"),
        ("away_", "away_team", "away_goals", "home_goals"),
    ]:
        for w in windows:
            gf_roll = result.groupby(team_col)[gf_col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            result[f"{team_prefix}avg_goals_for_{w}"] = gf_roll

            ga_roll = result.groupby(team_col)[ga_col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            result[f"{team_prefix}avg_goals_against_{w}"] = ga_roll
            result[f"{team_prefix}avg_goal_diff_{w}"] = gf_roll - ga_roll

            # Points
            pts = result.apply(lambda r: 3 if r[gf_col] > r[ga_col]
                              else 1 if r[gf_col] == r[ga_col] else 0, axis=1)
            pts_roll = pts.groupby(result[team_col]).transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            result[f"{team_prefix}avg_points_{w}"] = pts_roll

            # Win rate
            win = (result[gf_col] > result[ga_col]).astype(float)
            win_roll = win.groupby(result[team_col]).transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            result[f"{team_prefix}win_rate_{w}"] = win_roll

    return result


def get_base_feature_cols(df: pd.DataFrame) -> list:
    """Get original rolling feature columns."""
    return [c for c in df.columns if any(
        c.startswith(p) for p in ["home_avg_", "away_avg_", "home_win_", "away_win_"]
    )]


def get_extra_feature_cols() -> list:
    """Get footystats extra feature names."""
    return ["home_ppg", "away_ppg", "ppg_diff",
            "pre_match_xg_home", "pre_match_xg_away", "pre_match_xg_diff",
            "home_possession", "away_possession", "possession_diff",
            "pre_match_ppg_home", "pre_match_ppg_away",
            "pre_match_xg_avail"]


def fill_na(X: np.ndarray) -> np.ndarray:
    X = X.copy().astype(float)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X


def temporal_split_by_season(df: pd.DataFrame, test_seasons: list):
    train = df[~df["season"].isin(test_seasons)].copy()
    test = df[df["season"].isin(test_seasons)].copy()
    return train, test


def train_and_eval(X_train, y_train, X_test, y_test, label=""):
    X_tr = fill_na(X_train.values if hasattr(X_train, 'values') else X_train)
    X_te = fill_na(X_test.values if hasattr(X_test, 'values') else X_test)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    lr = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                            max_iter=1000, random_state=42, multi_class="multinomial")
    lr.fit(X_tr, y_train)
    y_pred = lr.predict(X_te)
    y_prob = lr.predict_proba(X_te)

    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average="weighted")
    ll = log_loss(y_test, y_prob)

    brier_scores = [brier_score_loss((y_test == i).astype(int), y_prob[:, i]) for i in range(3)]
    brier_mu = np.mean(brier_scores)

    print(f"  [{label}]")
    print(f"    Accuracy: {acc:.4f}  F1: {f1_w:.4f}  Brier: {brier_mu:.4f}  LogLoss: {ll:.4f}")
    return {"accuracy": acc, "f1_weighted": f1_w, "log_loss": ll,
            "brier_mean": brier_mu, "y_pred": y_pred, "y_prob": y_prob}


def main():
    print("=" * 65)
    print("  Merge Comparison: Original Pipeline + FootyStats Features")
    print("  Premier League 2020-2025")
    print("=" * 65)

    data_dir = PROJ_ROOT / "league" / "data" / "raw"
    raw_dir = PROJ_ROOT / "data" / "footystats" / "raw"

    # ── Load ──
    print("\n[1/4] Loading data...")
    orig = load_original_pl(data_dir)
    ft = load_footystats_pl(raw_dir)

    print("\n[2/4] Merging footystats extra features...")
    merged = merge_extra_features(orig, ft)

    # ── Feature engineering ──
    print("\n[3/4] Computing features...")
    merged = compute_rolling_features(merged, windows=[5, 10])

    base_cols = get_base_feature_cols(merged)
    extra_cols = [c for c in get_extra_feature_cols() if c in merged.columns]

    print(f"  Base features: {len(base_cols)}")
    print(f"  FootyStats extras: {len(extra_cols)}")
    print(f"    {extra_cols}")

    # ── Temporal split ──
    test_seasons = ["2023-24", "2024-25"]
    train, test = temporal_split_by_season(merged, test_seasons)
    y_train = train["target_result"].values
    y_test = test["target_result"].values
    print(f"\n  Train: {len(train)} ({train['season'].nunique()} seasons)")
    print(f"  Test:  {len(test)} ({test['season'].nunique()} seasons)")

    # ── Evaluate ──
    print("\n[4/4] Evaluation...\n")

    # A) Original rolling features only
    r1 = train_and_eval(train[base_cols], y_train, test[base_cols], y_test,
                        label="A) ORIGINAL: rolling stats only")

    # B) Original + footystats extras
    comb_cols = list(set(base_cols + extra_cols))
    r2 = train_and_eval(train[comb_cols], y_train, test[comb_cols], y_test,
                        label="B) ORIGINAL + FOOTYSTATS extras")

    # C) Footystats extras only
    r3 = train_and_eval(train[extra_cols], y_train, test[extra_cols], y_test,
                        label="C) FOOTYSTATS extras only")

    # D) Market odds benchmark (from the original data's Bet365 odds)
    if all(c in test.columns for c in ["odds_home", "odds_draw", "odds_away"]):
        # Compute implied probabilities
        market_prob = np.column_stack([
            1.0 / test["odds_away"].replace(0, np.nan),
            1.0 / test["odds_draw"].replace(0, np.nan),
            1.0 / test["odds_home"].replace(0, np.nan),
        ])
        total = market_prob.sum(axis=1, keepdims=True)
        market_prob = market_prob / total
        market_pred = np.argmax(market_prob, axis=1)
        m_acc = accuracy_score(y_test, market_pred)
        m_ll = log_loss(y_test, market_prob)
        m_brier = np.mean([brier_score_loss((y_test == i).astype(int), market_prob[:, i])
                          for i in range(3)])
        print(f"\n  D) Market odds (Bet365)")
        print(f"    Accuracy: {m_acc:.4f}  Brier: {m_brier:.4f}  LogLoss: {m_ll:.4f}")
    else:
        m_acc, m_brier, m_ll = 0, 0, 0

    # ── Summary ──
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  {'Method':<30} {'Acc':<8} {'Brier':<8} {'LogLoss':<8} {'Feat':<6}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    print(f"  {'A) Rolling only':<30} {r1['accuracy']:<8.4f} {r1['brier_mean']:<8.4f} {r1['log_loss']:<8.4f} {len(base_cols):<6}")
    print(f"  {'B) +FootyStats extras':<30} {r2['accuracy']:<8.4f} {r2['brier_mean']:<8.4f} {r2['log_loss']:<8.4f} {len(comb_cols):<6}")
    print(f"  {'C) FootyStats only':<30} {r3['accuracy']:<8.4f} {r3['brier_mean']:<8.4f} {r3['log_loss']:<8.4f} {len(extra_cols):<6}")
    if m_acc > 0:
        print(f"  {'D) Market odds':<30} {m_acc:<8.4f} {m_brier:<8.4f} {m_ll:<8.4f} {'--':<6}")

    delta = r2["accuracy"] - r1["accuracy"]
    if delta > 0:
        print(f"\n  >>> FootyStats extras improve accuracy by {delta*100:.2f}% ✅")
    else:
        print(f"\n  >>> FootyStats extras change accuracy by {delta*100:.2f}%")

    # Save predictions
    test_out = test[["season", "date", "home_team", "away_team",
                     "home_goals", "away_goals"]].copy()
    test_out["actual"] = y_test
    test_out["base_pred"] = r1["y_pred"]
    test_out["enhanced_pred"] = r2["y_pred"]
    path = OUTPUT_DIR / "predictions_merge.csv"
    test_out.to_csv(path, index=False)
    print(f"\n  Predictions: {path}")


if __name__ == "__main__":
    main()
