#!/usr/bin/env python3
"""
Experiment: Player-Level Squad Features for League Match Prediction
====================================================================

Hypothesis: Aggregating player data (goals, assists, clean sheets, etc.)
into team-level "squad strength" features can improve match result
prediction beyond rolling-window team stats alone.

Data: footystats.org free tier — 2018/19 Premier League
  - england-premier-league-matches-2018-to-2019-stats.csv (380 matches)
  - england-premier-league-players-2018-to-2019-stats.csv (572 players)

Method:
  1. Build baseline: rolling team stats → LR → accuracy
  2. Build +player features: squad aggregates → +baseline → LR → accuracy
  3. Compare: does player data break the ~62% ceiling?

Output: prints results, saves comparison table and predictions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────
DOWNLOAD_DIR = Path(r"C:\Users\-jmmmm\Downloads")
MATCH_FILE = DOWNLOAD_DIR / "england-premier-league-matches-2018-to-2019-stats.csv"
PLAYER_FILE = DOWNLOAD_DIR / "england-premier-league-players-2018-to-2019-stats.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "player_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load data ───────────────────────────────────────────────────
def load_match_data(path: Path) -> pd.DataFrame:
    """Load footystats match CSV, add target result column."""
    df = pd.read_csv(path, na_values=["N/A", "NA", ""])

    # Result target
    df["target_result"] = df.apply(
        lambda r: (2 if r["home_team_goal_count"] > r["away_team_goal_count"]
                   else 1 if r["home_team_goal_count"] == r["away_team_goal_count"]
                   else 0),
        axis=1,
    )
    # Total goals
    df["total_goal_count"] = df["home_team_goal_count"] + df["away_team_goal_count"]
    df["target_over_2.5"] = (df["total_goal_count"] > 2.5).astype(int)

    # Parse date properly
    df["date"] = pd.to_datetime(df["date_GMT"], format="%b %d %Y - %I:%M%p", errors="coerce")

    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"  Matches: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # Check pre-match xG availability
    xg_available = (df["Home Team Pre-Match xG"].notna() & (df["Home Team Pre-Match xG"] > 0)).sum()
    print(f"  Pre-match xG available: {xg_available}/{len(df)}")

    return df


def load_player_data(path: Path) -> pd.DataFrame:
    """Load footystats player CSV."""
    df = pd.read_csv(path, na_values=["N/A", "NA", ""])
    print(f"  Players: {len(df)}")
    print(f"  Teams: {df['Current Club'].nunique()}")
    print(f"  Positions: {df['position'].value_counts().to_dict()}")
    return df


# ── 2. Rolling team stats (baseline features) ──────────────────────
def compute_rolling_features(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """Compute rolling-window team stats from footystats match data.

    Mirrors league/src/features.py but uses footystats column names.
    """
    if windows is None:
        windows = [5, 10]

    result = df.copy()

    for w in windows:
        # ── Per-team rolling features ──
        for team_prefix, team_col, gf_col, ga_col in [
            ("home_", "home_team_name", "home_team_goal_count", "away_team_goal_count"),
            ("away_", "away_team_name", "away_team_goal_count", "home_team_goal_count"),
        ]:
            # Goals for / against (rolling mean)
            gf_roll = result.groupby(team_col)[gf_col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            result[f"{team_prefix}avg_goals_for_{w}"] = gf_roll

            ga_roll = result.groupby(team_col)[ga_col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            result[f"{team_prefix}avg_goals_against_{w}"] = ga_roll

            # Goal diff
            result[f"{team_prefix}avg_goal_diff_{w}"] = gf_roll - ga_roll

            # Points per game (rolling mean)
            def points(row):
                if row[gf_col] > row[ga_col]:
                    return 3
                elif row[gf_col] == row[ga_col]:
                    return 1
                return 0

            result["_pts"] = result.apply(points, axis=1)
            pts_roll = result.groupby(team_col)["_pts"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            result[f"{team_prefix}avg_points_{w}"] = pts_roll
            result.drop(columns=["_pts"], inplace=True)

            # Win / draw / loss rates
            result["_is_win"] = (result[gf_col] > result[ga_col]).astype(float)
            win_roll = result.groupby(team_col)["_is_win"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            result[f"{team_prefix}win_rate_{w}"] = win_roll
            result.drop(columns=["_is_win"], inplace=True)

            # Shot-based rolling (if available)
            if f"{team_prefix}team_shots" in result.columns:
                s_roll = result.groupby(team_col)[f"{team_prefix}team_shots"].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean()
                )
                result[f"{team_prefix}avg_shots_{w}"] = s_roll

            if f"{team_prefix}team_shots_on_target" in result.columns:
                sot_roll = result.groupby(team_col)[f"{team_prefix}team_shots_on_target"].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean()
                )
                result[f"{team_prefix}avg_sot_{w}"] = sot_roll

    # ── Possession features (footystats-specific) ──
    if "home_team_possession" in result.columns:
        result["home_possession"] = result["home_team_possession"]
        result["away_possession"] = result["away_team_possession"]
        result["possession_diff"] = result["home_possession"] - result["away_possession"]

    # ── Betting odds → implied probabilities ──
    if all(c in result.columns for c in ["odds_ft_home_team_win", "odds_ft_draw", "odds_ft_away_team_win"]):
        for prefix, oh, od, oa in [
            ("", "odds_ft_home_team_win", "odds_ft_draw", "odds_ft_away_team_win"),
        ]:
            impl_h = 1.0 / result[oh]
            impl_d = 1.0 / result[od]
            impl_a = 1.0 / result[oa]
            total = impl_h + impl_d + impl_a
            result["impl_home_prob"] = impl_h / total
            result["impl_draw_prob"] = impl_d / total
            result["impl_away_prob"] = impl_a / total
            result["overround"] = total - 1.0

    return result


def get_rolling_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get all rolling-window feature columns (excludes ID/label columns)."""
    exclude = {
        "timestamp", "date_GMT", "date", "status", "attendance",
        "home_team_name", "away_team_name", "stadium_name", "referee",
        "Game Week", "home_team_goal_count", "away_team_goal_count",
        "total_goal_count", "total_goals_at_half_time",
        "home_team_goal_count_half_time", "away_team_goal_count_half_time",
        "home_team_goal_timings", "away_team_goal_timings",
        "home_team_corner_count", "away_team_corner_count",
        "home_team_yellow_cards", "home_team_red_cards",
        "away_team_yellow_cards", "away_team_red_cards",
        "home_team_first_half_cards", "home_team_second_half_cards",
        "away_team_first_half_cards", "away_team_second_half_cards",
        "home_team_shots", "away_team_shots",
        "home_team_shots_on_target", "away_team_shots_on_target",
        "home_team_shots_off_target", "away_team_shots_off_target",
        "home_team_fouls", "away_team_fouls",
        "home_team_possession", "away_team_possession",
        "Home Team Pre-Match xG", "Away Team Pre-Match xG",
        "team_a_xg", "team_b_xg",
        "odds_ft_home_team_win", "odds_ft_draw", "odds_ft_away_team_win",
        "odds_ft_over15", "odds_ft_over25", "odds_ft_over35", "odds_ft_over45",
        "odds_btts_yes", "odds_btts_no",
        "average_goals_per_match_pre_match",
        "btts_percentage_pre_match",
        "over_15_percentage_pre_match", "over_25_percentage_pre_match",
        "over_35_percentage_pre_match", "over_45_percentage_pre_match",
        "over_15_HT_FHG_percentage_pre_match", "over_05_HT_FHG_percentage_pre_match",
        "over_15_2HG_percentage_pre_match", "over_05_2HG_percentage_pre_match",
        "average_corners_per_match_pre_match", "average_cards_per_match_pre_match",
        "target_result", "target_over_2.5",
    }

    # Also exclude raw odds columns
    return [c for c in df.columns if c not in exclude and c not in [
        "home_possession", "away_possession", "possession_diff",
    ]]


# ── 3. Player → Squad features ─────────────────────────────────────
def build_squad_features(player_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player-level data into team-level squad features.

    Returns DataFrame indexed by team name with squad feature columns.
    """
    records = []

    for team, squad in player_df.groupby("Current Club"):
        squad = squad.copy()

        # Basic squad info
        n_players = len(squad)
        avg_age = squad["age"].mean()
        total_minutes = squad["minutes_played_overall"].sum()
        avg_min_per_match = squad["min_per_match"].mean()

        # Players with significant minutes (at least half a season)
        regulars = squad[squad["minutes_played_overall"] >= 900]
        n_regulars = len(regulars)

        # Playing style / performance features
        total_goals = squad["goals_overall"].sum()
        total_assists = squad["assists_overall"].sum()
        total_goal_involvement = total_goals + total_assists

        # Goals by position
        forward_goals = squad[squad["position"] == "Forward"]["goals_overall"].sum()
        midfielder_goals = squad[squad["position"] == "Midfielder"]["goals_overall"].sum()
        defender_goals = squad[squad["position"] == "Defender"]["goals_overall"].sum()

        # Per-90 rates (weighted by minutes)
        def weighted_avg(col):
            """Weighted average by minutes played."""
            return np.average(squad[col], weights=squad["minutes_played_overall"]) if squad["minutes_played_overall"].sum() > 0 else 0

        team_goals_per90 = weighted_avg("goals_per_90_overall")
        team_assists_per90 = weighted_avg("assists_per_90_overall")
        team_conceded_per90 = weighted_avg("conceded_per_90_overall")
        team_cards_per90 = weighted_avg("cards_per_90_overall")

        # Squad depth / concentration metrics
        # Gini-like concentration: how concentrated are goals in a few players?
        if total_goals > 0:
            goal_shares = squad["goals_overall"] / total_goals
            goal_concentration = (goal_shares ** 2).sum()  # Herfindahl index
        else:
            goal_concentration = 1.0 / n_players

        # Minutes concentration (squad rotation / depth)
        if total_minutes > 0:
            min_shares = squad["minutes_played_overall"] / total_minutes
            min_concentration = (min_shares ** 2).sum()
        else:
            min_concentration = 1.0 / n_players

        # Clean sheets (defenders + GK)
        cs_defenders = squad[squad["position"].isin(["Defender", "Goalkeeper"])]["clean_sheets_overall"].sum()

        # Goal involvement across positions — balance
        if total_goal_involvement > 0:
            forward_involvement = squad[squad["position"] == "Forward"]["goals_involved_per_90_overall"].sum()
            midfield_involvement = squad[squad["position"] == "Midfielder"]["goals_involved_per_90_overall"].sum()
        else:
            forward_involvement = 0
            midfield_involvement = 0

        # Top scorer dominance
        top_scorer_goals = squad["goals_overall"].max()

        # Rank-based quality indicators
        avg_attacker_rank = squad["rank_in_league_top_attackers"].mean()
        avg_defender_rank = squad["rank_in_league_top_defenders"].mean()
        avg_midfielder_rank = squad["rank_in_league_top_midfielders"].mean()

        records.append({
            "team": team,
            # Squad size & depth
            "n_players": n_players,
            "n_regulars": n_regulars,        # ≥900 min
            "regular_ratio": n_regulars / n_players if n_players > 0 else 0,
            "avg_age": avg_age,
            "total_minutes": total_minutes,
            "avg_min_per_match": avg_min_per_match,
            # Production
            "total_goals": total_goals,
            "total_assists": total_assists,
            "total_goal_involvement": total_goal_involvement,
            "forward_goals": forward_goals,
            "midfielder_goals": midfielder_goals,
            "defender_goals": defender_goals,
            # Per-90 rates
            "team_goals_per90": team_goals_per90,
            "team_assists_per90": team_assists_per90,
            "team_conceded_per90": team_conceded_per90,
            "team_cards_per90": team_cards_per90,
            # Concentration / balance
            "goal_concentration": goal_concentration,   # higher = more reliant on few players
            "min_concentration": min_concentration,     # higher = less rotation
            "forward_goal_share": forward_goals / total_goals if total_goals > 0 else 0,
            "midfielder_goal_share": midfielder_goals / total_goals if total_goals > 0 else 0,
            # Defense
            "clean_sheets_def_gk": cs_defenders,
            "team_conceded_total": squad["conceded_overall"].sum(),
            # Top player dominance
            "top_scorer_goals": top_scorer_goals,
            "top_scorer_share": top_scorer_goals / total_goals if total_goals > 0 else 0,
            # Scout rankings
            "avg_attacker_rank": avg_attacker_rank,
            "avg_midfielder_rank": avg_midfielder_rank,
            "avg_defender_rank": avg_defender_rank,
        })

    squad_df = pd.DataFrame(records)
    # Fill potential inf/NaN from division by zero
    squad_df = squad_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    squad_df.set_index("team", inplace=True)

    print(f"\n  Squad features: {len(squad_df.columns)} columns")
    print(f"  Teams: {len(squad_df)}")
    print(f"  Feature list: {list(squad_df.columns)}")

    return squad_df


def merge_squad_features(match_df: pd.DataFrame, squad_df: pd.DataFrame) -> pd.DataFrame:
    """Merge squad features into match DataFrame for home and away teams."""
    result = match_df.copy()

    # Merge home team squad features
    home_squad = squad_df.add_prefix("home_")
    result = result.merge(home_squad, left_on="home_team_name", right_index=True, how="left")

    # Merge away team squad features
    away_squad = squad_df.add_prefix("away_")
    result = result.merge(away_squad, left_on="away_team_name", right_index=True, how="left")

    # Create differential features
    for col in squad_df.columns:
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        if home_col in result.columns and away_col in result.columns:
            result[f"squad_diff_{col}"] = result[home_col] - result[away_col]

    return result


# ── 4. Training & Evaluation ───────────────────────────────────────
def fill_na(X: np.ndarray) -> np.ndarray:
    """Fill NaN with column mean."""
    X = X.copy().astype(float)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X


def temporal_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """Split by time (chronological), not random."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test, split_idx


def train_and_eval(X_train, y_train, X_test, y_test, label: str = ""):
    """Train LR and return metrics."""
    # Fill NaN
    X_train_f = fill_na(X_train)
    X_test_f = fill_na(X_test)

    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_f)
    X_test_s = scaler.transform(X_test_f)

    # Train LR
    lr = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)

    y_pred = lr.predict(X_test_s)
    y_prob = lr.predict_proba(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    ll = log_loss(y_test, y_prob)

    print(f"\n  [{label}] Results on {len(y_test)} test matches:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    F1 (weighted): {f1:.4f}")
    print(f"    Log-loss: {ll:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Away", "Draw", "Home"]))

    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "log_loss": ll,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "model": lr,
        "scaler": scaler,
    }


# ── 5. Main experiment ────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Experiment: Player-Level Squad Features")
    print("  Dataset: 2018/19 Premier League (footystats.org)")
    print("=" * 65)

    # ── Load ──
    print("\n[1/5] Loading match data...")
    matches = load_match_data(MATCH_FILE)

    print("\n[2/5] Loading player data...")
    players = load_player_data(PLAYER_FILE)

    # ── Baseline rolling features ──
    print("\n[3/5] Computing rolling features (baseline)...")
    matches_feat = compute_rolling_features(matches, windows=[5, 10])
    rolling_cols = get_rolling_feature_cols(matches_feat)
    print(f"  Rolling feature columns: {len(rolling_cols)}")

    # ── Player-derived squad features ──
    print("\n[4/5] Building squad-level player features...")
    squad_feat = build_squad_features(players)
    matches_full = merge_squad_features(matches_feat, squad_feat)

    # Identify squad feature columns
    squad_cols = [c for c in matches_full.columns if c.startswith(("home_n_players", "away_n_players")) or
                  c.startswith("squad_diff_") or c.startswith("home_avg_age") or c.startswith("away_avg_age") or
                  c.startswith("home_total_goals") or c.startswith("away_total_goals") or
                  c.startswith("home_forward_goals") or c.startswith("away_forward_goals") or
                  c.startswith("home_midfielder_goals") or c.startswith("away_midfielder_goals") or
                  c.startswith("home_team_goals_per90") or c.startswith("away_team_goals_per90") or
                  c.startswith("home_team_conceded_per90") or c.startswith("away_team_conceded_per90") or
                  c.startswith("home_clean_sheets") or c.startswith("away_clean_sheets") or
                  c.startswith("home_top_scorer") or c.startswith("away_top_scorer") or
                  c.startswith("home_avg_attacker_rank") or c.startswith("away_avg_attacker_rank") or
                  c.startswith("home_avg_defender_rank") or c.startswith("away_avg_defender_rank") or
                  c.startswith("home_avg_midfielder_rank") or c.startswith("away_avg_midfielder_rank") or
                  c.startswith("home_goal_concentration") or c.startswith("away_goal_concentration") or
                  c.startswith("home_min_concentration") or c.startswith("away_min_concentration") or
                  c.startswith("home_n_regulars") or c.startswith("away_n_regulars") or
                  c.startswith("home_regular_ratio") or c.startswith("away_regular_ratio") or
                  c.startswith("home_avg_min_per_match") or c.startswith("away_avg_min_per_match")]
    print(f"  Squad feature columns: {len(squad_cols)}")

    # ── Temporal split ──
    train, test, split_idx = temporal_split(matches_full, test_ratio=0.2)
    target = "target_result"

    y_train = train[target].values.astype(int)
    y_test = test[target].values.astype(int)

    print(f"\n[5/5] Training & evaluation...")
    print(f"  Train: {len(train)} matches (GW1-GW{int(train['Game Week'].max())})")
    print(f"  Test:  {len(test)} matches (GW{int(test['Game Week'].min())}-GW{int(test['Game Week'].max())})")

    # ── Baseline: rolling features only ──
    X_train_base = train[rolling_cols].values.astype(float)
    X_test_base = test[rolling_cols].values.astype(float)
    base_results = train_and_eval(X_train_base, y_train, X_test_base, y_test, label="BASELINE (rolling stats only)")

    # ── Squad features only ──
    X_train_squad = train[squad_cols].values.astype(float)
    X_test_squad = test[squad_cols].values.astype(float)
    squad_only_results = train_and_eval(X_train_squad, y_train, X_test_squad, y_test, label="SQUAD ONLY (player agg)") if len(squad_cols) > 0 else None

    # ── Combined: rolling + squad ──
    combined_cols = rolling_cols + squad_cols
    X_train_comb = train[combined_cols].values.astype(float)
    X_test_comb = test[combined_cols].values.astype(float)
    comb_results = train_and_eval(X_train_comb, y_train, X_test_comb, y_test, label="COMBINED (rolling + squad)")

    # ── Rolling + pre-match xG (if available) ──
    xg_cols = []
    if "Home Team Pre-Match xG" in train.columns and "Away Team Pre-Match xG" in train.columns:
        # Only use rows where pre-match xG > 0
        xg_cols = ["Home Team Pre-Match xG", "Away Team Pre-Match xG"]
        xg_rolling_cols = rolling_cols + xg_cols
        X_train_xg = train[xg_rolling_cols].values.astype(float)
        X_test_xg = test[xg_rolling_cols].values.astype(float)
        xg_results = train_and_eval(X_train_xg, y_train, X_test_xg, y_test, label="BASELINE + Pre-Match xG")

    # ── Rolling + squad + pre-match xG ──
    if len(xg_cols) > 0:
        all_cols = rolling_cols + squad_cols + xg_cols
        X_train_all = train[all_cols].values.astype(float)
        X_test_all = test[all_cols].values.astype(float)
        all_results = train_and_eval(X_train_all, y_train, X_test_all, y_test, label="ALL (rolling + squad + xG)")
    else:
        all_results = comb_results

    # ── Summary ──
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  {'Feature Set':<35} {'Acc':<8} {'F1':<8} {'LogLoss':<8} {'Feat':<6}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    summary = [
        ("Rolling stats only", base_results),
        ("Squad features only", squad_only_results) if squad_only_results else None,
        ("Rolling + Squad", comb_results),
    ]
    if len(xg_cols) > 0:
        summary.append(("Rolling + xG", xg_results))
        summary.append(("All combined", all_results))

    for name, res in summary:
        if res is not None:
            n_feat = X_train_base.shape[1] if name == "Rolling stats only" else (
                X_train_squad.shape[1] if name == "Squad features only" else (
                    X_train_comb.shape[1] if name == "Rolling + Squad" else (
                        X_train_all.shape[1] if name == "All combined" else len(rolling_cols)
                    )
                )
            )
            print(f"  {name:<35} {res['accuracy']:<8.4f} {res['f1_weighted']:<8.4f} {res['log_loss']:<8.4f} {len(combined_cols) if 'Combined' in name or name == 'Rolling + Squad' else (len(squad_cols) if 'Squad' in name and 'Only' in name else (len(xg_cols) if 'xG' in name and 'Rolling' in name else (len(all_cols) if 'All' in name else n_feat))):<6}")

    # ── Save predictions ──
    test_out = test[["date", "home_team_name", "away_team_name",
                     "home_team_goal_count", "away_team_goal_count"]].copy()
    test_out["actual"] = y_test
    test_out["baseline_pred"] = base_results["y_pred"]
    test_out["combined_pred"] = comb_results["y_pred"]
    test_out["baseline_correct"] = (y_test == base_results["y_pred"]).astype(int)
    test_out["combined_correct"] = (y_test == comb_results["y_pred"]).astype(int)

    pred_file = OUTPUT_DIR / "predictions_comparison.csv"
    test_out.to_csv(pred_file, index=False)
    print(f"\n  Predictions saved: {pred_file}")

    # ── Feature importance (for combined model) ──
    if comb_results:
        print("\n  Top 15 feature coefficients (combined model):")
        coef = comb_results["model"].coef_[0]
        feat_names = combined_cols
        importance = pd.DataFrame({"feature": feat_names, "coef": np.abs(coef)})
        importance = importance.sort_values("coef", ascending=False).head(15)
        for _, row in importance.iterrows():
            direction = "+" if coef[list(feat_names).index(row["feature"])] > 0 else "-"
            print(f"    {direction} {row['feature']:<40} {row['coef']:.4f}")

    # ── Conclusion ──
    print("\n" + "=" * 65)
    delta = comb_results["accuracy"] - base_results["accuracy"]
    print(f"  Δ Accuracy (combined - baseline): {delta:+.4f}")
    if delta > 0:
        print(f"  ✅ Player-level squad features IMPROVE prediction by {delta*100:.1f}%")
    else:
        print(f"  X Player-level squad features did not help on this dataset")
    print("=" * 65)


if __name__ == "__main__":
    main()
