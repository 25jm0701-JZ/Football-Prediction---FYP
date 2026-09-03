#!/usr/bin/env python3
"""
Walk-forward value betting backtest across all English league levels (E0-E3).
====================================================================
For each season in 2020-21 to 2025-26, trains on ALL previous seasons
and evaluates betting performance on the current season.

Outputs per-season and aggregate ROI, plus comparison across divisions.

Usage:
    python scripts/value_betting_walkforward.py
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "league"))
from src.data_loader import load_data
from src.features import build_features, get_feature_columns
from src.models import fill_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

OUT = PROJ / "outputs" / "value_betting"
OUT.mkdir(parents=True, exist_ok=True)

# ── Config ──
LEAGUES = {"E0": "Premier League", "E1": "Championship", "E2": "League One", "E3": "League Two"}
ALL_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]  # 2019-20 to 2025-26

def load_league_data(league_code):
    """Load all seasons for one league, return sorted DataFrame with season labels."""
    frames = []
    for s in ALL_SEASONS:
        path = PROJ / "league/data/english_raw" / f"{league_code}{s}.csv"
        if not path.exists():
            continue
        df = load_data(path)
        # season label for temporal ordering
        year = int(s[:2]) + 2000
        df["season_num"] = year if s[2:] == "20" else year + 1  # 1920 → 2020, 2021 → 2021
        df["season_label"] = f"{year}-{s[2:]}"
        df["league_code"] = league_code
        frames.append(df)
    full = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    return full

def build_features_for_model(df):
    """Build rolling features. No FootyStats, no odds."""
    df_feat = build_features(df, windows=[5, 10], include_shots=True,
                              include_h2h=True, include_footystats=False, include_odds=False)
    df_feat["league_level"] = df["league_code"].map({"E0": 0, "E1": 1, "E2": 2, "E3": 3})
    feat_cols = get_feature_columns(df_feat) + ["league_level"]
    # Exclude non-numeric columns that leak in from the original df
    exclude_extra = {"season_label", "season_num", "league_code", "season"}
    feat_cols = [c for c in feat_cols if c not in exclude_extra and c in df_feat.columns]
    return df_feat, feat_cols

# ── Collect all data ──
print("Loading data...")
all_dfs = []
for lcode, lname in LEAGUES.items():
    df = load_league_data(lcode)
    all_dfs.append(df)
    print(f"  {lname} ({lcode}): {len(df)} matches ({df['season_num'].nunique()} seasons)")

full = pd.concat(all_dfs, ignore_index=True).sort_values("date").reset_index(drop=True)
full["season_num"] = full["season_num"].astype(int)
print(f"\nTotal: {len(full)} matches")

# Build features once
print("Building features...")
df_feat, feat_cols = build_features_for_model(full)
# Match df_feat index to full (build_features resets index, but row order is same)
df_feat.index = full.index
print(f"Features: {len(feat_cols)}")

# ── Walk-forward backtest ──
# For each season, train on all previous seasons, test on current
test_seasons = sorted(full["season_num"].unique())
# Need at least 2 seasons of training data
train_seasons_pool = [s for s in test_seasons if s < 2026]  # up to 2024-25

print(f"\nWalk-forward seasons: {len(train_seasons_pool)} training pools → {len(test_seasons)} test seasons")
print()

all_bets = []  # Collect every individual bet
results = []

for test_s in sorted(test_seasons):
    # Train: all seasons BEFORE test_s
    train_mask = full["season_num"] < test_s
    test_mask = full["season_num"] == test_s

    n_train = train_mask.sum()
    n_test = test_mask.sum()

    if n_train < 500 or n_test < 50:
        continue

    X_tr = fill_features(df_feat[train_mask][feat_cols].values.astype(float))
    X_te = fill_features(df_feat[test_mask][feat_cols].values.astype(float))
    y_tr = df_feat[train_mask]["target_result"].values.astype(int)
    y_te = df_feat[test_mask]["target_result"].values.astype(int)

    # Scale & train
    ss = StandardScaler()
    X_tr_s = ss.fit_transform(X_tr)
    X_te_s = ss.transform(X_te)

    lr = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                             max_iter=4000, random_state=42)
    lr.fit(X_tr_s, y_tr)
    te_prob = lr.predict_proba(X_te_s)
    te_pred = np.argmax(te_prob, axis=1)

    # Accuracy
    acc = accuracy_score(y_te, te_pred)

    # ── Value betting on this season ──
    season_bets = []
    n_matches_checked = 0
    for row_pos, idx in enumerate(full[test_mask].index):
        n_matches_checked += 1
        # Get market odds from the original data
        orig_row = full.loc[idx]
        try:
            # 市场基线统一为收盘价（Bet365 B365C）——投注赔率用收盘，不用开盘
            odds_h = float(orig_row["odds_home_close"]) if pd.notna(orig_row.get("odds_home_close")) else 0
            odds_d = float(orig_row["odds_draw_close"]) if pd.notna(orig_row.get("odds_draw_close")) else 0
            odds_a = float(orig_row["odds_away_close"]) if pd.notna(orig_row.get("odds_away_close")) else 0
        except (ValueError, TypeError):
            continue
        if odds_h <= 0 or odds_d <= 0 or odds_a <= 0:
            continue

        # Market implied probabilities (de-ordered)
        raw_odds = np.array([odds_a, odds_d, odds_h], dtype=float)
        m_probs = (1.0 / raw_odds)
        m_probs = m_probs / m_probs.sum()

        # Model probabilities for this match (use position within test set)
        p_model = te_prob[row_pos]

        # For each outcome
        outcome_map = {0: "Away", 1: "Draw", 2: "Home"}
        for outcome in range(3):
            model_p = p_model[outcome]
            market_p = m_probs[outcome]
            edge = model_p / market_p - 1 if market_p > 0 else 0
            payout_odds = raw_odds[outcome]

            bet_rec = {
                "season": int(test_s),
                "league": orig_row.get("league_code", "?"),
                "home": orig_row["home_team"],
                "away": orig_row["away_team"],
                "outcome": outcome,
                "outcome_label": outcome_map[outcome],
                "model_prob": round(float(model_p), 4),
                "market_prob": round(float(market_p), 4),
                "edge": round(float(edge), 4),
                "payout_odds": round(float(payout_odds), 2),
                "actual": int(orig_row.get("target_result", -1)),
            }
            all_bets.append(bet_rec)
            season_bets.append(bet_rec)

    season_bets = pd.DataFrame(season_bets)
    if len(season_bets) == 0:
        continue

    # Simulate: bet on highest-edge > 10% per match
    season_value = season_bets[season_bets["edge"] > 0.10].copy()
    if len(season_value) > 0:
        # One bet per match (highest edge)
        best_per_match = season_value.loc[season_value.groupby(["season", "home", "away"])["edge"].idxmax()]
        n_bets = len(best_per_match)
        wins = (best_per_match["outcome"] == best_per_match["actual"]).sum()
        profit = sum(best_per_match.apply(
            lambda r: r["payout_odds"] - 1 if r["outcome"] == r["actual"] else -1, axis=1))
        roi = profit / n_bets * 100 if n_bets > 0 else 0
    else:
        n_bets = wins = profit = roi = 0

    results.append({
        "season": int(test_s),
        "n_train": n_train,
        "n_test": n_test,
        "accuracy": round(float(acc), 4),
        "value_bets": n_bets,
        "wins": wins,
        "win_rate": round(wins / n_bets * 100, 1) if n_bets > 0 else 0,
        "profit": round(float(profit), 2),
        "roi": round(float(roi), 2),
    })

    season_str = f"{test_s-1}-{str(test_s)[-2:]}"
    print(f"  Season {season_str:<9}: train={n_train:>5d} test={n_test:>5d} "
          f"acc={acc:.4f} bets={n_bets:>4d} roi={roi:>+7.2f}%")

# ── Aggregate results ──
print("\n" + "=" * 70)
print("  WALK-FORWARD VALUE BETTING — AGGREGATE")
print("=" * 70)

all_bets_df = pd.DataFrame(all_bets)
value_bets = all_bets_df[all_bets_df["edge"] > 0.10].copy()
best_per_match = value_bets.loc[value_bets.groupby(["season", "home", "away"])["edge"].idxmax()].copy()

total_bets = len(best_per_match)
total_wins = (best_per_match["outcome"] == best_per_match["actual"]).sum()
total_profit = sum(best_per_match.apply(
    lambda r: r["payout_odds"] - 1 if r["outcome"] == r["actual"] else -1, axis=1))
total_roi = total_profit / total_bets * 100

print(f"\n  Total bets:          {total_bets}")
print(f"  Wins:                {total_wins} ({total_wins/total_bets*100:.1f}%)")
print(f"  Total profit:        {total_profit:+.2f}")
print(f"  Overall ROI:         {total_roi:+.2f}%")
print(f"\n  Test seasons:        {len(results)}")
print(f"  Total matches tested: {sum(r['n_test'] for r in results)}")

# By league
print(f"\n  {'='*50}")
print(f"  {'League':<20} {'Bets':<8} {'WinRate':<10} {'ROI':<10}")
print(f"  {'-'*50}")
for lcode, lname in LEAGUES.items():
    mask = best_per_match["league"] == lcode
    n = mask.sum()
    if n < 10:
        continue
    w = (best_per_match.loc[mask, "outcome"] == best_per_match.loc[mask, "actual"]).sum()
    p = sum(best_per_match.loc[mask].apply(
        lambda r: r["payout_odds"] - 1 if r["outcome"] == r["actual"] else -1, axis=1))
    print(f"  {lname:<20} {n:<8} {w/n*100:<10.1f} {p/n*100:<+10.2f}%")
print(f"  {'-'*50}")

# By season
print(f"\n  {'='*50}")
print(f"  {'Season':<12} {'Bets':<8} {'Accuracy':<12} {'ROI':<12}")
print(f"  {'-'*50}")
for r in sorted(results, key=lambda x: x["season"]):
    s_label = f"{r['season']-1}-{str(r['season'])[-2:]}"
    print(f"  {s_label:<12} {r['value_bets']:<8} {r['accuracy']:<12.4f} {r['roi']:<+12.2f}%")
# Add note about train_seasons
print(f"  {'='*50}")
print(f"  * Each row: test season predicted by model trained on ALL prior seasons")

# ── Statistical significance ──
print(f"\n  {'='*50}")
print(f"  STATISTICAL SIGNIFICANCE")
print(f"  {'='*50}")

# Simple binomial test: is win rate > 1/3? (since 3 outcomes)
from scipy.stats import binomtest
p_value = binomtest(total_wins, total_bets, 1/3, alternative='greater').pvalue
print(f"  Win rate vs random (33.3%): {total_wins}/{total_bets} = {total_wins/total_bets*100:.1f}%")
print(f"  Binomial p-value:           {p_value:.6f}")
print(f"  Statistically significant:  {'YES (p<0.001)' if p_value < 0.001 else 'YES (p<0.05)' if p_value < 0.05 else 'NO'}")

# Also test if profit > 0
# Simple: proportion of positive-ROI seasons
positive_seasons = sum(1 for r in results if r["roi"] > 0)
print(f"  Positive ROI seasons: {positive_seasons}/{len(results)} ({positive_seasons/len(results)*100:.0f}%)")

# ── Save ──
best_per_match.to_csv(OUT / "all_value_bets.csv", index=False)
pd.DataFrame(results).to_csv(OUT / "by_season_results.csv", index=False)

# Summary
summary = {
    "total_bets": int(total_bets),
    "wins": int(total_wins),
    "win_rate": round(float(total_wins / total_bets), 4),
    "profit": round(float(total_profit), 2),
    "roi": round(float(total_roi), 2),
    "n_seasons": len(results),
    "n_matches": int(sum(r["n_test"] for r in results)),
    "binomial_p": round(float(p_value), 6),
    "positive_season_ratio": f"{positive_seasons}/{len(results)}",
}
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Results saved to: {OUT}/")
print("  Done!")
