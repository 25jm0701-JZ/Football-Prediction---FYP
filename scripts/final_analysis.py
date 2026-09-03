#!/usr/bin/env python3
"""
Final analysis: walk-forward backtest with self-computed PPG + value betting.
======================================================================
- Computes PPG from ALL historical match data (no FootyStats API needed)
- Walk-forward across E0-E3, 7 seasons
- Reports accuracy + ROI per season, per league, and aggregate
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
from sklearn.metrics import accuracy_score

OUT = PROJ / "outputs" / "final_analysis"
OUT.mkdir(parents=True, exist_ok=True)

# ── 1. Load all English league data ──
print("Loading data...")
LEAGUES = {"E0": "Premier League", "E1": "Championship", "E2": "League One", "E3": "League Two"}
SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]

frames = []
for lcode in LEAGUES:
    for s in SEASONS:
        path = PROJ / "league/data/english_raw" / f"{lcode}{s}.csv"
        if not path.exists():
            continue
        df = load_data(path)
        year = int(s[:2]) + 2000
        df["league"] = lcode
        df["season_num"] = year if s[2:] == "20" else year + 1
        frames.append(df)

full = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
full["target_result"] = full["target_result"].fillna(-1).astype(int)
full["odds_h"] = full["odds_home"].fillna(0)
full["odds_d"] = full["odds_draw"].fillna(0)
full["odds_a"] = full["odds_away"].fillna(0)
print(f"  Total: {len(full)} matches")
for l in LEAGUES:
    print(f"    {l}: {(full['league']==l).sum()}")

# ── 2. Compute PPG for every team from all historical data ──
# PPG = total_points / total_matches (cumulative, no look-ahead)
# We compute it dynamically per match using only data BEFORE that match
print("\nComputing cumulative PPG...")
ppg_home = np.full(len(full), np.nan)
ppg_away = np.full(len(full), np.nan)
ppg_diff = np.full(len(full), np.nan)

team_points = {}
team_games = {}
for i in range(len(full)):
    r = full.iloc[i]
    home, away = r["home_team"], r["away_team"]
    ppg_home[i] = team_points.get(home, 0) / max(team_games.get(home, 1), 1)
    ppg_away[i] = team_points.get(away, 0) / max(team_games.get(away, 1), 1)
    ppg_diff[i] = ppg_home[i] - ppg_away[i]
    hg, ag = r["home_goals_full"], r["away_goals_full"]
    if pd.notna(hg) and pd.notna(ag):
        team_points.setdefault(home, 0)
        team_points.setdefault(away, 0)
        team_games.setdefault(home, 0)
        team_games.setdefault(away, 0)
        if hg > ag:
            team_points[home] += 3
        elif hg == ag:
            team_points[home] += 1
            team_points[away] += 1
        else:
            team_points[away] += 3
        team_games[home] += 1
        team_games[away] += 1

full["ppg_home"] = ppg_home
full["ppg_away"] = ppg_away
full["ppg_diff"] = ppg_diff

# ── 3. Build features (rolling stats + self-computed PPG) ──
print("Building features...")

# Base rolling features (no FootyStats)
df_feat = build_features(full, windows=[5, 10], include_shots=True,
                          include_h2h=True, include_footystats=False, include_odds=False)

# Get feature columns (exclude all non-numeric metadata)
feat_cols = get_feature_columns(df_feat)
extra_exclude = {"league", "season_num", "date"}
feat_cols = [c for c in feat_cols if c not in extra_exclude and c in df_feat.columns]

# Add self-computed PPG and season_num to df_feat for later use
df_feat["ppg_home"] = ppg_home
df_feat["ppg_away"] = ppg_away
df_feat["ppg_diff"] = ppg_diff
df_feat["league"] = full["league"].values
df_feat["season_num"] = full["season_num"].values
df_feat["target_result"] = full["target_result"].values

# Fill PPG NaN (first match for each team) with league-season mean
for l in LEAGUES:
    mask = df_feat["league"] == l
    mean_ppg = df_feat.loc[mask, "ppg_home"].mean()
    df_feat.loc[mask, "ppg_home"] = df_feat.loc[mask, "ppg_home"].fillna(mean_ppg)
    df_feat.loc[mask, "ppg_away"] = df_feat.loc[mask, "ppg_away"].fillna(mean_ppg)
df_feat["ppg_diff"] = df_feat["ppg_diff"].fillna(0)

feat_with_ppg = feat_cols + ["ppg_home", "ppg_away", "ppg_diff"]
feat_with_ppg = [c for c in feat_with_ppg if c in df_feat.columns]

# Also build baseline (no PPG) for comparison
feat_no_ppg = [c for c in feat_cols if c in df_feat.columns]

print(f"  Features without PPG: {len(feat_no_ppg)}")
print(f"  Features with PPG:    {len(feat_with_ppg)}")

# ── 4. Walk-forward backtest ──
print("\n" + "=" * 70)
print("WALK-FORWARD BACKTEST")
print("=" * 70)

test_seasons = sorted(full["season_num"].unique())

all_results = []

for cfg_name, cfg_feats, cfg_label in [
    ("no_ppg", feat_no_ppg, "No PPG"),
    ("with_ppg", feat_with_ppg, "With PPG"),
]:
    print(f"\n--- {cfg_label} ---")
    results_list = []

    for test_s in test_seasons:
        train_mask = df_feat["season_num"] < test_s
        test_mask = df_feat["season_num"] == test_s
        n_train, n_test = train_mask.sum(), test_mask.sum()
        if n_train < 500 or n_test < 50:
            continue

        X_tr = fill_features(df_feat.loc[train_mask, cfg_feats].values.astype(float))
        X_te = fill_features(df_feat.loc[test_mask, cfg_feats].values.astype(float))
        y_tr = df_feat.loc[train_mask, "target_result"].values.astype(int)
        y_te = df_feat.loc[test_mask, "target_result"].values.astype(int)

        ss = StandardScaler()
        lr = LogisticRegression(C=1.0, max_iter=4000, random_state=42)
        lr.fit(ss.fit_transform(X_tr), y_tr)
        te_prob = lr.predict_proba(ss.transform(X_te))
        te_pred = np.argmax(te_prob, axis=1)
        acc = accuracy_score(y_te, te_pred)

        # ---- Value betting ----
        test_df = df_feat.loc[test_mask].copy()
        test_orig = full.loc[test_mask].copy()
        bets = []
        for i in range(len(test_df)):
            odds = np.array([
                test_orig.iloc[i]["odds_a"],
                test_orig.iloc[i]["odds_d"],
                test_orig.iloc[i]["odds_h"],
            ], dtype=float)
            if (odds <= 0).any():
                continue
            m_probs = (1.0 / odds)
            m_probs = m_probs / m_probs.sum()
            for outcome in range(3):
                edge = te_prob[i][outcome] / m_probs[outcome] - 1
                bets.append({
                    "league": test_orig.iloc[i]["league"],
                    "home_team": test_orig.iloc[i]["home_team"],
                    "away_team": test_orig.iloc[i]["away_team"],
                    "outcome": outcome,
                    "edge": edge,
                    "payout": odds[outcome],
                    "actual": int(y_te[i]),
                })

        bets_df = pd.DataFrame(bets)
        value = bets_df[bets_df["edge"] > 0.10]
        if len(value) == 0:
            continue

        best = value.loc[value.groupby(["home_team", "away_team"])["edge"].idxmax()]
        wins = (best["outcome"] == best["actual"]).sum()
        profit = sum(best.apply(lambda r: r["payout"] - 1 if r["outcome"] == r["actual"] else -1, axis=1))
        roi = profit / len(best) * 100

        s_label = f"{test_s-1}-{str(test_s)[-2:]}"
        print(f"  {s_label:<9} train={n_train:>5d} test={n_test:>5d} acc={acc:.4f} bets={len(best):>4d} roi={roi:>+7.2f}%")

        results_list.append({
            "config": cfg_name, "season": test_s, "season_label": s_label,
            "n_train": n_train, "n_test": n_test, "accuracy": round(float(acc), 4),
            "n_bets": len(best), "n_wins": int(wins), "win_rate": round(wins / len(best), 4) if len(best) > 0 else 0,
            "profit": round(float(profit), 2), "roi": round(float(roi), 2),
        })

    all_results.extend(results_list)

    # Per-season summary for this config
    if results_list:
        total_bets = sum(r["n_bets"] for r in results_list)
        total_profit = sum(r["profit"] for r in results_list)
        print(f"  [{cfg_label}] All seasons: {total_bets} bets, profit={total_profit:.2f}, roi={total_profit/total_bets*100:.2f}%")

# ── 5. Detailed comparison table ──
print("\n" + "=" * 70)
print("COMPARISON: No PPG vs With PPG")
print("=" * 70)

for cfg_name, cfg_label in [("no_ppg", "No PPG"), ("with_ppg", "With PPG")]:
    r = [x for x in all_results if x["config"] == cfg_name]
    if not r:
        continue
    total_bets = sum(x["n_bets"] for x in r)
    total_profit = sum(x["profit"] for x in r)
    avg_acc = np.mean([x["accuracy"] for x in r])
    print(f"\n  {cfg_label}:")
    print(f"    Avg accuracy:   {avg_acc:.4f}")
    print(f"    Total bets:     {total_bets}")
    print(f"    Total profit:   {total_profit:.2f}")
    print(f"    ROI:            {total_profit/total_bets*100:.2f}%")
    for lcode, lname in LEAGUES.items():
        # Can't filter by league from aggregated results easily
        pass

# ── 6. By league breakdown (with PPG only) ──
print("\n" + "=" * 70)
print("BY LEAGUE (With PPG)")
print("=" * 70)

# Re-run just the value betting for per-league stats
cfg_feats = feat_with_ppg
league_bets = {l: [] for l in LEAGUES}

for test_s in test_seasons:
    train_mask = df_feat["season_num"] < test_s
    test_mask = df_feat["season_num"] == test_s
    if train_mask.sum() < 500 or test_mask.sum() < 50:
        continue

    X_tr = fill_features(df_feat.loc[train_mask, cfg_feats].values.astype(float))
    X_te = fill_features(df_feat.loc[test_mask, cfg_feats].values.astype(float))
    y_tr = df_feat.loc[train_mask, "target_result"].values.astype(int)
    y_te = df_feat.loc[test_mask, "target_result"].values.astype(int)

    ss = StandardScaler()
    lr = LogisticRegression(C=1.0, max_iter=4000, random_state=42)
    lr.fit(ss.fit_transform(X_tr), y_tr)
    te_prob = lr.predict_proba(ss.transform(X_te))

    test_orig = full.loc[test_mask].copy()
    for i, (idx, _) in enumerate(test_orig.iterrows()):
        odds = np.array([test_orig.iloc[i]["odds_a"], test_orig.iloc[i]["odds_d"], test_orig.iloc[i]["odds_h"]], dtype=float)
        if (odds <= 0).any():
            continue
        m_probs = (1.0 / odds)
        m_probs = m_probs / m_probs.sum()
        for outcome in range(3):
            edge = te_prob[i][outcome] / m_probs[outcome] - 1
            if edge > 0.10:
                league_bets[test_orig.iloc[i]["league"]].append({
                    "outcome": outcome, "actual": int(y_te[i]),
                    "payout": odds[outcome], "edge": edge,
                })

for lcode, lname in LEAGUES.items():
    bets = league_bets[lcode]
    if len(bets) == 0:
        continue
    # One bet per match (take highest edge per match)
    # Since we can't group by match easily, approximate with unique edge values
    # Better: just compute on all value outcomes
    wins = sum(1 for b in bets if b["outcome"] == b["actual"])
    profit = sum(b["payout"] - 1 if b["outcome"] == b["actual"] else -1 for b in bets)
    roi = profit / len(bets) * 100
    print(f"  {lname:<20} {lcode}: {len(bets):>4d} outcomes >10% edge, {wins:>3d} wins ({wins/len(bets)*100:.1f}%), roi={roi:>+7.2f}%")

# ── 7. Save ──
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT / "walkforward_results.csv", index=False)

summary = {
    "n_matches": len(full),
    "n_seasons": len(test_seasons),
    "features_no_ppg": len(feat_no_ppg),
    "features_with_ppg": len(feat_with_ppg),
}
for cfg_name in ["no_ppg", "with_ppg"]:
    r = [x for x in all_results if x["config"] == cfg_name]
    if r:
        summary[f"{cfg_name}_seasons"] = len(r)
        summary[f"{cfg_name}_avg_acc"] = float(np.mean([x["accuracy"] for x in r]))
        summary[f"{cfg_name}_total_bets"] = sum(x["n_bets"] for x in r)
        summary[f"{cfg_name}_total_profit"] = round(float(sum(x["profit"] for x in r)), 2)
        summary[f"{cfg_name}_roi"] = round(float(sum(x["profit"] for x in r) / max(sum(x["n_bets"] for x in r), 1) * 100), 2)

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {OUT}/")
