#!/usr/bin/env python3
"""
Buchdahl-style walk-forward: PL only, WITH FootyStats, 58 features.
====================================================================
Methodology aligned with Buchdahl (2003):
  - Rating system → fair odds → compare with bookmaker odds → value bet
  - But we use LR + 58 features instead of simple goal supremacy

Walk-forward across 6 PL seasons (2019-2025):
  Train on seasons [1..N], test on season N+1
  Report accuracy + value betting ROI per season

Model vs market: per-season + aggregate closing-odds (B365C) favorite
  accuracy with McNemar significance. Market = benchmark only, never a
  feature. Betting odds also use closing (B365C), not opening.

Two betting strategies for comparison:
  A) Buchdahl-style: bet home win ONLY when edge > threshold
  B) All-outcomes: bet ANY outcome with edge > threshold (Reade-style)

Usage:
    python scripts/buchdahl_ppg_walkforward.py
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

OUT = PROJ / "outputs" / "buchdahl_walkforward"
OUT.mkdir(parents=True, exist_ok=True)

# ── 1. Load PL data with FootyStats ──
print("Loading PL data with FootyStats...")

SEASONS_MAP = {
    "2019-20": "PL1920.csv", "2020-21": "PL2021.csv",
    "2021-22": "PL2122.csv", "2022-23": "PL2223.csv",
    "2023-24": "PL2324.csv", "2024-25": "PL2425.csv",
}

# Build a mapping from football-data.co.uk file to season number
FILE_TO_SEASON = {
    "PL1920.csv": 2020, "PL2021.csv": 2021, "PL2122.csv": 2022,
    "PL2223.csv": 2023, "PL2324.csv": 2024, "PL2425.csv": 2025,
}

frames = []
for season, fname in SEASONS_MAP.items():
    path = PROJ / "league/data/raw" / fname
    if not path.exists():
        continue
    df = load_data(path)
    df["season_num"] = FILE_TO_SEASON[fname]
    df["season_label"] = season
    df["season"] = season  # FootyStats merge requires a "season" column
    frames.append(df)

full = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
# 市场基线统一为收盘价（Bet365 B365C）——投注赔率用收盘，不用开盘
full["odds_h"] = full["odds_home_close"].fillna(0).astype(float)
full["odds_d"] = full["odds_draw_close"].fillna(0).astype(float)
full["odds_a"] = full["odds_away_close"].fillna(0).astype(float)
full["target_result"] = full["target_result"].fillna(-1).astype(int)

print(f"  Total: {len(full)} matches")
for s in sorted(full["season_num"].unique()):
    n = (full["season_num"] == s).sum()
    print(f"    {s}-{str(s+1)[-2:]}: {n} matches")

# ── 2. Build features WITH FootyStats ──
print("\nBuilding features (71-dim with FootyStats)...")
df_feat = build_features(full, windows=[5, 10], include_shots=True,
                          include_h2h=True, include_footystats=True,
                          include_odds=False, footystats_season_col="season")

feat_cols = get_feature_columns(df_feat)
exclude_meta = {"date", "season_num", "season_label", "season", "target_result", "league"}
feat_cols = [c for c in feat_cols if c not in exclude_meta and c in df_feat.columns]
# Add back metadata we need
df_feat["season_num"] = full["season_num"].values
df_feat["season_label"] = full["season_label"].values
df_feat["target_result"] = full["target_result"].values

print(f"  Features: {len(feat_cols)}")

# ── 3. Walk-forward ──
print("\n" + "=" * 70)
print("WALK-FORWARD: PL + FOOTYSTATS")
print("=" * 70)

test_seasons = sorted(full["season_num"].unique())
all_records = []  # all bet records for analysis

# Accumulators for walk-forward model vs closing-market comparison
wf_model_pred = []
wf_market_pred = []
wf_y = []

for test_s in test_seasons:
    train_mask = full["season_num"] < test_s
    test_mask = full["season_num"] == test_s
    n_train, n_test = train_mask.sum(), test_mask.sum()
    if n_train < 100 or n_test < 50:
        print(f"  Season {test_s}: SKIP (n_train={n_train})")
        continue

    X_tr = fill_features(df_feat.loc[train_mask, feat_cols].values.astype(float))
    X_te = fill_features(df_feat.loc[test_mask, feat_cols].values.astype(float))
    # Fill any remaining NaN (e.g., all-NaN columns) with 0
    X_tr = np.nan_to_num(X_tr, nan=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0)
    y_tr = df_feat.loc[train_mask, "target_result"].values.astype(int)
    y_te = df_feat.loc[test_mask, "target_result"].values.astype(int)

    ss = StandardScaler()
    lr = LogisticRegression(C=1.0, max_iter=4000, random_state=42)
    lr.fit(ss.fit_transform(X_tr), y_tr)
    te_prob = lr.predict_proba(ss.transform(X_te))
    te_pred = np.argmax(te_prob, axis=1)
    acc = accuracy_score(y_te, te_pred)

    # Market (closing B365C) favorite accuracy for this test season
    # Same convention as trackb_clean_eval: de-vig + argmax, home-first tie-break
    mo = full.loc[test_mask, ["odds_home_close", "odds_draw_close", "odds_away_close"]].values.astype(float)
    mo = np.nan_to_num(mo, nan=0.0)
    valid = (mo > 1.01).all(axis=1)
    imp = np.zeros_like(mo)
    imp[valid] = 1.0 / mo[valid]
    imp[valid] = imp[valid] / imp[valid].sum(axis=1, keepdims=True)
    mp = np.zeros(len(mo), dtype=int)
    mp[valid] = imp[valid].argmax(axis=1)  # 0=home,1=draw,2=away
    macc = (((mp == 2) & (y_te == 0)) | ((mp == 1) & (y_te == 1)) | ((mp == 0) & (y_te == 2))).mean()
    if valid.sum() < len(mo):
        print(f"    [warn] {len(mo)-valid.sum()} test matches missing closing odds (excluded from market acc)")

    wf_model_pred.append(te_pred)
    wf_market_pred.append(mp)
    wf_y.append(y_te)

    # Accuracy by class
    home_acc = (te_pred[y_te == 2] == 2).mean() if (y_te == 2).sum() > 0 else 0
    draw_acc = (te_pred[y_te == 1] == 1).mean() if (y_te == 1).sum() > 0 else 0
    away_acc = (te_pred[y_te == 0] == 0).mean() if (y_te == 0).sum() > 0 else 0

    # ── Value betting for this season ──
    test_orig = full.loc[test_mask].copy()
    season_records = []

    for i in range(len(test_orig)):
        row = test_orig.iloc[i]
        odds = np.array([row["odds_a"], row["odds_d"], row["odds_h"]], dtype=float)
        if (odds <= 0).any():
            continue

        # Market implied probabilities
        m_probs = (1.0 / odds)
        m_probs = m_probs / m_probs.sum()

        # For each outcome, compute edge
        for outcome in range(3):
            model_p = te_prob[i][outcome]
            market_p = m_probs[outcome]
            edge = model_p / market_p - 1 if market_p > 0 else 0

            season_records.append({
                "season": int(test_s),
                "home": row["home_team"],
                "away": row["away_team"],
                "outcome": outcome,  # 0=Away, 1=Draw, 2=Home
                "actual": int(y_te[i]),
                "model_prob": round(float(model_p), 4),
                "market_prob": round(float(market_p), 4),
                "edge": round(float(edge), 4),
                "payout_odds": round(float(odds[outcome]), 2),
                "correct": int(outcome == y_te[i]),
            })

    season_df = pd.DataFrame(season_records)
    all_records.append(season_df)

    # --- Strategy A: Buchdahl-style (home win only) ---
    home_value = season_df[(season_df["outcome"] == 2) & (season_df["edge"] > 0.10)]
    n_bets_h = len(home_value)
    profit_h = sum(home_value.apply(lambda r: r["payout_odds"] - 1 if r["correct"] else -1, axis=1))
    roi_h = profit_h / n_bets_h * 100 if n_bets_h > 0 else 0

    # --- Strategy B: All outcomes (Reade-style, multiple bets per match) ---
    all_value = season_df[season_df["edge"] > 0.10]
    n_bets_a = len(all_value)
    profit_a = sum(all_value.apply(lambda r: r["payout_odds"] - 1 if r["correct"] else -1, axis=1))
    roi_a = profit_a / n_bets_a * 100 if n_bets_a > 0 else 0

    # --- Strategy C: One best per match ---
    best_per_match = all_value.loc[all_value.groupby(["home", "away"])["edge"].idxmax()] if len(all_value) > 0 else pd.DataFrame()
    n_bests = len(best_per_match)
    profit_c = sum(best_per_match.apply(lambda r: r["payout_odds"] - 1 if r["correct"] else -1, axis=1)) if n_bests > 0 else 0
    roi_c = profit_c / n_bests * 100 if n_bests > 0 else 0

    s_label = f"{test_s-1}-{str(test_s)[-2:]}"
    print(f"\n  {s_label}")
    print(f"    Acc: {acc:.4f} (H={home_acc:.0%} D={draw_acc:.0%} A={away_acc:.0%})")
    print(f"    Market (closing):    {macc:.4f}  delta {acc-macc:+.4f}")
    print(f"    Buchdahl (Home only): {n_bets_h:>3d} bets, ROI {roi_h:>+7.2f}%")
    print(f"    All outcomes:         {n_bets_a:>3d} bets, ROI {roi_a:>+7.2f}%")
    print(f"    Best per match:       {n_bests:>3d} bets, ROI {roi_c:>+7.2f}%")

# ── 3b. Model vs closing market (walk-forward aggregate) ──
print("\n" + "=" * 70)
print("MODEL vs CLOSING MARKET (walk-forward aggregate)")
print("=" * 70)
pred_m = np.concatenate(wf_model_pred)      # y encoding: 0=away,1=draw,2=home
pred_k = np.concatenate(wf_market_pred)     # argmax encoding: 0=home,1=draw,2=away
y_all = np.concatenate(wf_y)                # 0=away,1=draw,2=home
pred_k_y = np.where(pred_k == 0, 2, np.where(pred_k == 2, 0, pred_k))  # map to y encoding
acc_m = accuracy_score(y_all, pred_m)
acc_k = accuracy_score(y_all, pred_k_y)
c1 = (pred_m == y_all)
c2 = (pred_k_y == y_all)
b = int(np.sum(c1 & ~c2))
c = int(np.sum(~c1 & c2))
n_disc = b + c
if n_disc > 0:
    from scipy.stats import binomtest
    pv = binomtest(b, n_disc, 0.5).pvalue
else:
    pv = 1.0
print(f"  Model:  {acc_m*100:.2f}%  ({len(y_all)} walk-forward test matches)")
print(f"  Market: {acc_k*100:.2f}%  (closing B365C)")
print(f"  Delta:  {(acc_m-acc_k)*100:+.2f}pp")
print(f"  McNemar: b(model-only-correct)={b} c(market-only-correct)={c} p={pv:.4f}")
if pv < 0.05:
    print(f"  Verdict: {'BEAT closing market' if acc_m > acc_k else 'LOST to closing market'}")
else:
    print(f"  Verdict: TIE with closing market (n.s.)")

# ── 4. Aggregate ──
print("\n" + "=" * 70)
print("AGGREGATE RESULTS")
print("=" * 70)

all_df = pd.concat(all_records, ignore_index=True)
print(f"\n  Total records: {len(all_df)} ({len(all_df) // 3} matches)")

for strategy_name, outcome_filter in [
    ("Buchdahl (Home only, edge>10%)", 2),
    ("Away only (edge>10%)", 0),
    ("Draw only (edge>10%)", 1),
]:
    value = all_df[(all_df["outcome"] == outcome_filter) & (all_df["edge"] > 0.10)]
    if len(value) == 0:
        continue
    profit = sum(value.apply(lambda r: r["payout_odds"] - 1 if r["correct"] else -1, axis=1))
    roi = profit / len(value) * 100
    wins = value["correct"].sum()
    print(f"\n  {strategy_name:<35}")
    print(f"    Bets: {len(value):>4d}, Wins: {wins:>3d} ({wins/len(value)*100:.1f}%), "
          f"Profit: {profit:>+8.2f}, ROI: {roi:>+7.2f}%")

for strategy_name, edge_thresh in [
    ("All outcomes (edge>5%)", 0.05),
    ("All outcomes (edge>10%)", 0.10),
    ("All outcomes (edge>15%)", 0.15),
    ("All outcomes (edge>20%)", 0.20),
]:
    value = all_df[all_df["edge"] > edge_thresh]
    if len(value) == 0:
        continue
    profit = sum(value.apply(lambda r: r["payout_odds"] - 1 if r["correct"] else -1, axis=1))
    roi = profit / len(value) * 100
    wins = value["correct"].sum()
    n_matches = value.groupby(["home", "away"]).ngroups
    print(f"\n  {strategy_name:<35}")
    print(f"    Bets: {len(value):>4d} on {n_matches:>3d} matches, Wins: {wins:>3d} ({wins/len(value)*100:.1f}%), "
          f"Profit: {profit:>+8.2f}, ROI: {roi:>+7.2f}%")

# Best per match
for edge_thresh in [0.05, 0.10, 0.15, 0.20]:
    value = all_df[all_df["edge"] > edge_thresh]
    if len(value) == 0:
        continue
    best = value.loc[value.groupby(["home", "away"])["edge"].idxmax()]
    profit = sum(best.apply(lambda r: r["payout_odds"] - 1 if r["correct"] else -1, axis=1))
    roi = profit / len(best) * 100
    wins = best["correct"].sum()
    print(f"\n  Best per match (edge>{edge_thresh:.0%}):{' ':>14}")
    print(f"    Bets: {len(best):>4d}, Wins: {wins:>3d} ({wins/len(best)*100:.1f}%), "
          f"Profit: {profit:>+8.2f}, ROI: {roi:>+7.2f}%")

# ── 5. Save ──
all_df.to_csv(OUT / "all_bets.csv", index=False)

summary = {
    "total_matches": len(all_df) // 3,
    "seasons": len(test_seasons),
    "strategy_buchdahl_home": {
        "bets": int(((all_df["outcome"] == 2) & (all_df["edge"] > 0.10)).sum()),
    },
    "walkforward_vs_market": {
        "n": int(len(y_all)),
        "model_acc": round(float(acc_m), 4),
        "market_acc_closing": round(float(acc_k), 4),
        "delta_pp": round(float((acc_m - acc_k) * 100), 2),
        "mcnemar_p": round(float(pv), 4),
    },
}
# Add more detail to summary
for label, out_filter, edge_th in [
    ("buchdahl_home", 2, 0.10),
    ("all_outcomes_10", None, 0.10),
    ("best_per_match_10", None, 0.10),
]:
    if out_filter is not None:
        value = all_df[(all_df["outcome"] == out_filter) & (all_df["edge"] > edge_th)]
    else:
        value = all_df[all_df["edge"] > edge_th]
        if label.startswith("best"):
            value = value.loc[value.groupby(["home", "away"])["edge"].idxmax()]

    if len(value) == 0:
        continue
    profit = float(sum(value.apply(lambda r: r["payout_odds"] - 1 if r["correct"] else -1, axis=1)))
    summary[label] = {
        "bets": len(value),
        "wins": int(value["correct"].sum()),
        "win_rate": round(float(value["correct"].mean()), 4),
        "profit": round(profit, 2),
        "roi": round(profit / len(value) * 100, 2),
    }

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Results saved to {OUT}/")
print("  Done!")
