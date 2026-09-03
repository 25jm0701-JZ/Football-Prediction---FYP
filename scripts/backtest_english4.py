"""
English 4-league value betting backtest & comparison
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "league")
import numpy as np
import pandas as pd
from pathlib import Path
from src.data_loader import load_data
from src.features import build_features, get_feature_columns
from src.models import fill_features
from sklearn.linear_model import LogisticRegression

DIV_LEVEL = {"E0": 0, "E1": 1, "E2": 2, "E3": 3}

# ── 1. Load English 4-league data ──
print("Loading English 4-league data...")
all_frames = []
for div, level in DIV_LEVEL.items():
    files = sorted(Path("league/data/english_raw").glob(f"{div}*.csv"))
    for f in files:
        # Track A 统一口径 = 7 季（2019/20–2025/26，13,988 场），见 project-baseline
        df = load_data(f)
        df["league_level"] = level
        df["division"] = div
        all_frames.append(df)

hist = pd.concat(all_frames, ignore_index=True)
hist = hist.sort_values("date").reset_index(drop=True)

# Build features
hist_feat = build_features(hist, windows=[5,10], include_odds=False, include_shots=True,
                           include_h2h=True, include_footystats=False)
feat_cols = get_feature_columns(hist_feat)
if "league_level" not in hist_feat.columns:
    hist_feat["league_level"] = hist["league_level"].values
feat_cols = list(set(feat_cols + ["league_level"]))

# Also build with odds for comparison
hist_odds = build_features(hist, windows=[5,10], include_odds=True, include_shots=True,
                           include_h2h=True, include_footystats=False)
if "league_level" not in hist_odds.columns:
    hist_odds["league_level"] = hist["league_level"].values

# Season ID for split
def sid(d):
    y = d.year
    return f"{y}-{str(y+1)[-2:]}" if d.month >= 8 else f"{y-1}-{str(y)[-2:]}"
hist["season_id"] = hist["date"].apply(sid)
hist_feat["season_id"] = hist["season_id"].values
hist_odds["season_id"] = hist["season_id"].values

all_seasons = sorted(hist["season_id"].unique())
print(f"Data: {len(hist)} matches, {len(all_seasons)} seasons: {all_seasons}")
print(f"Divisions: {hist['division'].value_counts().to_dict()}")
print(f"Feats: {len(feat_cols)}")

# ── 2. 5-fold walk-forward CV ──
STAKE = 100
results = []
print("\n=== 5-fold Walk-Forward CV (edge >= 30%) ===")
for fold, test_s in enumerate(all_seasons[1:], 1):
    train_s = [s for s in all_seasons if s < test_s]
    tr_mask = hist_feat["season_id"].isin(train_s).values
    te_mask = (hist_feat["season_id"] == test_s).values

    X_tr = fill_features(hist_feat.loc[tr_mask, feat_cols].values)
    y_tr = hist_feat.loc[tr_mask, "target_result"].values.astype(int)
    X_te = fill_features(hist_feat.loc[te_mask, feat_cols].values)
    y_te = hist_feat.loc[te_mask, "target_result"].values.astype(int)

    # Market odds
    mkt_r = hist_odds.loc[te_mask, ["impl_home_prob_close","impl_draw_prob_close","impl_away_prob_close"]].values.astype(float)
    for j in range(len(mkt_r)):
        if np.any(np.isnan(mkt_r[j])): mkt_r[j] = [1/3,1/3,1/3]
    mkt_r = mkt_r / mkt_r.sum(axis=1, keepdims=True)
    mkt_p = np.column_stack([mkt_r[:,2], mkt_r[:,1], mkt_r[:,0]])

    raw_o = hist_odds.loc[te_mask, ["odds_home_close","odds_draw_close","odds_away_close"]].values.astype(float)
    raw_o[raw_o <= 0] = 999.0; raw_o[np.isnan(raw_o)] = 999.0
    raw_o = np.column_stack([raw_o[:,2], raw_o[:,1], raw_o[:,0]])

    lr = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=4000, C=1.0, random_state=42)
    lr.fit(X_tr, y_tr)
    yp = lr.predict_proba(X_te)

    n_te = len(y_te)
    mod_acc = (lr.predict(X_te) == y_te).mean()
    mkt_acc = (np.argmax(mkt_p,1) == y_te).mean()

    n_b=0; n_w=0; pnl=0
    for i in range(n_te):
        be=0; bo=-1
        for o in range(3):
            e = yp[i,o]/mkt_p[i,o]-1
            if e > be: be, bo = e, o
        if be >= 0.30:
            n_b += 1
            won = bo == y_te[i]
            po = raw_o[i, bo]
            pnl += (po-1)*STAKE if won else -STAKE
            if won: n_w += 1

    roi = pnl/(n_b*STAKE)*100 if n_b else 0
    wr = n_w/n_b*100 if n_b else 0
    results.append((test_s, n_te, mod_acc, mkt_acc, n_b, wr, pnl, roi))
    print(f"Fold {fold}: {test_s}  model={mod_acc*100:.1f}% mkt={mkt_acc*100:.1f}%  "
          f"bets={n_b}/{n_te}({n_b/n_te*100:.0f}%)  win={wr:.0f}%  P&L={pnl:+7.0f}  ROI={roi:+6.1f}%")

avg_mod = np.mean([r[2] for r in results])
avg_mkt = np.mean([r[3] for r in results])
tot_b = sum(r[4] for r in results)
tot_m = sum(r[1] for r in results)
tot_p = sum(r[6] for r in results)
avg_r = tot_p/(tot_b*STAKE)*100
print(f"\n=== Summary ===")
print(f"Model acc: {avg_mod*100:.1f}% | Market acc: {avg_mkt*100:.1f}% | Diff: {(avg_mod-avg_mkt)*100:+.1f}pp")
print(f"Total: {tot_m} matches, {tot_b} bets ({tot_b/tot_m*100:.0f}%), P&L={tot_p:+9.0f}, ROI={avg_r:+6.1f}%")

# ── 3. Single 80/20 sweep for threshold comparison ──
print("\n=== Threshold sweep (80/20 split) ===")
split_idx = int(len(hist_feat) * 0.8)
tr2 = hist_feat.iloc[:split_idx]; te2 = hist_feat.iloc[split_idx:]
te_odds2 = hist_odds.iloc[split_idx:]

X_tr2 = fill_features(tr2[feat_cols].values)
y_tr2 = tr2["target_result"].values.astype(int)
X_te2 = fill_features(te2[feat_cols].values)
y_te2 = te2["target_result"].values.astype(int)

lr2 = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=4000, C=1.0, random_state=42)
lr2.fit(X_tr2, y_tr2)
yp2 = lr2.predict_proba(X_te2)

m2 = hist_odds.loc[te_odds2.index, ["impl_home_prob_close","impl_draw_prob_close","impl_away_prob_close"]].values.astype(float)
for j in range(len(m2)):
    if np.any(np.isnan(m2[j])): m2[j] = [1/3,1/3,1/3]
m2 = m2 / m2.sum(axis=1, keepdims=True)
m2 = np.column_stack([m2[:,2], m2[:,1], m2[:,0]])

r2 = hist_odds.loc[te_odds2.index, ["odds_home_close","odds_draw_close","odds_away_close"]].values.astype(float)
r2[r2 <= 0] = 999.0; r2[np.isnan(r2)] = 999.0
r2 = np.column_stack([r2[:,2], r2[:,1], r2[:,0]])

n2 = len(y_te2)
mod_acc2 = (lr2.predict(X_te2) == y_te2).mean()
mkt_acc2 = (np.argmax(m2,1) == y_te2).mean()
print(f"Test: {n2} matches | Model: {mod_acc2*100:.1f}% | Market: {mkt_acc2*100:.1f}%")

print(f"\n{'Thresh':>7} {'Bets':>6} {'%':>6} {'Wins':>5} {'Win%':>7} {'P&L':>10} {'ROI':>8}")
for t in [0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]:
    nb=0; nw=0; pnl=0
    for i in range(n2):
        be=0; bo=-1
        for o in range(3):
            e = yp2[i,o]/m2[i,o]-1
            if e > be: be, bo = e, o
        if be >= t:
            nb += 1
            won = bo == y_te2[i]
            po = r2[i, bo]
            pnl += (po-1)*STAKE if won else -STAKE
            if won: nw += 1
    roi = pnl/(nb*STAKE)*100 if nb else 0
    wr = nw/nb*100 if nb else 0
    print(f"{t:>6.0%} {nb:>6} {nb/n2*100:>5.1f}% {nw:>5} {wr:>6.1f}% {pnl:>+9.0f} {roi:>+7.1f}%")
