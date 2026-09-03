#!/usr/bin/env python3
"""
Re-run 2025-26 backtest WITH historical PPG features.
FootyStats PPG is cross-season, so we compute it from 2019-2024 PL data.
"""
import json, sys
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
from sklearn.metrics import accuracy_score, log_loss, f1_score, brier_score_loss

# ── 1. Compute PPG from 2019-2024 PL data ──
print("[1] Computing PPG from 2019-2024 PL data...")
SEASONS = ['PL1920.csv','PL2021.csv','PL2122.csv','PL2223.csv','PL2324.csv','PL2425.csv']

team_points, team_matches = {}, {}
for fname in SEASONS:
    df = load_data(PROJ / 'league/data/raw' / fname)
    for _, row in df.iterrows():
        home, away = row['home_team'], row['away_team']
        hg, ag = row['home_goals_full'], row['away_goals_full']
        team_matches.setdefault(home, 0); team_points.setdefault(home, 0)
        team_matches.setdefault(away, 0); team_points.setdefault(away, 0)
        if hg > ag: team_points[home] += 3
        elif hg == ag: team_points[home] += 1; team_points[away] += 1
        else: team_points[away] += 3
        team_matches[home] += 1; team_matches[away] += 1

team_ppg = {t: team_points[t] / team_matches[t] for t in team_points}
mean_ppg = float(np.mean(list(team_ppg.values())))
print(f"  {len(team_ppg)} teams, mean PPG={mean_ppg:.3f}")

# ── 2. Build training set (2019-2024, with PPG) ──
print("\n[2] Building training features (2019-2024)...")
train_frames = []
for fname in SEASONS:
    df = load_data(PROJ / 'league/data/raw' / fname)
    df['season'] = fname.replace('.csv', '')
    train_frames.append(df)
train_all = pd.concat(train_frames, ignore_index=True).sort_values('date').reset_index(drop=True)
train_feat = build_features(train_all, windows=[5, 10], include_shots=True,
                             include_h2h=True, include_footystats=False)

# Add computed PPG
ppg_cols = ['home_ppg', 'away_ppg', 'ppg_diff']
train_feat['home_ppg'] = train_feat['home_team'].map(lambda t: team_ppg.get(t, mean_ppg))
train_feat['away_ppg'] = train_feat['away_team'].map(lambda t: team_ppg.get(t, mean_ppg))
train_feat['ppg_diff'] = train_feat['home_ppg'] - train_feat['away_ppg']

# Use full feature set
all_cols = get_feature_columns(train_feat) + ppg_cols
feat_cols = [c for c in all_cols if c in train_feat.columns]
print(f"  Features: {len(feat_cols)} ({len(get_feature_columns(train_feat))} base + {len(ppg_cols)} PPG)")

# ── 3. Build test set (2025-26, with PPG) ──
print("\n[3] Building test features (2025-26 PL)...")
df_2526 = load_data(PROJ / 'league/data/raw/PL2526.csv')
df_2526['season'] = '2025-26'
df_2526_feat = build_features(df_2526, windows=[5, 10], include_shots=True,
                               include_h2h=True, include_footystats=False)
df_2526_feat['home_ppg'] = df_2526_feat['home_team'].map(lambda t: team_ppg.get(t, mean_ppg))
df_2526_feat['away_ppg'] = df_2526_feat['away_team'].map(lambda t: team_ppg.get(t, mean_ppg))
df_2526_feat['ppg_diff'] = df_2526_feat['home_ppg'] - df_2526_feat['away_ppg']

# ── 4. Train & Evaluate ──
print("\n[4] Training LR...")
X_tr = fill_features(train_feat[feat_cols].values.astype(float))
X_te = fill_features(df_2526_feat[feat_cols].values.astype(float))
y_tr = train_feat['target_result'].values.astype(int)
y_te = df_2526_feat['target_result'].values.astype(int)

ss = StandardScaler()
X_tr_s = ss.fit_transform(X_tr)
X_te_s = ss.transform(X_te)

lr = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=4000, random_state=42)
lr.fit(X_tr_s, y_tr)
te_prob = lr.predict_proba(X_te_s)
te_pred = np.argmax(te_prob, axis=1)

acc = accuracy_score(y_te, te_pred)
ll = log_loss(y_te, te_prob)
brier = float(np.mean([brier_score_loss((y_te == i).astype(int), te_prob[:, i]) for i in range(3)]))
f1s = f1_score(y_te, te_pred, average=None, labels=[0, 1, 2])

# Market odds
mpr = np.column_stack([1.0 / df_2526_feat['odds_away'].replace(0, np.nan),
                       1.0 / df_2526_feat['odds_draw'].replace(0, np.nan),
                       1.0 / df_2526_feat['odds_home'].replace(0, np.nan)])
mpr = mpr / mpr.sum(axis=1, keepdims=True)
m_pred = np.argmax(mpr, axis=1)
macc = accuracy_score(y_te, m_pred)
mll = log_loss(y_te, mpr)
mbrier = float(np.mean([brier_score_loss((y_te == i).astype(int), mpr[:, i]) for i in range(3)]))
mf1s = f1_score(y_te, m_pred, average=None, labels=[0, 1, 2])

# ── 5. Results ──
print("\n" + "=" * 60)
print(f"  {'Model':<24} {'Acc':<8} {'LogLoss':<8} {'Brier':<8} {'DrawF1':<8}")
print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'LR + PPG (2025-26)':<24} {acc:<8.4f} {ll:<8.4f} {brier:<8.4f} {f1s[1]:<8.4f}")
print(f"  {'Market (Bet365)':<24} {macc:<8.4f} {mll:<8.4f} {mbrier:<8.4f} {mf1s[1]:<8.4f}")
print(f"  {'Delta':<24} {(acc-macc):<+8.4f} {(ll-mll):<+8.4f} {(brier-mbrier):<+8.4f} {(f1s[1]-mf1s[1]):<+8.4f}")
print("=" * 60)

# Compare with previous results
print(f"\n── Three-way comparison ──")
print(f"  Historical LR+FS test (2019-24, WITH FootyStats):       ~69.74%")
print(f"  New: LR + computed PPG (2025-26, PPG only):             {acc*100:.2f}%")
print(f"  Old: LR fallback (2025-26, NO FootyStats at all):       60.26%")
print(f"  Market (Bet365, 2025-26):                                {macc*100:.2f}%")
print(f"\n  PPG alone recovers {(acc-0.6026)*100:.2f}pp of the ~9.5pp gap from missing FootyStats!")

# Save predictions
preds = pd.DataFrame({
    'home': df_2526_feat['home_team'],
    'away': df_2526_feat['away_team'],
    'actual': y_te, 'pred': te_pred,
    'p_home': te_prob[:, 2], 'p_draw': te_prob[:, 1], 'p_away': te_prob[:, 0],
    'm_home': mpr[:, 2], 'm_draw': mpr[:, 1], 'm_away': mpr[:, 0],
})
preds.to_csv(PROJ / 'outputs/ensemble/backtest_2526_with_ppg.csv', index=False)
print(f"\nPredictions saved to outputs/ensemble/backtest_2526_with_ppg.csv")
