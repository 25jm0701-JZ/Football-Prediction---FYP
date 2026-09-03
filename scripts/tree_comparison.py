#!/usr/bin/env python3
"""
Tree models (RF / XGBoost) vs LR on full E0-E3 — CLEAN features (no leakage).
============================================================================
Re-runs the Experiment 017 comparison after the leakage fix:
  12 current-match stat columns removed from get_feature_columns.

Data: E0-E3, 7 seasons (2019/20–2025/26, 13,988 matches), walk-forward CV.
Scope mirrors scripts/backtest_english4.py so LR is directly comparable.

Usage:
    python scripts/tree_comparison.py
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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, f1_score

DIV_LEVEL = {"E0": 0, "E1": 1, "E2": 2, "E3": 3}

# ── 1. Load E0-E3 (7 seasons, same as backtest_english4.py) ──
print("Loading English 4-league data...")
all_frames = []
for div, level in DIV_LEVEL.items():
    files = sorted(Path("league/data/english_raw").glob(f"{div}*.csv"))
    for f in files:
        df = load_data(f)
        df["league_level"] = level
        df["division"] = div
        all_frames.append(df)

hist = pd.concat(all_frames, ignore_index=True)
hist = hist.sort_values("date").reset_index(drop=True)

hist_feat = build_features(hist, windows=[5, 10], include_odds=False,
                           include_shots=True, include_h2h=True,
                           include_footystats=False)
feat_cols = get_feature_columns(hist_feat)
if "league_level" not in hist_feat.columns:
    hist_feat["league_level"] = hist["league_level"].values
feat_cols = list(set(feat_cols + ["league_level"]))
feat_cols = [c for c in feat_cols if c in hist_feat.columns]
feat_cols = sorted(feat_cols)

def sid(d):
    y = d.year
    return f"{y}-{str(y+1)[-2:]}" if d.month >= 8 else f"{y-1}-{str(y)[-2:]}"

hist_feat["season_id"] = hist["date"].apply(sid)
all_seasons = sorted(hist_feat["season_id"].unique())
print(f"Data: {len(hist)} matches, {len(all_seasons)} seasons, {len(feat_cols)} features")

# ── 2. Walk-forward: LR vs RF vs XGBoost ──
models = {
    "LR": LogisticRegression(penalty="l2", solver="lbfgs", max_iter=4000, C=1.0, random_state=42),
    "RF": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric="logloss"),
}

print("\n=== Walk-Forward (7 seasons) ===")
accs = {m: [] for m in models}
loses = {m: [] for m in models}
drawf1 = {m: [] for m in models}
for test_s in all_seasons[1:]:
    train_s = [s for s in all_seasons if s < test_s]
    tr_mask = hist_feat["season_id"].isin(train_s).values
    te_mask = (hist_feat["season_id"] == test_s).values
    X_tr = fill_features(hist_feat.loc[tr_mask, feat_cols].values)
    X_te = fill_features(hist_feat.loc[te_mask, feat_cols].values)
    y_tr = hist_feat.loc[tr_mask, "target_result"].values.astype(int)
    y_te = hist_feat.loc[te_mask, "target_result"].values.astype(int)
    X_tr = np.nan_to_num(X_tr, nan=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0)

    row = [test_s]
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        yp = m.predict_proba(X_te)
        ypp = m.predict(X_te)
        a = accuracy_score(y_te, ypp)
        ll = log_loss(y_te, yp, labels=[0, 1, 2])
        df1 = f1_score(y_te, ypp, average=None, labels=[0, 1, 2])[1]
        accs[name].append(a); loses[name].append(ll); drawf1[name].append(df1)
        row.append(f"{a*100:.1f}%")
    print("  " + " | ".join(row))

print("\n=== Aggregate (mean over test seasons) ===")
print(f"{'Model':<10}{'Acc':>8}{'LogLoss':>10}{'DrawF1':>9}")
for name in models:
    print(f"{name:<10}{np.mean(accs[name])*100:>7.2f}%{np.mean(loses[name]):>10.4f}{np.mean(drawf1[name]):>9.3f}")

lr = np.mean(accs["LR"])
for name in models:
    if name == "LR":
        continue
    print(f"\n{name} vs LR: {np.mean(accs[name])-lr:+.2f}pp")
