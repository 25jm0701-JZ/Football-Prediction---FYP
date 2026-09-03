#!/usr/bin/env python3
"""
Over/Under 2.5 goals — CLEAN features (no leakage).
====================================================
Re-runs Experiment 012 after the leakage fix (12 current-match stat columns
removed from get_feature_columns). PL-only + FootyStats, 80/20 temporal split.

Usage:
    python scripts/over_under_experiment.py
"""
import sys
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
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss

SEASONS_MAP = {
    "2019-20": "PL1920.csv", "2020-21": "PL2021.csv",
    "2021-22": "PL2122.csv", "2022-23": "PL2223.csv",
    "2023-24": "PL2324.csv", "2024-25": "PL2425.csv",
}
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
    df["season"] = season
    frames.append(df)

full = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
print(f"Total: {len(full)} matches")

# Over/under 2.5 target
full["target_over_2.5"] = (full["home_goals_full"] + full["away_goals_full"] > 2.5).astype(int)

df_feat = build_features(full, windows=[5, 10], include_shots=True,
                         include_h2h=True, include_footystats=True,
                         include_odds=False, footystats_season_col="season")
feat_cols = get_feature_columns(df_feat)
feat_cols = [c for c in feat_cols if c not in {"date", "season_num", "season", "target_result", "league", "target_over_2.5"}
             and c in df_feat.columns]
df_feat["target_over_2.5"] = full["target_over_2.5"].values
df_feat["season_num"] = full["season_num"].values
print(f"Features: {len(feat_cols)}")

# 80/20 chronological split
split = int(len(df_feat) * 0.8)
tr, te = df_feat.iloc[:split], df_feat.iloc[split:]
X_tr = fill_features(tr[feat_cols].values.astype(float))
X_te = fill_features(te[feat_cols].values.astype(float))
X_tr = np.nan_to_num(X_tr, nan=0.0)
X_te = np.nan_to_num(X_te, nan=0.0)
y_tr = tr["target_over_2.5"].values
y_te = te["target_over_2.5"].values
print(f"Split: train={len(tr)} test={len(te)} | Over rate in test: {y_te.mean():.2%}")

ss = StandardScaler()
lr = LogisticRegression(C=1.0, max_iter=4000, random_state=42)
lr.fit(ss.fit_transform(X_tr), y_tr)
p = lr.predict_proba(ss.transform(X_te))[:, 1]
pred = (p >= 0.5).astype(int)

acc = accuracy_score(y_te, pred)
ll = log_loss(y_te, p)
auc = roc_auc_score(y_te, p)
brier = brier_score_loss(y_te, p)

print("\n=== O/U 2.5 (clean features) ===")
print(f"  Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
print(f"  LogLoss:  {ll:.4f}")
print(f"  ROC AUC:  {auc:.4f}")
print(f"  Brier:    {brier:.4f}")
print(f"  Naive (always Over): {y_te.mean():.2%}")
