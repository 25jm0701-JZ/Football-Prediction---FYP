#!/usr/bin/env python3
"""
Track B clean evaluation (no leakage) — 80/20 and cross-season splits.
=====================================================================
Reproduces the two honest Track B accuracy numbers after the leakage fix,
with the FULL thesis pipeline (build_features + get_feature_columns):

  - 80/20 split (456 test):  model vs closing-odds market
  - Cross-season 4/2 (760):  model vs closing-odds market

Usage:
    python scripts/trackb_clean_eval.py
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
from sklearn.metrics import accuracy_score

SEASONS = {"2019-20": "PL1920.csv", "2020-21": "PL2021.csv",
           "2021-22": "PL2122.csv", "2022-23": "PL2223.csv",
           "2023-24": "PL2324.csv", "2024-25": "PL2425.csv"}
F2S = {"PL1920.csv": 2020, "PL2021.csv": 2021, "PL2122.csv": 2022,
       "PL2223.csv": 2023, "PL2324.csv": 2024, "PL2425.csv": 2025}

frames = []
for season, fname in SEASONS.items():
    df = load_data(PROJ / "league/data/raw" / fname)
    df["season_num"] = F2S[fname]
    df["season"] = season
    frames.append(df)
full = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)

df_feat = build_features(full, windows=[5, 10], include_shots=True,
                         include_h2h=True, include_footystats=True,
                         include_odds=False, footystats_season_col="season")
feat_cols = get_feature_columns(df_feat)
exclude_meta = {"date", "season", "season_num", "target_result", "league"}
feat_cols = [c for c in feat_cols if c not in exclude_meta and c in df_feat.columns]
df_feat["target_result"] = full["target_result"].values
df_feat["season_num"] = full["season_num"].values
print(f"Features: {len(feat_cols)}")


def market_acc(indices, y):
    o = full.loc[indices, ["odds_home_close", "odds_draw_close", "odds_away_close"]].values.astype(float)
    imp = 1.0 / np.maximum(o, 1.01)
    imp = imp / imp.sum(axis=1, keepdims=True)
    mp = imp.argmax(axis=1)  # 0=home,1=draw,2=away
    return (((mp == 2) & (y == 0)) | ((mp == 1) & (y == 1)) | ((mp == 0) & (y == 2))).mean()


def run(tr, te, label):
    Xtr = np.nan_to_num(fill_features(tr[feat_cols].values.astype(float)), nan=0.0)
    Xte = np.nan_to_num(fill_features(te[feat_cols].values.astype(float)), nan=0.0)
    ytr, yte = tr["target_result"].values.astype(int), te["target_result"].values.astype(int)
    lr = LogisticRegression(C=1.0, max_iter=4000, random_state=42)
    lr.fit(Xtr, ytr)
    acc = accuracy_score(yte, lr.predict(Xte))
    mk = market_acc(te.index, yte)
    print(f"\n[{label}] test={len(te)}  model={acc*100:.2f}%  market(closing)={mk*100:.2f}%  delta={acc*100-mk*100:+.2f}pp")
    return acc, mk


# 80/20 chronological split
split = int(len(df_feat) * 0.8)
run(df_feat.iloc[:split], df_feat.iloc[split:], "80/20 split")

# Cross-season: train 2019-2022, test 2023-24 + 2024-25
run(df_feat[df_feat["season_num"] < 2024],
    df_feat[df_feat["season_num"] >= 2024], "Cross-season 4/2")
