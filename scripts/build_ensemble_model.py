#!/usr/bin/env python3
"""
Build Ensemble Model + Backtest 2025-26
========================================
Trains two models:
  A) Ensemble (LR + MLP-Deep) — for teams WITH FootyStats coverage
  B) Fallback LR — for teams WITHOUT FootyStats (E0-E3, promoted, gap seasons)

Then backtests on 2025-26 PL season (no FootyStats available) to validate
the fallback approach.

Train data:
  Ensemble: PL 2019-20 to 2024-25 (6 seasons, 2,280 matches) + FootyStats
  Fallback: E0-E3 2019-20 to 2024-25 (7 seasons, ~12k matches) + league_level

Usage:
    python scripts/build_ensemble_model.py
"""
import json, pickle, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, f1_score, brier_score_loss

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "outputs" / "ensemble"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJ / "league"))
from src.data_loader import load_data
from src.models import fill_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")

# ═══════════════════════════════════════════════════════════════
# 0. MLP-Deep Model Definition (same as Exp 015)
# ═══════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.net(x)

class MLPDeep(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden, dropout) for _ in range(4)])
        self.head = nn.Linear(hidden, 3)
    def forward(self, x):
        h = F.relu(self.bn(self.proj(x)))
        for b in self.blocks:
            h = b(h)
        return self.head(h)

# ═══════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_pl_with_footystats() -> pd.DataFrame:
    """Load 6 PL seasons with FootyStats. This is the ensemble training set."""
    from src.features import build_features, get_feature_columns

    SEASONS = {
        "2019-20": "PL1920.csv", "2020-21": "PL2021.csv",
        "2021-22": "PL2122.csv", "2022-23": "PL2223.csv",
        "2023-24": "PL2324.csv", "2024-25": "PL2425.csv",
    }
    frames = []
    for season, fname in SEASONS.items():
        df = load_data(PROJ / "league/data/raw" / fname)
        df["season"] = season
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    print(f"[PL+FootyStats] {len(df)} matches, {df['season'].nunique()} seasons")

    df_feat = build_features(df, windows=[5, 10], include_shots=True,
                              include_h2h=True, include_footystats=True,
                              include_odds=False, footystats_season_col="season")
    feat_cols = get_feature_columns(df_feat)
    print(f"  Features: {len(feat_cols)} (incl. 9 FootyStats)")
    return df_feat, feat_cols


def load_e0e3_for_fallback() -> pd.DataFrame:
    """Load E0-E3 2019-20 to 2024-25 (NO FootyStats) for fallback training.

    Returns feature DataFrame with league_level column.
    """
    from src.features import build_features, get_feature_columns

    LEAGUES = {
        "E0": 0, "E1": 1, "E2": 2, "E3": 3,
    }
    SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425"]

    frames = []
    for league, lvl in LEAGUES.items():
        for season in SEASONS:
            fname = f"{league}{season}.csv"
            path = PROJ / "league/data/english_raw" / fname
            if not path.exists():
                continue
            df = load_data(path)
            df["league_level"] = lvl
            df["season"] = f"20{season[:2]}-20{season[2:]}"
            # Infer season tag for cross-season rolling
            frames.append(df)

    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    print(f"\n[E0-E3 Fallback] {len(df)} matches ({df['league_level'].nunique()} divisions)")

    df_feat = build_features(df, windows=[5, 10], include_shots=True,
                              include_h2h=True, include_footystats=False,
                              include_odds=False)
    df_feat["league_level"] = df["league_level"].values  # preserve after build_features
    feat_cols = get_feature_columns(df_feat) + ["league_level"]
    print(f"  Features: {len(feat_cols)} (no FootyStats, + league_level)")
    return df_feat, feat_cols


# ═══════════════════════════════════════════════════════════════
# 2. TRAINING
# ═══════════════════════════════════════════════════════════════

def temporal_split_sorted(df, test_ratio=0.1):
    """Chronological split, sorted by date."""
    df = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * (1 - test_ratio))
    return df.iloc[:split], df.iloc[split:]


def train_lr(X_tr, y_tr, X_val, y_val):
    """Train LR with L2 regularization."""
    lr = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                             max_iter=4000, random_state=42)
    lr.fit(X_tr, y_tr)
    val_prob = lr.predict_proba(X_val)
    val_ll = log_loss(y_val, val_prob)
    print(f"  LR val log_loss: {val_ll:.4f}")
    return lr


def train_mlp(X_tr, y_tr, X_val, y_val, in_dim, verbose=False):
    """Train MLP-Deep with early stopping."""
    torch.manual_seed(42)
    model = MLPDeep(in_dim, hidden=128, dropout=0.3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    X_tr_t = torch.FloatTensor(X_tr).to(DEVICE)
    y_tr_t = torch.LongTensor(y_tr).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.LongTensor(y_val).to(DEVICE)

    best_loss = float("inf")
    best_state = None
    wait = 0
    n = len(X_tr)

    for epoch in range(1, 501):
        model.train()
        perm = torch.randperm(n)
        loss_sum = 0
        for start in range(0, n, 32):
            idx = perm[start:start + 32]
            opt.zero_grad()
            loss = criterion(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        sched.step(val_loss)

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 20 and epoch > 50:
                break
        if verbose and epoch % 100 == 0:
            print(f"    epoch {epoch}: train={loss_sum/(n//32+1):.4f} val={val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        val_prob = F.softmax(model(X_val_t), dim=1).cpu().numpy()
    val_ll = log_loss(y_val, val_prob)
    print(f"  MLP-Deep val log_loss: {val_ll:.4f}")
    return model


def find_ensemble_weight(lr_prob, mlp_prob, y_val):
    """Grid search over ensemble weight w: P = w*LR + (1-w)*MLP."""
    best_w, best_ll = 0.5, float("inf")
    for w in np.arange(0.3, 0.8, 0.05):
        ensemble = w * lr_prob + (1 - w) * mlp_prob
        ll = log_loss(y_val, ensemble)
        if ll < best_ll:
            best_ll, best_w = ll, w
    return best_w, best_ll


# ═══════════════════════════════════════════════════════════════
# 3. EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate(y_true, y_prob, label=""):
    y_pred = np.argmax(y_prob, axis=1)
    f1s = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    market_prob = None
    return {
        "model": label,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "log_loss": round(float(log_loss(y_true, y_prob)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "brier": round(float(np.mean([
            brier_score_loss((y_true == i).astype(int), y_prob[:, i]) for i in range(3)])), 4),
        "draw_f1": round(float(f1s[1]), 4),
        "n_test": len(y_true),
    }


# ═══════════════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  ENSEMBLE MODEL + 2025-26 BACKTEST")
    print("=" * 72)

    # ── Phase 1: Train Ensemble on PL+FootyStats ──
    print("\n" + "─" * 60)
    print("[Phase 1] Training Ensemble on PL 2019-2024 + FootyStats")
    print("─" * 60)

    df_pl, feat_cols_pl = load_pl_with_footystats()
    n_feat_pl = len(feat_cols_pl)

    # Split: 80/10/10 temporal
    train_df, temp = temporal_split_sorted(df_pl, 0.2)
    val_df, test_df = temporal_split_sorted(temp, 0.5)

    X_tr = fill_features(train_df[feat_cols_pl].values.astype(float))
    X_val = fill_features(val_df[feat_cols_pl].values.astype(float))
    X_te = fill_features(test_df[feat_cols_pl].values.astype(float))
    y_tr = train_df["target_result"].values.astype(int)
    y_val = val_df["target_result"].values.astype(int)
    y_te = test_df["target_result"].values.astype(int)

    print(f"  Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}")

    # Scale
    scaler_pl = StandardScaler()
    X_tr_s = scaler_pl.fit_transform(X_tr)
    X_val_s = scaler_pl.transform(X_val)
    X_te_s = scaler_pl.transform(X_te)

    # Train LR
    print("\n  Training LR...")
    lr_model = train_lr(X_tr_s, y_tr, X_val_s, y_val)
    lr_val_prob = lr_model.predict_proba(X_val_s)

    # Train MLP-Deep
    print("  Training MLP-Deep...")
    mlp_model = train_mlp(X_tr_s, y_tr, X_val_s, y_val, n_feat_pl)

    with torch.no_grad():
        mlp_val_prob = F.softmax(mlp_model(torch.FloatTensor(X_val_s).to(DEVICE)), dim=1).cpu().numpy()

    # Find ensemble weight
    w, w_ll = find_ensemble_weight(lr_val_prob, mlp_val_prob, y_val)
    print(f"\n  Best ensemble weight: LR={w:.2f}, MLP={1-w:.2f} (val LL={w_ll:.4f})")

    # Evaluate on test set
    lr_te_prob = lr_model.predict_proba(X_te_s)
    with torch.no_grad():
        mlp_te_prob = F.softmax(mlp_model(torch.FloatTensor(X_te_s).to(DEVICE)), dim=1).cpu().numpy()
    ensemble_te_prob = w * lr_te_prob + (1 - w) * mlp_te_prob

    res_lr = evaluate(y_te, lr_te_prob, "LR")
    res_mlp = evaluate(y_te, mlp_te_prob, "MLP-Deep")
    res_ens = evaluate(y_te, ensemble_te_prob, "Ensemble")
    print(f"\n  Test set ({len(y_te)} matches) — WITH FootyStats:")
    for r in [res_lr, res_mlp, res_ens]:
        print(f"    {r['model']:<12} acc={r['accuracy']:.4f} ll={r['log_loss']:.4f} "
              f"drawF1={r['draw_f1']:.4f}")

    # Save ensemble models
    torch.save(mlp_model.state_dict(), OUT / "mlp_deep.pt")
    with open(OUT / "lr_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    with open(OUT / "scaler_pl.pkl", "wb") as f:
        pickle.dump(scaler_pl, f)
    pd.Series({"weight_lr": w, "weight_mlp": 1 - w, "n_features": n_feat_pl}).to_csv(
        OUT / "ensemble_config.csv")
    print(f"\n  [Saved] Ensemble models to {OUT}")

    # ── Phase 2: Train Fallback on E0-E3 (no FootyStats) ──
    print("\n" + "─" * 60)
    print("[Phase 2] Training Fallback on E0-E3 2019-2024 (no FootyStats)")
    print("─" * 60)

    df_fb, feat_cols_fb = load_e0e3_for_fallback()
    n_feat_fb = len(feat_cols_fb)
    print(f"  Features: {n_feat_fb}")

    # Split
    fb_train, fb_test = temporal_split_sorted(df_fb, 0.1)
    # Further split train for val
    fb_train, fb_val = temporal_split_sorted(fb_train, 0.1 / 0.9)

    X_fb_tr = fill_features(fb_train[feat_cols_fb].values.astype(float))
    X_fb_val = fill_features(fb_val[feat_cols_fb].values.astype(float))
    X_fb_te = fill_features(fb_test[feat_cols_fb].values.astype(float))
    y_fb_tr = fb_train["target_result"].values.astype(int)
    y_fb_val = fb_val["target_result"].values.astype(int)
    y_fb_te = fb_test["target_result"].values.astype(int)

    print(f"  Train: {len(X_fb_tr)} | Val: {len(X_fb_val)} | Test: {len(X_fb_te)}")

    scaler_fb = StandardScaler()
    X_fb_tr_s = scaler_fb.fit_transform(X_fb_tr)
    X_fb_val_s = scaler_fb.transform(X_fb_val)
    X_fb_te_s = scaler_fb.transform(X_fb_te)

    # Train fallback LR
    print("\n  Training Fallback LR...")
    fb_lr = train_lr(X_fb_tr_s, y_fb_tr, X_fb_val_s, y_fb_val)

    # Eval on E0-E3 test
    fb_te_prob = fb_lr.predict_proba(X_fb_te_s)
    res_fb = evaluate(y_fb_te, fb_te_prob, "Fallback LR")
    print(f"\n  E0-E3 test ({len(y_fb_te)} matches):")
    print(f"    acc={res_fb['accuracy']:.4f} ll={res_fb['log_loss']:.4f} "
          f"drawF1={res_fb['draw_f1']:.4f}")

    # Save fallback
    with open(OUT / "fallback_lr.pkl", "wb") as f:
        pickle.dump(fb_lr, f)
    with open(OUT / "scaler_fallback.pkl", "wb") as f:
        pickle.dump(scaler_fb, f)
    pd.Series({"n_features": n_feat_fb}).to_csv(OUT / "fallback_config.csv")
    print(f"  [Saved] Fallback model to {OUT}")

    # ── Phase 3: Backtest on 2025-26 PL ──
    print("\n" + "─" * 60)
    print("[Phase 3] Backtest on 2025-26 Premier League (no FootyStats)")
    print("─" * 60)

    from src.features import build_features, get_feature_columns

    df_2526 = load_data(PROJ / "league/data/raw/PL2526.csv")
    df_2526["league_level"] = 0
    df_2526["season"] = "2025-26"

    df_2526_feat = build_features(df_2526, windows=[5, 10], include_shots=True,
                                    include_h2h=True, include_footystats=False,
                                    include_odds=True)  # keep odds for market comparison
    df_2526_feat["league_level"] = 0

    # Get fallback feature columns (intersection)
    avail_cols = [c for c in feat_cols_fb if c in df_2526_feat.columns]
    missing = set(feat_cols_fb) - set(avail_cols)
    if missing:
        print(f"  Missing columns (will be zero-filled): {missing}")
        for c in missing:
            df_2526_feat[c] = 0.0

    X_2526 = fill_features(df_2526_feat[feat_cols_fb].values.astype(float))
    X_2526_s = scaler_fb.transform(X_2526)
    y_2526 = df_2526_feat["target_result"].values.astype(int)

    # Fallback predictions
    fb_2526_prob = fb_lr.predict_proba(X_2526_s)
    res_2526_fb = evaluate(y_2526, fb_2526_prob, "Fallback")
    market_2526_prob = None

    # Market odds benchmark — unified on Bet365 CLOSING odds (B365C)
    if all(c in df_2526.columns for c in ["odds_home_close", "odds_draw_close", "odds_away_close"]):
        mpr = np.column_stack([
            1.0 / df_2526["odds_away_close"].replace(0, np.nan),
            1.0 / df_2526["odds_draw_close"].replace(0, np.nan),
            1.0 / df_2526["odds_home_close"].replace(0, np.nan),
        ])
        mpr = mpr / mpr.sum(axis=1, keepdims=True)
        m_pred = np.argmax(mpr, axis=1)
        market_acc = accuracy_score(y_2526, m_pred)
        market_ll = log_loss(y_2526, mpr)
        market_brier = np.mean([
            brier_score_loss((y_2526 == i).astype(int), mpr[:, i]) for i in range(3)])
        market_f1s = f1_score(y_2526, m_pred, average=None, labels=[0, 1, 2])
        market_2526_prob = mpr
    else:
        market_acc = market_ll = market_brier = 0
        market_f1s = [0, 0, 0]

    # ── Print final table ──
    print(f"\n  {'='*60}")
    print(f"  {'Model':<24} {'Accuracy':<10} {'LogLoss':<10} {'DrawF1':<10}")
    print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Fallback LR (2025-26)':<24} {res_2526_fb['accuracy']:<10.4f} "
          f"{res_2526_fb['log_loss']:<10.4f} {res_2526_fb['draw_f1']:<10.4f}")
    if market_2526_prob is not None:
        print(f"  {'Market Odds (Bet365 CLOSING)':<24} {market_acc:<10.4f} {market_ll:<10.4f} "
              f"{market_f1s[1]:<10.4f}")
        print(f"  {'Delta (Model - Market)':<24} {res_2526_fb['accuracy'] - market_acc:<+10.4f} "
              f"{res_2526_fb['log_loss'] - market_ll:<+10.4f} "
              f"{res_2526_fb['draw_f1'] - market_f1s[1]:<+10.4f}")

    # Historical comparison
    print(f"\n  {'='*60}")
    print(f"  {'Historical Comparison':^60}")
    print(f"  {'='*60}")
    print(f"  {'Model / Season':<30} {'Acc':<8} {'DrawF1':<8} {'Note':<16}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*16}")
    print(f"  {'LR + FootyStats (2019-24)':<30} {res_lr['accuracy']:<8.4f} {res_lr['draw_f1']:<8.4f} {'with FS':<16}")
    print(f"  {'Ensemble (2019-24)':<30} {res_ens['accuracy']:<8.4f} {res_ens['draw_f1']:<8.4f} {'LR+MLP':<16}")
    print(f"  {'Fallback (E0-E3 test)':<30} {res_fb['accuracy']:<8.4f} {res_fb['draw_f1']:<8.4f} {'no FS':<16}")
    print(f"  {'Fallback (2025-26 PL)':<30} {res_2526_fb['accuracy']:<8.4f} {res_2526_fb['draw_f1']:<8.4f} {'no FS':<16}")
    if market_2526_prob is not None:
        print(f"  {'Market (2025-26)':<30} {market_acc:<8.4f} {market_f1s[1]:<8.4f} {'Bet365 CLOSING':<16}")

    # ── Save results ──
    all_results = {
        "ensemble_on_fs": {
            "train_seasons": "2019-20 to 2024-25",
            "n_train": len(X_tr),
            "n_val": len(X_val),
            "n_test": len(X_te),
            "features": n_feat_pl,
            "ensemble_weight_lr": w,
            "lr": res_lr,
            "mlp": res_mlp,
            "ensemble": res_ens,
        },
        "fallback": {
            "train_data": "E0-E3 2019-20 to 2024-25",
            "n_train": len(X_fb_tr),
            "model": res_fb,
            "features": n_feat_fb,
        },
        "backtest_2526": {
            "season": "2025-26",
            "n_matches": len(y_2526),
            "fallback": res_2526_fb,
            "market": {
                "accuracy": round(float(market_acc), 4),
                "log_loss": round(float(market_ll), 4),
                "brier": round(float(market_brier), 4),
                "draw_f1": round(float(market_f1s[1]), 4),
            } if market_2526_prob is not None else None,
        },
    }

    with open(OUT / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  [Saved] Full results to {OUT / 'results.json'}")

    # Save predictions
    preds = pd.DataFrame({
        "home": df_2526_feat["home_team"].values,
        "away": df_2526_feat["away_team"].values,
        "actual": y_2526,
        "pred": np.argmax(fb_2526_prob, axis=1),
        "p_home": fb_2526_prob[:, 2],
        "p_draw": fb_2526_prob[:, 1],
        "p_away": fb_2526_prob[:, 0],
    })
    if market_2526_prob is not None:
        preds["market_home"] = market_2526_prob[:, 2]
        preds["market_draw"] = market_2526_prob[:, 1]
        preds["market_away"] = market_2526_prob[:, 0]
        preds["market_pred"] = np.argmax(market_2526_prob, axis=1)

    preds.to_csv(OUT / "backtest_2526_predictions.csv", index=False)
    print(f"  [Saved] Predictions to {OUT / 'backtest_2526_predictions.csv'}")

    # Save final summary
    summary = [
        "=" * 62,
        "  FINAL SYSTEM — 2-TIER ENSEMBLE ARCHITECTURE",
        "=" * 62,
        "",
        "  TIER 1 — Teams WITH FootyStats (established PL teams):",
        f"    Model: Ensemble (w_LR={w:.2f} + w_MLP={1-w:.2f})",
        f"    Features: {n_feat_pl} (rolling + shots + h2h + FootyStats)",
        f"    Accuracy: {res_ens['accuracy']:.4f}",
        f"    DrawF1:   {res_ens['draw_f1']:.4f}",
        f"    LogLoss:  {res_ens['log_loss']:.4f}",
        "",
        "  TIER 2 — Teams WITHOUT FootyStats (promoted / gap seasons):",
        f"    Model: LogisticRegression (L2)",
        f"    Features: {n_feat_fb} (rolling + shots + h2h + league_level)",
        f"    Trained on: E0-E3 2019-2024 ({len(X_fb_tr) + len(X_fb_val)} matches)",
        f"    E0-E3 test: {res_fb['accuracy']:.4f} acc, {res_fb['draw_f1']:.4f} drawF1",
        "",
        f"  2025-26 BACKTEST (PL, no FootyStats available):",
        f"    Fallback accuracy: {res_2526_fb['accuracy']:.4f}",
        f"    Fallback drawF1:   {res_2526_fb['draw_f1']:.4f}",
    ]
    if market_2526_prob is not None:
        summary += [
            f"    Market accuracy:   {market_acc:.4f}",
            f"    Market drawF1:     {market_f1s[1]:.4f}",
            f"    Delta (model - market): +{(res_2526_fb['accuracy'] - market_acc)*100:.2f}pp acc",
        ]

    summary += [
        "",
        "  SAVED MODELS:",
        f"    {OUT / 'lr_model.pkl'}",
        f"    {OUT / 'mlp_deep.pt'}",
        f"    {OUT / 'fallback_lr.pkl'}",
        f"    {OUT / 'scaler_pl.pkl'}",
        f"    {OUT / 'scaler_fallback.pkl'}",
        "=" * 62,
    ]

    summary_text = "\n".join(summary)
    with open(OUT / "summary.txt", "w") as f:
        f.write(summary_text)
    print(f"\n  [Saved] Summary to {OUT / 'summary.txt'}")
    print(summary_text)


if __name__ == "__main__":
    main()
