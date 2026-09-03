#!/usr/bin/env python3
"""
Deep Learning Experiment — League Prediction
============================================
Compares PyTorch-based deep learning models against the LR + FootyStats baseline.
Uses the EXACT same feature pipeline for fair comparison.

Models:
  A) Logistic Regression (baseline, re-run for fair comparison)
  B) PyTorch FNN variants (MLP-Base, MLP-Deep, MLP-Wide)
  C) GRU sequence models (GRU-Simple, GRU-Dual, GRU-Attn)

Usage:
    python scripts/deep_learning_experiment.py
"""
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, f1_score, brier_score_loss

PROJ_ROOT = Path(__file__).resolve().parent.parent

# Import existing pipeline modules
sys.path.insert(0, str(PROJ_ROOT / "league"))
from src.data_loader import load_data, load_multiple
from src.features import build_features, get_feature_columns, _footystats_extra_features
from src.models import temporal_split, fill_features

OUTPUT_DIR = PROJ_ROOT / "outputs" / "deep_learning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# 1. DATA LOADING (using existing pipeline)
# ============================================================

ORIG_FILES = {
    "2019-20": "PL1920.csv", "2020-21": "PL2021.csv",
    "2021-22": "PL2122.csv", "2022-23": "PL2223.csv",
    "2023-24": "PL2324.csv", "2024-25": "PL2425.csv",
}


def load_pl_data() -> pd.DataFrame:
    """Load 6 PL seasons using the existing pipeline's data_loader."""
    frames = []
    for season, fname in ORIG_FILES.items():
        path = PROJ_ROOT / "league/data/raw" / fname
        if not path.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        df = load_data(path)
        df["season"] = season
        # Ensure we have goals_full columns (data_loader creates them)
        if "home_goals_full" not in df.columns:
            df.rename(columns={"home_goals": "home_goals_full",
                               "away_goals": "away_goals_full"}, inplace=True)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values("date").reset_index(drop=True)
    print(f"[Data] {len(all_df)} matches, {all_df['season'].nunique()} seasons")
    for s in sorted(all_df["season"].unique()):
        n = (all_df["season"] == s).sum()
        print(f"  {s}: {n} matches")
    return all_df


def build_features_for_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """Build features using the existing pipeline function.

    Uses the same configuration as the best LR model:
    windows=[5,10], include_footystats=True, include_odds=False,
    include_shots=True, include_h2h=True
    """
    print("\n[Feature Engineering]")
    df_feat = build_features(
        df,
        windows=[5, 10],
        include_shots=True,
        include_odds=False,
        include_h2h=True,
        include_footystats=True,
        footystats_season_col="season",
    )
    feat_cols = get_feature_columns(df_feat)
    print(f"  Total features: {len(feat_cols)}")
    return df_feat, feat_cols


def split_data(df_feat: pd.DataFrame, feat_cols: list[str]):
    """Split into train/val/test with temporal ordering.

    Uses 80/10/10: first 80% train, next 10% val, last 10% test.
    """
    train_df, test_df = temporal_split(df_feat, test_ratio=0.1)
    # Further split train into train + val (last 10% of train)
    train_df, val_df = temporal_split(train_df, test_ratio=0.1 / 0.9)

    print(f"\n[Split]")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    X_tr = fill_features(train_df[feat_cols].values.astype(float))
    X_val = fill_features(val_df[feat_cols].values.astype(float))
    X_te = fill_features(test_df[feat_cols].values.astype(float))

    y_tr = train_df["target_result"].values.astype(int)
    y_val = val_df["target_result"].values.astype(int)
    y_te = test_df["target_result"].values.astype(int)

    # Standardize
    ss = StandardScaler()
    X_tr_s = ss.fit_transform(X_tr)
    X_val_s = ss.transform(X_val)
    X_te_s = ss.transform(X_te)

    return X_tr_s, X_val_s, X_te_s, y_tr, y_val, y_te, train_df, val_df, test_df


# ============================================================
# 2. GRU SEQUENCE BUILDER
# ============================================================

def build_sequences(match_df: pd.DataFrame, K: int = 10):
    """Build per-team match sequences for GRU models.

    For each match, extract the last K matches for home and away teams
    as (goals_for, goals_conceded, is_home) vectors.

    Returns:
        seq_home, seq_away: (n, K, 3) arrays
        mask: (n, K) — 1=valid, 0=padded
    """
    team_histories: dict[str, list[dict]] = {}
    n = len(match_df)
    seq_home = np.zeros((n, K, 3), dtype=np.float32)
    seq_away = np.zeros((n, K, 3), dtype=np.float32)
    mask = np.zeros((n, K), dtype=np.float32)

    for i, (_, row) in enumerate(match_df.iterrows()):
        home = row["home_team"]
        away = row["away_team"]

        # Home history
        hist = team_histories.get(home, [])
        last_k = hist[-K:]
        for t, m in enumerate(last_k):
            seq_home[i, K - len(last_k) + t] = [m["gf"], m["ga"], float(m["is_home"])]
            mask[i, K - len(last_k) + t] = 1.0

        # Away history
        hist = team_histories.get(away, [])
        last_k = hist[-K:]
        for t, m in enumerate(last_k):
            seq_away[i, K - len(last_k) + t] = [m["gf"], m["ga"], float(m["is_home"])]
            mask[i, K - len(last_k) + t] = 1.0

        # Update with current match
        hg = int(row["home_goals_full"]) if pd.notna(row.get("home_goals_full")) else 0
        ag = int(row["away_goals_full"]) if pd.notna(row.get("away_goals_full")) else 0
        team_histories.setdefault(home, []).append({"gf": hg, "ga": ag, "is_home": 1})
        team_histories.setdefault(away, []).append({"gf": ag, "ga": hg, "is_home": 0})

    return seq_home, seq_away, mask


# ============================================================
# 3. PYTOCH MODELS
# ============================================================

class MLPBase(nn.Module):
    """3-layer MLP with BatchNorm + Dropout."""
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.BatchNorm1d(hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, 3),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """MLP block with residual connection (if dims match)."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(dropout)
        self.use_skip = (in_dim == out_dim)

    def forward(self, x):
        out = self.drop(F.relu(self.bn(self.fc(x))))
        return out + x if self.use_skip else out


class MLPDeep(nn.Module):
    """5-block MLP with residual connections."""
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(in_dim, hidden, dropout),
            ResidualBlock(hidden, hidden, dropout),
            ResidualBlock(hidden, hidden, dropout),
            ResidualBlock(hidden, hidden, dropout),
            ResidualBlock(hidden, hidden, dropout),
        ])
        self.head = nn.Linear(hidden, 3)

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        return self.head(h)


class MLPWide(nn.Module):
    """2-layer wide MLP with Dropout."""
    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, x):
        return self.net(x)


class GRUPredictor(nn.Module):
    """GRU-based sequence model for match prediction.

    Encodes each team's last K matches via GRU, then combines
    with static FootyStats features (PPG, xG, possession).
    """
    def __init__(
        self,
        hidden: int = 128,
        static_dim: int = 0,
        dropout: float = 0.3,
        share_weights: bool = True,
        use_attention: bool = False,
    ):
        super().__init__()
        self.share_weights = share_weights
        self.use_attention = use_attention
        self.hidden = hidden

        self.gru = nn.GRU(3, hidden, batch_first=True)
        if not share_weights:
            self.gru_away = nn.GRU(3, hidden, batch_first=True)
        else:
            self.gru_away = self.gru

        if use_attention:
            self.attn = nn.Linear(hidden, 1)

        total_dim = hidden * 2 + static_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 3),
        )

    def forward(self, seq_h, seq_a, mask, static_feats=None):
        h_h = self._encode(seq_h, mask, self.gru)
        h_a = self._encode(seq_a, mask, self.gru_away)
        combined = [h_h, h_a]
        if static_feats is not None:
            combined.append(static_feats)
        h = torch.cat(combined, dim=1)
        return self.classifier(h)

    def _encode(self, seq, mask, gru):
        """Encode sequence into a single vector."""
        B, K, _ = seq.shape

        if self.use_attention:
            out, _ = gru(seq)
            scores = self.attn(out).squeeze(-1)
            # Mask out padded positions
            scores = scores.masked_fill(mask == 0, -1e9)
            attn_weights = F.softmax(scores, dim=1)
            h = (out * attn_weights.unsqueeze(-1)).sum(dim=1)
        else:
            lengths = mask.sum(dim=1).clamp(min=1).long().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                seq, lengths, batch_first=True, enforce_sorted=False)
            _, h_n = gru(packed)
            h = h_n[-1]
        return h


# ============================================================
# 4. TRAINING
# ============================================================

def train_model(
    model: nn.Module,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    max_epochs: int = 500,
    patience: int = 20,
    is_gru: bool = False,
    seq_h_tr=None, seq_a_tr=None, mask_tr=None,
    seq_h_val=None, seq_a_val=None, mask_val=None,
    verbose: bool = True,
):
    """Train with early stopping + LR scheduling."""
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    # Prepare tensors
    if is_gru:
        tensors = lambda d: (
            torch.FloatTensor(d[0]).to(DEVICE),
            torch.FloatTensor(d[1]).to(DEVICE),
            torch.FloatTensor(d[2]).to(DEVICE),
        )
        tr_t = tensors((seq_h_tr, seq_a_tr, mask_tr))
        val_t = tensors((seq_h_val, seq_a_val, mask_val))
        tr_stat = torch.FloatTensor(X_tr).to(DEVICE) if X_tr.shape[1] > 0 else None
        val_stat = torch.FloatTensor(X_val).to(DEVICE) if X_val.shape[1] > 0 else None
    else:
        tr_t = torch.FloatTensor(X_tr).to(DEVICE)
        val_t = torch.FloatTensor(X_val).to(DEVICE)

    y_tr_t = torch.LongTensor(y_tr).to(DEVICE)
    y_val_t = torch.LongTensor(y_val).to(DEVICE)

    n = len(X_tr)
    best_loss = float("inf")
    best_state = None
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(1, max_epochs + 1):
        # --- Train ---
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        nb = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()

            if is_gru:
                out = model(tr_t[0][idx], tr_t[1][idx], tr_t[2][idx],
                            static_feats=tr_stat[idx] if tr_stat is not None else None)
            else:
                out = model(tr_t[idx])

            loss = criterion(out, y_tr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            nb += 1

        avg_loss = epoch_loss / nb
        train_losses.append(avg_loss)

        # --- Val ---
        model.eval()
        with torch.no_grad():
            if is_gru:
                v_out = model(val_t[0], val_t[1], val_t[2],
                              static_feats=val_stat if val_stat is not None else None)
            else:
                v_out = model(val_t)
            val_loss = criterion(v_out, y_val_t).item()
        val_losses.append(val_loss)

        sched.step(val_loss)

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience and epoch > 50:
                break

        if verbose and epoch % 100 == 0:
            print(f"    Epoch {epoch}: train={avg_loss:.4f} val={val_loss:.4f} "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    return model, train_losses, val_losses


# ============================================================
# 5. EVALUATION
# ============================================================

def evaluate(y_true: np.ndarray, y_prob: np.ndarray, label: str) -> dict:
    """Compute all metrics."""
    y_pred = np.argmax(y_prob, axis=1)
    f1s = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    return {
        "model": label,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "log_loss": round(float(log_loss(y_true, y_prob)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "brier": round(float(np.mean([
            brier_score_loss((y_true == i).astype(int), y_prob[:, i])
            for i in range(3)])), 4),
        "draw_f1": round(float(f1s[1]), 4),
        "home_f1": round(float(f1s[2]), 4),
        "away_f1": round(float(f1s[0]), 4),
    }


# ============================================================
# 6. MODEL REGISTRIES
# ============================================================

FNN_VARIANTS = {
    "MLP-Base": lambda d: MLPBase(d, hidden=256, dropout=0.3),
    "MLP-Wide": lambda d: MLPWide(d, hidden=512, dropout=0.3),
    "MLP-Deep": lambda d: MLPDeep(d, hidden=128, dropout=0.3),
}

GRU_VARIANTS = {
    "GRU-Simple": lambda s: GRUPredictor(hidden=128, static_dim=s, share_weights=True, use_attention=False),
    "GRU-Dual":   lambda s: GRUPredictor(hidden=128, static_dim=s, share_weights=False, use_attention=False),
    "GRU-Attn":   lambda s: GRUPredictor(hidden=128, static_dim=s, share_weights=False, use_attention=True),
}


def predict_model(model, is_gru: bool,
                   X_te, seq_h_te, seq_a_te, mask_te, X_static_te):
    """Get predictions from a trained model."""
    model.eval()
    with torch.no_grad():
        if is_gru:
            out = model(
                torch.FloatTensor(seq_h_te).to(DEVICE),
                torch.FloatTensor(seq_a_te).to(DEVICE),
                torch.FloatTensor(mask_te).to(DEVICE),
                static_feats=torch.FloatTensor(X_static_te).to(DEVICE),
            )
        else:
            out = model(torch.FloatTensor(X_te).to(DEVICE))
        return F.softmax(out, dim=1).cpu().numpy()


# ============================================================
# 7. MAIN
# ============================================================

def main():
    print("=" * 72)
    print("  DEEP LEARNING vs LR EXPERIMENT")
    print("  Premier League 2019-2025, temporal split")
    print("=" * 72)

    torch.manual_seed(42)
    np.random.seed(42)

    # ========== LOAD DATA ==========
    print("\n[1] Loading data...")
    df = load_pl_data()

    # ========== BUILD FEATURES ==========
    df_feat, feat_cols = build_features_for_experiment(df)

    # ========== SPLIT ==========
    X_tr, X_val, X_te, y_tr, y_val, y_te, train_df, val_df, test_df = \
        split_data(df_feat, feat_cols)
    n_feat = X_tr.shape[1]
    print(f"  Feature dim: {n_feat}")

    # ========== BUILD SEQUENCES FOR GRU ==========
    print("\n[Sequence Building]")
    K = 10
    # Build on full sorted data, then split
    full_sorted = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_sorted = full_sorted.sort_values("date").reset_index(drop=True)
    seq_h, seq_a, masks = build_sequences(full_sorted, K)

    n_train, n_val = len(train_df), len(val_df)
    seq_h_tr, seq_a_tr, mask_tr = seq_h[:n_train], seq_a[:n_train], masks[:n_train]
    seq_h_val, seq_a_val, mask_val = seq_h[n_train:n_train + n_val], seq_a[n_train:n_train + n_val], masks[n_train:n_train + n_val]
    seq_h_te = seq_h[n_train + n_val:]
    seq_a_te = seq_a[n_train + n_val:]
    mask_te = masks[n_train + n_val:]

    # Static features for GRU (FootyStats only)
    ft_cols = [c for c in ["pp_home", "pp_away", "ppg_diff",
                            "xg_home", "xg_away", "xg_diff",
                            "poss_home", "poss_away", "poss_diff"]
               if c in df_feat.columns]

    def extract_static(part_df):
        vals = part_df[ft_cols].values.astype(float)
        cm = np.nanmean(vals, axis=0)
        for i in range(vals.shape[1]):
            m = cm[i] if not (np.isnan(cm[i]) or cm[i] != cm[i]) else 0.0
            vals[np.isnan(vals[:, i]), i] = m
        return vals

    ss_ft = StandardScaler()
    X_ft_tr = ss_ft.fit_transform(extract_static(pd.concat([train_df, val_df], ignore_index=True)))
    X_ft_te = ss_ft.transform(extract_static(test_df))

    static_dim = X_ft_tr.shape[1]
    static_tr = X_ft_tr[:n_train]
    static_val = X_ft_tr[n_train:]
    static_te = X_ft_te

    # ========== RUN MODELS ==========
    print("\n" + "=" * 72)
    print("[2] TRAINING MODELS")
    print("=" * 72)
    results = []

    # --- A) LR Baseline ---
    print("\n--- A) Logistic Regression ---")
    lr = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                             max_iter=4000, random_state=42)
    lr.fit(X_tr, y_tr)
    lr_prob = lr.predict_proba(X_te)
    lr_res = evaluate(y_te, lr_prob, "LR")
    results.append(lr_res)
    print(f"  {lr_res}")

    # --- B) FNN Variants ---
    print("\n--- B) PyTorch FNN Variants ---")

    fnn_hp = [
        {"lr": 1e-3, "wd": 1e-5},
        {"lr": 5e-4, "wd": 1e-4},
        {"lr": 1e-3, "wd": 0.0},
    ]

    for name, builder in FNN_VARIANTS.items():
        print(f"\n  [{name}]")
        best_val = float("inf")
        best_res = None

        for hp in fnn_hp:
            model = builder(n_feat)
            print(f"    lr={hp['lr']}, wd={hp['wd']}...", end=" ")
            model, tl, vl = train_model(
                model, X_tr, y_tr, X_val, y_val,
                lr=hp["lr"], weight_decay=hp["wd"],
                is_gru=False, verbose=False)
            # Quick val check
            model.eval()
            with torch.no_grad():
                vp = F.softmax(model(torch.FloatTensor(X_val).to(DEVICE)), dim=1).cpu().numpy()
            vll = log_loss(y_val, vp)
            print(f"val_ll={vll:.4f}")

            if vll < best_val:
                best_val = vll
                # Run full test eval
                te_prob = predict_model(model, False, X_te, None, None, None, None)
                best_res = evaluate(y_te, te_prob, name)

        if best_res:
            results.append(best_res)
            print(f"  >> {name}: acc={best_res['accuracy']}, ll={best_res['log_loss']}")

    # --- C) GRU Models ---
    print("\n--- C) GRU Sequence Models ---")

    for name, builder in GRU_VARIANTS.items():
        print(f"\n  [{name}]")
        best_val = float("inf")
        best_gru = None

        for hp in fnn_hp:  # reuse same HP grid
            model = builder(static_dim)
            print(f"    lr={hp['lr']}, wd={hp['wd']}...", end=" ")
            model, tl, vl = train_model(
                model, static_tr, y_tr, static_val, y_val,
                lr=hp["lr"], weight_decay=hp["wd"],
                is_gru=True,
                seq_h_tr=seq_h_tr, seq_a_tr=seq_a_tr, mask_tr=mask_tr,
                seq_h_val=seq_h_val, seq_a_val=seq_a_val, mask_val=mask_val,
                verbose=False)
            # Quick val check
            model.eval()
            with torch.no_grad():
                vo = model(torch.FloatTensor(seq_h_val).to(DEVICE),
                           torch.FloatTensor(seq_a_val).to(DEVICE),
                           torch.FloatTensor(mask_val).to(DEVICE),
                           static_feats=torch.FloatTensor(static_val).to(DEVICE))
                vll = log_loss(y_val, F.softmax(vo, dim=1).cpu().numpy())
            print(f"val_ll={vll:.4f}")

            if vll < best_val:
                best_val = vll
                best_gru = model

        if best_gru is not None:
            te_prob = predict_model(best_gru, True, None, seq_h_te, seq_a_te, mask_te, static_te)
            res = evaluate(y_te, te_prob, name)
            results.append(res)
            print(f"  >> {name}: acc={res['accuracy']}, ll={res['log_loss']}")

    # ========== RESULTS TABLE ==========
    print("\n" + "=" * 72)
    print("  FINAL COMPARISON TABLE")
    print("=" * 72)
    headers = ["Model", "Accuracy", "LogLoss", "F1(macro)", "Brier", "DrawF1"]
    print(f"  {headers[0]:<16} {headers[1]:<9} {headers[2]:<9} {headers[3]:<10} {headers[4]:<8} {headers[5]:<8}")
    print(f"  {'-'*16} {'-'*9} {'-'*9} {'-'*10} {'-'*8} {'-'*8}")

    results.sort(key=lambda r: r["accuracy"], reverse=True)
    for r in results:
        print(f"  {r['model']:<16} {r['accuracy']:<9.4f} {r['log_loss']:<9.4f} "
              f"{r['f1_macro']:<10.4f} {r['brier']:<8.4f} {r['draw_f1']:<8.4f}")

    # ========== SAVE ==========
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "metrics_comparison.csv", index=False)

    # Save predictions
    preds = pd.DataFrame({
        "home_team": test_df["home_team"].values,
        "away_team": test_df["away_team"].values,
        "date": test_df["date"].values,
        "actual": y_te,
    })
    for r in results:
        name = r["model"]
        if name == "LR":
            prob = lr.predict_proba(X_te)
        elif name in FNN_VARIANTS:
            model = FNN_VARIANTS[name](n_feat)
            model, _, _ = train_model(model, X_tr, y_tr, X_val, y_val, is_gru=False, verbose=False)
            prob = predict_model(model, False, X_te, None, None, None, None)
        elif name in GRU_VARIANTS:
            model = GRU_VARIANTS[name](static_dim)
            model, _, _ = train_model(model, static_tr, y_tr, static_val, y_val,
                                        is_gru=True,
                                        seq_h_tr=seq_h_tr, seq_a_tr=seq_a_tr, mask_tr=mask_tr,
                                        seq_h_val=seq_h_val, seq_a_val=seq_a_val, mask_val=mask_val,
                                        verbose=False)
            prob = predict_model(model, True, None, seq_h_te, seq_a_te, mask_te, static_te)
        else:
            continue
        preds[f"P_H_{name}"] = prob[:, 2]
        preds[f"P_D_{name}"] = prob[:, 1]
        preds[f"P_A_{name}"] = prob[:, 0]
        preds[f"pred_{name}"] = np.argmax(prob, axis=1)

    preds.to_csv(OUTPUT_DIR / "predictions_all.csv", index=False)

    # ========== SUMMARY ==========
    lr_acc = next(r["accuracy"] for r in results if r["model"] == "LR")
    best_acc = results[0]["accuracy"]
    print(f"\n  Baseline LR:      {lr_acc:.4f}")
    print(f"  Best DL model:    {best_acc:.4f} ({results[0]['model']})")
    print(f"  Delta:            {(best_acc - lr_acc) * 100:+.2f}%")
    dl_match = abs(best_acc - lr_acc) < 0.001
    print(f"  DL matches LR accuracy: {'YES' if dl_match else 'NO'}")
    print(f"  DL DrawF1 improvement:  +{results[0].get('draw_f1', 0) - lr_res.get('draw_f1', 0):+.4f}")
    print(f"\n  Results saved to: {OUTPUT_DIR}")
    print("  Done!")


if __name__ == "__main__":
    main()
