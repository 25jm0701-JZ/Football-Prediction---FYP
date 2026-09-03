#!/usr/bin/env python3
"""
Local GPU PyTorch MLP walk-forward for Atta Mills Phase 2 features.

Uses:
  - 44 Atta Mills-style pre-match features.
  - 16 rolling shot-efficiency features.
  - 5 H2H features.

Still excludes odds from training, half-time variables, O/U 2.5, and FootyStats.
Run with:
  .\\.venv_cuda\\Scripts\\python.exe scripts\\atta_mills_pl_gpu_mlp.py
"""
from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.atta_mills_pl_phase1 import CLASS_LABELS, market_probabilities
from scripts.atta_mills_pl_phase2 import build_phase2_features, load_pl_data

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "atta_mills_pl_walkforward" / "phase2_gpu_mlp"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def seed_everything() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.relu(self.bn1(self.fc1(x))))
        h = self.drop(self.bn2(self.fc2(h)))
        return F.relu(x + h)


class MLPDeep(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.30) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(128, dropout) for _ in range(4)])
        self.head = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.input(x)))


class MLPWide(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.30) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


MODEL_BUILDERS = {
    "torch_mlp_deep": MLPDeep,
    "torch_mlp_wide": MLPWide,
}


def prepare_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]):
    val_start = max(1, int(len(train_df) * 0.85))
    inner_train = train_df.iloc[:val_start].copy()
    val_df = train_df.iloc[val_start:].copy()

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(inner_train[feature_cols])).astype(np.float32)
    x_val = scaler.transform(imputer.transform(val_df[feature_cols])).astype(np.float32)
    x_test = scaler.transform(imputer.transform(test_df[feature_cols])).astype(np.float32)

    y_train = inner_train["target_result"].astype(int).to_numpy()
    y_val = val_df["target_result"].astype(int).to_numpy()
    y_test = test_df["target_result"].astype(int).to_numpy()
    return x_train, y_train, x_val, y_val, x_test, y_test


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    lr: float,
    weight_decay: float,
    batch_size: int = 64,
    max_epochs: int = 500,
    patience: int = 35,
) -> tuple[nn.Module, float, int]:
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
    )
    criterion = nn.CrossEntropyLoss()

    x_tr = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
    y_tr = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    x_v = torch.tensor(x_val, dtype=torch.float32, device=DEVICE)
    y_v = torch.tensor(y_val, dtype=torch.long, device=DEVICE)

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(x_tr), device=DEVICE)
        for start in range(0, len(x_tr), batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_v), y_v).item()
        scheduler.step(val_loss)

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience and epoch >= 60:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss, best_epoch


def predict_proba(model: nn.Module, x_test: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)
        return F.softmax(model(x), dim=1).detach().cpu().numpy()


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean([
        brier_score_loss((y_true == klass).astype(int), y_prob[:, i])
        for i, klass in enumerate(CLASS_LABELS)
    ]))


def metrics(model: str, fold: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = np.argmax(y_prob, axis=1)
    per_f1 = f1_score(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    return {
        "model": model,
        "fold": fold,
        "n_matches": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, y_prob, labels=CLASS_LABELS)),
        "brier": brier(y_true, y_prob),
        "away_f1": float(per_f1[0]),
        "draw_f1": float(per_f1[1]),
        "home_f1": float(per_f1[2]),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything()

    print("=" * 78)
    print("Local GPU MLP walk-forward on Atta Mills Phase 2 features")
    print("=" * 78)
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    df = load_pl_data()
    df_feat, feature_cols, _ = build_phase2_features(df)
    seasons = sorted(df_feat["season_id"].unique())
    print(f"Matches: {len(df_feat)} | Seasons: {seasons} | Features: {len(feature_cols)}")

    hp_grid = [
        {"lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.30},
        {"lr": 5e-4, "weight_decay": 1e-4, "dropout": 0.35},
        {"lr": 1e-3, "weight_decay": 1e-5, "dropout": 0.25},
    ]
    fold_rows = []
    prediction_rows = []
    all_true = []
    all_probs = {name: [] for name in MODEL_BUILDERS}
    all_market = []
    started = time.time()

    for fold_idx, test_season in enumerate(seasons[1:], start=1):
        train_df = df_feat[df_feat["season_id"] < test_season].copy()
        test_df = df_feat[df_feat["season_id"] == test_season].copy()
        x_train, y_train, x_val, y_val, x_test, y_test = prepare_fold(train_df, test_df, feature_cols)
        all_true.append(y_test)
        all_market.append(market_probabilities(test_df))
        print(f"\nFold {fold_idx}: train < {test_season} ({len(train_df)}), test {test_season} ({len(test_df)})")

        for name, builder in MODEL_BUILDERS.items():
            best = None
            for hp in hp_grid:
                seed_everything()
                model = builder(x_train.shape[1], dropout=hp["dropout"])
                model, val_loss, best_epoch = train_model(
                    model,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    lr=hp["lr"],
                    weight_decay=hp["weight_decay"],
                )
                probs = predict_proba(model, x_test)
                row = metrics(name, test_season, y_test, probs)
                row.update(hp)
                row["val_loss"] = float(val_loss)
                row["best_epoch"] = int(best_epoch)
                if best is None or val_loss < best["val_loss"]:
                    best = {**row, "probs": probs}

            assert best is not None
            probs = best.pop("probs")
            fold_rows.append(best)
            all_probs[name].append(probs)
            print(
                f"  {name}: acc={best['accuracy']:.3f}, "
                f"macro={best['f1_macro']:.3f}, draw={best['draw_f1']:.3f}, "
                f"ll={best['log_loss']:.3f}"
            )

            pred = np.argmax(probs, axis=1)
            for idx, (_, match) in enumerate(test_df.iterrows()):
                prediction_rows.append({
                    "fold": test_season,
                    "model": name,
                    "date": match["date"],
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "actual": int(y_test[idx]),
                    "predicted": int(pred[idx]),
                    "p_away": float(probs[idx, 0]),
                    "p_draw": float(probs[idx, 1]),
                    "p_home": float(probs[idx, 2]),
                })

    y_all = np.concatenate(all_true)
    aggregate_rows = [metrics(name, "ALL", y_all, np.vstack(chunks)) for name, chunks in all_probs.items()]
    aggregate_rows.append(metrics("bet365_closing", "ALL", y_all, np.vstack(all_market)))
    fold_df = pd.DataFrame(fold_rows)
    aggregate_df = pd.DataFrame(aggregate_rows).sort_values("accuracy", ascending=False)
    pred_df = pd.DataFrame(prediction_rows)
    runtime = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "runtime_seconds": round(time.time() - started, 2),
    }

    fold_df.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    aggregate_df.to_csv(OUTPUT_DIR / "model_comparison_walkforward.csv", index=False)
    pred_df.to_csv(OUTPUT_DIR / "predictions_by_fold.csv", index=False)
    (OUTPUT_DIR / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    print("\n=== Aggregate ===")
    print(aggregate_df[["model", "accuracy", "f1_macro", "draw_f1", "log_loss", "brier"]].to_string(index=False))
    print(f"\nRuntime: {runtime}")
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
