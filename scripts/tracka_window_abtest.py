#!/usr/bin/env python3
"""
Track A rolling-window A/B test.

Compares the current Track A window choice (5+10 matches) against a
Luiz et al. (2024)-inspired 20-match-only window. Everything else is kept
as close as possible: English E0-E3, clean pre-match features, no odds in
training, Bet365 closing odds as the market benchmark, and season-by-season
walk-forward validation.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, "league")

from src.data_loader import load_data
from src.features import build_features, get_feature_columns

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


DIV_LEVEL = {"E0": 0, "E1": 1, "E2": 2, "E3": 3}
OUT_DIR = Path("outputs/tracka_window_abtest")


def season_id(date: pd.Timestamp) -> str:
    year = date.year
    return f"{year}-{str(year + 1)[-2:]}" if date.month >= 8 else f"{year - 1}-{str(year)[-2:]}"


def load_english4() -> pd.DataFrame:
    frames = []
    for div, level in DIV_LEVEL.items():
        for path in sorted(Path("league/data/english_raw").glob(f"{div}*.csv")):
            df = load_data(path)
            df["league_level"] = level
            df["division"] = div
            frames.append(df)
    hist = pd.concat(frames, ignore_index=True)
    hist = hist.sort_values("date").reset_index(drop=True)
    hist["season_id"] = hist["date"].apply(season_id)
    return hist


def market_probabilities(df: pd.DataFrame) -> np.ndarray:
    probs = df[["impl_home_prob_close", "impl_draw_prob_close", "impl_away_prob_close"]].to_numpy(dtype=float)
    missing = np.isnan(probs).any(axis=1)
    probs[missing] = [1 / 3, 1 / 3, 1 / 3]
    probs = probs / probs.sum(axis=1, keepdims=True)
    return np.column_stack([probs[:, 2], probs[:, 1], probs[:, 0]])


def brier_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    one_hot = np.eye(3)[y_true]
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1) / 3))


def model_registry() -> dict[str, object]:
    models: dict[str, object] = {
        "lr": make_pipeline(
            SimpleImputer(strategy="mean"),
            LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=4000,
                C=1.0,
                random_state=42,
            ),
        ),
        "rf": make_pipeline(
            SimpleImputer(strategy="mean"),
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=1,
            ),
        ),
    }
    if os.getenv("INCLUDE_MLP", "").lower() in {"1", "true", "yes"}:
        from sklearn.neural_network import MLPClassifier

        models["mlp"] = make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 128, 128, 128),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate="adaptive",
                learning_rate_init=0.002,
                max_iter=250,
                batch_size=32,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42,
                verbose=False,
            ),
        )
    if HAS_XGB:
        models["xgb"] = make_pipeline(
            SimpleImputer(strategy="mean"),
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=1,
                eval_metric="mlogloss",
            ),
        )
    return models


def predict_prob(model: object, x_test: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(x_test)
    classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    aligned = np.zeros((len(probs), 3), dtype=float)
    for src_idx, cls in enumerate(classes):
        aligned[:, int(cls)] = probs[:, src_idx]
    return aligned


def evaluate_config(hist: pd.DataFrame, label: str, windows: list[int]) -> tuple[pd.DataFrame, dict]:
    print(f"\n=== Config {label}: windows={windows} ===", flush=True)
    feat_df = build_features(
        hist,
        windows=windows,
        include_odds=False,
        include_shots=True,
        include_h2h=True,
        include_footystats=False,
    )
    odds_df = build_features(
        hist,
        windows=windows,
        include_odds=True,
        include_shots=True,
        include_h2h=True,
        include_footystats=False,
    )

    feat_df["season_id"] = hist["season_id"].values
    odds_df["season_id"] = hist["season_id"].values
    if "league_level" not in feat_df.columns:
        feat_df["league_level"] = hist["league_level"].values

    feature_cols = get_feature_columns(feat_df)
    feature_cols = [c for c in feature_cols if c != "season_id"]
    feature_cols = sorted({*feature_cols, "league_level"} & set(feat_df.columns))
    seasons = sorted(feat_df["season_id"].unique())
    models = model_registry()
    print(f"Matches={len(feat_df)} seasons={len(seasons)} features={len(feature_cols)} models={list(models)}", flush=True)

    rows = []
    for test_season in seasons[1:]:
        train_seasons = [s for s in seasons if s < test_season]
        train_mask = feat_df["season_id"].isin(train_seasons).to_numpy()
        test_mask = (feat_df["season_id"] == test_season).to_numpy()

        x_train = feat_df.loc[train_mask, feature_cols].to_numpy(dtype=float)
        y_train = feat_df.loc[train_mask, "target_result"].to_numpy(dtype=int)
        x_test = feat_df.loc[test_mask, feature_cols].to_numpy(dtype=float)
        y_test = feat_df.loc[test_mask, "target_result"].to_numpy(dtype=int)

        market_prob = market_probabilities(odds_df.loc[test_mask])
        rows.append(
            {
                "config": label,
                "windows": "+".join(map(str, windows)),
                "season": test_season,
                "model": "bet365_closing",
                "n_matches": int(len(y_test)),
                "n_features": 0,
                "accuracy": accuracy_score(y_test, np.argmax(market_prob, axis=1)),
                "macro_f1": f1_score(y_test, np.argmax(market_prob, axis=1), labels=[0, 1, 2], average="macro"),
                "draw_f1": f1_score(y_test, np.argmax(market_prob, axis=1), labels=[0, 1, 2], average=None)[1],
                "log_loss": log_loss(y_test, market_prob, labels=[0, 1, 2]),
                "brier": brier_multiclass(y_test, market_prob),
            }
        )

        line = [test_season]
        for model_name, base_model in models.items():
            model = clone(base_model)
            model.fit(x_train, y_train)
            prob = predict_prob(model, x_test)
            pred = np.argmax(prob, axis=1)
            acc = accuracy_score(y_test, pred)
            rows.append(
                {
                    "config": label,
                    "windows": "+".join(map(str, windows)),
                    "season": test_season,
                    "model": model_name,
                    "n_matches": int(len(y_test)),
                    "n_features": int(len(feature_cols)),
                    "accuracy": acc,
                    "macro_f1": f1_score(y_test, pred, labels=[0, 1, 2], average="macro"),
                    "draw_f1": f1_score(y_test, pred, labels=[0, 1, 2], average=None)[1],
                    "log_loss": log_loss(y_test, prob, labels=[0, 1, 2]),
                    "brier": brier_multiclass(y_test, prob),
                }
            )
            line.append(f"{model_name}={acc * 100:.1f}%")
        print("  " + " | ".join(line), flush=True)

    fold_df = pd.DataFrame(rows)
    summary = (
        fold_df.groupby(["config", "windows", "model"], as_index=False)
        .agg(
            n_matches=("n_matches", "sum"),
            n_features=("n_features", "max"),
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            draw_f1=("draw_f1", "mean"),
            log_loss=("log_loss", "mean"),
            brier=("brier", "mean"),
        )
        .sort_values(["config", "accuracy"], ascending=[True, False])
    )
    return fold_df, {"label": label, "windows": windows, "feature_count": len(feature_cols), "summary": summary}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hist = load_english4()
    print(
        f"Loaded E0-E3: {len(hist)} matches, "
        f"{hist['season_id'].nunique()} seasons, divisions={hist['division'].value_counts().to_dict()}"
    , flush=True)

    configs = {
        "A_5_10": [5, 10],
        "B_20_only": [20],
    }

    all_folds = []
    summaries = []
    metadata = {
        "purpose": "A/B test current Track A 5+10 rolling windows against 20-only windows.",
        "data": "football-data.co.uk English E0-E3, clean pre-match features, no odds in training.",
        "market_benchmark": "Bet365 closing implied probabilities.",
        "configs": configs,
    }
    for label, windows in configs.items():
        fold_df, result = evaluate_config(hist, label, windows)
        all_folds.append(fold_df)
        summaries.append(result["summary"])
        metadata[label] = {"windows": windows, "feature_count": result["feature_count"]}

    folds = pd.concat(all_folds, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    folds.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== Aggregate Summary ===", flush=True)
    display = summary.copy()
    for col in ["accuracy", "macro_f1", "draw_f1", "log_loss", "brier"]:
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    print(display.to_string(index=False), flush=True)
    print(f"\nSaved: {OUT_DIR / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
