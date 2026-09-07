#!/usr/bin/env python3
"""
Track A high-confidence prediction analysis.

Uses the current clean Track A 5+10 baseline and evaluates whether model
predictions become more useful when restricted to matches where the model's
maximum predicted class probability is high.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, log_loss

warnings.filterwarnings("ignore")
sys.path.insert(0, "league")
sys.path.insert(0, "scripts")

from src.features import build_features, get_feature_columns
from tracka_window_abtest import (
    brier_multiclass,
    load_english4,
    market_probabilities,
    model_registry,
    predict_prob,
)


OUT_DIR = Path("outputs/tracka_confidence_analysis")
WINDOWS = [5, 10]
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
LABELS = {0: "A", 1: "D", 2: "H"}


def raw_closing_odds(df: pd.DataFrame) -> np.ndarray:
    odds = df[["odds_home_close", "odds_draw_close", "odds_away_close"]].to_numpy(dtype=float)
    odds[odds <= 0] = np.nan
    return np.column_stack([odds[:, 2], odds[:, 1], odds[:, 0]])


def value_roi(
    y_true: np.ndarray,
    model_prob: np.ndarray,
    market_prob: np.ndarray,
    closing_odds: np.ndarray,
    mask: np.ndarray,
    edge_threshold: float = 0.10,
) -> tuple[int, int, float, float]:
    bets = 0
    wins = 0
    pnl = 0.0
    stake = 100.0
    for i in np.where(mask)[0]:
        best_edge = 0.0
        best_outcome = -1
        for outcome in range(3):
            if market_prob[i, outcome] <= 0:
                continue
            edge = model_prob[i, outcome] / market_prob[i, outcome] - 1
            if edge > best_edge:
                best_edge = edge
                best_outcome = outcome
        if best_edge < edge_threshold or best_outcome < 0:
            continue
        price = closing_odds[i, best_outcome]
        if np.isnan(price):
            continue
        bets += 1
        won = best_outcome == y_true[i]
        wins += int(won)
        pnl += (price - 1) * stake if won else -stake
    roi = pnl / (bets * stake) if bets else np.nan
    win_rate = wins / bets if bets else np.nan
    return bets, wins, win_rate, roi


def summarize_thresholds(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model_name, group in predictions.groupby("model"):
        y_true = group["y_true"].to_numpy(dtype=int)
        model_prob = group[["prob_A", "prob_D", "prob_H"]].to_numpy(dtype=float)
        market_prob = group[["market_prob_A", "market_prob_D", "market_prob_H"]].to_numpy(dtype=float)
        odds = group[["odds_A", "odds_D", "odds_H"]].to_numpy(dtype=float)
        pred = group["pred"].to_numpy(dtype=int)
        conf = group["confidence"].to_numpy(dtype=float)
        market_pred = np.argmax(market_prob, axis=1)

        for threshold in THRESHOLDS:
            mask = conf >= threshold
            n = int(mask.sum())
            if n == 0:
                continue
            prob_subset = model_prob[mask]
            pred_subset = pred[mask]
            y_subset = y_true[mask]
            market_subset = market_prob[mask]
            market_pred_subset = market_pred[mask]

            bets, wins, bet_win_rate, roi = value_roi(
                y_true=y_true,
                model_prob=model_prob,
                market_prob=market_prob,
                closing_odds=odds,
                mask=mask,
                edge_threshold=0.10,
            )
            counts = pd.Series(pred_subset).value_counts().to_dict()
            rows.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    "n_matches": n,
                    "coverage": n / len(group),
                    "accuracy": accuracy_score(y_subset, pred_subset),
                    "market_accuracy_same_matches": accuracy_score(y_subset, market_pred_subset),
                    "macro_f1": f1_score(y_subset, pred_subset, labels=[0, 1, 2], average="macro"),
                    "draw_f1": f1_score(y_subset, pred_subset, labels=[0, 1, 2], average=None)[1],
                    "log_loss": log_loss(y_subset, prob_subset, labels=[0, 1, 2]),
                    "brier": brier_multiclass(y_subset, prob_subset),
                    "pred_A": int(counts.get(0, 0)),
                    "pred_D": int(counts.get(1, 0)),
                    "pred_H": int(counts.get(2, 0)),
                    "value_bets_edge10": bets,
                    "value_bet_wins_edge10": wins,
                    "value_bet_win_rate_edge10": bet_win_rate,
                    "value_bet_roi_edge10": roi,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hist = load_english4()
    print(
        f"Loaded E0-E3: {len(hist)} matches, "
        f"{hist['season_id'].nunique()} seasons, divisions={hist['division'].value_counts().to_dict()}",
        flush=True,
    )

    feat_df = build_features(
        hist,
        windows=WINDOWS,
        include_odds=False,
        include_shots=True,
        include_h2h=True,
        include_footystats=False,
    )
    odds_df = build_features(
        hist,
        windows=WINDOWS,
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
    print(f"Features={len(feature_cols)} models={list(models)}", flush=True)

    prediction_rows = []
    fold_rows = []
    for test_season in seasons[1:]:
        train_seasons = [s for s in seasons if s < test_season]
        train_mask = feat_df["season_id"].isin(train_seasons).to_numpy()
        test_mask = (feat_df["season_id"] == test_season).to_numpy()

        x_train = feat_df.loc[train_mask, feature_cols].to_numpy(dtype=float)
        y_train = feat_df.loc[train_mask, "target_result"].to_numpy(dtype=int)
        x_test = feat_df.loc[test_mask, feature_cols].to_numpy(dtype=float)
        y_test = feat_df.loc[test_mask, "target_result"].to_numpy(dtype=int)
        match_df = feat_df.loc[test_mask, ["date", "division", "home_team", "away_team"]].reset_index(drop=True)
        market_prob = market_probabilities(odds_df.loc[test_mask])
        odds = raw_closing_odds(odds_df.loc[test_mask])

        line = [test_season]
        for model_name, base_model in models.items():
            model = clone(base_model)
            model.fit(x_train, y_train)
            prob = predict_prob(model, x_test)
            pred = np.argmax(prob, axis=1)
            conf = np.max(prob, axis=1)
            acc = accuracy_score(y_test, pred)
            line.append(f"{model_name}={acc * 100:.1f}%")
            fold_rows.append(
                {
                    "season": test_season,
                    "model": model_name,
                    "n_matches": int(len(y_test)),
                    "accuracy": acc,
                    "macro_f1": f1_score(y_test, pred, labels=[0, 1, 2], average="macro"),
                    "draw_f1": f1_score(y_test, pred, labels=[0, 1, 2], average=None)[1],
                    "log_loss": log_loss(y_test, prob, labels=[0, 1, 2]),
                    "brier": brier_multiclass(y_test, prob),
                }
            )
            for i in range(len(y_test)):
                prediction_rows.append(
                    {
                        "season": test_season,
                        "model": model_name,
                        "date": match_df.loc[i, "date"],
                        "division": match_df.loc[i, "division"],
                        "home_team": match_df.loc[i, "home_team"],
                        "away_team": match_df.loc[i, "away_team"],
                        "y_true": int(y_test[i]),
                        "true_label": LABELS[int(y_test[i])],
                        "pred": int(pred[i]),
                        "pred_label": LABELS[int(pred[i])],
                        "confidence": float(conf[i]),
                        "prob_A": float(prob[i, 0]),
                        "prob_D": float(prob[i, 1]),
                        "prob_H": float(prob[i, 2]),
                        "market_prob_A": float(market_prob[i, 0]),
                        "market_prob_D": float(market_prob[i, 1]),
                        "market_prob_H": float(market_prob[i, 2]),
                        "odds_A": float(odds[i, 0]) if not np.isnan(odds[i, 0]) else np.nan,
                        "odds_D": float(odds[i, 1]) if not np.isnan(odds[i, 1]) else np.nan,
                        "odds_H": float(odds[i, 2]) if not np.isnan(odds[i, 2]) else np.nan,
                    }
                )
        print("  " + " | ".join(line), flush=True)

    predictions = pd.DataFrame(prediction_rows)
    folds = pd.DataFrame(fold_rows)
    threshold_summary = summarize_thresholds(predictions)
    aggregate = (
        folds.groupby("model", as_index=False)
        .agg(
            n_matches=("n_matches", "sum"),
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            draw_f1=("draw_f1", "mean"),
            log_loss=("log_loss", "mean"),
            brier=("brier", "mean"),
        )
        .sort_values("accuracy", ascending=False)
    )

    predictions.to_csv(OUT_DIR / "predictions.csv", index=False)
    folds.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    threshold_summary.to_csv(OUT_DIR / "confidence_summary.csv", index=False)
    aggregate.to_csv(OUT_DIR / "aggregate_metrics.csv", index=False)
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "purpose": "High-confidence analysis for clean Track A 5+10 baseline.",
                "data": "football-data.co.uk English E0-E3, no odds in training.",
                "windows": WINDOWS,
                "thresholds": THRESHOLDS,
                "market_benchmark": "Bet365 closing, evaluated on the same selected matches.",
                "value_betting": "Best-per-match, edge >= 10%, stake=100.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Aggregate Metrics ===", flush=True)
    display = aggregate.copy()
    for col in ["accuracy", "macro_f1", "draw_f1", "log_loss", "brier"]:
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    print(display.to_string(index=False), flush=True)

    print("\n=== Confidence Summary ===", flush=True)
    compact = threshold_summary[
        [
            "model",
            "threshold",
            "n_matches",
            "coverage",
            "accuracy",
            "market_accuracy_same_matches",
            "pred_A",
            "pred_D",
            "pred_H",
            "value_bets_edge10",
            "value_bet_roi_edge10",
        ]
    ].copy()
    for col in ["coverage", "accuracy", "market_accuracy_same_matches", "value_bet_roi_edge10"]:
        compact[col] = compact[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    print(compact.to_string(index=False), flush=True)
    print(f"\nSaved: {OUT_DIR / 'confidence_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
