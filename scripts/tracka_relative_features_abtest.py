#!/usr/bin/env python3
"""
Track A Luiz-inspired relative-feature A/B test.

Compares the current clean Track A 5+10 rolling-window baseline against the
same feature set plus interpretable relative-strength features inspired by
Luiz et al. (2024). The experiment keeps the data, validation, model set, and
market benchmark aligned with scripts/tracka_window_abtest.py.
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


OUT_DIR = Path("outputs/tracka_relative_features_abtest")
WINDOWS = [5, 10]


def add_relative_features(df: pd.DataFrame, windows: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """Add relative home-vs-away features using already-lagged rolling inputs."""
    result = df.copy()
    added: list[str] = []

    def add_col(name: str, values: pd.Series) -> None:
        result[name] = values
        added.append(name)

    for w in windows:
        required = [
            f"home_avg_goals_for_{w}",
            f"away_avg_goals_for_{w}",
            f"home_avg_goals_against_{w}",
            f"away_avg_goals_against_{w}",
            f"home_avg_goal_diff_{w}",
            f"away_avg_goal_diff_{w}",
            f"home_avg_points_{w}",
            f"away_avg_points_{w}",
            f"home_win_rate_{w}",
            f"away_win_rate_{w}",
            f"home_draw_rate_{w}",
            f"away_draw_rate_{w}",
            f"home_loss_rate_{w}",
            f"away_loss_rate_{w}",
        ]
        if not all(c in result.columns for c in required):
            missing = [c for c in required if c not in result.columns]
            raise KeyError(f"Missing rolling columns for window {w}: {missing}")

        add_col(f"rel_avg_goals_for_{w}", result[f"home_avg_goals_for_{w}"] - result[f"away_avg_goals_for_{w}"])
        add_col(
            f"rel_avg_goals_against_{w}",
            result[f"away_avg_goals_against_{w}"] - result[f"home_avg_goals_against_{w}"],
        )
        add_col(f"rel_avg_goal_diff_{w}", result[f"home_avg_goal_diff_{w}"] - result[f"away_avg_goal_diff_{w}"])
        add_col(f"rel_avg_points_{w}", result[f"home_avg_points_{w}"] - result[f"away_avg_points_{w}"])
        add_col(f"rel_win_rate_{w}", result[f"home_win_rate_{w}"] - result[f"away_win_rate_{w}"])
        add_col(f"rel_draw_rate_{w}", result[f"home_draw_rate_{w}"] - result[f"away_draw_rate_{w}"])
        add_col(f"rel_loss_rate_{w}", result[f"away_loss_rate_{w}"] - result[f"home_loss_rate_{w}"])

        home_attack = result[f"home_avg_goals_for_{w}"] - result[f"away_avg_goals_against_{w}"]
        away_attack = result[f"away_avg_goals_for_{w}"] - result[f"home_avg_goals_against_{w}"]
        add_col(f"home_attack_vs_away_defense_{w}", home_attack)
        add_col(f"away_attack_vs_home_defense_{w}", away_attack)
        add_col(f"relative_attack_defense_gap_{w}", home_attack - away_attack)

        shot_cols = [
            f"home_avg_shots_{w}",
            f"away_avg_shots_{w}",
            f"home_avg_sot_{w}",
            f"away_avg_sot_{w}",
            f"home_shot_accuracy_{w}",
            f"away_shot_accuracy_{w}",
            f"home_conversion_rate_{w}",
            f"away_conversion_rate_{w}",
        ]
        if all(c in result.columns for c in shot_cols):
            add_col(f"rel_avg_shots_{w}", result[f"home_avg_shots_{w}"] - result[f"away_avg_shots_{w}"])
            add_col(f"rel_avg_sot_{w}", result[f"home_avg_sot_{w}"] - result[f"away_avg_sot_{w}"])
            add_col(
                f"rel_shot_accuracy_{w}",
                result[f"home_shot_accuracy_{w}"] - result[f"away_shot_accuracy_{w}"],
            )
            add_col(
                f"rel_conversion_rate_{w}",
                result[f"home_conversion_rate_{w}"] - result[f"away_conversion_rate_{w}"],
            )

    return result, added


def evaluate(hist: pd.DataFrame, label: str, include_relative: bool) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    print(f"\n=== Config {label}: relative={include_relative} ===", flush=True)
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

    relative_cols: list[str] = []
    if include_relative:
        feat_df, relative_cols = add_relative_features(feat_df, WINDOWS)

    feature_cols = get_feature_columns(feat_df)
    feature_cols = [c for c in feature_cols if c != "season_id"]
    feature_cols = sorted({*feature_cols, "league_level", *relative_cols} & set(feat_df.columns))
    seasons = sorted(feat_df["season_id"].unique())
    models = model_registry()
    print(
        f"Matches={len(feat_df)} seasons={len(seasons)} features={len(feature_cols)} "
        f"relative_features={len(relative_cols)} models={list(models)}",
        flush=True,
    )

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
        market_pred = np.argmax(market_prob, axis=1)
        rows.append(
            {
                "config": label,
                "relative": include_relative,
                "season": test_season,
                "model": "bet365_closing",
                "n_matches": int(len(y_test)),
                "n_features": 0,
                "n_relative_features": 0,
                "accuracy": accuracy_score(y_test, market_pred),
                "macro_f1": f1_score(y_test, market_pred, labels=[0, 1, 2], average="macro"),
                "draw_f1": f1_score(y_test, market_pred, labels=[0, 1, 2], average=None)[1],
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
                    "relative": include_relative,
                    "season": test_season,
                    "model": model_name,
                    "n_matches": int(len(y_test)),
                    "n_features": int(len(feature_cols)),
                    "n_relative_features": int(len(relative_cols)),
                    "accuracy": acc,
                    "macro_f1": f1_score(y_test, pred, labels=[0, 1, 2], average="macro"),
                    "draw_f1": f1_score(y_test, pred, labels=[0, 1, 2], average=None)[1],
                    "log_loss": log_loss(y_test, prob, labels=[0, 1, 2]),
                    "brier": brier_multiclass(y_test, prob),
                }
            )
            line.append(f"{model_name}={acc * 100:.1f}%")
        print("  " + " | ".join(line), flush=True)

    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby(["config", "relative", "model"], as_index=False)
        .agg(
            n_matches=("n_matches", "sum"),
            n_features=("n_features", "max"),
            n_relative_features=("n_relative_features", "max"),
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            draw_f1=("draw_f1", "mean"),
            log_loss=("log_loss", "mean"),
            brier=("brier", "mean"),
        )
        .sort_values(["config", "accuracy"], ascending=[True, False])
    )
    return folds, summary, len(relative_cols)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hist = load_english4()
    print(
        f"Loaded E0-E3: {len(hist)} matches, "
        f"{hist['season_id'].nunique()} seasons, divisions={hist['division'].value_counts().to_dict()}",
        flush=True,
    )

    configs = {
        "A_5_10_baseline": False,
        "B_5_10_relative": True,
    }
    all_folds = []
    all_summaries = []
    metadata = {
        "purpose": "A/B test Luiz-inspired relative features on the current Track A 5+10 setup.",
        "data": "football-data.co.uk English E0-E3, clean pre-match features, no odds in training.",
        "windows": WINDOWS,
        "market_benchmark": "Bet365 closing implied probabilities.",
        "configs": configs,
    }
    for label, include_relative in configs.items():
        folds, summary, rel_count = evaluate(hist, label, include_relative)
        all_folds.append(folds)
        all_summaries.append(summary)
        metadata[label] = {"include_relative": include_relative, "relative_feature_count": rel_count}

    fold_df = pd.concat(all_folds, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    fold_df.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== Aggregate Summary ===", flush=True)
    display = summary_df.copy()
    for col in ["accuracy", "macro_f1", "draw_f1", "log_loss", "brier"]:
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    print(display.to_string(index=False), flush=True)
    print(f"\nSaved: {OUT_DIR / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
