#!/usr/bin/env python3
"""
English E0-E3 version of the Phase 3 half-time reproduction.

This keeps the experiment close to the paper-style feature set:
  - 44 Atta Mills-style rolling team-state features.
  - Current-match half-time goals/result features.
  - league_level as an English4 control variable.
  - Optional Bet365 opening odds probabilities.

It excludes the project's original H2H, rolling shot-efficiency, FootyStats,
player, and O/U 2.5 features.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "league"))

from scripts.atta_mills_pl_phase1 import build_atta_mills_features, run_walk_forward  # noqa: E402
from scripts.atta_mills_pl_phase3_halftime import (  # noqa: E402
    build_halftime_features,
    build_opening_odds_features,
)
from src.data_loader import load_data  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "atta_mills_english4_phase3_halftime"
DIV_LEVEL = {"E0": 0, "E1": 1, "E2": 2, "E3": 3}
WINDOWS = [5, 10]


def season_from_date(date: pd.Timestamp) -> str:
    year = int(date.year)
    if int(date.month) >= 8:
        return f"{year}-{str(year + 1)[-2:]}"
    return f"{year - 1}-{str(year)[-2:]}"


def load_english4_data() -> pd.DataFrame:
    frames = []
    for div, level in DIV_LEVEL.items():
        for path in sorted((PROJECT_ROOT / "league" / "data" / "english_raw").glob(f"{div}*.csv")):
            df = load_data(path)
            df["division"] = div
            df["league_level"] = float(level)
            df["source_file"] = path.name
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No E0-E3 files found under league/data/english_raw.")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    df["season_id"] = df["date"].apply(season_from_date)
    return df


def build_english4_models() -> dict[str, object]:
    """Fast LR-only model set for the larger 14k-match half-time run."""
    models: dict[str, object] = {
        "lr": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=4000, C=1.0, solver="lbfgs", random_state=42)),
        ]),
        "lr_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=4000,
                C=1.0,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
            )),
        ]),
    }
    return models


def build_features(df: pd.DataFrame, include_opening_odds: bool) -> tuple[pd.DataFrame, list[str]]:
    df_feat, atta_cols = build_atta_mills_features(df, windows=WINDOWS)
    ht_features, ht_cols, _ = build_halftime_features(df_feat)
    for col in ht_cols:
        df_feat[col] = ht_features[col].values

    feature_cols = atta_cols + ht_cols + ["league_level"]
    if include_opening_odds:
        odds_features, odds_cols, _ = build_opening_odds_features(df_feat)
        for col in odds_cols:
            df_feat[col] = odds_features[col].values
        feature_cols += odds_cols
    return df_feat, feature_cols


def run_config(name: str, include_opening_odds: bool, df: pd.DataFrame) -> dict[str, object]:
    import scripts.atta_mills_pl_phase1 as phase1

    out_dir = OUTPUT_ROOT / name
    phase1.OUTPUT_DIR = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_feat, feature_cols = build_features(df, include_opening_odds=include_opening_odds)
    matrix_cols = ["date", "season_id", "division", "home_team", "away_team", "target_result", *feature_cols]
    df_feat[matrix_cols].to_csv(out_dir / "feature_matrix.csv", index=False)

    fold_metrics, aggregate_metrics, predictions = run_walk_forward(df_feat, feature_cols, build_english4_models)
    fold_metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(out_dir / "predictions_by_fold.csv", index=False)

    aggregate_metrics = aggregate_metrics.sort_values(
        ["source", "accuracy", "f1_macro", "draw_f1"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    models = aggregate_metrics[aggregate_metrics["source"] == "model"].copy()
    market = aggregate_metrics[aggregate_metrics["source"] == "market"].iloc[0]
    models.to_csv(out_dir / "model_comparison_walkforward.csv", index=False)
    aggregate_metrics[aggregate_metrics["source"] == "market"].to_csv(
        out_dir / "market_comparison_bet365_closing.csv",
        index=False,
    )
    aggregate_metrics.to_csv(out_dir / "aggregate_metrics_all.csv", index=False)

    best = models.sort_values(["accuracy", "f1_macro", "draw_f1"], ascending=False).iloc[0]
    return {
        "config": name,
        "features": len(feature_cols),
        "best_model": best["model"],
        "accuracy": float(best["accuracy"]),
        "macro_f1": float(best["f1_macro"]),
        "draw_f1": float(best["draw_f1"]),
        "log_loss": float(best["log_loss"]),
        "brier": float(best["brier"]),
        "market_accuracy": float(market["accuracy"]),
        "market_log_loss": float(market["log_loss"]),
    }


def main() -> None:
    print("=" * 78)
    print("Atta Mills Phase 3: English E0-E3 half-time reproduction")
    print("=" * 78)

    df = load_english4_data()
    print(f"[Data] Matches: {len(df)}")
    print(f"[Data] Divisions: {sorted(df['division'].unique())}")
    print(f"[Data] Seasons: {sorted(df['season_id'].unique())}")

    summaries = [
        run_config("english4_ht_no_odds", False, df),
        run_config("english4_ht_with_b365_opening", True, df),
    ]
    summary_df = pd.DataFrame(summaries)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_ROOT / "summary.csv", index=False)
    (OUTPUT_ROOT / "metadata.json").write_text(json.dumps({
        "data": "English E0-E3 football-data.co.uk",
        "matches": int(len(df)),
        "divisions": sorted(df["division"].unique()),
        "seasons": sorted(df["season_id"].unique()),
        "excluded": ["H2H", "rolling shots", "FootyStats", "player features", "O/U 2.5"],
    }, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
