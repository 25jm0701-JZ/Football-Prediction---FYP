#!/usr/bin/env python3
"""
Track A: Atta Mills-style English four-league walk-forward experiment.

This extends the PL-only Atta Mills branch to English E0-E3 as one integrated
dataset. The promotion/relegation logic follows the original Track A idea:
teams keep their historical rolling form across seasons and divisions, while a
`league_level` feature marks the match level.

Scope:
  - Data: E0, E1, E2, E3 football-data.co.uk files.
  - Target: full-time H/D/A result.
  - Training features: Phase 2 clean pre-match features + league_level.
  - Excludes: odds as training features, half-time variables, O/U 2.5,
    FootyStats, and current-match leaked statistics.
  - Market benchmark: Bet365 closing odds where available.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "league"))

from scripts.atta_mills_pl_phase1 import run_walk_forward
from src.data_loader import load_data
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "atta_mills_tracka_english4"
DIV_LEVEL = {"E0": 0, "E1": 1, "E2": 2, "E3": 3}


def build_tracka_models() -> dict[str, object]:
    """Smaller CPU model set for the 14k-match E0-E3 experiment."""
    models: dict[str, object] = {
        "lr": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", LogisticRegression(max_iter=4000, C=1.0, solver="lbfgs", random_state=42)),
        ]),
        "lr_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", LogisticRegression(
                max_iter=4000,
                C=1.0,
                solver="lbfgs",
                random_state=42,
                class_weight="balanced",
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", RandomForestClassifier(
                n_estimators=250,
                max_depth=10,
                min_samples_leaf=8,
                n_jobs=1,
                random_state=42,
            )),
        ]),
        "random_forest_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", RandomForestClassifier(
                n_estimators=250,
                max_depth=10,
                min_samples_leaf=8,
                class_weight="balanced",
                n_jobs=1,
                random_state=42,
            )),
        ]),
    }

    if XGBClassifier is not None:
        models["xgboost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", XGBClassifier(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="multi:softprob",
                eval_metric="mlogloss",
                n_jobs=1,
                random_state=42,
            )),
        ])
        models["voting_rf_xgb"] = VotingClassifier(
            estimators=[
                ("rf", clone(models["random_forest"])),
                ("xgb", clone(models["xgboost"])),
            ],
            voting="soft",
        )
    return models


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
            df["league_level"] = level
            df["source_file"] = path.name
            frames.append(df)

    if not frames:
        raise FileNotFoundError("No E0-E3 files found under league/data/english_raw.")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    df["season_id"] = df["date"].apply(season_from_date)
    return df


def _empty_stats() -> dict[str, list[float]]:
    return {
        "gf": [],
        "ga": [],
        "points": [],
        "result": [],
        "goal_diff": [],
        "win_margin": [],
        "loss_margin": [],
        "shots": [],
        "sot": [],
    }


def _rolling_mean(values: list[float], window: int) -> float:
    clean = [v for v in values[-window:] if not math.isnan(v)]
    return float(np.mean(clean)) if clean else np.nan


def _rolling_rate(values: list[float], window: int, target: float) -> float:
    if not values:
        return np.nan
    recent = values[-window:]
    return float(np.mean([v == target for v in recent]))


def _team_features(
    stats: dict[str, list[float]],
    prefix: str,
    window: int,
    league_avg_goals: float,
) -> dict[str, float]:
    avg_gf = _rolling_mean(stats["gf"], window)
    avg_ga = _rolling_mean(stats["ga"], window)
    avg_shots = _rolling_mean(stats["shots"], window)
    avg_sot = _rolling_mean(stats["sot"], window)
    shot_accuracy = avg_sot / avg_shots if avg_shots and not math.isnan(avg_shots) else np.nan

    gf_recent = [v for v in stats["gf"][-window:] if not math.isnan(v)]
    shots_recent = [v for v in stats["shots"][-window:] if not math.isnan(v)]
    conversion_rate = (
        float(np.sum(gf_recent) / np.sum(shots_recent))
        if shots_recent and np.sum(shots_recent) > 0
        else np.nan
    )

    return {
        f"{prefix}_team_state_points_{window}": _rolling_mean(stats["points"], window),
        f"{prefix}_attack_strength_{window}": avg_gf / league_avg_goals if league_avg_goals and not math.isnan(avg_gf) else np.nan,
        f"{prefix}_defense_strength_{window}": avg_ga / league_avg_goals if league_avg_goals and not math.isnan(avg_ga) else np.nan,
        f"{prefix}_goals_for_{window}": avg_gf,
        f"{prefix}_goals_against_{window}": avg_ga,
        f"{prefix}_goal_differential_{window}": _rolling_mean(stats["goal_diff"], window),
        f"{prefix}_win_rate_{window}": _rolling_rate(stats["result"], window, 1.0),
        f"{prefix}_draw_rate_{window}": _rolling_rate(stats["result"], window, 0.0),
        f"{prefix}_loss_rate_{window}": _rolling_rate(stats["result"], window, -1.0),
        f"{prefix}_win_margin_goals_{window}": _rolling_mean(stats["win_margin"], window),
        f"{prefix}_loss_margin_goals_{window}": _rolling_mean(stats["loss_margin"], window),
        f"{prefix}_avg_shots_{window}": avg_shots,
        f"{prefix}_avg_sot_{window}": avg_sot,
        f"{prefix}_shot_accuracy_{window}": shot_accuracy,
        f"{prefix}_conversion_rate_{window}": conversion_rate,
    }


def build_tracka_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    df = df.sort_values("date").reset_index(drop=True).copy()
    histories: dict[str, dict[str, list[float]]] = defaultdict(_empty_stats)
    pair_histories: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    league_goals: list[float] = []
    rows = []

    for _, match in df.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        league_avg_goals = float(np.mean(league_goals)) if league_goals else np.nan
        row_features = {}

        for window in [5, 10]:
            row_features.update(_team_features(histories[home], "home", window, league_avg_goals))
            row_features.update(_team_features(histories[away], "away", window, league_avg_goals))

        pair = tuple(sorted([home, away]))
        previous = pair_histories[pair][-5:]
        row_features["h2h_home_wins"] = sum(1 for item in previous if item["winner"] == home)
        row_features["h2h_away_wins"] = sum(1 for item in previous if item["winner"] == away)
        row_features["h2h_draws"] = sum(1 for item in previous if item["winner"] == "D")
        row_features["h2h_home_avg_goals"] = (
            float(np.mean([
                item["home_goals"] if item["home"] == home else item["away_goals"]
                for item in previous
            ]))
            if previous else 0.0
        )
        row_features["h2h_away_avg_goals"] = (
            float(np.mean([
                item["away_goals"] if item["away"] == away else item["home_goals"]
                for item in previous
            ]))
            if previous else 0.0
        )
        row_features["league_level"] = float(match["league_level"])
        rows.append(row_features)

        hg = int(match["home_goals_full"])
        ag = int(match["away_goals_full"])
        hs = float(match["home_shots"]) if pd.notna(match.get("home_shots")) else np.nan
        away_shots = float(match["away_shots"]) if pd.notna(match.get("away_shots")) else np.nan
        hst = float(match["home_shots_target"]) if pd.notna(match.get("home_shots_target")) else np.nan
        ast = float(match["away_shots_target"]) if pd.notna(match.get("away_shots_target")) else np.nan

        updates = [
            (home, hg, ag, hs, hst),
            (away, ag, hg, away_shots, ast),
        ]
        for team, gf, ga, shots, sot in updates:
            points = 3 if gf > ga else 1 if gf == ga else 0
            result = 1.0 if gf > ga else 0.0 if gf == ga else -1.0
            diff = gf - ga
            histories[team]["gf"].append(float(gf))
            histories[team]["ga"].append(float(ga))
            histories[team]["points"].append(float(points))
            histories[team]["result"].append(result)
            histories[team]["goal_diff"].append(float(diff))
            histories[team]["win_margin"].append(float(max(diff, 0)))
            histories[team]["loss_margin"].append(float(max(-diff, 0)))
            histories[team]["shots"].append(shots)
            histories[team]["sot"].append(sot)

        pair_histories[pair].append({
            "home": home,
            "away": away,
            "home_goals": hg,
            "away_goals": ag,
            "winner": home if hg > ag else away if ag > hg else "D",
        })
        league_goals.extend([float(hg), float(ag)])

    features = pd.DataFrame(rows)
    df_feat = pd.concat([df, features], axis=1)
    feature_cols = list(features.columns)

    mapping_rows = []
    for col in feature_cols:
        if col == "league_level":
            family = "Track A: League Level"
            desc = "0=Premier League, 1=Championship, 2=League One, 3=League Two"
        elif "avg_shots" in col:
            family = "Original Extra: Rolling Shots"
            desc = "Rolling average shots"
        elif "avg_sot" in col:
            family = "Original Extra: Rolling Shots on Target"
            desc = "Rolling average shots on target"
        elif "shot_accuracy" in col:
            family = "Original Extra: Shot Accuracy"
            desc = "Rolling shots-on-target divided by shots"
        elif "conversion_rate" in col:
            family = "Original Extra: Conversion Rate"
            desc = "Rolling goals divided by shots"
        elif col.startswith("h2h_"):
            family = "Original Extra: Head-to-Head"
            desc = "Previous meetings between the teams"
        elif "team_state_points" in col:
            family = "Team State"
            desc = "Rolling average points"
        elif "attack_strength" in col:
            family = "Attack Strength"
            desc = "Rolling goals-for relative to league average"
        elif "defense_strength" in col:
            family = "Defense Strength"
            desc = "Rolling goals-against relative to league average"
        elif "goals_for" in col:
            family = "Goals Forward"
            desc = "Rolling average goals scored"
        elif "goals_against" in col:
            family = "Goals Against"
            desc = "Rolling average goals conceded"
        elif "goal_differential" in col:
            family = "Goal Differential"
            desc = "Rolling average goal difference"
        elif "win_rate" in col:
            family = "Win History"
            desc = "Rolling win rate"
        elif "draw_rate" in col:
            family = "Draw History"
            desc = "Rolling draw rate"
        elif "loss_rate" in col:
            family = "Loss History"
            desc = "Rolling loss rate"
        elif "win_margin" in col:
            family = "Win Margin Goals"
            desc = "Rolling average winning margin"
        elif "loss_margin" in col:
            family = "Loss Margin Goals"
            desc = "Rolling average losing margin"
        else:
            family = "Other"
            desc = ""
        mapping_rows.append({
            "feature": col,
            "feature_family": family,
            "home_or_away": "home" if col.startswith("home_") else "away" if col.startswith("away_") else "match",
            "window": 5 if col.endswith("_5") else 10 if col.endswith("_10") else "",
            "interpretation": desc,
            "used_in_tracka": True,
        })

    mapping = pd.DataFrame(mapping_rows)
    return df_feat, feature_cols, mapping


def write_doc(
    df: pd.DataFrame,
    feature_cols: list[str],
    aggregate_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> None:
    doc = PROJECT_ROOT / "docs" / "atta_mills_tracka_english4_results.md"
    models = aggregate_metrics[aggregate_metrics["source"] == "model"].copy()
    market = aggregate_metrics[aggregate_metrics["source"] == "market"].iloc[0]
    best = models.sort_values(["accuracy", "f1_macro", "draw_f1"], ascending=False).iloc[0]

    model_table = models[[
        "model", "accuracy", "f1_macro", "f1_weighted", "draw_f1", "log_loss", "brier"
    ]].copy()
    for col in ["accuracy", "f1_macro", "f1_weighted", "draw_f1", "log_loss", "brier"]:
        model_table[col] = model_table[col].map(lambda x: f"{x:.4f}")
    model_lines = "\n".join([
        f"| {r.model} | {r.accuracy} | {r.f1_macro} | {r.f1_weighted} | {r.draw_f1} | {r.log_loss} | {r.brier} |"
        for r in model_table.itertuples(index=False)
    ])

    division_counts = df["division"].value_counts().sort_index()
    div_lines = "\n".join([f"| {div} | {count} |" for div, count in division_counts.items()])

    fold_best = fold_metrics[fold_metrics["source"] == "model"].copy()
    fold_best = fold_best.sort_values(["fold", "accuracy"], ascending=[True, False]).groupby("fold").head(1)
    fold_lines = "\n".join([
        f"| {r.fold} | {r.model} | {r.accuracy:.4f} | {r.f1_macro:.4f} | {r.draw_f1:.4f} |"
        for r in fold_best.itertuples(index=False)
    ])

    content = f"""# Atta Mills Track A English4 Results

## Scope

- Data: English E0-E3 football-data.co.uk files.
- Seasons: {', '.join(sorted(df['season_id'].unique()))}.
- Matches: {len(df)}.
- Target: full-time H/D/A result only.
- Validation: season-by-season walk-forward.
- Training exclusions: bookmaker odds, half-time variables, O/U 2.5, FootyStats.
- Market benchmark: Bet365 closing odds, converted to de-vigged implied probabilities.

## Promotion/Relegation Logic

This experiment uses E0-E3 as one integrated chronological dataset. Team rolling
histories are keyed by team name rather than division, so promoted/relegated
teams carry their previous form into the next division. The `league_level`
feature marks match level:

- E0 = 0
- E1 = 1
- E2 = 2
- E3 = 3

## Data Coverage

| Division | Matches |
|---|---:|
{div_lines}

## Feature Set

Total features: {len(feature_cols)}.

- 44 Atta Mills-style pre-match features.
- 16 rolling shot-efficiency features.
- 5 H2H features.
- 1 league-level feature for Track A promotion/relegation handling.

Feature mapping is saved in `outputs/atta_mills_tracka_english4/feature_set.csv`.

## Aggregate Results

| Model | Accuracy | Macro F1 | Weighted F1 | Draw F1 | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|---:|
{model_lines}

Bet365 closing benchmark:

| Model | Accuracy | Macro F1 | Draw F1 | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|
| {market['model']} | {market['accuracy']:.4f} | {market['f1_macro']:.4f} | {market['draw_f1']:.4f} | {market['log_loss']:.4f} | {market['brier']:.4f} |

## Best Model By Fold

| Fold/Test Season | Best Model | Accuracy | Macro F1 | Draw F1 |
|---|---|---:|---:|---:|
{fold_lines}

## Interpretation

Best model by aggregate accuracy is `{best['model']}` at {best['accuracy']:.2%}.
This Track A version tests whether adding E1-E3 improves the clean pre-match
Atta Mills branch by increasing training volume and preserving team history
across promotion/relegation.
"""
    doc.write_text(content, encoding="utf-8")


def main() -> None:
    import scripts.atta_mills_pl_phase1 as phase1

    phase1.OUTPUT_DIR = OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Atta Mills Track A: English E0-E3 walk-forward")
    print("=" * 78)
    df = load_english4_data()
    print(f"[Data] Matches: {len(df)}")
    print(f"[Data] Seasons: {sorted(df['season_id'].unique())}")
    print(f"[Data] Divisions: {df['division'].value_counts().sort_index().to_dict()}")

    df_feat, feature_cols, mapping = build_tracka_features(df)
    print(f"[Features] Total: {len(feature_cols)}")
    mapping.to_csv(OUTPUT_DIR / "feature_set.csv", index=False)
    matrix_cols = ["date", "season_id", "division", "league_level", "home_team", "away_team", "target_result", *feature_cols]
    df_feat[matrix_cols].to_csv(OUTPUT_DIR / "feature_matrix.csv", index=False)

    fold_metrics, aggregate_metrics, predictions = run_walk_forward(df_feat, feature_cols, build_tracka_models)
    fold_metrics.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "predictions_by_fold.csv", index=False)

    aggregate_metrics = aggregate_metrics.sort_values(
        ["source", "accuracy", "f1_macro", "draw_f1"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    model_summary = aggregate_metrics[aggregate_metrics["source"] == "model"].copy()
    market_summary = aggregate_metrics[aggregate_metrics["source"] == "market"].copy()
    model_summary.to_csv(OUTPUT_DIR / "model_comparison_walkforward.csv", index=False)
    market_summary.to_csv(OUTPUT_DIR / "market_comparison_bet365_closing.csv", index=False)
    aggregate_metrics.to_csv(OUTPUT_DIR / "aggregate_metrics_all.csv", index=False)

    metadata = {
        "data": "English E0-E3 football-data.co.uk",
        "matches": int(len(df)),
        "seasons": sorted(df["season_id"].unique()),
        "divisions": {k: int(v) for k, v in df["division"].value_counts().sort_index().to_dict().items()},
        "features": int(len(feature_cols)),
        "promotion_relegation_logic": "team rolling histories are carried across divisions; league_level marks division strength",
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_doc(df, feature_cols, aggregate_metrics, fold_metrics)

    print("\n=== Aggregate model comparison ===")
    print(model_summary[["model", "accuracy", "f1_macro", "draw_f1", "log_loss", "brier"]].to_string(index=False))
    print("\n=== Market benchmark ===")
    print(market_summary[["model", "accuracy", "f1_macro", "draw_f1", "log_loss", "brier"]].to_string(index=False))
    best = model_summary.sort_values(["accuracy", "f1_macro", "draw_f1"], ascending=False).iloc[0]
    market = market_summary.iloc[0]
    print(f"\nBest model: {best['model']} accuracy={best['accuracy']:.2%}, draw_f1={best['draw_f1']:.4f}")
    print(f"Bet365 closing: accuracy={market['accuracy']:.2%}, draw_f1={market['draw_f1']:.4f}")
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
