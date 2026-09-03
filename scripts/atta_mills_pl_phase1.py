#!/usr/bin/env python3
"""
Phase 1: Atta Mills-style Premier League result prediction.

Scope:
  - Premier League only, football-data.co.uk CSV files.
  - Target: full-time result H/D/A only.
  - Features: Atta Mills-style pre-match team state, attack/defense,
    goals for/against, goal differential, W/D/L history, win/loss margins.
  - Excludes odds, half-time variables, O/U 2.5, FootyStats, H2H, and shots.
  - Evaluation: season-by-season walk-forward.
  - Market benchmark: Bet365 closing odds, de-vigged to implied probabilities.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "league"))

from src.data_loader import load_data  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "atta_mills_pl_walkforward" / "phase1_atta_mills_only"
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["Away", "Draw", "Home"]
RESULT_TO_TEXT = {0: "A", 1: "D", 2: "H"}
PL_FILES = [
    "PL1920.csv",
    "PL2021.csv",
    "PL2122.csv",
    "PL2223.csv",
    "PL2324.csv",
    "PL2425.csv",
    "PL2526.csv",
]
WINDOWS = [5, 10]


def season_from_date(date: pd.Timestamp) -> str:
    year = int(date.year)
    if int(date.month) >= 8:
        return f"{year}-{str(year + 1)[-2:]}"
    return f"{year - 1}-{str(year)[-2:]}"


def load_pl_data() -> pd.DataFrame:
    frames = []
    for fname in PL_FILES:
        path = PROJECT_ROOT / "league" / "data" / "raw" / fname
        if not path.exists():
            print(f"[Data] Skip missing file: {path}")
            continue
        df = load_data(path)
        df["source_file"] = fname
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No Premier League CSV files found.")

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
    }


def _rolling_mean(values: list[float], window: int) -> float:
    if not values:
        return np.nan
    return float(np.mean(values[-window:]))


def _rate(values: list[float], window: int, target: float) -> float:
    if not values:
        return np.nan
    recent = values[-window:]
    return float(np.mean([v == target for v in recent]))


def _team_features(
    team_stats: dict[str, list[float]],
    prefix: str,
    window: int,
    league_avg_gf: float,
    league_avg_ga: float,
) -> dict[str, float]:
    avg_gf = _rolling_mean(team_stats["gf"], window)
    avg_ga = _rolling_mean(team_stats["ga"], window)
    avg_points = _rolling_mean(team_stats["points"], window)
    avg_goal_diff = _rolling_mean(team_stats["goal_diff"], window)
    avg_win_margin = _rolling_mean(team_stats["win_margin"], window)
    avg_loss_margin = _rolling_mean(team_stats["loss_margin"], window)

    attack_strength = avg_gf / league_avg_gf if league_avg_gf and not math.isnan(avg_gf) else np.nan
    defense_strength = avg_ga / league_avg_ga if league_avg_ga and not math.isnan(avg_ga) else np.nan

    return {
        f"{prefix}_team_state_points_{window}": avg_points,
        f"{prefix}_attack_strength_{window}": attack_strength,
        f"{prefix}_defense_strength_{window}": defense_strength,
        f"{prefix}_goals_for_{window}": avg_gf,
        f"{prefix}_goals_against_{window}": avg_ga,
        f"{prefix}_goal_differential_{window}": avg_goal_diff,
        f"{prefix}_win_rate_{window}": _rate(team_stats["result"], window, 1.0),
        f"{prefix}_draw_rate_{window}": _rate(team_stats["result"], window, 0.0),
        f"{prefix}_loss_rate_{window}": _rate(team_stats["result"], window, -1.0),
        f"{prefix}_win_margin_goals_{window}": avg_win_margin,
        f"{prefix}_loss_margin_goals_{window}": avg_loss_margin,
    }


def build_atta_mills_features(df: pd.DataFrame, windows: list[int] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Build only the pre-match feature families used in Phase 1."""
    if windows is None:
        windows = WINDOWS

    df = df.sort_values("date").reset_index(drop=True).copy()
    histories: dict[str, dict[str, list[float]]] = defaultdict(_empty_stats)
    league_gf_history: list[float] = []
    rows: list[dict[str, float]] = []

    for _, match in df.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        league_avg_gf = float(np.mean(league_gf_history)) if league_gf_history else np.nan
        league_avg_ga = league_avg_gf

        row_features: dict[str, float] = {}
        for window in windows:
            row_features.update(_team_features(histories[home], "home", window, league_avg_gf, league_avg_ga))
            row_features.update(_team_features(histories[away], "away", window, league_avg_gf, league_avg_ga))
        rows.append(row_features)

        hg = int(match["home_goals_full"])
        ag = int(match["away_goals_full"])
        home_points = 3 if hg > ag else 1 if hg == ag else 0
        away_points = 3 if ag > hg else 1 if hg == ag else 0
        home_result = 1.0 if hg > ag else 0.0 if hg == ag else -1.0
        away_result = 1.0 if ag > hg else 0.0 if hg == ag else -1.0

        updates = [
            (home, hg, ag, home_points, home_result),
            (away, ag, hg, away_points, away_result),
        ]
        for team, gf, ga, points, result in updates:
            diff = gf - ga
            histories[team]["gf"].append(float(gf))
            histories[team]["ga"].append(float(ga))
            histories[team]["points"].append(float(points))
            histories[team]["result"].append(float(result))
            histories[team]["goal_diff"].append(float(diff))
            histories[team]["win_margin"].append(float(max(diff, 0)))
            histories[team]["loss_margin"].append(float(max(-diff, 0)))

        league_gf_history.extend([float(hg), float(ag)])

    feature_df = pd.DataFrame(rows)
    result = pd.concat([df, feature_df], axis=1)
    feature_cols = list(feature_df.columns)
    return result, feature_cols


def market_probabilities(df: pd.DataFrame) -> np.ndarray:
    """Return Bet365 closing implied probabilities in class order A/D/H."""
    odds_cols = ["odds_home_close", "odds_draw_close", "odds_away_close"]
    odds = df[odds_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    probs = np.full((len(df), 3), np.nan)

    valid = np.all(np.isfinite(odds) & (odds > 1.0), axis=1)
    implied = 1.0 / odds[valid]
    implied = implied / implied.sum(axis=1, keepdims=True)
    probs[valid] = np.column_stack([implied[:, 2], implied[:, 1], implied[:, 0]])

    missing = ~valid
    if missing.any():
        probs[missing] = np.array([1 / 3, 1 / 3, 1 / 3])
    return probs


def multiclass_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean([
        brier_score_loss((y_true == klass).astype(int), y_prob[:, i])
        for i, klass in enumerate(CLASS_LABELS)
    ]))


def summarize_predictions(
    model_name: str,
    fold: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    source: str = "model",
) -> dict[str, float | str | int]:
    y_pred = np.argmax(y_prob, axis=1)
    per_f1 = f1_score(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    return {
        "source": source,
        "model": model_name,
        "fold": fold,
        "n_matches": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, y_prob, labels=CLASS_LABELS)),
        "brier": multiclass_brier(y_true, y_prob),
        "away_f1": float(per_f1[0]),
        "draw_f1": float(per_f1[1]),
        "home_f1": float(per_f1[2]),
    }


def build_models() -> dict[str, object]:
    models: dict[str, object] = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
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
                random_state=42,
                class_weight="balanced",
            )),
        ]),
        "svm": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", SVC(C=1.0, kernel="rbf", probability=True, random_state=42)),
        ]),
        "naive_bayes": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=8,
                class_weight=None,
                n_jobs=1,
                random_state=42,
            )),
        ]),
        "random_forest_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=8,
                class_weight="balanced",
                n_jobs=1,
                random_state=42,
            )),
        ]),
        "fnn_mlp": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", __import__("sklearn.neural_network").neural_network.MLPClassifier(
                hidden_layer_sizes=(64, 128, 128, 128),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate="adaptive",
                learning_rate_init=0.002,
                max_iter=500,
                batch_size=32,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=25,
                random_state=42,
            )),
        ]),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=3,
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
            n_jobs=None,
        )
    except Exception as exc:
        print(f"[Models] XGBoost unavailable, skipping xgboost/voting_rf_xgb: {exc}")

    return models


def feature_mapping_table(feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    mapping_rules: list[tuple[str, str, str]] = [
        ("team_state_points", "Team State", "Rolling average points per match"),
        ("attack_strength", "Attack Strength", "Rolling goals-for relative to league average"),
        ("defense_strength", "Defense Strength", "Rolling goals-against relative to league average"),
        ("goals_for", "Goals Forward", "Rolling average goals scored"),
        ("goals_against", "Goals Against", "Rolling average goals conceded"),
        ("goal_differential", "Goal Differential", "Rolling average goal difference"),
        ("win_rate", "Win History", "Rolling win rate"),
        ("draw_rate", "Draw History", "Rolling draw rate"),
        ("loss_rate", "Loss History", "Rolling loss rate"),
        ("win_margin_goals", "Win Margin Goals", "Rolling average winning margin, zero on non-wins"),
        ("loss_margin_goals", "Loss Margin Goals", "Rolling average losing margin, zero on non-losses"),
    ]
    for col in feature_cols:
        paper_family = "Unmapped"
        interpretation = ""
        for token, family, desc in mapping_rules:
            if token in col:
                paper_family = family
                interpretation = desc
                break
        rows.append({
            "feature": col,
            "paper_family": paper_family,
            "home_or_away": "home" if col.startswith("home_") else "away",
            "window": 5 if col.endswith("_5") else 10 if col.endswith("_10") else "",
            "interpretation": interpretation,
            "used_in_phase1": True,
        })
    return pd.DataFrame(rows)


def write_confusion_and_reports(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    name: str,
    out_dir: Path,
) -> None:
    y_pred = np.argmax(y_prob, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out_dir / f"{name}_confusion_matrix.csv")
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    (out_dir / f"{name}_classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_walk_forward(
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    model_builders: Callable[[], dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seasons = sorted(df_feat["season_id"].dropna().unique())
    if len(seasons) < 2:
        raise ValueError("Need at least two seasons for walk-forward evaluation.")

    fold_rows = []
    prediction_rows = []
    aggregate_probs: dict[str, list[np.ndarray]] = defaultdict(list)
    aggregate_true: list[np.ndarray] = []

    print(f"[Walk-forward] Seasons: {seasons}")
    print(f"[Walk-forward] Features: {len(feature_cols)}")

    for fold_idx, test_season in enumerate(seasons[1:], start=1):
        train_mask = df_feat["season_id"] < test_season
        test_mask = df_feat["season_id"] == test_season
        train_df = df_feat.loc[train_mask].copy()
        test_df = df_feat.loc[test_mask].copy()

        X_train = train_df[feature_cols]
        y_train = train_df["target_result"].astype(int).to_numpy()
        X_test = test_df[feature_cols]
        y_test = test_df["target_result"].astype(int).to_numpy()
        aggregate_true.append(y_test)

        print(f"  Fold {fold_idx}: train < {test_season} ({len(train_df)}), test {test_season} ({len(test_df)})")

        market_prob = market_probabilities(test_df)
        fold_rows.append(summarize_predictions("bet365_closing", test_season, y_test, market_prob, source="market"))
        aggregate_probs["bet365_closing"].append(market_prob)

        fold_models = model_builders()
        for model_name, model in fold_models.items():
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            if hasattr(fitted, "predict_proba"):
                y_prob = fitted.predict_proba(X_test)
            else:
                y_pred = fitted.predict(X_test)
                y_prob = np.zeros((len(y_pred), 3), dtype=float)
                y_prob[np.arange(len(y_pred)), y_pred] = 1.0
            fold_rows.append(summarize_predictions(model_name, test_season, y_test, y_prob))
            aggregate_probs[model_name].append(y_prob)

            pred = np.argmax(y_prob, axis=1)
            for idx, (_, row) in enumerate(test_df.iterrows()):
                prediction_rows.append({
                    "fold": test_season,
                    "model": model_name,
                    "date": row["date"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "actual": int(y_test[idx]),
                    "actual_text": RESULT_TO_TEXT[int(y_test[idx])],
                    "predicted": int(pred[idx]),
                    "predicted_text": RESULT_TO_TEXT[int(pred[idx])],
                    "p_away": float(y_prob[idx, 0]),
                    "p_draw": float(y_prob[idx, 1]),
                    "p_home": float(y_prob[idx, 2]),
                })

    y_all = np.concatenate(aggregate_true)
    aggregate_rows = []
    conf_dir = OUTPUT_DIR / "confusion_matrices"
    conf_dir.mkdir(parents=True, exist_ok=True)
    for name, probs_by_fold in aggregate_probs.items():
        probs = np.vstack(probs_by_fold)
        source = "market" if name == "bet365_closing" else "model"
        aggregate_rows.append(summarize_predictions(name, "ALL", y_all, probs, source=source))
        write_confusion_and_reports(y_all, probs, name, conf_dir)

    return pd.DataFrame(fold_rows), pd.DataFrame(aggregate_rows), pd.DataFrame(prediction_rows)


def write_phase1_doc(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_summary: pd.DataFrame,
    market_row: pd.Series,
    best_row: pd.Series,
) -> None:
    doc = PROJECT_ROOT / "docs" / "atta_mills_phase1_results.md"
    feature_count_by_family = feature_mapping_table(feature_cols)["paper_family"].value_counts().sort_index()
    family_lines = "\n".join([f"| {family} | {count} |" for family, count in feature_count_by_family.items()])
    table = model_summary[[
        "model", "accuracy", "f1_macro", "f1_weighted", "draw_f1", "log_loss", "brier"
    ]].copy()
    for col in ["accuracy", "f1_macro", "f1_weighted", "draw_f1", "log_loss", "brier"]:
        table[col] = table[col].map(lambda x: f"{x:.4f}")
    model_lines = "\n".join([
        f"| {r.model} | {r.accuracy} | {r.f1_macro} | {r.f1_weighted} | {r.draw_f1} | {r.log_loss} | {r.brier} |"
        for r in table.itertuples(index=False)
    ])

    content = f"""# Atta Mills Phase 1 Results

## Scope

- Data: Premier League football-data.co.uk files, {df['season_id'].nunique()} seasons, {len(df)} matches.
- Target: full-time H/D/A result only.
- Validation: season-by-season walk-forward.
- Training exclusions: Bet365 odds, half-time variables, O/U 2.5, FootyStats, H2H, shots.
- Market benchmark: Bet365 closing odds, converted to de-vigged implied probabilities.

## Feature Set

Phase 1 uses {len(feature_cols)} Atta Mills-style pre-match features.

| Paper Feature Family | Count |
|---|---:|
{family_lines}

Detailed feature mapping is saved in `outputs/atta_mills_pl_walkforward/phase1_atta_mills_only/feature_set.csv`.

## Aggregate Results

| Model | Accuracy | Macro F1 | Weighted F1 | Draw F1 | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|---:|
{model_lines}

## Initial Interpretation

Best model by accuracy: `{best_row['model']}` at {best_row['accuracy']:.2%}.
Bet365 closing market accuracy: {market_row['accuracy']:.2%}.

The next decision is whether Phase 1 is strong enough to keep as the main model, or whether to start Phase 2 by adding the project's original extra features such as H2H and rolling shot efficiency.
"""
    doc.write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("Atta Mills Phase 1: PL-only H/D/A walk-forward")
    print("=" * 78)

    df = load_pl_data()
    print(f"[Data] Matches: {len(df)}")
    print(f"[Data] Seasons: {sorted(df['season_id'].unique())}")

    df_feat, feature_cols = build_atta_mills_features(df, windows=WINDOWS)
    feature_map = feature_mapping_table(feature_cols)
    feature_map.to_csv(OUTPUT_DIR / "feature_set.csv", index=False)
    df_feat[["date", "season_id", "home_team", "away_team", "target_result", *feature_cols]].to_csv(
        OUTPUT_DIR / "feature_matrix.csv",
        index=False,
    )

    fold_metrics, aggregate_metrics, predictions = run_walk_forward(df_feat, feature_cols, build_models)
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

    best = model_summary.sort_values(["accuracy", "f1_macro", "draw_f1"], ascending=False).iloc[0]
    market = market_summary.iloc[0]
    write_phase1_doc(df, feature_cols, model_summary, market, best)

    print("\n=== Aggregate model comparison ===")
    print(model_summary[["model", "accuracy", "f1_macro", "draw_f1", "log_loss", "brier"]].to_string(index=False))
    print("\n=== Market benchmark ===")
    print(market_summary[["model", "accuracy", "f1_macro", "draw_f1", "log_loss", "brier"]].to_string(index=False))
    print(f"\nBest model: {best['model']} accuracy={best['accuracy']:.2%}, draw_f1={best['draw_f1']:.4f}")
    print(f"Bet365 closing: accuracy={market['accuracy']:.2%}, draw_f1={market['draw_f1']:.4f}")
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
