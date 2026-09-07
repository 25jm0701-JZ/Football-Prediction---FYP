#!/usr/bin/env python3
"""
Track A Poisson Goals Baseline.

This is a Loukas et al. (2024)-style independent double-Poisson model for the
clean Track A English4 setting. It predicts home/away goals first, derives a
scoreline probability matrix, and then aggregates scoreline probabilities into
H/D/A probabilities.

Scope:
  - Data: E0, E1, E2, E3 football-data.co.uk files.
  - Model inputs: historical full-time goals only.
  - Excludes: odds as model inputs, half-time/in-play variables, FootyStats,
    current-match raw statistics, and O/U 2.5.
  - Validation: season-by-season walk-forward.
  - Market benchmark: Bet365 closing H/D/A odds where available.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "league"))

from scripts.atta_mills_pl_phase1 import (  # noqa: E402
    RESULT_TO_TEXT,
    market_probabilities,
    season_from_date,
    summarize_predictions,
)
from src.data_loader import load_data  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tracka_poisson_goals"
DOC_PATH = PROJECT_ROOT / "docs" / "tracka_poisson_goals_results.md"
DIV_LEVEL = {"E0": 0, "E1": 1, "E2": 2, "E3": 3}


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
    df = df.dropna(subset=["date", "home_team", "away_team", "home_goals_full", "away_goals_full"])
    df = df.sort_values("date").reset_index(drop=True)
    df["season_id"] = df["date"].apply(season_from_date)
    return df


def poisson_pmf(lam: float, max_goals: int) -> np.ndarray:
    lam = max(float(lam), 1e-9)
    return np.array([
        math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))
        for k in range(max_goals + 1)
    ])


@dataclass
class LoukasPoissonModel:
    """Independent double-Poisson model with Loukas-style attack/defence terms."""

    max_goals: int = 10
    smoothing: float = 0.05

    mu: float = 0.0
    mu_home: float = 0.0
    avg_home_goals: float = 0.0
    avg_away_goals: float = 0.0
    team_attack: dict[str, float] | None = None
    team_defence: dict[str, float] | None = None
    teams: list[str] | None = None

    def fit(self, matches: pd.DataFrame) -> "LoukasPoissonModel":
        train = matches.copy()
        train["home_goals_full"] = pd.to_numeric(train["home_goals_full"], errors="coerce")
        train["away_goals_full"] = pd.to_numeric(train["away_goals_full"], errors="coerce")
        train = train.dropna(subset=["home_goals_full", "away_goals_full"])

        self.avg_home_goals = float(train["home_goals_full"].mean())
        self.avg_away_goals = float(train["away_goals_full"].mean())
        self.mu = math.log(max(self.avg_away_goals, self.smoothing))
        self.mu_home = math.log(max(self.avg_home_goals, self.smoothing)) - self.mu

        long_rows = []
        for row in train.itertuples(index=False):
            long_rows.append({
                "team": row.home_team,
                "goals_for": float(row.home_goals_full),
                "goals_against": float(row.away_goals_full),
            })
            long_rows.append({
                "team": row.away_team,
                "goals_for": float(row.away_goals_full),
                "goals_against": float(row.home_goals_full),
            })
        team_df = pd.DataFrame(long_rows)
        grouped = team_df.groupby("team").agg(
            matches=("goals_for", "size"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
        )

        self.team_attack = {}
        self.team_defence = {}
        for team, row in grouped.iterrows():
            n = float(row["matches"])
            avg_for = (float(row["goals_for"]) + self.smoothing) / (n + self.smoothing)
            avg_against = (float(row["goals_against"]) + self.smoothing) / (n + self.smoothing)
            self.team_attack[str(team)] = math.log(max(avg_for, self.smoothing)) - self.mu
            self.team_defence[str(team)] = math.log(max(avg_against, self.smoothing)) - self.mu

        self.teams = sorted(self.team_attack)
        return self

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        if self.team_attack is None or self.team_defence is None:
            raise RuntimeError("Model must be fitted before prediction.")

        home_attack = self.team_attack.get(home_team, 0.0)
        away_attack = self.team_attack.get(away_team, 0.0)
        home_defence = self.team_defence.get(home_team, 0.0)
        away_defence = self.team_defence.get(away_team, 0.0)

        lambda_home = math.exp(self.mu + self.mu_home + home_attack + away_defence)
        lambda_away = math.exp(self.mu + away_attack + home_defence)
        return float(lambda_home), float(lambda_away)

    def score_matrix(self, lambda_home: float, lambda_away: float) -> np.ndarray:
        home_pmf = poisson_pmf(lambda_home, self.max_goals)
        away_pmf = poisson_pmf(lambda_away, self.max_goals)
        matrix = np.outer(home_pmf, away_pmf)
        total_mass = matrix.sum()
        if total_mass > 0:
            matrix = matrix / total_mass
        return matrix

    def predict_match(self, home_team: str, away_team: str) -> dict[str, float | int | str]:
        lambda_home, lambda_away = self.expected_goals(home_team, away_team)
        matrix = self.score_matrix(lambda_home, lambda_away)
        p_home = float(np.tril(matrix, k=-1).sum())
        p_draw = float(np.trace(matrix))
        p_away = float(np.triu(matrix, k=1).sum())
        probs = np.array([p_away, p_draw, p_home], dtype=float)
        probs = probs / probs.sum()

        score_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        predicted_class = int(np.argmax(probs))
        return {
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            "pred_home_goals": int(score_idx[0]),
            "pred_away_goals": int(score_idx[1]),
            "p_away": float(probs[0]),
            "p_draw": float(probs[1]),
            "p_home": float(probs[2]),
            "predicted": predicted_class,
            "predicted_text": RESULT_TO_TEXT[predicted_class],
        }

    def predict_frame(self, matches: pd.DataFrame) -> pd.DataFrame:
        rows = [
            self.predict_match(str(row.home_team), str(row.away_team))
            for row in matches.itertuples(index=False)
        ]
        return pd.DataFrame(rows)

    def parameter_frame(self) -> pd.DataFrame:
        if self.team_attack is None or self.team_defence is None or self.teams is None:
            raise RuntimeError("Model must be fitted before parameters can be exported.")
        return pd.DataFrame([
            {
                "team": team,
                "attack_parameter": self.team_attack.get(team, 0.0),
                "defence_parameter": self.team_defence.get(team, 0.0),
            }
            for team in self.teams
        ])


def goal_metrics(predictions: pd.DataFrame) -> dict[str, float | int | str]:
    actual_home = predictions["home_goals_full"].to_numpy(dtype=float)
    actual_away = predictions["away_goals_full"].to_numpy(dtype=float)
    lambda_home = predictions["lambda_home"].to_numpy(dtype=float)
    lambda_away = predictions["lambda_away"].to_numpy(dtype=float)
    pred_home = predictions["pred_home_goals"].to_numpy(dtype=float)
    pred_away = predictions["pred_away_goals"].to_numpy(dtype=float)

    return {
        "n_matches": int(len(predictions)),
        "home_goals_mae_lambda": float(np.mean(np.abs(lambda_home - actual_home))),
        "away_goals_mae_lambda": float(np.mean(np.abs(lambda_away - actual_away))),
        "total_goals_mae_lambda": float(np.mean(np.abs((lambda_home + lambda_away) - (actual_home + actual_away)))),
        "home_goals_exact": float(np.mean(pred_home == actual_home)),
        "away_goals_exact": float(np.mean(pred_away == actual_away)),
        "exact_score": float(np.mean((pred_home == actual_home) & (pred_away == actual_away))),
        "home_goals_within_1": float(np.mean(np.abs(pred_home - actual_home) <= 1)),
        "away_goals_within_1": float(np.mean(np.abs(pred_away - actual_away) <= 1)),
        "total_goals_within_1_lambda": float(
            np.mean(np.abs((lambda_home + lambda_away) - (actual_home + actual_away)) <= 1)
        ),
    }


def summarize_result_predictions(
    model_name: str,
    fold: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    source: str = "model",
) -> dict[str, float | str | int]:
    row = summarize_predictions(model_name, fold, y_true, y_prob, source=source)
    row["home_probability_mean"] = float(np.mean(y_prob[:, 2]))
    row["draw_probability_mean"] = float(np.mean(y_prob[:, 1]))
    row["away_probability_mean"] = float(np.mean(y_prob[:, 0]))
    return row


def run_walk_forward(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seasons = sorted(df["season_id"].dropna().unique())
    if len(seasons) < 2:
        raise ValueError("Need at least two seasons for walk-forward evaluation.")

    fold_rows = []
    goal_rows = []
    prediction_frames = []
    parameter_frames = []
    aggregate_true = []
    aggregate_probs = {"loukas_poisson": [], "bet365_closing": []}

    print(f"[Walk-forward] Seasons: {seasons}")
    for fold_idx, test_season in enumerate(seasons[1:], start=1):
        train = df[df["season_id"] < test_season].copy()
        test = df[df["season_id"] == test_season].copy()
        print(f"  Fold {fold_idx}: train < {test_season} ({len(train)}), test {test_season} ({len(test)})")

        model = LoukasPoissonModel(max_goals=10).fit(train)
        pred = model.predict_frame(test)
        out = pd.concat([test.reset_index(drop=True), pred], axis=1)
        out["fold"] = test_season
        out["actual"] = out["target_result"].astype(int)
        out["actual_text"] = out["actual"].map(RESULT_TO_TEXT)
        out["correct"] = (out["actual"] == out["predicted"]).astype(int)
        out["scoreline_predicted"] = out["pred_home_goals"].astype(str) + "-" + out["pred_away_goals"].astype(str)
        out["scoreline_actual"] = (
            out["home_goals_full"].astype(int).astype(str) + "-" + out["away_goals_full"].astype(int).astype(str)
        )
        prediction_frames.append(out)

        y_true = out["actual"].to_numpy(dtype=int)
        poisson_prob = out[["p_away", "p_draw", "p_home"]].to_numpy(dtype=float)
        market_prob = market_probabilities(test)
        aggregate_true.append(y_true)
        aggregate_probs["loukas_poisson"].append(poisson_prob)
        aggregate_probs["bet365_closing"].append(market_prob)

        fold_rows.append(summarize_result_predictions("loukas_poisson", test_season, y_true, poisson_prob))
        fold_rows.append(summarize_result_predictions("bet365_closing", test_season, y_true, market_prob, source="market"))

        fold_goal = goal_metrics(out)
        fold_goal["fold"] = test_season
        goal_rows.append(fold_goal)

        params = model.parameter_frame()
        params["fold"] = test_season
        params["mu_log_away_avg"] = model.mu
        params["mu_home_log_advantage"] = model.mu_home
        params["avg_home_goals_train"] = model.avg_home_goals
        params["avg_away_goals_train"] = model.avg_away_goals
        parameter_frames.append(params)

    y_all = np.concatenate(aggregate_true)
    aggregate_rows = []
    for name, fold_probs in aggregate_probs.items():
        probs = np.vstack(fold_probs)
        source = "market" if name == "bet365_closing" else "model"
        aggregate_rows.append(summarize_result_predictions(name, "ALL", y_all, probs, source=source))

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    goal_rows.append({"fold": "ALL", **goal_metrics(all_predictions)})
    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(goal_rows),
        pd.concat(parameter_frames, ignore_index=True),
        all_predictions,
    )


def write_results_doc(
    df: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    goal_summary: pd.DataFrame,
) -> None:
    poisson = aggregate_metrics[aggregate_metrics["model"] == "loukas_poisson"].iloc[0]
    market = aggregate_metrics[aggregate_metrics["model"] == "bet365_closing"].iloc[0]
    goals = goal_summary[goal_summary["fold"] == "ALL"].iloc[0]

    fold_table = fold_metrics.copy()
    fold_table = fold_table[["fold", "model", "accuracy", "f1_macro", "draw_f1", "log_loss", "brier"]]
    fold_lines = "\n".join([
        f"| {r.fold} | {r.model} | {r.accuracy:.4f} | {r.f1_macro:.4f} | {r.draw_f1:.4f} | {r.log_loss:.4f} | {r.brier:.4f} |"
        for r in fold_table.itertuples(index=False)
    ])

    goal_cols = [
        "fold",
        "home_goals_mae_lambda",
        "away_goals_mae_lambda",
        "total_goals_mae_lambda",
        "home_goals_exact",
        "away_goals_exact",
        "exact_score",
        "home_goals_within_1",
        "away_goals_within_1",
    ]
    goal_lines = "\n".join([
        (
            f"| {r.fold} | {r.home_goals_mae_lambda:.4f} | {r.away_goals_mae_lambda:.4f} | "
            f"{r.total_goals_mae_lambda:.4f} | {r.home_goals_exact:.4f} | {r.away_goals_exact:.4f} | "
            f"{r.exact_score:.4f} | {r.home_goals_within_1:.4f} | {r.away_goals_within_1:.4f} |"
        )
        for r in goal_summary[goal_cols].itertuples(index=False)
    ])

    content = f"""# Track A Poisson Goals Baseline Results

## Scope

- Branch: `codex/tracka-poisson-goals`.
- Main-line parent: `codex/atta-mills-tracka-english4`.
- Data: English E0-E3 football-data.co.uk files.
- Seasons: {', '.join(sorted(df['season_id'].unique()))}.
- Matches: {len(df)}.
- Model: Loukas et al. (2024)-style independent double Poisson.
- Inputs: historical full-time goals only.
- Validation: season-by-season walk-forward.
- Exclusions: no odds as model inputs, no half-time/in-play variables, no current-match raw statistics, no FootyStats, no O/U 2.5.

## Method

For each test season, the model is fitted only on previous seasons. It estimates
a pooled away-goal baseline, a pooled home advantage, and team attack/defence
parameters from historical goals scored and conceded.

```text
log(lambda_home) = mu + mu_home + attack_home + defence_away
log(lambda_away) = mu + attack_away + defence_home
```

A scoreline matrix is generated from two independent Poisson distributions.
H/D/A probabilities are then aggregated from the scoreline matrix:

```text
P(H) = sum P(home_goals > away_goals)
P(D) = sum P(home_goals = away_goals)
P(A) = sum P(home_goals < away_goals)
```

This keeps the branch as a Track A clean goal-based baseline, not a Track B
FootyStats-enhanced model.

## Aggregate H/D/A Results

| Model | Accuracy | Macro F1 | Draw F1 | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|
| Loukas-style Poisson | {poisson['accuracy']:.4f} | {poisson['f1_macro']:.4f} | {poisson['draw_f1']:.4f} | {poisson['log_loss']:.4f} | {poisson['brier']:.4f} |
| Bet365 closing | {market['accuracy']:.4f} | {market['f1_macro']:.4f} | {market['draw_f1']:.4f} | {market['log_loss']:.4f} | {market['brier']:.4f} |

## Aggregate Goal Results

| Metric | Value |
|---|---:|
| Home goals MAE vs lambda | {goals['home_goals_mae_lambda']:.4f} |
| Away goals MAE vs lambda | {goals['away_goals_mae_lambda']:.4f} |
| Total goals MAE vs lambda total | {goals['total_goals_mae_lambda']:.4f} |
| Exact home goals from modal score | {goals['home_goals_exact']:.4f} |
| Exact away goals from modal score | {goals['away_goals_exact']:.4f} |
| Exact score from modal score | {goals['exact_score']:.4f} |
| Home goals within +/-1 | {goals['home_goals_within_1']:.4f} |
| Away goals within +/-1 | {goals['away_goals_within_1']:.4f} |

## Fold H/D/A Results

| Fold | Model | Accuracy | Macro F1 | Draw F1 | Log Loss | Brier |
|---|---|---:|---:|---:|---:|---:|
{fold_lines}

## Fold Goal Results

| Fold | Home MAE | Away MAE | Total MAE | Home Exact | Away Exact | Exact Score | Home +/-1 | Away +/-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{goal_lines}

## Interpretation

This first run intentionally stays close to Loukas et al.'s simple structure.
It is stricter than the paper's random-sample validation because each fold only
uses seasons before the test season, avoiding same-season leakage.

The branch should remain under Track A as a clean goal-based baseline. A later
decision can test whether Poisson probabilities are useful as additional Track A
features, but this first version evaluates the standalone score model.

## Outputs

- `scripts/tracka_poisson_goals.py`
- `docs/tracka_poisson_goals_results.md`
- `outputs/tracka_poisson_goals/fold_metrics.csv`
- `outputs/tracka_poisson_goals/aggregate_metrics.csv`
- `outputs/tracka_poisson_goals/goal_metrics.csv`
- `outputs/tracka_poisson_goals/predictions_by_fold.csv`
- `outputs/tracka_poisson_goals/poisson_parameters_by_fold.csv`
"""
    DOC_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("Track A Poisson Goals Baseline")
    print("=" * 78)
    df = load_english4_data()
    print(f"[Data] Matches: {len(df)}")
    print(f"[Data] Seasons: {sorted(df['season_id'].unique())}")
    print(f"[Data] Divisions: {df['division'].value_counts().sort_index().to_dict()}")

    fold_metrics, aggregate_metrics, goal_metrics_df, params, predictions = run_walk_forward(df)

    fold_metrics.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    aggregate_metrics.to_csv(OUTPUT_DIR / "aggregate_metrics.csv", index=False)
    goal_metrics_df.to_csv(OUTPUT_DIR / "goal_metrics.csv", index=False)
    params.to_csv(OUTPUT_DIR / "poisson_parameters_by_fold.csv", index=False)

    output_cols = [
        "fold",
        "date",
        "division",
        "home_team",
        "away_team",
        "home_goals_full",
        "away_goals_full",
        "scoreline_actual",
        "scoreline_predicted",
        "lambda_home",
        "lambda_away",
        "p_home",
        "p_draw",
        "p_away",
        "actual_text",
        "predicted_text",
        "correct",
    ]
    predictions[output_cols].to_csv(OUTPUT_DIR / "predictions_by_fold.csv", index=False)

    metadata = {
        "branch": "codex/tracka-poisson-goals",
        "parent": "codex/atta-mills-tracka-english4",
        "method": "Loukas et al. 2024 style independent double Poisson",
        "max_goals": 10,
        "data": "English E0-E3 football-data.co.uk",
        "matches": int(len(df)),
        "seasons": sorted(df["season_id"].unique()),
        "excluded": ["odds as inputs", "half-time/in-play variables", "FootyStats", "O/U 2.5"],
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_results_doc(df, fold_metrics, aggregate_metrics, goal_metrics_df)
    print("\n=== Aggregate H/D/A comparison ===")
    print(aggregate_metrics[["model", "accuracy", "f1_macro", "draw_f1", "log_loss", "brier"]].to_string(index=False))
    print("\n=== Aggregate goal metrics ===")
    all_goals = goal_metrics_df[goal_metrics_df["fold"] == "ALL"].iloc[0]
    for col in [
        "home_goals_mae_lambda",
        "away_goals_mae_lambda",
        "total_goals_mae_lambda",
        "home_goals_exact",
        "away_goals_exact",
        "exact_score",
        "home_goals_within_1",
        "away_goals_within_1",
    ]:
        print(f"{col}: {all_goals[col]:.4f}")
    print(f"\nSaved outputs to: {OUTPUT_DIR}")
    print(f"Updated doc: {DOC_PATH}")


if __name__ == "__main__":
    main()
