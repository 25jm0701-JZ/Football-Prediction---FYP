from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.international_training_data import standardize_team


def poisson_probabilities(expected_goals: float, max_goals: int) -> np.ndarray:
    if expected_goals <= 0:
        raise ValueError("expected_goals must be positive.")
    probabilities = np.empty(max_goals + 1, dtype=float)
    probabilities[0] = math.exp(-expected_goals)
    for goals in range(1, max_goals + 1):
        probabilities[goals] = (
            probabilities[goals - 1] * expected_goals / goals
        )
    return probabilities


def score_matrix(
    home_expected_goals: float,
    away_expected_goals: float,
    max_goals: int = 10,
) -> np.ndarray:
    home = poisson_probabilities(home_expected_goals, max_goals)
    away = poisson_probabilities(away_expected_goals, max_goals)
    matrix = np.outer(home, away)
    return matrix / matrix.sum()


def score_summary(
    home_expected_goals: float,
    away_expected_goals: float,
    max_goals: int = 10,
    top_n: int = 3,
) -> dict:
    matrix = score_matrix(home_expected_goals, away_expected_goals, max_goals)
    home_probability = float(np.tril(matrix, k=-1).sum())
    draw_probability = float(np.trace(matrix))
    away_probability = float(np.triu(matrix, k=1).sum())

    ranked = np.argsort(matrix.ravel())[::-1][:top_n]
    scores = []
    for flat_index in ranked:
        home_goals, away_goals = np.unravel_index(flat_index, matrix.shape)
        scores.append(
            {
                "score": f"{home_goals}-{away_goals}",
                "probability": float(matrix[home_goals, away_goals]),
            }
        )
    return {
        "P_H": home_probability,
        "P_D": draw_probability,
        "P_A": away_probability,
        "scores": scores,
    }


@dataclass
class WeightedPoissonModel:
    teams: list[str] | None = None
    intercept: float = 0.0
    home_advantage: float = 0.0
    attack: np.ndarray | None = None
    defence_weakness: np.ndarray | None = None
    l2: float = 0.08
    learning_rate: float = 0.03
    max_iter: int = 12000
    tolerance: float = 1e-10
    fitted_iterations: int = 0
    training_loss: float | None = None

    def fit(self, matches: pd.DataFrame) -> "WeightedPoissonModel":
        required = {
            "Home",
            "Away",
            "HomeGoals",
            "AwayGoals",
            "FinalWeight",
            "TrueHome",
        }
        missing = required - set(matches.columns)
        if missing:
            raise ValueError(f"Training data is missing columns: {sorted(missing)}")

        frame = matches.copy()
        frame["Home"] = frame["Home"].map(standardize_team)
        frame["Away"] = frame["Away"].map(standardize_team)
        self.teams = sorted(set(frame["Home"]) | set(frame["Away"]))
        team_index = {team: index for index, team in enumerate(self.teams)}
        home_index = frame["Home"].map(team_index).to_numpy(dtype=int)
        away_index = frame["Away"].map(team_index).to_numpy(dtype=int)
        home_goals = frame["HomeGoals"].to_numpy(dtype=float)
        away_goals = frame["AwayGoals"].to_numpy(dtype=float)
        weights = frame["FinalWeight"].to_numpy(dtype=float)
        true_home = (
            pd.to_numeric(frame["TrueHome"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        if np.any(weights <= 0):
            raise ValueError("FinalWeight values must be positive.")

        n_teams = len(self.teams)
        denominator = 2.0 * weights.sum()
        weighted_goal_mean = float(
            np.sum(weights * (home_goals + away_goals)) / denominator
        )
        intercept = math.log(max(weighted_goal_mean, 1e-6))
        home_advantage = 0.0
        attack = np.zeros(n_teams, dtype=float)
        defence = np.zeros(n_teams, dtype=float)

        parameter_count = 2 + 2 * n_teams
        first_moment = np.zeros(parameter_count, dtype=float)
        second_moment = np.zeros(parameter_count, dtype=float)
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        previous_loss = np.inf

        for iteration in range(1, self.max_iter + 1):
            home_eta = (
                intercept
                + attack[home_index]
                + defence[away_index]
                + home_advantage * true_home
            )
            away_eta = intercept + attack[away_index] + defence[home_index]
            home_lambda = np.exp(np.clip(home_eta, -5.0, 5.0))
            away_lambda = np.exp(np.clip(away_eta, -5.0, 5.0))

            data_loss = np.sum(
                weights
                * (
                    home_lambda
                    - home_goals * home_eta
                    + away_lambda
                    - away_goals * away_eta
                )
            ) / denominator
            penalty = 0.5 * self.l2 * (
                np.mean(attack**2) + np.mean(defence**2)
            )
            loss = float(data_loss + penalty)

            home_error = weights * (home_lambda - home_goals) / denominator
            away_error = weights * (away_lambda - away_goals) / denominator
            gradient_intercept = float(home_error.sum() + away_error.sum())
            gradient_home = float(np.sum(home_error * true_home))
            gradient_attack = (
                np.bincount(home_index, weights=home_error, minlength=n_teams)
                + np.bincount(away_index, weights=away_error, minlength=n_teams)
                + self.l2 * attack / n_teams
            )
            gradient_defence = (
                np.bincount(away_index, weights=home_error, minlength=n_teams)
                + np.bincount(home_index, weights=away_error, minlength=n_teams)
                + self.l2 * defence / n_teams
            )
            gradient = np.concatenate(
                (
                    [gradient_intercept, gradient_home],
                    gradient_attack,
                    gradient_defence,
                )
            )

            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
            first_hat = first_moment / (1.0 - beta1**iteration)
            second_hat = second_moment / (1.0 - beta2**iteration)
            step = (
                self.learning_rate
                * first_hat
                / (np.sqrt(second_hat) + epsilon)
            )
            intercept -= step[0]
            home_advantage -= step[1]
            attack -= step[2 : 2 + n_teams]
            defence -= step[2 + n_teams :]

            attack_mean = float(attack.mean())
            defence_mean = float(defence.mean())
            attack -= attack_mean
            defence -= defence_mean
            intercept += attack_mean + defence_mean

            if abs(previous_loss - loss) < self.tolerance:
                self.fitted_iterations = iteration
                self.training_loss = loss
                break
            previous_loss = loss
        else:
            self.fitted_iterations = self.max_iter
            self.training_loss = loss

        self.intercept = float(intercept)
        self.home_advantage = float(home_advantage)
        self.attack = attack
        self.defence_weakness = defence
        return self

    def _team_index(self) -> dict[str, int]:
        if self.teams is None or self.attack is None or self.defence_weakness is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return {team: index for index, team in enumerate(self.teams)}

    def expected_goals(
        self,
        home: str,
        away: str,
        host_advantage: int = 0,
    ) -> tuple[float, float]:
        index = self._team_index()
        home = standardize_team(home)
        away = standardize_team(away)
        missing = {home, away} - set(index)
        if missing:
            raise ValueError(f"Teams not found in Poisson model: {sorted(missing)}")

        home_host = max(int(host_advantage), 0)
        away_host = max(-int(host_advantage), 0)
        home_eta = (
            self.intercept
            + self.attack[index[home]]
            + self.defence_weakness[index[away]]
            + self.home_advantage * home_host
        )
        away_eta = (
            self.intercept
            + self.attack[index[away]]
            + self.defence_weakness[index[home]]
            + self.home_advantage * away_host
        )
        return (
            float(np.exp(np.clip(home_eta, -5.0, 5.0))),
            float(np.exp(np.clip(away_eta, -5.0, 5.0))),
        )

    def predict_fixtures(
        self,
        fixtures: pd.DataFrame,
        max_goals: int = 10,
        top_n: int = 3,
    ) -> pd.DataFrame:
        output = fixtures.copy()
        rows = []
        for _, row in output.iterrows():
            host_advantage = int(row.get("HostAdvantage", 0))
            home_xg, away_xg = self.expected_goals(
                row["Home"],
                row["Away"],
                host_advantage,
            )
            summary = score_summary(home_xg, away_xg, max_goals, top_n)
            scores = summary["scores"]
            result = {
                "Poisson_xG_H": home_xg,
                "Poisson_xG_A": away_xg,
                "PoissonP_H": summary["P_H"],
                "PoissonP_D": summary["P_D"],
                "PoissonP_A": summary["P_A"],
                "MostLikelyScore": scores[0]["score"],
                "MostLikelyScoreP": scores[0]["probability"],
            }
            for rank, score in enumerate(scores, start=1):
                result[f"Score{rank}"] = score["score"]
                result[f"Score{rank}P"] = score["probability"]
            rows.append(result)
        return pd.concat([output.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    def training_metrics(self, matches: pd.DataFrame) -> dict:
        expected_home = []
        expected_away = []
        for _, row in matches.iterrows():
            true_home = 0 if pd.isna(row["TrueHome"]) else int(row["TrueHome"])
            home_xg, away_xg = self.expected_goals(
                row["Home"],
                row["Away"],
                host_advantage=true_home,
            )
            expected_home.append(home_xg)
            expected_away.append(away_xg)

        home_goals = matches["HomeGoals"].to_numpy(dtype=float)
        away_goals = matches["AwayGoals"].to_numpy(dtype=float)
        home_lambda = np.asarray(expected_home)
        away_lambda = np.asarray(expected_away)
        weights = matches["FinalWeight"].to_numpy(dtype=float)
        return {
            "matches": int(len(matches)),
            "teams": int(len(self.teams or [])),
            "weighted_mae_home": float(
                np.average(np.abs(home_goals - home_lambda), weights=weights)
            ),
            "weighted_mae_away": float(
                np.average(np.abs(away_goals - away_lambda), weights=weights)
            ),
            "weighted_actual_goals": float(
                np.average(home_goals + away_goals, weights=weights)
            ),
            "weighted_expected_goals": float(
                np.average(home_lambda + away_lambda, weights=weights)
            ),
            "training_loss": self.training_loss,
            "iterations": self.fitted_iterations,
        }

    def to_dict(self) -> dict:
        self._team_index()
        return {
            "method": "weighted independent double Poisson",
            "teams": self.teams,
            "intercept": self.intercept,
            "home_advantage": self.home_advantage,
            "attack": self.attack.tolist(),
            "defence_weakness": self.defence_weakness.tolist(),
            "l2": self.l2,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "fitted_iterations": self.fitted_iterations,
            "training_loss": self.training_loss,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WeightedPoissonModel":
        return cls(
            teams=list(payload["teams"]),
            intercept=float(payload["intercept"]),
            home_advantage=float(payload["home_advantage"]),
            attack=np.asarray(payload["attack"], dtype=float),
            defence_weakness=np.asarray(payload["defence_weakness"], dtype=float),
            l2=float(payload.get("l2", 0.08)),
            learning_rate=float(payload.get("learning_rate", 0.03)),
            max_iter=int(payload.get("max_iter", 12000)),
            tolerance=float(payload.get("tolerance", 1e-10)),
            fitted_iterations=int(payload.get("fitted_iterations", 0)),
            training_loss=payload.get("training_loss"),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "WeightedPoissonModel":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
