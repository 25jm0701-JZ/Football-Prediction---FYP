"""
Poisson model enhanced with player-quality features as additional linear terms.

The log-lambda equation becomes::

    home_eta = intercept + attack[h] + defence[a] + home_adv * true_home
             + Σ βₚ · (player_p[h] − player_p[a])

    away_eta = intercept + attack[a] + defence[h]
             + Σ βₚ · (player_p[a] − player_p[h])

The βₚ coefficients are learned jointly with attack/defence during the
weighted Poisson optimisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.international_training_data import standardize_team


@dataclass
class PlayerAwarePoisson:
    """
    Weighted double-Poisson model with additional player-quality features.

    Parameters
    ----------
    player_features : list[str]
        Column names in the match DataFrame that contain player-quality
        differentials (home − away).  Each gets a learned coefficient βₚ.
    l2 : float
        L2 regularisation strength applied to attack, defence AND βₚ.
    learning_rate : float
    max_iter : int
    tolerance : float
    """

    teams: list[str] | None = None
    intercept: float = 0.0
    home_advantage: float = 0.0
    attack: np.ndarray | None = None
    defence_weakness: np.ndarray | None = None
    player_weights: np.ndarray | None = None  # β coefficients

    player_features: tuple[str, ...] = ()
    l2: float = 0.08
    learning_rate: float = 0.03
    max_iter: int = 12000
    tolerance: float = 1e-10
    fitted_iterations: int = 0
    training_loss: float | None = None

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(
        self,
        matches: pd.DataFrame,
    ) -> "PlayerAwarePoisson":
        required = {"Home", "Away", "HomeGoals", "AwayGoals", "FinalWeight", "TrueHome"}
        missing = required - set(matches.columns)
        if missing:
            raise ValueError(f"Training data is missing columns: {sorted(missing)}")

        frame = matches.copy()
        frame["Home"] = frame["Home"].map(standardize_team)
        frame["Away"] = frame["Away"].map(standardize_team)
        self.teams = sorted(set(frame["Home"]) | set(frame["Away"]))
        team_index = {team: idx for idx, team in enumerate(self.teams)}
        home_idx = frame["Home"].map(team_index).to_numpy(dtype=int)
        away_idx = frame["Away"].map(team_index).to_numpy(dtype=int)
        home_goals = frame["HomeGoals"].to_numpy(dtype=float)
        away_goals = frame["AwayGoals"].to_numpy(dtype=float)
        weights = frame["FinalWeight"].to_numpy(dtype=float)
        true_home = (
            pd.to_numeric(frame.get("TrueHome", pd.Series(0.0, index=frame.index)),
                          errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        if np.any(weights <= 0):
            raise ValueError("FinalWeight values must be positive.")

        n_teams = len(self.teams)

        # ---- player feature matrix (n_matches × n_feat) ---------------- #
        pf_names = [c for c in self.player_features if c in frame.columns]
        n_pf = len(pf_names)
        pf_matrix = frame[pf_names].to_numpy(dtype=float) if n_pf else np.empty(
            (len(frame), 0)
        )

        # ---- initialisation -------------------------------------------- #
        denominator = 2.0 * weights.sum()
        weighted_goal_mean = float(
            np.sum(weights * (home_goals + away_goals)) / denominator
        )
        intercept = math.log(max(weighted_goal_mean, 1e-6))
        home_advantage = 0.0
        attack = np.zeros(n_teams, dtype=float)
        defence = np.zeros(n_teams, dtype=float)
        beta = np.zeros(n_pf, dtype=float)

        n_params = 2 + 2 * n_teams + n_pf
        first_moment = np.zeros(n_params, dtype=float)
        second_moment = np.zeros(n_params, dtype=float)
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        prev_loss = np.inf

        # Adam loop ------------------------------------------------------ #
        for iteration in range(1, self.max_iter + 1):
            pf_home_contrib = pf_matrix @ beta  # (n_matches,) home - away
            pf_away_contrib = -pf_home_contrib  # away - home

            home_eta = (
                intercept
                + attack[home_idx]
                + defence[away_idx]
                + home_advantage * true_home
                + pf_home_contrib
            )
            away_eta = (
                intercept
                + attack[away_idx]
                + defence[home_idx]
                + pf_away_contrib
            )
            home_lambda = np.exp(np.clip(home_eta, -5.0, 5.0))
            away_lambda = np.exp(np.clip(away_eta, -5.0, 5.0))

            data_loss = np.sum(
                weights
                * (home_lambda - home_goals * home_eta
                   + away_lambda - away_goals * away_eta)
            ) / denominator
            penalty = 0.5 * self.l2 * (
                np.mean(attack ** 2) + np.mean(defence ** 2) + np.mean(beta ** 2)
            )
            loss = float(data_loss + penalty)

            home_error = weights * (home_lambda - home_goals) / denominator
            away_error = weights * (away_lambda - away_goals) / denominator

            grad_intercept = float(home_error.sum() + away_error.sum())
            grad_home = float(np.sum(home_error * true_home))
            grad_attack = (
                np.bincount(home_idx, weights=home_error, minlength=n_teams)
                + np.bincount(away_idx, weights=away_error, minlength=n_teams)
                + self.l2 * attack / n_teams
            )
            grad_defence = (
                np.bincount(away_idx, weights=home_error, minlength=n_teams)
                + np.bincount(home_idx, weights=away_error, minlength=n_teams)
                + self.l2 * defence / n_teams
            )
            # gradient for each βₚ
            grad_beta = np.zeros(n_pf, dtype=float)
            for p in range(n_pf):
                diff = pf_matrix[:, p]  # home − away for feature p
                grad_beta[p] = float(
                    np.sum(home_error * diff) + np.sum(away_error * (-diff))
                    + self.l2 * beta[p] / max(n_pf, 1)
                )

            gradient = np.concatenate(
                ([grad_intercept, grad_home], grad_attack, grad_defence, grad_beta)
            )

            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * gradient ** 2
            first_hat = first_moment / (1.0 - beta1 ** iteration)
            second_hat = second_moment / (1.0 - beta2 ** iteration)
            step = self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)

            intercept -= step[0]
            home_advantage -= step[1]
            attack -= step[2: 2 + n_teams]
            defence -= step[2 + n_teams: 2 + 2 * n_teams]
            beta -= step[2 + 2 * n_teams:]

            # Identifiability constraint
            attack_mean = float(attack.mean())
            defence_mean = float(defence.mean())
            attack -= attack_mean
            defence -= defence_mean
            intercept += attack_mean + defence_mean

            if abs(prev_loss - loss) < self.tolerance:
                self.fitted_iterations = iteration
                self.training_loss = loss
                break
            prev_loss = loss
        else:
            self.fitted_iterations = self.max_iter
            self.training_loss = loss

        self.intercept = float(intercept)
        self.home_advantage = float(home_advantage)
        self.attack = attack
        self.defence_weakness = defence
        self.player_weights = beta
        return self

    # ------------------------------------------------------------------ #
    # Prediction helpers
    # ------------------------------------------------------------------ #

    def _team_index(self) -> dict[str, int]:
        if self.teams is None:
            raise RuntimeError("Model must be fitted first.")
        return {team: idx for idx, team in enumerate(self.teams)}

    def expected_goals(
        self,
        home: str,
        away: str,
        host_advantage: int = 0,
        player_diff: np.ndarray | None = None,
    ) -> tuple[float, float]:
        idx = self._team_index()
        home = standardize_team(home)
        away = standardize_team(away)
        missing = {home, away} - set(idx)
        if missing:
            raise ValueError(f"Teams not in model: {sorted(missing)}")

        home_host = max(int(host_advantage), 0)
        away_host = max(-int(host_advantage), 0)
        home_eta = (
            self.intercept
            + self.attack[idx[home]]
            + self.defence_weakness[idx[away]]
            + self.home_advantage * home_host
        )
        away_eta = (
            self.intercept
            + self.attack[idx[away]]
            + self.defence_weakness[idx[home]]
            + self.home_advantage * away_host
        )
        # Add player-quality contribution
        if player_diff is not None and self.player_weights is not None:
            contrib = float(np.dot(self.player_weights, player_diff))
            home_eta += contrib
            away_eta -= contrib

        return (
            float(np.exp(np.clip(home_eta, -5.0, 5.0))),
            float(np.exp(np.clip(away_eta, -5.0, 5.0))),
        )

    def predict_fixtures(
        self,
        fixtures: pd.DataFrame,
        team_features: pd.DataFrame | None = None,
        max_goals: int = 10,
        top_n: int = 3,
    ) -> pd.DataFrame:
        """Predict fixture outcomes, optionally incorporating player features."""
        from src.world_cup_model import format_probability_columns
        from src.player_features import add_player_features_to_matches

        output = fixtures.copy()
        if team_features is not None and self.player_features:
            output["Year"] = 2026
            output = add_player_features_to_matches(
                output, team_features, feature_names=list(self.player_features)
            )
            output = output.drop(columns="Year")

        rows = []
        for _, row in output.iterrows():
            host_adv = int(row.get("HostAdvantage", 0))
            pf_diff = None
            if self.player_features:
                diff_vals = row.get(self.player_features[0])
                if isinstance(diff_vals, (int, float)):
                    pf_diff = np.array([row.get(pf, 0.0) for pf in self.player_features],
                                       dtype=float)
                else:
                    pf_diff = np.array(
                        [row.get(pf, 0.0) for pf in self.player_features], dtype=float
                    )
            home_xg, away_xg = self.expected_goals(
                row["Home"], row["Away"], host_adv, player_diff=pf_diff
            )
            summary = self._score_summary(home_xg, away_xg, max_goals, top_n)
            scores = summary["scores"]
            result = {
                "Poisson_xG_H": home_xg,
                "Poisson_xG_A": away_xg,
                "PoissonP_H": summary["P_H"],
                "PoissonP_D": summary["P_D"],
                "PoissonP_A": summary["P_A"],
                "MostLikelyScore": scores[0]["score"] if scores else "0-0",
                "MostLikelyScoreP": scores[0]["probability"] if scores else 0.0,
            }
            for rank, sc in enumerate(scores, 1):
                result[f"Score{rank}"] = sc["score"]
                result[f"Score{rank}P"] = sc["probability"]
            rows.append(result)
        return pd.concat([output.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    @staticmethod
    def _score_summary(
        home_xg: float, away_xg: float, max_goals: int = 10, top_n: int = 3
    ) -> dict:
        from src.poisson_model import score_summary
        return score_summary(home_xg, away_xg, max_goals, top_n)

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "weighted double Poisson + player-quality features",
            "teams": self.teams,
            "intercept": self.intercept,
            "home_advantage": self.home_advantage,
            "attack": self.attack.tolist() if self.attack is not None else None,
            "defence_weakness": self.defence_weakness.tolist() if self.defence_weakness is not None else None,
            "player_weights": self.player_weights.tolist() if self.player_weights is not None else None,
            "player_features": list(self.player_features),
            "l2": self.l2,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "fitted_iterations": self.fitted_iterations,
            "training_loss": self.training_loss,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PlayerAwarePoisson":
        return cls(
            teams=list(payload["teams"]),
            intercept=float(payload["intercept"]),
            home_advantage=float(payload["home_advantage"]),
            attack=np.asarray(payload["attack"], dtype=float),
            defence_weakness=np.asarray(payload["defence_weakness"], dtype=float),
            player_weights=np.asarray(payload["player_weights"], dtype=float)
            if payload.get("player_weights") else None,
            player_features=tuple(payload.get("player_features", [])),
            l2=float(payload.get("l2", 0.08)),
            learning_rate=float(payload.get("learning_rate", 0.03)),
            max_iter=int(payload.get("max_iter", 12000)),
            tolerance=float(payload.get("tolerance", 1e-10)),
            fitted_iterations=int(payload.get("fitted_iterations", 0)),
            training_loss=payload.get("training_loss"),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "PlayerAwarePoisson":
        import json
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
