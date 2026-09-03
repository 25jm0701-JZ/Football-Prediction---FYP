"""Poisson rating model — Goalence-style Pi-Ratings, with Dixon-Coles (1997)
low-score adjustment for realistic draw probabilities.

Each team carries four learnable parameters:
  home_attack (ha), home_defense (hd),
  away_attack (aa), away_defense (ad).

For a match between home i and away j:
  λ_home = exp(ha[i] + ad[j] + home_adv)
  λ_away = exp(aa[j] + hd[i])

Match probabilities derived from the joint Poisson distribution with
a Dixon-Coles adjustment (ρ) that inflates low-scoring draws (0-0, 1-1)
and deflates narrow wins (1-0, 0-1) to match real football's higher
draw rate.

After each match, parameters are updated via Poisson gradient ascent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson


class PoissonRatingModel:
    """Goalence-style Poisson rating system with Dixon-Coles draw correction.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient ascent updates.
    home_adv : float
        Home advantage offset (added in log space).
    l2_reg : float
        L2 regularization strength to prevent parameter drift.
    initial_rating : float
        Starting value for all team parameters.
    rho : float
        Dixon-Coles adjustment for low scores. Negative values inflate
        0-0 and 1-1 draws, deflate 1-0 and 0-1 narrow wins.
        rho=0 → standard independent Poisson.
    max_goals : int
        Truncation limit for the Poisson distribution convolution.
    """

    def __init__(
        self,
        learning_rate: float = 0.03,
        home_adv: float = 0.1,
        l2_reg: float = 1e-4,
        initial_rating: float = 0.0,
        rho: float = -0.2,
        max_goals: int = 8,
    ):
        self.learning_rate = learning_rate
        self.home_adv = home_adv
        self.l2_reg = l2_reg
        self.initial_rating = initial_rating
        self.rho = rho
        self.max_goals = max_goals

        # team -> {ha, hd, aa, ad}
        self.ratings: dict[str, dict[str, float]] = {}
        self.games_played: dict[str, int] = {}
        self._history: list[dict] = []

    # ── Parameter access ───────────────────────────────────────────

    def _init_team(self, team: str):
        if team not in self.ratings:
            self.ratings[team] = {
                "ha": self.initial_rating,
                "hd": self.initial_rating,
                "aa": self.initial_rating,
                "ad": self.initial_rating,
            }
            self.games_played[team] = 0

    # ── Prediction ─────────────────────────────────────────────────

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """Return (λ_home, λ_away) for a match."""
        self._init_team(home)
        self._init_team(away)
        hp = self.ratings[home]
        ap = self.ratings[away]

        lam_h = np.exp(hp["ha"] + ap["ad"] + self.home_adv)
        lam_a = np.exp(ap["aa"] + hp["hd"])
        return float(lam_h), float(lam_a)

    def match_probabilities(
        self, home: str, away: str
    ) -> tuple[float, float, float]:
        """Return P(away), P(draw), P(home) via Dixon-Coles adjusted Poisson.

        Builds a joint Poisson probability matrix over (home_goals, away_goals)
        up to `max_goals`, applies the Dixon-Coles τ adjustment for low scores
        (0-0, 1-1, 0-1, 1-0), then marginalises to H/D/A probabilities.

        With ρ < 0 (typical): 0-0 and 1-1 draws get inflated, 1-0 and 0-1
        narrow wins get deflated, matching real football's higher draw rate.
        """
        lam_h, lam_a = self.expected_goals(home, away)
        mg = self.max_goals

        pmf_h = np.array([poisson.pmf(g, lam_h) for g in range(mg + 1)])
        pmf_a = np.array([poisson.pmf(g, lam_a) for g in range(mg + 1)])

        # Joint probability matrix: joint[i,j] = P(home=i, away=j)
        joint = np.outer(pmf_h, pmf_a)

        # Dixon-Coles adjustment for the four low-scoring cells
        rho = self.rho
        if rho != 0.0:
            # τ(0,0) = 1 - λ_h·λ_a·ρ  (>1 when ρ<0 → inflates 0-0)
            joint[0, 0] *= 1.0 - lam_h * lam_a * rho
            # τ(1,1) = 1 - ρ           (>1 when ρ<0 → inflates 1-1)
            joint[1, 1] *= 1.0 - rho
            # τ(0,1) = 1 + λ_h·ρ      (<1 when ρ<0 → deflates away 1-0)
            joint[0, 1] *= 1.0 + lam_h * rho
            # τ(1,0) = 1 + λ_a·ρ      (<1 when ρ<0 → deflates home 1-0)
            joint[1, 0] *= 1.0 + lam_a * rho

            # Clip any negative values from extreme ρ + large λ combos
            joint = np.maximum(joint, 0.0)
            # Renormalise (the τ adjustments change total mass slightly)
            joint /= joint.sum()

        # H/D/A from joint matrix
        # joint[i,j] = P(home_goals=i, away_goals=j)
        # Home wins: i > j  →  lower triangle (below diagonal)
        # Draws:     i = j  →  diagonal
        # Away wins: i < j  →  upper triangle (above diagonal)
        ph = float(np.tril(joint, k=-1).sum())   # lower triangle = home wins
        pd_ = float(np.trace(joint))              # diagonal = draws
        pa = float(np.triu(joint, k=1).sum())     # upper triangle = away wins

        # Safety clamp
        eps = 1e-15
        return (
            float(np.clip(pa, eps, 1.0 - eps)),
            float(np.clip(pd_, eps, 1.0 - eps)),
            float(np.clip(ph, eps, 1.0 - eps)),
        )

    # ── Online update ──────────────────────────────────────────────

    def update(
        self, home: str, away: str,
        goals_home: int, goals_away: int,
    ):
        """Gradient ascent on Poisson log-likelihood.

        ∂logL/∂θ = observed_goals - λ - l2_reg × θ
        """
        lam_h, lam_a = self.expected_goals(home, away)
        hp = self.ratings[home]
        ap = self.ratings[away]

        # Gradients
        hp["ha"] += self.learning_rate * (goals_home - lam_h - self.l2_reg * hp["ha"])
        ap["ad"] += self.learning_rate * (goals_home - lam_h - self.l2_reg * ap["ad"])
        ap["aa"] += self.learning_rate * (goals_away - lam_a - self.l2_reg * ap["aa"])
        hp["hd"] += self.learning_rate * (goals_away - lam_a - self.l2_reg * hp["hd"])

        self.games_played[home] += 1
        self.games_played[away] += 1

    # ── Batch processing ───────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> list[dict]:
        """Run model chronologically; store pre-match predictions.

        For each match:
          1. Predict from current ratings.
          2. Store pre-match probabilities + expected goals.
          3. Update ratings from actual result.
        """
        df = df.sort_values("date").reset_index(drop=True)
        self._history = []

        for _, row in df.iterrows():
            home, away = row["home_team"], row["away_team"]
            gh, ga = int(row["home_goals_full"]), int(row["away_goals_full"])

            lam_h, lam_a = self.expected_goals(home, away)
            pa, pd_, ph = self.match_probabilities(home, away)

            self._history.append({
                "date": row["date"],
                "home": home, "away": away,
                "lambda_home": lam_h,
                "lambda_away": lam_a,
                "p_away": pa, "p_draw": pd_, "p_home": ph,
                "goals_home": gh, "goals_away": ga,
            })

            self.update(home, away, gh, ga)

        return self._history

    # ── Evaluation ─────────────────────────────────────────────────

    def evaluate_sequential(
        self, df: pd.DataFrame, update: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict sequentially; optionally continue updating.

        Returns (y_true, y_prob) where:
          y_true: 0=A, 1=D, 2=H
          y_prob: (n, 3) in [A, D, H] order
        """
        from sklearn.metrics import accuracy_score

        y_true: list[int] = []
        y_prob: list[list[float]] = []
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)

        for i, (_, row) in enumerate(df.iterrows()):
            home, away = row["home_team"], row["away_team"]
            gh, ga = int(row["home_goals_full"]), int(row["away_goals_full"])

            pa, pd_, ph = self.match_probabilities(home, away)
            y_prob.append([pa, pd_, ph])

            # Label: 2=H, 1=D, 0=A
            y_true.append(2 if gh > ga else 1 if gh == ga else 0)

            if update:
                self.update(home, away, gh, ga)

        return np.array(y_true), np.array(y_prob)

    # ── Likelihood & optimization ─────────────────────────────────

    def log_likelihood(self, df: pd.DataFrame) -> float:
        """Walk-forward negative log-likelihood (lower = better)."""
        self.reset()
        self.fit(df)
        ll = 0.0
        for h in self._history:
            if h["goals_home"] > h["goals_away"]:
                p = h["p_home"]
            elif h["goals_home"] == h["goals_away"]:
                p = h["p_draw"]
            else:
                p = h["p_away"]
            ll += np.log(max(p, 1e-15))
        return -ll

    def optimize_params(
        self, df: pd.DataFrame,
        n_iter: int = 100,
        early_stop: int = 30,
        verbose: bool = True,
    ) -> tuple[dict, float]:
        """Random search over learning_rate, home_adv, l2_reg, initial_rating.

        Args:
            n_iter: Max random search iterations.
            early_stop: Stop after this many iters without improvement.
        """
        import itertools
        import random

        param_grid = {
            "learning_rate": [0.005, 0.008, 0.01, 0.02, 0.03, 0.05,
                              0.08, 0.1, 0.15, 0.2, 0.3, 0.5],
            "home_adv": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
            "l2_reg": [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            "initial_rating": [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
            "rho": [-0.3, -0.25, -0.2, -0.18, -0.15,
                    -0.12, -0.1, -0.08, -0.05, -0.02, 0.0],
        }

        keys = list(param_grid.keys())
        all_combos = list(itertools.product(*[param_grid[k] for k in keys]))
        combos = random.sample(all_combos, min(n_iter, len(all_combos)))

        best_nll = float("inf")
        best_params = {}
        no_improve = 0

        if verbose:
            print(f"  Random search {len(combos)} combos"
                  f" ({len(all_combos)} in grid)")

        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            for k, v in params.items():
                setattr(self, k, v)

            try:
                nll = self.log_likelihood(df)
            except Exception:
                continue

            if nll < best_nll:
                best_nll = nll
                best_params = params.copy()
                no_improve = 0
                if verbose:
                    print(f"  [{i+1}/{len(combos)}] new best:"
                          f" lr={params['learning_rate']:.4f}"
                          f" ha={params['home_adv']:.3f}"
                          f" reg={params['l2_reg']:.6f}"
                          f" init={params['initial_rating']:.2f}"
                          f" rho={params['rho']:.3f}"
                          f"  NLL={nll:.2f}")
            else:
                no_improve += 1

            if early_stop and no_improve >= early_stop:
                if verbose:
                    print(f"  Early stop at iter {i+1}"
                          f" ({no_improve} iters without improvement)")
                break

            if verbose and (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(combos)}] best NLL={best_nll:.2f}")

        # Restore best
        for k, v in best_params.items():
            setattr(self, k, v)
        self.reset()

        if verbose:
            print(f"\n  OK Best: {best_params}  NLL={best_nll:.2f}")

        return best_params, best_nll

    def reset(self):
        """Clear all team states."""
        self.ratings.clear()
        self.games_played.clear()
        self._history.clear()

    # ── Diagnostics ────────────────────────────────────────────────

    def get_team_strengths(self) -> pd.DataFrame:
        """Return current team parameter estimates as a DataFrame."""
        rows = []
        for team, p in self.ratings.items():
            rows.append({
                "team": team,
                "home_attack": p["ha"],
                "home_defense": p["hd"],
                "away_attack": p["aa"],
                "away_defense": p["ad"],
                "net_attack": p["ha"] + p["aa"],   # overall attacking strength
                "net_defense": p["hd"] + p["ad"],   # overall defensive weakness
                "games": self.games_played.get(team, 0),
            })
        return pd.DataFrame(rows).sort_values("net_attack", ascending=False)
