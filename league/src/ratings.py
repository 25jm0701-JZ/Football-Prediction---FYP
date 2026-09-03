"""
Pi-rating system for football team strength estimation.

Implements the ordered-probit rating model from Luiz et al. (2024):
  "A deep learning approach for football match prediction"

Each team's strength is modeled as N(mu, sigma^2). Match outcomes
follow an ordered probit model. After each match, ratings update
via gradient ascent on the log-likelihood, scaled by pi (the
learning rate).

Usage:
    pr = PiRating(pi=0.01)
    pr.fit(df_sorted_by_date)          # compute all pre-match ratings
    df_with_ratings = pr.add_features(df)  # attach rating columns
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


class PiRating:
    """Pi-rating system.

    Parameters
    ----------
    pi : float
        Learning / update rate (optimized via grid search).
    initial_mu : float
        Starting strength for new teams.
    initial_sigma : float
        Starting uncertainty for new teams.
    home_adv : float
        Home advantage offset in the ordered probit model.
    delta : float
        Threshold parameter for the draw outcome.
    sigma_match : float
        Match-level variance (aleatoric uncertainty).
    sigma_decay : float
        Multiplicative decay applied to sigma after each update.
    """

    def __init__(
        self,
        pi: float = 0.015,
        initial_mu: float = 0.0,
        initial_sigma: float = 2.0,
        home_adv: float = 0.3,
        delta: float = 0.5,
        sigma_match: float = 1.2,
        sigma_decay: float = 0.999,
    ):
        self.pi = pi
        self.initial_mu = initial_mu
        self.initial_sigma = initial_sigma
        self.home_adv = home_adv
        self.delta = delta
        self.sigma_match = sigma_match
        self.sigma_decay = sigma_decay

        # Team states
        self.mu: dict[str, float] = {}
        self.sigma: dict[str, float] = {}
        self.games_played: dict[str, int] = {}

        # Tracking
        self._history: list[dict] = []

    # ── Helpers ──────────────────────────────────────────────────

    def _get_mu(self, team: str) -> float:
        return self.mu.get(team, self.initial_mu)

    def _get_sigma(self, team: str) -> float:
        return self.sigma.get(team, self.initial_sigma)

    # ── Expected outcome ─────────────────────────────────────────

    def expected_outcome(
        self, home: str, away: str
    ) -> tuple[float, float, float]:
        """Return P(home win), P(draw), P(away win) before a match."""
        mu_h = self._get_mu(home)
        mu_a = self._get_mu(away)
        sig_h = self._get_sigma(home)
        sig_a = self._get_sigma(away)

        tau = np.sqrt(sig_h**2 + sig_a**2 + self.sigma_match**2)
        theta = mu_h - mu_a + self.home_adv

        z1 = (self.delta - theta) / tau
        z2 = (-self.delta - theta) / tau

        p_home = 1.0 - norm.cdf(z1)
        p_draw = norm.cdf(z1) - norm.cdf(z2)
        p_away = norm.cdf(z2)

        # Safety clamp
        eps = 1e-15
        p_home = float(np.clip(p_home, eps, 1 - eps))
        p_draw = float(np.clip(p_draw, eps, 1 - eps))
        p_away = float(np.clip(p_away, eps, 1 - eps))

        return p_home, p_draw, p_away

    # ── Rating update ────────────────────────────────────────────

    def update(self, home: str, away: str, goals_home: int, goals_away: int):
        """Update ratings after a match using ordered-probit gradient."""
        mu_h = self._get_mu(home)
        mu_a = self._get_mu(away)
        sig_h = self._get_sigma(home)
        sig_a = self._get_sigma(away)
        tau = np.sqrt(sig_h**2 + sig_a**2 + self.sigma_match**2)
        theta = mu_h - mu_a + self.home_adv
        z1 = (self.delta - theta) / tau
        z2 = (-self.delta - theta) / tau

        # Outcome index: 0=home win, 1=draw, 2=away win
        if goals_home > goals_away:
            y = 0
        elif goals_home == goals_away:
            y = 1
        else:
            y = 2

        # Gradient of log-likelihood w.r.t. theta
        phi_z1 = norm.pdf(z1)
        phi_z2 = norm.pdf(z2)

        if y == 0:  # home win
            grad = phi_z1 / (tau * (1.0 - norm.cdf(z1) + 1e-15))
        elif y == 1:  # draw
            grad = (phi_z2 - phi_z1) / (tau * (norm.cdf(z1) - norm.cdf(z2) + 1e-15))
        else:  # away win
            grad = -phi_z2 / (tau * (norm.cdf(z2) + 1e-15))

        # Update mu: gradient × pi × sigma^2 (natural gradient)
        self.mu[home] = mu_h + self.pi * grad * sig_h**2
        self.mu[away] = mu_a - self.pi * grad * sig_a**2

        # Sigma decays gradually
        self.sigma[home] = sig_h * self.sigma_decay
        self.sigma[away] = sig_a * self.sigma_decay

        # Track games played
        self.games_played[home] = self.games_played.get(home, 0) + 1
        self.games_played[away] = self.games_played.get(away, 0) + 1

    # ── Batch fit ────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> list[dict]:
        """Run pi-rating over a chronologically-sorted DataFrame.

        For each match, first stores pre-match ratings, then updates
        post-match. Returns the history of pre-match states.

        Required columns: date, home_team, away_team,
                          home_goals_full, away_goals_full
        """
        df = df.sort_values("date").reset_index(drop=True)
        self._history = []

        for _, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            # Store pre-match ratings
            mu_h = self._get_mu(home)
            mu_a = self._get_mu(away)
            sig_h = self._get_sigma(home)
            sig_a = self._get_sigma(away)
            p_h, p_d, p_a = self.expected_outcome(home, away)

            self._history.append({
                "date": row["date"],
                "home": home,
                "away": away,
                "mu_home": mu_h,
                "mu_away": mu_a,
                "sigma_home": sig_h,
                "sigma_away": sig_a,
                "p_home": p_h,
                "p_draw": p_d,
                "p_away": p_a,
                "home_goals": row["home_goals_full"],
                "away_goals": row["away_goals_full"],
            })

            # Update after the match
            self.update(home, away, row["home_goals_full"], row["away_goals_full"])

        return self._history

    # ── Feature generation ───────────────────────────────────────

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pre-match pi-rating columns to a DataFrame.

        Must call fit() first, or the history won't exist.
        """
        if not self._history:
            raise RuntimeError("Call fit() before add_features().")

        result = df.copy()
        n = min(len(result), len(self._history))
        hist = self._history[:n]

        result["home_pi_mu"] = [h["mu_home"] for h in hist]
        result["away_pi_mu"] = [h["mu_away"] for h in hist]
        result["home_pi_sigma"] = [h["sigma_home"] for h in hist]
        result["away_pi_sigma"] = [h["sigma_away"] for h in hist]
        result["pi_mu_diff"] = [h["mu_home"] - h["mu_away"] for h in hist]
        result["pi_home_win_prob"] = [h["p_home"] for h in hist]
        result["pi_draw_prob"] = [h["p_draw"] for h in hist]
        result["pi_away_win_prob"] = [h["p_away"] for h in hist]

        return result

    # ── Pi optimization ──────────────────────────────────────────

    def log_likelihood(self, df: pd.DataFrame, pi: float | None = None) -> float:
        """Compute negative log-likelihood of match outcomes under pi.

        Lower is better. Uses pi from attribute or supplied value.
        """
        if pi is not None:
            self.pi = pi
        self.reset()
        self.fit(df)
        ll = 0.0
        for h in self._history:
            probs = [h["p_home"], h["p_draw"], h["p_away"]]
            if h["home_goals"] > h["away_goals"]:
                ll += np.log(probs[0])
            elif h["home_goals"] == h["away_goals"]:
                ll += np.log(probs[1])
            else:
                ll += np.log(probs[2])
        return -ll  # negative log-likelihood

    def optimize_pi(
        self, df: pd.DataFrame, pi_values: list[float] | None = None
    ) -> tuple[float, float]:
        """Grid search over pi values to find the best one.

        Returns (best_pi, best_nll).
        """
        if pi_values is None:
            pi_values = [0.001, 0.003, 0.005, 0.008, 0.01, 0.015,
                         0.02, 0.03, 0.05, 0.08, 0.1]

        best_pi = self.pi
        best_nll = float("inf")

        print("  Optimizing pi (lower NLL = better fit):")
        for p in pi_values:
            nll = self.log_likelihood(df, pi=p)
            marker = "  <-- best" if nll < best_nll else ""
            print(f"    pi={p:.4f}  NLL={nll:.2f}{marker}")
            if nll < best_nll:
                best_nll = nll
                best_pi = p

        self.pi = best_pi
        return best_pi, best_nll

    def optimize_params(
        self, df: pd.DataFrame, n_iter: int = 200, verbose: bool = True,
    ) -> tuple[dict, float]:
        """Random search over ALL Pi-Rating parameters.

        Optimizes pi, delta, home_adv, sigma_match, and sigma_decay
        jointly to minimize negative log-likelihood.

        Args:
            df: DataFrame sorted by date.
            n_iter: Number of random-search iterations.
            verbose: Print progress.

        Returns:
            (best_params_dict, best_nll)
        """
        import itertools
        import random

        # Coarse grids for each parameter
        param_grid = {
            "pi": [0.001, 0.003, 0.005, 0.008, 0.01, 0.012, 0.015,
                   0.02, 0.03, 0.05, 0.08, 0.1],
            "delta": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                      1.0, 1.2, 1.5],
            "home_adv": [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                         0.5, 0.6, 0.7, 0.8, 1.0],
            "sigma_match": [0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8,
                            2.0, 2.5],
            "sigma_decay": [0.99, 0.992, 0.995, 0.997, 0.998, 0.999,
                            0.9995, 1.0],
        }

        keys = list(param_grid.keys())
        all_combos = list(itertools.product(*[param_grid[k] for k in keys]))
        combos = random.sample(all_combos, min(n_iter, len(all_combos)))

        # Store originals for restoring after search
        original = {k: getattr(self, k) for k in keys}

        best_nll = float("inf")
        best_params = {}

        if verbose:
            print(f"  Random search over {len(combos)} combos"
                  f" ({len(all_combos)} total in grid)")

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
                if verbose:
                    print(f"  [{i+1}/{len(combos)}] New best: "
                          f"{params}  NLL={nll:.2f}")

            if verbose and (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(combos)}] current best NLL={best_nll:.2f}")

        # Restore best params and re-fit with them
        for k, v in best_params.items():
            setattr(self, k, v)
        self.reset()

        if verbose:
            print(f"\n  OK Best params: {best_params}")
            print(f"  OK Best NLL:   {best_nll:.2f}")

        return best_params, best_nll

    def reset(self):
        """Clear all team states."""
        self.mu.clear()
        self.sigma.clear()
        self.games_played.clear()
        self._history.clear()

    def evaluate_sequential(
        self, df: pd.DataFrame, update: bool = True, verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run model sequentially over matches, returning predictions.

        For each match (in chronological order):
          1. Predict H/D/A probs from current ratings.
          2. Optionally update ratings after the actual result.

        This is the "online" evaluation mode — what Goalence does
        in production.

        Args:
            df: DataFrame sorted by date.
            update: Whether to update ratings after each match
                    (True = online, False = frozen ratings).
            verbose: Print progress.

        Returns:
            (y_true, y_prob) arrays.
        """
        from sklearn.metrics import accuracy_score

        y_true: list[int] = []
        y_prob: list[float] = []

        df_sorted = df.sort_values("date").reset_index(drop=True)
        n = len(df_sorted)

        for i, (_, row) in enumerate(df_sorted.iterrows()):
            home, away = row["home_team"], row["away_team"]

            # Pre-match prediction
            p_h, p_d, p_a = self.expected_outcome(home, away)
            y_prob.append([p_a, p_d, p_h])  # [A, D, H] order

            # Actual result label: 2=H, 1=D, 0=A
            if row["home_goals_full"] > row["away_goals_full"]:
                y_true.append(2)
            elif row["home_goals_full"] == row["away_goals_full"]:
                y_true.append(1)
            else:
                y_true.append(0)

            # Post-match update (unless frozen)
            if update:
                self.update(
                    home, away,
                    row["home_goals_full"], row["away_goals_full"],
                )

            if verbose and (i + 1) % 500 == 0:
                acc = accuracy_score(y_true, np.argmax(y_prob, axis=1))
                print(f"  [{i+1}/{n}] running acc={acc:.4f}")

        return np.array(y_true), np.array(y_prob)
