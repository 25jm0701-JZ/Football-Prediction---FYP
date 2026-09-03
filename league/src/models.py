"""
Model definitions for football match prediction.

Three approaches:
  1. Logistic Regression — direct H/D/A classification (primary)
  2. Feedforward Neural Network / MLPClassifier
  3. Poisson GLM — goal-based prediction (classic football model)

The Poisson model predicts goals, then derives H/D/A probabilities.
Research shows Poisson and LR have similar accuracy (Dobson & Goddard 2007),
but Poisson enables scoreline and O/U predictions.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

try:
    import statsmodels.api as sm
    HAS_STATS = True
except ImportError:
    HAS_STATS = False


# ── Label encoding ───────────────────────────────────────────────
RESULT_MAP = {"H": 2, "D": 1, "A": 0}
REVERSE_MAP = {2: "H", 1: "D", 0: "A"}


# ── Train/validation split ───────────────────────────────────────
def temporal_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    min_train_matches: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time (chronological), not random.

    Trains on earlier matches, tests on later matches.
    """
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    if split_idx < min_train_matches:
        split_idx = min_train_matches
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


# ── Imputation ───────────────────────────────────────────────────
def fill_features(X: np.ndarray) -> np.ndarray:
    """Fill NaN with column mean."""
    X = X.copy().astype(float)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X


# ── 1. Logistic Regression ──────────────────────────────────────
def build_lr(
    random_state: int = 42,
    **kwargs,
) -> LogisticRegression:
    """Logistic Regression with l2 penalty and lbfgs solver.

    Uses multinomial loss (native, no OneVsRest wrapper needed).
    l2/lbfgs is preferred over l1/liblinear for multiclass.
    """
    params = {
        "penalty": "l2",
        "solver": "lbfgs",
        "C": 1.0,
        "max_iter": 4000,
        "random_state": random_state,
    }
    params.update(kwargs)
    return LogisticRegression(**params)


# ── 2. Feedforward Neural Network ───────────────────────────────
def build_fnn(
    random_state: int = 42,
    **kwargs,
):
    """Feedforward Neural Network using sklearn MLPClassifier.

    Paper 2 FNN: 4 layers (64->128->128->128), ReLU, l2, dropout 0.5,
    batch normalization, Yogi optimizer, lr=0.002.

    Scikit-learn's MLPClassifier doesn't have batch norm or dropout,
    but provides a reasonable approximation of the Paper 2 architecture.
    """
    params = {
        "hidden_layer_sizes": (64, 128, 128, 128),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.001,        # l2 regularization
        "learning_rate": "adaptive",
        "learning_rate_init": 0.002,
        "max_iter": 400,
        "batch_size": 32,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 20,
        "random_state": random_state,
        "verbose": False,
    }
    params.update(kwargs)
    return MLPClassifier(**params)


# ── 3. Poisson GLM (goal-based prediction) ──────────────────────
class PoissonGLM:
    """Classic Poisson football model — predicts goals first, then results.

    Fits two independent Poisson regressions (home goals, away goals)
    and derives H/D/A probabilities from their distributions.

    References:
      - Maher (1982) Modelling association football scores
      - Dixon & Coles (1997) Modelling association football scores
      - Dobson & Goddard (2007) fixed-odds betting market efficiency

    Example:
        model = PoissonGLM()
        model.fit(X_train, y_train_df[['home_goals','away_goals']])
        proba = model.predict_proba(X_test)  # (n, 3) = [P(A), P(D), P(H)]
    """
    def __init__(self, max_goals: int = 8):
        self.max_goals = max_goals
        self.model_h: Any = None   # statsmodels GLM for home goals
        self.model_a: Any = None   # statsmodels GLM for away goals
        self.scaler: Any = None    # sklearn StandardScaler
        self.feature_names: list = []

    def fit(self, X, y_goals):
        """Fit two Poisson regressions.

        Args:
            X: feature array (n_samples, n_features) or DataFrame
            y_goals: array (n_samples, 2) with columns [home_goals, away_goals]
        """
        if not HAS_STATS:
            raise ImportError("statsmodels is required for PoissonGLM. Install: pip install statsmodels")

        from sklearn.preprocessing import StandardScaler

        X_arr = np.asarray(X).astype(float)
        y_arr = np.asarray(y_goals)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        home_g = y_arr[:, 0].astype(int)
        away_g = y_arr[:, 1].astype(int)

        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else []

        # Impute NaN
        col_mean = np.nanmean(X_arr, axis=0)
        for i in range(X_arr.shape[1]):
            m = col_mean[i]
            if m != m:
                m = 0
            X_arr[np.isnan(X_arr[:, i]), i] = m

        # Standardize
        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X_arr)
        X_s = sm.add_constant(X_s)

        # Fit Poisson GLMs (disp=0 suppresses convergence messages)
        self.model_h = sm.GLM(home_g, X_s, family=sm.families.Poisson()).fit(disp=0)
        self.model_a = sm.GLM(away_g, X_s, family=sm.families.Poisson()).fit(disp=0)
        return self

    def predict_proba(self, X):
        """Predict H/D/A probabilities via goal distributions.

        Returns:
            array (n_samples, 3): [P(away_win), P(draw), P(home_win)]
        """
        X_arr = np.asarray(X).astype(float)
        col_mean = np.nanmean(X_arr, axis=0)
        for i in range(X_arr.shape[1]):
            m = col_mean[i]
            if m != m:
                m = 0
            X_arr[np.isnan(X_arr[:, i]), i] = m

        X_s = self.scaler.transform(X_arr)
        X_s = sm.add_constant(X_s)

        lam_h = self.model_h.predict(X_s)
        lam_a = self.model_a.predict(X_s)

        probs = np.zeros((len(lam_h), 3))
        mg = self.max_goals

        for i in range(len(lam_h)):
            lh = lam_h.iloc[i] if hasattr(lam_h, "iloc") else lam_h[i]
            la = lam_a.iloc[i] if hasattr(lam_a, "iloc") else lam_a[i]

            pmf_h = np.array([poisson.pmf(g, lh) for g in range(mg + 1)])
            pmf_a = np.array([poisson.pmf(g, la) for g in range(mg + 1)])

            # Home win: sum_{i>j} P(h=i) * P(a=j)
            ph = 0.0
            for hi in range(1, mg + 1):
                for ai in range(hi):
                    ph += pmf_h[hi] * pmf_a[ai]

            # Draw
            pd_d = float(np.sum(pmf_h * pmf_a))
            # Away win
            pa = float(1.0 - ph - pd_d)
            probs[i] = [pa, pd_d, ph]

        return probs

    def predict(self, X):
        """Predict class labels (0=away, 1=draw, 2=home)."""
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_goals(self, X):
        """Return expected goal values (lambda_h, lambda_a)."""
        X_arr = np.asarray(X).astype(float)
        col_mean = np.nanmean(X_arr, axis=0)
        for i in range(X_arr.shape[1]):
            m = col_mean[i]
            if m != m:
                m = 0
            X_arr[np.isnan(X_arr[:, i]), i] = m
        X_s = self.scaler.transform(X_arr)
        X_s = sm.add_constant(X_s)
        lam_h = self.model_h.predict(X_s)
        lam_a = self.model_a.predict(X_s)
        return np.column_stack([
            lam_h.values if hasattr(lam_h, "values") else lam_h,
            lam_a.values if hasattr(lam_a, "values") else lam_a,
        ])

    def predict_ou_proba(self, X, threshold: float = 2.5):
        """Predict probability of over `threshold` total goals."""
        goals = self.predict_goals(X)
        probs = np.zeros(len(goals))
        mg = self.max_goals
        for i in range(len(goals)):
            lh, la = goals[i]
            p = 0.0
            for hi in range(mg + 1):
                for ai in range(mg + 1):
                    if hi + ai > threshold:
                        p += poisson.pmf(hi, lh) * poisson.pmf(ai, la)
            probs[i] = min(p, 1.0)
        return probs


def build_poisson(max_goals: int = 8) -> PoissonGLM:
    """Build a Poisson GLM model for goal-based prediction.

    Returns:
        PoissonGLM instance (fit manually with .fit(X, y_goals)).
    """
    return PoissonGLM(max_goals=max_goals)


# ── Model registry ──────────────────────────────────────────────
MODEL_REGISTRY: dict[str, tuple[str, Any]] = {
    "lr": ("Logistic Regression", build_lr),
    "fnn": ("FNN (MLP)", build_fnn),
}
# Poisson is conditionally available
if HAS_STATS:
    MODEL_REGISTRY["poisson"] = ("Poisson GLM (goals)", build_poisson)


def list_available_models() -> list[str]:
    """List model keys that can be used right now."""
    return list(MODEL_REGISTRY.keys())


def get_model(name: str, **kwargs):
    """Get a model by name, with optional override kwargs."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list_available_models()}"
        )
    _, builder = MODEL_REGISTRY[name]
    return builder(**kwargs)
