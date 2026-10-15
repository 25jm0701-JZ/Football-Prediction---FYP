"""Mathematically checked Poisson scoreline reweighting with LR H/D/A marginals.

Index convention: score[home_goals, away_goals]; class vector [Away, Draw, Home].
No train/test data are consumed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson

CLASS_NAMES = ("away", "draw", "home")
ATOL = 3e-12


@dataclass(frozen=True)
class ScoreGrid:
    matrix: np.ndarray
    max_goals: int
    omitted_mass: float


def class_masks(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("Need at least a 2x2 score grid to represent H/D/A")
    h, a = np.indices((n, n))
    return (h < a, h == a, h > a)


def aggregate_hda(matrix: np.ndarray) -> np.ndarray:
    q = _validate_matrix(matrix)
    return np.asarray([q[mask].sum() for mask in class_masks(len(q))], dtype=float)


def _validate_matrix(matrix: np.ndarray) -> np.ndarray:
    q = np.asarray(matrix, dtype=float)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or q.shape[0] < 2:
        raise ValueError("Score matrix must be square and at least 2x2")
    if not np.all(np.isfinite(q)) or np.any(q < 0):
        raise ValueError("Score matrix contains negative or non-finite values")
    if not np.isclose(q.sum(), 1, rtol=0, atol=ATOL):
        raise ValueError(f"Score matrix is not normalized: {q.sum()}")
    return q


def checked_probabilities(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if p.shape != (3,) or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("H/D/A probabilities must be 3 finite nonnegative values")
    if not np.isclose(p.sum(), 1.0, rtol=0, atol=ATOL):
        raise ValueError("H/D/A probabilities must sum to 1")
    return p


def reweight_scores(poisson_score: np.ndarray, lr_hda: np.ndarray) -> np.ndarray:
    """Exact within-class rescaling. Never clamp an impossible zero denominator.

    If Q(class)=0 but LR(class)>0, no within-class-preserving distribution
    exists. Fail explicitly instead of silently injecting arbitrary scores.
    """
    q = _validate_matrix(poisson_score)
    p = checked_probabilities(lr_hda)
    result = np.zeros_like(q)
    for c, mask in enumerate(class_masks(len(q))):
        mass = float(q[mask].sum())
        if mass == 0:
            if p[c] > 0:
                raise ValueError(f"Poisson mass is zero for {CLASS_NAMES[c]} but LR assigns {p[c]}")
            continue
        # Dividing before multiplying avoids overflow when mass is subnormal.
        result[mask] = (q[mask] / mass) * p[c]
    if np.any(result < 0) or not np.all(np.isfinite(result)):
        raise ArithmeticError("Invalid reweighted score distribution")
    if not np.isclose(result.sum(), 1., atol=ATOL, rtol=0):
        raise ArithmeticError("Reweighted probabilities do not sum to 1")
    if not np.allclose(aggregate_hda(result), p, atol=ATOL, rtol=0):
        raise ArithmeticError("Reweighted H/D/A marginals disagree with LR")
    return result


def adaptive_poisson_grid(lambda_home: float, lambda_away: float,
                          tail_tolerance: float = 1e-12,
                          max_support: int = 256) -> ScoreGrid:
    """Choose support solely using pre-match lambdas, never realized scores.

    The omitted unnormalized joint probability is <= tail_tolerance.
    Unlike the legacy max_goals=10 grid, this adapts to high-goal tails.
    """
    if any(not np.isfinite(x) or x <= 0 for x in (lambda_home, lambda_away)):
        raise ValueError("Poisson rate parameters must be positive and finite")
    if not (0 < tail_tolerance < 0.01) or max_support < 10:
        raise ValueError("Invalid grid tolerance or support limit")
    # Divide the tail budget across the two one-dimensional marginals.
    quantile = 1 - tail_tolerance / 4
    n = max(10, int(poisson.ppf(quantile, lambda_home)),
            int(poisson.ppf(quantile, lambda_away)))
    if n > max_support:
        raise ArithmeticError(f"Required score support {n} exceeds configured limit {max_support}")
    while True:
        contained = float(poisson.cdf(n, lambda_home) * poisson.cdf(n, lambda_away))
        if 1.0 - contained <= tail_tolerance:
            break
        n += 1
        if n > max_support:
            raise ArithmeticError("Could not bound omitted scoreline tail")
    h = poisson.pmf(np.arange(n + 1), lambda_home)
    a = poisson.pmf(np.arange(n + 1), lambda_away)
    mat = np.outer(h, a)
    raw_mass = float(mat.sum())
    if raw_mass <= 0 or not np.isfinite(raw_mass):
        raise ArithmeticError("All Poisson mass underflowed")
    return ScoreGrid(mat / raw_mass, n, max(0.0, 1.0 - raw_mass))


def score_statistics(matrix: np.ndarray, actual_home: int, actual_away: int,
                     lambda_home: float | None = None, lambda_away: float | None = None,
                     target_hda: np.ndarray | None = None) -> dict[str, float | int]:
    q = _validate_matrix(matrix)
    n = len(q) - 1
    if actual_home < 0 or actual_away < 0 or not float(actual_home).is_integer() or not float(actual_away).is_integer():
        raise ValueError("Goals must be nonnegative integers")
    outside = actual_home > n or actual_away > n
    if outside:
        if lambda_home is None or lambda_away is None:
            raise ArithmeticError("True score outside fixed prediction grid: supply Poisson rates")
        if not np.isfinite(lambda_home) or not np.isfinite(lambda_away):
            raise ValueError("Invalid Poisson parameters")
        # The prediction grid is NOT enlarged using the observed result. Evaluate
        # a rare outside-grid score by the analytical Poisson PMF instead.
        lp = float(poisson.logpmf(actual_home, lambda_home) +
                   poisson.logpmf(actual_away, lambda_away))
        contained = float(poisson.cdf(n, lambda_home) * poisson.cdf(n, lambda_away))
        if target_hda is None:
            true_logp = lp - np.log(contained)
        else:
            target = checked_probabilities(target_hda)
            cls = 0 if actual_home < actual_away else 1 if actual_home == actual_away else 2
            base_h = poisson.pmf(np.arange(n+1), lambda_home)
            base_a = poisson.pmf(np.arange(n+1), lambda_away)
            base_group = float(np.outer(base_h, base_a)[class_masks(n+1)[cls]].sum())
            if base_group <= 0 or target[cls] <= 0:
                raise ArithmeticError("Outside-support score has zero class probability")
            true_logp = np.log(target[cls]) + lp - np.log(base_group)
        pmf = float(np.exp(true_logp))
    else:
        pmf = float(q[int(actual_home), int(actual_away)])
        true_logp = float(np.log(pmf)) if pmf > 0 else -np.inf
    if not np.isfinite(true_logp):
        raise ArithmeticError("True score probability has invalid logarithm")
    axes = np.arange(n + 1, dtype=float)
    expected_h = float((q.sum(axis=1) * axes).sum())
    expected_a = float((q.sum(axis=0) * axes).sum())
    modal_h, modal_a = np.unravel_index(q.argmax(), q.shape)
    return {
        "expected_home_goals": expected_h, "expected_away_goals": expected_a,
        "mode_home_goals": int(modal_h), "mode_away_goals": int(modal_a),
        "true_score_probability": float(pmf), "true_score_nll": float(-true_logp),
        "actual_outside_grid": int(outside),
        "exact_score_hit": int(modal_h == actual_home and modal_a == actual_away),
        "home_abs_error": float(abs(expected_h - actual_home)),
        "away_abs_error": float(abs(expected_a - actual_away)),
        "home_within_1": int(abs(expected_h - actual_home) <= 1.0),
        "away_within_1": int(abs(expected_a - actual_away) <= 1.0),
        "total_abs_error": float(abs((expected_h + expected_a) - (actual_home + actual_away))),
        "total_within_1": int(abs((expected_h + expected_a) - (actual_home + actual_away)) <= 1.0),
    }
