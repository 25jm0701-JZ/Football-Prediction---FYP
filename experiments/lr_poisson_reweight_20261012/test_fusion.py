"""Run: python -m unittest discover -s experiments/lr_poisson_reweight_20261012 -p 'test_*.py' -v"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_core import (aggregate_hda, adaptive_poisson_grid, reweight_scores,
                         score_statistics)


class ConditionalReweightingTests(unittest.TestCase):
    def test_normalization_marginals_ratios_and_no_negative(self):
        grid = adaptive_poisson_grid(1.6, 1.2).matrix
        target = np.array([0.18, 0.35, 0.47])  # [away, draw, home]
        combined = reweight_scores(grid, target)
        np.testing.assert_allclose(aggregate_hda(combined), target, atol=1e-14)
        self.assertAlmostEqual(combined.sum(), 1.0)
        self.assertTrue(np.all(combined >= 0))
        # Same result: (2,0)/(3,0) stays invariant.
        self.assertAlmostEqual(combined[2, 0]/combined[3, 0], grid[2, 0]/grid[3, 0])
        # Conditional draw scores 0-0 and 1-1 have same preserved ratio.
        self.assertAlmostEqual(combined[0, 0]/combined[1, 1], grid[0, 0]/grid[1, 1])

    def test_identity_when_marginals_match_poisson(self):
        q = adaptive_poisson_grid(1.1, 1.9).matrix
        np.testing.assert_allclose(reweight_scores(q, aggregate_hda(q)), q, atol=1e-15)

    def test_zero_lr_class_is_allowed(self):
        q = adaptive_poisson_grid(1.4, 1.1).matrix
        p = [0.0, 0.1, 0.9]
        fused = reweight_scores(q, p)
        np.testing.assert_allclose(aggregate_hda(fused), p, atol=1e-15)

    def test_zero_poisson_class_with_positive_target_fails(self):
        q = np.array([[0.5, 0.0], [0.0, 0.5]])  # no H or A possible
        with self.assertRaisesRegex(ValueError, "Poisson mass is zero"):
            reweight_scores(q, [0.2, 0.6, 0.2])

    def test_zero_zero_class_is_acceptable(self):
        q = np.array([[0.4, 0], [0, 0.6]])
        np.testing.assert_allclose(reweight_scores(q, [0, 1, 0]), q)

    def test_tiny_nonzero_denominator(self):
        tiny = 1e-300
        q = np.array([[0.7, tiny], [0.3 - tiny, 0.]])
        p = [0.3, 0.2, 0.5]
        fused = reweight_scores(q, p)
        np.testing.assert_allclose(aggregate_hda(fused), p, atol=1e-15)
        self.assertTrue(np.all(np.isfinite(fused)))

    def test_reject_corrupt_probabilities(self):
        q = np.array([[0.5, 0.1], [0.1, 0.3]])
        for p in [[0.2, 0.3, 0.3], [-0.1, 0.6, 0.5], [float('nan'), 0.5, 0.5]]:
            with self.assertRaises(ValueError): reweight_scores(q, p)
        with self.assertRaises(ValueError): reweight_scores(q*0.5, [0.2, 0.3, 0.5])
        bad = q.copy();bad[1,0]=-0.1
        with self.assertRaises(ValueError): reweight_scores(bad, [0.2, 0.3, 0.5])

    def test_tail_coverage_and_finite_rare_score(self):
        grid = adaptive_poisson_grid(5.5, 0.7, tail_tolerance=1e-13)
        self.assertLessEqual(grid.omitted_mass, 1.0001e-13)
        self.assertAlmostEqual(grid.matrix.sum(), 1)
        self.assertTrue(np.all(np.isfinite(aggregate_hda(grid.matrix))))
        fused = reweight_scores(grid.matrix, [0.1, 0.1, 0.8])
        s = score_statistics(fused, 5, 2)
        self.assertTrue(np.isfinite(s['true_score_nll']))

    def test_mean_vs_mode_are_separate(self):
        q = adaptive_poisson_grid(1.6, 1.2).matrix
        s = score_statistics(q, 2, 1)
        self.assertAlmostEqual(s['expected_home_goals'], 1.6, delta=1e-8)
        self.assertAlmostEqual(s['expected_away_goals'], 1.2, delta=1e-8)
        self.assertEqual(s['mode_home_goals'], 1)
        self.assertEqual(s['mode_away_goals'], 1)

    def test_rare_mass_with_nonzero_lr_not_silently_clipped(self):
        q = np.array([[0.55, 1e-18], [0.45-1e-18, 0.0]])
        p = [0.1, 0.1, 0.8]
        fused = reweight_scores(q, p)
        np.testing.assert_allclose(aggregate_hda(fused), p, atol=1e-15)


class OutsideGridNllTests(unittest.TestCase):
    def test_analytic_outside_support(self):
        import math
        from scipy.stats import poisson
        p = adaptive_poisson_grid(1.1, 1.3).matrix
        actual = len(p) + 2
        exp0 = score_statistics(p, actual, 0, lambda_home=1.1, lambda_away=1.3)
        self.assertEqual(exp0['actual_outside_grid'], 1)
        mass = poisson.cdf(len(p)-1, 1.1) * poisson.cdf(len(p)-1, 1.3)
        manual = -(poisson.logpmf(actual, 1.1) + poisson.logpmf(0, 1.3) - np.log(mass))
        self.assertAlmostEqual(exp0['true_score_nll'], manual)
        target = np.array([0.24, 0.30, 0.46])
        f = reweight_scores(p, target)
        exp1 = score_statistics(f, actual, 0, lambda_home=1.1,
                                lambda_away=1.3, target_hda=target)
        self.assertEqual(exp1['actual_outside_grid'], 1)
        self.assertAlmostEqual(exp1['true_score_nll'] - exp0['true_score_nll'],
                               np.log(aggregate_hda(p)[2] / target[2]), places=9)


class NllIdentityTests(unittest.TestCase):
    def test_score_nll_change_equals_outcome_nll_change(self):
        p = adaptive_poisson_grid(1.5, 0.95).matrix
        q = np.array([0.22, 0.25, 0.53])
        f = reweight_scores(p, q)
        base_outcomes = aggregate_hda(p)
        for home, away in [(0, 0), (2, 0), (0, 2), (3, 2), (1, 3)]:
            c = 0 if home < away else 1 if home == away else 2
            a = score_statistics(p, home, away)['true_score_nll']
            b = score_statistics(f, home, away)['true_score_nll']
            self.assertAlmostEqual(b-a, np.log(base_outcomes[c])-np.log(q[c]), places=12)


if __name__ == '__main__': unittest.main()
