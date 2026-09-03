import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

import numpy as np
import pandas as pd

from src.poisson_model import (
    WeightedPoissonModel,
    score_matrix,
    score_summary,
)


class PoissonModelTests(unittest.TestCase):
    def test_score_matrix_sums_to_one(self):
        matrix = score_matrix(1.7, 0.9, max_goals=10)
        self.assertAlmostEqual(float(matrix.sum()), 1.0)

    def test_result_probabilities_sum_to_one(self):
        summary = score_summary(1.7, 0.9)
        self.assertAlmostEqual(
            summary["P_H"] + summary["P_D"] + summary["P_A"],
            1.0,
        )
        self.assertEqual(summary["scores"][0]["score"], "1-0")

    def test_model_learns_stronger_attack(self):
        matches = pd.DataFrame(
            {
                "Home": ["Strong", "Weak"] * 20,
                "Away": ["Weak", "Strong"] * 20,
                "HomeGoals": [3, 0] * 20,
                "AwayGoals": [0, 2] * 20,
                "FinalWeight": [1.0] * 40,
                "TrueHome": [0] * 40,
            }
        )
        model = WeightedPoissonModel(
            max_iter=4000,
            tolerance=1e-9,
        ).fit(matches)
        index = {team: i for i, team in enumerate(model.teams)}
        self.assertGreater(
            model.attack[index["Strong"]],
            model.attack[index["Weak"]],
        )
        strong_xg, weak_xg = model.expected_goals("Strong", "Weak")
        self.assertGreater(strong_xg, weak_xg)

    def test_serialization_preserves_predictions(self):
        model = WeightedPoissonModel(
            teams=["A", "B"],
            intercept=0.2,
            home_advantage=0.1,
            attack=np.array([0.3, -0.3]),
            defence_weakness=np.array([-0.2, 0.2]),
        )
        restored = WeightedPoissonModel.from_dict(model.to_dict())
        self.assertTrue(
            np.allclose(
                model.expected_goals("A", "B", 1),
                restored.expected_goals("A", "B", 1),
            )
        )


if __name__ == "__main__":
    unittest.main()
