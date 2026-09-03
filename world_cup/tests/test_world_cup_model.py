import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

import numpy as np
import pandas as pd

from src.world_cup_model import (
    fifa_expected_score,
    remove_overround,
    round_probability_columns,
    update_fifa_points,
    update_live_ratings,
)


class WorldCupModelTests(unittest.TestCase):
    def test_overround_probabilities_sum_to_one(self):
        probabilities = remove_overround([2.0], [3.5], [4.0])
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)

    def test_equal_ratings_have_equal_expected_scores(self):
        self.assertAlmostEqual(fifa_expected_score(1800, 1800), 0.5)

    def test_rounded_probabilities_still_sum_to_one(self):
        frame = pd.DataFrame(
            {"P_H": [1 / 3], "P_D": [1 / 3], "P_A": [1 / 3]}
        )
        rounded = round_probability_columns(frame, ["P_H", "P_D", "P_A"])
        self.assertEqual(float(rounded.iloc[0].sum()), 1.0)
        self.assertTrue(all(len(str(value).split(".")[-1]) <= 3 for value in rounded.iloc[0]))

    def test_standard_match_update_is_zero_sum(self):
        updated_home, updated_away = update_fifa_points(
            1800, 1700, 2, 1, importance=50
        )
        self.assertAlmostEqual(updated_home + updated_away, 3500)
        self.assertGreater(updated_home, 1800)
        self.assertLess(updated_away, 1700)

    def test_live_table_is_reranked(self):
        ratings = pd.DataFrame(
            {
                "Country": ["A", "B"],
                "Rank": [1, 2],
                "Points": [1800.0, 1799.0],
            }
        )
        updated = update_live_ratings(ratings, "B", "A", 1, 0, importance=50)
        self.assertEqual(updated.iloc[0]["Country"], "B")
        self.assertTrue(np.array_equal(updated["Rank"].to_numpy(), [1, 2]))


if __name__ == "__main__":
    unittest.main()
