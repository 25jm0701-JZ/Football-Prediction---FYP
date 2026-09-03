import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

import numpy as np
import pandas as pd

from src.blend_backtest import blend_probabilities, optimize_weights
from src.historical_poisson import WORLD_CUP_CYCLES


class HistoricalBacktestTests(unittest.TestCase):
    def test_cycle_cutoffs_are_before_world_cup_start(self):
        starts = {
            2014: pd.Timestamp("2014-06-12"),
            2018: pd.Timestamp("2018-06-14"),
            2022: pd.Timestamp("2022-11-20"),
        }
        for year, cycle in WORLD_CUP_CYCLES.items():
            self.assertLess(cycle.cutoff_date, starts[year])

    def test_weight_optimizer_prefers_perfect_result_model(self):
        frame = pd.DataFrame(
            {
                "Result": ["H", "D", "A"] * 4,
                "ResultP_H": [0.9, 0.05, 0.05] * 4,
                "ResultP_D": [0.05, 0.9, 0.05] * 4,
                "ResultP_A": [0.05, 0.05, 0.9] * 4,
                "MarketProxyP_H": [1 / 3] * 12,
                "MarketProxyP_D": [1 / 3] * 12,
                "MarketProxyP_A": [1 / 3] * 12,
                "PoissonP_H": [1 / 3] * 12,
                "PoissonP_D": [1 / 3] * 12,
                "PoissonP_A": [1 / 3] * 12,
            }
        )
        weights, _ = optimize_weights(frame, step=0.1)
        self.assertEqual(weights, (1.0, 0.0, 0.0))
        probabilities = blend_probabilities(frame, weights)
        self.assertTrue(np.allclose(probabilities[:, 0].max(), 0.9))


if __name__ == "__main__":
    unittest.main()
