import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

from src.platform_prediction import consistency_label, predicted_outcome


class PlatformPredictionTests(unittest.TestCase):
    def test_consistency_thresholds(self):
        self.assertEqual(consistency_label(0.10), "High")
        self.assertEqual(consistency_label(0.15), "Medium")
        self.assertEqual(consistency_label(0.21), "Low")

    def test_predicted_outcome(self):
        self.assertEqual(predicted_outcome([0.6, 0.2, 0.2]), "Home")
        self.assertEqual(predicted_outcome([0.2, 0.5, 0.3]), "Draw")
        self.assertEqual(predicted_outcome([0.2, 0.3, 0.5]), "Away")


if __name__ == "__main__":
    unittest.main()
