import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

import pandas as pd

from src.international_training_data import (
    PreparationConfig,
    add_training_weights,
    apply_known_result_adjustments,
    competition_category,
    core_duplicate_indices,
    filter_external_results,
    match_key,
    standardize_team,
)


class InternationalTrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.config = PreparationConfig(
            start_date=pd.Timestamp("2022-12-19"),
            cutoff_date=pd.Timestamp("2026-06-10"),
            half_life_days=540,
        )

    def test_competition_categories(self):
        self.assertEqual(
            competition_category("FIFA World Cup qualification"),
            "world_cup_qualification",
        )
        self.assertEqual(
            competition_category("UEFA Nations League"),
            "nations_league",
        )
        self.assertEqual(competition_category("Friendly"), "friendly")
        self.assertEqual(
            competition_category("Mauritius Four Nations Cup"),
            "other_senior",
        )

    def test_filter_keeps_current_senior_teams_only(self):
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    ["2025-01-01", "2025-01-02", "2025-01-03", "2020-01-01"]
                ),
                "Home": ["A", "A", "A U-23", "A"],
                "Away": ["B", "Outside", "B", "B"],
                "HomeGoals": [1, 1, 1, 1],
                "AwayGoals": [0, 0, 0, 0],
                "Tournament": [
                    "Friendly",
                    "Friendly",
                    "Olympic Games",
                    "Friendly",
                ],
            }
        )
        filtered, audit = filter_external_results(
            frame,
            team_universe={"A", "B", "A U-23"},
            config=self.config,
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["Away"], "B")
        self.assertEqual(audit["after_score_validation"], 1)

    def test_time_weight_halves_after_540_days(self):
        frame = pd.DataFrame(
            {
                "Date": [self.config.cutoff_date - pd.Timedelta(days=540)],
                "Tournament": ["FIFA World Cup qualification"],
            }
        )
        weighted = add_training_weights(frame, self.config)
        self.assertAlmostEqual(float(weighted.iloc[0]["TimeWeight"]), 0.5)
        self.assertAlmostEqual(float(weighted.iloc[0]["FinalWeight"]), 0.5)

    def test_match_key_ignores_home_away_order(self):
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2025-03-01", "2025-03-01"]),
                "Home": ["A", "B"],
                "Away": ["B", "A"],
            }
        )
        keys = match_key(frame)
        self.assertEqual(keys.iloc[0], keys.iloc[1])

    def test_team_aliases_match_project_names(self):
        self.assertEqual(standardize_team("Republic of Ireland"), "Ireland")
        self.assertEqual(standardize_team("Taiwan"), "Chinese Taipei")
        self.assertEqual(
            standardize_team("Central African Republic"),
            "Central Africa",
        )

    def test_cross_source_duplicate_allows_one_day_date_shift(self):
        core = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2025-06-08"]),
                "Home": ["Bahamas"],
                "Away": ["Costa Rica"],
                "HomeGoals": [0],
                "AwayGoals": [8],
            }
        )
        external = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2025-06-07", "2025-01-01"]),
                "Home": ["Bahamas", "Bahamas"],
                "Away": ["Costa Rica", "Costa Rica"],
                "HomeGoals": [0, 0],
                "AwayGoals": [8, 8],
            }
        )
        duplicates = core_duplicate_indices(core, external)
        self.assertEqual(duplicates, {0})

    def test_known_forfeit_restores_played_score_and_unplayed_match_is_removed(self):
        loaded = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-11-15", "2025-10-09"]),
                "Home": ["Equatorial Guinea", "Malawi"],
                "Away": ["Namibia", "Equatorial Guinea"],
                "HomeGoals": [0, 3],
                "AwayGoals": [3, 0],
                "Tournament": [
                    "FIFA World Cup qualification",
                    "FIFA World Cup qualification",
                ],
            }
        )
        loaded = apply_known_result_adjustments(loaded)
        filtered, audit = filter_external_results(
            loaded,
            team_universe={"Equatorial Guinea", "Namibia", "Malawi"},
            config=self.config,
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(int(filtered.iloc[0]["HomeGoals"]), 1)
        self.assertEqual(int(filtered.iloc[0]["AwayGoals"]), 0)
        self.assertTrue(bool(filtered.iloc[0]["ScoreAdjusted"]))
        self.assertEqual(audit["known_score_adjustments"], 2)
        self.assertEqual(audit["known_unplayed_matches"], 1)


if __name__ == "__main__":
    unittest.main()
