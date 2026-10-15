"""Tests for the new experiment's season and selection boundaries."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from run_experiment import fit_lr, frozen_selection, season_from_source


class ProtocolTests(unittest.TestCase):
    def test_file_season_does_not_depend_on_match_month(self):
        self.assertEqual(season_from_source('E01920.csv'), '2019-20')
        self.assertEqual(season_from_source('E12223.csv'), '2022-23')
        self.assertEqual(season_from_source('E32526.csv'), '2025-26')

    def test_invalid_source_season_fails(self):
        for name in ['E12224.csv', 'E42223.csv', '../E12223.csv', 'PL2223.csv']:
            with self.assertRaises(ValueError):
                season_from_source(name)

    def test_fit_rejects_future_dates_before_training(self):
        train = pd.DataFrame(dict(date=pd.to_datetime(['2022-09-01']), season_id=['2021-22']))
        test = pd.DataFrame(dict(date=pd.to_datetime(['2022-08-01']), season_id=['2022-23']))
        with self.assertRaisesRegex(AssertionError, 'future'):
            fit_lr(train, test, ['example'])

    def test_fit_rejects_target_season_even_with_earlier_date(self):
        train = pd.DataFrame(dict(date=pd.to_datetime(['2022-07-01']), season_id=['2022-23']))
        test = pd.DataFrame(dict(date=pd.to_datetime(['2022-08-01']), season_id=['2022-23']))
        with self.assertRaisesRegex(AssertionError, 'test-season'):
            fit_lr(train, test, ['example'])

    def test_selection_accepts_new_count_and_uses_validation_loss(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            record = dict(config='new_two', C=1.0, n_features=2, features=['a', 'b'])
            (directory / 'frozen_selections.json').write_text(json.dumps({'lr__logloss': record}))
            (directory / 'protocol.json').write_text(json.dumps(dict(completed=True, primary_selection_metric='log_loss')))
            (directory / 'candidate_features_lr.json').write_text(json.dumps({'new_two': ['a', 'b']}))
            pd.DataFrame([dict(config='all_three', C=1.0, n_features=3, log_loss=1.1),
                          dict(config='new_two', C=1.0, n_features=2, log_loss=1.0)]).to_csv(directory / 'validation_grid_lr.csv', index=False)
            columns, _ = frozen_selection(directory, ['a', 'b', 'c'])
            self.assertEqual(columns, ['a', 'b'])
            record['features'] = ['a', 'c']
            (directory / 'frozen_selections.json').write_text(json.dumps({'lr__logloss': record}))
            with self.assertRaises(AssertionError):
                frozen_selection(directory, ['a', 'b', 'c'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
