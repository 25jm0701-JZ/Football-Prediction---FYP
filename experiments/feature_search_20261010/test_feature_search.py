"""Meaningful invariants for the chronological feature-search experiment."""
from pathlib import Path
import argparse
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_feature_search as experiment

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[2])
_args, _unittest_args = _parser.parse_known_args()
PROJECT = _args.project_root.resolve()
if not (PROJECT/"scripts/atta_mills_tracka_english4.py").is_file():
    raise FileNotFoundError("Run inside the cloned FYP repository, or pass --project-root PATH.")
sys.path.insert(0, str(PROJECT))
from scripts import atta_mills_tracka_english4 as existing


def fixture():
    records = [
        ("2022-07-28", "A", "B", 1, 0),
        ("2022-07-28", "C", "D", 0, 1),
        ("2022-07-29", "A", "C", 2, 1),
        ("2022-07-29", "B", "D", 1, 1),
        ("2022-07-30", "A", "B", 0, 2),
    ]
    return pd.DataFrame([dict(date=pd.Timestamp(date),home_team=h,away_team=a,
        home_goals_full=hg,away_goals_full=ag,home_shots=10.,away_shots=9.,
        home_shots_target=4.,away_shots_target=3.,league_level=1,division="E1",
        source_file="E12223.csv",season_id="2021-22",target_result=2 if hg>ag else 0 if hg<ag else 1)
        for date,h,a,hg,ag in records])


class TemporalInvariants(unittest.TestCase):
    def build(self, raw, name):
        out=Path(__file__).resolve().parent/"checks"/name
        out.mkdir(parents=True,exist_ok=True)
        with patch.object(existing,"load_english4_data",return_value=raw):
            return experiment.build_data(PROJECT,out)

    def test_current_and_same_day_outcomes_cannot_change_features(self):
        original=fixture()
        changed=fixture()
        # Change one game: its own and the other match on that date stay unchanged.
        changed.loc[2,["home_goals_full","away_goals_full","target_result"]]=[9,0,2]
        changed.loc[2,"home_shots_target"]=9.
        base,cols=self.build(original,"original")
        altered,_=self.build(changed,"altered")
        before=base.date<=pd.Timestamp("2022-07-29")
        np.testing.assert_allclose(base.loc[before,cols],altered.loc[before,cols],equal_nan=True)
        later=base.date==pd.Timestamp("2022-07-30")
        self.assertFalse(np.allclose(base.loc[later,cols],altered.loc[later,cols],equal_nan=True))

    def test_july_uses_source_season_and_league_is_not_duplicated(self):
        df,cols=self.build(fixture(),"season")
        self.assertEqual(set(df.season_id),{"2022-23"})
        self.assertEqual(cols.count("league_level"),1)
        self.assertEqual(df.columns.tolist().count("league_level"),1)
        self.assertEqual(len(cols),66)

    def test_validation_training_excludes_evaluation_and_future(self):
        # A deliberately overlapping chronology must fail before fitting.
        df=pd.DataFrame(dict(season_id=["2021-22","2022-23"],
            date=pd.to_datetime(["2022-08-01","2022-07-01"]),
            target_result=[0,1],example=[1.,2.]))
        with self.assertRaisesRegex(AssertionError,"Chronology overlaps"):
            experiment.fit_predict(df,["example"],"2022-23","lr")

    def test_metrics_use_away_draw_home_and_existing_brier_convention(self):
        y=np.array([0,1,2])
        p=np.array([[.7,.2,.1],[.1,.8,.1],[.2,.2,.6]])
        report=experiment.metrics(y,p)
        self.assertEqual(report["accuracy"],1.)
        self.assertEqual(report["draw_f1"],1.)
        self.assertAlmostEqual(report["brier"],np.mean((np.eye(3)-p)**2))
        self.assertAlmostEqual(report["log_loss"],-np.mean(np.log([.7,.8,.6])))

    def test_group_permutation_preserves_joint_rows_and_league(self):
        x=np.array([[1,10,0],[2,20,0],[3,30,1],[4,40,1]],dtype=float)
        groups=np.array(["E0","E0","E1","E1"])
        shuffled=experiment.shuffle_rows(x,[0,1],groups,np.random.default_rng(42))
        np.testing.assert_equal(shuffled[:,1],10*shuffled[:,0])
        np.testing.assert_equal(shuffled[:,2],x[:,2])
        self.assertEqual(set(shuffled[:2,0]),{1.,2.})
        self.assertEqual(set(shuffled[2:,0]),{3.,4.})


if __name__=="__main__": unittest.main(argv=[sys.argv[0],*_unittest_args],verbosity=2)
