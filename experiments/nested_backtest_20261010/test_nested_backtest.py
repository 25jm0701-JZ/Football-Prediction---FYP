"""Checks for nested temporal selection, including the short-history folds."""
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import run_nested_backtest as nested


def fixture():
    rows=[]
    for year in range(2019,2026):
        for i in range(30):
            rows.append(dict(date=pd.Timestamp(year,8,1)+pd.Timedelta(days=i),
                season_id=f"{year}-{str(year+1)[-2:]}", target_result=i%3, example=float(i)))
    return pd.DataFrame(rows)


class NestedChronology(unittest.TestCase):
    def test_all_six_outer_splits_exclude_test_and_future(self):
        data=fixture()
        for season in sorted(data.season_id.unique())[1:]:
            view,rank,valid=nested.inner_protocol(data[data.season_id<season])
            self.assertLess(view.date.max(),data.loc[data.season_id==season,"date"].min())
            self.assertLess(view.loc[view.season_id<rank,"date"].max(),view.loc[view.season_id==rank,"date"].min())
            self.assertLess(view.loc[view.season_id==rank,"date"].max(),view.loc[view.season_id.isin(valid),"date"].min())

    def test_poisoning_outer_and_future_labels_does_not_change_inner_view(self):
        data=fixture(); changed=data.copy()
        changed.loc[changed.season_id>="2022-23",["target_result","example"]]=[999,1e9]
        a,rank_a,val_a=nested.inner_protocol(data[data.season_id<"2022-23"])
        b,rank_b,val_b=nested.inner_protocol(changed[changed.season_id<"2022-23"])
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual((rank_a,val_a),(rank_b,val_b))

    def test_last_fold_matches_previous_search_calendar(self):
        data=fixture()
        _,rank,valid=nested.inner_protocol(data[data.season_id<"2025-26"])
        self.assertEqual(rank,"2022-23")
        self.assertEqual(valid,["2023-24","2024-25"])

    def test_overlapping_validation_is_rejected_before_cache_or_fit(self):
        data=pd.DataFrame(dict(date=pd.to_datetime(["2022-08-01","2022-07-01"]),
            season_id=["a","b"],target_result=[0,1],example=[1.,2.]))
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(nested,"ORIGINAL_FIT") as mocked:
                with self.assertRaisesRegex(AssertionError,"Chronology overlaps"):
                    nested.cached_fit(Path(temporary),"test")(data,["example"],"b","lr")
                mocked.assert_not_called()

    def test_identical_predictions_have_zero_paired_difference_and_interval(self):
        a=fixture().rename_axis("row_id").reset_index()
        a["config"]="all_66"; a[["p_away","p_draw","p_home"]]=[.3,.3,.4]
        b=a.copy(); b["config"]="selected_logloss"
        ci=nested.stratified_ci(pd.concat([a,b]),"all_66","selected_logloss",100)
        for value in ci.values():
            for field in ["delta","ci95_low","ci95_high"]:
                self.assertAlmostEqual(value[field],0.)


if __name__=="__main__": unittest.main(verbosity=2)
