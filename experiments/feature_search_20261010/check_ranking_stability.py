"""Descriptive ranking stability on development seasons; does not select models."""
from pathlib import Path
import json
import pandas as pd
from threadpoolctl import threadpool_limits
import run_feature_search as experiment


def main():
    here=Path(__file__).resolve().parent
    root=here/"results"
    df=pd.read_csv(root/"feature_matrix.csv",parse_dates=["date"],float_precision="round_trip")
    cols=pd.read_csv(root/"feature_groups.csv").feature.to_list()
    factory=experiment.make_model
    def single_thread(kind,c=1.0,seed=42):
        pipeline=factory(kind,c,seed)
        if kind=="rf": pipeline.set_params(model__n_jobs=1)
        return pipeline
    experiment.make_model=single_thread
    with threadpool_limits(limits=1):
        for kind in ["lr","rf"]:
            frames=[pd.read_csv(root/f"importance_{kind}.csv").assign(season="2022-23")]
            for season in ["2023-24","2024-25"]:
                experiment.RANK_SEASON=season
                out=root/"ranking_stability"/season
                out.mkdir(parents=True,exist_ok=True)
                frames.append(experiment.importance(df,cols,kind,out,repeats=5).assign(season=season))
            combined=pd.concat(frames)
            combined.to_csv(root/f"importance_by_development_season_{kind}.csv",index=False)
            summary=combined.groupby("feature").agg(
                mean_logloss_increase=("permutation_logloss_mean","mean"),
                mean_rank=("rank","mean"),worst_rank=("rank","max"),
                positive_seasons=("permutation_logloss_mean",lambda x:int((x>0).sum())),
                top15_seasons=("rank",lambda x:int((x<=15).sum())))
            summary=summary.sort_values(["mean_logloss_increase","mean_rank"],ascending=[False,True])
            summary.to_csv(root/f"importance_stability_{kind}.csv")
            print("STABILITY",kind,summary.head(10).to_string(),flush=True)
    (root/"ranking_stability_note.json").write_text(json.dumps(dict(
        seasons=["2022-23","2023-24","2024-25"],purpose="descriptive only",
        not_used_to_change_frozen_feature_sets=True,final_test_not_used=True),indent=2),encoding="utf-8")


if __name__=="__main__": main()
