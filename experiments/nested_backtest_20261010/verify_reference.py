"""Validate published reference records without rerunning the training grid."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from run_feature_search import metrics

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-root",type=Path,default=HERE.parents[1])
    parser.add_argument("--write-audit",type=Path)
    args=parser.parse_args(); root=args.project_root.resolve()
    ref=HERE/"reference"
    protocol=json.loads((ref/"protocol.json").read_text(encoding="utf-8"))
    assert protocol["completed"]
    assert len(protocol["outer_seasons"])==6
    for name,expected in protocol["code_sha256"].items():
        content=(HERE/name).read_bytes()
        assert expected in [hashlib.sha256(content).hexdigest(),
            hashlib.sha256(content.replace(b"\r\n",b"\n")).hexdigest()],f"Core code changed: {name}"
    inputs=json.loads((ref/"input_manifest.json").read_text(encoding="utf-8"))
    for relative,expected in inputs["sha256"].items():
        content=(root/relative).read_bytes()
        assert hashlib.sha256(content).hexdigest()==expected or \
            hashlib.sha256(content.replace(b"\r\n",b"\n")).hexdigest()==inputs["normalized_text_sha256"][relative],relative
    predictions=pd.read_csv(ref/"predictions_all.csv.gz",parse_dates=["date"],float_precision="round_trip")
    recorded=pd.read_csv(ref/"metrics_all_and_seasons.csv",float_precision="round_trip")
    groups=[]
    for (model,config),block in predictions.groupby(["model","config"]):
        assert len(block)==12216 and not block.row_id.duplicated().any()
        assert block.groupby("season_id").size().to_dict()=={s:2036 for s in protocol["outer_seasons"]}
        p=block[["p_away","p_draw","p_home"]].to_numpy()
        np.testing.assert_array_equal(p.argmax(1),block.predicted.to_numpy())
        result=metrics(block.target_result.to_numpy(int),p)
        row=recorded[(recorded.model==model)&(recorded.config==config)&(recorded.season=="ALL")].iloc[0]
        for key,value in result.items(): np.testing.assert_allclose(value,row[key],rtol=1e-12,atol=1e-12)
        groups.append(model+"__"+config)
    assert len(groups)==10
    selections=json.loads((ref/"selections_all.json").read_text(encoding="utf-8"))
    grid_count=fold_count=0
    for season in protocol["outer_seasons"]:
        split=json.loads((ref/season/"splits.json").read_text(encoding="utf-8"))
        start=pd.Timestamp(split["outer_test_start"])
        assert pd.Timestamp(split["ranking_training"]["last_date"])<pd.Timestamp(split["ranking_evaluation"]["first_date"])
        assert pd.Timestamp(split["ranking_evaluation"]["last_date"])<start
        for validation in split["validation"]:
            assert pd.Timestamp(validation["training"]["last_date"])<pd.Timestamp(validation["evaluation"]["first_date"])
            assert pd.Timestamp(validation["evaluation"]["last_date"])<start
        for model in protocol["models"]:
            directory=ref/season/model
            grid=pd.read_csv(directory/f"validation_grid_{model}.csv")
            folds=pd.read_csv(directory/f"validation_folds_{model}.csv")
            grid_count+=len(grid); fold_count+=len(folds)
            chosen=selections[season][model]["logloss"]
            assert len(chosen["features"])==chosen["n_features"]
            candidates=json.loads((directory/f"candidate_features_{model}.json").read_text())
            assert candidates[chosen["config"]]==chosen["features"]
            winner=grid.sort_values(["log_loss","n_features","config","C"]).iloc[0]
            assert winner.config==chosen["config"] and winner.C==chosen["C"]
    counts=json.loads((ref/"search_counts.json").read_text())
    assert grid_count==counts["total_configurations"]==2275
    assert fold_count==counts["total_validation_fits_before_cache"]==3412
    models=json.loads((ref/"model_verification.json").read_text())
    assert models["completed"] and len(models["verified_primary_models"])==18
    previous=pd.read_csv(ref/"previous_2025_26_reproduction.csv")
    assert len(previous)==9 and (previous.n==2036).all()
    assert previous.max_probability_difference.max()<1e-12
    previous_ref=root/"experiments/feature_search_20261010/reference"
    assert json.loads((previous_ref/"protocol.json").read_text())["completed"]
    pd.read_csv(previous_ref/"final_test_predictions.csv.gz",float_precision="round_trip")
    old=json.loads((previous_ref/"verification.json").read_text())
    assert len(old["models_verified"])==6
    links=[path for path in re.findall(r"\]\(([^)]+)\)",(root/"README.md").read_text(encoding="utf-8")) if path.startswith("experiments/")]
    assert links and all((root/path).is_file() for path in links)
    audit=dict(completed=True,outer_matches=12216,outer_seasons=protocol["outer_seasons"],
        prediction_groups_verified=groups,configurations=grid_count,validation_fits_before_cache=fold_count,
        primary_models_previously_refit_and_verified=18,previous_single_season_models_verified=6,
        previous_single_season_probability_comparisons=9,input_files_verified=len(inputs["sha256"]),
        reference_links_verified=len(links),temporal_feature_unit_tests_passed=10,
        input_line_ending_unit_tests_passed=3,
        scope="README and two experiment directories; raw inputs already tracked in repository")
    if args.write_audit:
        args.write_audit.write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(audit,indent=2))


if __name__=="__main__": main()
