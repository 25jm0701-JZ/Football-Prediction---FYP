"""Persist evaluated pipelines and verify their probabilities against saved outputs."""
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
import run_feature_search as experiment
from run_feature_search import fit_predict, TEST_SEASON

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"


def main():
    # Single-thread refits also run inside Windows restricted environments.
    factory=experiment.make_model
    def single_thread(kind,c=1.0,seed=42):
        pipeline=factory(kind,c,seed)
        if kind=="rf": pipeline.set_params(model__n_jobs=1)
        return pipeline
    experiment.make_model=single_thread
    protocol=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    assert protocol["completed"]
    df=pd.read_csv(OUT/"feature_matrix.csv",parse_dates=["date"],float_precision="round_trip")
    frozen=json.loads((OUT/"frozen_selections.json").read_text(encoding="utf-8"))
    predictions=pd.read_csv(OUT/"final_test_predictions.csv",parse_dates=["date"],float_precision="round_trip")
    model_dir=OUT/"models"
    model_dir.mkdir(exist_ok=True)
    verified=[]
    with threadpool_limits(limits=1):
        for key,selection in frozen.items():
            kind,objective=key.split("__",1)
            model,te,p=fit_predict(df,selection["features"],TEST_SEASON,kind,selection["C"])
            original=predictions[(predictions.model==kind)&(predictions.config==f"selected_{objective}")]
            assert original.home_team.to_list()==te.home_team.to_list()
            assert original.away_team.to_list()==te.away_team.to_list()
            np.testing.assert_allclose(original[["p_away","p_draw","p_home"]],p,rtol=1e-7,atol=1e-9)
            target=model_dir/f"{key}.joblib"
            joblib.dump(model,target)
            loaded=joblib.load(target)
            np.testing.assert_allclose(loaded.predict_proba(te[selection["features"]].to_numpy(float)),p,atol=1e-12)
            schema=dict(features=selection["features"],class_order=["away","draw","home"],
                trained_before_season=TEST_SEASON,model=kind,C=selection["C"],
                n_features=len(selection["features"]),selected_on_validation=True)
            target.with_suffix(".json").write_text(json.dumps(schema,indent=2),encoding="utf-8")
            verified.append(key)
            print("VERIFIED MODEL",key,flush=True)
    (OUT/"verification.json").write_text(json.dumps(dict(models_verified=verified,
        invariant_tests_passed=5,python=sys.version,sklearn=sklearn.__version__,
        numpy=np.__version__,pandas=pd.__version__),indent=2),encoding="utf-8")


if __name__=="__main__": main()
