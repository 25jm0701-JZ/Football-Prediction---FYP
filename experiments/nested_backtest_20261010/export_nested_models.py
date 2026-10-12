"""Refit and serialize primary selected models, checking all six outer folds."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import run_feature_search as fs

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=HERE/"results")
    args=parser.parse_args(); out=args.output_dir.resolve()
    protocol=json.loads((out/"protocol.json").read_text())
    assert protocol["completed"]
    data=pd.read_csv(out/"feature_matrix.csv",parse_dates=["date"],float_precision="round_trip")
    predictions=pd.read_csv(out/"predictions_all.csv",parse_dates=["date"],float_precision="round_trip")
    choices=json.loads((out/"selections_all.json").read_text())
    factory=fs.make_model
    def single_thread(kind,c=1.,seed=42):
        model=factory(kind,c,seed)
        if kind=="rf": model.set_params(model__n_jobs=1)
        return model
    fs.make_model=single_thread
    folder=out/"models"; folder.mkdir(exist_ok=True)
    verified=[]
    with threadpool_limits(limits=1):
        for season in protocol["outer_seasons"]:
            for kind in protocol["models"]:
                selection=choices[season][kind]["logloss"]
                model,test,p=fs.fit_predict(data,selection["features"],season,kind,selection["C"])
                original=predictions[(predictions.season_id==season)&(predictions.model==kind)&(predictions.config=="selected_logloss")]
                assert original.row_id.tolist()==test.index.tolist()
                np.testing.assert_allclose(original[["p_away","p_draw","p_home"]],p,rtol=1e-7,atol=1e-9)
                name=season+"__"+kind
                joblib.dump(model,folder/(name+".joblib"))
                reloaded=joblib.load(folder/(name+".joblib"))
                np.testing.assert_allclose(reloaded.predict_proba(test[selection["features"]].to_numpy(float)),p,rtol=1e-12,atol=1e-12)
                schema=dict(features=selection["features"],C=selection["C"],model=kind,
                    class_order=["away","draw","home"],trained_before_season=season,selected_on_inner_validation=True)
                (folder/(name+".json")).write_text(json.dumps(schema,indent=2),encoding="utf-8")
                verified.append(name)
                print("VERIFIED",name,flush=True)
    (out/"model_verification.json").write_text(json.dumps(dict(verified_primary_models=verified,completed=True),indent=2),encoding="utf-8")


if __name__=="__main__": main()
