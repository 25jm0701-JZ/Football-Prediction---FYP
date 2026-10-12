"""Nested chronological English4 feature selection; no outer labels enter selection."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import run_feature_search as fs

HERE = Path(__file__).resolve().parent
MODELS = ["lr", "rf", "lr_scaled"]
ORIGINAL_FIT = fs.fit_predict


def chronological_cut(frame, fraction):
    # Approximate a fraction of matches, preserving whole dates at boundaries.
    counts = frame.groupby("date").size().sort_index()
    return counts.index[np.searchsorted(counts.cumsum().to_numpy(),len(frame)*fraction)]


def inner_protocol(past):
    """Use only past seasons; short histories require date-block inner splits."""
    seasons = sorted(past.season_id.unique())
    view = past.copy()
    if len(seasons) == 1:
        cut1, cut2 = chronological_cut(view,.6), chronological_cut(view,.8)
        view["season_id"] = np.where(view.date < cut1, "2019-20-a",
            np.where(view.date < cut2, "2019-20-b", "2019-20-c"))
        rank, valid = "2019-20-b", ["2019-20-c"]
    elif len(seasons) == 2:
        mask = view.season_id == seasons[0]
        cut = chronological_cut(view.loc[mask],.6)
        view.loc[mask, "season_id"] = np.where(view.loc[mask,"date"] < cut,
            "2019-20-a", "2019-20-b")
        rank, valid = "2019-20-b", [seasons[-1]]
    elif len(seasons) == 3:
        rank, valid = seasons[-2], [seasons[-1]]
    else:
        rank, valid = seasons[-3], seasons[-2:]
    assert not view.empty
    assert view.loc[view.season_id < rank, "date"].max() < view.loc[view.season_id == rank,"date"].min()
    assert view.loc[view.season_id == rank,"date"].max() < view.loc[view.season_id.isin(valid),"date"].min()
    for season in valid:
        assert view.loc[view.season_id < season,"date"].max() < view.loc[view.season_id == season,"date"].min()
    return view, rank, valid


def split_description(view, rank, valid):
    def describe(x):
        return dict(n=len(x), first_date=str(x.date.min().date()), last_date=str(x.date.max().date()))
    return dict(ranking_training=describe(view[view.season_id < rank]),
        ranking_evaluation=dict(block=rank, **describe(view[view.season_id == rank])),
        validation=[dict(block=s, training=describe(view[view.season_id < s]),
            evaluation=describe(view[view.season_id == s])) for s in valid])


def cached_fit(cache, data_hash):
    def fit(view, cols, season, kind, c=1., seed=42):
        tr, te = view[view.season_id < season], view[view.season_id == season]
        assert tr.date.max() < te.date.min(), "Chronology overlaps"
        identity = dict(data=data_hash, features=cols, train=tr.index.tolist(),
            test=te.index.tolist(), model=kind, C=c, seed=seed)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        dest = cache / (digest + ".npy")
        if dest.exists():
            p = np.load(dest, allow_pickle=False)
        else:
            _, _, p = ORIGINAL_FIT(view, cols, season, kind, c, seed)
            temp = dest.with_suffix(".tmp")
            with temp.open("wb") as handle: np.save(handle, p, allow_pickle=False)
            temp.replace(dest)
        assert p.shape == (len(te),3) and np.isfinite(p).all()
        return None, te, p
    return fit


def stratified_ci(predictions, base, selected, repeats=4000):
    """Paired date clusters, resampled separately inside each outer season."""
    a = predictions[predictions.config == base].sort_values("row_id")
    b = predictions[predictions.config == selected].sort_values("row_id")
    assert a.row_id.tolist() == b.row_id.tolist()
    y = a.target_result.to_numpy(int)
    pa, pb = a[["p_away","p_draw","p_home"]].to_numpy(), b[["p_away","p_draw","p_home"]].to_numpy()
    deltas = pd.DataFrame(dict(season=a.season_id.to_numpy(), date=a.date.to_numpy(),
        accuracy=(pb.argmax(1)==y).astype(float)-(pa.argmax(1)==y).astype(float),
        logloss=-np.log(np.clip(pb[np.arange(len(y)),y],1e-15,1))+np.log(np.clip(pa[np.arange(len(y)),y],1e-15,1))))
    rng = np.random.default_rng(123)
    result = {}
    for field in ["accuracy", "logloss"]:
        sums_total = np.zeros(repeats); counts_total = np.zeros(repeats)
        for _, block in deltas.groupby("season"):
            agg = block.groupby("date")[field].agg(["sum","count"])
            sums, counts = agg["sum"].to_numpy(), agg["count"].to_numpy()
            indices = rng.integers(0,len(sums),size=(repeats,len(sums)))
            sums_total += sums[indices].sum(1); counts_total += counts[indices].sum(1)
        samples = sums_total / counts_total
        result[field] = dict(delta=float(deltas[field].mean()), ci95_low=float(np.quantile(samples,.025)),
            ci95_high=float(np.quantile(samples,.975)), repeats=repeats)
    return result


def save_summaries(out, completed_seasons):
    frames = [pd.read_csv(out/s/"outer_predictions.csv",float_precision="round_trip",parse_dates=["date"]) for s in completed_seasons]
    all_predictions = pd.concat(frames,ignore_index=True)
    all_predictions.to_csv(out/"predictions_all.csv",index=False)
    metric_rows, league_rows = [], []
    for (kind, config), group in all_predictions.groupby(["model","config"]):
        for season, block in [("ALL",group), *list(group.groupby("season_id"))]:
            p = block[["p_away","p_draw","p_home"]].to_numpy()
            row = dict(model=kind,config=config,season=season,**fs.metrics(block.target_result.to_numpy(int),p))
            row["correct"] = int((p.argmax(1)==block.target_result.to_numpy(int)).sum())
            metric_rows.append(row)
        for div, block in group.groupby("division"):
            league_rows.append(dict(model=kind,config=config,division=div,
                **fs.metrics(block.target_result.to_numpy(int),block[["p_away","p_draw","p_home"]].to_numpy())))
    pd.DataFrame(metric_rows).to_csv(out/"metrics_all_and_seasons.csv",index=False)
    pd.DataFrame(league_rows).to_csv(out/"metrics_by_league.csv",index=False)
    paired = {}
    for kind in MODELS:
        sub = all_predictions[all_predictions.model==kind]
        for config in ["selected_logloss", "selected_accuracy"]:
            paired[kind+"__"+config] = stratified_ci(sub,"all_66",config)
    (out/"paired_uncertainty.json").write_text(json.dumps(paired,indent=2),encoding="utf-8")
    frozen = {s: json.loads((out/s/"selection_before_test.json").read_text(encoding="utf-8")) for s in completed_seasons}
    (out/"selections_all.json").write_text(json.dumps(frozen,indent=2),encoding="utf-8")
    return all_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root",type=Path,default=HERE.parents[1])
    parser.add_argument("--output-dir",type=Path,default=HERE/"results")
    args = parser.parse_args()
    root, out = args.project_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    cache = out/"validation_cache"; cache.mkdir(exist_ok=True)
    start = time.time()
    with threadpool_limits(limits=1):
        df, cols = fs.build_data(root,out)
        digest = hashlib.sha256((out/"feature_matrix.csv").read_bytes()).hexdigest()
        source_hashes = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root/"league/data/english_raw").glob("E[0-3]*.csv"))}
        protocol = dict(outer_seasons=sorted(df.season_id.unique())[1:], models=MODELS,
            primary_selection_metric="log_loss", secondary_selection_metric="accuracy",
            feature_policy="same selected columns for all four divisions within each outer fold",
            selection_grid="same 63 family subsets plus ranking/window/correlation sets as prior experiment",
            C_grid=dict(lr=[1.],rf=[1.],lr_scaled=[.1,1.,10.]),
            short_history="first fold chronological approximately 60/20/20 of matches, keeping whole dates; second fold rank on latter approximately 40% of initial season and validate on next season",
            longer_history="one ranking season followed by up to two validation seasons, all preceding outer test",
            retrospective=True, limitation="Historical seasons and the 2025-26 experiment have already been observed; temporal exclusion does not imply prospective untouched data.",
            seed=42, feature_matrix_sha256=digest, source_sha256=source_hashes,
            code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),HERE/"run_feature_search.py"]},
            completed=False)
        (out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
        done = []
        for season in protocol["outer_seasons"]:
            fold = out/season; fold.mkdir(exist_ok=True)
            if (fold/"completed.json").exists():
                previous = json.loads((fold/"completed.json").read_text())
                assert previous["feature_matrix_sha256"] == digest
                assert previous["code_sha256"] == protocol["code_sha256"], "Code changed: use a new output directory"
                done.append(season); print("RESUME",season,flush=True); continue
            past, test = df[df.season_id<season],df[df.season_id==season]
            assert past.date.max() < test.date.min()
            view, rank, valid = inner_protocol(past)
            fs.RANK_SEASON, fs.VALID_SEASONS = rank, valid
            split = dict(outer_test_season=season, outer_train_n=len(past), outer_test_n=len(test),
                outer_test_start=str(test.date.min().date()), **split_description(view,rank,valid))
            (fold/"splits.json").write_text(json.dumps(split,indent=2),encoding="utf-8")
            print("OUTER",season,json.dumps(split),flush=True)
            selections = {}
            for kind in MODELS:
                model_out = fold/kind; model_out.mkdir(exist_ok=True)
                if (model_out/"selection.json").exists():
                    candidates = json.loads((model_out/f"candidate_features_{kind}.json").read_text())
                    winners = json.loads((model_out/"selection.json").read_text())
                    print("RESUME GRID",season,kind,flush=True)
                else:
                    fs.fit_predict = ORIGINAL_FIT
                    ranking = fs.importance(view,cols,kind,model_out)
                    fs.fit_predict = cached_fit(cache,digest)
                    candidates,winners = fs.select_grid(view,cols,ranking,kind,model_out)
                    (model_out/"selection.json").write_text(json.dumps(winners,indent=2),encoding="utf-8")
                selections[kind] = {objective:dict(**winner,features=candidates[winner["config"]]) for objective,winner in winners.items()}
            # All feature lists/hyperparameters are frozen before evaluating outer targets.
            (fold/"selection_before_test.json").write_text(json.dumps(selections,indent=2),encoding="utf-8")
            predictions = []
            for kind in MODELS:
                configs = {"all_66":dict(features=cols,C=1.),
                    **{"selected_"+k:v for k,v in selections[kind].items()}}
                identical = {}
                for config, spec in configs.items():
                    key = (tuple(spec["features"]),float(spec["C"]))
                    if key not in identical:
                        _, _, p = ORIGINAL_FIT(df,spec["features"],season,kind,float(spec["C"]))
                        identical[key] = p
                    p = identical[key]
                    prediction = test[["date","season_id","division","home_team","away_team","target_result"]].copy()
                    prediction["row_id"] = test.index
                    prediction["model"],prediction["config"] = kind,config
                    prediction[["p_away","p_draw","p_home"]] = p
                    prediction["predicted"] = p.argmax(1)
                    predictions.append(prediction)
                    print("OUTER METRIC",season,kind,config,len(spec["features"]),fs.metrics(test.target_result.to_numpy(int),p),flush=True)
            odds = test[["odds_away_close","odds_draw_close","odds_home_close"]].to_numpy(float)
            valid_odds = np.isfinite(odds).all(1)&(odds>1).all(1)
            market = test.loc[valid_odds,["date","season_id","division","home_team","away_team","target_result"]].copy()
            market["row_id"] = market.index
            market["model"],market["config"] = "bet365_closing","market"
            p = 1/odds[valid_odds]; p /= p.sum(1,keepdims=True)
            market[["p_away","p_draw","p_home"]] = p; market["predicted"] = p.argmax(1)
            predictions.append(market)
            pd.concat(predictions).to_csv(fold/"outer_predictions.csv",index=False)
            (fold/"completed.json").write_text(json.dumps(dict(feature_matrix_sha256=digest,code_sha256=protocol["code_sha256"])),encoding="utf-8")
            done.append(season)
            save_summaries(out,done)
            print("FOLD COMPLETE",season,"elapsed",round(time.time()-start,1),flush=True)
        save_summaries(out,done)
        protocol.update(completed=True,elapsed_seconds=time.time()-start)
        (out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
        print("COMPLETE",round(time.time()-start,1),flush=True)


if __name__=="__main__": main()
