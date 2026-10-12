"""Chronological, reproducible feature ablation for the existing English4 Track A.

The source project is read-only. All outputs go to the explicit output directory.
Feature ranking: 2022-23, models trained on preceding seasons.
Feature/hyperparameter selection: rolling 2023-24 and 2024-25 validation.
Final evaluation: 2025-26, never used for ranking or selection by this script.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

RANK_SEASON = "2022-23"
VALID_SEASONS = ["2023-24", "2024-25"]
TEST_SEASON = "2025-26"
SEED = 42


def metrics(y, p):
    assert p.shape == (len(y), 3)
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1.0)
    pred = p.argmax(axis=1)
    onehot = np.eye(3)[y]
    return {
        "n_matches": len(y), "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0),
        "draw_f1": f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)[1],
        "log_loss": log_loss(y, p, labels=[0, 1, 2]),
        # Existing project convention: average of three one-vs-rest Brier scores.
        "brier": np.mean((onehot - p) ** 2),
    }


def make_model(kind, c=1.0, seed=SEED):
    steps = [("imputer", SimpleImputer(strategy="mean", keep_empty_features=True))]
    if kind == "lr_scaled":
        steps.append(("scaler", StandardScaler()))
    if kind.startswith("lr"):
        model = LogisticRegression(C=c, max_iter=4000, solver="lbfgs", random_state=seed)
    else:
        model = RandomForestClassifier(n_estimators=250, max_depth=10,
                    min_samples_leaf=8, n_jobs=4, random_state=seed)
    return Pipeline([*steps, ("model", model)])


def fit_predict(df, cols, season, kind, c=1.0, seed=SEED):
    tr = df[df.season_id < season]
    te = df[df.season_id == season]
    assert tr.date.max() < te.date.min(), f"Chronology overlaps in {season}"
    model = make_model(kind, c, seed)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(tr[cols].to_numpy(float), tr.target_result.to_numpy(int))
    if any(isinstance(w.message, ConvergenceWarning) for w in captured):
        raise RuntimeError(f"Nonconvergence: {kind}, {season}, {len(cols)} features, C={c}")
    assert np.array_equal(model.classes_, [0, 1, 2])
    p = model.predict_proba(te[cols].to_numpy(float))
    return model, te, p


def build_data(root, out):
    sys.path.insert(0, str(root))
    from scripts.atta_mills_tracka_english4 import (
        load_english4_data, _empty_stats, _team_features,
    )
    raw = load_english4_data()
    old_seasons = raw.season_id.copy()
    source = raw.source_file.str.extract(r"([0-9]{2})([0-9]{2})\.csv$")
    assert source.notna().all().all()
    raw["season_id"] = "20" + source[0] + "-" + source[1]
    changed = raw[old_seasons != raw.season_id].copy()
    changed["old_season_id"] = old_seasons[changed.index]
    changed.to_csv(out / "season_assignment_fixes.csv", index=False)
    raw = raw.sort_values(["date", "division", "home_team", "away_team"], kind="stable").reset_index(drop=True)
    assert raw.date.notna().all() and raw.target_result.notna().all()
    assert not raw.duplicated(["date", "home_team", "away_team"]).any()
    # Without reliable kickoff times, exclude all same-day results from features.
    appearances = pd.concat([raw[["date", "home_team"]].rename(columns={"home_team": "team"}),
                             raw[["date", "away_team"]].rename(columns={"away_team": "team"})])
    assert not appearances.duplicated(["date", "team"]).any()
    histories = defaultdict(_empty_stats)
    pairs = defaultdict(list)
    goals_total, goals_count = 0.0, 0
    rows = []
    for day, matches in raw.groupby("date", sort=False):
        avg = goals_total / goals_count if goals_count else np.nan
        for m in matches.itertuples(index=False):
            home, away = m.home_team, m.away_team
            row = {}
            for window in [5, 10]:
                row.update(_team_features(histories[home], "home", window, avg))
                row.update(_team_features(histories[away], "away", window, avg))
            previous = pairs[tuple(sorted([home, away]))][-5:]
            row.update({
                "h2h_home_wins": sum(a["winner"] == home for a in previous),
                "h2h_away_wins": sum(a["winner"] == away for a in previous),
                "h2h_draws": sum(a["winner"] == "D" for a in previous),
                "h2h_home_avg_goals": float(np.mean([a["hg"] if a["home"] == home else a["ag"] for a in previous])) if previous else 0.0,
                "h2h_away_avg_goals": float(np.mean([a["ag"] if a["away"] == away else a["hg"] for a in previous])) if previous else 0.0,
                "league_level": float(m.league_level),
            })
            rows.append(row)
        for m in matches.itertuples(index=False):
            home, away = m.home_team, m.away_team
            hg, ag = int(m.home_goals_full), int(m.away_goals_full)
            for team, gf, ga, shots, sot in [
                (home, hg, ag, m.home_shots, m.home_shots_target),
                (away, ag, hg, m.away_shots, m.away_shots_target),
            ]:
                diff = gf - ga
                values = dict(gf=gf, ga=ga, points=3 if diff > 0 else 1 if diff == 0 else 0,
                    result=1 if diff > 0 else 0 if diff == 0 else -1, goal_diff=diff,
                    win_margin=max(diff, 0), loss_margin=max(-diff, 0), shots=shots, sot=sot)
                for key, value in values.items(): histories[team][key].append(float(value))
            pairs[tuple(sorted([home, away]))].append(dict(home=home, away=away, hg=hg, ag=ag,
                winner=home if hg > ag else away if ag > hg else "D"))
            goals_total += hg + ag
            goals_count += 2
    features = pd.DataFrame(rows)
    cols = features.columns.tolist()
    assert len(cols) == len(set(cols)) == 66
    df = pd.concat([raw.drop(columns=["league_level"]), features], axis=1)
    assert df.columns.is_unique
    df[["date", "season_id", "division", "home_team", "away_team", "target_result", *cols]].to_csv(out / "feature_matrix.csv", index=False)
    audit = dict(matches=len(df), features=len(cols), original_effective_features=67,
        season_fixes=len(changed), same_day_results_excluded=True,
        seasons=df.groupby("season_id").size().to_dict(),
        date_min=str(df.date.min()), date_max=str(df.date.max()))
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("DATA", json.dumps(audit), flush=True)
    return df, cols


def family(col):
    if col == "league_level": return "league"
    if col.startswith("h2h_"): return "h2h"
    if any(x in col for x in ["avg_shots", "avg_sot", "shot_accuracy", "conversion_rate"]): return "shooting"
    if "margin" in col: return "margins"
    if any(x in col for x in ["team_state", "win_rate", "draw_rate", "loss_rate"]): return "form"
    return "goals_strength"


def shuffle_rows(x, indices, divisions, rng):
    result = x.copy()
    # Preserve league contexts when shuffling performance stats; league itself is global.
    blocks = [np.arange(len(x))] if len(indices) == 1 and indices[0] == x.shape[1] - 1 else [
        np.flatnonzero(divisions == d) for d in np.unique(divisions)]
    for block in blocks:
        shuffled = rng.permutation(block)
        result[np.ix_(block, indices)] = x[np.ix_(shuffled, indices)]
    return result


def importance(df, cols, kind, out, repeats=5):
    model, rank, p = fit_predict(df, cols, RANK_SEASON, kind)
    y = rank.target_result.to_numpy(int)
    x = rank[cols].to_numpy(float)
    base = metrics(y, p)
    rng = np.random.default_rng(SEED)
    rows = []
    for i, col in enumerate(cols):
        losses, accuracies = [], []
        for _ in range(repeats):
            perturbed = shuffle_rows(x, [i], rank.division.to_numpy(), rng)
            ps = model.predict_proba(perturbed)
            losses.append(log_loss(y, ps, labels=[0, 1, 2]) - base["log_loss"])
            accuracies.append(base["accuracy"] - accuracy_score(y, ps.argmax(axis=1)))
        rows.append(dict(feature=col, family=family(col),
            permutation_logloss_mean=np.mean(losses), permutation_logloss_std=np.std(losses, ddof=1),
            permutation_accuracy_mean=np.mean(accuracies), permutation_accuracy_std=np.std(accuracies, ddof=1)))
        if (i + 1) % 16 == 0: print(f"IMPORTANCE {kind} {i+1}/{len(cols)}", flush=True)
    result = pd.DataFrame(rows).sort_values("permutation_logloss_mean", ascending=False, kind="stable")
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    final = model.named_steps["model"]
    if kind == "rf":
        own = dict(zip(cols, final.feature_importances_))
        result["rf_impurity_importance"] = result.feature.map(own)
    else:
        # Center across classes; express coefficients per one training SD.
        coefficients = final.coef_ - final.coef_.mean(axis=0, keepdims=True)
        if kind == "lr":
            train_x = model.named_steps["imputer"].transform(df.loc[df.season_id < RANK_SEASON, cols].to_numpy(float))
            coefficients *= train_x.std(axis=0)
        for i, label in enumerate(["away", "draw", "home"]):
            result[f"centered_coef_per_sd_{label}"] = result.feature.map(dict(zip(cols, coefficients[i])))
        result["coefficient_mean_abs"] = result.feature.map(dict(zip(cols, np.abs(coefficients).mean(axis=0))))
    result.to_csv(out / f"importance_{kind}.csv", index=False)
    group_rows = []
    for group in sorted({family(c) for c in cols}):
        ids = [i for i,c in enumerate(cols) if family(c) == group]
        losses, accs = [], []
        for _ in range(repeats):
            ps = model.predict_proba(shuffle_rows(x, ids, rank.division.to_numpy(), rng))
            losses.append(log_loss(y, ps, labels=[0,1,2]) - base["log_loss"])
            accs.append(base["accuracy"] - accuracy_score(y, ps.argmax(axis=1)))
        group_rows.append(dict(family=group, n_features=len(ids),
            permutation_logloss_mean=np.mean(losses), permutation_logloss_std=np.std(losses,ddof=1),
            permutation_accuracy_mean=np.mean(accs)))
    pd.DataFrame(group_rows).sort_values("permutation_logloss_mean", ascending=False).to_csv(out / f"group_importance_{kind}.csv", index=False)
    print("RANKING", kind, result.head(8)[["feature", "permutation_logloss_mean"]].to_dict("records"), flush=True)
    return result


def candidate_sets(df, cols, ranking):
    groups = sorted({family(c) for c in cols})
    candidates = {"all_66": cols}
    for size in range(1, len(groups)+1):
        for subset in combinations(groups, size):
            if size == len(groups): continue
            name = "groups__" + "__".join(subset)
            candidates[name] = [c for c in cols if family(c) in subset]
    ranked = ranking.feature.tolist()
    for k in [8, 16, 24, 32, 48, 60]:
        candidates[f"top_{k}"] = [c for c in cols if c in ranked[:k]]
    for k in [5, 10, 20]:
        candidates[f"drop_bottom_{k}"] = [c for c in cols if c in ranked[:-k]]
    positive = ranking[ranking.permutation_logloss_mean > 0].feature.tolist()
    candidates["positive_permutation"] = [c for c in cols if c in positive]
    for w in [5, 10]:
        candidates[f"window_{w}_only"] = [c for c in cols if c.endswith(f"_{w}") or family(c) in ["h2h", "league"]]
    # Correlation pruning is based only on data preceding the importance season.
    corr = df.loc[df.season_id < RANK_SEASON, cols].corr().abs()
    keep = []
    for c in ranked:
        if all(not np.isfinite(corr.loc[c, kept]) or corr.loc[c, kept] < 0.95 for kept in keep): keep.append(c)
    candidates["correlation_pruned_095"] = [c for c in cols if c in keep]
    # Same feature set may appear under multiple informative names: evaluate it once.
    unique = {}
    aliases = {}
    for name, selected in candidates.items():
        key = tuple(selected)
        if selected and key not in aliases:
            aliases[key] = name
            unique[name] = selected
    return unique


def validation(df, selected, kind, c):
    ys, ps, folds = [], [], []
    for season in VALID_SEASONS:
        _, te, p = fit_predict(df, selected, season, kind, c)
        y = te.target_result.to_numpy(int)
        folds.append(dict(season=season, **metrics(y, p)))
        ys.append(y); ps.append(p)
    return metrics(np.concatenate(ys), np.vstack(ps)), folds


def select_grid(df, cols, ranking, kind, out):
    candidates = candidate_sets(df, cols, ranking)
    (out / f"candidate_features_{kind}.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    rows, fold_rows = [], []
    cs = [1.0] if kind in ["lr", "rf"] else [0.1, 1.0, 10.0]
    for number, (name, selected) in enumerate(candidates.items(), 1):
        for c in cs:
            overall, folds = validation(df, selected, kind, c)
            row = dict(model=kind, config=name, C=c, n_features=len(selected), **overall)
            row["min_season_accuracy"] = min(f["accuracy"] for f in folds)
            row["max_season_log_loss"] = max(f["log_loss"] for f in folds)
            rows.append(row)
            fold_rows.extend(dict(model=kind, config=name, C=c, n_features=len(selected), **f) for f in folds)
        pd.DataFrame(rows).to_csv(out / f"validation_grid_{kind}.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(out / f"validation_folds_{kind}.csv", index=False)
        if number % 8 == 0 or number == len(candidates):
            best = min(rows, key=lambda r:r["log_loss"])
            print(f"GRID {kind} {number}/{len(candidates)}, best={best['config']} C={best['C']} LL={best['log_loss']:.6f}", flush=True)
    table = pd.DataFrame(rows)
    # Predeclared primary target is probability quality; accuracy winner is also frozen.
    winners = {
        "logloss": table.sort_values(["log_loss", "n_features", "config", "C"]).iloc[0].to_dict(),
        "accuracy": table.sort_values(["accuracy", "log_loss", "n_features"], ascending=[False, True, True]).iloc[0].to_dict(),
    }
    return candidates, winners


def paired_ci(y, base, selected, dates, repeats=2000):
    # Resample whole match dates, keeping same-day outcomes together.
    loss_b = -np.log(np.clip(base[np.arange(len(y)), y], 1e-15, 1))
    loss_s = -np.log(np.clip(selected[np.arange(len(y)), y], 1e-15, 1))
    delta = pd.DataFrame(dict(date=np.asarray(dates),
        accuracy=(selected.argmax(axis=1) == y).astype(float)-(base.argmax(axis=1)==y).astype(float),
        logloss=loss_s-loss_b)).groupby("date").agg(["sum", "count"])
    rng = np.random.default_rng(123)
    output = {}
    for field in ["accuracy", "logloss"]:
        sums, counts = delta[(field,"sum")].to_numpy(), delta[(field,"count")].to_numpy()
        samples = rng.integers(0,len(sums),size=(repeats,len(sums)))
        values = sums[samples].sum(axis=1)/counts[samples].sum(axis=1)
        output[field] = dict(delta=float(sums.sum()/counts.sum()),
            ci95_low=float(np.quantile(values,0.025)), ci95_high=float(np.quantile(values,0.975)))
    return output


def evaluate_frozen(df, cols, selections, out):
    rows, predictions, manifest, comparisons = [], [], {}, {}
    for kind, (candidates, winners) in selections.items():
        evaluation = {"all_66": (cols, 1.0)}
        for objective, winner in winners.items():
            evaluation[f"selected_{objective}"] = (candidates[winner["config"]], float(winner["C"]))
            manifest[f"{kind}__{objective}"] = dict(**winner, features=candidates[winner["config"]])
        if kind in ["lr", "rf"]:
            # Predeclared controlled ablations: all feature families minus one.
            for group in sorted({family(c) for c in cols}):
                evaluation[f"drop_{group}"] = ([c for c in cols if family(c)!=group], 1.0)
        base_p = None
        for name, (features, c) in evaluation.items():
            model, te, p = fit_predict(df, features, TEST_SEASON, kind, c)
            y = te.target_result.to_numpy(int)
            row = dict(model=kind, config=name, C=c, n_features=len(features), season=TEST_SEASON, **metrics(y,p))
            rows.append(row)
            if name == "all_66": base_p = p
            if name.startswith("selected"):
                comparisons[f"{kind}__{name}"] = paired_ci(y,base_p,p,te.date)
            prediction = te[["date","season_id","division","home_team","away_team","target_result"]].copy()
            prediction["model"], prediction["config"] = kind, name
            prediction[["p_away","p_draw","p_home"]] = p
            prediction["predicted"] = p.argmax(axis=1)
            predictions.append(prediction)
            print("FINAL", row, flush=True)
    te = df[df.season_id == TEST_SEASON]
    odds = te[["odds_away_close","odds_draw_close","odds_home_close"]].to_numpy(float)
    valid = np.isfinite(odds).all(axis=1) & (odds>1).all(axis=1)
    p_market = 1/odds[valid]
    p_market /= p_market.sum(axis=1,keepdims=True)
    rows.append(dict(model="bet365_closing",config="market",C=np.nan,n_features=0,season=TEST_SEASON,
        **metrics(te.target_result.to_numpy(int)[valid],p_market)))
    pd.DataFrame(rows).to_csv(out/"final_test_metrics.csv",index=False)
    pd.concat(predictions).to_csv(out/"final_test_predictions.csv",index=False)
    (out/"frozen_selections.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (out/"paired_uncertainty.json").write_text(json.dumps(comparisons,indent=2),encoding="utf-8")
    # No final-test result is used to change a feature list or model parameter.
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent/"results")
    parser.add_argument("--models",nargs="+",default=["lr","rf","lr_scaled"])
    args=parser.parse_args()
    args.project_root=args.project_root.resolve()
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    with threadpool_limits(limits=1):
        df,cols=build_data(args.project_root,out)
        pd.DataFrame(dict(feature=cols,family=[family(c) for c in cols])).to_csv(out/"feature_groups.csv",index=False)
        metadata=dict(seed=SEED,ranking_season=RANK_SEASON,validation_seasons=VALID_SEASONS,
            final_test_season=TEST_SEASON,primary_selection_metric="log_loss",
            brier_definition="mean squared error over matches and 3 classes",
            permutation_repeats=5,permutation_blocks="within division except league_level",
            retrospective_test=True, note="2025-26 was previously inspected by the project; reserved only for this new search.",
            source_sha256={str(p.relative_to(args.project_root)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((args.project_root/"league/data/english_raw").glob("E[0-3]*.csv"))})
        (out/"protocol.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        rankings={}
        # Rank before any candidate validation and before the final test is evaluated.
        for kind in args.models:
            rankings[kind]=importance(df,cols,kind,out)
        selections={}
        for kind in args.models:
            selections[kind]=select_grid(df,cols,rankings[kind],kind,out)
        # Freeze the selections on disk before accessing test targets for evaluation.
        frozen={kind:winners for kind,(_,winners) in selections.items()}
        (out/"selection_before_test.json").write_text(json.dumps(frozen,indent=2),encoding="utf-8")
        evaluate_frozen(df,cols,selections,out)
        metadata["elapsed_seconds"]=time.time()-start
        metadata["completed"]=True
        (out/"protocol.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        print(f"COMPLETE in {time.time()-start:.1f}s",flush=True)


if __name__ == "__main__": main()
