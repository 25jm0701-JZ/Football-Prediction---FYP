"""Freshly selected fixed-feature LR versus original LR and common Poisson model."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'league'))
sys.path.insert(0, str(ROOT / 'experiments/lr_poisson_reweight_20261012'))
from fusion_core import aggregate_hda, adaptive_poisson_grid, reweight_scores, score_statistics
from experiments.feature_search_20261010 import run_feature_search as fs
from scripts.tracka_poisson_goals import LoukasPoissonModel

MODELS = ['poisson', 'lr66', 'lr_selected', 'fusion66', 'fusion_selected']
SCORE_MODELS = ['poisson', 'fusion66', 'fusion_selected']
PROB_COLS = ['away', 'draw', 'home']
KEYS = ['date', 'fold', 'division', 'home_team', 'away_team']
POINT_FIELDS = ['accuracy', 'log_loss', 'brier', 'exact_score_accuracy', 'score_nll',
                'home_goal_mae', 'away_goal_mae', 'total_goal_mae',
                'home_within_1', 'away_within_1', 'total_within_1']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def season_from_source(name):
    match = re.fullmatch(r'E([0-3])(\d{2})(\d{2})\.csv', name)
    if not match or int(match[3]) != int(match[2]) + 1:
        raise ValueError(f'Invalid source season: {name}')
    return f'20{match[2]}-{match[3]}'


def raw_season_audit(df, out):
    sources, registry = [], []
    paths = sorted((ROOT / 'league/data/english_raw').glob('E[0-3]*.csv'))
    assert len(paths) == 28
    for path in paths:
        raw = pd.read_csv(path)
        raw = raw.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'])
        dates = pd.to_datetime(raw.Date, dayfirst=True, errors='raise')
        season = season_from_source(path.name)
        assert raw.Div.eq(path.name[:2]).all()
        goals_label = np.where(raw.FTHG > raw.FTAG, 2, np.where(raw.FTHG == raw.FTAG, 1, 0))
        np.testing.assert_array_equal(goals_label, raw.FTR.map({'A': 0, 'D': 1, 'H': 2}))
        frame = pd.DataFrame(dict(date=dates, division=raw.Div, home_team=raw.HomeTeam,
                                 away_team=raw.AwayTeam, source_file=path.name, season_id=season,
                                 home_goals_full=raw.FTHG, away_goals_full=raw.FTAG,
                                 target_result=goals_label))
        registry.append(frame)
        sources.append(dict(source_file=path.name, assigned_season=season, matches=len(raw),
                            date_min=str(dates.min().date()), date_max=str(dates.max().date()),
                            july_matches=int(dates.dt.month.eq(7).sum()), sha256=sha(path)))
    registry = pd.concat(registry, ignore_index=True)
    key = ['date', 'division', 'home_team', 'away_team']
    merged = df.merge(registry, on=key, suffixes=('_built', '_raw'), how='outer',
                      indicator=True, validate='one_to_one')
    assert len(merged) == 13988 and merged['_merge'].eq('both').all()
    for field in ['season_id', 'source_file', 'target_result', 'home_goals_full', 'away_goals_full']:
        assert merged[f'{field}_built'].eq(merged[f'{field}_raw']).all(), field
    month_seasons = np.where(registry.date.dt.month >= 8,
                            registry.date.dt.year.astype(str) + '-' + (registry.date.dt.year + 1).astype(str).str[-2:],
                            (registry.date.dt.year - 1).astype(str) + '-' + registry.date.dt.year.astype(str).str[-2:])
    registry['old_month_assigned_season'] = month_seasons
    differences = registry.loc[registry.season_id.ne(registry.old_month_assigned_season)]
    assert len(differences) == 35 and differences.season_id.eq('2022-23').all()
    registry.to_csv(out / 'source_match_registry.csv.gz', index=False, compression='gzip')
    differences.to_csv(out / 'season_assignment_corrections.csv', index=False)
    pd.DataFrame(sources).to_csv(out / 'source_file_audit.csv', index=False)
    return sources


def frozen_selection(development, cols):
    record = json.loads((development / 'frozen_selections.json').read_text(encoding='utf-8'))['lr__logloss']
    protocol = json.loads((development / 'protocol.json').read_text(encoding='utf-8'))
    assert protocol['completed'] and protocol['primary_selection_metric'] == 'log_loss'
    grid = pd.read_csv(development / 'validation_grid_lr.csv')
    winner = grid.sort_values(['log_loss', 'n_features', 'config', 'C']).iloc[0]
    assert record['config'] == winner['config'] and record['C'] == winner.C == 1.0
    candidates = json.loads((development / 'candidate_features_lr.json').read_text(encoding='utf-8'))
    selected = candidates[record['config']]
    assert selected == record['features'] and len(selected) == record['n_features']
    assert selected and len(set(selected)) == len(selected) and set(selected) <= set(cols)
    return selected, record


def fit_lr(train, test, cols):
    if train.date.max() >= test.date.min() or not (train.season_id < test.season_id.iloc[0]).all():
        raise AssertionError('Training includes test-season/future matches')
    model = fs.make_model('lr', c=1.0)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(train[cols].to_numpy(float), train.target_result.to_numpy(int))
    if any(isinstance(w.message, ConvergenceWarning) for w in caught):
        raise RuntimeError('LR nonconvergence')
    fit_time = time.perf_counter() - start
    np.testing.assert_array_equal(model.classes_, [0, 1, 2])
    start = time.perf_counter()
    p = model.predict_proba(test[cols].to_numpy(float))
    predict_time = time.perf_counter() - start
    np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-13)
    stats = dict(fit_seconds=fit_time, predict_seconds=predict_time,
                 n_features=len(cols), n_parameters=int(model[-1].coef_.size + model[-1].intercept_.size),
                 iterations=int(model[-1].n_iter_[0]))
    return model, p, stats


def predict_fold(train, test, cols, selected, out, cfg):
    season = test.season_id.iloc[0]
    print(f'FOLD {season}: train={len(train)} test={len(test)} selected={len(selected)}', flush=True)
    probabilities, audits = {}, []
    for name, features in [('lr66', cols), ('lr_selected', selected)]:
        model, prob, stats = fit_lr(train, test, features)
        dest = out / 'models' / f'{season}_{name}.joblib'
        joblib.dump(dict(model=model, features=features, class_order=PROB_COLS), dest, compress=3)
        stats.update(model=name, fold=season, model_bytes=dest.stat().st_size,
                     feature_list_sha256=hashlib.sha256(json.dumps(features).encode()).hexdigest(),
                     imputer_statistics=model['imputer'].statistics_.tolist())
        audits.append(stats)
        probabilities[name] = prob
        np.testing.assert_allclose(model.predict_proba(test[features].to_numpy(float)),
                                   joblib.load(dest)['model'].predict_proba(test[features].to_numpy(float)), atol=0)
    poisson = LoukasPoissonModel(max_goals=10, smoothing=cfg['poisson']['smoothing']).fit(train)
    params = poisson.parameter_frame()
    params['fold'] = season
    params['mu'], params['mu_home'] = poisson.mu, poisson.mu_home
    rows = []
    for idx, match in enumerate(test.itertuples(index=False)):
        lh, la = poisson.expected_goals(match.home_team, match.away_team)
        grid = adaptive_poisson_grid(lh, la, cfg['poisson']['joint_tail_tolerance'])
        base = grid.matrix
        q66, qs = probabilities['lr66'][idx], probabilities['lr_selected'][idx]
        mats = dict(poisson=base, fusion66=reweight_scores(base, q66), fusion_selected=reweight_scores(base, qs))
        probs = dict(lr66=q66, lr_selected=qs, **{k: aggregate_hda(v) for k, v in mats.items()})
        np.testing.assert_allclose(probs['fusion66'], q66, rtol=0, atol=3e-12)
        np.testing.assert_allclose(probs['fusion_selected'], qs, rtol=0, atol=3e-12)
        if probs['poisson'].min() < 1e-12:
            raise ArithmeticError('Near-zero Poisson result-class mass')
        hg, ag, y = int(match.home_goals_full), int(match.away_goals_full), int(match.target_result)
        row = dict(fold=season, date=str(match.date.date()), division=match.division,
                   home_team=match.home_team, away_team=match.away_team, actual_hda=y,
                   actual_home_goals=hg, actual_away_goals=ag,
                   lambda_home=lh, lambda_away=la, grid_max_goals=grid.max_goals,
                   poisson_tail_omitted=grid.omitted_mass, min_poisson_class_mass=float(probs['poisson'].min()))
        for name, p in probs.items():
            row.update({f'{name}_p_{c}': float(p[j]) for j, c in enumerate(PROB_COLS)})
            row[f'{name}_predicted'] = int(p.argmax())
        for name, mat in mats.items():
            target = None if name == 'poisson' else q66 if name == 'fusion66' else qs
            stats = score_statistics(mat, hg, ag, lambda_home=lh, lambda_away=la, target_hda=target)
            row.update({f'{name}_{k}': v for k, v in stats.items()})
        for fusion, lr in [('fusion66', 'lr66'), ('fusion_selected', 'lr_selected')]:
            delta = row[f'{fusion}_true_score_nll'] - row['poisson_true_score_nll']
            np.testing.assert_allclose(delta, np.log(probs['poisson'][y]) - np.log(probs[lr][y]), rtol=0, atol=5e-10)
        rows.append(row)
    fold_audit = dict(fold=season, train_seasons=','.join(sorted(train.season_id.unique())),
                      n_train=len(train), n_test=len(test), train_last_date=str(train.date.max().date()),
                      test_first_date=str(test.date.min().date()), test_last_date=str(test.date.max().date()),
                      selected_features=len(selected), parameters_frozen_within_test_season=True,
                      unknown_test_teams=len((set(test.home_team) | set(test.away_team)) - set(poisson.teams)))
    return pd.DataFrame(rows), audits, params, fold_audit


def per_match_metrics(pred):
    rows = []
    y = pred.actual_hda.to_numpy(int)
    for name in MODELS:
        frame = pred[KEYS + ['actual_hda']].copy()
        frame['model'] = name
        p = pred[[f'{name}_p_{c}' for c in PROB_COLS]].to_numpy()
        frame['predicted'] = p.argmax(axis=1)
        frame['accuracy'] = (p.argmax(axis=1) == y).astype(float)
        frame['log_loss'] = -np.log(p[np.arange(len(y)), y])
        frame['brier'] = ((np.eye(3)[y] - p) ** 2).mean(axis=1)
        if name in SCORE_MODELS:
            fields = dict(exact_score_accuracy='exact_score_hit', score_nll='true_score_nll',
                          home_goal_mae='home_abs_error', away_goal_mae='away_abs_error',
                          total_goal_mae='total_abs_error', home_within_1='home_within_1',
                          away_within_1='away_within_1', total_within_1='total_within_1')
            for metric, suffix in fields.items():
                frame[metric] = pred[f'{name}_{suffix}'].to_numpy()
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def summarize(per_match):
    groups = [('overall', 'ALL', per_match)]
    for field, scope in [('fold', 'season'), ('division', 'league'), ('actual_hda', 'actual_outcome')]:
        groups.extend((scope, str(key), frame) for key, frame in per_match.groupby(field, sort=True))
    groups.extend(('season_league', f'{s}/{d}', frame) for (s, d), frame in per_match.groupby(['fold', 'division'], sort=True))
    rows = []
    for scope, group, sample in groups:
        for name, frame in sample.groupby('model', sort=False):
            row = dict(scope=scope, group=group, model=name, n_matches=len(frame), correct=int(frame.accuracy.sum()))
            row.update({field: float(frame[field].mean()) for field in POINT_FIELDS if field in frame and frame[field].notna().all()})
            row['macro_f1'] = float(f1_score(frame.actual_hda, frame.predicted, labels=[0, 1, 2], average='macro', zero_division=0))
            row['draw_f1'] = float(f1_score(frame.actual_hda, frame.predicted, labels=[0, 1, 2], average=None, zero_division=0)[1])
            rows.append(row)
    return pd.DataFrame(rows)


def paired_intervals(per_match, repeats=4000, seed=123):
    pairs = [('lr66', 'lr_selected'), ('poisson', 'fusion66'),
             ('poisson', 'fusion_selected'), ('fusion66', 'fusion_selected')]
    rows = []
    for base, selected in pairs:
        b = per_match.loc[per_match.model.eq(base)].set_index(KEYS)
        s = per_match.loc[per_match.model.eq(selected)].set_index(KEYS).reindex(b.index)
        assert s.actual_hda.notna().all() and s.actual_hda.eq(b.actual_hda).all()
        fields = [c for c in POINT_FIELDS if c in s and s[c].notna().all() and b[c].notna().all()]
        delta = (s[fields] - b[fields]).reset_index()
        point = delta[fields].mean().to_numpy()
        rng = np.random.default_rng(seed)
        grouped = delta.groupby(['fold', 'date'], sort=True)
        sums, counts = grouped[fields].sum(), grouped.size()
        strata = [(sums.loc[season].to_numpy(), counts.loc[season].to_numpy()) for season in sorted(delta.fold.unique())]
        boot = np.empty((repeats, len(fields)))
        for i in range(repeats):
            total, n = np.zeros(len(fields)), 0
            for block, sizes in strata:
                idx = rng.integers(len(sizes), size=len(sizes))
                total += block[idx].sum(axis=0)
                n += sizes[idx].sum()
            boot[i] = total / n
        low, high = np.quantile(boot, [0.025, 0.975], axis=0)
        ss = delta.groupby('fold')[fields].sum().to_numpy()
        nn = delta.groupby('fold').size().to_numpy()
        idx = rng.integers(len(nn), size=(repeats, len(nn)))
        seasonal = ss[idx].sum(axis=1) / nn[idx].sum(axis=1)[:, None]
        slo, shi = np.quantile(seasonal, [0.025, 0.975], axis=0)
        for j, field in enumerate(fields):
            rows.append(dict(comparison=f'{selected}_minus_{base}', metric=field, delta=float(point[j]),
                             ci95_low=float(low[j]), ci95_high=float(high[j]),
                             season_ci95_low=float(slo[j]), season_ci95_high=float(shi[j]), repeats=repeats))
    return pd.DataFrame(rows)


def redundancy_audit(df, cols, selected, out):
    # Never use outer-test scores for feature diagnostics or choices.
    history = df.loc[df.season_id < fs.RANK_SEASON, cols]
    corr = history.corr().abs()
    corr.to_csv(out / 'development_feature_abs_correlations.csv')
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if np.isfinite(corr.loc[a, b]) and corr.loc[a, b] >= 0.95:
                pairs.append(dict(feature_a=a, feature_b=b, absolute_correlation=float(corr.loc[a, b])))
    pd.DataFrame(pairs, columns=['feature_a', 'feature_b', 'absolute_correlation']).to_csv(out / 'high_correlation_pairs.csv', index=False)
    table = pd.DataFrame(dict(feature=cols, family=[fs.family(c) for c in cols], selected=[c in selected for c in cols]))
    table.to_csv(out / 'feature_retention.csv', index=False)
    return len(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-dir', type=Path, default=HERE / 'development')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    out, development = args.output_dir.resolve(), args.development_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use an empty output directory to preserve previous results')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'models').mkdir()
    cfg = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    start = time.perf_counter()
    code_paths = [HERE / 'run_experiment.py', HERE / 'config.json',
                  ROOT / 'experiments/feature_search_20261010/run_feature_search.py',
                  ROOT / 'experiments/lr_poisson_reweight_20261012/fusion_core.py',
                  ROOT / 'scripts/tracka_poisson_goals.py', ROOT / 'scripts/atta_mills_tracka_english4.py',
                  ROOT / 'league/src/data_loader.py']
    manifest = dict(complete=False, config=cfg, code_hashes={str(p.relative_to(ROOT)): sha(p) for p in code_paths},
                    python=platform.python_version(), libraries={m.__name__: m.__version__ for m in [np, pd, scipy, sklearn, joblib]},
                    command=sys.argv)
    dump(out / 'run_manifest.json', manifest)
    with threadpool_limits(limits=1):
        df, cols = fs.build_data(ROOT, out)
        sources = raw_season_audit(df, out)
        selected, record = frozen_selection(development, cols)
        # The winner is accepted as produced by the new search; no feature count is enforced.
        freeze = dict(record=record, feature_names=selected, dropped_features=[c for c in cols if c not in selected],
                      frozen_before_six_season_evaluation=True, retrospective=True,
                      development_files={p.name: sha(p) for p in development.glob('*') if p.is_file()})
        dump(out / 'frozen_selection.json', freeze)
        print(f"FRESH SELECTION: {record['config']}, {len(selected)}/{len(cols)} features", flush=True)
        high_corr = redundancy_audit(df, cols, selected, out)
        predictions, fit_audits, parameters, folds = [], [], [], []
        for season in cfg['test_seasons']:
            train, test = df.loc[df.season_id < season], df.loc[df.season_id == season]
            assert len(test) == 2036 and train.date.max() < test.date.min()
            pred, audit, params, fold = predict_fold(train, test, cols, selected, out, cfg)
            predictions.append(pred); fit_audits.extend(audit); parameters.append(params); folds.append(fold)
        pred = pd.concat(predictions, ignore_index=True)
        assert len(pred) == 12216 and not pred.duplicated(KEYS).any()
        pred.to_csv(out / 'predictions.csv.gz', index=False, compression='gzip', float_format='%.17g')
        per_match = per_match_metrics(pred)
        per_match.to_csv(out / 'per_match_metrics.csv.gz', index=False, compression='gzip', float_format='%.17g')
        metrics = summarize(per_match)
        metrics.to_csv(out / 'metrics_all_scopes.csv', index=False, float_format='%.17g')
        for scope, frame in metrics.groupby('scope'):
            frame.to_csv(out / f'metrics_{scope}.csv', index=False, float_format='%.17g')
        print('Computing paired date/season cluster intervals...', flush=True)
        intervals = paired_intervals(per_match, cfg['bootstrap']['repeats'], cfg['bootstrap']['seed'])
        intervals.to_csv(out / 'paired_intervals.csv', index=False, float_format='%.17g')
        pd.DataFrame(folds).to_csv(out / 'walk_forward_folds.csv', index=False)
        pd.concat(parameters).to_csv(out / 'poisson_parameters.csv', index=False)
        dump(out / 'fit_audit.json', fit_audits)
        manifest.update(complete=True, inputs=sources, n_source_matches=len(df), n_evaluated=len(pred),
                        n_original_features=len(cols), n_selected_features=len(selected), selected_config=record['config'],
                        high_correlation_pairs_ge_095=high_corr, elapsed_seconds=time.perf_counter() - start,
                        development_protocol_sha256=sha(development / 'protocol.json'))
        dump(out / 'run_manifest.json', manifest)
        print(metrics.loc[metrics.scope.eq('overall')].to_string(index=False), flush=True)
        print(f'COMPLETE: {out}', flush=True)


if __name__ == '__main__':
    main()
