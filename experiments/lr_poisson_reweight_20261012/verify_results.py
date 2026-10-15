"""Audit saved fixed-66 walk-forward results against inputs and recomputed metrics."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-zip', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output_dir.resolve()
    meta = json.loads((out / 'run_metadata.json').read_text(encoding='utf-8'))
    assert meta['complete'] and meta['features'] == 66
    assert len(set(meta['feature_names'])) == 66
    assert len(meta['inputs']) == 28
    for item in meta['inputs']:
        assert hashlib.sha256((root / item['path']).read_bytes()).hexdigest() == item['sha256']
    features = pd.read_csv(out / 'feature_matrix.csv', parse_dates=['date'])
    pred = pd.read_csv(out / 'predictions_by_match.csv.gz', parse_dates=['date'], float_precision='round_trip')
    assert len(features) == 13988 and len(pred) == 12216
    keys = ['date', 'division', 'home_team', 'away_team']
    assert not features.duplicated(keys).any() and not pred.duplicated(keys).any()
    expected = features.loc[features.season_id != '2019-20', keys + ['season_id', 'target_result']]
    matched = pred.merge(expected, on=keys, how='outer', validate='one_to_one', indicator=True)
    assert matched['_merge'].eq('both').all()
    assert matched.fold.eq(matched.season_id).all()
    assert matched.actual_hda.eq(matched.target_result).all()
    raw_labels = np.where(pred.actual_home_goals > pred.actual_away_goals, 2,
                          np.where(pred.actual_home_goals == pred.actual_away_goals, 1, 0))
    np.testing.assert_array_equal(raw_labels, pred.actual_hda)
    folds = []
    for season in sorted(pred.fold.unique()):
        train = features.loc[features.season_id < season]
        test = features.loc[features.season_id == season]
        assert train.date.max() < test.date.min()
        assert len(test) == 2036 and pred.fold.eq(season).sum() == 2036
        folds.append(dict(test_season=season, train_seasons=','.join(sorted(train.season_id.unique())),
                          n_train=len(train), n_test=len(test), train_last_date=str(train.date.max().date()),
                          test_first_date=str(test.date.min().date()), test_last_date=str(test.date.max().date()),
                          features=66, model_parameters_frozen_within_season=True))
    pd.DataFrame(folds).to_csv(out / 'walk_forward_folds.csv', index=False)
    probs = {}
    hda = pd.read_csv(out / 'hda_metrics.csv')
    for name in ['poisson', 'lr', 'fusion']:
        p = pred[[f'{name}_p_{c}' for c in ['away', 'draw', 'home']]].to_numpy()
        assert np.isfinite(p).all() and (p >= 0).all()
        np.testing.assert_allclose(p.sum(axis=1), 1., rtol=0, atol=3e-12)
        probs[name] = p
        for fold in [*sorted(pred.fold.unique()), 'ALL']:
            mask = np.ones(len(pred), dtype=bool) if fold == 'ALL' else pred.fold.eq(fold).to_numpy()
            p_sub, y = p[mask], pred.actual_hda.to_numpy(int)[mask]
            row = hda.loc[hda.model.eq(name) & hda.fold.eq(fold)].iloc[0]
            assert int(row.n_matches) == len(y)
            np.testing.assert_allclose(row.accuracy, (p_sub.argmax(axis=1) == y).mean(), atol=1e-14)
            np.testing.assert_allclose(row.hda_log_loss, -np.log(p_sub[np.arange(len(y)), y]).mean(), atol=1e-13)
    np.testing.assert_allclose(probs['fusion'], probs['lr'], rtol=0, atol=3e-12)
    y = pred.actual_hda.to_numpy(int)
    nll_delta = pred.fusion_true_score_nll - pred.poisson_true_score_nll
    hda_delta = np.log(probs['poisson'][np.arange(len(y)), y]) - np.log(probs['lr'][np.arange(len(y)), y])
    np.testing.assert_allclose(nll_delta, hda_delta, rtol=0, atol=5e-10)
    goals = pd.read_csv(out / 'score_metrics.csv')
    for name in ['poisson', 'fusion']:
        for fold in [*sorted(pred.fold.unique()), 'ALL']:
            sample = pred if fold == 'ALL' else pred.loc[pred.fold.eq(fold)]
            row = goals.loc[goals.model.eq(name) & goals.fold.eq(fold)].iloc[0]
            for suffix, metric in [('true_score_nll', 'true_score_nll'),
                                   ('exact_score_hit', 'modal_exact_score_accuracy'),
                                   ('home_abs_error', 'home_goal_mae_expected'),
                                   ('away_abs_error', 'away_goal_mae_expected'),
                                   ('total_abs_error', 'total_goal_mae_expected')]:
                np.testing.assert_allclose(row[metric], sample[f'{name}_{suffix}'].mean(), atol=1e-13)
    assert pred.poisson_tail_omitted.max() <= 1e-12
    code_paths = [Path(__file__).parent / n for n in ['run_experiment.py', 'fusion_core.py', 'test_fusion.py']]
    code_paths += [root / 'experiments/feature_search_20261010/run_feature_search.py',
                   root / 'scripts/tracka_poisson_goals.py', root / 'scripts/atta_mills_tracka_english4.py',
                   root / 'league/src/data_loader.py', Path(__file__)]
    archive_unchanged = None
    if args.source_zip:
        with zipfile.ZipFile(args.source_zip) as archive:
            for name in ['run_experiment.py', 'fusion_core.py', 'test_fusion.py']:
                source = archive.read('experiments/lr_poisson_reweight_20261012/' + name)
                assert source == (Path(__file__).parent / name).read_bytes()
        archive_unchanged = True
    audit = dict(passed=True, matches=len(pred), features=66, test_seasons=6,
                 raw_input_hashes_verified=28, original_archive_computation_code_unchanged=archive_unchanged,
                 model='fixed66, unscaled LR C=1; adaptive Poisson score grid',
                 max_marginal_error=float(np.abs(probs['lr'] - probs['fusion']).max()),
                 max_nll_identity_error=float(np.abs(nll_delta - hda_delta).max()),
                 folds=folds, code_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in code_paths})
    (out / 'verification_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k not in ['folds', 'code_hashes']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
