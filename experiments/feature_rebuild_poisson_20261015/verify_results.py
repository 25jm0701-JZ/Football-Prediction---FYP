"""Independent saved-prediction, model-loading, season and metric audit."""
from pathlib import Path
import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson as scipy_poisson
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from run_experiment import HERE, ROOT, KEYS, MODELS, SCORE_MODELS, PROB_COLS, POINT_FIELDS, dump, sha


def audit(out):
    manifest = json.loads((out / 'run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['complete'] and manifest['n_evaluated'] == 12216
    for path, digest in manifest['code_hashes'].items():
        assert sha(ROOT / path) == digest, path
    for item in manifest['inputs']:
        assert sha(ROOT / 'league/data/english_raw' / item['source_file']) == item['sha256']
    pred = pd.read_csv(out / 'predictions.csv.gz', parse_dates=['date'], float_precision='round_trip')
    features = pd.read_csv(out / 'feature_matrix.csv', parse_dates=['date'], float_precision='round_trip')
    registry = pd.read_csv(out / 'source_match_registry.csv.gz', parse_dates=['date'])
    keys = ['date', 'division', 'home_team', 'away_team']
    assert len(pred) == 12216 and not pred.duplicated(KEYS).any()
    reg_test = registry.loc[registry.season_id.ne('2019-20')]
    joined = pred.merge(reg_test, on=keys, how='outer', suffixes=('_prediction', '_raw'), validate='one_to_one', indicator=True)
    assert joined['_merge'].eq('both').all()
    assert joined.fold.eq(joined.season_id).all()
    for a, b in [('actual_home_goals', 'home_goals_full'), ('actual_away_goals', 'away_goals_full'), ('actual_hda', 'target_result')]:
        np.testing.assert_array_equal(joined[a], joined[b])
    labels = pred.actual_hda.to_numpy(int)
    frozen = json.loads((out / 'frozen_selection.json').read_text(encoding='utf-8'))
    model_error, imputer_error = 0., 0.
    for season in sorted(pred.fold.unique()):
        train = features.loc[features.season_id < season]
        test = features.loc[features.season_id.eq(season)]
        assert train.date.max() < test.date.min() and len(test) == 2036
        target = pred.loc[pred.fold.eq(season)]
        for name in ['lr66', 'lr_selected']:
            saved = joblib.load(out / 'models' / f'{season}_{name}.joblib')
            cols = saved['features']
            if name == 'lr_selected':
                assert cols == frozen['feature_names']
            trained_means = train[cols].mean().fillna(0.).to_numpy()
            mean_error = float(np.abs(trained_means - saved['model']['imputer'].statistics_).max())
            imputer_error = max(imputer_error, mean_error)
            np.testing.assert_allclose(trained_means, saved['model']['imputer'].statistics_, atol=1e-12)
            p = saved['model'].predict_proba(test[cols].to_numpy(float))
            expected = target[[f'{name}_p_{c}' for c in PROB_COLS]].to_numpy()
            model_error = max(model_error, float(np.abs(p - expected).max()))
            np.testing.assert_allclose(p, expected, rtol=0, atol=1e-12)
    nll_error, mass_error = 0., 0.
    for match in pred.itertuples(index=False):
        axis = np.arange(match.grid_max_goals + 1)
        base = np.outer(scipy_poisson.pmf(axis, match.lambda_home), scipy_poisson.pmf(axis, match.lambda_away))
        base /= base.sum()
        h, a = np.indices(base.shape)
        masks = [h < a, h == a, h > a]
        pp = np.array([base[m].sum() for m in masks])
        expected_pp = np.array([getattr(match, f'poisson_p_{c}') for c in PROB_COLS])
        np.testing.assert_allclose(pp, expected_pp, rtol=0, atol=2e-14)
        for name in SCORE_MODELS:
            if name == 'poisson':
                mat = base
            else:
                lr = 'lr66' if name == 'fusion66' else 'lr_selected'
                q = np.array([getattr(match, f'{lr}_p_{c}') for c in PROB_COLS])
                mat = base.copy()
                for c, mask in enumerate(masks):
                    mat[mask] = base[mask] * q[c] / pp[c]
                np.testing.assert_allclose([mat[m].sum() for m in masks], q, rtol=0, atol=3e-12)
            mass_error = max(mass_error, abs(float(mat.sum()) - 1.))
            eh, ea = float((mat * h).sum()), float((mat * a).sum())
            np.testing.assert_allclose([eh, ea], [getattr(match, f'{name}_expected_home_goals'), getattr(match, f'{name}_expected_away_goals')], atol=1e-12)
            mh, ma = np.unravel_index(mat.argmax(), mat.shape)
            assert (mh, ma) == (getattr(match, f'{name}_mode_home_goals'), getattr(match, f'{name}_mode_away_goals'))
            if match.actual_home_goals > match.grid_max_goals or match.actual_away_goals > match.grid_max_goals:
                raise AssertionError('Unexpected outside-grid case: audit requires explicit extension')
            nll = -np.log(mat[match.actual_home_goals, match.actual_away_goals])
            nll_error = max(nll_error, abs(float(nll) - getattr(match, f'{name}_true_score_nll')))
            np.testing.assert_allclose(nll, getattr(match, f'{name}_true_score_nll'), atol=2e-12)
    per_match = pd.read_csv(out / 'per_match_metrics.csv.gz', parse_dates=['date'], float_precision='round_trip')
    table = pd.read_csv(out / 'metrics_all_scopes.csv', float_precision='round_trip')
    assert len(per_match) == 5 * len(pred)
    for name in MODELS:
        p = pred[[f'{name}_p_{c}' for c in PROB_COLS]].to_numpy()
        assert (p >= 0).all() and np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(axis=1), 1., rtol=0, atol=3e-12)
        pm = per_match.loc[per_match.model.eq(name)]
        np.testing.assert_array_equal(pm.predicted.to_numpy(), p.argmax(axis=1))
        np.testing.assert_allclose(pm.log_loss, -np.log(p[np.arange(len(pred)), labels]), atol=1e-13)
        np.testing.assert_allclose(pm.brier, ((np.eye(3)[labels] - p) ** 2).mean(axis=1), atol=1e-13)
    for row in table.itertuples(index=False):
        sample = per_match.loc[per_match.model.eq(row.model)]
        if row.scope == 'season': sample = sample.loc[sample.fold.eq(row.group)]
        elif row.scope == 'league': sample = sample.loc[sample.division.eq(row.group)]
        elif row.scope == 'actual_outcome': sample = sample.loc[sample.actual_hda.eq(int(row.group))]
        elif row.scope == 'season_league':
            season, div = row.group.split('/')
            sample = sample.loc[sample.fold.eq(season) & sample.division.eq(div)]
        assert len(sample) == row.n_matches and sample.accuracy.sum() == row.correct
        for field in POINT_FIELDS:
            if field in sample and sample[field].notna().all():
                np.testing.assert_allclose(sample[field].mean(), getattr(row, field), atol=2e-13)
        np.testing.assert_allclose(row.macro_f1, f1_score(sample.actual_hda, sample.predicted, labels=[0, 1, 2], average='macro', zero_division=0), atol=1e-13)
    # Saved 66-column run provides an external check after the new evaluation is complete.
    previous = ROOT / 'experiments/lr_poisson_reweight_20261012/results_20261015/predictions_by_match.csv.gz'
    previous_error = None
    if previous.exists():
        old = pd.read_csv(previous, parse_dates=['date'], float_precision='round_trip').set_index(KEYS).reindex(pred.set_index(KEYS).index)
        assert old.actual_hda.notna().all()
        previous_error = float(np.abs(old[[f'lr_p_{c}' for c in PROB_COLS]].to_numpy() - pred[[f'lr66_p_{c}' for c in PROB_COLS]].to_numpy()).max())
        assert previous_error < 1e-12
    result = dict(passed=True, n_matches=len(pred), models=5, reconstructed_score_matrices=3 * len(pred),
                  input_files_verified=28, max_loaded_model_probability_error=model_error,
                  max_training_only_imputer_error=imputer_error, max_reconstructed_nll_error=nll_error,
                  max_matrix_mass_error=mass_error, prior66_probability_error=previous_error,
                  selected_features=len(frozen['feature_names']),
                  result_hashes={p.name: sha(p) for p in out.glob('*') if p.is_file() and p.name != 'verification_audit.json'})
    dump(out / 'verification_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'result_hashes'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        audit(args.output_dir.resolve())
