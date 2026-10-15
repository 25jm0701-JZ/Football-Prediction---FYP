# Clean Track A: Logistic-regression-guided Poisson scoreline distribution

This experiment is separate from the historical Poisson and LR baseline outputs.

## Completed local run: 2026-10-15

The original attachment's fixed 66-feature, unscaled LR (C=1) experiment has now completed all six expanding-window season tests: 12,216 predictions from 28 raw CSVs. Training and score-distribution computation code are byte-identical to the supplied ZIP.

- [Chinese empirical report](results_20261015/report_CN.md)
- [Training/test boundaries for all six seasons](results_20261015/walk_forward_folds.csv)
- [Saved-result verification and source hashes](results_20261015/verification_audit.json)
- [Actual environment versions and input hashes](results_20261015/run_metadata.json)

Overall Poisson versus fusion: H/D/A accuracy 42.0187% → 45.6287%; H/D/A Log Loss 1.093788 → 1.056095; exact-score accuracy 11.8533% → 12.1807%; exact-score NLL 3.005417 → 2.967725. Exact-score accuracy increases by 40 matches, but its paired date-cluster 95% interval is [-0.2110, +0.8540] percentage points and includes zero. NLL and expected-goal MAE differences favor fusion with intervals excluding zero. Fusion H/D/A probabilities equal LR probabilities by construction.

The 12 fusion tests and 5 existing temporal-feature tests passed. Saved predictions were independently checked against feature rows, labels, seasonal boundaries, recomputed metrics, and the NLL identity. Each test season uses all earlier seasons to fit frozen model parameters; earlier dates in the test season update rolling feature histories only. All matches on the same date are excluded from one another's features.

Reproduce into a new output directory:

```powershell
python -B experiments/lr_poisson_reweight_20261012/run_experiment.py --output-dir experiments/lr_poisson_reweight_20261012/results_new
python -B experiments/lr_poisson_reweight_20261012/verify_results.py --output-dir experiments/lr_poisson_reweight_20261012/results_new
```

## Mathematical definition

For observed home score `i` and away score `j`, set class `c=0` if `i<j` (away win), `c=1` if `i=j` (draw), `c=2` if `i>j` (home win). Given independent-Poisson score PMF `P(i,j)` and LR three-class distribution `q=(q_A,q_D,q_H)`, define `S_c=sum_{(i,j) in c}P(i,j)` and `P_fused(i,j)=q_c P(i,j)/S_c`. This exactly preserves within-class Poisson ratios and guarantees fused class marginals equal LR probabilities. For actual outcome `y`, the exact score NLL improvement identity is `NLL_fused_score - NLL_poisson_score = -log(q_y) + log(S_y)`; its dataset mean **must** equal the LR-vs-Poisson H/D/A log-loss difference. The runner checks this equality independently over every test match. All three LR classes are used; no inversions or Poisson parameter recovery are attempted.

## Reproduce

Run from the project root (Python 3.11 recommended):

```bash
python -m pip install -r experiments/lr_poisson_reweight_20261012/requirements.txt
python -m unittest discover -s experiments/lr_poisson_reweight_20261012 -p 'test_*.py' -v
python experiments/lr_poisson_reweight_20261012/run_experiment.py
```

Outputs are written to `experiments/lr_poisson_reweight_20261012/results/` by default, **never** to the existing baseline directories. An alternate `--output-dir` is supported. Input 28 English4 CSVs must be present under `league/data/english_raw/`.

- `predictions_by_match.csv.gz`: 12,216 walk-forward fixtures, Poisson and LR H/D/A probabilities, fused H/D/A probabilities, Poisson λ, both models' expected and modal scores, true-score NLL, goal errors, and audit fields. Storing full score matrices would create an unwieldy artifact; every matrix is deterministically reconstructible from each fixture's λ, LR probabilities, and recorded tolerance.
- `hda_metrics.csv`: all six test seasons and full combined coverage for Poisson, LR and fusion.
- `score_metrics.csv`: exact score mode accuracy, full-distribution true-score NLL, expected-goal MAE and within ±1 for Poisson and fusion.
- `paired_uncertainty.csv`: outcome-wise paired bootstrap CIs, resampling entire match dates within each tested season; 1000 replicates.
- `report_CN.md`: numerical, empirical (not predetermined) results; `run_metadata.json` contains library versions and input hashes.

## Strict protocol

The LR classifier reuses the 66 existing Clean Track A historical features via `experiments/feature_search_20261010/run_feature_search.py:build_data`. Feature snapshots are constructed before any outcomes from the **same date** are added to rolling histories. Test-season earlier fixtures can inform later fixtures' historical rolling features, but the LR/Poisson models themselves are fitted only on completed prior *seasons*. A held-out test season is never used for fitting, tuning, calibration, feature selection, or imputation fitting. No odds, match-time raw match stats, halftime or FootyStats data are predictors. This test is *fixed-feature, fixed-hyperparameter*: 66 features, LR C=1.0, same Loukas historical-goals model and smoothing=0.05 as before. The baseline attack/defence model has not been changed. LR predicts [Away, Draw, Home].

The score-grid support adapts solely to predicted pre-match λ until unnormalized omitted joint tail <=1e-12. If observed scores rarely fall outside this support, true-score PMF is obtained from the exact Poisson formula, not by expanding support using the score. H/D/A group denominators must be non-zero; very small denominators are checked and rejected rather than silently replaced. All probabilities must be finite, nonnegative, normalized and class-consistent. No score distribution is inferred from LR alone: only the two scoreline models are evaluated at score level.

Historic Poisson baseline tables use `max_goals=10`; the new adaptive grid and expected-value-based MAE may yield slight numerical discrepancies. Within-1 home/away metrics in the old report use the *rounded mode*, which is not directly comparable to the expected-value-based within-1 rates here. For fair new comparisons, **both** Poisson and fusion use the same new score-level metrics.

**Status:** the complete local run is available in `results_20261015/`. `analysis_CN.md` records the original attachment's pre-run review; its pending-run statements describe that earlier delivery, not the completed 2026-10-15 run.
