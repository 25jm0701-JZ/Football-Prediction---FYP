# Atta Mills Phase 2 Results

## Scope

- Data: Premier League football-data.co.uk files, 7 seasons, 2660 matches.
- Target: full-time H/D/A result only.
- Validation: season-by-season walk-forward.
- Training exclusions: Bet365 odds, half-time variables, O/U 2.5, FootyStats.
- Added after Phase 1: original clean pre-match H2H and rolling shot-efficiency features.
- Market benchmark: Bet365 closing odds, converted to de-vigged implied probabilities.

## Feature Set

Phase 2 uses 65 features:

- 44 Atta Mills-style features from Phase 1.
- 21 original clean extras: 16 rolling shot-efficiency features and 5 H2H features.

Detailed feature mapping is saved in `outputs/atta_mills_pl_walkforward/phase2_with_original_features/feature_set.csv`.

## Aggregate Results

| Model | Accuracy | Macro F1 | Weighted F1 | Draw F1 | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|---:|
| random_forest | 0.5101 | 0.3773 | 0.4384 | 0.0073 | 1.0151 | 0.2020 |
| svm | 0.4996 | 0.3638 | 0.4247 | 0.0000 | 1.0277 | 0.2053 |
| fnn_mlp | 0.4921 | 0.3595 | 0.4187 | 0.0110 | 1.0356 | 0.2059 |
| voting_rf_xgb | 0.4904 | 0.3867 | 0.4387 | 0.0801 | 1.0307 | 0.2051 |
| random_forest_balanced | 0.4860 | 0.4327 | 0.4718 | 0.1971 | 1.0285 | 0.2054 |
| lr | 0.4860 | 0.4071 | 0.4534 | 0.1376 | 1.0708 | 0.2120 |
| naive_bayes | 0.4825 | 0.4134 | 0.4573 | 0.1447 | 3.6296 | 0.2958 |
| xgboost | 0.4768 | 0.3964 | 0.4408 | 0.1384 | 1.0807 | 0.2132 |
| lr_balanced | 0.4496 | 0.4304 | 0.4569 | 0.2786 | 1.0980 | 0.2187 |
| dummy_most_frequent | 0.4311 | 0.2008 | 0.2598 | 0.0000 | 20.5038 | 0.3792 |

## Comparison

Phase 1 best model was `lr` at 49.69% accuracy and 0.0443 Draw F1.

Phase 2 best model is `random_forest` at 51.01% accuracy and 0.0073 Draw F1.
Bet365 closing market accuracy is 54.87%, with 0.0000 Draw F1.

## Interpretation

Phase 2 tests whether the project's original clean pre-match features add useful signal beyond the strict Atta Mills-style reproduction. Model choice remains data-driven: the final candidate should be selected from walk-forward accuracy, macro F1, Draw F1, log loss, and Brier score rather than from the reference paper alone.
