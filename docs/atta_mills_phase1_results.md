# Atta Mills Phase 1 Results

## Scope

- Data: Premier League football-data.co.uk files, 7 seasons, 2660 matches.
- Target: full-time H/D/A result only.
- Validation: season-by-season walk-forward.
- Training exclusions: Bet365 odds, half-time variables, O/U 2.5, FootyStats, H2H, shots.
- Market benchmark: Bet365 closing odds, converted to de-vigged implied probabilities.

## Feature Set

Phase 1 uses 44 Atta Mills-style pre-match features.

| Paper Feature Family | Count |
|---|---:|
| Attack Strength | 4 |
| Defense Strength | 4 |
| Draw History | 4 |
| Goal Differential | 4 |
| Goals Against | 4 |
| Goals Forward | 4 |
| Loss History | 4 |
| Loss Margin Goals | 4 |
| Team State | 4 |
| Win History | 4 |
| Win Margin Goals | 4 |

Detailed feature mapping is saved in `outputs/atta_mills_pl_walkforward/phase1_atta_mills_only/feature_set.csv`.

## Aggregate Results

| Model | Accuracy | Macro F1 | Weighted F1 | Draw F1 | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|---:|
| lr | 0.4969 | 0.3810 | 0.4368 | 0.0443 | 1.0339 | 0.2065 |
| fnn_mlp | 0.4956 | 0.3750 | 0.4315 | 0.0356 | 1.0328 | 0.2063 |
| random_forest | 0.4952 | 0.3709 | 0.4288 | 0.0247 | 1.0353 | 0.2066 |
| svm | 0.4899 | 0.3538 | 0.4140 | 0.0000 | 1.0367 | 0.2075 |
| naive_bayes | 0.4868 | 0.4087 | 0.4543 | 0.1245 | 3.0521 | 0.2906 |
| voting_rf_xgb | 0.4838 | 0.3861 | 0.4360 | 0.0929 | 1.0544 | 0.2104 |
| xgboost | 0.4658 | 0.3883 | 0.4319 | 0.1427 | 1.1041 | 0.2186 |
| random_forest_balanced | 0.4570 | 0.4130 | 0.4483 | 0.2036 | 1.0493 | 0.2101 |
| lr_balanced | 0.4425 | 0.4172 | 0.4459 | 0.2447 | 1.0595 | 0.2129 |
| dummy_most_frequent | 0.4311 | 0.2008 | 0.2598 | 0.0000 | 20.5038 | 0.3792 |

## Initial Interpretation

Best model by accuracy: `lr` at 49.69%.
Bet365 closing market accuracy: 54.87%.

The next decision is whether Phase 1 is strong enough to keep as the main model, or whether to start Phase 2 by adding the project's original extra features such as H2H and rolling shot efficiency.
