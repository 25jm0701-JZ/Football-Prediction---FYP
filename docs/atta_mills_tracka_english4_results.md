# Atta Mills Track A English4 Results

## Scope

- Data: English E0-E3 football-data.co.uk files.
- Seasons: 2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25, 2025-26.
- Matches: 13988.
- Target: full-time H/D/A result only.
- Validation: season-by-season walk-forward.
- Training exclusions: bookmaker odds, half-time variables, O/U 2.5, FootyStats.
- Market benchmark: Bet365 closing odds, converted to de-vigged implied probabilities.

## Promotion/Relegation Logic

This experiment uses E0-E3 as one integrated chronological dataset. Team rolling
histories are keyed by team name rather than division, so promoted/relegated
teams carry their previous form into the next division. The `league_level`
feature marks match level:

- E0 = 0
- E1 = 1
- E2 = 2
- E3 = 3

## Data Coverage

| Division | Matches |
|---|---:|
| E0 | 2660 |
| E1 | 3864 |
| E2 | 3712 |
| E3 | 3752 |

## Feature Set

Total features: 66.

- 44 Atta Mills-style pre-match features.
- 16 rolling shot-efficiency features.
- 5 H2H features.
- 1 league-level feature for Track A promotion/relegation handling.

Feature mapping is saved in `outputs/atta_mills_tracka_english4/feature_set.csv`.

## Aggregate Results

| Model | Accuracy | Macro F1 | Weighted F1 | Draw F1 | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|---:|
| random_forest | 0.4601 | 0.3395 | 0.3877 | 0.0204 | 1.0553 | 0.2119 |
| lr | 0.4562 | 0.3520 | 0.3969 | 0.0415 | 1.0562 | 0.2120 |
| voting_rf_xgb | 0.4537 | 0.3487 | 0.3933 | 0.0544 | 1.0606 | 0.2129 |
| xgboost | 0.4435 | 0.3554 | 0.3961 | 0.0897 | 1.0758 | 0.2159 |
| random_forest_balanced | 0.4321 | 0.3870 | 0.4159 | 0.1818 | 1.0696 | 0.2152 |
| lr_balanced | 0.4188 | 0.3950 | 0.4156 | 0.2369 | 1.0809 | 0.2175 |

Bet365 closing benchmark:

| Model | Accuracy | Macro F1 | Draw F1 | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|
| bet365_closing | 0.4938 | 0.3653 | 0.0006 | 1.0189 | 0.2036 |

## Best Model By Fold

| Fold/Test Season | Best Model | Accuracy | Macro F1 | Draw F1 |
|---|---|---:|---:|---:|
| 2020-21 | random_forest | 0.4288 | 0.3300 | 0.0963 |
| 2021-22 | random_forest | 0.4597 | 0.3420 | 0.0110 |
| 2022-23 | random_forest | 0.4693 | 0.3406 | 0.0000 |
| 2023-24 | lr | 0.4764 | 0.3524 | 0.0201 |
| 2024-25 | lr | 0.4769 | 0.3541 | 0.0073 |
| 2025-26 | lr | 0.4676 | 0.3412 | 0.0000 |

## Interpretation

Best model by aggregate accuracy is `random_forest` at 46.01%.
This Track A version tests whether adding E1-E3 improves the clean pre-match
Atta Mills branch by increasing training volume and preserving team history
across promotion/relegation.
