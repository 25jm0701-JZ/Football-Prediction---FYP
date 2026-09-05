# Atta Mills Phase 3 Half-Time Results

## Scope

- Data: Premier League football-data.co.uk files, 7 seasons, 2660 matches.
- Target: full-time H/D/A result.
- Validation: season-by-season walk-forward.
- Purpose: reproduce the paper-style in-play setting by adding half-time information.
- Excluded project extras: H2H, rolling shot-efficiency features, FootyStats, player features, and O/U 2.5 target/features.
- Market benchmark: Bet365 closing odds, converted to de-vigged implied probabilities.

## Aggregate Comparison

Phase 1 strict pre-match best: `lr` at 49.69% accuracy.

| Config | Features | Best Model | Accuracy | Macro F1 | Draw F1 | Log Loss | Market Accuracy |
|---|---:|---|---:|---:|---:|---:|---:|
| paper_ht_no_odds | 51 | random_forest_balanced | 0.6088 | 0.5828 | 0.3879 | 0.8695 | 0.5487 |
| paper_ht_with_b365_opening | 55 | random_forest | 0.6215 | 0.5555 | 0.2806 | 0.8434 | 0.5487 |

## Detailed Results

### paper_ht_no_odds

Feature set: 51 features. Includes 44 Atta Mills-style team-state features and 7 half-time features.

| Model | Accuracy | Macro F1 | Weighted F1 | Draw F1 | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|---:|
| random_forest_balanced | 0.6088 | 0.5828 | 0.6141 | 0.3879 | 0.8695 | 0.1696 |
| lr | 0.6070 | 0.5390 | 0.5832 | 0.2591 | 0.8518 | 0.1662 |
| random_forest | 0.6066 | 0.5394 | 0.5827 | 0.2663 | 0.8594 | 0.1671 |
| voting_rf_xgb | 0.6061 | 0.5367 | 0.5811 | 0.2546 | 0.8639 | 0.1681 |
| svm | 0.6044 | 0.5052 | 0.5585 | 0.1601 | 0.8912 | 0.1730 |
| xgboost | 0.5969 | 0.5325 | 0.5757 | 0.2594 | 0.9005 | 0.1739 |
| fnn_mlp | 0.5864 | 0.4921 | 0.5443 | 0.1592 | 0.8864 | 0.1718 |
| lr_balanced | 0.5807 | 0.5659 | 0.5944 | 0.3849 | 0.8750 | 0.1717 |
| naive_bayes | 0.5798 | 0.5402 | 0.5769 | 0.3012 | 2.6480 | 0.2324 |
| dummy_most_frequent | 0.4311 | 0.2008 | 0.2598 | 0.0000 | 20.5038 | 0.3792 |

Best model: `random_forest_balanced` at 60.88% accuracy and 0.3879 Draw F1.
Bet365 closing market benchmark: 54.87% accuracy, 0.0000 Draw F1.

### paper_ht_with_b365_opening

Feature set: 55 features. Includes 44 Atta Mills-style team-state features, 7 half-time features, and 4 Bet365 opening-odds features.

| Model | Accuracy | Macro F1 | Weighted F1 | Draw F1 | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|---:|
| random_forest | 0.6215 | 0.5555 | 0.5987 | 0.2806 | 0.8434 | 0.1638 |
| random_forest_balanced | 0.6193 | 0.5876 | 0.6212 | 0.3750 | 0.8528 | 0.1665 |
| lr | 0.6184 | 0.5518 | 0.5967 | 0.2626 | 0.8362 | 0.1629 |
| svm | 0.6162 | 0.5270 | 0.5771 | 0.2022 | 0.8717 | 0.1687 |
| voting_rf_xgb | 0.6145 | 0.5467 | 0.5912 | 0.2611 | 0.8460 | 0.1647 |
| fnn_mlp | 0.6070 | 0.5247 | 0.5728 | 0.2142 | 0.8826 | 0.1700 |
| xgboost | 0.6053 | 0.5398 | 0.5844 | 0.2529 | 0.8824 | 0.1706 |
| naive_bayes | 0.5882 | 0.5518 | 0.5872 | 0.3208 | 2.9222 | 0.2364 |
| lr_balanced | 0.5860 | 0.5691 | 0.5991 | 0.3723 | 0.8606 | 0.1690 |
| dummy_most_frequent | 0.4311 | 0.2008 | 0.2598 | 0.0000 | 20.5038 | 0.3792 |

Best model: `random_forest` at 62.15% accuracy and 0.2806 Draw F1.
Bet365 closing market benchmark: 54.87% accuracy, 0.0000 Draw F1.


## Interpretation

This experiment is not a deployable pre-match predictor because current-match
half-time goals/result are only known after the first half. It is an explicit
methodology check: if accuracy jumps toward the Atta Mills et al. headline range
after adding half-time variables, the gap is likely explained by the paper's
in-play information rather than by the clean pre-match feature set.

## Recommendation

This branch is not recommended as the final project direction. Keep it as a
backup and methodology comparison only.
