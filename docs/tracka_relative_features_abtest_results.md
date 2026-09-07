# Track A Relative-Feature A/B Test Results

## Purpose

This experiment tests whether Luiz et al. (2024)-inspired relative-strength
features improve the current clean Track A English4 setup.

The test keeps the rolling window fixed at `[5, 10]` and changes only the
feature representation:

- A: current Track A 5+10 baseline
- B: current Track A 5+10 baseline plus relative features

## Data And Validation

- Data: football-data.co.uk English E0-E3
- Seasons: 2019/20 through 2025/26
- Matches: 13,988
- Walk-forward test matches: 12,216
- Training exclusions: bookmaker odds, half-time variables, current-match raw
  statistics, FootyStats
- Market benchmark: Bet365 closing implied probabilities
- Models: LR, RF, XGB

## Added Relative Features

For each window in `[5, 10]`, the script adds 14 relative features:

- relative goals for
- relative goals against, expressed so positive means the home team has the
  stronger defensive record
- relative goal difference
- relative points
- relative win, draw, and loss rates
- home attack versus away defense
- away attack versus home defense
- relative attack-defense gap
- relative shots
- relative shots on target
- relative shot accuracy
- relative conversion rate

Total added features: 28.

These features are built only from already-lagged rolling inputs, so they do not
introduce current-match leakage.

## Aggregate Results

| Config | Model | Features | Relative Features | Accuracy | Macro F1 | Draw F1 | Log Loss | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A: baseline | Bet365 closing | 0 | 0 | **49.37%** | 0.3650 | 0.0007 | **1.0190** | **0.2036** |
| A: baseline | RF | 50 | 0 | **45.18%** | 0.3214 | 0.0136 | 1.0586 | 0.2127 |
| A: baseline | LR | 50 | 0 | 45.06% | 0.3394 | 0.0418 | 1.0629 | 0.2136 |
| A: baseline | XGB | 50 | 0 | 40.83% | 0.3486 | 0.1644 | 1.1901 | 0.2351 |
| B: +relative | Bet365 closing | 0 | 0 | **49.37%** | 0.3650 | 0.0007 | **1.0190** | **0.2036** |
| B: +relative | RF | 78 | 28 | **45.06%** | 0.3231 | 0.0173 | 1.0584 | 0.2127 |
| B: +relative | LR | 78 | 28 | 45.05% | 0.3402 | 0.0426 | 1.0638 | 0.2138 |
| B: +relative | XGB | 78 | 28 | 40.86% | 0.3472 | 0.1584 | 1.1935 | 0.2359 |

## Interpretation

Relative features do not materially improve Track A under the current clean
football-data.co.uk feature set.

- LR accuracy is effectively unchanged: 45.06% -> 45.05%.
- RF accuracy falls slightly: 45.18% -> 45.06%.
- XGB changes only trivially: 40.83% -> 40.86%.
- Draw F1 improves only marginally for LR/RF and falls for XGB.
- Bet365 closing remains clearly stronger than all model configurations.

The likely explanation is that LR, RF, and XGB can already infer many of these
simple home-away differences from the original paired home and away rolling
features. The relative features are more interpretable, but they do not add
much new information.

## Decision

Do not add this full 28-feature relative block as a default Track A feature set.

The result is still useful for the thesis: it shows that a Luiz-inspired
feature-engineering idea was tested in a strict pre-match setting, but did not
transfer into meaningful predictive improvement with football-data.co.uk data.

The next useful step is high-confidence evaluation on the existing best Track A
baseline, rather than adding more similar difference features.

## Outputs

- `scripts/tracka_relative_features_abtest.py`
- `outputs/tracka_relative_features_abtest/fold_metrics.csv`
- `outputs/tracka_relative_features_abtest/summary.csv`
- `outputs/tracka_relative_features_abtest/metadata.json`
