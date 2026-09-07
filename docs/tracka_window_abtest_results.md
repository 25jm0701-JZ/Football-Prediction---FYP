# Track A Window A/B Test Results

## Purpose

This experiment tests whether the Luiz et al. (2024)-inspired 20-match rolling
window should replace the current Track A 5+10 rolling-window setup.

The test is intentionally narrow:

- A: current Track A windows `[5, 10]`
- B: 20-match-only windows `[20]`
- Same English E0-E3 football-data.co.uk data
- Same clean pre-match feature policy
- No bookmaker odds in training
- Bet365 closing odds used only as market benchmark
- Season-by-season walk-forward validation

## Data

- Divisions: E0, E1, E2, E3
- Seasons: 2019/20 through 2025/26
- Matches: 13,988
- Walk-forward test matches: 12,216

## Aggregate Results

| Config | Windows | Model | Features | Accuracy | Macro F1 | Draw F1 | Log Loss | Brier |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| A | 5+10 | Bet365 closing | 0 | 49.37% | 0.3650 | 0.0007 | 1.0190 | 0.2036 |
| A | 5+10 | RF | 50 | 45.18% | 0.3214 | 0.0136 | 1.0586 | 0.2127 |
| A | 5+10 | LR | 50 | 45.06% | 0.3394 | 0.0418 | 1.0629 | 0.2136 |
| A | 5+10 | XGB | 50 | 40.83% | 0.3486 | 0.1644 | 1.1901 | 0.2351 |
| B | 20 | Bet365 closing | 0 | 49.37% | 0.3650 | 0.0007 | 1.0190 | 0.2036 |
| B | 20 | LR | 28 | 44.70% | 0.3258 | 0.0235 | 1.0612 | 0.2133 |
| B | 20 | RF | 28 | 44.24% | 0.3122 | 0.0122 | 1.0634 | 0.2139 |
| B | 20 | XGB | 28 | 40.52% | 0.3475 | 0.1671 | 1.1798 | 0.2344 |

## Interpretation

The 20-match-only setup does not improve Track A. Compared with the current
5+10 setup:

- LR drops from 45.06% to 44.70%.
- RF drops from 45.18% to 44.24%.
- XGB is effectively unchanged but remains much weaker than LR/RF.
- Bet365 closing remains clearly stronger than both model configurations.

This means the Luiz et al. (2024) 20-match-window finding does not transfer
directly to our football-data.co.uk English4 setting when the 20-match window
replaces the current short/medium windows.

## Decision

Do not replace the current Track A 5+10 window setup with 20-only windows.

The next reasonable experiment is not "20-only as the new default", but either:

1. test `[5, 10, 20]` as an additive multi-scale window setup; or
2. keep `[5, 10]` and add Luiz-inspired relative features first.

The output files for this run are saved in:

- `outputs/tracka_window_abtest/fold_metrics.csv`
- `outputs/tracka_window_abtest/summary.csv`
- `outputs/tracka_window_abtest/metadata.json`
