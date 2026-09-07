# Football Prediction FYP - NN Methodology Side Branch

This branch is a methodology side experiment built on top of the clean
Atta Mills Track A English4 baseline.

It tests whether selected ideas from Luiz, Fialho, and Teixeira (2024),
*Is Football Unpredictable? Predicting Matches Using Neural Networks*, improve
the project's strict pre-match English E0-E3 model.

Current answer: no. The tested NN-methodology ideas do not improve the clean
Track A English4 baseline, so this branch should be treated as negative-result
evidence rather than the active main line.

## Branch Context

| Item | Value |
|---|---|
| Main Track A branch | `codex/atta-mills-tracka-english4` |
| This side branch | `codex/football-nn-methodology-plan` |
| Parent commit | `020c9d8` |
| Full experiment summary | `docs/tracka_luiz_inspired_summary.md` |
| Full experiment log | `docs/experiment_log.md` |

## What This Branch Tests

The branch keeps the Track A data source and leakage rules fixed:

- football-data.co.uk English E0-E3
- 2019/20 through 2025/26
- clean pre-match features only
- no bookmaker odds in training
- no half-time or in-play variables
- no current-match raw statistics
- no FootyStats or WhoScored data

The market benchmark is Bet365 closing odds evaluated on the same matches.

This branch then tests three additions around the current Track A setup:

1. replacing the current 5+10 rolling windows with a 20-match-only window
2. adding Luiz-inspired relative-strength features
3. checking whether high-confidence model predictions beat the market on the
   same selected matches

## Results

### Experiment 021 - Window A/B Test

| Config | Best Model | Accuracy |
|---|---|---:|
| 5+10 baseline | RF | 45.18% |
| 20-only | LR | 44.70% |
| Bet365 closing | Benchmark | 49.37% |

Decision: do not replace the current 5+10 setup with 20-only windows.

### Experiment 022 - Relative-Feature A/B Test

| Config | Best Model | Features | Accuracy |
|---|---|---:|---:|
| 5+10 baseline | RF | 50 | 45.18% |
| 5+10 + relative features | RF | 78 | 45.06% |
| Bet365 closing | Benchmark | 0 | 49.37% |

Decision: do not adopt the full 28-feature relative block as a default Track A
feature set. The features are interpretable, but mostly duplicate signal
already present in paired home/away rolling features.

### Experiment 023 - High-Confidence Analysis

| Model | Threshold | Coverage | Model Accuracy | Market Accuracy Same Matches |
|---|---:|---:|---:|---:|
| LR | 0.60 | 7.24% | 62.37% | 63.05% |
| LR | 0.70 | 0.96% | 68.38% | 69.23% |
| RF | 0.60 | 4.81% | 65.42% | 66.27% |
| RF | 0.70 | 0.71% | 73.56% | 74.71% |

Decision: high-confidence filtering raises model accuracy, but Bet365 closing
remains more accurate on the same selected matches.

## Final Decision

Keep the original clean Track A English4 baseline as the main model. Do not use
this `football-nn-methodology-plan` branch as the active main line.

The useful contribution of this branch is methodological: it shows that ideas
from a WhoScored-based neural-network paper do not directly transfer to the
project's stricter football-data.co.uk English4 setting.

## Run

From the repository root:

```powershell
python scripts/tracka_window_abtest.py
python scripts/tracka_relative_features_abtest.py
python scripts/tracka_confidence_analysis.py
```

Generated outputs are written locally under:

```text
outputs/tracka_window_abtest/
outputs/tracka_relative_features_abtest/
outputs/tracka_confidence_analysis/
```

Generated outputs are ignored by Git.

## Key Files

| Path | Purpose |
|---|---|
| `docs/tracka_luiz_inspired_summary.md` | Branch-level conclusion |
| `docs/tracka_window_abtest_results.md` | Window A/B test results |
| `docs/tracka_relative_features_abtest_results.md` | Relative-feature A/B test results |
| `docs/tracka_confidence_analysis_results.md` | High-confidence analysis results |
| `scripts/tracka_window_abtest.py` | Window A/B test runner |
| `scripts/tracka_relative_features_abtest.py` | Relative-feature A/B test runner |
| `scripts/tracka_confidence_analysis.py` | High-confidence analysis runner |
