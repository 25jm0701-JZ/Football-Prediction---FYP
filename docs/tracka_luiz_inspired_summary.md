# Track A Luiz-Inspired Experiment Summary

## Branch

`codex/football-nn-methodology-plan`

## Purpose

This branch tests whether selected methodology ideas from Luiz, Fialho, and
Teixeira (2024), *Is Football Unpredictable? Predicting Matches Using Neural
Networks*, improve the project's clean Track A English4 baseline.

The branch keeps the project data source fixed:

- football-data.co.uk English E0-E3
- 2019/20 through 2025/26
- clean pre-match features only
- no bookmaker odds in training
- no half-time or in-play variables
- no current-match raw statistics
- no FootyStats or WhoScored data

The market benchmark is Bet365 closing odds evaluated on the same matches.

## Experiments

### Experiment 021: Window A/B Test

Question: should the current Track A `[5, 10]` rolling windows be replaced by a
20-match-only window inspired by Luiz et al. (2024)?

Result:

| Config | Best Model | Accuracy |
|---|---|---:|
| 5+10 baseline | RF | 45.18% |
| 20-only | LR | 44.70% |
| Bet365 closing | Benchmark | 49.37% |

Decision: do not replace the current 5+10 setup with 20-only windows.

### Experiment 022: Relative-Feature A/B Test

Question: do Luiz-inspired relative-strength features improve the current 5+10
Track A baseline?

Result:

| Config | Best Model | Features | Accuracy |
|---|---|---:|---:|
| 5+10 baseline | RF | 50 | 45.18% |
| 5+10 + relative features | RF | 78 | 45.06% |
| Bet365 closing | Benchmark | 0 | 49.37% |

Decision: do not adopt the full 28-feature relative block as a default Track A
feature set. The features are interpretable but mostly redundant with the
paired home/away rolling features already present.

### Experiment 023: High-Confidence Analysis

Question: even if Track A does not win overall, does it become useful on
matches where the model is highly confident?

Key result:

| Model | Threshold | Coverage | Model Accuracy | Market Accuracy Same Matches |
|---|---:|---:|---:|---:|
| LR | 0.60 | 7.24% | 62.37% | 63.05% |
| LR | 0.70 | 0.96% | 68.38% | 69.23% |
| RF | 0.60 | 4.81% | 65.42% | 66.27% |
| RF | 0.70 | 0.71% | 73.56% | 74.71% |

Decision: high-confidence filtering increases model accuracy, but Bet365
closing remains more accurate on the same selected matches. This should be kept
as supplementary analysis, not as evidence that Track A beats the market.

## Thesis-Ready Conclusion

> We tested two feature-engineering ideas inspired by Luiz et al. (2024): a
> 20-match rolling window and relative-strength features. Under a strict
> pre-match football-data.co.uk English4 setting, neither improved the Track A
> baseline. High-confidence filtering increased model accuracy, but the same
> subsets were also easier for the betting market, and Bet365 closing odds
> remained more accurate.

## Final Decision

Keep the original clean Track A baseline as the main English4 model. Do not
adopt 20-only windows or the full relative-feature block. The Luiz-inspired
experiments are useful as negative-results evidence showing that ideas from a
WhoScored-based neural-network paper do not directly transfer to the project's
football-data.co.uk English4 setting.

## Files

- `scripts/tracka_window_abtest.py`
- `scripts/tracka_relative_features_abtest.py`
- `scripts/tracka_confidence_analysis.py`
- `docs/tracka_window_abtest_results.md`
- `docs/tracka_relative_features_abtest_results.md`
- `docs/tracka_confidence_analysis_results.md`
- `outputs/tracka_window_abtest/`
- `outputs/tracka_relative_features_abtest/`
- `outputs/tracka_confidence_analysis/`
