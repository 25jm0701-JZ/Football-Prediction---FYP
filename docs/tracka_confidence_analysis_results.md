# Track A High-Confidence Prediction Analysis

## Purpose

This experiment evaluates whether the current clean Track A 5+10 baseline is
more useful when restricted to matches where the model is highly confident.

The model and features are not changed. This is a post-model evaluation:

- Data: football-data.co.uk English E0-E3
- Seasons: 2019/20 through 2025/26
- Features: current clean Track A 5+10 baseline, 50 features
- Training exclusions: bookmaker odds, half-time variables, current-match raw
  statistics, FootyStats
- Validation: season-by-season walk-forward
- Test matches: 12,216
- Models: LR, RF, XGB
- Market benchmark: Bet365 closing, evaluated on the same selected matches

## Overall Model Metrics

| Model | Accuracy | Macro F1 | Draw F1 | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|
| RF | **45.18%** | 0.3214 | 0.0136 | 1.0586 | 0.2127 |
| LR | 45.06% | 0.3394 | 0.0418 | 1.0629 | 0.2136 |
| XGB | 40.83% | 0.3486 | 0.1644 | 1.1901 | 0.2351 |

## Confidence Threshold Results

High confidence means the model's largest predicted probability is at least the
threshold. For example, if LR predicts `[A=0.15, D=0.23, H=0.62]`, then the
confidence is `0.62`.

| Model | Threshold | Matches | Coverage | Model Acc | Market Acc Same Matches | Pred A | Pred D | Pred H | Edge>10% ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LR | 0.50 | 3,566 | 29.19% | 53.11% | **54.12%** | 694 | 31 | 2,841 | -6.72% |
| LR | 0.60 | 885 | 7.24% | 62.37% | **63.05%** | 146 | 1 | 738 | -5.39% |
| LR | 0.65 | 340 | 2.78% | 66.18% | **66.76%** | 51 | 0 | 289 | +12.74% |
| LR | 0.70 | 117 | 0.96% | 68.38% | **69.23%** | 22 | 0 | 95 | +28.09% |
| RF | 0.50 | 2,738 | 22.41% | 55.48% | **55.77%** | 231 | 0 | 2,507 | -7.06% |
| RF | 0.60 | 587 | 4.81% | 65.42% | **66.27%** | 27 | 0 | 560 | -9.24% |
| RF | 0.65 | 265 | 2.17% | 69.06% | **70.57%** | 8 | 0 | 257 | -3.43% |
| RF | 0.70 | 87 | 0.71% | 73.56% | **74.71%** | 1 | 0 | 86 | -9.26% |
| XGB | 0.50 | 7,263 | 59.45% | 44.09% | **50.94%** | 1,848 | 646 | 4,769 | -9.63% |
| XGB | 0.60 | 4,086 | 33.45% | 47.06% | **51.96%** | 941 | 273 | 2,872 | -6.04% |
| XGB | 0.70 | 2,098 | 17.17% | 49.76% | **54.10%** | 420 | 117 | 1,561 | -8.12% |

## Interpretation

The high-confidence filter works in the narrow sense that model accuracy rises
as confidence increases. For example:

- LR rises from 45.06% overall to 62.37% at threshold 0.60.
- RF rises from 45.18% overall to 65.42% at threshold 0.60.
- RF reaches 73.56% at threshold 0.70, but only covers 87 matches.

However, this does not show a market edge. On the same selected matches,
Bet365 closing accuracy remains higher at every reported threshold.

The high-confidence subsets are also heavily skewed toward home wins. RF at
threshold 0.70 predicts 86 home wins, 1 away win, and 0 draws. This means the
model's "confidence" mostly identifies strong home-favourite situations, which
the betting market also recognizes.

The LR edge>10% ROI becomes positive at thresholds 0.65 and 0.70, but those
samples are very small: 340 and 117 selected matches respectively. This is not
strong enough to overturn the main conclusion without further statistical
testing.

## Decision

High-confidence analysis is useful as a supplementary evaluation, but it should
not change the main Track A conclusion.

Recommended thesis wording:

> The model becomes more accurate on high-confidence subsets, but the same
> subsets are also easier for the betting market. Bet365 closing odds remain
> more accurate on the selected matches, suggesting that Track A confidence is
> mostly identifying obvious favourites rather than producing independent
> predictive edge.

Combined Luiz-inspired conclusion:

> We tested two feature-engineering ideas inspired by Luiz et al. (2024): a
> 20-match rolling window and relative-strength features. Under a strict
> pre-match football-data.co.uk English4 setting, neither improved the Track A
> baseline. High-confidence filtering increased model accuracy, but the same
> subsets were also easier for the betting market, and Bet365 closing odds
> remained more accurate.

## Outputs

- `scripts/tracka_confidence_analysis.py`
- `outputs/tracka_confidence_analysis/predictions.csv`
- `outputs/tracka_confidence_analysis/fold_metrics.csv`
- `outputs/tracka_confidence_analysis/aggregate_metrics.csv`
- `outputs/tracka_confidence_analysis/confidence_summary.csv`
- `outputs/tracka_confidence_analysis/metadata.json`
