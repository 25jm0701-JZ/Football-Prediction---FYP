# Football Prediction FYP

This repository contains the main Final Year Project experiments for football
match outcome prediction, value betting analysis, and a separate World Cup
prediction subproject.

The main project studies whether data-driven football features can produce
useful pre-match predictions and whether model probabilities can identify
value against bookmaker markets.

## Project Structure

| Track | Scope | Data | Main Model | Purpose |
|---|---|---|---|---|
| Track A | English E0-E3 leagues | football-data.co.uk | Logistic Regression | Core market-comparison model |
| Track B | Premier League only | football-data.co.uk + FootyStats | Logistic Regression | Enhanced feature experiment |
| Deep Learning | Premier League | Same feature set as Track B | PyTorch MLP / GRU | Model-family comparison |
| World Cup | International tournaments | World Cup match data + FIFA ranking/player features | Softmax / Poisson variants | Separate tournament prediction subproject |

Full experiment history is recorded in:

```text
docs/experiment_log.md
```

## Main Results

### Track A: English Four-League Baseline

Track A uses English professional league data from football-data.co.uk:

- Premier League, Championship, League One, League Two
- 2019/20 through 2025/26
- Rolling team form, goals, shot efficiency, head-to-head, and league-level
  features
- No bookmaker odds as training features

Key result:

| Validation | Model Accuracy | Market Accuracy | Difference | Value Betting ROI |
|---|---:|---:|---:|---:|
| 80/20 temporal split | **58.60%** | 49.70% | **+8.90pp** | +62.8% |
| Walk-forward | 57.43% | - | - | +55.0% |

The main takeaway is that a clean rolling-statistics Logistic Regression model
can outperform the market benchmark on the selected English four-league setup.

### Track B: Premier League FootyStats Enhancement

Track B focuses on the Premier League and adds FootyStats features:

- Pre-match PPG
- Pre-match xG
- Possession-related features

Key result:

| Configuration | Accuracy | Lift |
|---|---:|---:|
| Rolling-stat baseline | 62.28% | - |
| Logistic Regression + FootyStats | **68.42%** | **+6.14pp** |
| Bet365 market benchmark | 57.02% | - |

This is the strongest evidence that richer pre-match team-strength features
matter more than simply switching model families.

### Deep Learning Comparison

Deep learning models were tested against the Logistic Regression baseline using
the same feature set.

| Model | Accuracy | LogLoss | Macro F1 | Brier | Draw F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | **0.7018** | **0.7057** | 0.5942 | **0.1385** | 0.2059 |
| MLP-Wide | **0.7018** | 0.7574 | 0.6139 | 0.1467 | 0.2857 |
| MLP-Deep | **0.7018** | 0.7482 | **0.6429** | 0.1459 | **0.3810** |

The deep models did not beat LR on accuracy, but the residual MLP improved
draw detection substantially.

## Value Betting Method

Model probabilities are compared with bookmaker implied probabilities:

```text
Edge = P_model / P_market - 1
```

If the edge exceeds a threshold, the result is treated as a value bet. Three
strategies are tested:

| Strategy | Description |
|---|---|
| Buchdahl-style | Only bet home wins |
| Best per match | Bet only the highest-edge outcome per match |
| All outcomes | Bet every outcome above threshold |

At the 30% threshold in the English four-league walk-forward experiment, all
three strategies produced similar ROI, around +52% to +56%.

## Atta Mills Experiment Branch

A separate branch tests a stricter reproduction of the feature framework from
Atta Mills et al. (2024):

```text
codex/atta-mills-pl-walkforward
```

That branch intentionally excludes:

- Odds from training
- Half-time variables
- Over/Under 2.5
- FootyStats

Its current conclusion is that clean pre-match Atta Mills-style features do not
beat Bet365 closing odds on Premier League walk-forward validation. See the
branch README for details.

## World Cup Subproject

The `world_cup/` folder contains a separate international tournament prediction
pipeline.

Main idea:

- Predict World Cup knockout-stage outcomes.
- Use FIFA ranking differences, host advantage, knockout indicators, and
  player-level aggregate features.
- Compare Softmax, Poisson, and blended model variants.

Representative result:

| Model | LogLoss | Accuracy |
|---|---:|---:|
| Base Softmax | 0.978 | 57.8% |
| Softmax + player features | **0.917** | **64.1%** |

## Quick Start

Run from the repository root:

```powershell
# Track A: English four-league backtest and value betting
python scripts/backtest_english4.py

# Track B: Premier League FootyStats/Buchdahl-style comparison
python scripts/buchdahl_ppg_walkforward.py

# Deep learning model comparison
python scripts/deep_learning_experiment.py

# Two-tier ensemble experiment
python scripts/build_ensemble_model.py

# Generate thesis draft/output document
python scripts/build_partial_thesis.py
```

## Important Files

| Path | Purpose |
|---|---|
| `league/src/` | League data loading, feature engineering, models, evaluation |
| `scripts/backtest_english4.py` | Track A main backtest |
| `scripts/buchdahl_ppg_walkforward.py` | Track B Buchdahl/FootyStats experiment |
| `scripts/deep_learning_experiment.py` | PyTorch deep learning comparison |
| `scripts/build_ensemble_model.py` | Two-tier model training |
| `world_cup/` | World Cup prediction subproject |
| `docs/experiment_log.md` | Full experiment log |
| `论文点_更新版.md` | Thesis-point summary |

## Notes

- Generated outputs are ignored by Git under `outputs/`.
- Betting odds are generally used as market benchmarks, not default training
  features, unless an experiment explicitly says otherwise.
- Walk-forward or chronological validation is preferred over random splitting.
- Reported ROI values are experimental and should not be treated as financial
  advice.

## References

1. Buchdahl, J. (2003). *Rating Systems for Fixed Odds Football Match Prediction*.
2. Reade, J., Singleton, C. & Vaughan Williams, L. (2020). Betting markets and statistical models.
3. Luiz et al. (2024). A deep learning approach for football match prediction.
4. Atta Mills et al. (2024). Data-driven prediction of soccer outcomes using enhanced machine and deep learning techniques.
