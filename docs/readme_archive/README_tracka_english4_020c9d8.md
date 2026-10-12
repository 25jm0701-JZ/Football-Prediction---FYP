# Football Prediction FYP — Atta Mills Track A English4 Branch

This branch extends the Atta Mills Premier League experiment to English E0-E3
as one integrated Track A dataset.

The question tested here:

> If we add Championship, League One, and League Two data to the clean pre-match
> Atta Mills-style setup, does the larger training set and promotion/relegation
> continuity improve performance?

Current answer: no. The extra leagues increase sample size and preserve team
history across promotion/relegation, but the best model still underperforms the
Bet365 closing market.

## Branch Context

| Item | Value |
|---|---|
| Parent experiment branch | `codex/atta-mills-pl-walkforward` |
| This branch | `codex/atta-mills-tracka-english4` |
| Baseline backup tag | `backup/current-experiment-2026-09-03` |
| Full experiment log | `docs/experiment_log.md` |

## Data

Data comes from football-data.co.uk English league CSV files:

| Division | Meaning | Matches |
|---|---|---:|
| E0 | Premier League | 2,660 |
| E1 | Championship | 3,864 |
| E2 | League One | 3,712 |
| E3 | League Two | 3,752 |
| **Total** |  | **13,988** |

Seasons covered:

```text
2019/20 through 2025/26
```

Target:

```text
Full-time H/D/A result
```

## Promotion/Relegation Logic

This branch follows the original Track A idea:

- E0-E3 are merged into one chronological dataset.
- Team rolling histories are keyed by team name, not by division.
- A promoted or relegated team carries its historical form into the next
  division.
- `league_level` tells the model the match level:
  - E0 = 0
  - E1 = 1
  - E2 = 2
  - E3 = 3

This avoids the PL-only problem where promoted teams can have weak or missing
Premier League history.

## Feature Set

The experiment uses 66 clean pre-match features:

- 44 Atta Mills-style features:
  - team state
  - attack strength
  - defense strength
  - goals for/against
  - goal differential
  - win/draw/loss history
  - win/loss margin goals
- 16 rolling shot-efficiency features
- 5 H2H features
- 1 `league_level` feature

Excluded from training:

- Bet365 odds and other bookmaker odds
- Half-time result/goals
- O/U 2.5
- FootyStats
- Current-match raw statistics

## Validation

The experiment uses season-by-season walk-forward validation:

- Train on all seasons before the test season.
- Test on the next season.
- Repeat from 2020/21 through 2025/26.

Total walk-forward test matches: 12,216.

## Results

| Model | Accuracy | Macro F1 | Draw F1 | LogLoss | Brier |
|---|---:|---:|---:|---:|---:|
| Random Forest | **46.01%** | 0.3395 | 0.0204 | 1.0553 | 0.2119 |
| Logistic Regression | 45.62% | 0.3520 | 0.0415 | 1.0562 | 0.2120 |
| Voting RF+XGB | 45.37% | 0.3487 | 0.0544 | 1.0606 | 0.2129 |
| XGBoost | 44.35% | 0.3554 | 0.0897 | 1.0758 | 0.2159 |
| Random Forest balanced | 43.21% | 0.3870 | 0.1818 | 1.0696 | 0.2152 |
| LR balanced | 41.88% | 0.3950 | **0.2369** | 1.0809 | 0.2175 |
| Bet365 closing | **49.38%** | 0.3653 | 0.0006 | **1.0189** | **0.2036** |

## Interpretation

Adding E1-E3 did not improve the clean Atta Mills branch. It solved part of the
promotion/relegation history problem, but the combined E0-E3 task is more
heterogeneous than PL-only prediction. Under this feature set, the extra data
does not compensate for cross-division differences.

The result also reinforces the earlier finding: Atta Mills et al.'s high
headline accuracy is not directly comparable to this clean pre-match setting,
because their framework includes half-time/in-play information.

## Run

From the repository root:

```powershell
python scripts/atta_mills_tracka_english4.py
```

Outputs are written locally to:

```text
outputs/atta_mills_tracka_english4/
```

Generated outputs are ignored by Git.

## Key Files

| Path | Purpose |
|---|---|
| `scripts/atta_mills_tracka_english4.py` | E0-E3 Track A experiment runner |
| `docs/atta_mills_tracka_english4_results.md` | Result summary |
| `docs/experiment_log.md` | Full experiment log |
| `scripts/atta_mills_pl_phase1.py` | PL-only Phase 1 reference |
| `scripts/atta_mills_pl_phase2.py` | PL-only Phase 2 reference |
