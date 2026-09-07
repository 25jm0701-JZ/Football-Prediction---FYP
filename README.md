# Football Prediction FYP - Track A Poisson Goals Baseline

This branch adds a Loukas et al. (2024)-style Poisson goals baseline under the
clean Track A English4 line.

It is not Track B. In this project, Track B is reserved for FootyStats-enhanced
models. This branch uses only football-data.co.uk match results and historical
full-time goals, then predicts:

- expected home goals
- expected away goals
- modal scoreline
- H/D/A probabilities aggregated from the scoreline matrix

## Branch Context

| Item | Value |
|---|---|
| Main Track A branch | `codex/atta-mills-tracka-english4` |
| This branch | `codex/tracka-poisson-goals` |
| Result document | `docs/tracka_poisson_goals_results.md` |
| Output directory | `outputs/tracka_poisson_goals/` |

## Conclusion

This branch should remain a Track A goal-based baseline, not the main Track A
classifier and not Track B.

The model is useful because it adds interpretable expected goals, modal
scorelines, and full scoreline-derived H/D/A probabilities using only clean
football-data.co.uk match results. However, as a standalone H/D/A predictor it
does not outperform either the Bet365 closing market or the existing Track A
classification models.

| Model | H/D/A Accuracy | Macro F1 | Draw F1 | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|
| Loukas-style Poisson | 42.00% | 0.2974 | 0.0000 | 1.0945 | 0.2213 |
| Bet365 closing | 49.38% | 0.3653 | 0.0006 | 1.0189 | 0.2036 |

Goal prediction is more encouraging than result classification:

| Metric | Value |
|---|---:|
| Home goals MAE vs lambda | 1.0508 |
| Away goals MAE vs lambda | 0.9457 |
| Total goals MAE vs lambda total | 1.4030 |
| Exact score from modal score | 11.80% |
| Home goals within +/-1 | 80.57% |
| Away goals within +/-1 | 86.13% |

Compared with Loukas et al. (2024), the same broad pattern appears: most goal
estimates are close, especially within +/-1 goal. The project conclusion is more
conservative because this branch uses season-by-season walk-forward validation
instead of same-season random sampling.

## Method

The model follows the independent double-Poisson structure from Loukas et al.
(2024):

```text
log(lambda_home) = mu + mu_home + attack_home + defence_away
log(lambda_away) = mu + attack_away + defence_home
```

The scoreline matrix is converted to H/D/A probabilities by summing all home-win,
draw, and away-win score probabilities.

## Validation

The evaluation uses the Track A season-by-season walk-forward protocol:

- train on all seasons before the test season
- test on the next season
- repeat through the available English4 data

This avoids the same-season leakage risk in random-sample validation.

## Run

From the repository root:

```powershell
python scripts/tracka_poisson_goals.py
```

Generated outputs are ignored by Git.
