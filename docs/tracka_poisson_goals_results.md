# Track A Poisson Goals Baseline Results

## Scope

- Branch: `codex/tracka-poisson-goals`.
- Main-line parent: `codex/atta-mills-tracka-english4`.
- Data: English E0-E3 football-data.co.uk files.
- Seasons: 2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25, 2025-26.
- Matches: 13988.
- Model: Loukas et al. (2024)-style independent double Poisson.
- Inputs: historical full-time goals only.
- Validation: season-by-season walk-forward.
- Exclusions: no odds as model inputs, no half-time/in-play variables, no current-match raw statistics, no FootyStats, no O/U 2.5.

## Method

For each test season, the model is fitted only on previous seasons. It estimates
a pooled away-goal baseline, a pooled home advantage, and team attack/defence
parameters from historical goals scored and conceded.

```text
log(lambda_home) = mu + mu_home + attack_home + defence_away
log(lambda_away) = mu + attack_away + defence_home
```

A scoreline matrix is generated from two independent Poisson distributions.
H/D/A probabilities are then aggregated from the scoreline matrix:

```text
P(H) = sum P(home_goals > away_goals)
P(D) = sum P(home_goals = away_goals)
P(A) = sum P(home_goals < away_goals)
```

This keeps the branch as a Track A clean goal-based baseline, not a Track B
FootyStats-enhanced model.

## Original Paper Comparison

Loukas et al. (2024) evaluate the model on the 2022/23 English Premier League
season only:

- 380 matches.
- 1084 goals.
- 1.63 average home goals.
- 1.22 average away goals.
- 48.4% home wins, 22.9% draws, and 28.7% away wins.

Their validation has three parts. First, they estimate parameters from all 380
matches and inspect a random 20% sample of 76 matches. In that sample, rounded
goal differences within +/-1 are 75.0% for home goals and 90.8% for away goals.
Second, they simulate full-season goals for Manchester City, Fulham, and
Southampton using the same-season parameters. Third, they run a cleaner
independent check by estimating parameters from the first 19 fixtures and
predicting fixtures 20-24, a 50-match sample. In that independent sample,
rounded goal differences within +/-1 are 72.0% for home goals and 76.0% for
away goals.

This branch uses a stricter project-level validation:

- 13,988 English E0-E3 matches.
- 12,216 walk-forward test matches.
- train only on seasons before the test season.
- no same-season random sampling.

Using the paper's rounded-difference style, this branch achieves 73.3% home
goals within +/-1 and 81.3% away goals within +/-1 across all walk-forward test
matches. This is less optimistic than the paper's random 76-match same-season
check, but broadly consistent with its independent 50-match check.

## Aggregate H/D/A Results

| Model | Accuracy | Macro F1 | Draw F1 | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|
| Loukas-style Poisson | 0.4200 | 0.2974 | 0.0000 | 1.0945 | 0.2213 |
| Bet365 closing | 0.4938 | 0.3653 | 0.0006 | 1.0189 | 0.2036 |

## Aggregate Goal Results

| Metric | Value |
|---|---:|
| Home goals MAE vs lambda | 1.0508 |
| Away goals MAE vs lambda | 0.9457 |
| Total goals MAE vs lambda total | 1.4030 |
| Exact home goals from modal score | 0.3322 |
| Exact away goals from modal score | 0.3518 |
| Exact score from modal score | 0.1180 |
| Home goals within +/-1 | 0.8057 |
| Away goals within +/-1 | 0.8613 |

## Fold H/D/A Results

| Fold | Model | Accuracy | Macro F1 | Draw F1 | Log Loss | Brier |
|---|---|---:|---:|---:|---:|---:|
| 2020-21 | loukas_poisson | 0.3865 | 0.2849 | 0.0000 | 1.1608 | 0.2360 |
| 2020-21 | bet365_closing | 0.4725 | 0.3549 | 0.0000 | 1.0378 | 0.2079 |
| 2021-22 | loukas_poisson | 0.4099 | 0.2957 | 0.0000 | 1.0883 | 0.2201 |
| 2021-22 | bet365_closing | 0.5089 | 0.3770 | 0.0000 | 1.0073 | 0.2006 |
| 2022-23 | loukas_poisson | 0.4118 | 0.2921 | 0.0000 | 1.1002 | 0.2230 |
| 2022-23 | bet365_closing | 0.4868 | 0.3591 | 0.0000 | 1.0235 | 0.2050 |
| 2023-24 | loukas_poisson | 0.4489 | 0.3132 | 0.0000 | 1.0583 | 0.2131 |
| 2023-24 | bet365_closing | 0.5088 | 0.3708 | 0.0042 | 1.0018 | 0.1994 |
| 2024-25 | loukas_poisson | 0.4209 | 0.2882 | 0.0000 | 1.0842 | 0.2190 |
| 2024-25 | bet365_closing | 0.4926 | 0.3654 | 0.0000 | 1.0200 | 0.2039 |
| 2025-26 | loukas_poisson | 0.4420 | 0.3043 | 0.0000 | 1.0754 | 0.2168 |
| 2025-26 | bet365_closing | 0.4926 | 0.3625 | 0.0000 | 1.0233 | 0.2049 |

## Fold Goal Results

| Fold | Home MAE | Away MAE | Total MAE | Home Exact | Away Exact | Exact Score | Home +/-1 | Away +/-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020-21 | 1.1464 | 0.9769 | 1.5252 | 0.3183 | 0.3448 | 0.1071 | 0.7667 | 0.8369 |
| 2021-22 | 0.9986 | 0.9483 | 1.3413 | 0.3423 | 0.3409 | 0.1188 | 0.8199 | 0.8576 |
| 2022-23 | 1.0636 | 0.9660 | 1.4516 | 0.3328 | 0.3338 | 0.1129 | 0.8156 | 0.8631 |
| 2023-24 | 1.0271 | 0.9413 | 1.3400 | 0.3330 | 0.3531 | 0.1213 | 0.7967 | 0.8492 |
| 2024-25 | 1.0530 | 0.9277 | 1.4064 | 0.3217 | 0.3679 | 0.1243 | 0.8134 | 0.8841 |
| 2025-26 | 1.0173 | 0.9141 | 1.3556 | 0.3448 | 0.3703 | 0.1238 | 0.8222 | 0.8772 |
| ALL | 1.0508 | 0.9457 | 1.4030 | 0.3322 | 0.3518 | 0.1180 | 0.8057 | 0.8613 |

## Interpretation

This first run intentionally stays close to Loukas et al.'s simple structure.
It is stricter than the paper's random-sample validation because each fold only
uses seasons before the test season, avoiding same-season leakage.

The branch should remain under Track A as a clean goal-based baseline. It should
not replace the main Track A classifier and should not be labelled as Track B,
because Track B is reserved for FootyStats-enhanced modelling.

The result is mixed. The model is useful for interpretable expected goals,
modal scorelines, and full scoreline-derived H/D/A probabilities. The goal
prediction results are close enough to support the paper's broad claim that a
simple Poisson model can estimate many team goal counts within one goal.
However, the H/D/A comparison is clearly negative: the Poisson model reaches
42.00% accuracy, below the Bet365 closing benchmark at 49.38% and below the
existing Track A classifiers.

The zero Draw F1 confirms the known limitation of independent Poisson models:
draw probabilities are non-zero, but they rarely become the argmax prediction.
Loukas et al. (2024) do not report Draw F1, so this is an additional
classification-oriented diagnostic from this project rather than a direct
paper-to-project comparison.

Final decision: keep this branch as a documented Track A Poisson goals baseline.
A later experiment can test Dixon-Coles correction or Poisson probabilities as
features for the Track A classifier, but those should be separate follow-up
extensions.

## Outputs

- `scripts/tracka_poisson_goals.py`
- `docs/tracka_poisson_goals_results.md`
- `outputs/tracka_poisson_goals/fold_metrics.csv`
- `outputs/tracka_poisson_goals/aggregate_metrics.csv`
- `outputs/tracka_poisson_goals/goal_metrics.csv`
- `outputs/tracka_poisson_goals/predictions_by_fold.csv`
- `outputs/tracka_poisson_goals/poisson_parameters_by_fold.csv`
