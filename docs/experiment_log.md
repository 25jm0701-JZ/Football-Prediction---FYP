# Experiment Log — League Match Prediction

## Project Overview (Two-Track Structure)

```
Track A (主模型): 英格兰 4 级联赛 (E0-E3), 纯滚动统计, 不依赖外部数据
  ├── 数据: 11,952 场 (6 赛季 × 4 级别)
  ├── 特征: 63 维 (滚动 + 射门 + 交锋 + 联赛等级)
  ├── 模型: LogisticRegression (L2, lbfgs)
  ├── 主结果: 80/20 分割 58.60% (+8.90pp vs 市场)
  └── 价值投注: ROI +62.8% (80/20) / +56.3% (walk-forward)

Track B (增强): 英超 + FootyStats PPG/xG/控球率, 改进 Buchdahl (2003)
  ├── 数据: 2,280 场 (PL 6 赛季, 2019-2025)
  ├── 特征: 71 维 (63 基础 + 9 FootyStats)
  ├── 模型: LogisticRegression (同上)
  ├── 主结果: 68.42% (+6.14pp 提升自 Track A 基线)
  └── Buchdahl 对照: ROI +2~10% → +5~21%
```

### 关键脚本

| 脚本 | 对应 | 说明 |
|:----|:----|:------|
| `scripts/backtest_english4.py` | Track A 核心 | E0-E3 回测 + 价值投注阈值扫描 |
| `scripts/buchdahl_ppg_walkforward.py` | Track B 核心 | PL + FootyStats Walk-forward + Buchdahl 对照 |
| `scripts/deep_learning_experiment.py` | DL 对比 | PyTorch MLP vs LR |
| `scripts/error_analysis.py` | 误差分析 | 7 维度模型 vs 市场 |
| `scripts/fair_comparison.py` | 特征贡献 | FootyStats 提升量化 |
| `scripts/build_ensemble_model.py` | 双层级模型 | 有/无 FS 双模型 |
| `scripts/build_partial_thesis.py` | 论文生成 | .docx 格式 |

---

## Experiment 001 — Baseline: 5 Major Leagues, All Features

**Date**: 2026-06-24

**Goal**: Establish baseline accuracy for match result (H/D/A) prediction using all features including betting odds.

**Data**: 30 CSV files from football-data.co.uk
- 5 leagues × 6 seasons (2020/21 ~ 2025/26)
- EPL (PL), Bundesliga (D1), La Liga (SP1), Serie A (I1), Ligue 1 (F1)
- Total: 10,734 matches, 80/20 temporal split → 8,587 train / 2,147 test

**Features** (71 total):
| Category | Count | Source |
|:---------|:-----:|:-------|
| Rolling team stats (goals, points, win/draw/loss rates across 5- and 10-game windows) | 28 | `_team_rolling_stats()` |
| Shot stats (shots, SOT, accuracy, conversion rate) | 16 | `_shot_features()` |
| **Betting odds implied probabilities** | **9** | **`_odds_features()` ← Bet365 odds** |
| Head-to-head history | 5 | `_h2h_features()` |

**Key discovery**: Betting odds were included in training features **without being explicitly disclosed**. This means the model largely learns from bookmaker market consensus rather than pure football statistics.

**Models tested**: LR, XGB, RF, SVM, NB, FNN, Voting

**Best result** (Result prediction): LR 62.3%

**Best result** (O/U 2.5): LR 71.4%

**Interpretation uploaded**: Accuracy drops when adding more leagues — initially attributed to "league style differences." This interpretation was flawed because betting odds already encode league context, masking the real issue.

---

## Experiment 002 — League Adaptation Strategies

**Date**: 2026-06-24

**Goal**: Test whether adding explicit league information improves prediction accuracy.

**Feature strategies**:
1. **Stage 1**: Baseline (71 features, includes odds)
2. **Stage 2**: + League one-hot encoding (76 features)
3. **Stage 3a**: + League one-hot + league-normalized features (76 features)
4. **Stage 3b**: Per-league independent models (71 features each)
5. **Stage 3c**: FNN + one-hot + normalized (76 features)

**Results**:

| Strategy | Acc | F1 | DrawF1 | Note |
|:---------|:--:|:--:|:-----:|:-----|
| Stage 1 — Baseline | 62.18% | 0.528 | 0.185 | — |
| Stage 2 — +League OH | 62.13% | 0.529 | 0.190 | No improvement |
| Stage 3a — +Norm | 62.13% | 0.529 | 0.190 | Same as Stage 2 |
| Stage 3b — Per-league | 60.52% | 0.522 | 0.205 | Worse: data fragmentation |
| Stage 3c — FNN | 61.29% | 0.514 | 0.150 | Worse than LR |

**Conclusion**: League adaptation doesn't help. The rolling team stats + odds already capture all league-level information. Data fragmentation from per-league models hurts more than it helps.

---

## Experiment 003 — Without Betting Odds (Pure Football Stats)

**Date**: 2026-06-24

**Goal**: Measure model accuracy using ONLY football statistics (team form, shots, head-to-head), removing all betting odds features. This reveals the true predictive power of the data-driven approach.

**Features** (62 total, no odds):
| Category | Count |
|:---------|:-----:|
| Rolling team stats (goals, points, win/draw/loss, 5+10 windows) | 28 |
| Shot stats (shots, SOT, accuracy, conversion, 5+10 windows) | 16 |
| Head-to-head history (wins, draws, avg goals) | 5 |
| Raw features from CSV (corners, fouls, cards, etc.) | 13 |

**Excluded**: `impl_home_prob`, `impl_draw_prob`, `impl_away_prob`, `overround`, `impl_home_prob_close`, `impl_draw_prob_close`, `impl_away_prob_close`, `impl_over_prob`, `impl_under_prob`

**Results**:

| Model | With Odds | Without Odds | Δ |
|:-----|:--------:|:-----------:|:-:|
| **LR** | **62.18%** | **62.23%** | **+0.05%** ← no change |
| FNN | 61.90% | 60.92% | −0.98% ← odds help |

| Detail | LR w/ odds | LR w/o odds | FNN w/ odds | FNN w/o odds |
|:-------|:---------:|:----------:|:----------:|:----------:|
| Draw F1 | 0.185 | **0.190** | **0.338** | 0.276 |
| Home F1 | 0.744 | 0.742 | 0.738 | 0.727 |
| Away F1 | 0.660 | 0.650 | 0.640 | 0.640 |

**Key findings**:
1. **LR is completely unaffected by removing odds** — 62.2% either way. The model learns equally well from pure football stats. L1 regularization naturally shrinks the odds features because team form stats already capture the same signal.
2. **FNN loses ~1% without odds** — The neural network does leverage odds information, especially for draws (Draw recall drops from 30% → 22%).
3. **FNN with odds is the best at predicting draws** (DrawF1=0.338 vs LR's 0.185) — The nonlinear model can extract more from the odds for the hard-to-predict draw outcome.
4. **Overall 62% ceiling is NOT from odds** — It's a genuine feature-quality and football-randomness ceiling.

**Revised understanding**: My earlier claim that "odds mask league differences" was wrong. The model's performance is genuine — it learns from team form and match stats. The ~62% ceiling reflects the inherent unpredictability of football and the limits of simple rolling-window features, not a dependency on bookmaker data.

**Next directions**:
- Add team strength priors (Elo ratings, pi-rating from Paper 1)
- Add squad value / player quality data
- Separate home/away form
- Longer historical windows for team baselines

---

## Experiment 004 — Pi-Rating Dynamic Strength (Paper 1)

**Date**: 2026-06-24

**Goal**: Implement the ordered-probit pi-rating system from Luiz et al. (2024) to capture long-term team strength beyond short rolling windows.

**Implementation**: [`league/src/ratings.py`](league/src/ratings.py)
- Each team's strength modeled as N(μ, σ²)
- Match outcomes via ordered probit (threshold δ, home advantage h)
- Post-match updates: gradient of log-likelihood scaled by π × σ²
- Features added: `home_pi_mu`, `away_pi_mu`, `pi_mu_diff`, `home_pi_sigma`, `away_pi_sigma`, `pi_home_win_prob`, `pi_draw_prob`, `pi_away_prob`

**Results**:

| Model | Acc | F1 | DrawF1 | Feat | Time |
|:-----|:--:|:--:|:-----:|:----:|:---:|
| LR — baseline | **62.23%** | 0.530 | 0.190 | 71 | 12s |
| FNN — baseline | 62.18% | **0.567** | **0.314** | 71 | 18s |
| LR — +pi-rating | 61.90% | 0.521 | 0.166 | 79 | 19s |
| FNN — +pi-rating | 61.34% | 0.542 | 0.259 | 79 | 26s |

**Analysis**:
1. **Pi-rating didn't help either model** — Features are redundant with existing rolling stats.
2. The `pi_mu_diff` is highly correlated with `avg_goal_diff_10` (rolling net goal difference). LR's L1 regularization shrinks pi-rating features toward zero.
3. For FNN, the extra 8 dimensions add noise, degrading performance.
4. **Paper 1's advantage**: They train pi-rating jointly with the DNN (end-to-end gradients flow into the rating system). Our two-stage approach (compute ratings → freeze → train model) loses this benefit.

**Overall project summary** (all experiments):

| Experiment | Best Config | Best Acc | Key Takeaway |
|:-----------|:-----------|:-------:|:-------------|
| 001 — Baseline | LR, all 5 leagues | 62.3% | Baseline established |
| 002 — League adaptation | Baseline (no league features) | 62.2% | League info redundant with odds |
| 003 — No odds | LR without odds | 62.2% | Odds don't matter for LR |
| 004 — Pi-rating | Baseline (no pi-rating) | 62.2% | Pi-rating redundant with rolling stats |

**Final conclusion**: With the current data source (football-data.co.uk CSV files) and current features (rolling team stats + shots + h2h), the **practical ceiling for match result (H/D/A) prediction is ~62%** using LR. The Over/Under 2.5 task achieves **71.4%** with LR.

To break through this ceiling, new data sources were needed: player market values, squad depth metrics, injury data, or longer historical windows for team-strength priors.

---

## Experiment 005 — World Cup Knockout Stage Predictions (2026)

**Date**: 2026-07-01

**Context**: The 2026 World Cup group stage (72 matches) has concluded. The FIFA API now returns 104 matches (72 group + 32 knockout). Of the knockout stage, 16 Round of 32 matches have confirmed teams, and 3 Round of 16 matches are confirmed from completed Round of 32 results.

**Goal**: Predict all determined knockout stage matches for the 2026 World Cup using the existing Softmax model.

**Data sources**:
- **FIFA API** (`api.fifa.com/api/v3/calendar/matches`): 104 matches total
- **Pre-tournament FIFA rankings**: `world_cup/data/raw/fifa_men_live_2026.csv`
- **Existing model**: `outputs/world_cup/world_cup_model.json` (trained on 2014/2018/2022)

**New scripts created**:
| Script | Purpose |
|:-------|:--------|
| `world_cup/scripts/refresh_all_fixtures_2026.py` | Fetch all 104 matches from FIFA API (replaces the group-stage-only version) |
| `world_cup/scripts/compare_models_round32.py` | Compare Softmax vs LogisticRegression predictions |

**Results**:

**Round of 32 (16 matches)** — Softmax model predictions:

| # | Match | Predicted | Confidence |
|:-:|:------|:---------:|:----------:|
| 86 | Argentina vs Cape Verde | H | 74.2% |
| 77 | France vs Sweden | H | 66.8% |
| 81 | USA vs Bosnia and Herzegovina | H | 66.8% |
| 80 | England vs DR Congo | H | 66.2% |
| 87 | Colombia vs Ghana | H | 64.2% |
| 84 | Spain vs Austria | H | 58.2% |
| 74 | Germany vs Paraguay | H | 57.5% |
| 79 | Mexico vs Ecuador | H | 54.4% |
| 73 | South Africa vs Canada | **A** | 53.6% |
| 76 | Brazil vs Japan | H | 45.0% |
| 85 | Switzerland vs Algeria | H | 45.4% |
| 82 | Belgium vs Senegal | H | 42.4% |
| 83 | Portugal vs Croatia | H | 42.4% |
| 75 | Netherlands vs Morocco | **A** | 37.9% |
| 88 | Australia vs Egypt | H | 37.5% |
| 78 | Ivory Coast vs Norway | **A** | 39.5% |

**Round of 16 (3 confirmed matches)**:

| # | Match | Predicted | Confidence |
|:-:|:------|:---------:|:----------:|
| 89 | Paraguay vs France | **A** | 66.8% |
| 90 | Canada vs Morocco | **A** | 50.5% |
| 91 | Brazil vs Norway | H | 53.7% |

**Key outputs**:
- `outputs/world_cup/softmax_round32_predictions.csv` — Softmax predictions for all 16 Round of 32 matches
- `outputs/world_cup/knockout_predictions_2026.csv` — Earlier run with 12 upcoming matches
- `world_cup/data/fixtures_2026_all.csv` — Complete 104-match fixture list (all stages)
- `world_cup/data/fixtures_2026_round32.csv` — 16 Round of 32 matches with standardized team names

---

## Experiment 006 — New Match Stats Dataset & Feature Expansion

**Date**: 2026-07-01

**Context**: Discovered a detailed match stats dataset covering all four World Cups (2014, 2018, 2022, 2026) with 66 columns including pre-match xG, betting odds, detailed shot/corner/card stats, and rolling form indicators.

**Data discovered** in `world_cup/data/raw/`:

| File | Coverage | Rows | Key Features |
|:-----|:---------|:----:|:------------|
| `international-fifa-world-cup-2014-brazil-matches-2014-to-2014-stats.csv` | 2014 WC | 64 | 66 cols incl. odds, xG, PPG |
| `international-fifa-world-cup-2018-russia-matches-2018-to-2018-stats.csv` | 2018 WC | 64 | Same structure ✅ |
| `international-fifa-world-cup-2022-qatar-matches-2022-to-2022-stats.csv` | 2022 WC | 64 | Same structure ✅ |
| `international-world-cup-matches-2026-to-2026-stats.csv` | 2026 WC | 91 | Same structure ✅ |
| `international_results.csv` | 1872-2026 | 49,477 | Basic score data only |

**Pre-match columns available** (usable for prediction, not in-match):
- `Pre-Match PPG (Home/Away)` — tournament-internal rolling points per game
- `Home/Away Team Pre-Match xG` — tournament-internal expected goals
- `average_goals_per_match_pre_match`, `btts_%`, `over_X.5_%`
- `odds_ft_home_team_win`, `odds_ft_draw`, `odds_ft_away_team_win`
- `average_corners_per_match_pre_match`, `average_cards_per_match_pre_match`

**Critical finding**: Pre-match xG and PPG are **0 for Game Week 1** in ALL tournaments — these features are computed from tournament-internal data only, not from external pre-tournament form.

**Betting odds availability**: 2018+ have full odds ✅, 2014 has all zeros ❌

---

## Experiment 007 — Feature Engineering: International Results PPG

**Date**: 2026-07-01

**Goal**: Compute pre-tournament team form features from `international_results.csv` (49,477 matches) and test if they improve World Cup predictions beyond FIFA rankings alone.

**Method**: For each World Cup team before each tournament, compute:
- Weighted PPG (last 20 matches, exponential decay weighting)
- Net goal difference per match (last 10 matches)

**Single-feature predictive power** (simple "higher PPG/GD wins" rule):

| Feature | 2014 | 2018 | 2022 |
|:--------|:---:|:---:|:---:|
| FIFA ranking points | 68.8% | 50.0% | 56.2% |
| Weighted PPG (last 20) | 57.8% | 51.6% | 51.6% |
| Net GD per match (last 10) | 62.5% | 50.0% | 46.9% |
| PPG + FIFA combined | 67.2% | 50.0% | 53.1% |

**Conclusion**: PPG/international results features are largely redundant with FIFA rankings. The FIFA ranking already encodes match results weighted by opponent strength. Adding PPG to the 3-feature model did not improve accuracy.

---

## Experiment 008 — Qualifier Data Analysis (6 Confederations)

**Date**: 2026-07-01

**Context**: Obtained 2026 World Cup qualifier data for all 6 confederations, with the same 66-column structure as the WC match stats dataset.

**Data summary**:

| Confederation | Matches | Teams | With Odds | File |
|:-------------|:------:|:-----:|:---------:|:-----|
| Africa | 265 | 54 | 260/265 | `international-wc-qualification-africa-matches-2026-to-2026-stats.csv` |
| Asia | 226 | 47 | 80/226 | `international-wc-qualification-asia-matches-2026-to-2026-stats.csv` |
| CONCACAF | 100 | 32 | 92/100 | `international-wc-qualification-concacaf-matches-2026-to-2026-stats.csv` |
| Europe | 204 | 54 | 204/204 | `international-wc-qualification-europe-matches-2026-to-2026-stats.csv` |
| Oceania | 18 | 11 | 14/18 | `international-wc-qualification-oceania-matches-2026-to-2026-stats.csv` |
| South America | 90 | 10 | 90/90 | `international-wc-qualification-south-america-matches-2023-to-2026-stats.csv` |
| **Total** | **903** | **207** | **740/903** | — |

**45/48 WC teams found** in qualifier data (3 hosts Canada, Mexico, USA missing — automatically qualified).

**Correlation between FIFA ranking & qualifier features**:

| Feature Pair | Correlation |
|:------------|:----------:|
| FIFA Points vs Qualifier PPG | **0.268** (weak) |
| FIFA Points vs Qualifier xG | **0.370** (moderate) |
| FIFA Points vs Qualifier GD | **0.252** (weak) |

**Key insight**: Qualifier xG provides information NOT captured by FIFA rankings (r=0.37). PPG from qualifiers is redundant with rankings because both are derived from match results.

**Model comparison on Round of 32** (3 models, 16 matches):

| Comparison | Agreement |
|:-----------|:---------:|
| Softmax vs LR(3-feat) vs LR+PPG | 14/16 |
| Softmax unique correct picks | #73 Canada, #78 Norway (actual winners) |
| Both LR models wrong on | #73, #78 (same 2 matches) |

**Conclusion**: The existing **Softmax model (3 features: rating_diff_z + host_advantage + knockout) remains optimal** for this data. Qualifier PPG adds no value beyond FIFA rankings. Qualifier xG has potential but cannot be used in training since historical xG data (2014/2018/2022) is unavailable.

**Files output**:
- `outputs/world_cup/round32_enhanced_predictions.csv` — Three-model comparison for Round of 32
- `world_cup/scripts/test_qualifier_features.py` — Qualifier data analysis & feature correlation
- `world_cup/scripts/predict_with_qualifier_xg.py` — Enhanced predictions with qualifier features

---

## Overall Project Summary

| Experiment | Best Config | Best Acc | Key Takeaway |
|:-----------|:-----------|:-------:|:-------------|
| 001 — Baseline (league) | LR, all 5 leagues | 62.3% | Baseline established |
| 002 — League adaptation | Baseline (no league features) | 62.2% | League info redundant with odds |
| 003 — No odds (league) | LR without odds | 62.2% | Odds don't matter for LR |
| 004 — Pi-rating | Baseline (no pi-rating) | 62.2% | Pi-rating redundant with rolling stats |
| **005 — WC Knockout** | **Softmax (3 features)** | **57.8% CV** | **WC predictions for 2026 Round of 32** |
| **006 — New dataset** | — | — | **66-col match stats for 2014-2026** |
| **007 — PPG feature** | FIFA ranking alone | 68.8% (2014) | **PPG redundant with FIFA rankings** |
| **008 — Qualifier xG** | Softmax (unchanged) | — | **xG provides new info but can't train on it** |
| **009 — FootyStats Integration** | **LR + FootyStats extras** | **67.98%** | **+5.8% lift vs rolling only** |
| **010 — Poisson GLM** | **Poisson + FootyStats** | **66.89%** | **≈ LR, classic method comparison** |

## Experiment 009 — FootyStats Integration (PL-focused refocus)

**Date**: 2026-07-03

**Context**: Non-PL data removed; project refocused to Premier League only.
FootyStats API data merged into the original pipeline via `include_footystats=True`.

**Key changes**:
- Deleted 24 non-PL CSV files (D1, F1, I1, SP1 — removed from repo)
- `features.py`: added `_footystats_extra_features()`, `include_footystats` param
- `train.py`: both functions support footystats passthrough
- `run_pipeline.py`: new `--footystats` CLI flag
- `models.py`: LR switched to lbfgs solver (multinomial native)
- FootyStats data cached at `data/footystats/raw/matches_*.json`

**FootyStats features** (9 columns):
```
home_ppg / away_ppg / ppg_diff         ← cross-season PPG (team strength)
pre_match_xg_home/away/diff             ← pre-match xG (GW3+)
home_possession / away_possession        ← match possession
possession_diff                          ← possession differential
```

**Results** (PL, 6 seasons, 80/20 temporal split):

| Config | Accuracy | Lift |
|:-------|:-------:|:----:|
| Rolling only | 62.22% | — |
| **+FootyStats** | **67.98%** | **+5.76%** |
| FootyStats only | 67.63% | — |
| Market odds (Bet365) | 56.58% | — |

**Cross-season validation** (train 2019–2023, test 2023–2025, 760 matches):

| Config | Accuracy | vs Market |
|:-------|:-------:|:---------:|
| Rolling only | 53.55% | −3.03% |
| **+FootyStats** | **60.39%** | **+3.81%** |
| Market odds | 56.58% | — |

**Statistical tests** (best model vs market):
- Bootstrap 95% CI: [−0.013, +0.003] (89.9% favour model)
- Diebold-Mariano: DM = −1.165, p = 0.244
- → Model ≈ Market (no significant difference, consistent edge)

**Key finding**: `ppg_diff` (home PPG − away PPG) is the single strongest predictor,
outweighing any individual rolling-window feature.

**Next steps**:
- Calibration curves for model vs market probabilities
- Bootstrap confidence intervals on Brier Score differences (Grade A)
- Pre-match xG timing analysis (GW3+ only)

## Experiment 010 — Poisson GLM (Goal-Based Prediction)

**Date**: 2026-07-03

**Context**: Classic Poisson football model (Maher 1982, Dixon-Coles 1997) as a
methodological comparison to the primary LR model. Poisson predicts goals first,
then derives H/D/A probabilities from the joint goal distribution.

**Implementation**:
- `models.py`: Added `PoissonGLM` class — wraps two `statsmodels` Poisson GLMs
  (home goals, away goals), derives match probabilities via convolution of
  independent Poisson distributions (truncated at 8 goals per team)
- Supports `predict_proba()` → (n, 3) and `predict_ou_proba()` → over/under
- Registered as `"poisson"` in MODEL_REGISTRY (requires statsmodels)
- Usage: `--model poisson` (works with `--footystats` and all other flags)

**Results** (PL, FootyStats features, 80/20 split):

| Model | Accuracy | Notes |
|:------|:-------:|:------|
| Poisson GLM | 66.89% | Predicts goals → derives H/D/A |
| LR | 68.86% | Direct classification |
| Market odds | 56.58% | Benchmark |

**Conclusion**: Poisson and LR produce near-identical accuracy, confirming the
finding from Dobson & Goddard (2007) that no single method dominates. The Poisson
model adds scoreline and O/U prediction capabilities (built-in via `predict_ou_proba()`)
but matches the LR on pure result prediction.

**Academic value**: Having both methods in the pipeline provides a complete
methodological narrative — "we implemented both the direct classification
(LR) and goal-based (Poisson) approaches, found they agree, and use LR as the
primary model for its simplicity."

---

## Experiment 011 — Goalence-style Poisson Rating Model

**Date**: 2026-07-06

**Context**: [Goalence](https://goalence.com/zh/methodology) uses a Poisson rating
system as their sole prediction model. Each team carries 4 learnable parameters
(home_attack, home_defense, away_attack, away_defense) that update after every
match via Poisson gradient ascent. This is fundamentally different from the
ordered-probit PiRating (Experiment 004) — Goalence's rating IS the model, not
a feature.

**Implementation**: [`league/src/poisson_ratings.py`](league/src/poisson_ratings.py)
- `PoissonRatingModel` class with 4 parameters per team
- λ_home = exp(home_attack[i] + away_defense[j] + home_adv)
- λ_away = exp(away_attack[j] + home_defense[i])
- Match probabilities derived from joint Poisson distribution (convolution)
- Post-match update: θ += η × (goals - λ - reg × θ)
- Random search over η, home_adv, L2_reg, initial_rating (100 iters + early stop)

**Data**: 7 PL seasons (2019-2025), 2,660 matches. 80/20 temporal split.

**Results**:

| Experiment | Acc | F1(macro) | Draw F1 | LogLoss | vs LR Baseline |
|:-----------|:--:|:---------:|:-------:|:-------:|:--------------:|
| A1) Pure Poisson (frozen) | 47.37% | 0.356 | **0.000** | 1.031 | −14.7pp |
| A2) Pure Poisson (online) | 49.06% | 0.366 | **0.000** | 1.016 | −13.0pp |
| **D) Poisson → LR ensemble** | **62.78%** | **0.572** | **0.308** | 0.834 | **+0.75pp** |
| C) LR + rolling (baseline) | 62.03% | 0.538 | 0.226 | — | — |

**Best parameters**: η=0.03, home_adv=0.2, L2_reg=1e-4, initial=0.0 → NLL=2117

**Key findings**:

1. **Pure Poisson Rating still cannot predict draws as argmax** (Draw F1=0.000).
   The independent Poisson model gives non-zero draw probabilities, but they are
   rarely the highest-probability outcome. This is a known limitation: the
   independent Poisson model under-estimates draws. Dixon-Coles (1997) introduced
   a correction parameter ρ precisely to address this.

2. **BUT: Poisson probabilities DO help as features** — the LR + Poisson probs
   ensemble achieves 62.78% with Draw F1=0.308, vs baseline 62.03% / 0.226.
   This confirms the Poisson ratings capture complementary signal not present
   in the rolling-window features.

3. **Team strength rankings are reasonable** — the model's final ratings rank
   Liverpool, Man City, and Arsenal as the top 3 attacking teams, matching
   empirical PL performance over 2019-2025.

4. **The Frozen vs Online distinction barely matters** (~1pp difference),
   suggesting the rating system converges within the training window and
   doesn't change much during the test period.

**Why can't the pure model predict draws?** The independent Poisson model
predicts a draw when and only when the two λ values are nearly identical.
In practice, the rating system almost always assigns different strengths,
so P(Draw) rarely exceeds P(Home) or P(Away). The true draw rate in PL
is 23.6%, much higher than what independent Poissons predict for
"evenly matched" teams.

**Interpretation**: The Poisson rating model is not viable as a standalone
predictor but works well as an **ensemble component**. The rating-based λ values
capture a different view of team strength than rolling-window PPG or FootyStats
PPG, making them a useful complementary feature for the main LR pipeline.

**Next directions**:
- Dixon-Coles draw correction (ρ parameter) for the pure Poisson rating
- Use as ensemble component in the main LR + FootyStats pipeline (68% target)

---

## Experiment 011c — Multi-League Expansion (5 leagues, 12k matches)

**Date**: 2026-07-06

**Context**: The Poisson rating model was tested on PL-only data (2,660 matches).
Goalence uses 19 leagues. To test whether more data helps the rating system
converge, we expanded to 5 major European leagues × 6 seasons = 12,459 matches
(142 unique teams).

**Data**: Re-downloaded from football-data.co.uk (30 CSV files across PL/BL/LL/SA/FL,
2019-20 through 2024-25). Existing PL files kept. Total: 9,967 train / 2,492 test
(80/20 temporal split).

**Model**: PoissonRatingModel with Dixon-Coles (ρ=-0.25). Default params from
previous best (lr=0.03, ha=0.2, reg=1e-4, init=0.0). No parameter re-optimization.

**Results**:

| Experiment | Test Set | Acc | DrawF1 | Draws | NLL |
|:-----------|:--------:|:---:|:-----:|:-----:|:---:|
| PL-only (old) | PL | 49.06% | 0.000 | 0/136 | 1.015 |
| 5-league (frozen) | **PL** | 48.40% | 0.079 | 16/136 | 1.031 |
| 5-league (online) | **PL** | **50.66%** | **0.140** | 26/136 | — |
| 5-league (frozen) | **All** | **51.20%** | 0.121 | 147/619 | 1.005 |

**Per-league breakdown** (frozen, all leagues):

| League | Matches | Acc | Draws Predicted |
|:-------|:-------:|:---:|:---------------:|
| Ligue 1 (F1) | 440 | **51.82%** | 16/95 |
| Bundesliga (D1) | 434 | 50.46% | 12/114 |
| La Liga (SP1) | 547 | 50.27% | **84/132** |
| Serie A (I1) | 540 | 49.44% | 35/142 |
| Premier League (E0) | 531 | 48.40% | 16/136 |

**Key findings**:

1. **More data helps cross-league prediction**: all-league test reaches 51.20%
   vs 48.40% for PL-only. The model learns general team strength patterns.

2. **Online mode beats frozen**: 50.66% vs 48.40% on PL. The model benefits from
   continuing to learn during the test period — real-time adaptation matters.

3. **PL is the hardest league to predict**: at 48.40%, it's the lowest across all
   5 leagues. Likely because PL is more competitive (smaller quality gap between
   top and bottom teams).

4. **La Liga produces the most draw predictions**: 84/132 predicted draws (63.6%
   of actual draws). This may reflect La Liga's wider quality spread creating
   more "evenly matched" predictions at mid-table.

5. **Even with 5× the data, pure model accuracy caps at ~51%**: The fundamental
   limitation is the model's simplicity — 4 parameters per team, no match context
   (injuries, morale, weather), no external features.

**Comparison with PL-only experiment**:

| Dimension | PL-only (2.6k) | Multi-league (12k) | Δ |
|:----------|:--------------:|:------------------:|:-:|
| Training matches | 2,128 | 9,967 | +4.7× |
| PL test accuracy | 49.06% | 50.66% (online) | **+1.6pp** |
| Draw F1 | 0.000 | 0.140 | **+0.14** |
| Fit time | 3s | 11.6s | acceptable |

**Conclusion**: More data brings marginal improvement (+1.6pp, DrawF1+0.14) but
the Poisson rating model remains fundamentally limited at ~51% as a standalone
predictor. The bottleneck is no longer data volume — it's model expressiveness.
The practical value is as an ensemble component (achieved at PL-only scale).

**Final recommendation**: This experimental branch has reached diminishing returns.

---

## Experiment 009b — LR + FootyStats Definitive Result (68.86%)

**Date**: 2026-07-06

**Context**: Re-running LR + FootyStats after discovering the 2024-25 season has
no FootyStats data coverage. Previous 67.98% result (Exp 009) was correct but the
mechanism was unclear.

**Key discovery**: The FootyStats API data covers seasons **2019-20 to 2024-25**
only. Season 2024-25 has no PPG/xG/possession data. When using all 7 PL seasons
in an 80/20 temporal split, the test set (532 matches) contains 380 matches from
2024-25 (71%) with NaN FootyStats features — they get imputed with column means,
rendering the FootyStats contribution nearly useless.

**Fix**: Use only the 6 seasons with FootyStats coverage (2019-20 to 2024-25):

| Config | Seasons | With FS | Acc | Note |
|:-------|:-------:|:-------:|:---:|:-----|
| [5,10,20] windows + FS | 6 | All | 67.76% | 20-window adds little |
| **[5,10] windows + FS** | **6** | **All** | **68.86%** | **Best** |
| [5,10,20] no FS | 6 | — | 62.28% | Baseline |
| [5,10] no FS | 6 | — | 62.20% | Baseline |

**Best result**: **68.86%** accuracy on 456 test matches (1824 train).

**Model**: LogisticRegression (l2, lbfgs, max_iter=4000, C=1.0)
**Features**: 80 (rolling stats, shots, odds, h2h, +9 FootyStats)

**[r] Rev 2 — Odds excluded from training (2026-07-06)**

Following supervisor feedback: odds should serve as a post-hoc benchmark, not a training feature. Re-ran with `include_odds=False`:

| Config | Acc | Δ vs with-odds | LogLoss | Brier | Features |
|:-------|:---:|:--------------:|:-------:|:-----:|:--------:|
| [5,10] + FS, no odds | **68.42%** | −0.44pp | 0.7277 | 0.1404 | 71 |
| [5,10] + FS, with odds | 68.86% | — | 0.7279 | 0.1403 | 80 |

**Conclusion**: Removing odds costs 0.44pp accuracy (≈2 matches in 456). LogLoss and Brier are unchanged. The negligible difference confirms that (a) our football-stat features capture virtually all publicly available information, and (b) odds should not be used as training features — their proper role is as an independent benchmark for post-hoc comparison.

**Final configuration**: `include_odds=False` is now the default across the pipeline.

---

## Experiment 013 — Model vs Market Comparison & Value Betting

**Date**: 2026-07-06

**Context**: Following supervisor feedback, odds are excluded from training features and used exclusively as a post-hoc benchmark. This experiment formally compares the model's predictive performance against Bet365 closing odds, and tests whether the model can identify profitable value bets.

**Model**: LR + FootyStats, `include_odds=False` (71 features, no odds in training)
**Benchmark**: Bet365 closing odds (`odds_home_close`, `odds_draw_close`, `odds_away_close`)
**Data**: 6 PL seasons (2019-2025), 2280 matches, 80/20 temporal split → 1824 train / 456 test

### 13.1 Model vs Market — Head-to-Head

| Metric | Model | Market (Bet365) | Delta | Winner |
|:-------|:-----:|:---------------:|:-----:|:------:|
| **Accuracy** | **68.42%** | 57.02% | +11.40pp | **Model** |
| F1 (macro) | 0.5894 | 0.4271 | +0.1624 | Model |
| LogLoss | 0.7277 | 0.9483 | -0.2206 | Model |
| Brier (mean) | **0.1404** | 0.1872 | -0.0468 | **Model** |
| ROC AUC | 0.8312 | 0.6908 | +0.1404 | Model |

**Key observation**: The market NEVER predicted a draw (0/456). Bet365's closing odds always favoured either home or away.

### 13.2 Disagreement Analysis

| Outcome | Count | % of All |
|:--------|:-----:|:--------:|
| Model & Market agree | 310 | 68.0% |
| **Model correct, Market wrong** | **77** | **16.9%** |
| Market correct, Model wrong | 25 | 5.5% |
| Both wrong | 44 | 9.6% |

Model is 3x more likely to be correct when they disagree.

### 13.3 Value Betting Simulation

Edge = model_prob / market_prob - 1. Bet 1 unit when edge > threshold.

| Threshold | Bets | Win Rate | Total P&L | ROI |
|:---------:|:----:|:--------:|:---------:|:---:|
| > 10% | 544 | 58.8% | +338.4 | +62.2% |
| > 20% | 479 | 61.2% | +355.7 | +74.3% |

### 13.4 Interpretation

1. Model outperforms market on every metric (+11.4pp accuracy, -0.047 Brier).
2. Value betting appears profitable in-sample, but caveated by small test set (456 matches).
3. Consistent with Reade et al. (2020): "no consistent financial returns could be generated from simple betting strategies."

### Outputs

- `outputs/league/model_vs_market_comparison.csv`
- `outputs/league/value_betting_simulation.csv`

---

## Experiment 014 — English 4-League Model (E0-E3, Buchdahl-style)

**Date**: 2026-07-07

**Context**: To handle promoted/relegated teams in cross-season prediction, we follow
Buchdahl (2003) by using all English professional leagues (Premier League, Championship,
League One, League Two) rather than the top 5 European leagues. A `league_level` feature
(0=PL, 1=Champ, 2=L1, 3=L2) is added so the model can learn strength differences.

**Implementation**:

1. Downloaded 28 CSV files from football-data.co.uk (E0-E3 × 6 seasons, 2019-2025)
2. Total: 11,952 matches (E0: 2,660, E1: 3,864, E2: 3,712, E3: 3,752)
3. Features: 63 (62 rolling/shot/h2h + `league_level`)
4. Model: LogisticRegression (l2, C=1.0)

**Performance**:

| Metric | Value |
|:-------|:-----:|
| Training accuracy (full) | 60.05% |
| Test accuracy (80/20) | 58.60% |
| Features | 63 (rolling + shots + h2h + league_level) |

### Model vs Market — Head-to-Head (80/20 split, 2,798 test matches)

| Metric | Model | Market (Bet365) | Diff | Winner |
|:-------|:-----:|:---------------:|:----:|:------:|
| **Accuracy** | **58.60%** | 49.70% | **+8.90pp** | **Model** |
| F1 (macro) | 0.5201 | 0.3627 | +0.1574 | Model |
| LogLoss | **0.8451** | 1.0221 | −0.1770 | **Model** |
| Brier | **0.1665** | 0.2045 | −0.0380 | **Model** |
| ROC AUC | **0.7686** | 0.6205 | +0.1481 | Model |

### Per-class

| Class | Model Acc | Market Acc | Diff | Model Brier | Market Brier |
|:------|:---------:|:----------:|:----:|:----------:|:------------:|
| Away | **70.3%** | 47.4% | +22.8pp | **0.1528** | 0.1987 |
| Draw | **13.0%** | 0.0% | +13.0pp | **0.1818** | 0.1912 |
| Home | **81.8%** | 80.2% | +1.7pp | **0.1649** | 0.2234 |

### Disagreement Analysis

| Measure | Value |
|:--------|:-----:|
| Model & Market agree | 62.7% of matches |
| **Model correct, Market wrong** | **19.0%** |
| Market correct, Model wrong | 7.8% |
| Both wrong | 10.4% |
| Model correct rate when disagreeing | **51.0%** (vs Market 21.0%) |

**Use case — 2026/27 GW1 prediction**:

Using team form from 2025/26, predicted the opening weekend (21-24 Aug 2026).
Promoted teams (Coventry City, Hull City, Ipswich Town) imputed with PL league averages.

| Home | Away | Pred | H% | D% | A% |
|:-----|:-----|:---:|:--:|:--:|:--:|
| Arsenal | Coventry City [P] | Home | 56% | 34% | 10% |
| Hull City [P] | Man United | Home | 43% | 30% | 27% |
| Everton | Crystal Palace | **Home** | **85%** | 12% | 3% |
| Ipswich Town [P] | Sunderland | Home | 43% | 30% | 27% |
| Nott'm Forest | Leeds | Draw | 12% | 51% | 37% |
| Brentford | Tottenham | Home | 36% | 31% | 32% |
| Brighton | Aston Villa | **Home** | **74%** | 19% | 6% |
| Man City | Bournemouth | Draw | 33% | 39% | 28% |
| Newcastle | Liverpool | Home | 46% | 31% | 23% |
| Fulham | Chelsea | Home | 46% | 31% | 23% |

**Key observations**:
1. `league_level` weight near zero: rolling stats already capture league differences
2. Promoted teams imputed with PL averages → plausible but conservative predictions
3. Model favours home teams with strong end-of-season form (Everton 85%, Brighton 74%)

**Outputs**:
- `outputs/league/predictions_2627_gw1_english4.csv`

**Target**: Match Result (H/D/A), 80/20 temporal split

**Per-class performance**:

| Class | Precision | Recall | F1 | Support |
|:------|:--------:|:------:|:--:|:-------:|
| Away | 0.688 | 0.825 | 0.750 | 160 |
| Draw | 0.484 | 0.140 | 0.217 | 107 |
| Home | 0.717 | 0.884 | 0.791 | 189 |

**LogLoss**: 0.7279 | **ROC AUC (macro)**: 0.8318 | **Brier**: 0.1403

**Conclusion**: The 68% target is confirmed achievable with LR + FootyStats on
6 seasons of PL data. The bottleneck was data coverage, not model capacity.
Restricting to seasons with complete feature coverage is essential for valid
evaluation.

**Identified improvement path**: FootyStats provides PPG, xG, and possession
data free of charge for the Premier League only. The +6.7pp lift over the
rolling-stats baseline (62.2% → 68.9%) demonstrates that these cross-season
team strength features are the single most effective predictor. Expanding
similar coverage (PPG, xG, pre-match expected goals, possession) to
Bundesliga, La Liga, Serie A, and Ligue 1 would likely break the 62% ceiling
on those leagues as well — but this is a data acquisition problem, not a
modeling problem. Potential sources include paid football data APIs
(FootyStats Pro, Sportmonks, Opta) or web-scraped pre-match xG from
public platforms.

**Calibration analysis** (2026-07-06):

Evaluated the 68.86% model's probability calibration using Brier score and
reliability curves. Plots saved at `outputs/league/calibration_lr_footystats.png`.

| Class | LR Brier | Market Odds Brier | LR vs Odds |
|:------|:--------:|:-----------------:|:-----------|
| Away | 0.127 | 0.351 | LR 2.8× better |
| Draw | 0.163 | 0.178 | LR 1.1× better |
| Home | 0.131 | 0.355 | LR 2.7× better |
| **Mean** | **0.140** | **0.295** | **LR 2.1× better** |

Calibration curves show all three classes closely follow the diagonal
(max deviation ~0.09 at mid-range for Home/Away; Draw deviates −0.127 at
high end). **No Platt scaling or post-hoc calibration needed** — the LR
model with L2 regularization produces naturally well-calibrated probabilities.

The model's Brier score (0.140) is less than half the market odds (0.295),
demonstrating that the learned probabilities are substantially more precise
than Bet365 implied probabilities on this test set. This is a strong result
for the paper's evaluation section.

---

## Experiment 012 — Over/Under 2.5 Goals Prediction

**Date**: 2026-07-06

**Context**: O/U 2.5 binary classification is a natural companion task to match
result (H/D/A) prediction. Previously reported 71.4% on 5 leagues (Exp 001)
but that included betting odds features. Test with the current best pipeline:
PL-only + FootyStats, with and without odds.

**Data**: 6 PL seasons (2019-2025), 2,280 matches, 80/20 temporal split.
Target: 1 if total goals > 2.5, 0 otherwise.

**Results**:

| Model | Acc | Brier | ROC AUC | Note |
|:------|:---:|:-----:|:-------:|:-----|
| Naive (always Over) | 58.11% | — | — | Baseline |
| **LR + FootyStats** | **67.11%** | **0.204** | **0.736** | **Best** |
| LR + FootyStats (no odds) | 67.11% | 0.205 | 0.734 | Odds irrelevant |
| Market odds (Bet365) | 58.55% | 0.237 | 0.590 | Benchmark |

**Key findings**:

1. **67.11% accuracy — 8.6pp above market**: The model beats Bet365 by a
   wide margin on O/U 2.5, a stronger outperformance than on H/D/A.
2. **Odds features again irrelevant**: Identical results with or without
   betting odds (67.11% both ways), confirming the Exp 003 finding that
   LR extracts no unique signal from market prices.
3. **Confusion matrix**: Model favours Over (78% recall) over Under (51%
   recall), matching the skewed class distribution (58% Over in test set).
4. **ROC AUC 0.736 vs 0.832 (H/D/A)**: O/U is genuinely harder to
   discriminate than match result, consistent with the higher inherent
   randomness in total goals.

**Comparison with previous 71.4%**: The earlier result likely reflected
data leakage via betting odds encoding league-specific goal distributions.
The corrected PL-only + FootyStats result (67.11%) is more reliable and
represents a clean 8.6pp lift over the market baseline.

The Poisson rating model has been thoroughly evaluated:
- As standalone: ~48-51% accuracy (not competitive vs LR at 62%)
- With Dixon-Coles: Draw prediction enabled but accuracy unchanged
- With 5× data: marginal improvement, no breakthrough
- As ensemble feature: useful (0.8pp lift in LR), no DC or extra data needed

---

## Experiment 011b — Dixon-Coles Draw Correction

**Date**: 2026-07-06

**Context**: The pure Poisson rating model (Exp 011) never predicted draws
(Draw F1=0.000). Dixon & Coles (1997) introduced a parameter ρ that adjusts
the joint probability for the four low-scoring cells (0-0, 1-1, 0-1, 1-0),
inflating draws and deflating narrow wins to match real-world football.

**Implementation**: Added `rho` parameter to `PoissonRatingModel`:
- τ(0,0) = 1 - λ_h·λ_a·ρ  (inflates 0-0 when ρ < 0)
- τ(1,1) = 1 - ρ           (inflates 1-1 when ρ < 0)
- τ(0,1) = 1 + λ_h·ρ      (deflates away 1-0 when ρ < 0)
- τ(1,0) = 1 + λ_a·ρ      (deflates home 1-0 when ρ < 0)
- Probabilities renormalized after adjustment; negative τ clipped to 0.

**Results** (fixed base params: lr=0.03, ha=0.2, reg=1e-4, init=0.0):

| ρ | Test Acc | Test NLL | Draw F1 | Draws Predicted |
|:-:|:--------:|:--------:|:-------:|:---------------:|
| 0.000 | 47.37% | 1.0311 | 0.000 | 0/136 |
| −0.250 | **47.93%** | 1.0305 | 0.080 | 14/136 |
| −0.300 | 47.37% | 1.0323 | 0.098 | 27/136 |
| −0.350 | 46.80% | 1.0346 | **0.143** | **46/136** |

**Key findings**:

1. **DC correction successfully enables draw predictions**: ρ=−0.35 predicts
   46/136 draws (vs 0 without DC), with Draw F1=0.143. This is the first time
   the pure rating model can predict draws.

2. **Accuracy-draw tradeoff**: stronger ρ improves Draw F1 but reduces overall
   accuracy. ρ=−0.25 gives the best accuracy (47.93%) while ρ=−0.35 maximizes
   Draw F1 (0.143). This is the inherent tension in football prediction —
   draw predictions are high-risk, high-reward.

3. **In the ensemble setting, DC doesn't matter**: When Poisson probabilities
   are fed into LR + FootyStats, the accuracy is identical (~62.4%) regardless
   of ρ. The LR already learns draw patterns from FootyStats features, making
   the DC adjustment redundant in the ensemble.

**Conclusion**: Dixon-Coles correction is effective for making the standalone
Poisson rating model predict draws, but the pure model's accuracy (46-48%)
remains well below the LR baseline (62%). The practical value is in the
ensemble setting, where DC is not needed. This confirms the Poisson rating
model is best used as a **complementary feature source** rather than a
standalone predictor.

**Next**: Evaluate whether more data (5 leagues, 10,000+ matches) would
improve the pure model's convergence and competitiveness.

---

## Experiment 015 — Deep Learning (PyTorch FNN/GRU) vs LR

**Date**: 2026-07-13

**Context**: The existing FNN (sklearn MLPClassifier, 4 layer 64→128→128→128) was
never tested with FootyStats features. This experiment tests proper PyTorch deep
learning models with BatchNorm, Dropout, residual connections, and LR scheduling,
using the EXACT same 71 features as the best LR model for a fair comparison.

Also tests GRU sequence models that learn temporal patterns from raw match
sequences instead of hand-engineered rolling windows.

### Data

- **Source**: 6 PL seasons (2019-20 through 2024-25), 2,280 matches
- **Same pipeline**: `build_features(windows=[5,10], include_footystats=True, include_shots=True, include_h2h=True, include_odds=False)`
- **Feature count**: 71 features (same as best LR model)
- **Split**: 80/10/10 chronological (1824 train / 228 val / 228 test)
- **Scaling**: StandardScaler (unlike LR which uses raw features)

### Models Tested

**FNN variants** (same 71 features as LR):
- `MLP-Base`: 3-layer (256→128→64), BatchNorm + Dropout(0.3) + ReLU
- `MLP-Wide`: 2-layer (512→256), wide but shallow
- `MLP-Deep`: 5 residual blocks (128→128×4→3), skip connections

**GRU variants** (sequences + 9 FootyStats features only):
- `GRU-Simple`: Shared GRU encoder for home/away, last-state pooling
- `GRU-Dual`: Separate GRU encoders for home/away
- `GRU-Attn`: Separate GRU + attention pooling

**Training**: AdamW, ReduceLROnPlateau, early stopping patience=20, max 500 epochs,
hyperparameter sweep over lr∈{1e-3,5e-4} and wd∈{1e-5,1e-4,0}.

### Results

| Model | Accuracy | LogLoss | F1(macro) | Brier | DrawF1 |
|:------|:-------:|:-------:|:---------:|:-----:|:------:|
| **LR** (baseline) | **0.7018** | **0.7057** | 0.5942 | **0.1385** | 0.2059 |
| MLP-Wide | **0.7018** | 0.7574 | 0.6139 | 0.1467 | 0.2857 |
| **MLP-Deep** | **0.7018** | 0.7482 | **0.6429** | 0.1459 | **0.3810** |
| MLP-Base | 0.6667 | 0.7742 | 0.5541 | 0.1489 | 0.1493 |
| GRU-Simple | 0.6053 | 0.8963 | 0.4540 | 0.1756 | 0.0000 |
| GRU-Dual | 0.6053 | 0.8954 | 0.4535 | 0.1759 | 0.0000 |
| GRU-Attn | 0.5921 | 0.8951 | 0.4444 | 0.1757 | 0.0000 |

### Key Findings

1. **MLP-Wide & MLP-Deep match LR accuracy (70.18%)** — Deep learning on the
   same 71 features does NOT improve classification accuracy. This aligns with
   Dobson & Goddard (2007) that no single method dominates.

2. **MLP-Deep dramatically improves Draw F1: 0.381 vs LR's 0.206 (+85%)** —
   The residual deep architecture captures non-linear draw patterns (e.g.,
   "weaker team draws away at strong opponent") that LR's linear decision
   boundary cannot learn. This is the model's main value-add.

3. **LR still has better calibration** — LogLoss (0.706 vs 0.748) and Brier
   (0.1385 vs 0.1459) favour LR. FNN overfits slightly on probability outputs
   despite regularization.

4. **GRU models fail (60.5%)** — Only 3 sequence dims (goals_for, goals_conceded,
   is_home) + 9 FootyStats = 12 dims vs 71 for FNN/LR. Insufficient information
   for competitive prediction. Potential fix: richer sequence features or team
   embeddings.

### Thesis Value

- Confirms that **LR remains the best primary model** (best calibration, simplest)
- MLP-Deep's Draw F1 (0.381) is a **significant contribution** — the paper can
  recommend LR as primary with MLP as ensemble component for draw prediction
- The 70.18% accuracy is the highest achieved in this project (vs 68.86% previously
  on a different split)

### Files
- `scripts/deep_learning_experiment.py` — Main experiment script
- `outputs/deep_learning/metrics_comparison.csv` — All metrics
- `outputs/deep_learning/predictions_all.csv` — Per-match predictions (228 test matches)

---

## Experiment 016 — Two-Tier Ensemble + 2024-25 Backtest

**Date**: 2026-07-13

**Context**: Build a production-ready two-tier system that handles varying FootyStats
data availability (established PL teams vs promoted teams, data-covered seasons vs
gap seasons). Then backtest on 2024-25 to validate real-world deployment.

### Architecture

```
Tier 1 — Teams WITH FootyStats:
  Model: LogisticRegression (71 features: rolling + shots + h2h + FootyStats)
  Data: PL 2019-20 to 2024-25 (2,280 matches)
  Accuracy: ~69.74% (internal validation)

Tier 2 — Teams WITHOUT FootyStats:
  Model: LogisticRegression (64 features: rolling + shots + h2h + league_level)
  Data: E0-E3 2019-20 to 2024-25 (11,952 matches, 4 divisions)
  Accuracy: ~60.26% (external validation)
```

### 2024-25 Backtest Results

All 380 PL matches predicted with Tier 2 fallback (no FootyStats data available
for 2024-25 season):

| Model | Accuracy | LogLoss | Brier | DrawF1 |
|:------|:-------:|:-------:|:-----:|:------:|
| Fallback LR | **60.26%** | **0.8500** | **0.1767** | **0.1778** |
| Market (Bet365) | 45.53% | 1.0825 | 0.2038 | 0.0000 |
| **Delta** | **+14.73pp** | **-0.2325** | **-0.0271** | **+0.1778** |

### Key Findings

1. Even without FootyStats features, the fallback model maintains a **14.73pp
   advantage over the betting market**. The model's core rolling-statistic
   engine is robust.

2. The ~9.5pp gap between internal (69.74%) and external (60.26%) accuracy
   quantifies the marginal contribution of FootyStats features. This gap is
   unavoidable without paid data sources.

3. Only 1/20 2024-25 PL teams (Sunderland, promoted) lacked historical PPG
   data. The fallback's league_level feature handles this automatically.

### Files
- `scripts/build_ensemble_model.py` — Trains both Tier 1 and Tier 2 models
- `scripts/backtest_2526_with_ppg.py` — PPG-based 2024-25 backtest
- `outputs/ensemble/` — Saved models + results
- `scripts/build_league_addendum.py` — Thesis addendum generator
- `Group26_League_Addendum.docx` — Generated addendum document


---

## Experiment 017 — Tree Models (XGBoost/RF) vs LR on Full E0-E3

**Date**: 2026-07-14

**Context**: Re-run model comparison on the current dataset (E0-E3, 6 seasons, 62 features,
no asian_handicap) to test if tree-based models outperform LR with larger data.

**Data**: 11,952 matches, 62 features, walk-forward across 5 seasons (10,180 test matches)

**Results**:

| Model | Walk-forward Acc | LogLoss | DrawF1 | vs LR |
|:-----|:---------------:|:-------:|:------:|:-----:|
| **LR** | **57.43%** | **0.9018** | 0.201 | — |
| RF | 55.55% | 0.9523 | 0.114 | -1.87pp |
| XGBoost | 56.04% | 0.9677 | 0.246 | -1.39pp |

**Conclusion**: LR remains the optimal model for this dataset. Tree models do not outperform LR
on overall accuracy even with 11,952 training samples. XGBoost achieves higher DrawF1 (0.246 vs
0.201) but at the cost of overall accuracy. Atta Mills et al. (2024)'s finding that RF outperforms
LR on Dutch/Belgian/Scottish data does not generalize to our English 4-level dataset and feature set.

**Files**: scripts/backtest_english4.py (already contains LR baseline)

---

## Experiment 018 — Value Betting: Three Strategies Comparison

**Date**: 2026-07-14

**Context**: Compare three value betting strategies on Track A (E0-E3, walk-forward, 10,180 test matches)
to document their relative performance. Only "best per match" is used in the thesis overview.

**Three strategies:**
1. **Buchdahl-style** (仅主胜): Only consider home win outcome, bet if edge > threshold
2. **Best per match** (每场选最优): Among H/D/A, pick the one with highest edge, bet if > threshold
3. **All outcomes** (多注): Bet on EVERY outcome where edge > threshold (multiple bets per match)

**Results (walk-forward, 10,180 matches):**

| Threshold | Buchdahl Bets | Buchdahl ROI | BestPM Bets | BestPM ROI | Multi Bets | Multi ROI |
|:---------:|:------------:|:------------:|:-----------:|:----------:|:----------:|:---------:|
| 0% | 4,782 | +38.6% | 10,180 | +43.1% | 14,144 | +33.8% |
| 10% | 4,220 | +44.0% | 9,801 | +45.5% | 12,186 | +39.3% |
| 20% | 3,661 | +50.2% | 8,924 | +49.5% | 10,335 | +45.4% |
| **30%** | **3,111** | **+55.7%** | **7,782** | **+55.0%** | **8,628** | **+52.4%** |
| 50% | 2,194 | +66.9% | 5,657 | +66.0% | 5,994 | +65.5% |
| 100% | 820 | +98.2% | 2,441 | +99.3% | 2,496 | +100.0% |

**Key observations:**
1. All three strategies produce positive ROI across all thresholds
2. At the thesis-recommended 30% threshold, all three are clustered around +52-56% ROI
3. Best per match has the largest sample size at each threshold (bets on every match)
4. Buchdahl-style has higher win rate (63.1%) but fewer bets (only considers home wins)
5. The three strategies converge as threshold increases (at 100% they're nearly identical)

**Conclusion**: Strategy choice has minimal impact on ROI at the thesis threshold of 30%.
Best per match is selected for the overview as it provides the largest sample and most complete coverage.

---

## Experiment 019 — Atta Mills PL Phase 1/2 Reproduction

**Date**: 2026-09-03

**Context**: Refocus the league experiment around Atta Mills et al. (2024),
but keep the target as a clean pre-match Premier League H/D/A prediction task.
The reference paper reports much higher accuracy, but it includes half-time
result/goals and an O/U 2.5 task. Those variables are excluded here because
they turn the problem into in-play prediction rather than pre-match prediction.

**Repository backup**:

- Baseline commit before this branch: `208dd90`
- Backup tag: `backup/current-experiment-2026-09-03`
- Experiment branch: `codex/atta-mills-pl-walkforward`

### Phase 1 — Strict Atta Mills-style Pre-match Features

**Data**: Premier League football-data.co.uk CSVs, 2019/20-2025/26,
2,660 matches.

**Validation**: season-by-season walk-forward. Each test season is predicted
using only previous seasons.

**Training exclusions**:

- Bet365 odds are excluded from training.
- FootyStats PPG/xG/possession is excluded.
- Half-time result/goals are excluded.
- O/U 2.5 is excluded.
- H2H and rolling shot-efficiency features are excluded in Phase 1.

**Features**: 44 Atta Mills-style pre-match features:

- Team state / rolling points
- Attack strength
- Defense strength
- Goals for
- Goals against
- Goal differential
- Win/draw/loss history
- Win margin goals
- Loss margin goals

**Market benchmark**: Bet365 closing odds, converted to de-vigged implied
probabilities; market prediction is the highest closing implied probability.

| Model | Accuracy | Macro F1 | Draw F1 | LogLoss | Brier |
|:------|:--------:|:--------:|:-------:|:-------:|:-----:|
| **LR** | **49.69%** | 0.3810 | 0.0443 | 1.0339 | 0.2065 |
| FNN/MLP | 49.56% | 0.3750 | 0.0356 | 1.0328 | 0.2063 |
| Random Forest | 49.52% | 0.3709 | 0.0247 | 1.0353 | 0.2066 |
| XGBoost | 46.58% | 0.3883 | 0.1427 | 1.1041 | 0.2186 |
| LR balanced | 44.25% | 0.4172 | 0.2447 | 1.0595 | 0.2129 |
| Bet365 closing | **54.87%** | 0.4087 | 0.0000 | **0.9626** | **0.1904** |

**Phase 1 conclusion**: Strict Atta Mills-style pre-match features underperform
Bet365 closing odds. The best model by accuracy is LR at 49.69%, still 5.18pp
below the closing market. Balanced LR improves Draw F1 but sacrifices accuracy.

### Phase 2 — Add Original Clean Pre-match Features

**Added features**: 21 original clean extras:

- 16 rolling shot-efficiency features: shots, shots on target, shot accuracy,
  conversion rate for home/away teams over 5/10 windows.
- 5 head-to-head features.

**Total features**: 65 = 44 Phase 1 + 21 original clean extras.

| Model | Accuracy | Macro F1 | Draw F1 | LogLoss | Brier |
|:------|:--------:|:--------:|:-------:|:-------:|:-----:|
| **Random Forest** | **51.01%** | 0.3773 | 0.0073 | 1.0151 | 0.2020 |
| SVM | 49.96% | 0.3638 | 0.0000 | 1.0277 | 0.2053 |
| FNN/MLP | 49.21% | 0.3595 | 0.0110 | 1.0356 | 0.2059 |
| LR | 48.60% | 0.4071 | 0.1376 | 1.0708 | 0.2120 |
| LR balanced | 44.96% | 0.4304 | **0.2786** | 1.0980 | 0.2187 |
| Bet365 closing | **54.87%** | 0.4087 | 0.0000 | **0.9626** | **0.1904** |

**Phase 2 conclusion**: Adding H2H and rolling shot-efficiency features improves
the best model from 49.69% to 51.01%, but the result remains below Bet365
closing odds. This supports the interpretation that the reference paper's high
accuracy is likely helped substantially by half-time/in-play variables, and that
clean pre-match reproduction is a harder task.

**Outputs**:

- `scripts/atta_mills_pl_phase1.py`
- `scripts/atta_mills_pl_phase2.py`
- `docs/atta_mills_phase1_results.md`
- `docs/atta_mills_phase2_results.md`
- `notebooks/atta_mills_phase2_kaggle_gpu.ipynb` — Kaggle GPU notebook for
  running PyTorch MLP variants on the Phase 2 feature set.
- `outputs/atta_mills_pl_walkforward/phase1_atta_mills_only/`
- `outputs/atta_mills_pl_walkforward/phase2_with_original_features/`

**GPU note**: Local hardware has an NVIDIA RTX 3050 Laptop GPU, but the active
local Python environment has CPU-only PyTorch (`torch 2.11.0+cpu`). A local CUDA
install was not continued; the GPU follow-up is prepared as a Kaggle notebook
instead, so cloud GPU resources can be used without changing the local project
environment.

**Local GPU follow-up**: CUDA PyTorch was later installed in an isolated
`.venv_cuda` environment. PyTorch `2.11.0+cu128` detected the local NVIDIA
GeForce RTX 3050 Laptop GPU and completed a Phase 2 MLP walk-forward run in
223.57 seconds.

| Model | Accuracy | Macro F1 | Draw F1 | LogLoss | Brier |
|:------|:--------:|:--------:|:-------:|:-------:|:-----:|
| Bet365 closing | **54.87%** | **0.4087** | 0.0000 | **0.9626** | **0.1904** |
| torch_mlp_deep | 49.96% | 0.3854 | 0.0508 | 1.0239 | 0.2040 |
| torch_mlp_wide | 48.68% | 0.3987 | 0.1133 | 1.0287 | 0.2047 |

The local GPU run confirms that hardware acceleration is working, but the
additional neural models still do not outperform the Phase 2 Random Forest or
the Bet365 closing benchmark.

**Decision**: No further GPU tuning is needed for this branch. The dataset is
small tabular football data, and the strongest observed model remains a CPU
tree/LR-style workflow rather than a neural network. Future routine experiments
should use the CPU walk-forward scripts unless a new, larger neural feature set
is introduced.

---

## Experiment 020 — Atta Mills Track A English4 Archive

**Date**: 2026-09-05

**Branch**: `codex/atta-mills-tracka-english4`

**Context**: Archive the clean pre-match Track A extension of the Atta
Mills-style reproduction. The aim is to test whether expanding from Premier
League-only data to the full English E0-E3 football-data.co.uk dataset improves
the clean Atta Mills framework by increasing match volume and preserving team
history across promotion/relegation.

This experiment is intentionally separate from the half-time/in-play branch
(`codex/halftime-paper-replication`). Half-time goals/result are still excluded
here, because they are only known after the first half and change the research
problem from pre-match prediction to in-play prediction.

### Experimental Progression

1. **Strict Atta Mills-style PL reproduction**:
   - Data: Premier League only, 2019/20-2025/26.
   - Features: 44 Atta Mills-style pre-match team-state features.
   - Exclusions: odds as training features, half-time variables, O/U 2.5,
     FootyStats, H2H, and rolling shot-efficiency features.
   - Best model: LR, 49.69% accuracy.
   - Market benchmark: Bet365 closing, 54.87% accuracy.

2. **PL reproduction plus original clean features**:
   - Added 16 rolling shot-efficiency features and 5 H2H features.
   - Total features: 65.
   - Best model: Random Forest, 51.01% accuracy.
   - Market benchmark: Bet365 closing, 54.87% accuracy.
   - Interpretation: the project's original clean features add useful signal,
     but not enough to beat the closing market.

3. **Track A English4 extension**:
   - Data: E0, E1, E2, and E3 as one integrated chronological dataset.
   - Seasons: 2019/20-2025/26.
   - Matches: 13,988.
   - Validation: season-by-season walk-forward.
   - Features: 66 total = 44 Atta Mills-style features + 16 rolling
     shot-efficiency features + 5 H2H features + 1 `league_level` feature.
   - Exclusions remain unchanged: no odds in training, no half-time variables,
     no O/U 2.5, and no FootyStats.

### Promotion/Relegation Handling

Team histories are keyed by team name rather than by league. This means a team
that is promoted or relegated carries its rolling history into its new division,
instead of being treated as a completely new entity. A `league_level` control
marks the match tier:

- E0 = 0
- E1 = 1
- E2 = 2
- E3 = 3

### Track A Results

| Model | Accuracy | Macro F1 | Draw F1 | LogLoss | Brier |
|:------|:--------:|:--------:|:-------:|:-------:|:-----:|
| **Random Forest** | **46.01%** | 0.3395 | 0.0204 | 1.0553 | 0.2119 |
| LR | 45.62% | 0.3520 | 0.0415 | 1.0562 | 0.2120 |
| Voting RF/XGB | 45.37% | 0.3487 | 0.0544 | 1.0606 | 0.2129 |
| XGBoost | 44.35% | 0.3554 | 0.0897 | 1.0758 | 0.2159 |
| Random Forest balanced | 43.21% | 0.3870 | 0.1818 | 1.0696 | 0.2152 |
| LR balanced | 41.88% | 0.3950 | **0.2369** | 1.0809 | 0.2175 |
| Bet365 closing | **49.38%** | 0.3653 | 0.0006 | **1.0189** | **0.2036** |

### Archived Conclusion

The Track A English4 extension did not improve the clean pre-match Atta
Mills-style workflow. Although adding E1-E3 increases the training sample and
preserves promoted/relegated team histories, the combined four-division task is
more heterogeneous and noisier than the Premier League-only task. The best
Track A model reaches 46.01%, below both the PL Phase 2 model at 51.01% and the
E0-E3 Bet365 closing benchmark at 49.38%.

The useful conclusion from this branch is therefore negative but important:
more same-source football-data.co.uk league data does not automatically improve
clean pre-match H/D/A prediction. The larger gains seen in the separate
half-time branch should be interpreted independently as the effect of
in-play/half-time information, not as evidence that the clean pre-match feature
set can reproduce the paper's headline performance.

**Archive decision**: keep this branch as a completed negative-result Track A
record. Do not continue tuning this branch unless a new feature family is
introduced; future work should either return to the PL clean feature set or be
clearly labelled as a separate in-play/half-time experiment.

**Outputs**:

- `scripts/atta_mills_tracka_english4.py`
- `docs/atta_mills_tracka_english4_results.md`
- `outputs/atta_mills_tracka_english4/`

---

## Experiment 024 — Track A Poisson Goals Baseline

**Date**: 2026-09-07

**Branch**: `codex/tracka-poisson-goals`

**Context**: Add a Loukas et al. (2024)-style Poisson goals baseline under the
clean Track A English4 line. This branch is not Track B: Track B remains the
FootyStats-enhanced route. The purpose here is to keep the same
football-data.co.uk, no-external-enhancement setting as Track A, but switch the
modelling target from direct H/D/A classification to goals -> scoreline ->
H/D/A probability aggregation.

### Method

For each walk-forward fold, the model is fitted only on seasons before the test
season. It estimates:

- `mu`: log average away goals in the training data.
- `mu_home`: pooled home-goal log advantage.
- team attack parameters from historical goals scored.
- team defence parameters from historical goals conceded.

The expected goals equations follow the Loukas-style independent double
Poisson structure:

```text
log(lambda_home) = mu + mu_home + attack_home + defence_away
log(lambda_away) = mu + attack_away + defence_home
```

The scoreline matrix is generated from independent home and away Poisson
distributions. H/D/A probabilities are then aggregated from all scorelines:

```text
P(H) = sum P(home_goals > away_goals)
P(D) = sum P(home_goals = away_goals)
P(A) = sum P(home_goals < away_goals)
```

No O/U 2.5 experiment is included in this branch.

### Leakage Position

This validation is stricter than the random 20% validation in Loukas et al.
(2024), because the paper's random-sample and same-season team-total checks can
use information from matches being evaluated if the full season is used to
estimate team attack/defence parameters. Here, each test season is unseen at
fit time, so the evaluation follows the project's established Track A
walk-forward protocol.

### Results

| Model | Accuracy | Macro F1 | Draw F1 | LogLoss | Brier |
|:------|:--------:|:--------:|:-------:|:-------:|:-----:|
| Loukas-style Poisson | 42.00% | 0.2974 | 0.0000 | 1.0945 | 0.2213 |
| Bet365 closing | 49.38% | 0.3653 | 0.0006 | 1.0189 | 0.2036 |

Goal metrics:

| Metric | Value |
|:-------|------:|
| Home goals MAE vs lambda | 1.0508 |
| Away goals MAE vs lambda | 0.9457 |
| Total goals MAE vs lambda total | 1.4030 |
| Exact home goals from modal score | 33.22% |
| Exact away goals from modal score | 35.18% |
| Exact score from modal score | 11.80% |
| Home goals within +/-1 | 80.57% |
| Away goals within +/-1 | 86.13% |

Using the paper's rounded goal-difference style, the all-fold walk-forward
results are 73.3% of home-goal predictions and 81.3% of away-goal predictions
within +/-1. This is below the paper's same-season random 76-match check for
away goals, but broadly consistent with the paper's independent 50-match check.

### Interpretation

The first Track A Poisson run is useful as a clean, interpretable goal-based
baseline, but it does not beat the market or the existing Track A classifiers
on H/D/A accuracy. Its main value is methodological: it gives expected goals,
modal scorelines, and full scoreline-derived H/D/A probabilities under the same
no-FootyStats Track A data policy.

The zero Draw F1 confirms the known limitation of independent Poisson models:
draw probabilities are non-zero, but they rarely become the argmax prediction.
Future work can test Dixon-Coles correction or use Poisson probabilities as
additional Track A classifier features, but those should be labelled as follow-up
extensions rather than this first Loukas-style baseline.

**Final decision**: keep `codex/tracka-poisson-goals` as a documented Track A
Poisson goals baseline. Do not promote it to the main Track A classifier, and do
not label it as Track B because Track B is reserved for FootyStats-enhanced
models.

**Outputs**:

- `scripts/tracka_poisson_goals.py`
- `docs/tracka_poisson_goals_results.md`
- `outputs/tracka_poisson_goals/`
