# Football Prediction FYP

> 本文件以旧版项目总览为基础，保留其项目结构、Track A / Track B、深度学习和世界杯介绍，并在末尾追加后续实验记录。
> 旧版 Main Results 保留当时实验的原文和历史指标。最新 Clean Track A 统一口径结果请参阅末尾的六赛季回测记录及[中文报告](experiments/nested_backtest_20261010/reference/nested_backtest_report_CN.md)。
> [历史 README 原文入口](docs/readme_archive/README.md)包含项目总览、English4 基线和 Poisson 基线三个版本。

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

---

## 追加实验记录：Poisson 比分基线

This branch adds a Loukas et al. (2024)-style Poisson goals baseline under the
clean Track A English4 line.

It is not Track B. In this project, Track B is reserved for FootyStats-enhanced
models. This branch uses only football-data.co.uk match results and historical
full-time goals, then predicts:

- expected home goals
- expected away goals
- modal scoreline
- H/D/A probabilities aggregated from the scoreline matrix

### Branch Context

| Item | Value |
|---|---|
| Main Track A branch | `codex/atta-mills-tracka-english4` |
| This branch | `codex/tracka-poisson-goals` |
| Result document | `docs/tracka_poisson_goals_results.md` |
| Output directory | `outputs/tracka_poisson_goals/` |

### Conclusion

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

### Method

The model follows the independent double-Poisson structure from Loukas et al.
(2024):

```text
log(lambda_home) = mu + mu_home + attack_home + defence_away
log(lambda_away) = mu + attack_away + defence_home
```

The scoreline matrix is converted to H/D/A probabilities by summing all home-win,
draw, and away-win score probabilities.

### Validation

The evaluation uses the Track A season-by-season walk-forward protocol:

- train on all seasons before the test season
- test on the next season
- repeat through the available English4 data

This avoids the same-season leakage risk in random-sample validation.

### Run

From the repository root:

```powershell
python scripts/tracka_poisson_goals.py
```

Generated outputs are ignored by Git.

---

## 六赛季严格整体回测 2026-10-10

已完成 **2020/21–2025/26 六季、英超/英冠/英甲/英乙共 12,216 场** 的嵌套滚动回测。2019/20 的 1,772 场用作初始历史，没有更早训练数据，故不计作独立测试。

每季只用此前数据排序、验证并冻结特征，再用此前全部赛季训练。四级联赛合并使用同一模型、同一套当季特征。首次回测在初始季内按日期划分训练/排序/验证，随后逐步使用完整历史赛季；没有把后期选出的固定 36 列倒用到早期独立测试。

### 同口径完整基线与特征选择方案

| 模型 | 完整66列 Accuracy | 历史选特征 Accuracy | 净多预测正确 | 完整 Log Loss | 所选 Log Loss |
|---|---:|---:|---:|---:|---:|
| 逻辑回归 | 45.63% | 46.46% | +101 | 1.05610 | 1.05291 |
| 随机森林 | 45.99% | 45.61% | -46 | 1.05462 | 1.05445 |
| 标准化逻辑回归 | 45.71% | 46.74% | +126 | 1.05806 | 1.05064 |

Bet365 收盘市场：49.38%，Log Loss 1.01891（12,216 场）。主方案一直按内部验证 Log Loss 选择；另存准确率目标的全部结果，不能用外层测试再挑赢家。标准化 LR 的差异包含 C 的选择。

本项目的特征工程研究目标是：在相同比赛、训练区间和指标口径下，验证模型简化是否改善原模型，并确保全过程排除数据泄漏、可以复现。博彩公司市场作为外部参照，超过市场不作为毕业项目成功的必要条件。

逻辑回归：准确率变化 +0.83 个百分点，按赛季与比赛日期成组的 4,000 次配对 bootstrap 95% 区间为 [+0.31, +1.36] 个百分点；Log Loss 差异区间 [-0.00551, -0.00080]。准确率提高 4/6 季，Log Loss 改善 4/6 季。

随机森林：准确率变化 -0.38 个百分点，按赛季与比赛日期成组的 4,000 次配对 bootstrap 95% 区间为 [-0.80, +0.07] 个百分点；Log Loss 差异区间 [-0.00160, +0.00127]。准确率提高 0/6 季，Log Loss 改善 4/6 季。

标准化逻辑回归：准确率变化 +1.03 个百分点，按赛季与比赛日期成组的 4,000 次配对 bootstrap 95% 区间为 [+0.52, +1.54] 个百分点；Log Loss 差异区间 [-0.00990, -0.00492]。准确率提高 6/6 季，Log Loss 改善 4/6 季。

区间描述固定预测的抽样波动，没有覆盖完整选模和训练不确定性。历史数据及此前 2025/26 结果已被观察过，本次仍属于回顾性研究。网格最优只针对预设候选。

整体回测支持两种逻辑回归按历史选特征流程的准确率及 Log Loss 改善；随机森林准确率下降，Log Loss 的微小变化区间包含零。三个所选流程的整体 Macro F1 均未提高，平局识别仍弱，且均未超过收盘市场。

原逻辑回归的 101 场净增中，71 场来自 2021/22 当折选出的 8 列胜负分差组合。最近两折均事前选中 36 列 window_10_only：2024/25 准确率与完整模型持平，2025/26 多对 10 场。因此整体增益不能归因于固定 36 列，也不能据此称它跨全部六季稳定最优。

**0.83 个百分点描述的是每季根据历史重新选特征的流程，不能直接当作固定 36 列模型的提升。** 最近两个外层折恰好都选中了同一 36 列方案，两季共 4,072 场，完整 66 列预测正确 1,924 场，36 列预测正确 1,934 场，合计提升约 0.25 个百分点。该方案特征数减少约 45%，目前支持更简洁、略有改善的候选结论；这两季的点估计不等于跨全部六季稳定优于完整模型。

### 与旧结果对照

| 模型 | 旧历史结果 Accuracy | 本次修正后的完整基线 | 本次历史选特征 |
|---|---:|---:|---:|
| 逻辑回归 | 45.62% | 45.63% | 46.46% |
| 随机森林 | 46.01% | 45.99% | 45.61% |

旧历史结果也覆盖相同 12,216 场，已逐场匹配。旧流程有 league_level 重复列、35 场七月比赛的赛季归属错误和累计均值使用同日信息的问题；本次完整基线先修正这些问题。因此，旧结果到新方案的差异包含实现修正，特征选择效果应看本次基线与本次方案的差异。

本次 2025/26 的九组同名配置预测通过概率一致性核对，复现下文旧单季实验。此前逻辑回归单季多对 10 场只是 2025/26 的结果；本次整体净增场次包含全部六季。

七月修正的 35 场全部属于 2022/23：英冠 11 场、英甲 12 场、英乙 12 场，从 2021/22 移回 2022/23。依据源 CSV 所属赛季，比赛未被删除。

### 代码、结果与复现

- [整体中文报告](experiments/nested_backtest_20261010/reference/nested_backtest_report_CN.md)
- [逐季完整指标](experiments/nested_backtest_20261010/reference/metrics_all_and_seasons.csv)
- [逐季增减与所选配置](experiments/nested_backtest_20261010/reference/controlled_comparison_by_season.csv)
- [各季全部入选特征](experiments/nested_backtest_20261010/reference/selections_all.json)
- [逐场预测压缩快照](experiments/nested_backtest_20261010/reference/predictions_all.csv.gz)
- [代码与运行说明](experiments/nested_backtest_20261010/README.md)

10 项特征/时间边界单元检查通过，整体指标已由逐场预测重算；18 个主方案模型重新训练、保存、加载后的概率通过核对。依赖版本、31 个输入哈希、代码哈希和参考快照均已保存。Git 忽略的 results/ 保存完整生成物；GitHub 可提交的 reference/ 保留报告和比较证据。

发布前另通过 3 项输入换行兼容检查；[参考记录自检脚本](experiments/nested_backtest_20261010/verify_reference.py) 可从压缩逐场预测核对指标、2,275 个配置、3,412 次内部验证记录、特征选择与时间边界。[发布检查记录](experiments/nested_backtest_20261010/reference/publication_audit.json) 保存检查范围与完成状态。

## 特征工程与特征组合实验 2026-10-10

本次对 Clean Track A 做了文献重要性核查、当前模型重要性排序和特征消融。实验覆盖英超、英冠、英甲、英乙，共 13,988 场。四级联赛合并训练同一个分类器，使用同一套特征定义；球队升降级后延续历史记录，`league_level` 标记比赛所在联赛。逻辑回归、随机森林和标准化逻辑回归是用于选择方法的对照配置，各自都统一预测四级联赛。

### 特征选择与回测方式

- 2022/23：计算特征重要性，模型只用 2019/20–2021/22 训练。
- 2023/24、2024/25：按赛季滚动训练和验证，用于选择特征组合及参数。
- 2025/26：选择冻结后，用之前全部赛季训练，再评估这一季的 2,036 场比赛。
- 每个回测赛季重新估计模型参数，算法和特征方案保持一致；赔率只作 Bet365 收盘市场基准。

额外比较了 2022/23、2023/24、2024/25 的重要性排名是否相近。这三个赛季是用于分析和选择特征的历史数据；额外排名核查没有反过来修改冻结的选择。

特征分为六组：历史积分及胜平负状态、进失球及攻防强度、胜负分差、射门及射正效率、历史交锋、联赛级别。搜索所有 63 个非空特征组组合，并加入 Top-k、删除末位特征、正置换重要性列、5/10 场窗口删减以及相关性去冗余。去重后每种模型有 76 个特征组合；标准化逻辑回归另搜索 C=0.1/1/10，总计 380 个模型与参数配置、760 次验证训练。

主选择指标预先设为验证 Log Loss；同时冻结验证准确率最高的配置。本次两个目标在各模型中选出了相同的组合。特征重要性使用打乱后 Log Loss 上升及准确率下降，重复 5 次，并检查整组特征。低重要性不直接等于该足球信息无效，相关特征之间可能互相替代。

### 最后评估结果

下表全部为 2025/26 同一批 2,036 场，不能直接与旧报告多个赛季的整体准确率比较。Log Loss 越低越好。

| 模型与配置 | 特征数 | Accuracy | Log Loss |
|---|---:|---:|---:|
| 逻辑回归完整基线，C=1 | 66 | 46.71% | 1.05192 |
| 逻辑回归精简候选，C=1 | 36 | 47.20% | 1.04979 |
| 标准化逻辑回归完整基线，C=1 | 66 | 46.81% | 1.05314 |
| 标准化逻辑回归精简候选，C=0.1 | 36 | 47.25% | 1.05016 |
| 随机森林完整基线 | 66 | 45.87% | 1.05366 |
| 随机森林精简候选 | 45 | 45.63% | 1.05153 |
| Bet365 收盘市场 | — | 49.26% | 1.02331 |

按预先固定的验证 Log Loss 规则，全部已搜索配置中最优候选为原逻辑回归的 `window_10_only`：保留主客队近 10 场的 30 列、5 列历史交锋和 1 列联赛级别，删除全部近 5 场的 30 列。验证两季的准确率为 47.99%，Log Loss 为 1.03828。

随机森林的验证最优组合保留历史状态、交锋、胜负分差和射门组，共 45 列。其最后评估 Log Loss 改善，准确率略降。标准化逻辑回归的精简配置还改变了正则化参数，因此其变化不能全部归因于删特征。

### 为什么目前只能称为小幅改善

原逻辑回归完整基线预测正确 951 场，36 列候选预测正确 961 场，即多对 10 场，准确率增加约 0.49 个百分点。按比赛日期成组进行 2,000 次配对 bootstrap，准确率差异的 95% 区间为 **[-0.31, +1.36] 个百分点**，包含零；Log Loss 差异区间 **[-0.00493, +0.00073]** 也包含零。现有数据尚不能排除两种方案实际表现相近的情况。

因此，36 列方案是更简洁、值得继续验证的候选，暂不声称它已经稳定优于完整模型。这些区间只描述一个回顾性评估赛季，没有覆盖跨赛季、训练和选模的不确定性。2025/26 曾出现在项目此前实验中，也不能称为项目从未观察过的新数据。后续需要新赛季或其他独立数据确认；本次候选仍未跑赢收盘市场，精简模型的 Draw F1 仍为 0。

### 文献和实现核查

Atta Mills 等使用随机森林重要性，靠前特征包含半场信息和赔率；它们不符合当前 Clean Track A 的输入范围。Luiz 等使用打乱特征后的性能变化，排名靠前的是相对防守、组织、中场能力及主队预期进球；其 WhoScored 数据中的部分字段不在当前数据中。文献排序只提供假设，实际保留列由本项目验证数据决定。

本项目三个历史赛季的逻辑回归排序中，客队近 5 场攻击强度、主客队近 10 场射正、客队近期射门与净胜球较稳定靠前。单列排名不等于最佳组合：完整模型中的相关列会互相替代，所以需要重训练后的消融验证。

本次独立实验还修正了评估口径：

- 原 `league_level` 重复加入，实际传入 67 列；实验只保留一列。
- 按月份划分导致 35 场七月比赛归到上一赛季；改为按原 CSV 所属赛季确定。
- 构造特征时排除当场和同一天结果，避免累计联赛均值吸收同日信息。

原模型与既有结果保留，本次修正和实验代码位于独立目录。5 项时间顺序及指标检查通过；导出的 6 个冻结模型重新训练、加载后的预测概率已核对。

### 文件和复现

- [中文实验报告](experiments/feature_search_20261010/reference/feature_search_report_CN.md)
- [文献原文核查](experiments/feature_search_20261010/literature_review.md)
- [实验代码及运行说明](experiments/feature_search_20261010/README.md)
- [逻辑回归完整重要性排名](experiments/feature_search_20261010/reference/importance_lr.csv)
- [随机森林完整重要性排名](experiments/feature_search_20261010/reference/importance_rf.csv)
- [逻辑回归跨历史赛季排名](experiments/feature_search_20261010/reference/importance_stability_lr.csv)
- [冻结的保留特征列表](experiments/feature_search_20261010/reference/frozen_selections.json)
- [全部验证配置](experiments/feature_search_20261010/reference/validation_grid_all.csv)
- [最后评估逐场预测](experiments/feature_search_20261010/reference/final_test_predictions.csv.gz)
- [差异区间](experiments/feature_search_20261010/reference/paired_uncertainty.json)

从项目根目录运行：

```powershell
.\.venv_cuda\Scripts\python.exe -B -u experiments/feature_search_20261010/run_feature_search.py --project-root . --output-dir experiments/feature_search_20261010/results
.\.venv_cuda\Scripts\python.exe -B experiments/feature_search_20261010/test_feature_search.py
```
