# 数据泄漏修复 —— 干净数字审查文档（2026-08-31）

> **状态**：用户暂缓论文文档更新，先审数字。本文档是全部干净数字的权威对照 + 复现命令。
> **背景**：发现并修复了 `league/src/features.py` 的数据泄漏。此前的论文/README 数字全部作废，见下表。

---

## 1. TL;DR

- **泄漏问题**：`get_feature_columns` 把 12 个"当场统计列"当特征透传（未滞后）：射门/射正/犯规/角球/黄牌/红牌 × 主客。这些是**比赛自身的结果指标**——用它预测比赛 = 泄漏。
- **修复**：加入 `LEAKED_MATCH_STATS` 排除集。修复后特征数：Track A 63→**50**，Track B 71→**58**。
- **后果**：所有声称"模型优于市场"的数字（Track A 58.60%、Track B 68.42%、O/U 67.11%、ensemble 69.74% 等）**全部是泄漏假象**。
- **唯一存活信号**：① 市场基线统一为**收盘赔率**（B365C）后，模型**不跑赢市场**——主评估（全季 walk-forward 50.21% vs 56.32% **显著输 −6.11pp**；80/20 54.61% vs 57.24% 输），附加 4/2 精简模型 +1.45pp/+0.92pp 不显著；② FootyStats 特征确有增量（+5.5pp vs 纯滚动）；③ 干净模型下 Buchdahl 价值投注 ROI 仍为正（收盘价 B365C，+3.69~+11.95%）；④ DL/树模型都不如 LR。

---

## 2. 泄漏问题与修复

**12 个泄漏列**：
```
home_shots, away_shots, home_shots_target, away_shots_target,
home_fouls, away_fouls, home_corners, away_corners,
home_yellow, away_yellow, home_red, away_red
```

**修复位置**：`league/src/features.py` → `get_feature_columns` 的 exclude 集合
（新增 `LEAKED_MATCH_STATS` 常量，`exclude |= LEAKED_MATCH_STATS`）。

**受影响的脚本**（都用 `get_feature_columns`，自动继承修复）：
`backtest_english4.py`、`buchdahl_ppg_walkforward.py`、`build_ensemble_model.py`、
`deep_learning_experiment.py`、`value_betting_walkforward.py`、`over_under_experiment.py`、`tree_comparison.py`。

**不受影响**（特征只取滚动均值/前缀过滤，本就无泄漏）：
`fair_comparison.py`（滚动均值+FootyStats 25 特征）、`footystats_backtest.py`（滚动+阵容+赛前）、World Cup 子项目（独立管线）。

---

## 3. 完整对照表（每条论文数字 → 干净数字）

### 3.1 Track A（E0-E3，7 季 13,988 场，50 特征）

| 指标 | 论文（泄漏） | **干净** | 市场 | 复现 |
|:--|:--:|:--:|:--:|:--|
| 80/20 准确率（2,798 测试） | 58.60% (+8.90pp) | **46.4%** (−2.9pp) | 49.3%（收盘） | `python scripts/backtest_english4.py` |
| Walk-forward 准确率 | 57.43% | **45.1%** (−4.3pp) | 49.4%（收盘） | 同上 |
| 价值投注 ROI（best-per-match，edge>10%，收盘 B365C） | +38~+99% | **−6.40%**（胜率 27.1% 低于随机 1/3，0/6 季为正） | — | `python scripts/value_betting_walkforward.py` |
| 价值投注 ROI（30% 阈值） | +55.0% | **−9.6%**（WF）/ **−12.2%**（80/20） | — | `backtest_english4.py` |
| 树模型 LR/RF/XGB（7 季 WF） | 57.43/56.04/55.55 | **45.05/45.17/41.31** | — | `python scripts/tree_comparison.py` |

> 备注：论文"6 季 11,952、2,391 测试"与确认约定"7 季 13,988、2,798 测试"不一致；已把 `backtest_english4.py` 的 2025/26 排除行移除，统一到 7 季口径。

### 3.2 Track B（英超，6 季 2,280 场，58 特征）

| 指标 | 论文（泄漏） | **干净** | 市场 | 复现 |
|:--|:--:|:--:|:--:|:--|
| 80/20 准确率（456 测试，全 58 特征管线，主评估） | 68.42% (+11.4pp) | **54.61%** (−2.6pp) | **57.24%**（收盘） | 见 §5 脚本 B1 |
| **Walk-forward（全季，1,900 测试，主评估）** | 52.2%（复现） | **50.21%** | **56.32%**（收盘） | −6.11pp，McNemar p<0.001 显著输 | `python scripts/buchdahl_ppg_walkforward.py` |
| 跨季 4/2 分割（760 测试，全管线）〔附加〕 | — | **53.29%** (−4.3pp) | **57.63%**（收盘） | 见 §5 脚本 B2 |
| 跨季 4/2（**精简 25 特征**，fair_comparison）〔附加〕 | — | **59.08%** (+1.45pp) | 57.63%（**收盘**） | `python scripts/fair_comparison.py` |
| 跨季 4/2（footystats_backtest，滚动+赛前）〔附加〕 | — | **58.55%** (+0.92pp) | 57.63%（**收盘**） | `python scripts/footystats_backtest.py` |
| Buchdahl 主胜 ROI（edge>10%，收盘 B365C） | +5.41%（不可复现） | **+3.69%** | — | 同上（summary.json） |
| 全部 outcome ROI（收盘 B365C） | +9.77%（不可复现） | **+2.12%** | — | 同上 |
| 最优单注 ROI（收盘 B365C） | +22.71%（不可复现） | **+11.95%** | — | 同上 |

**关键洞察：统一收盘基线（B365C）后，模型不跑赢市场。**
- 市场基线统一为**收盘赔率**（Bet365 收盘线 B365C，开赛前的市场共识）。不再分开盘/收盘两套口径。
- **主评估 = 全季 walk-forward + 80/20**：walk-forward（58 特征，1,900 测试）模型 **50.21%** vs 收盘 **56.32%**，**−6.11pp，McNemar p<0.001 显著跑输**；80/20 全管线 54.61% vs 57.24%（−2.6pp）。
- **附加实验 4/2**（FootyStats 免费数据到 24/25 而定的整季 hold-out）上，精简模型 +1.45pp / +0.92pp（fair_comparison / footystats_backtest），**不显著**，且非主评估测试集。
- 原论文口径问题：Track A 用收盘价、Track B 用开盘价，两套口径不一致，曾造成"赢开盘"的假象。

### 3.3 其余实验（全部干净版）

| 实验 | 论文（泄漏） | **干净** | 复现 |
|:--|:--:|:--:|:--|
| DL：LR / MLP-Deep 准确率 | 70.18% | **61.4% / 61.4%**（80/10/10 分割） | `python scripts/deep_learning_experiment.py` |
| DL：MLP-Deep 平局 F1 | **0.381**（+85% vs LR） | **0.000**（LR 0.036）——主张作废 | 同上 |
| Two-tier：Tier1（有 FS） | 69.74% | **61.84%**（58 特征，DrawF1 0.097） | `python scripts/build_ensemble_model.py` |
| Two-tier：Tier2（无 FS fallback） | 60.26% | **48.58%**（51 特征） | 同上 |
| 2025-26 回测 vs 市场 | +14.73pp | **+1.31pp**（46.05% vs 44.74% 收盘） | 同上 |
| O/U 2.5 准确率 | 67.11% (+8.6pp) | 55.48%（AUC 0.5302；低于"全押 Over"58.11% 与市场 58.55%）——**已剔除，不纳入论文** | — |
| Poisson 作为 ensemble 特征（LR +0.8pp） | 有效 | 未复跑（待定） | — |

---

## 4. 存活信号（干净后仍为正的结果）

1. **模型不跑赢收盘市场（严格二分）**：主评估（全季 walk-forward 50.21% vs 56.32% **McNemar p<0.001 显著输**；80/20 54.61% vs 57.24% 输）下模型全部跑输或持平；附加 4/2 精简模型 +1.45pp/+0.92pp 不显著。按"不显著=输"，模型未跑赢市场。
2. **FootyStats 增量真实**：fair_comparison 滚动 53.55% → +FootyStats 59.08%（+5.5pp）。
3. **Buchdahl 价值投注 ROI 仍正（收盘价 B365C）**：干净模型下主胜 +3.69%、全部 outcome +2.12%、最优单注 +11.95%（edge>10%）。注：模型整体准确率低于市场却 ROI 为正，说明利润来自特定方向的系统性偏差，需统计检验后再入论文。
4. **模型选择结论不变**：LR ≥ 树模型 ≥ DL（DL/树都不如 LR，且无平局优势）。
5. **泄漏量级本身是发现**：当场统计列抬高了 10~14pp 准确率——这是足球预测 ML 里一个有分量的方法论警示。

---

## 5. 复现命令索引

```powershell
# Track A：7 季 walk-forward + 价值投注 + 阈值扫描（50 特征）
python scripts/backtest_english4.py

# Track A：三策略价值投注（已修复 bug，含二项检验）
python scripts/value_betting_walkforward.py

# Track A：树模型对比 LR/RF/XGB（7 季 WF）【新脚本】
python scripts/tree_comparison.py

# Track B：全季 walk-forward + 每季收盘市场基线 + Buchdahl ROI（58 特征，投注赔率=收盘 B365C，summary.json 含 walkforward_vs_market）
python scripts/buchdahl_ppg_walkforward.py

# Track B：fair_comparison（25 特征，4/2 分割，59.08% vs 收盘市场 57.63%）
python scripts/fair_comparison.py

# Track B：footystats_backtest（滚动+阵容+赛前，58.55% vs 收盘市场 57.63%）
python scripts/footystats_backtest.py

# Track B：80/20 + 跨季 4/2 干净评估（58 特征全管线）【新脚本】
python scripts/trackb_clean_eval.py

# Track B：O/U 2.5【新脚本，已剔除，不纳入论文】
python scripts/over_under_experiment.py

# Two-tier ensemble（Tier1 61.84% / Tier2 48.58% / 2025-26 +0.52pp）
python scripts/build_ensemble_model.py

# DL（LR/MLP/GRU 对比，58 特征）
python scripts/deep_learning_experiment.py
```

**Track B 干净评估脚本**（`scripts/trackb_clean_eval.py`，正式入库）：

- 80/20 时序分割（456 测试）：**54.61%** vs 收盘市场 **57.24%**（−2.63pp）。
- 跨季 4/2（train 2019-2022 / test 2023-24+2024-25，760 测试，**附加**）：**53.29%** vs 收盘市场 **57.63%**（−4.34pp）；无 FS 对照 **50.66%**。
- 管线：`build_features(windows=[5,10], include_footystats=True, include_odds=False)` → `get_feature_columns` → LR(C=1.0, max_iter=4000, seed 42)。市场用 `odds_*_close` 收盘赔率去抽水后 argmax 取 favorite。

---

## 6. 不可复现的论文数字（转录无源，需重跑后替换）

- Track B 68.42% / 基线 62.28%（Exp 009b 的 71 特征配置已不存在于脚本）
- Buchdahl ROI +5.41% / +9.77% / +22.71%（既不对上泄漏复现 +11.69/+16.51/+31.47，也不对上干净收盘 +3.69/+2.12/+11.95）
- Track A 80/20 +62.8% ROI、WF +52~56% ROI、分类 70.3%/13.0%/81.8%
- 模型 vs 市场分歧 3.1x / 32.0%（fair_comparison 干净输出的是 1.8x/38.6%，见 build_overview_docx 里的另一组数字）
- O/U 67.11%、ensemble 69.74%/60.26%、DL 70.18%/DrawF1 0.381

**原则**：以上全部以"当前脚本复跑输出"为准，见 §5。

---

## 7. 关键文件

| 文件 | 说明 |
|:--|:--|
| `league/src/features.py` | 泄漏修复处（`LEAKED_MATCH_STATS` + exclude） |
| `scripts/tree_comparison.py` | 【新】树模型对比（Exp 017 干净版） |
| `scripts/over_under_experiment.py` | 【新】O/U 2.5（Exp 012 干净版） |
| `scripts/value_betting_walkforward.py` | 修复 2 个 bug（`test_df`、`season_bets`、JSON int64） |
| `scripts/backtest_english4.py` | 移除 2025/26 排除行，统一 7 季口径 |
| `outputs/buchdahl_walkforward/summary.json` | 干净 Buchdahl ROI 存档 |
| `outputs/value_betting/summary.json` | 干净 Track A 价值投注存档 |
| `outputs/ensemble/results.json` | 干净 ensemble 存档 |
| `outputs/deep_learning/` | 干净 DL 存档 |
| `outputs/footystats_backtest/` | 干净 footystats_backtest 存档 |
| `memory/leakage-fix-2026.md` | 记忆（本结论的持久化） |
