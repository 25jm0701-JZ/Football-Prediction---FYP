# 足球赛果预测 —— 项目说明

基于逻辑回归与滚动统计特征的英格兰职业足球赛果预测系统。
附：基于 Buchdahl (2003) 框架的价值投注改进、World Cup 预测子项目、深度学习等模型对比。

> **论文概要基准**：`论文点_更新版.md`；**全部实验过程**：`docs/experiment_log.md`。
> 本文档为项目总览，数字若与两者冲突，以 `论文点_更新版.md` 为准。

---

## 双轨结构

| | Track A（主模型） | Track B（增强） |
|:--|:----------------|:---------------|
| **数据** | E0-E3 四级联赛，11,952 场，6 季 | 英超，2,280 场，6 季 |
| **特征** | 63 维（滚动统计+射门+交锋+联赛等级） | 71 维（62 基础 + 9 FootyStats） |
| **外部数据** | 无 | FootyStats API（PPG/xG/控球率） |
| **定位** | 核心结果：证明模型优于市场 | 改进 Buchdahl 评分系统 |
| **参考论文** | Atta Mills et al. (2024) + Buchdahl (2003) | Buchdahl (2003) |

---

## Track A：主模型

### 结果

| 验证方式 | 模型准确率 | 市场准确率 | Δ | 价值投注 ROI* |
|:--------|:---------:|:---------:|:-:|:---:|
| 80/20 分割 | **58.60%** | 49.70% | **+8.90pp** | +62.8% |
| Walk-forward（逐季滚动） | 57.43% | — | — | +55.0% |

*Edge>30% 阈值、最优单注策略。80/20 与 walk-forward 数据量不同，ROI 不可直接比较（见"价值投注方法论"）。

**结论：即使无任何外部数据源，纯滚动统计 LR 在英格兰全级别联赛上稳定优于市场 8.9pp（80/20 分割）。**

### 数据

| 联赛 | 代码 | 场次（6 季） |
|:----|:---:|:----:|
| 英超 | E0 | 2,280 |
| 英冠 | E1 | 3,312 |
| 英甲 | E2 | 3,160 |
| 英乙 | E3 | 3,200 |
| **合计** | | **11,952** |

赛季范围 2019/20–2024/25。

### 特征（63 维）
- 滚动球队状态（进球、积分、胜平负率 5/10 场窗口）：28
- 射门数据（射门、射正、命中率、转化率）：16
- 历史交锋：5
- 原始统计（角球、犯规、黄红牌）：13
- 联赛等级标识（E0=0 至 E3=3）：1
- **不含任何赔率特征**（赔率仅训练后作对比基准）

### 分类表现（80/20 分割）

| 类别 | 模型准确率 | 市场准确率 | Δ |
|:----|:---------:|:---------:|:-:|
| 客胜 | **70.3%** | 47.4% | **+22.8pp** |
| 平局 | **13.0%** | 0.0% | +13.0pp |
| 主胜 | **81.8%** | 80.2% | +1.7pp |

### 价值投注（80/20 分割，Edge 阈值扫描）

| 阈值 | 投注数 | 胜率 | ROI |
|:---:|:-----:|:----:|:---:|
| 0% | 2,798 | 55.9% | +50.6% |
| 10% | 2,704 | 56.5% | +52.8% |
| 20% | 2,414 | 58.0% | +58.6% |
| **30%** | **2,041** | **59.0%** | **+62.8%** |
| 50% | 1,381 | 61.8% | +84.4% |

---

## Track B：FootyStats 增强

### 结果

| 配置 | 准确率 | Δ vs 基线 |
|:----|:-----:|:---------:|
| 滚动特征基线 | 62.28% | — |
| **+FootyStats（最终模型，71 维）** | **68.42%** | **+6.14pp** |
| 市场赔率（Bet365） | 57.02% | — |

### Buchdahl 改进对照

| 策略 | ROI | 说明 |
|:----|:---:|:------|
| Buchdahl 原文（2001/02） | +2%~+10% | 简单进球差评分 |
| 本论文：Buchdahl 策略（仅主胜） | **+5.41%** | LR + 71 维 |
| 本论文：全部 outcome | **+9.77%** | LR + 71 维，多注策略 |
| 本论文：最优单注 | **+22.71%** | LR + 71 维，每场选最佳 |

---

## 模型 vs 市场分歧（PL 测试集 456 场）

分歧比例：**32.0%**
分歧时模型正确率：**52.7%**
分歧时市场正确率：17.1%
模型/市场正确比：**3.1x**

---

## 价值投注方法论

### Edge（优势）
```
Edge = P_model / P_market - 1
```
正 Edge = 模型认为市场低估了该结果。P_model 为 LR 预测概率，P_market 为 Bet365 收盘赔率去抽水后的概率。

### Threshold（阈值）
Edge 超过阈值时才下注。阈值越高→投注越少→单注质量越高。

### 三种策略
| 策略 | 做法 |
|:----|:------|
| Buchdahl-style | 只看主胜，Edge>阈值则投 |
| All outcomes | 所有结果中 Edge>阈值的都投（一场可多注） |
| Best per match | 每场只选 Edge 最高的一个结果投 |

阈值 30% 时三种策略 ROI 均在 +52%~+56%（walk-forward），策略选择对 ROI 影响不大。

### 案例：利物浦 vs 水晶宫
- 模型看客胜概率 28.1%，市场仅 8.6%（赔率 11.0）
- Edge = 227%
- 实际：水晶宫客胜 ✅

**注意**：ROI 依赖验证方法。80/20 分割 ROI 偏高（+62.8%），walk-forward 更贴近真实部署（+55.0%）；Track A 与 Track B 的 ROI 口径不同，不可直接横向比较。

---

## World Cup 预测子项目（`world_cup/`）

独立于双轨结构的第二个子项目：Softmax 排名模型预测世界杯淘汰赛赛果。

- **模型**：SoftmaxModel，3 基础特征（FIFA 排名差分、东道主优势、淘汰赛标识）+ 4 球员级特征（前锋进球参与、中场助攻、后卫失球、阵容年龄）
- **数据**：2014/2018/2022 三届世界杯 + 2026 淘汰赛赛程
- **结果**（leave-one-year-out）：LogLoss **0.978 → 0.917（+6.1%）**，准确率 57.8% → 64.1%
- 关键发现：FIFA 排名只反映国家队层面，球员个人能力提供正交增量信号（与排名相关性仅 0.21~0.37）
- 球员特征加权聚合（按分钟加权）另带来 +0.92% LogLoss 改善；树模型在小样本（192 场）下全部过拟合，Softmax 仍是最优

---

## 模型方法对比（`docs/experiment_log.md` 全记录）

| 实验 | 结论 |
|:----|:----|
| **深度学习**（PyTorch FNN/GRU vs LR，同 71 特征） | FNN 准确率与 LR 持平（70.18%）；但 **MLP-Deep 平局 F1 +85%**（0.381 vs 0.206）；LR 校准更好（LogLoss/Brier 更优），仍是主模型 |
| **Poisson 评分**（Goalence 风格，4 参数/队） | 纯模型仅 48-51%，不可独立用；作为 ensemble 特征给 LR +0.8pp（Draw F1 0.226→0.308） |
| **树模型**（XGBoost/RF vs LR，E0-E3 全量） | LR 最优：57.43% > XGB 56.04% > RF 55.55%；XGB 平局 F1 略高但总准确率下降 |
| **Two-tier ensemble** | Tier1 有 FS 69.74% / Tier2 无 FS fallback 60.26%；2024-25 回测 380 场 **+14.73pp** vs 市场 |
| **O/U 2.5 进球预测** | LR + FS 67.11%，超市场 8.6pp |

---

## 快速开始

```powershell
# Track A：英格兰 4 级联赛回测 + 价值投注
python scripts/backtest_english4.py

# Track B：Buchdahl 改进对照
python scripts/buchdahl_ppg_walkforward.py

# 深度学习对比
python scripts/deep_learning_experiment.py

# Two-tier ensemble（有/无 FootyStats 双模型）
python scripts/build_ensemble_model.py

# 论文生成（.docx）
python scripts/build_partial_thesis.py
```

## 核心脚本

| 脚本 | 对应 | 说明 |
|:----|:----|:------|
| `scripts/backtest_english4.py` | Track A | E0-E3 回测 + 价值投注阈值扫描 |
| `scripts/buchdahl_ppg_walkforward.py` | Track B | Buchdahl 对照 |
| `scripts/footystats_backtest.py` | Track B | FootyStats 特征量化 |
| `scripts/deep_learning_experiment.py` | DL | PyTorch FNN/GRU vs LR |
| `scripts/poisson_experiment.py` | Poisson | Goalence 风格评分模型 |
| `scripts/experiment_player_features.py` | 球员特征 | 球员级特征实验 |
| `scripts/build_ensemble_model.py` | Two-tier | 双层级模型 + 回测 |
| `scripts/error_analysis.py` | 误差分析 | 模型 vs 市场多维对比 |
| `scripts/build_partial_thesis.py` | 论文生成 | .docx 格式 |

## 参考文献

1. Buchdahl, J. (2003). Rating Systems for Fixed Odds Football Match Prediction.
2. Reade, J., Singleton, C. & Vaughan Williams, L. (2020). Betting markets and statistical models.
3. Luiz et al. (2024). A deep learning approach for football match prediction.
4. Atta Mills et al. (2024). Comparing machine learning models for football prediction.
