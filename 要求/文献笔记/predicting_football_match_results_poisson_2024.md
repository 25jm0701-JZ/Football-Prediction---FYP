# Predicting Football Match Results Using a Poisson Regression Model - 方法论笔记

## 基本信息

- 原文件: `要求/文献/Predicting_Football_Match_Resu.pdf`
- 标题: Predicting Football Match Results Using a Poisson Regression Model
- 作者: Konstantinos Loukas, Dimitrios Karapiperis, Georgios Feretzakis, Vassilios S. Verykios
- 发表: Applied Sciences, 2024
- 数据: English Premier League 2022/23
- 主题: 用双泊松/泊松回归预测足球比分和赛果

## 研究目标

文章希望用球队历史表现估计主客队 expected goals，再用泊松分布计算具体比分概率和主胜/平/客胜概率。

## 数据

- 数据来源: Kaggle
- 联赛: English Premier League 2022/23
- 比赛数: 380
- 总进球: 1084
- 场均进球: 2.85
- 主队进球: 621，场均 1.63
- 客队进球: 463，场均 1.22
- 主胜: 184 场，48.4%
- 客胜: 109 场，28.7%
- 平局: 87 场，22.9%

## 核心模型

文章假设主队和客队进球数分别服从泊松分布，并且两队进球相互独立。

单队进球概率:

```text
P(k) = exp(-lambda) * lambda^k / k!
```

双泊松比分概率:

```text
P(Home=k, Away=h) = P(Home=k) * P(Away=h)
```

赛果概率由比分概率求和得到:

```text
P(Home win) = sum P(k,h), k > h
P(Draw) = sum P(k,h), k = h
P(Away win) = sum P(k,h), k < h
```

## Expected goals 估计

主队 expected goals:

```text
log(lambda_home) = mu + mu_home + att_home + def_away
```

客队 expected goals:

```text
log(lambda_away) = mu + att_away + def_home
```

其中:

- `mu`: 客队平均进球的 log average。
- `mu_home`: 主场优势，等于主场平均进球和客场平均进球的 log 差。
- `att_team`: 球队攻击能力。
- `def_team`: 球队防守参数。

文章假设:

- 主场优势对所有球队相同。
- 球队攻击/防守能力不区分主场和客场。
- 主客队进球独立。

## 参数计算步骤

作者提出的算法:

1. 计算所有比赛主队和客队进球数的 log average。
2. 对每支球队，计算所有比赛的平均进球和平均失球。
3. `mu` 取客队平均进球的 log average。
4. `mu_home` 取主队与客队平均进球 log 差。
5. 攻击和防守参数通过球队进球/失球 log average 减去 `mu` 得到。
6. 代入主客队 expected goals 公式。
7. 使用泊松分布生成比分矩阵。
8. 由比分矩阵求主胜、平局、客胜概率。

## 独立性检验

为了验证主客队进球独立假设，作者使用 chi-square test。

结果:

- Chi-square = 20.37
- df = 16
- p-value = 0.2039

在 0.05 显著性水平下，不拒绝“主客队进球无关系”的原假设。因此作者认为独立性假设在该数据中可接受。

## 示例

Manchester City vs Liverpool:

```text
xG_ManCity = 3.4
xG_Liverpool = 1.4
```

模型给出的最高概率比分为 3-1，概率约 7.6%。实际结果为 4-1。

赛果概率:

- Manchester City win: 74.6%
- Draw: 13.1%
- Liverpool win: 12.3%

## 验证方式

文章使用三类测试:

1. 从 380 场中随机抽 20%，即 76 场，比较预测进球与实际进球差。
2. 对 Manchester City、Fulham、Southampton 三支不同水平球队模拟完整赛季 38 场总进球。
3. 用前 19 轮估计参数，预测第 20-24 轮的 50 场比赛，作为独立样本测试。

结果概括:

- 76 场随机样本中，大多数预测误差在正负 1 球以内。
- 独立样本 50 场中，主客队进球误差大多在 0 到 2 球，正负 1 球以内的比例超过 70%。
- 模型对强弱差距明显的比赛表现较好，但对爆冷结果表现较差。

## 主要局限

文章明确指出模型低估 0-0 比分。

2022/23 EPL 中:

- 23/380 场为 0-0，约 6%。

普通泊松模型无法很好处理“零值过多”的问题，因此作者建议未来使用 zero-inflated Poisson model。

其他局限:

- 没考虑伤病、停赛、阵容变化。
- 没考虑近期比赛比旧比赛更重要。
- 没考虑赛季末球队动机变化。
- 用整季数据估计部分测试存在信息泄漏风险，虽然作者另做了独立样本测试。

## 对 FYP 的启发

这篇可作为可实现的 Poisson baseline。适合用于:

- 由历史进失球估计 expected goals。
- 生成比分概率矩阵。
- 再聚合为赛果概率。
- 与 Elo/FIFA ranking 或机器学习模型比较。

如果用于世界杯，应注意国家队样本少，可以把 Poisson 模型和 FIFA/Elo rating 结合，而不是只用单届赛事进失球估计参数。

