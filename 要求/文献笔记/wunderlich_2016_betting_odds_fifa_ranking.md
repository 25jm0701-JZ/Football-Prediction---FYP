# Wunderlich 2016 - Betting Odds 与 FIFA Ranking 预测质量笔记

## 基本信息

- 原文件: `要求/文献/Wunderlich_2016_Analysis-of-the-predictive-qualities-of-betting-odds-and-FIFA-World-Ranking.pdf`
- 标题: Analysis of the predictive qualities of betting odds and FIFA World Ranking: evidence from the 2006, 2010 and 2014 Football World Cups
- 作者: Fabian Wunderlich, Daniel Memmert
- 发表: Journal of Sports Sciences, 2016
- 主题: 比较 Betfair 赔率与 FIFA World Ranking 对世界杯比赛的预测能力

## 研究目标

文章提出一个框架，把 FIFA World Ranking 从单纯排名转换为三结果概率预测，然后与投注市场概率预测进行比较。核心问题是: 排名系统能否生成接近甚至优于 betting odds 的预测？

## 数据

使用 2006、2010、2014 三届男足世界杯，共 183 场比赛。

预测信息来源:

- Betfair exchange 赔率，记为 BET。
- 各届世界杯前 FIFA World Ranking，记为 RANK。

淘汰赛按常规时间结果处理，因此平局仍是可能结果。

## 研究假设

文章提出三条假设:

1. BET 和 RANK 都应比无信息基准预测更准确。
2. FIFA Ranking 在 2006 年规则改革后预测质量应提高。
3. BET 应比 RANK 更好，因为赔率理论上包含更多即时信息，如主场优势、伤病、阵容、赛程形势等。

## BET 概率构造

从 Betfair 取每个赛果中交易量最高的 odds。

赔率转概率:

```text
p = 1 / odds
```

再归一化，使主胜、平局、客胜概率和为 100%。

文章选择 Betfair 而不是传统博彩公司，是因为交易所赔率没有传统 bookmaker margin，更接近市场概率。

## RANK 概率构造

FIFA Ranking 本身不是概率预测，因此作者提出两步转换:

1. 把 ranking points 差异转为两队 expected goals。
2. 用双变量泊松分布把 expected goals 转为主胜、平局、客胜概率。

设:

- `ptsMAX`: 排名积分较高球队的积分
- `ptsMIN`: 排名积分较低球队的积分
- `expMAX`: 强队 expected goals
- `expMIN`: 弱队 expected goals
- `g_hat`: 上一届世界杯平均总进球数
- `alpha`: 调整排名积分差异强度的参数

模型思想:

```text
expMAX / (expMAX + expMIN)
```

由两队排名积分比例和 `alpha` 决定。

同时约束:

```text
expMAX + expMIN = g_hat
```

也就是说，排名只决定两队 expected goals 的分配比例，总进球期望由上一届世界杯平均进球数给定。

随后使用 Karlis and Ntzoufras 的 bivariate Poisson distribution，把 expected goals 转成三结果概率。

## 示例

2010 世界杯决赛 Spain vs Netherlands:

- 使用 2006 世界杯平均总进球数，预期总进球为 2.25。
- FIFA points: Spain 1565, Netherlands 1231。
- `alpha = 1.382`。
- 转换得到 expected goals: Spain 1.434, Netherlands 0.816。
- 双变量泊松转为结果概率: Spain 51.48%, Draw 28.64%, Netherlands 19.88%。

## 评价指标

文章使用两类评价:

### 1. Model-free prediction

只判断谁是 favourite、谁是 outsider。

评价标准不是胜场数，而是 favourite 与 outsider 的进球数。理由:

- 净胜球/进球差比胜负更能反映球队实力。
- 进球数量比比赛胜负样本更多，更容易做显著性判断。
- 进球可自然分为 favourite goals 与 outsider goals。

### 2. Percental prediction

评价主胜、平局、客胜概率预测。

使用 log-likelihood，与无信息 benchmark 比较。benchmark 用上一届世界杯平局频率估计平局概率，其余概率平均分给两队胜。

## 主要结果

- BET 和 RANK 一般都优于无信息预测。
- 2006 年 RANK 表现明显弱于 BET，且 RANK 未能显著优于 benchmark。
- 2010 年 BET 略优于 RANK，但差异不显著。
- 2014 年 RANK 显著优于 BET。
- 合并 2010 与 2014 数据后，RANK 仍显著优于 BET。
- 文章认为 FIFA Ranking 在 2006 年计算规则改革后预测质量明显提高。

## 文章结论

文章的反直觉结论是: 虽然 betting odds 理论上包含更多即时信息，但改革后的 FIFA Ranking 在世界杯场景中可能具有很强预测能力，甚至在 2014 年和 2010-2014 合并样本中超过 Betfair 市场预测。

## 方法论优点

- 将 ranking points 转成 expected goals，再转成概率，解决了排名不能直接输出概率的问题。
- 同时评价 favourite 判断和三结果概率预测。
- 使用 log-likelihood 而不是只看 accuracy，更适合概率预测。
- 区分 2006 前后 FIFA Ranking 规则变化。

## 方法论局限

- 新规则下样本只有 2010 和 2014 两届，共 128 场，样本偏小。
- FIFA Ranking 到 expected goals 的转换依赖若干假设，例如总进球等于上一届均值。
- ranking 无法捕捉伤病、阵容、战术、主场优势等即时信息。
- 世界杯比赛与友谊赛、预选赛结构不同，结果未必能外推。

## 对 FYP 的启发

这篇非常适合支持“FIFA Ranking 可以转为概率特征”的方法设计。可借鉴:

```text
FIFA points difference
-> expected goals allocation
-> bivariate Poisson
-> W/D/L probabilities
-> log-likelihood evaluation
```

如果你的项目是世界杯预测，这篇比联赛文献更贴近场景。它也提醒我们: 对国家队比赛，FIFA Ranking 或 Elo 这类累积评分可能比俱乐部联赛中的短期赔率更稳定。

