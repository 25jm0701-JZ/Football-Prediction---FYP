# Betting markets for English Premier League results and scorelines - 方法论笔记

## 基本信息

- 原文件: `要求/文献/reade_singleton_scorelines.pdf`
- 标题: Betting markets for English Premier League results and scorelines: evaluating a forecasting model
- 作者: J. James Reade, Carl Singleton, Leighton Vaughan Williams
- 发表: Economic Issues, 2020
- 研究对象: English Premier League 2016/17 与 2017/18 赛季
- 主题: 比较博彩公司赔率与统计模型在赛果、准确比分、净胜球、总进球市场上的预测表现

## 研究问题

文章关注两个预测来源:

1. 博彩市场赔率隐含的概率预测。
2. 一个标准统计模型生成的概率预测。

核心问题是:

- 博彩市场对 EPL 赛果和准确比分的预测是否有效率？
- 标准统计比分模型是否比博彩公司赔率更好？
- 如果用该模型下注，能否获得稳定正收益？
- 比分市场是否存在 favourite-longshot bias？

## 数据

样本为 EPL 两个赛季:

- 2016/17: 380 场
- 2017/18: 380 场
- 总计: 760 场

比赛结果数据来自 Soccerbase.com。

赔率数据来自 Oddsportal.com，包括:

- 51 家博彩公司
- 1 个 betting exchange: Matchbook
- 赔率时间点: 开赛前

分析的市场包括:

- match result: 主胜、平局、客胜
- exact scoreline: 准确比分
- margin of victory: 净胜球差
- total goals: 总进球数

## 赔率转概率

文章将十进制赔率转换为隐含概率:

```text
p = 1 / d
```

其中:

- `d` 是 decimal odds
- `p` 是原始隐含概率

由于博彩公司赔率包含 overround，即所有结果的隐含概率之和大于 1，作者使用简单归一化修正:

```text
修正后概率 = 某结果原始隐含概率 / 所有结果原始隐含概率之和
```

文中提到:

- 赛果市场平均 overround 约 4%。
- 准确比分市场平均 overround 约 12%。

这意味着比分市场对下注者更不友好，模型必须明显优于市场概率，才可能通过简单策略盈利。

## 统计模型: 双变量泊松回归

文章使用标准的 bivariate Poisson regression model 预测比分。该模型来源于 Maher (1982)、Dixon and Coles (1997)、Karlis and Ntzoufras (2003, 2005) 等足球比分建模传统。

设第 `i` 场比赛:

- 主队进球数为 `h_i`
- 客队进球数为 `a_i`

模型假设主队和客队进球共同服从双变量泊松分布:

```text
(h_i, a_i) ~ BP(lambda_i1, lambda_i2, lambda_i3)
```

其中可以理解为:

```text
h_i = X_i1 + X_i3
a_i = X_i2 + X_i3
```

- `X_i1`: 主队自身进球强度因素
- `X_i2`: 客队自身进球强度因素
- `X_i3`: 双方共享的比赛环境因素，例如天气、赛程、月份等

回归形式:

```text
log(lambda_ik) = w'_ik beta_k, k = 1, 2, 3
```

## 模型变量

模型包含以下解释变量:

- 双方球队固定效应，用于捕捉球队进攻和防守强度。
- 星期几固定效应。
- 月份固定效应。
- 是否处于国际比赛日之后。
- 滞后的联赛排名。
- 近期状态。
- 动态 Elo rating。
- Elo implied match outcome probabilities。
- 是否仍在 FA Cup。
- 是否仍有机会进入联赛前二。
- 是否刚踢完欧洲赛事回到国内联赛。

这些变量的目的，是把球队实力、状态、赛程压力、杯赛影响和比赛环境共同纳入进球过程。

## 模型估计与预测

模型使用 maximum likelihood estimation。

估计方式:

- 在每一轮 EPL 比赛前估计模型。
- 使用过去一个日历年的比赛作为训练数据。
- 每轮包含 20 支球队的 10 场比赛。
- 估计出的 `lambda` 参数用于预测即将到来的比赛。

模型输出:

- 每个可能比分的概率。
- 主胜、平局、客胜概率，可由比分概率求和得到。
- 期望总进球数。
- 双方球队各自的进球率。

## Point forecast 的两种构造

文章不只使用概率预测，还从模型中生成比分 point forecast，用于投注策略。

第一种是 unconditional forecast:

```text
直接选择模型认为概率最高的准确比分
```

第二种是 conditional forecast:

```text
先判断最可能的赛果类型，再在该赛果类型内部选择概率最高的比分
```

例如:

- 如果所有主胜比分概率之和大于平局和客胜，则先确定预测赛果为主胜。
- 然后只在主胜比分中选择概率最高的准确比分。

这样做的原因是，足球中最常见的单个比分可能是 1-1，但最常见的赛果类型通常是主胜。因此 unconditional 和 conditional 可能给出不同 picks。

## 评价对象

文章把比分视为最基础 outcome，其他 outcome 都是比分的函数。

定义如下:

```text
scoreline: s_i = (h_i, a_i)
result: r_i = r(s_i)
margin: m_i = h_i - a_i
total goals: t_i = h_i + a_i
```

赛果编码为:

```text
r_i = 0   if h_i < a_i
r_i = 0.5 if h_i = a_i
r_i = 1   if h_i > a_i
```

## ROI 评价

文章用简单投注策略评估模型是否能产生经济收益。

对每场比赛下注 1 unit，如果预测 outcome 命中，则获得对应 decimal odds，否则损失本金。

公式:

```text
ROI_i = d_i * 1{s_i = predicted_s_i} - 1
```

说明:

- 准确比分、总进球、净胜球市场使用收集到的博彩公司平均赔率。
- 赛果市场使用样本中的 best available bookmaker odds。

这个设计让文章不仅评价预测准确性，也评价市场是否可能存在可利用的低效率。

## 回归评价: Mincer-Zarnowitz 框架

文章使用 Mincer and Zarnowitz (1969) 的 forecast evaluation regression。

设:

- `p_hat_ij`: 第 `i` 场比赛中 outcome `j` 的预测概率
- `y_ij`: outcome `j` 是否实际发生，发生为 1，否则为 0

弱效率检验:

```text
y_ij = alpha + beta * p_hat_ij + error_ij
```

若预测有效率，需要满足:

```text
alpha = 0
beta = 1
```

强效率检验加入其他公开信息:

```text
y_ij = alpha + beta * p_hat_ij + z'_i gamma + error_ij
```

若强效率成立，需要满足:

```text
alpha = 0
beta = 1
gamma = 0
```

加入的公开信息包括:

- Elo 预测
- 历史比分频率
- 主队当前积分
- 主队近 6 场积分
- 主客队积分差
- 主客队近期状态差

## Forecast encompassing

文章进一步用 forecast encompassing 检验两个预测来源谁包含更多信息。

核心思想:

- 如果模型预测可以解释博彩公司预测误差，而博彩公司预测不能解释模型预测误差，则模型 encompass 博彩公司。
- 如果两者都能解释对方误差，则二者线性组合可能优于单一预测。

文章结果显示:

- 模型概率显著 encompass 博彩公司比分概率。
- 也就是说，在准确比分预测上，标准统计模型的信息含量优于 bookmaker odds-implied forecasts。

## 主要实证结果

### 1. 比分赔率存在 favourite-longshot bias

博彩公司在准确比分市场上显著高估冷门比分的概率，例如 4-4 这类罕见比分；同时相对低估常见比分，例如 1-0、1-1。

这属于 favourite-longshot bias:

```text
longshot outcomes are overvalued / overestimated
favourite outcomes are undervalued / underestimated
```

但这种偏误在 match result 市场中不明显。

### 2. 标准统计模型也有偏误

模型对准确比分预测并非完全有效率。文章认为模型有 reverse favourite-longshot bias，倾向于低估高比分和意外比分，可能与泊松假设有关。

### 3. 赛果市场比比分市场更有效率

无论是博彩公司预测还是模型预测，对主胜、平局、客胜这三个赛果 outcome 来说，效率检验一般不拒绝强效率假设。

这说明 EPL 赛果市场在样本期内整体较有效率。

### 4. 模型比分概率优于博彩公司比分概率

Forecast encompassing 结果表明，标准双变量泊松模型在比分概率预测上优于博彩公司赔率隐含概率。

但“预测概率更好”不等于“简单下注能赚钱”。

### 5. 简单投注策略无法稳定盈利

表 7 的 ROI 显示:

| 赛季 | Forecast | Result | Scoreline | Margin | Total Goals |
|---|---|---:|---:|---:|---:|
| 2016/17 | Unconditional | -3.4% | -10.8% | -3.9% | -1.9% |
| 2016/17 | Conditional | 12.7% | -5.2% | 3.1% | -1.6% |
| 2017/18 | Unconditional | 4.8% | -25.8% | -6.7% | -6.5% |
| 2017/18 | Conditional | -0.2% | -26.6% | -6.5% | -7.6% |

结论是:

- 赛果投注有少量正收益迹象，但不稳定。
- 准确比分投注整体亏损明显。
- 净胜球和总进球市场也没有稳定正收益。
- 因此不能说明这些市场存在可由该简单模型稳定利用的无效率。

## 文章结论

文章的结论可以概括为:

- 准确比分市场存在显著 mispricing，尤其是 favourite-longshot bias。
- 标准统计模型比分概率在统计意义上优于 bookmaker odds-implied forecasts。
- 但由于比分市场 overround 高、模型也有偏误，简单投注策略无法稳定产生正收益。
- 赛果市场整体更接近有效率。
- 未来如果使用更好的模型，尤其结合 betting exchange 的赔率，比分市场可能仍有研究和盈利空间。

## 方法论优点

- 明确区分 probability forecast 与 point forecast。
- 同时比较统计模型和市场赔率。
- 不只看 accuracy，还用 ROI、效率检验和 forecast encompassing 多角度评价。
- 使用比分作为基础 outcome，再自然推导赛果、净胜球、总进球。
- 使用双变量泊松处理主客队进球相关性，比独立泊松更合理。
- 使用 rolling estimation，更接近真实预测场景。

## 方法论局限

- 样本只有两个 EPL 赛季，外推性有限。
- 使用平均赔率或 best odds 的设定会影响 ROI 结论。
- 简单投注策略没有真正做 value selection，只是按模型 point forecast 下注。
- 模型是 benchmark bivariate Poisson，可能不足以捕捉高比分、战术变化、阵容伤停等因素。
- 比分 outcome 很稀疏，许多比分发生频率极低，回归和 ROI 评价都受到稀疏性的影响。

## 对 FYP 的启发

这篇文章非常适合用于你的 FYP 方法论和评价体系设计，尤其是:

- 如果研究足球预测，应该区分赛果预测和比分预测。
- 比分预测可以作为更底层的建模任务，因为赛果、净胜球、总进球都是比分的函数。
- 赔率可以被看作市场概率预测，但需要处理 overround。
- 模型评估不应只用 accuracy，也可以使用:
  - calibration / bias
  - Mincer-Zarnowitz regression
  - forecast encompassing
  - ROI
  - favourite-longshot bias
- 即便模型统计上优于博彩公司赔率，也不一定能产生正收益，因为市场 overround 和投注策略会吞掉优势。

如果后续项目要建立一个比赛预测模型，可以借鉴本文结构:

```text
historical match data + team variables + Elo/form controls
-> bivariate Poisson scoreline model
-> scoreline probabilities
-> derived result / margin / total goals probabilities
-> compare with bookmaker odds
-> test efficiency + forecast encompassing + ROI
```
