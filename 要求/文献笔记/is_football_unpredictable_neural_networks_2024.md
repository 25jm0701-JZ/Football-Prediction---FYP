# Is Football Unpredictable? Predicting Matches Using Neural Networks - 方法论笔记

## 基本信息

- 原文件: `要求/文献/Is_Football_Unpredictable_Predicting_Matches_Using.pdf`
- 标题: Is Football Unpredictable? Predicting Matches Using Neural Networks
- 作者: Luiz E. Luiz, Gabriel Fialho, Joao P. Teixeira
- 发表: Forecasting, 2024
- 主题: 使用深度神经网络和特征工程预测足球赛果

## 研究目标

文章试图回答“足球是否可预测”。作者用公开比赛统计数据构建深度神经网络，预测主胜、平局、客胜，并分析哪些特征对预测最重要。

## 数据

数据来自 WhoScored。

- 总比赛数: 24,760
- 联赛数: 13
- 时间跨度: 不同联赛约 2 到 10 年
- 测试集: 2,589 场，主要来自 2018 和 2019 年

联赛包括:

- Argentina First Division
- Brazilian Championship
- China Super League
- England Championship
- England Premier League
- France League
- Germany Bundesliga
- Italy Serie A
- Eredivisie
- Portugal NOS League
- Russia Premier League
- Spain League
- Turkey Super Lig

原始特征包括:

- 主客场
- 进球
- 球员评分
- 射门
- 射正
- 传球成功
- 空中对抗成功
- 盘带成功
- 抢回球权
- 控球率
- 球队评分
- 拦截
- 丢失球权
- 角球
- 联赛

## 数据组织与预处理

文章比较了不同历史窗口和数据组织方式:

- 最近 20 场简单平均
- 最近 7 场简单平均
- 指数加权移动平均

结果显示，最近 20 场简单平均表现最好。

预处理方面比较:

- normalization
- standardization

结果显示 min-max normalization 优于 standardization。

## 特征工程

作者生成了一系列新特征，用来帮助神经网络理解球队强弱关系，而不是只输入原始统计。

重要特征包括:

- goal difference
- pi-rating home
- pi-rating away
- relative attacking power
- relative defending power
- relative midfield power
- goal expectancy for home team
- goal expectancy for away team
- shot accuracy
- relative ball loss
- relative corner power
- relative playmaking power

## Pi-rating

pi-rating 是文中使用的重要基准特征，来自 Constantinou and Fenton。它是一种动态球队评分，只根据预测进球差与实际进球差之间的误差更新。

特点:

- 每队有 home pi-rating 和 away pi-rating。
- 最近结果比旧结果更重要。
- 胜负比单纯扩大净胜球更重要。
- 更新中使用对数函数压缩大比分误差。

文章先验证 pi-rating 相比原始数据的重要性，再与作者提出的新特征比较。

## 特征重要性分析

文章使用两种思路:

1. 决策树特征重要性，用于初步比较原始变量和 pi-rating。
2. permutation/shuffling importance，即随机打乱某特征后观察模型准确率下降程度。

最终最重要的特征为:

| 特征 | 重要性 |
|---|---:|
| Relative Defending Power | 0.85 |
| Relative Playmaking Power | 0.81 |
| Relative Midfield Power | 0.77 |
| Goal Expectancy Home Team | 0.77 |
| Pi-rating Home Team | 0.73 |
| Pi-rating Away Team | 0.73 |

一个关键结论是: 作者提出的 relative defending、playmaking、midfield power 和 home goal expectancy 比 pi-rating 更重要。

## 神经网络模型

最终模型为深度神经网络:

```text
Layer 1: 100 neurons
Layer 2: 80 neurons
Layer 3: 5 neurons
Output: 3 neurons
```

输出对应:

- home win
- draw
- away win

模型使用 dropout 减少 overfitting，并用 grid search 寻找最优超参数。

## 主要结果

整体测试集:

- 2,589 场比赛
- 总准确率: 52.8%

按联赛:

- Portugal NOS League: 60.9%
- Turkey Super Lig: 58.1%
- English Premier League: 57.2%
- Italy Serie A: 56.5%
- German Bundesliga: 41.3%

高置信预测:

- 当模型给主队胜率 > 50% 时，准确率 78.8%。
- 当模型给主队胜率 > 60% 时，准确率 80.3%。
- 当模型给客队胜率 > 50% 时，准确率 76.6%。

平局预测较弱:

- draw accuracy 约 32%。

## 文章结论

作者认为足球不是完全不可预测的。神经网络可以从历史统计中学习到影响结果的主要因素，在整体上明显高于随机三分类 33.3% 的水平，并在高置信场景下达到较高准确率。

## 方法论优点

- 数据规模较大，覆盖 13 个联赛。
- 同时使用球队层面、球员层面和派生特征。
- 明确比较历史窗口、归一化方法和特征重要性。
- 输出可解释的特征重要性结果。
- 高置信度子集分析对实际投注/决策很有参考价值。

## 方法论局限

- 主要评价指标是 accuracy，缺少概率校准、log loss、Brier score 等概率评价。
- 平局预测能力较弱，这是足球三分类预测中的常见难点。
- 高置信子集准确率高，但样本占比和收益情况需要进一步说明。
- 不直接与 bookmaker odds 做系统比较。
- WhoScored 数据采集和特征可复现性可能是问题。

## 对 FYP 的启发

这篇适合支持机器学习/深度学习路线，尤其是:

- 不要只用原始统计，要做相对强弱特征。
- 最近 20 场平均可能是一个合理窗口。
- 需要单独处理平局难预测问题。
- 可以报告 feature importance，让模型更可解释。
- 高置信预测可以作为投注或推荐策略筛选条件，但必须再用 ROI 或 log loss 验证。

