# 参考链条文档

> 目标：按实际阅读顺序记录文献如何一步一步启发本项目。本文档先只记录已经明确的推导，不提前写后续项目操作。

## 1. Suzuki & Ohmori：赛前 Ranking 作为起点

本项目最初受到 Suzuki & Ohmori 关于 FIFA/Coca-Cola World Ranking 预测世界杯成绩的研究启发。该研究使用世界杯开赛前的 FIFA 排名作为球队实力代理，检验排名是否能够解释或预测世界杯决赛圈成绩。

这个思路的优点是简单、直观，而且数据时间点清楚：FIFA ranking 在比赛前已经存在，因此适合作为赛前预测变量。

但它也有两个限制：

- FIFA ranking 主要适用于国家队，尤其是世界杯语境。
- 单一 ranking 指标不够精细，难以表达联赛球队的近期状态、主客场差异和具体比赛上下文。

因此，下一步需要寻找一个更适合联赛比赛的 rating 或历史表现建模方法。

## 2. Buchdahl：从 Ranking 转向联赛 Rating System

在这个基础上，本项目进一步阅读 Buchdahl 的 `Rating Systems for Fixed Odds Football Match Prediction`。这篇文章提供了一个更适合联赛场景的思路：不用固定的国家队 ranking，而是根据球队过去比赛表现构造 rating，再用主客队 rating 差来预测即将发生的比赛。

Buchdahl 的方法依然保持了 Suzuki & Ohmori 那种“赛前实力指标 -> 比赛结果预测”的清晰结构，但它把实力指标从 FIFA ranking 换成了联赛球队的近期表现。

## 3. ROI 思路的来源

Buchdahl 这篇文章还有一个重要启发：它不只讨论预测准确性，也把模型概率进一步用于投注回测。文章先根据 rating 推出 fair odds，再把 fair odds 和 bookmaker odds 比较。如果 bookmaker odds 高于模型给出的 fair odds，就认为可能存在 value bet。

因此，ROI 或 yield 分析可以作为赛果预测之外的一个拓展评估方向。它关注的不是“单场是否猜中”，而是模型概率和市场赔率之间是否存在可利用的定价差异。

## 4. Buchdahl `ratings.pdf` 方法论摘要

Buchdahl 的方法可以概括为六步：

1. **定义球队 rating**：rating 是一支球队相对对手的量化优势，来自过去比赛表现。
2. **选择近期表现窗口**：示例系统使用最近 6 场比赛。
3. **计算 goal superiority**：每队最近 6 场进球数减失球数，得到球队近期净胜球 rating。
4. **计算 match rating**：用主队 rating 减客队 rating，得到单场比赛的相对实力差。
5. **映射为概率和 fair odds**：在历史样本中统计不同 match rating 对应的主胜、平局、客胜比例，并用拟合曲线平滑成概率；fair odds 等于 `100 / result probability`。
6. **做 value betting 回测**：当 bookmaker odds 高于模型 fair odds 时下注 1 单位，随后按赛季统计 profit/yield，并与盲押主胜比较。

这篇文章最重要的价值不是模型复杂度，而是完整展示了一个可复现的预测-定价-投注链条：历史表现形成 rating，rating 形成概率，概率形成 fair odds，fair odds 再与市场赔率比较得到 value bet。
