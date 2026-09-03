# 足球预测文献总整理矩阵

## 文献分类

| 类别 | 文献 |
|---|---|
| Rating / Ranking baseline | Buchdahl 2003; Suzuki & Ohmori 2008; FIFA Ranking Rules; Wunderlich & Memmert 2016 |
| Poisson / Scoreline model | Loukas et al. 2024; Reade et al. 2020; Wunderlich & Memmert 2016 |
| Betting market efficiency | Buchdahl 2003; Reade et al. 2020; Wunderlich & Memmert 2016 |
| Machine learning / Deep learning | Luiz et al. 2024; Atta Mills et al. 2024 |
| FIFA / World Cup specific | Suzuki & Ohmori 2008; Wunderlich & Memmert 2016; FIFA Ranking Rules |

## 总览矩阵

| 文献 | 数据 | 方法 | 评价指标 | 主要结论 | 对 FYP 的用途 |
|---|---|---|---|---|---|
| Buchdahl 2003, Rating Systems | 英格兰联赛 1993/94-2000/01，回测 2001/02 | 最近 6 场净胜球 rating，评分差映射概率，拟合 fair odds | 历史频率、拟合 R2、ROI/yield | 简单 rating 可生成主胜概率和 value bet；中间评分区间更可靠 | 最基础 baseline；说明概率预测和赔率比较 |
| Reade, Singleton & Vaughan Williams 2020 | EPL 2016/17-2017/18，760 场，Oddsportal 赔率 | 双变量泊松比分模型；赔率转概率；Mincer-Zarnowitz；forecast encompassing | efficiency tests、encompassing、ROI | 模型比分概率优于博彩公司，但简单投注不稳定盈利；比分市场有 favourite-longshot bias | 评价体系模板；赔率市场效率；比分到赛果聚合 |
| Suzuki & Ohmori 2008 | 1994、1998、2002、2006 世界杯 | FIFA Ranking 规则预测；Fisher exact、chi-square、Spearman | 晋级准确率、胜率、相关系数 | FIFA Ranking 对小组赛和淘汰赛晋级有一定预测力，相关约 0.40 | 支持 FIFA Ranking 作为国家队强弱特征 |
| FIFA Ranking Rules | FIFA 官方 ranking 规则 | SUM/Elo 式公式 `P=Pbefore+I(W-We)` | 非实证文档 | FIFA points 本身是动态强度评分，包含比赛重要性和对手强弱 | 解释 FIFA ranking points/rank difference/We 特征 |
| Wunderlich & Memmert 2016 | 2006、2010、2014 世界杯，183 场，Betfair + FIFA Ranking | FIFA points -> expected goals -> bivariate Poisson -> W/D/L probabilities；与 Betfair 比较 | goals by favourite、log-likelihood、LR test | 2006 后 FIFA Ranking 预测质量提高；2014 和 2010-2014 合并样本中 RANK 优于 BET | 世界杯预测最相关；排名转概率方法 |
| Loukas et al. 2024 | EPL 2022/23，380 场 | 双泊松/泊松回归；攻击、防守、主场优势参数 | 进球误差、chi-square 独立性检验 | 多数预测误差在正负 1 球内；低估 0-0 | 可实现 Poisson baseline；比分概率矩阵 |
| Luiz, Fialho & Teixeira 2024 | WhoScored，13 联赛，24,760 场 | DNN + 特征工程 + pi-rating + grid search + feature importance | accuracy、特征重要性 | 总准确率 52.8%；高置信主胜可到 80.3%；平局弱 | ML/DL 路线；特征工程；高置信筛选 |
| Atta Mills et al. 2024 | 荷甲、比甲、苏超，合并 5,329 场 | LR/XGBoost/RF/SVM/NB/FNN/RNN/Voting；采样和归一化 | accuracy、precision、recall、F1、AUC | Voting Model 结果预测最强；LR 对 O/U 2.5 最强；平局仍难 | 多模型比较；类别不平衡处理；赛前/赛中变量区分 |

## 方法论脉络

### 1. 从简单评分到概率

早期方法从直观评分开始。Buchdahl 用最近 6 场净胜球构造球队强弱，Suzuki & Ohmori 用 FIFA Ranking 判断国家队强弱。这类方法可解释性强，但变量少、概率建模弱。

### 2. 从评分到 expected goals

Wunderlich & Memmert 把 FIFA ranking points 转成 expected goals，再通过双变量泊松得到三结果概率。这一步很适合世界杯预测，因为国家队样本少，直接训练复杂模型可能不稳。

### 3. 从比分模型到多个市场

Reade et al. 和 Loukas et al. 都强调比分是更底层的 outcome。只要有比分概率，就能聚合出:

- 主胜/平/客胜
- 净胜球
- 总进球
- exact scoreline

### 4. 从统计模型到机器学习

Luiz et al. 和 Atta Mills et al. 说明现代模型更依赖特征工程、多模型比较和类别不平衡处理。它们能提高 accuracy，但解释性、数据需求和过拟合风险也更高。

### 5. 从预测准确性到市场效率

Buchdahl 和 Reade et al. 都提醒: 准确预测不等于能赚钱。要判断投注价值，需要:

- 模型概率
- bookmaker implied probability
- overround adjustment
- value threshold
- ROI/yield backtest

## 可用于 FYP 的建议框架

一个稳妥的 FYP 方法设计可以这样组织:

```text
Baseline 1: FIFA ranking / Elo difference
Baseline 2: recent form / goal difference rating
Statistical model: Poisson or bivariate Poisson expected goals
ML model: logistic regression, random forest, XGBoost, neural network
Evaluation: accuracy + log loss/Brier + calibration + ROI/backtest
Market comparison: bookmaker implied probabilities after overround correction
```

## 各文献可引用位置

| 论文部分 | 推荐引用 |
|---|---|
| 足球预测任务背景 | Reade et al. 2020; Luiz et al. 2024 |
| FIFA Ranking 作为强弱指标 | Suzuki & Ohmori 2008; FIFA Ranking Rules; Wunderlich & Memmert 2016 |
| Poisson 比分模型 | Loukas et al. 2024; Reade et al. 2020 |
| Betting odds 作为概率预测 | Reade et al. 2020; Wunderlich & Memmert 2016 |
| Value betting / ROI | Buchdahl 2003; Reade et al. 2020 |
| ML/DL 特征工程 | Luiz et al. 2024; Atta Mills et al. 2024 |
| 类别不平衡和平局难预测 | Luiz et al. 2024; Atta Mills et al. 2024 |

## 总结性观点

这些文献整体说明，足球预测可以分成三条路线:

1. 评分/排名路线: 简单、可解释、适合 baseline。
2. 泊松比分路线: 适合输出完整比分和赛果概率。
3. 机器学习路线: 适合吸收大量特征，但需要严格验证和解释。

对你的 FYP 来说，最稳的写法不是只押一个模型，而是设计“baseline -> statistical model -> ML model -> bookmaker comparison”的递进结构。这样文献综述和方法论会自然连起来。

