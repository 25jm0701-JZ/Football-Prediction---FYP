# 特征重要性文献核查

核查对象为 FYP 项目原始 PDF；同时检查论文出版社页面。Atta Mills 的第 7 页表 1、Luiz 的第 14 页表 7 已渲染并核对，下面是文献结果，不能当作 English4 模型的实际重要性。

## Atta Mills 等 2024

论文：Data-driven prediction of soccer outcomes using enhanced machine and deep learning techniques。

链接：https://link.springer.com/article/10.1186/s40537-024-01008-2

作者使用随机森林的平均不纯度下降进行特征重要性分析。表 1 对赛果和总进球大小两个目标分别排序；本实验参考赛果一列。作者最终保留所有特征，因为重要性均非零；因此该论文并没有证明“删除最低重要性的变量”能提高预测。

| 原文赛果排序 | 特征 | 原文重要性 | 对当前 Track A 的适用性 |
|---:|---|---:|---|
| 1 | Half-time result | 0.0922 | 半场信息，排除 |
| 2 | Home win bet odds | 0.0786 | 市场输入，排除 |
| 3 | Away win bet odds | 0.0757 | 市场输入，排除 |
| 4 | Home team half-time goals | 0.0707 | 半场信息，排除 |
| 5 | Away team half-time goals | 0.0637 | 半场信息，排除 |
| 6 | Draw bet odds | 0.0604 | 市场输入，排除 |
| 7 | Away wins in the window | 0.0416 | 历史状态，概念可参考 |
| 8 | Home wins in the window | 0.0406 | 历史状态，概念可参考 |
| 9 | Away team state | 0.0314 | 历史状态，概念可参考 |
| 10 | Home team state | 0.0279 | 历史状态，概念可参考 |
| 11 | Total away loss rate | 0.0269 | 历史失利率 |
| 12 | Total away win rate | 0.0265 | 历史胜率 |
| 13 | Total home win rate | 0.0263 | 历史胜率 |
| 14 | Total home loss rate | 0.0258 | 历史失利率 |
| 15 | Home attack strength | 0.0240 | 定义需核对 |
| 16 | Total away draw rate | 0.0236 | 历史平局率 |
| 17 | Total home draw rate | 0.0234 | 历史平局率 |
| 18 | Away attack strength | 0.0231 | 定义需核对 |
| 19 | Home goal differential | 0.0212 | 历史净胜球 |
| 20 | Away goals against | 0.0203 | 历史失球 |
| 21 | Away goal differential | 0.0200 | 历史净胜球 |
| 22 | Away goals forward | 0.0199 | 历史进球 |
| 23 | Home goals forward | 0.0185 | 历史进球 |
| 24 | Home goals against | 0.0184 | 历史失球 |
| 25 | Away loss | 0.0118 | 仅参考历史版本 |
| 26 | Home wins margin goal | 0.0113 | 历史胜场分差 |
| 27 | Away win | 0.0111 | 仅参考历史版本 |
| 28 | Home draw | 0.0108 | 仅参考历史版本 |
| 29 | Home win | 0.0108 | 仅参考历史版本 |
| 30 | Away draw | 0.0099 | 仅参考历史版本 |
| 31 | Away losses margin goal | 0.0095 | 历史负场分差 |
| 32 | Home loss | 0.0090 | 仅参考历史版本 |
| 33 | Away wins margin goal | 0.0076 | 历史胜场分差 |
| 34 | Home losses margin goal | 0.0075 | 历史负场分差 |

应用限制：包含赛中变量的模型，其重要性分配与纯赛前模型不同。当前代码与原文的 Team State、Attack Strength 等也未必逐项同定义，不能把名字相似当成完整复现。

## Luiz 等 2024

论文：Is Football Unpredictable? Predicting Matches Using Neural Networks。

链接：https://www.mdpi.com/2571-9394/6/4/57

作者先用决策树检查原始统计与 pi-rating，再在神经网络上打乱每个特征，以原性能减去打乱后性能衡量重要性。第 14 页表 7 的顺序如下。

| 排序 | 特征 | 原文表 7 数值 |
|---:|---|---:|
| 1 | Relative Defending Power | 0.85 |
| 2 | Relative Playmaking Power | 0.81 |
| 3 | Relative Midfield Power | 0.77 |
| 3 | Goal Expectancy Home Team | 0.77 |
| 5 | Pi-rating Home Team | 0.73 |
| 5 | Pi-rating Away Team | 0.73 |
| 7 | Goal Expectancy Away Team | 0.70 |
| 8 | Shot Accuracy Away Team | 0.64 |
| 9 | Shot Accuracy Home Team | 0.62 |
| 9 | Relative Ball Loss | 0.62 |
| 11 | Goal difference | 0.46 |
| 11 | Relative Attacking Power | 0.46 |
| 13 | Relative Corner Power | 0.35 |

数值按原文保留。表格没有清晰标出其换算单位，本文不将 0.85 擅自解释成下降 85 个百分点，也不和本实验的 Log Loss 增量直接比较。

论文有传球、控球、拦截等 WhoScored 输入。当前 football-data.co.uk 数据没有完整对应字段，Relative Playmaking Power 和 Relative Midfield Power 不能通过改名现有滚动特征来替代。当前已有攻防、进球、射门/射正以及 5/10 场窗口可做删减实验；新增评级或相对特征是后续独立实验。

## 本次实验采用的方法

官方方法说明：https://scikit-learn.org/stable/modules/permutation_importance.html

对逻辑回归、随机森林分别计算验证季上的置换重要性，记录打乱后 Log Loss 上升及准确率下降，各重复 5 次。历史表现特征在联赛内部打乱，避免引入跨联赛交换造成的分布偏移；league_level 单独全局打乱。另给出随机森林不纯度重要性和逻辑回归按训练标准差换算、跨类别中心化后的系数，作为辅助诊断。

相关变量会互相替代。单列重要性低不代表该足球信息无效，所以同时进行整组置换、整组删除、Top-k、相关性去冗余和所有非空特征组组合搜索。负重要性仅用于提出候选，是否保留由后续验证季决定。

预先固定 2022/23 用于计算特征重要性，2023/24 与 2024/25 用于滚动验证和选择特征，2025/26 为本次搜索的最后评估季。最后一季在该项目过去实验中已经出现，因此这里只能称为回顾性评估，不能宣称是项目从未看过的新数据。

在有限候选集合中寻找表现更好的组合，不宣称穷举了 66 列的全部 2^66 种可能。
