# Data-driven Prediction of Soccer Outcomes Using Enhanced Machine and Deep Learning Techniques - 方法论笔记

## 基本信息

- 原文件: `要求/文献/s40537-024-01008-2.pdf`
- 标题: Data-driven prediction of soccer outcomes using enhanced machine and deep learning techniques
- 作者: Ebenezer Fiifi Emire Atta Mills, Zihui Deng, Zhuoqing Zhong, Jinger Li
- 发表: Journal of Big Data, 2024
- DOI: 10.1186/s40537-024-01008-2
- 主题: 多模型、多联赛、特征工程、采样和集成学习的足球预测框架

## 研究目标

文章提出一个足球比赛预测框架，比较多种机器学习和深度学习模型在两个任务上的表现:

1. 预测比赛结果: 主胜、平局、客胜。
2. 预测总进球是否 over/under 2.5。

研究强调:

- 加入实时/赛中信息，如半场结果和半场进球。
- 通过特征工程增强模型输入。
- 用采样方法缓解类别不平衡。
- 用 voting model 集成多个模型。
- 在多个联赛和合并数据集上检验泛化能力。

## 数据

主要数据来自 football-data.co.uk 一类历史比赛结果与赔率数据源。

联赛:

- Dutch Eredivisie
- Belgian Jupiler Pro League
- Scottish Premiership

荷甲数据:

- 时间: 2017/18 到 2024-05-19
- 比赛数: 1,411
- 主胜: 644
- 平局: 442
- 客胜: 325
- Under 2.5: 852
- Over 2.5: 559

合并数据集:

- 三个联赛合并
- 时间: 2017 至研究时点
- 比赛数: 5,329

## 总体框架

流程:

```text
data acquisition
-> data cleaning
-> target encoding
-> feature engineering
-> feature importance analysis
-> normalization/scaling
-> sampling for class imbalance
-> model training
-> hyperparameter tuning
-> voting model ensemble
-> evaluation on individual and merged leagues
```

## 特征工程

文章生成 28 个新特征，试图捕捉球队状态、攻防强度、进球趋势、盘口信息和比赛过程信息。

重要特征包括:

- Team State
- Attack Strength
- home/away goals forward
- home/away goals against
- goal differential
- win/draw/loss history
- win/loss margin goals
- bet odds
- half-time result
- half-time goals

文章使用 Random Forest classifier 评估特征重要性，并保留所有非零重要性的特征。

## 数据预处理

比较多种缩放方法:

- min-max scaling
- max-abs scaling
- standardization
- robust scaling

类别不平衡处理方法:

- random under-sampling
- Near-Miss
- random oversampling
- SMOTE-NN
- SVM-SMOTE

这些方法主要用于提升对少数类，尤其是平局的识别能力。

## 模型

比较模型:

- Logistic Regression
- XGBoost
- Random Forest
- Support Vector Machine
- Naive Bayes
- Feedforward Neural Network
- Vanilla Recurrent Neural Network
- Voting Model

深度学习模型使用 dropout、batch normalization、regularization 等增强泛化能力。

Voting Model 采用多个模型预测概率的集成，最终选择平均预测概率最高的类别。文章中最成功的组合是 Random Forest + XGBoost。

## 调参和验证

模型通过 grid search 和 Bayesian optimization 调参。

评价指标:

- accuracy
- precision
- recall
- F1-score
- AUC

部分实验使用 sevenfold cross-validation。

## 主要结果

### 荷甲结果预测

Voting Model 表现最好，Random Forest + XGBoost 组合准确率达到 0.83。

分类指标显示:

- Voting Model 对 home、draw、away 的 F1-score 分别约为 0.86、0.68、0.89。
- 单模型中 Feedforward Neural Network、SVM、Random Forest 表现较强。
- 平局仍然是最难预测类别，但 voting model 明显改善。

### Over/Under 2.5

Logistic Regression 表现最好，在荷甲实验中 accuracy 约 0.78。SVM 和 Random Forest 也接近，约 0.77。

### 比利时联赛

结果预测中，FNN 表现最好:

- accuracy: 0.66
- F1-score: 0.54
- precision: 0.70
- recall: 0.57

### 合并数据集

结果预测:

| 模型 | Accuracy | F1 |
|---|---:|---:|
| SVM | 0.67 | 0.62 |
| FNN | 0.67 | 0.61 |
| Logistic Regression | 0.66 | 0.65 |
| Random Forest | 0.66 | 0.60 |
| XGBoost | 0.65 | 0.64 |

Over/Under 2.5:

| 模型 | Accuracy | F1 |
|---|---:|---:|
| Logistic Regression | 0.77 | 0.77 |
| XGBoost | 0.76 | 0.75 |
| Random Forest | 0.76 | 0.74 |
| SVM | 0.74 | 0.71 |
| FNN | 0.73 | 0.71 |

## 文章结论

- 机器学习和深度学习可以有效预测足球比赛结果。
- FNN 是结果预测中表现稳定的模型之一。
- Logistic Regression 对 over/under 2.5 这类二分类任务非常有效。
- Voting Model，尤其 Random Forest + XGBoost，整体表现最强。
- 跨联赛合并数据能验证模型泛化能力，但不同联赛仍有差异。

## 方法论优点

- 同时覆盖赛果和 over/under 2.5 两类任务。
- 比较多个传统 ML 与 DL 模型。
- 系统处理类别不平衡。
- 使用多种评价指标，不只看 accuracy。
- 做跨联赛和合并数据实验，关注泛化能力。
- 引入半场数据，有利于赛中预测。

## 方法论局限

- 加入半场结果和半场进球后，任务更像 in-play prediction，不完全等同赛前预测。
- 如果 FYP 目标是赛前预测，应避免使用赛中变量。
- 模型很多，容易出现调参选择偏差，需要清晰验证集/测试集划分。
- 文章没有把预测直接转化为 betting ROI。
- 部分模型解释性较弱。

## 对 FYP 的启发

这篇适合支撑机器学习实验设计:

- 可以比较多模型，而不是只用一个模型。
- 对平局这种少数/难分类类别，可尝试 SMOTE、Near-Miss、class weight。
- 对赛前预测和赛中预测必须分开建模。
- 如果用赔率变量，要说明这是市场信息，和纯球队表现模型不同。
- 可以加入 voting/stacking ensemble 作为提升模型。

