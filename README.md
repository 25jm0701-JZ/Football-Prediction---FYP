# Football Prediction FYP

**研究主线：赛前 66 特征 LR → 筛选并冻结 36 特征 → 跨赛季、跨联赛检验 → LR 指导泊松比分分布。**

更新：2026-10-15。研究目标是用开赛前已知的历史信息输出可检验的胜平负与比分概率。通过开发期特征筛选，从 66 个特征中选出 36 个特征。

## 从这里开始

| 内容 | 入口 |
|---|---|
| 研究问题、方法与主线解释 | [主线说明](docs/research_mainline_CN.md) |
| 过往实验与结论如何演进 | [实验沿革](docs/experiment_history_CN.md) |
| 最新 66→36 与泊松融合代码、结果、复现 | [主实验](experiments/feature_rebuild_poisson_20261015/README.md) |
| 完整数值、分组结果与局限 | [中文实验报告](experiments/feature_rebuild_poisson_20261015/results/report_CN.md) |
| 文献依据 | [文献综述](docs/literature_review_CN.md) · [方法对应](docs/literature_method_mapping_CN.md) |
| 本次移除、替换与未上传内容 | [清理报告](docs/repository_cleanup_20261015.md) |

## 核心方法

1. 从 football-data.co.uk 的英格兰 E0–E3 历史比赛构造干净的 66 个赛前特征；赔率只作外部基准。
2. 以 2022/23 计算重要性，2023/24–2024/25 比较 76 个候选组合，按 Log Loss 选择后冻结一组 36 特征：双方近 10 场统计 30 列、交锋 5 列、联赛等级 1 列。
3. 原版与简化 LR 使用相同的训练集均值插补、C=1、未标准化设置。六个赛季分别只用此前赛季拟合，比较同场预测。
4. 同一套泊松攻防模型先给出比分矩阵，再用 LR 胜平负概率调整三个赛果区域的总概率；区域内部的比分比例保持原样。

对比分 `(i,j)` 所属赛果区域 `c`，融合公式为 `F(i,j)=q_LR(c)×P(i,j)/S_P(c)`。类别顺序统一为客胜、平局、主胜。LR 提供区域概率，泊松提供区域内部的比分结构。

## 当前结果

同一批 **12,216** 场比赛；2019/20 为初始训练历史，2020/21–2025/26 为六个评价季。全部原始数据为 28 份 CSV、**13,988** 场。

| 赛果模型 | 输入数 | 准确率 | Log Loss ↓ | Macro F1 | 平局 F1 |
|---|---:|---:|---:|---:|---:|
| 原版 LR | 66 | 45.63% | 1.056095 | 0.3523 | 0.0421 |
| 简化 LR | 36 | 46.07% | 1.053133 | 0.3493 | 0.0266 |

特征减少 45.45%，净多预测正确 54 场。Log Loss 在六季和四个联赛的汇总中均降低；准确率提升的日期聚类 95% 区间包含零，Macro F1 与平局 F1 下降。

| 比分模型 | 赛果准确率 | 精确比分命中率 | 比分 NLL ↓ |
|---|---:|---:|---:|
| 纯泊松 | 42.02% | 11.85% | 3.005417 |
| 66 特征 LR＋泊松 | 45.63% | 12.18% | 2.967725 |
| 36 特征 LR＋泊松 | 46.07% | 12.05% | 2.964762 |

36 特征融合的概率损失最低，但精确比分命中率低于 66 特征融合，进球 MAE 也略差。两项融合的精确比分差异区间均包含零。融合的赛果概率由构造保证等于对应 LR；比分 NLL 的改变量等于赛果 Log Loss 的改变量，不能算作额外学会区域内部比分结构的证据。

## 评价边界

- 赛季按源 CSV 文件归属；2020 年 7 月延赛仍属 2019/20，2022 年 7 月的 35 场属 2022/23。
- 模型参数在每个测试季内冻结；滚动历史只加入更早日期已结束的比赛，同日比赛互相不可见。
- 不使用当场原始统计、半场信息、赔率或 FootyStats 作为主线特征。
- **36 特征由较晚历史筛选后回看六季，所以整体是回顾性泛化检验，不是六个事先冻结后的独立留出集。**2025/26 已被此前研究观察；未来泛化仍需新数据。
- 早期 58.60%、68.42% 及高 ROI 等结果已被泄漏审查否定，不作当前结论；当前未证明盈利或普遍优于市场。

## 复现

Python 3.11，CPU 即可。先安装主实验实际执行版本的依赖；绘图额外使用 matplotlib。

```powershell
python -m pip install -r experiments/feature_rebuild_poisson_20261015/requirements.txt
python -B -u experiments/feature_search_20261010/run_feature_search.py --project-root . --output-dir experiments/feature_rebuild_poisson_20261015/development_reproduced --models lr
python -B -u experiments/feature_rebuild_poisson_20261015/run_experiment.py --development-dir experiments/feature_rebuild_poisson_20261015/development_reproduced --output-dir experiments/feature_rebuild_poisson_20261015/results_reproduced
python -B experiments/feature_rebuild_poisson_20261015/build_report.py --development-dir experiments/feature_rebuild_poisson_20261015/development_reproduced --output-dir experiments/feature_rebuild_poisson_20261015/results_reproduced
python -B experiments/feature_rebuild_poisson_20261015/verify_results.py --output-dir experiments/feature_rebuild_poisson_20261015/results_reproduced
python -B experiments/feature_rebuild_poisson_20261015/test_protocol.py
```

已保存新选择、逐场预测、12 个 LR 模型、六季泊松参数、配对区间及输入/代码哈希。直接核验已发布结果的方法见主实验 README。

## 仓库结构

| 路径 | 用途 |
|---|---|
| `experiments/feature_rebuild_poisson_20261015/` | 唯一当前主实验：66→36、固定子集检验、两种 LR→泊松融合 |
| `experiments/feature_search_20261010/` | 一次性筛选工具与早期开发记录 |
| `experiments/lr_poisson_reweight_20261012/` | 固定 66 特征融合基线及共用分区重加权函数 |
| `league/data/english_raw/` | 主实验 28 份输入，保留原始字节供哈希核验 |
| `scripts/`、`league/src/` | 赛前历史特征、泊松基线和历史实验代码 |
| `docs/` | 主线、实验沿革、文献、泄漏审查与清理记录 |
| `world_cup/` | 独立的世界杯支线 |
| `要求/` | 项目要求、原始文献与阅读笔记 |

历史 FootyStats、神经网络和投注代码保留供追溯，其结论和信息集见[实验沿革](docs/experiment_history_CN.md)。历史 README 只用于考据，最新数字以主实验报告为准。
