# 从原版66特征重新筛选，再验证泛化与泊松融合

2026-10-15 已完成。从66个赛前特征出发，经过重要性分析与候选组合验证，筛选得到固定36个特征，再进行泛化检验与泊松融合；赛季始终按源CSV归属。

本次从原始28份CSV重新构建特征，重新运行置换重要性和76个候选组合搜索。历史验证LL最低的组合仍为`window_10_only`，36列。随后固定这组新选择，重新训练六季的原版/简化版LR，以及同一套原泊松参数，计算两种融合。

[完整中文报告](results/report_CN.md) · [总体结果](results/metrics_overall.csv) · [原版与简化版逐季对照](results/lr_comparison_by_season.csv) · [逐联赛对照](results/lr_comparison_by_league.csv) · [逐场预测](results/predictions.csv.gz) · [保存结果独立审计](results/verification_audit.json)

## 结果

- 简化LR：输入66→36（减少45.45%），六季准确率45.6287%→46.0707%，净多正确54场；LL 1.056095→1.053133。LL六季、四联赛均改善；准确率差的主要日期聚类95%区间包含零，Macro F1及平局F1下降。
- 相比纯泊松：简化融合比分NLL 3.005417→2.964762，精确比分命中率11.8533%→12.0498%；比分命中提升区间包含零，期望进球MAE改善。
- 相比66列融合：简化融合比分NLL 2.967725→2.964762；精确比分命中率12.1807%→12.0498%，MAE略变差。不能宣称所有指标进一步提升。

这是一项回顾性跨季对照。特征选择使用2022/23排序、2023/24–2024/25验证，因此全六季并非固定子集事先冻结后的六次独立验证。2025/26按原开发时序在选择之后，但已被历史研究观察，单独列出其结果和区间。完整模型与简化模型具有相同插补、C=1、未标准化设置，区别仅在列集。

## 复现

从项目根目录运行，使用已有Python环境。精确的实际库版本见`results/run_manifest.json`；不需要重新训练RF或标准化LR。下面使用新的目录，避免覆盖本次完成结果：

```powershell
python -B -u experiments/feature_search_20261010/run_feature_search.py --project-root . --output-dir experiments/feature_rebuild_poisson_20261015/development_reproduced --models lr
python -B -u experiments/feature_rebuild_poisson_20261015/run_experiment.py --development-dir experiments/feature_rebuild_poisson_20261015/development_reproduced --output-dir experiments/feature_rebuild_poisson_20261015/results_reproduced
python -B experiments/feature_rebuild_poisson_20261015/build_report.py --development-dir experiments/feature_rebuild_poisson_20261015/development_reproduced --output-dir experiments/feature_rebuild_poisson_20261015/results_reproduced
python -B experiments/feature_rebuild_poisson_20261015/verify_results.py --output-dir experiments/feature_rebuild_poisson_20261015/results_reproduced
python -B experiments/feature_rebuild_poisson_20261015/test_protocol.py
```

计算使用单线程BLAS以减少线程开销；主脚本内部已有threadpool_limits。`config.json`在本次全六季计算前写入，原模型代码和旧结果不覆盖。新选择的验证LL优于原版（1.038277 vs 1.040944），满足继续检验融合的主指标条件。

## 赛季与核验

- `source_file_audit.csv`逐份列出文件对应赛季、日期范围、七月场次和哈希；`source_match_registry.csv.gz`逐场保留源赛季和真实比分。
- 35场2022年7月比赛归到2022/23；2020年7月延迟完赛的2019/20比赛仍在2019/20。没有按月份把比赛移入相邻赛季。
- 模型训练使用此前全部赛季并在测试季内冻结。滚动特征只使用此前日期，不含当场或同日赛果。
- 新增5项边界测试通过。独立审计重新加载12个LR模型，验证训练集插补均值与全部LR概率，重构36,648个比分矩阵，逐行对源比赛、逐范围重算指标。
- `feature_selection.png`展示新的重要性与验证候选；`results_overview.png`展示跨季LR与比分结果。

`superseded_initial_config.json`仅保留用户澄清前的初始方案历史，未用于本次计算；实际配置唯一来源是`config.json`。开发搜索最初在该初始目录启动，完成后迁入本目录，计算本身从原始数据重新执行，未加载旧重要性或旧选择。


## 已发布结果的直接核验

```powershell
python -m pip install -r experiments/feature_rebuild_poisson_20261015/requirements.txt
python -B experiments/feature_rebuild_poisson_20261015/test_protocol.py
python -B -m unittest discover -s experiments/lr_poisson_reweight_20261012 -p "test_*.py" -v
python -B experiments/feature_search_20261010/test_feature_search.py --project-root .
python -B experiments/feature_rebuild_poisson_20261015/verify_results.py --output-dir experiments/feature_rebuild_poisson_20261015/results
```

依赖版本以本目录 requirements.txt 和实际运行清单为准。开发筛选工具单独保留的旧 requirements.txt 对应其早期运行环境，不是本次实验环境。输入 CSV 与旧历史特征脚本由 .gitattributes 保留原字节，避免 Git 换行转换影响审计。
