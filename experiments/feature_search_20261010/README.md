# English4 特征工程实验

本目录包含重要性排序、组合搜索、控制消融和逐场审计代码。四级联赛合并训练；本次最终评估仅为 2025/26。六个历史赛季的严格嵌套回测位于相邻的 `nested_backtest_20261010` 目录。

`reference/` 保存已完成运行的报告、指标、特征列表和输入哈希，适合随 GitHub 代码提交。`results/` 是重新运行生成的完整输出，含逐场预测和模型；该目录被 Git 忽略。原有项目模型和输出不被覆盖。

## 在另一台电脑复现

从 FYP 仓库根目录运行，使用 Python 3.11。参考运行使用 Python 3.11.0，依赖版本固定在 requirements.txt。模型在 CPU 上训练，无需 CUDA。原项目必须包含 scripts/、league/src/ 和 28 份 league/data/english_raw/E[0-3]*.csv 输入文件。

```powershell
python -m venv .venv_reproduce
.\.venv_reproduce\Scripts\python.exe -m pip install -r experiments/feature_search_20261010/requirements.txt
.\.venv_reproduce\Scripts\python.exe -B experiments/feature_search_20261010/check_inputs.py --project-root .
.\.venv_reproduce\Scripts\python.exe -B experiments/feature_search_20261010/test_feature_search.py --project-root .
.\.venv_reproduce\Scripts\python.exe -B -u experiments/feature_search_20261010/run_feature_search.py --project-root .
.\.venv_reproduce\Scripts\python.exe -B experiments/feature_search_20261010/export_frozen_models.py
.\.venv_reproduce\Scripts\python.exe -B experiments/feature_search_20261010/check_ranking_stability.py
.\.venv_reproduce\Scripts\python.exe -B experiments/feature_search_20261010/summarize_results.py
```

已有同版本环境可直接使用其 Python。跨操作系统或数值库实现可能导致末位浮点差异。读取 feature_matrix.csv 必须使用 `float_precision="round_trip"`，否则优化结果可能因 CSV 精度发生变化。

输入清单保留原始字节哈希，并增加 CRLF 统一为 LF 后的文本哈希。check_inputs.py 只允许 Git 换行转换造成的差异，数据或代码内容变化仍会报错。

## 实验协议

2022/23 为重要性评估季，2023/24 和 2024/25 用于组合及参数选择，2025/26 为选择冻结后的最终评估。每个评估季仅使用之前赛季训练，插补和缩放只在训练数据拟合，特征排除当场及同一天结果。三个开发季的额外排名稳定性只是描述性诊断，没有修改冻结的选择。

搜索六组特征的全部 63 个非空组合，以及 Top-k、删除末位、正置换重要性、窗口删减和相关性去冗余。原 LR、RF 保留参数作为控制，标准化 LR 另搜索 C=0.1/1/10。主选择目标为验证 Log Loss，另保存准确率最优方案。最优仅指预设候选范围。

只有 results/protocol.json 中 completed=true 才代表完成。export_frozen_models.py 会重训练、保存模型并与原始逐场预测比较，检查失败会终止。summarize_results.py 无需 matplotlib；安装该包后可额外生成静态图。

## 文件

- literature_review.md：文献原文核查。
- reference/feature_search_report_CN.md：实验报告。
- reference/input_manifest.json：28 个数据文件和三个源代码文件的 SHA-256。
- reference/importance_*.csv、group_importance_*.csv：重要性排名。
- reference/validation_grid_*.csv：选择证据。
- reference/frozen_selections.json：保留列及参数。
- reference/final_test_metrics.csv、paired_uncertainty.json：指标及区间。
- 运行生成的 results/final_test_predictions.csv：逐场预测。
- 运行生成的 results/models/：冻结模型和列顺序。

将参考报告、代码、依赖和输入数据一起保留即可复现。历史数据和 2025/26 此前已被研究过，因此本次为回顾性实验，不能视为项目完全未见的新数据。
