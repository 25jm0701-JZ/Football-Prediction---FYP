# English4 严格整体滚动回测

预测范围：2020/21–2025/26 六季，四级联赛共 12,216 场。2019/20 作为初始训练历史。四级联赛合并使用同一个模型，同一折使用同一套特征；每季仅根据此前数据重新选择特征和参数。

完整结果和结论见 [中文报告](reference/nested_backtest_report_CN.md)，逐季特征列表见 [selections_all.json](reference/selections_all.json)。`reference/` 是可随 GitHub 提交的参考快照，包含压缩逐场预测、旧历史预测、全部内部搜索与审计文件。`results/` 包含运行生成的完整预测、缓存和模型，被 Git 忽略。

## 复现

从 FYP 仓库根目录执行，Python 3.11；参考版本为 3.11.0。31 个输入文件由 check_inputs.py 核对 SHA-256。CPU 训练，无需 CUDA。

输入清单保留参考运行的原始字节哈希，并额外记录 CRLF 统一为 LF 后的文本哈希。校验只允许 Git 换行转换造成的差异；数据内容或代码内容变化仍会报错。

安装依赖后，也可先核对已发布记录，无需重新训练：

```powershell
.\.venv_reproduce\Scripts\python.exe -B experiments/nested_backtest_20261010/verify_reference.py --project-root .
.\.venv_reproduce\Scripts\python.exe -B experiments/nested_backtest_20261010/test_input_checks.py
```

verify_reference.py 从压缩逐场预测重算指标，核对 2,275 个配置、3,412 次内部验证记录、每季选择及时间边界、输入哈希与报告链接。模型重训练一致性来自原实验记录的 18 个主方案模型检查；此命令不重新训练模型。发布检查记录见 reference/publication_audit.json。

```powershell
python -m venv .venv_reproduce
.\.venv_reproduce\Scripts\python.exe -m pip install -r experiments/nested_backtest_20261010/requirements.txt
.\.venv_reproduce\Scripts\python.exe -B experiments/nested_backtest_20261010/check_inputs.py --project-root .
.\.venv_reproduce\Scripts\python.exe -B experiments/nested_backtest_20261010/test_feature_search.py --project-root .
.\.venv_reproduce\Scripts\python.exe -B experiments/nested_backtest_20261010/test_nested_backtest.py
.\.venv_reproduce\Scripts\python.exe -B -u experiments/nested_backtest_20261010/run_nested_backtest.py --project-root .
.\.venv_reproduce\Scripts\python.exe -B experiments/nested_backtest_20261010/summarize_nested_backtest.py --project-root .
.\.venv_reproduce\Scripts\python.exe -B experiments/nested_backtest_20261010/export_nested_models.py
```

已有相同依赖的环境可直接运行。省略 --output-dir 时输出默认在本目录的 results/。完整折及验证预测会缓存，再运行可续接；改变数据或核心代码后请使用新的输出目录。脚本会拒绝用不同代码接续已完成折。排序和训练随机种子均为 42。

总结脚本优先核对本地旧输出；若旧 outputs/ 被 Git 忽略，则使用 reference/ 中保存的旧逐场预测，并从相邻 feature_search_20261010/reference/ 中读取此前 2025/26 预测。跨平台数值库可能产生末位差异；脚本保留严格的概率核对，失败时应检查输入、版本和优化过程。

## 协议与边界

- 内层只访问外层测试季之前的数据；重要性评估早于选择验证，所有选择先冻结再评估外层目标。
- 首折在 2019/20 内按日期保留整天边界，约 60% 训练、20% 排序、20% 验证。第二折在初始季内排序，2020/21 验证。历史增加后使用完整赛季排序和最多两季验证。
- 插补、标准化和相关性筛选只拟合对应训练集。训练特征排除同一天和当场结果。
- 测试季较早比赛完成后，可更新后来比赛的滚动特征；分类器参数只用赛季开始前历史训练。
- 66 列全特征是每折的受控基线。主选择目标为内部验证 Log Loss，同时单独保存准确率选择方案。
- 标准化 LR 还搜索 C=0.1/1/10，原 LR 和 RF 保留固定参数。
- 本次验证每季历史选模流程，不能声称此前用后期数据选出的固定 36 列已获得六季独立验证。
- 历史数据已经被研究过，协议源自此前实验；这是时间顺序严格的回顾性测试，而非完全未见数据的前瞻性测试。

只有 results/protocol.json 的 completed=true 才表示全部完成。总结脚本重新计算整体指标、检查六季场次、标签、时间边界，并核对 2025/26 与先前实验的九组预测。导出脚本重训练 18 个主方案模型并检查原始预测与加载后预测。

## 主要参考文件

- metrics_all_and_seasons.csv、metrics_by_league.csv：整体、逐季、逐联赛结果。
- controlled_comparison.csv、controlled_comparison_by_season.csv：完整基线与选特征方案的差异。
- paired_uncertainty.json：按赛季、日期成组的配对 bootstrap 区间。
- selections_all.json、各季/selection_before_test.json：每折特征及参数。
- 各季/model/validation_grid_*.csv、importance_*.csv：完整搜索与排序。
- season_assignment_fixes.csv：35 场七月比赛的赛季修正。
- original_historical_metrics.csv、previous_2025_26_reproduction.csv：与旧结果对照。
- verification.json、model_verification.json、protocol.json：验证、版本及输入/代码哈希。
