"""Create a Chinese research report and static figures from completed outputs."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_experiment import HERE, dump, paired_intervals


def markdown_table(frame, columns, names=None, percent=(), decimals=5):
    names = names or columns
    lines = ['| ' + ' | '.join(names) + ' |', '|' + '|'.join(['---'] * len(columns)) + '|']
    for _, row in frame.iterrows():
        values = []
        for c in columns:
            value = row[c]
            if pd.isna(value): value = '—'
            elif c in percent: value = f'{float(value) * 100:.3f}%'
            elif isinstance(value, (float, np.floating)): value = f'{value:.{decimals}f}'
            else: value = str(value)
            values.append(value)
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def contrast_table(table, scope, base, selected):
    t = table.loc[table.scope.eq(scope)]
    b, s = t.loc[t.model.eq(base)].set_index('group'), t.loc[t.model.eq(selected)].set_index('group')
    result = pd.DataFrame(dict(group=b.index, n_matches=b.n_matches, accuracy_base=b.accuracy,
                               accuracy_selected=s.accuracy, additional_correct=s.correct - b.correct,
                               log_loss_base=b.log_loss, log_loss_selected=s.log_loss,
                               log_loss_delta=s.log_loss - b.log_loss))
    return result.reset_index(drop=True)


def build(out, development):
    manifest = json.loads((out / 'run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['complete']
    selected = json.loads((out / 'frozen_selection.json').read_text(encoding='utf-8'))
    count = len(selected['feature_names'])
    labels = dict(poisson='纯泊松', lr66='原版 LR（66）', lr_selected=f'简化 LR（{count}）',
                  fusion66='66 特征 LR＋泊松', fusion_selected=f'{count} 特征 LR＋泊松')
    table = pd.read_csv(out / 'metrics_all_scopes.csv')
    intervals = pd.read_csv(out / 'paired_intervals.csv')
    overall = table.loc[table.scope.eq('overall')].set_index('model')
    original, reduced = overall.loc['lr66'], overall.loc['lr_selected']
    pm = pd.read_csv(out / 'per_match_metrics.csv.gz')
    last = paired_intervals(pm.loc[pm.fold.eq('2025-26')], repeats=4000, seed=123)
    last = last.drop(columns=['season_ci95_low', 'season_ci95_high'])
    last.to_csv(out / 'postselection_2025_26_intervals.csv', index=False)
    grid = pd.read_csv(development / 'validation_grid_lr.csv').sort_values(['log_loss', 'n_features', 'config', 'C'])
    ranking = pd.read_csv(development / 'importance_lr.csv')
    fit_audit = pd.DataFrame(json.loads((out / 'fit_audit.json').read_text(encoding='utf-8')))
    complexity = fit_audit.groupby('model').agg(n_features=('n_features', 'first'), n_parameters=('n_parameters', 'first'),
                    mean_model_bytes=('model_bytes', 'mean'), mean_fit_seconds=('fit_seconds', 'mean'),
                    mean_predict_seconds=('predict_seconds', 'mean'))
    complexity.to_csv(out / 'model_complexity.csv')
    seasons = contrast_table(table, 'season', 'lr66', 'lr_selected')
    leagues = contrast_table(table, 'league', 'lr66', 'lr_selected')
    cross = contrast_table(table, 'season_league', 'lr66', 'lr_selected')
    seasons.to_csv(out / 'lr_comparison_by_season.csv', index=False)
    leagues.to_csv(out / 'lr_comparison_by_league.csv', index=False)
    columns = ['group', 'n_matches', 'accuracy_base', 'accuracy_selected', 'additional_correct',
               'log_loss_base', 'log_loss_selected', 'log_loss_delta']
    cn = ['范围', '场次', '66 准确率', f'{count} 准确率', '净多正确', '66 LL', f'{count} LL', 'LL 差值']
    lines = [f'# 从原始 66 特征重新筛选，再验证泛化与泊松融合', '',
        '实验日期：2026-10-15。全部训练、重要性排序、候选搜索和逐场预测均在本次本地重新执行。', '',
        '## 先回答研究问题', '',
        f"1. **重新筛选得到什么？** 从 66 列出发，在 {len(grid)} 个候选组合中，以历史验证 Log Loss 为主指标，新一轮仍选中 `{selected['record']['config']}`：{count} 列。没有读取旧选择作为本次答案，也没有要求一定选出 36 列。",
        f"2. **简化 LR 是否优于原版？** 六季总体准确率 {original.accuracy:.3%} → {reduced.accuracy:.3%}，净多正确 {int(reduced.correct-original.correct)} 场；LL {original.log_loss:.6f} → {reduced.log_loss:.6f}，Brier 也降低。主指标改善，因此继续检验融合。但 Macro F1、平局 F1 下降，不能称为所有指标全面更好。",
        f"3. **能否认为准确率稳定提升？** 日期聚类区间下，六季准确率差的 95% CI 包含零；六季 LL 差的区间不包含零。按六个赛季重抽的敏感性区间另列，不能挑更乐观的区间下结论。",
        '4. **简化 LR 指导泊松后是否改善？** 相比纯泊松，赛果 LL、比分 NLL、主客及总进球 MAE 均改善；精确比分命中率点估计提高，但区间包含零。',
        '5. **是否超过原版 LR＋泊松？** 比分 NLL 进一步降低，但精确比分命中率下降，主客及总进球 MAE 也略变差，相关差异区间包含零。因此只支持概率评分进一步改善，不支持所有比分指标进一步改善。', '',
        '## 1. 实验流程与事先确定的规则', '',
        '原始 66 特征 LR → 重新做重要性与候选组合验证 → 冻结新胜出的列集 → 固定列集六季 walk-forward 对照 → 在主指标改善后分析泊松融合。', '',
        '- 原版和简化版均为训练集均值插补、未标准化 LogisticRegression，C=1、lbfgs、max_iter=4000、种子42。只改变输入列，避免把标准化或调参收益归于特征删减。',
        '- 排序：用 2019/20–2021/22 拟合，在 2022/23 做五次按联赛分块的置换重要性；组合验证：分别用此前全部赛季训练，验证 2023/24 和 2024/25。按合并验证 Log Loss 选定，2025/26 不参与候选排序或选择。',
        '- 候选包括特征族组合、重要性前若干列、去掉尾部若干列、正置换重要性列、仅近5场/仅近10场、相关性删减。重要性用于提出候选，最终以组合验证判断；不是简单把最末30列删除。',
        '- 泊松：复用原始 LoukasPoissonModel.fit 与 smoothing=0.05；三种比分模型共用由赛前 λ 决定的自适应网格，联合尾部遗漏≤1e-12。',
        '- 融合：三个赛果区域分别重加权，F(i,j)=q_LR(c)×P(i,j)/P(c)，alpha=1；没有根据测试比分调节融合强度。',
        '- 整个筛选和训练流程从原始 CSV 重新执行，没有使用旧逐场 LR 概率替代本次训练。', '',
        '## 2. 为什么选中这个简化组合', '',
        markdown_table(grid.head(10), ['config', 'n_features', 'accuracy', 'log_loss'],
                       ['候选组合', '列数', '验证准确率', '验证 Log Loss'], percent=['accuracy']), '',
        '完整 66 列验证基线：', '',
        markdown_table(grid.loc[grid.config.eq('all_66')], ['config', 'n_features', 'accuracy', 'log_loss'],
                       ['组合', '列数', '验证准确率', '验证 Log Loss'], percent=['accuracy']), '',
        f"选择后保留 {count} 列，删除 {66-count} 列，输入数量减少 {(66-count)/66:.2%}。该组合是所搜索候选中验证 LL 最低的方案；不意味着它是所有可能列集中的全局最优，也不是特征数量最少的方案。", '',
        '新排序前15列：', '',
        markdown_table(ranking.head(15), ['rank', 'feature', 'permutation_logloss_mean', 'permutation_logloss_std'],
                       ['排序', '特征', '置换后 LL 增量均值', '重复标准差']), '',
        f"仅在排序季之前的历史中，绝对相关系数≥0.95 的特征对有 {manifest['high_correlation_pairs_ge_095']} 对，详见 high_correlation_pairs.csv。相关性说明存在重叠信息，但并不能单独证明应该删哪一列。即使部分5场特征重要性较高，仍可能因与其他输入重叠而在组合删减时改善泛化。", '',
        '新组合保留全部近10场主客统计，以及5个交锋指标和联赛级别，删除近5场窗口。完整列清单见文末。', '',
        '![特征重要性与验证候选](feature_selection.png)', '',
        '## 3. 赛季归属与 walk-forward 边界', '',
        '**赛季以原始 E0–E3 CSV 文件名为准，绝不按八月或其他月份切分。** 独立逐行匹配源文件、比赛日期、球队、比分、标签和赛季。28份文件共13,988场，测试12,216场；初始2019/20共1,772场仅作训练历史。', '',
        '- 35场2022年7月比赛按 E12223/E22223/E32223 文件归到2022/23。',
        '- 疫情延迟至2020年7月的2019/20比赛，仍保留在2019/20。完整逐文件日期范围及七月场次见 source_file_audit.csv。',
        '- 每折用此前所有完整赛季训练；测试季模型参数冻结。近期特征可使用此前日期的已完赛比赛，不使用当场、同日或未来赛果。', '',
        markdown_table(pd.read_csv(out / 'walk_forward_folds.csv'),
                       ['fold', 'train_seasons', 'n_train', 'n_test', 'train_last_date', 'test_first_date'],
                       ['测试季', '训练赛季', '训练场次', '测试场次', '训练最后日期', '测试首日']), '',
        '## 4. 原版 LR 与简化 LR：先检验特征工程', '',
        markdown_table(overall.loc[['lr66', 'lr_selected']].reset_index().assign(model=lambda x: x.model.map(labels)),
                       ['model', 'n_matches', 'correct', 'accuracy', 'log_loss', 'brier', 'macro_f1', 'draw_f1'],
                       ['模型', '场次', '正确场次', '准确率', 'Log Loss', 'Brier', 'Macro F1', '平局 F1'], percent=['accuracy']), '',
        '逐季：', '', markdown_table(seasons, columns, cn, percent=['accuracy_base', 'accuracy_selected']), '',
        '逐联赛（E0英超、E1英冠、E2英甲、E3英乙）：', '',
        markdown_table(leagues, columns, cn, percent=['accuracy_base', 'accuracy_selected']), '',
        f"六季中准确率提高 {int((seasons.additional_correct>0).sum())}/6、持平 {int((seasons.additional_correct==0).sum())}/6；LL改善 {int((seasons.log_loss_delta<0).sum())}/6。四联赛 LL 改善 {int((leagues.log_loss_delta<0).sum())}/4；24个季×联赛单元 LL 改善 {int((cross.log_loss_delta<0).sum())}/24。分组结果为描述性分析，没有进行多重比较校正。", '',
        '模型输入与保存体积：', '',
        markdown_table(complexity.reset_index(), ['model', 'n_features', 'n_parameters', 'mean_model_bytes', 'mean_fit_seconds', 'mean_predict_seconds'],
                       ['模型', '输入列数', '系数及截距个数', '平均模型字节', '平均拟合秒', '平均预测秒']), '',
        '输入列数与参数数量的减少是确定的。时间仅为本机单次执行的描述，不是多轮性能基准；本实验先构建全部66列再取子集，没有测量精简特征生成后的端到端加速。', '',
        '## 5. 在此基础上检验泊松融合', '',
        markdown_table(overall.loc[['poisson', 'fusion66', 'fusion_selected']].reset_index().assign(model=lambda x: x.model.map(labels)),
                       ['model', 'accuracy', 'log_loss', 'exact_score_accuracy', 'score_nll', 'home_goal_mae', 'away_goal_mae', 'total_goal_mae'],
                       ['模型', '胜平负准确率', '胜平负 LL', '精确比分命中率', '比分 NLL', '主球 MAE', '客球 MAE', '总球 MAE'],
                       percent=['accuracy', 'exact_score_accuracy']), '',
        'MAE来自各自比分矩阵的期望进球，不沿用原泊松λ作为融合期望。精确比分命中率来自概率最高的比分；胜平负预测来自区域总概率。', '',
        '逐季比分结果：', '',
        markdown_table(table.loc[table.scope.eq('season') & table.model.isin(['poisson','fusion66','fusion_selected'])].assign(model=lambda x: x.model.map(labels)),
                       ['group', 'model', 'exact_score_accuracy', 'score_nll', 'home_goal_mae', 'away_goal_mae', 'total_goal_mae'],
                       ['赛季', '模型', '精确比分命中率', '比分 NLL', '主球 MAE', '客球 MAE', '总球 MAE'], percent=['exact_score_accuracy']), '',
        '逐联赛比分结果：', '',
        markdown_table(table.loc[table.scope.eq('league') & table.model.isin(['poisson','fusion66','fusion_selected'])].assign(model=lambda x: x.model.map(labels)),
                       ['group', 'model', 'exact_score_accuracy', 'score_nll', 'home_goal_mae', 'away_goal_mae', 'total_goal_mae'],
                       ['联赛', '模型', '精确比分命中率', '比分 NLL', '主球 MAE', '客球 MAE', '总球 MAE'], percent=['exact_score_accuracy']), '',
        '**融合的赛果概率按定义等于输入LR概率。逐场比分NLL差也严格等于赛果LL差。** 因而比分NLL改善来自赛果区域质量分配，不能把它解释成又学会了区域内部比分结构；精确比分命中和进球MAE没有自动改善保证。', '',
        '![跨季结果](results_overview.png)', '',
        '## 6. 配对差异和不确定性', '',
        '差值均为“后者−前者”。准确率、比分命中率正值更好；LL/NLL、MAE负值更好。',
        '主区间使用4,000次按赛季分层、完整比赛日期成组的配对bootstrap；另给出六个赛季成组的敏感性区间。六个赛季簇很少；两种方法均未涵盖特征选择及重新训练带来的全部不确定性。', '',
        markdown_table(intervals.loc[intervals.metric.isin(['accuracy','log_loss','exact_score_accuracy','score_nll','home_goal_mae','away_goal_mae','total_goal_mae'])],
                       ['comparison', 'metric', 'delta', 'ci95_low', 'ci95_high', 'season_ci95_low', 'season_ci95_high'],
                       ['比较', '指标', '差值', '日期95%下限', '日期95%上限', '赛季95%下限', '赛季95%上限']), '',
        '注意：准确率差在此表采用0–1比例。例如0.00442等于+0.442个百分点，不能当作+0.00442个百分点。', '',
        '## 7. 原选择区间之后的2025/26结果', '',
        '2025/26在原开发时序中位于选择之后，单独列出以避免把开发区间与之后测试混为一谈。但该季此前已被项目研究过，这次属于重新计算，不能称为首次未见或前瞻性验证。', '',
        markdown_table(table.loc[table.scope.eq('season') & table.group.eq('2025-26')].assign(model=lambda x: x.model.map(labels)),
                       ['model','accuracy','log_loss','exact_score_accuracy','score_nll'],
                       ['模型','准确率','Log Loss','精确比分命中率','比分 NLL'], percent=['accuracy','exact_score_accuracy']), '',
        markdown_table(last.loc[last.comparison.eq('lr_selected_minus_lr66')], ['metric','delta','ci95_low','ci95_high'],
                       ['简化−原版指标','差值','日期95%下限','日期95%上限']), '',
        '该季简化LR的准确率与LL改善均为点估计，区间包含零；不能以全六季更窄的区间替代这个选择之后赛季的独立证据。', '',
        '## 8. 可以写出的结论与局限', '',
        f"在同一清洁数据、训练设置和六季回顾性walk-forward下，重新筛选得到的{count}特征LR以更少输入获得更低总体Log Loss和略高准确率。跨季、跨联赛概率评分方向较一致，但平局识别减弱，准确率提升仍有不确定性。",
        '用简化LR对泊松比分分布重加权，相比纯泊松改善概率评分和期望进球误差；相比完整66特征融合，收益集中在概率评分，精确比分命中率和MAE未进一步改善。不能声称简化模型或融合在所有指标上全面最佳。', '',
        '**全六季固定子集结果是回顾性跨季检验。** 子集选择使用2022/23排序及2023/24–2024/25验证，因此早期赛季并不是该子集事先冻结之后的六次独立验证。各折模型训练没有未来数据，并不消除全局特征选择的后见性。所有历史曾被观察，进一步前瞻确认需要之后的新比赛。', '',
        '## 9. 可复现与核验', '',
        '- run_manifest.json：实际库版本、命令、配置、代码及28份输入哈希。',
        '- frozen_selection.json：本次新选择和完整列清单，六季评价前冻结。',
        '- verification_audit.json：从保存模型重算LR概率、核对训练集插补均值、全部36,648个比分矩阵重构、全部分组指标重算、原始逐场赛季及标签核验。',
        '- predictions.csv.gz、per_match_metrics.csv.gz：12,216场比赛上的五个模型预测及逐场评价。',
        '- metrics_all_scopes.csv：整体、六季、四联赛、24个季×联赛及实际赛果分层。',
        '- paired_intervals.csv、postselection_2025_26_intervals.csv：配对区间。',
        '- models/：本次训练的12个LR模型及各自特征列表；poisson_parameters.csv：六季泊松参数。',
        '- protocol_tests.log：新增5项源文件赛季、训练时间边界及不预设36列的测试。此前融合数学12项和特征时间边界5项测试已通过；本次另做完整保存结果审计。', '',
        '## 附：本次新选中列清单', '', '```text', *selected['feature_names'], '```', '']
    (out / 'report_CN.md').write_text('\n'.join(lines), encoding='utf-8')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
    top = ranking.head(15).iloc[::-1]
    colors = ['#087e8b' if f in selected['feature_names'] else '#bf8c52' for f in top.feature]
    ax[0].barh(top.feature, top.permutation_logloss_mean, color=colors)
    ax[0].set(title='Fresh permutation importance (2022/23)', xlabel='Log Loss increase after permutation')
    ax[0].text(.99, .02, 'Teal: retained / Ochre: removed', transform=ax[0].transAxes, ha='right', fontsize=9)
    short = grid.head(8).iloc[::-1].copy()
    ax[1].barh(short.config.str.replace('groups__', '').str.replace('__', '+'), short.log_loss, color='#087e8b')
    ax[1].set_xlim(short.log_loss.min() - .001, max(short.log_loss.max(), float(grid.loc[grid.config.eq('all_66'),'log_loss'].iloc[0])) + .0005)
    ax[1].axvline(float(grid.loc[grid.config.eq('all_66'),'log_loss'].iloc[0]), color='#c35e4f', linestyle='--', label='All66 baseline')
    ax[1].set(title='Development validation (2023/24 + 2024/25)', xlabel='Log Loss (lower is better)')
    ax[1].legend()
    fig.savefig(out / 'feature_selection.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), layout='constrained')
    x = np.arange(len(seasons))
    axes[0].bar(x - .18, seasons.accuracy_base * 100, .36, color='#758da0', label='LR66')
    axes[0].bar(x + .18, seasons.accuracy_selected * 100, .36, color='#087e8b', label=f'LR{count}')
    axes[0].set_xticks(x, seasons.group.str.replace('20', '', n=1), rotation=35)
    axes[0].set(title='Fixed-subset retrospective comparison', ylabel='W/D/L accuracy (%)', ylim=(37, 51))
    axes[0].legend()
    season_metrics = table.loc[table.scope.eq('season')]
    for name, color, label in [('poisson','#c35e4f','Poisson'),('fusion66','#758da0','LR66 + Poisson'),('fusion_selected','#087e8b',f'LR{count} + Poisson')]:
        frame = season_metrics.loc[season_metrics.model.eq(name)].set_index('group').loc[seasons.group]
        axes[1].plot(x, frame.score_nll, 'o-', color=color, label=label)
        axes[2].plot(x, frame.exact_score_accuracy * 100, 'o-', color=color, label=label)
    for axis in axes[1:]:
        axis.set_xticks(x, seasons.group.str.replace('20', '', n=1), rotation=35)
        axis.legend(fontsize=8)
    axes[1].set(title='Score distribution improves', ylabel='Exact-score NLL (lower is better)')
    axes[2].set(title='Modal score accuracy does not always improve', ylabel='Exact-score accuracy (%)')
    fig.savefig(out / 'results_overview.png', dpi=170)
    plt.close(fig)
    dump(out / 'report_summary.json', dict(selected_features=count, selected_config=selected['record']['config'],
          lr_additional_correct=int(reduced.correct-original.correct), lr_accuracy_delta=float(reduced.accuracy-original.accuracy),
          lr_log_loss_delta=float(reduced.log_loss-original.log_loss), seasons_accuracy_improved=int((seasons.additional_correct>0).sum()),
          seasons_log_loss_improved=int((seasons.log_loss_delta<0).sum()), leagues_log_loss_improved=int((leagues.log_loss_delta<0).sum()),
          season_league_log_loss_improved=int((cross.log_loss_delta<0).sum())))
    print(f'Report and figures saved to {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'results')
    parser.add_argument('--development-dir', type=Path, default=HERE / 'development')
    args = parser.parse_args()
    build(args.output_dir.resolve(), args.development_dir.resolve())
