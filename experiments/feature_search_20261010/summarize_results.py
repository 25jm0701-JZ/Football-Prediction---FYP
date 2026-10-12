"""Build the Chinese experiment report and static scientific plots from completed CSVs."""
from __future__ import annotations
import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"


def markdown_table(df):
    headers=[str(c) for c in df.columns]
    lines=["| "+" | ".join(headers)+" |", "| "+" | ".join("---" for _ in headers)+" |"]
    for row in df.itertuples(index=False,name=None):
        lines.append("| "+" | ".join(str(v) for v in row)+" |")
    return "\n".join(lines)


def formatted_metrics(df):
    x=df[["model","config","n_features","n_matches","accuracy","macro_f1","draw_f1","log_loss","brier"]].copy()
    x["accuracy"]=x.accuracy.map(lambda n:f"{n:.2%}")
    for c in ["macro_f1","draw_f1","log_loss","brier"]: x[c]=x[c].map(lambda n:f"{n:.5f}")
    return markdown_table(x)


def main():
    protocol=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    assert protocol.get("completed"), "Search has not completed; do not describe partial results as final."
    test=pd.read_csv(OUT/"final_test_metrics.csv")
    frozen=json.loads((OUT/"frozen_selections.json").read_text(encoding="utf-8"))
    uncertainty=json.loads((OUT/"paired_uncertainty.json").read_text(encoding="utf-8"))
    audit=json.loads((OUT/"data_audit.json").read_text(encoding="utf-8"))
    allgrid=pd.concat([pd.read_csv(OUT/f"validation_grid_{kind}.csv") for kind in ["lr","rf","lr_scaled"]])
    allgrid.to_csv(OUT/"validation_grid_all.csv",index=False)
    champion=allgrid.sort_values(["log_loss","n_features","model"]).iloc[0]
    lrfull=test[(test.model=="lr")&(test.config=="all_66")].iloc[0]
    lrchosen=test[(test.model=="lr")&(test.config=="selected_logloss")].iloc[0]
    lrci=uncertainty["lr__selected_logloss"]["accuracy"]
    lines=["# Clean Track A 特征重要性与组合消融实验", "",
        "本次完成文献原文核查、当前模型特征重要性排序、按组组合搜索与逐组消融。所有删减和参数选择使用历史验证赛季；最后评估赛季没有参与本次组合选择。", "",
        f"按预先设定的验证 Log Loss 选择规则，全部已搜索配置中的最优候选是 `{champion.model}` 的 `{champion.config}`，共 {champion.n_features} 列。原逻辑回归的 36 列方案删除所有 5 场窗口，保留 10 场窗口、交锋和联赛级别；最后评估准确率由 {lrfull.accuracy:.2%} 到 {lrchosen.accuracy:.2%}，约增加 {(lrchosen.accuracy-lrfull.accuracy)*100:.2f} 个百分点。差异的日期组 bootstrap 95% 区间为 [{lrci['ci95_low']*100:+.2f}, {lrci['ci95_high']*100:+.2f}] 个百分点，包含零，因此是有希望的简化候选，尚不能称为稳定提升。", "",
        "## 评估结果", "",
        formatted_metrics(test[test.config.isin(["all_66","selected_logloss","selected_accuracy","market"])]), "",
        "表中的 accuracy 为胜平负准确率。Log Loss 越低越好；Brier 沿用项目已有的三类平均均方误差口径。不同表中的整体历史准确率和单赛季准确率不能直接比较。", "",
        "主选择指标预先设为验证季 Log Loss，同时冻结验证准确率最高的配置作为第二个目标。两者可能选出不同特征集，最终测试结果不能用于在二者之间重新挑赢家。", "",
        "## 冻结的特征组合", ""]
    for key,chosen in frozen.items():
        kind,objective=key.split("__",1)
        row=test[(test.model==kind)&(test.config==f"selected_{objective}")].iloc[0]
        base=test[(test.model==kind)&(test.config=="all_66")].iloc[0]
        interval=uncertainty[f"{kind}__selected_{objective}"]
        ci=interval["accuracy"]
        ll=interval["logloss"]
        evidence="准确率区间包含零，不能据此声称稳定提升" if ci["ci95_low"]<=0<=ci["ci95_high"] else "准确率区间未包含零，仍需新赛季确认"
        parameters=f"C={chosen['C']:g}" if kind.startswith("lr") else "250 棵树、深度 10、叶节点最少 8"
        lines.extend([f"### {key}", "",
            f"验证选中 `{chosen['config']}`，保留 {chosen['n_features']} 列，{parameters}。验证准确率 {chosen['accuracy']:.2%}，验证 Log Loss {chosen['log_loss']:.5f}。", "",
            f"最后评估准确率 {row.accuracy:.2%}，相对同模型完整 66 列变化 {(row.accuracy-base.accuracy)*100:+.3f} 个百分点；Log Loss 变化 {row.log_loss-base.log_loss:+.5f}。", "",
            f"按比赛日期成组的 2,000 次配对 bootstrap：准确率变化 95% 区间 [{ci['ci95_low']*100:+.3f}, {ci['ci95_high']*100:+.3f}] 个百分点；Log Loss 变化区间 [{ll['ci95_low']:+.5f}, {ll['ci95_high']:+.5f}]。{evidence}。该区间只描述这个回顾性评估季，不包含跨赛季、训练或选模不确定性。", "",
            "```text", *chosen["features"], "```", ""])
    lines.extend(["## 当前模型的重要性排序", "",
        "以下排序用 2022/23 赛季，模型仅用之前赛季训练；每列在联赛内打乱 5 次。主排序数值是打乱后 Log Loss 增加量，正值越大表明该模型在该评估季越依赖该列。league_level 单独全局打乱。标准差反映 5 次打乱之间的变化，不是跨赛季置信区间。", "",
        "所有 66 列的完整排名已保存为 CSV。这里列出前 15 名及特征组重要性。", ""])
    if plt is not None:
        fig,axes=plt.subplots(1,2,figsize=(15,8),constrained_layout=True)
    else:
        axes=[None,None]
    for kind,ax in zip(["lr","rf"],axes):
        rank=pd.read_csv(OUT/f"importance_{kind}.csv")
        group=pd.read_csv(OUT/f"group_importance_{kind}.csv")
        shown=rank.head(15)[["rank","feature","permutation_logloss_mean","permutation_logloss_std","permutation_accuracy_mean"]].copy()
        for col in shown.columns[2:]: shown[col]=shown[col].map(lambda n:f"{n:.6f}")
        lines.extend([f"### {kind}", "", markdown_table(shown), "",
            markdown_table(group.round(6)), ""])
        if ax is not None:
            plot=rank.head(15).iloc[::-1]
            ax.barh(plot.feature,plot.permutation_logloss_mean,xerr=plot.permutation_logloss_std,color="#366c96",capsize=2)
            ax.set_title(f"{kind}: importance on 2022-23")
            ax.set_xlabel("Increase in log loss after permutation")
            ax.axvline(0,color="gray",linewidth=.8)
            ax.tick_params(axis="y",labelsize=9)
    if plt is not None:
        fig.savefig(OUT/"feature_importance_top15.png",dpi=160)
        plt.close(fig)
    lines.extend(["## 三个历史赛季的排名稳定性", "",
        "2022/23 是选择候选前的排序季。额外核查 2023/24、2024/25 的排名，每季仍只用此前赛季训练。这些结果是描述性稳定性分析，没有反过来改变已冻结特征组合，完全未使用最后评估季。", ""])
    for kind in ["lr","rf"]:
        stability=OUT/f"importance_stability_{kind}.csv"
        if stability.exists():
            lines.extend([f"### {kind} 跨开发季", "",
                markdown_table(pd.read_csv(stability).head(10).round(6)), ""])
    lines.extend(["## 逐组删除的控制实验", "",
        "这些删减项目在查看最后评估结果之前已固定，用于解释信息组贡献。它们不是根据最后评估季挑选的模型，不能据此追加特征选择。", "",
        formatted_metrics(test[(test.model.isin(["lr","rf"]))&((test.config=="all_66")|test.config.str.startswith("drop_"))]), "",
        "## 实验协议", "",
        f"数据为 English4 的 {audit['matches']:,} 场，来自 E0–E3 的 2019/20–2025/26。每场输入都在更新当场及当天结果之前构造，球队历史跨联赛延续。", "",
        "1. 重要性季：2022/23；训练季为 2019/20–2021/22。",
        "2. 组合选择：分别训练到 2022/23、2023/24，验证 2023/24、2024/25，再合并逐场预测计算指标。",
        "3. 最后评估：用 2019/20–2024/25 训练，评估 2025/26。",
        "4. 赔率只用于独立市场基准，不用于重要性排序、模型训练或候选选择。所有市场比较使用有效 Bet365 收盘赔率。",
        "5. 未使用半场信息、当场原始统计、FootyStats 或当前赛果标签作为特征。历史进球、射门和射正只在比赛完成后进入历史队列。", "",
        "## 搜索范围", "",
        "六组分别为 form（平均积分与胜平负历史）、goals_strength（进失球、净胜球、攻防强度）、margins（胜负分差）、shooting（射门、射正及效率）、h2h（交锋）、league（联赛级别）。所有非空组组合共有 63 种；另外加入 Top-8/16/24/32/48/60、删除末位 5/10/20、正置换重要性、5 场或 10 场窗口和相关性阈值 0.95 去冗余。重复特征集去重。", "",
        f"实际完成 {len(allgrid)} 个模型与参数配置、{2*len(allgrid)} 次验证训练。原逻辑回归 C=1；随机森林保持 250 树、深度 10、叶节点最少 8；标准化逻辑回归额外搜索 C=0.1/1/10。标准化比较是独立预处理分支，其改进不能全部归因于删特征。", "",
        "仅计算有限候选内的验证最优组合；没有穷举 66 列的全部组合，也不保证数学意义上的全局最优。", "",
        "## 数据及实现审查", "",
        "- 原构建代码将 league_level 同时保留在原表和特征表，选择时实际传入 67 列；本实验只保留一列。",
        f"- 按月份划分赛季会把七月的揭幕比赛归到上一季。本次按 CSV 所属赛季改正 {audit['season_fixes']} 场，并检查每折训练最大日期早于评估最小日期。",
        "- 原攻防归一化的累计联赛均值会吸收此前遍历的同一天比赛。本实验按日期批量构造，用前一天以前的结果计算均值。",
        "- 当前平均积分=3×胜率+平局率，净胜球=进球−失球，胜平负率之和=1，存在精确线性冗余。打乱单列会破坏这些关系；特征间的代偿也会影响数值，所以单列置换排名不能直接解释为因果作用或独立足球价值。",
        "- 五项自动检查已通过：当前及同日结果不影响输入、七月赛季归属与单列联赛标记、时间重叠拒绝、类别与 Brier 指标口径、整组置换保留行内关系及联赛范围。", "",
        "- 导出的六个冻结模型已与原逐场预测核对；重新加载模型后概率也一致。重读特征 CSV 必须使用 float_precision='round_trip'，保留浮点值的精确往返；否则在这些高度相关的变量下，微小解析差异可能改变优化停止点。", "",
        "## 结论的边界", "",
        "2025/26 曾用于该项目之前的实验。本次代码没有用它调参，但它不是项目从未观察过的新数据。只有一个最后评估赛季，仍需要新赛季或另一个独立联赛确认。低重要性并不证明信息无效，增加噪声列也未必让所有模型表现更差；删除效果必须看重训练后的实际验证结果。", "",
        "原报告 46.01% / 45.62% 是多个赛季的整体数字；本报告表格是 2025/26 单季。修正后的数据和输入列数也不同，因此不会将二者直接相减并称为提升。", "",
        "## 文献与复现", "",
        "文献排序见 `../literature_review.md`。", "",
        "```powershell",
        "python -B -u experiments/feature_search_20261010/run_feature_search.py --project-root . --output-dir experiments/feature_search_20261010/results",
        "```", "",
        "`protocol.json` 记录预先规定的分割、随机种子、原始 CSV 哈希；`selection_before_test.json` 在最后评估之前写入选择结果；`frozen_selections.json` 列出保留列；`final_test_predictions.csv` 可逐场审计。"])
    (OUT/"feature_search_report_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if plt is not None:
        fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
        for kind,ax in zip(["lr","rf","lr_scaled"],axes):
            grid=allgrid[allgrid.model==kind]
            ax.scatter(grid.n_features,grid.log_loss,s=18,alpha=.7,color="#366c96")
            chosen=frozen[f"{kind}__logloss"]
            ax.scatter([chosen["n_features"]],[chosen["log_loss"]],s=110,color="#cb673b",marker="*")
            ax.set_title(f"{kind}: 2023-25 validation")
            ax.set_xlabel("Number of retained features")
            ax.set_ylabel("Log loss (lower is better)")
        fig.savefig(OUT/"feature_grid_validation.png",dpi=160)
        plt.close(fig)
    print("REPORT",OUT/"feature_search_report_CN.md")
    print(test[test.config.isin(["all_66","selected_logloss","selected_accuracy","market"])].to_string(index=False))
    print("CONFIGS",len(allgrid))


if __name__=="__main__": main()
