"""Audit completed nested results, compare historical runs, and write a Chinese report."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import sklearn
from run_feature_search import metrics

HERE=Path(__file__).resolve().parent


def table(frame):
    lines=["| "+" | ".join(map(str,frame.columns))+" |", "| "+" | ".join(["---"]*len(frame.columns))+" |"]
    return "\n".join(lines+["| "+" | ".join(map(str,row))+" |" for row in frame.itertuples(index=False,name=None)])


def metric_table(frame):
    display=frame[["model","config","n_matches","accuracy","log_loss","macro_f1","draw_f1","brier","correct"]].copy()
    display.accuracy=display.accuracy.map(lambda x:f"{x:.2%}")
    for col in ["log_loss","macro_f1","draw_f1","brier"]: display[col]=display[col].map(lambda x:f"{x:.5f}")
    return table(display)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-root",type=Path,default=HERE.parents[1])
    parser.add_argument("--output-dir",type=Path,default=HERE/"results")
    parser.add_argument("--previous-dir",type=Path)
    args=parser.parse_args()
    root,out=args.project_root.resolve(),args.output_dir.resolve()
    previous=args.previous_dir or root/"experiments/feature_search_20261010/results"
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    assert protocol["completed"], "All six outer folds must finish before reporting"
    predictions=pd.read_csv(out/"predictions_all.csv",float_precision="round_trip",parse_dates=["date"])
    data=pd.read_csv(out/"feature_matrix.csv",float_precision="round_trip",parse_dates=["date"])
    summary=pd.read_csv(out/"metrics_all_and_seasons.csv",float_precision="round_trip")
    selected=json.loads((out/"selections_all.json").read_text(encoding="utf-8"))
    uncertainty=json.loads((out/"paired_uncertainty.json").read_text(encoding="utf-8"))
    audit=json.loads((out/"data_audit.json").read_text(encoding="utf-8"))
    expected=data[data.season_id.isin(protocol["outer_seasons"])]
    assert len(expected)==12216
    checks=[]
    for (kind,config),block in predictions.groupby(["model","config"]):
        assert not block.row_id.duplicated().any()
        if kind!="bet365_closing":
            assert set(block.row_id)==set(expected.index)
            assert block.groupby("season_id").size().to_dict()=={s:2036 for s in protocol["outer_seasons"]}
        truth=data.loc[block.row_id,"target_result"].to_numpy(int)
        np.testing.assert_array_equal(truth,block.target_result.to_numpy(int))
        probability=block[["p_away","p_draw","p_home"]].to_numpy()
        np.testing.assert_array_equal(block.predicted.to_numpy(int),probability.argmax(1))
        result=metrics(truth,probability)
        recorded=summary[(summary.model==kind)&(summary.config==config)&(summary.season=="ALL")].iloc[0]
        for key,value in result.items(): np.testing.assert_allclose(value,recorded[key],atol=1e-12,rtol=1e-12)
        checks.append(kind+"__"+config)
    for season in protocol["outer_seasons"]:
        split=json.loads((out/season/"splits.json").read_text())
        start=pd.Timestamp(split["outer_test_start"])
        assert pd.Timestamp(split["ranking_evaluation"]["last_date"])<start
        for fold in split["validation"]:
            assert pd.Timestamp(fold["training"]["last_date"])<pd.Timestamp(fold["evaluation"]["first_date"])
            assert pd.Timestamp(fold["evaluation"]["last_date"])<start
    # Compare old outputs on the identical fixture set, retaining the old model's original folds.
    old_path=root/"outputs/atta_mills_tracka_english4/predictions_by_fold.csv"
    if not old_path.exists(): old_path=HERE/"reference/original_predictions.csv.gz"
    original=pd.read_csv(old_path,
        float_precision="round_trip",parse_dates=["date"])
    keys=["date","home_team","away_team"]
    old_rows=[]
    for kind,old_kind in [("lr","lr"),("rf","random_forest")]:
        fresh=predictions[(predictions.model==kind)&(predictions.config=="all_66")]
        old=original[original.model==old_kind]
        assert not old.duplicated(keys).any()
        matched=fresh[keys+["target_result"]].merge(old,on=keys,validate="one_to_one")
        assert len(matched)==len(fresh)==len(old)==12216
        np.testing.assert_array_equal(matched.actual,matched.target_result)
        p=matched[["p_away","p_draw","p_home"]].to_numpy()
        old_rows.append(dict(model=kind,config="original_67_columns",**metrics(matched.actual.to_numpy(int),p),
            correct=int((p.argmax(1)==matched.actual.to_numpy()).sum())))
    old_table=pd.DataFrame(old_rows)
    old_table.to_csv(out/"original_historical_metrics.csv",index=False)
    # The last fold deliberately reproduces the previous single-season selection calendar.
    prior_path=previous/"final_test_predictions.csv"
    if not prior_path.exists(): prior_path=previous.parent/"reference/final_test_predictions.csv.gz"
    old_test=pd.read_csv(prior_path,float_precision="round_trip",parse_dates=["date"])
    comparisons=[]
    for kind in ["lr","rf","lr_scaled"]:
        for config in ["all_66","selected_logloss","selected_accuracy"]:
            a=predictions[(predictions.season_id=="2025-26")&(predictions.model==kind)&(predictions.config==config)]
            b=old_test[(old_test.model==kind)&(old_test.config==config)]
            matched=a.merge(b,on=keys,validate="one_to_one",suffixes=("_new","_old"))
            assert len(matched)==2036
            pa=matched[["p_away_new","p_draw_new","p_home_new"]].to_numpy()
            pb=matched[["p_away_old","p_draw_old","p_home_old"]].to_numpy()
            np.testing.assert_allclose(pa,pb,rtol=1e-7,atol=1e-9)
            comparisons.append(dict(model=kind,config=config,n=2036,max_probability_difference=float(np.max(np.abs(pa-pb)))))
    pd.DataFrame(comparisons).to_csv(out/"previous_2025_26_reproduction.csv",index=False)
    verification=dict(completed=True,outer_matches=12216,source_matches=len(data),
        invariant_unit_tests=5,feature_unit_tests=5,prediction_groups_verified=checks,
        previous_2025_26_reproduced=comparisons,python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__)
    (out/"verification.json").write_text(json.dumps(verification,indent=2),encoding="utf-8")
    # Explicitly separate data corrections from feature-selection effects.
    all_metrics=summary[summary.season=="ALL"]
    changes=[]
    fold_changes=[]
    for kind in ["lr","rf","lr_scaled"]:
        baseline=all_metrics[(all_metrics.model==kind)&(all_metrics.config=="all_66")].iloc[0]
        chosen=all_metrics[(all_metrics.model==kind)&(all_metrics.config=="selected_logloss")].iloc[0]
        ci=uncertainty[kind+"__selected_logloss"]
        changes.append(dict(model=kind,accuracy_delta_pp=100*(chosen.accuracy-baseline.accuracy),
            additional_correct=int(chosen.correct-baseline.correct),log_loss_delta=chosen.log_loss-baseline.log_loss,
            accuracy_ci95_pp=f"[{ci['accuracy']['ci95_low']*100:+.3f}, {ci['accuracy']['ci95_high']*100:+.3f}]",
            log_loss_ci95=f"[{ci['logloss']['ci95_low']:+.6f}, {ci['logloss']['ci95_high']:+.6f}]"))
        for season in protocol["outer_seasons"]:
            rows=summary[(summary.model==kind)&(summary.season==season)]
            a=rows[rows.config=="all_66"].iloc[0]; b=rows[rows.config=="selected_logloss"].iloc[0]
            choice=selected[season][kind]["logloss"]
            fold_changes.append(dict(season=season,model=kind,n=2036,baseline_accuracy=a.accuracy,
                selected_accuracy=b.accuracy,accuracy_delta_pp=100*(b.accuracy-a.accuracy),additional_correct=int(b.correct-a.correct),
                baseline_log_loss=a.log_loss,selected_log_loss=b.log_loss,log_loss_delta=b.log_loss-a.log_loss,
                selected_features=len(choice["features"]),selected_C=choice["C"],selected_config=choice["config"]))
    changes=pd.DataFrame(changes); fold_changes=pd.DataFrame(fold_changes)
    changes.to_csv(out/"controlled_comparison.csv",index=False)
    fold_changes.to_csv(out/"controlled_comparison_by_season.csv",index=False)
    lr_base=all_metrics[(all_metrics.model=="lr")&(all_metrics.config=="all_66")].iloc[0]
    lr_selected=all_metrics[(all_metrics.model=="lr")&(all_metrics.config=="selected_logloss")].iloc[0]
    lines=["# English4 六赛季严格嵌套滚动回测", "",
        "已完成 2020/21–2025/26 六个外层测试赛季，共 12,216 场，英超、英冠、英甲、英乙各季合并训练和预测。2019/20 的 1,772 场作为初始历史，无法在没有更早训练数据的条件下纳入独立测试。", "",
        f"逻辑回归按历史选特征的流程将整体准确率从 {lr_base.accuracy:.2%} 提高到 {lr_selected.accuracy:.2%}，净多对 {int(lr_selected.correct-lr_base.correct)} 场，Log Loss 也改善。其准确率与 Log Loss 配对区间均排除零，支持本次历史回测中的整体改善。标准化逻辑回归的增益还包含 C 的调整；随机森林删特征没有带来准确率收益，Log Loss 的微小差异区间包含零。所有方案仍低于收盘市场基准。", "",
        "这些流程没有改善整体 Macro F1，平局 F1 仍很低。因此准确率与概率质量的提高，不能解释成三类结果的识别都改善。", "",
        "原逻辑回归 101 场净增中，71 场来自 2021/22 当折选出的 8 列胜负分差组合。最近两折均事前选中 36 列 window_10_only：2024/25 准确率持平，2025/26 多对 10 场。整体增益不能归因于固定 36 列，仍不能称其跨六季稳定最优。", "",
        "0.83 个百分点描述的是每季按历史重新选特征的流程。最近两个外层折恰好都选中了同一 36 列方案：4,072 场中完整 66 列正确 1,924 场，36 列正确 1,934 场，合计提升约 0.25 个百分点。该两季点估计仍不能证明它跨全部六季稳定优于完整模型。", "",
        "研究目标是在同一数据和训练口径下验证模型简化是否改善原模型，并排除泄漏、保证可复现。市场仅作外部参照，超过博彩公司市场不是毕业项目成功的必要条件。", "",
        "每个测试季先用此前的数据做重要性排序、组合搜索并冻结特征和参数，再用该季之前全部赛季重训练并预测该季。四级联赛在同一折中使用同一个模型、同一套特征。", "",
        "## 整体受控结果", "",
        metric_table(all_metrics[all_metrics.config.isin(["all_66","selected_logloss","market"])]), "",
        "主方案始终按内部验证 Log Loss 选择，测试季准确率不参与选择。完整基线与主方案使用相同数据、相同外层训练区间。标准化 LR 还选择 C，因此其差异包含正则化参数影响。C 只适用于逻辑回归，RF 表中的 C=1 仅为统一表结构的占位值。", "",
        table(changes.round(6)), "",
        "区间为按赛季分别重采样比赛日期的 4,000 次配对 bootstrap，保留同日比赛成组。区间只反映固定预测的抽样波动，不覆盖完整训练及选模不确定性，也不能消除球队和相邻日期之间所有相关性。", "",
        "## 每季的表现与选择", ""]
    view=fold_changes.copy()
    for col in ["baseline_accuracy","selected_accuracy"]: view[col]=view[col].map(lambda x:f"{x:.2%}")
    for col in ["accuracy_delta_pp","baseline_log_loss","selected_log_loss","log_loss_delta"]: view[col]=view[col].map(lambda x:f"{x:+.5f}")
    lines += [table(view), ""]
    for kind,block in fold_changes.groupby("model"):
        lines += [f"{kind}：准确率提高 {int((block.accuracy_delta_pp>0).sum())}/6 季，下降 {int((block.accuracy_delta_pp<0).sum())}/6 季；Log Loss 改善 {int((block.log_loss_delta<0).sum())}/6 季。", ""]
    lines += ["## 与此前结果的关系", "", metric_table(old_table), "",
        "旧历史结果与本次覆盖相同 12,216 场，已逐场匹配确认。不过旧流程重复传入 league_level，按月份误分 35 场七月比赛，并且累计均值没有按日期统一排除同日结果。本次完整 66 列基线先修正这些问题；只有‘本次基线 → 本次所选方案’的差异可用于讨论本次特征选择的效果。‘旧结果 → 新结果’还包含数据与特征实现修正。", "",
        "2025/26 外层折沿用此前的排序季 2022/23、验证季 2023/24–2024/25。九组同名模型与配置的 2,036 场预测概率均通过一致性核对，结果复现此前单季实验；新增整体结果包含其他五季，不能把单季的 +10 场简单放大成六季增益。", "",
        table(pd.DataFrame(comparisons)), "",
        "## 内部时间划分", ""]
    for season in protocol["outer_seasons"]:
        split=json.loads((out/season/"splits.json").read_text())
        rank=split["ranking_evaluation"]
        validations=", ".join(v["block"] for v in split["validation"])
        lines += [f"- {season}：外层训练 {split['outer_train_n']} 场；排序评估 {rank['block']}（{rank['first_date']}–{rank['last_date']}，{rank['n']} 场）；选择验证 {validations}；外层测试 2,036 场。"]
    lines += ["", "前两折历史不足：首折按比赛日期保留整天边界，约 60% 训练、20% 排序、20% 选择验证；第二折仍在初始季内部排序，完整 2020/21 作选择验证。2019/20 疫情中断使最早折的历史更有限，不能假设其特征选择与后期一样可靠。三折起使用完整历史赛季排序和验证。", "",
        "训练集插补、缩放、相关性删减和模型拟合均排除其对应评估区间。比赛特征使用赛前可得的先前日期结果；在测试赛季内，已完成的较早比赛可更新后来比赛的历史特征，模型参数仍只在赛季开始前拟合。没有赔率或当前比赛统计作为训练特征。", "",
        "## 如何理解本次结论", "",
        "本次检验的是‘每季仅凭历史重新选择特征’的完整训练流程。它没有用 2025/26 选出的固定 36 列反向验证早期赛季；那样的回测会将未来选模信息带入历史。各季入选列可以变化，但同一季四级联赛始终共用特征和模型。若需要部署一个固定列集，应在当前全部开发历史中重新选择并冻结，然后用之后新数据确认。", "",
        "这是一项严格排除外层目标参与当折选模的回顾性测试。已有历史结果和此前 2025/26 实验已被观察，候选搜索方法也源于此前工作，因此仍不等同于真正前瞻性、从未观察过的测试。有限网格中的最优也不是所有可能特征组合的全局最优。", "",
        "## 文件与复现", "",
        "- metrics_all_and_seasons.csv：整体及六季指标。",
        "- metrics_by_league.csv：四个联赛分别的指标。",
        "- controlled_comparison*.csv：基线与所选方案的整体及逐季差异。",
        "- selections_all.json：每折、每模型、两个目标的全部列与参数。",
        "- 各季/model/validation_grid_*.csv：内部所有配置指标。",
        "- 各季/selection_before_test.json、splits.json：冻结选择与时间边界。",
        "- predictions_all.csv：全部逐场预测，支持核查和后续分析。",
        "- verification.json：指标重新计算、场次、标签和旧 2025/26 一致性核对。",
        "- protocol.json：数据和代码哈希、运行时间、完成状态。", "",
        "代码、版本固定依赖、输入清单和中文报告均可随 GitHub 提交；完整预测、缓存和模型在 results/ 生成。参考快照放在 reference/，避免忽略 results/ 后报告链接失效。", ""]
    (out/"nested_backtest_report_CN.md").write_text("\n".join(lines),encoding="utf-8")
    print("VERIFIED AND REPORTED",len(checks),"prediction groups",len(expected),"matches")
    print(all_metrics[all_metrics.config.isin(["all_66","selected_logloss","market"])].to_string(index=False))
    print(changes.to_string(index=False))


if __name__=="__main__": main()
