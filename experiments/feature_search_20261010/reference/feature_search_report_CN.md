# Clean Track A 特征重要性与组合消融实验

本次完成文献原文核查、当前模型特征重要性排序、按组组合搜索与逐组消融。所有删减和参数选择使用历史验证赛季；最后评估赛季没有参与本次组合选择。

按预先设定的验证 Log Loss 选择规则，全部已搜索配置中的最优候选是 `lr` 的 `window_10_only`，共 36 列。原逻辑回归的 36 列方案删除所有 5 场窗口，保留 10 场窗口、交锋和联赛级别；最后评估准确率由 46.71% 到 47.20%，约增加 0.49 个百分点。差异的日期组 bootstrap 95% 区间为 [-0.31, +1.36] 个百分点，包含零，因此是有希望的简化候选，尚不能称为稳定提升。

## 评估结果

| model | config | n_features | n_matches | accuracy | macro_f1 | draw_f1 | log_loss | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr | all_66 | 66 | 2036 | 46.71% | 0.34019 | 0.00000 | 1.05192 | 0.21117 |
| lr | selected_logloss | 36 | 2036 | 47.20% | 0.34270 | 0.00000 | 1.04979 | 0.21076 |
| lr | selected_accuracy | 36 | 2036 | 47.20% | 0.34270 | 0.00000 | 1.04979 | 0.21076 |
| rf | all_66 | 66 | 2036 | 45.87% | 0.33072 | 0.00000 | 1.05366 | 0.21158 |
| rf | selected_logloss | 45 | 2036 | 45.63% | 0.32856 | 0.00000 | 1.05153 | 0.21123 |
| rf | selected_accuracy | 45 | 2036 | 45.63% | 0.32856 | 0.00000 | 1.05153 | 0.21123 |
| lr_scaled | all_66 | 66 | 2036 | 46.81% | 0.34175 | 0.00738 | 1.05314 | 0.21143 |
| lr_scaled | selected_logloss | 36 | 2036 | 47.25% | 0.34191 | 0.00000 | 1.05016 | 0.21082 |
| lr_scaled | selected_accuracy | 36 | 2036 | 47.25% | 0.34191 | 0.00000 | 1.05016 | 0.21082 |
| bet365_closing | market | 0 | 2036 | 49.26% | 0.36246 | 0.00000 | 1.02331 | 0.20492 |

表中的 accuracy 为胜平负准确率。Log Loss 越低越好；Brier 沿用项目已有的三类平均均方误差口径。不同表中的整体历史准确率和单赛季准确率不能直接比较。

主选择指标预先设为验证季 Log Loss，同时冻结验证准确率最高的配置作为第二个目标。两者可能选出不同特征集，最终测试结果不能用于在二者之间重新挑赢家。

## 冻结的特征组合

### lr__logloss

验证选中 `window_10_only`，保留 36 列，C=1。验证准确率 47.99%，验证 Log Loss 1.03828。

最后评估准确率 47.20%，相对同模型完整 66 列变化 +0.491 个百分点；Log Loss 变化 -0.00213。

按比赛日期成组的 2,000 次配对 bootstrap：准确率变化 95% 区间 [-0.309, +1.364] 个百分点；Log Loss 变化区间 [-0.00493, +0.00073]。准确率区间包含零，不能据此声称稳定提升。该区间只描述这个回顾性评估季，不包含跨赛季、训练或选模不确定性。

```text
home_team_state_points_10
home_attack_strength_10
home_defense_strength_10
home_goals_for_10
home_goals_against_10
home_goal_differential_10
home_win_rate_10
home_draw_rate_10
home_loss_rate_10
home_win_margin_goals_10
home_loss_margin_goals_10
home_avg_shots_10
home_avg_sot_10
home_shot_accuracy_10
home_conversion_rate_10
away_team_state_points_10
away_attack_strength_10
away_defense_strength_10
away_goals_for_10
away_goals_against_10
away_goal_differential_10
away_win_rate_10
away_draw_rate_10
away_loss_rate_10
away_win_margin_goals_10
away_loss_margin_goals_10
away_avg_shots_10
away_avg_sot_10
away_shot_accuracy_10
away_conversion_rate_10
h2h_home_wins
h2h_away_wins
h2h_draws
h2h_home_avg_goals
h2h_away_avg_goals
league_level
```

### lr__accuracy

验证选中 `window_10_only`，保留 36 列，C=1。验证准确率 47.99%，验证 Log Loss 1.03828。

最后评估准确率 47.20%，相对同模型完整 66 列变化 +0.491 个百分点；Log Loss 变化 -0.00213。

按比赛日期成组的 2,000 次配对 bootstrap：准确率变化 95% 区间 [-0.309, +1.364] 个百分点；Log Loss 变化区间 [-0.00493, +0.00073]。准确率区间包含零，不能据此声称稳定提升。该区间只描述这个回顾性评估季，不包含跨赛季、训练或选模不确定性。

```text
home_team_state_points_10
home_attack_strength_10
home_defense_strength_10
home_goals_for_10
home_goals_against_10
home_goal_differential_10
home_win_rate_10
home_draw_rate_10
home_loss_rate_10
home_win_margin_goals_10
home_loss_margin_goals_10
home_avg_shots_10
home_avg_sot_10
home_shot_accuracy_10
home_conversion_rate_10
away_team_state_points_10
away_attack_strength_10
away_defense_strength_10
away_goals_for_10
away_goals_against_10
away_goal_differential_10
away_win_rate_10
away_draw_rate_10
away_loss_rate_10
away_win_margin_goals_10
away_loss_margin_goals_10
away_avg_shots_10
away_avg_sot_10
away_shot_accuracy_10
away_conversion_rate_10
h2h_home_wins
h2h_away_wins
h2h_draws
h2h_home_avg_goals
h2h_away_avg_goals
league_level
```

### rf__logloss

验证选中 `groups__form__h2h__margins__shooting`，保留 45 列，250 棵树、深度 10、叶节点最少 8。验证准确率 47.94%，验证 Log Loss 1.04159。

最后评估准确率 45.63%，相对同模型完整 66 列变化 -0.246 个百分点；Log Loss 变化 -0.00212。

按比赛日期成组的 2,000 次配对 bootstrap：准确率变化 95% 区间 [-1.044, +0.563] 个百分点；Log Loss 变化区间 [-0.00487, +0.00043]。准确率区间包含零，不能据此声称稳定提升。该区间只描述这个回顾性评估季，不包含跨赛季、训练或选模不确定性。

```text
home_team_state_points_5
home_win_rate_5
home_draw_rate_5
home_loss_rate_5
home_win_margin_goals_5
home_loss_margin_goals_5
home_avg_shots_5
home_avg_sot_5
home_shot_accuracy_5
home_conversion_rate_5
away_team_state_points_5
away_win_rate_5
away_draw_rate_5
away_loss_rate_5
away_win_margin_goals_5
away_loss_margin_goals_5
away_avg_shots_5
away_avg_sot_5
away_shot_accuracy_5
away_conversion_rate_5
home_team_state_points_10
home_win_rate_10
home_draw_rate_10
home_loss_rate_10
home_win_margin_goals_10
home_loss_margin_goals_10
home_avg_shots_10
home_avg_sot_10
home_shot_accuracy_10
home_conversion_rate_10
away_team_state_points_10
away_win_rate_10
away_draw_rate_10
away_loss_rate_10
away_win_margin_goals_10
away_loss_margin_goals_10
away_avg_shots_10
away_avg_sot_10
away_shot_accuracy_10
away_conversion_rate_10
h2h_home_wins
h2h_away_wins
h2h_draws
h2h_home_avg_goals
h2h_away_avg_goals
```

### rf__accuracy

验证选中 `groups__form__h2h__margins__shooting`，保留 45 列，250 棵树、深度 10、叶节点最少 8。验证准确率 47.94%，验证 Log Loss 1.04159。

最后评估准确率 45.63%，相对同模型完整 66 列变化 -0.246 个百分点；Log Loss 变化 -0.00212。

按比赛日期成组的 2,000 次配对 bootstrap：准确率变化 95% 区间 [-1.044, +0.563] 个百分点；Log Loss 变化区间 [-0.00487, +0.00043]。准确率区间包含零，不能据此声称稳定提升。该区间只描述这个回顾性评估季，不包含跨赛季、训练或选模不确定性。

```text
home_team_state_points_5
home_win_rate_5
home_draw_rate_5
home_loss_rate_5
home_win_margin_goals_5
home_loss_margin_goals_5
home_avg_shots_5
home_avg_sot_5
home_shot_accuracy_5
home_conversion_rate_5
away_team_state_points_5
away_win_rate_5
away_draw_rate_5
away_loss_rate_5
away_win_margin_goals_5
away_loss_margin_goals_5
away_avg_shots_5
away_avg_sot_5
away_shot_accuracy_5
away_conversion_rate_5
home_team_state_points_10
home_win_rate_10
home_draw_rate_10
home_loss_rate_10
home_win_margin_goals_10
home_loss_margin_goals_10
home_avg_shots_10
home_avg_sot_10
home_shot_accuracy_10
home_conversion_rate_10
away_team_state_points_10
away_win_rate_10
away_draw_rate_10
away_loss_rate_10
away_win_margin_goals_10
away_loss_margin_goals_10
away_avg_shots_10
away_avg_sot_10
away_shot_accuracy_10
away_conversion_rate_10
h2h_home_wins
h2h_away_wins
h2h_draws
h2h_home_avg_goals
h2h_away_avg_goals
```

### lr_scaled__logloss

验证选中 `window_10_only`，保留 36 列，C=0.1。验证准确率 48.18%，验证 Log Loss 1.03838。

最后评估准确率 47.25%，相对同模型完整 66 列变化 +0.442 个百分点；Log Loss 变化 -0.00299。

按比赛日期成组的 2,000 次配对 bootstrap：准确率变化 95% 区间 [-0.428, +1.333] 个百分点；Log Loss 变化区间 [-0.00627, +0.00029]。准确率区间包含零，不能据此声称稳定提升。该区间只描述这个回顾性评估季，不包含跨赛季、训练或选模不确定性。

```text
home_team_state_points_10
home_attack_strength_10
home_defense_strength_10
home_goals_for_10
home_goals_against_10
home_goal_differential_10
home_win_rate_10
home_draw_rate_10
home_loss_rate_10
home_win_margin_goals_10
home_loss_margin_goals_10
home_avg_shots_10
home_avg_sot_10
home_shot_accuracy_10
home_conversion_rate_10
away_team_state_points_10
away_attack_strength_10
away_defense_strength_10
away_goals_for_10
away_goals_against_10
away_goal_differential_10
away_win_rate_10
away_draw_rate_10
away_loss_rate_10
away_win_margin_goals_10
away_loss_margin_goals_10
away_avg_shots_10
away_avg_sot_10
away_shot_accuracy_10
away_conversion_rate_10
h2h_home_wins
h2h_away_wins
h2h_draws
h2h_home_avg_goals
h2h_away_avg_goals
league_level
```

### lr_scaled__accuracy

验证选中 `window_10_only`，保留 36 列，C=0.1。验证准确率 48.18%，验证 Log Loss 1.03838。

最后评估准确率 47.25%，相对同模型完整 66 列变化 +0.442 个百分点；Log Loss 变化 -0.00299。

按比赛日期成组的 2,000 次配对 bootstrap：准确率变化 95% 区间 [-0.428, +1.333] 个百分点；Log Loss 变化区间 [-0.00627, +0.00029]。准确率区间包含零，不能据此声称稳定提升。该区间只描述这个回顾性评估季，不包含跨赛季、训练或选模不确定性。

```text
home_team_state_points_10
home_attack_strength_10
home_defense_strength_10
home_goals_for_10
home_goals_against_10
home_goal_differential_10
home_win_rate_10
home_draw_rate_10
home_loss_rate_10
home_win_margin_goals_10
home_loss_margin_goals_10
home_avg_shots_10
home_avg_sot_10
home_shot_accuracy_10
home_conversion_rate_10
away_team_state_points_10
away_attack_strength_10
away_defense_strength_10
away_goals_for_10
away_goals_against_10
away_goal_differential_10
away_win_rate_10
away_draw_rate_10
away_loss_rate_10
away_win_margin_goals_10
away_loss_margin_goals_10
away_avg_shots_10
away_avg_sot_10
away_shot_accuracy_10
away_conversion_rate_10
h2h_home_wins
h2h_away_wins
h2h_draws
h2h_home_avg_goals
h2h_away_avg_goals
league_level
```

## 当前模型的重要性排序

以下排序用 2022/23 赛季，模型仅用之前赛季训练；每列在联赛内打乱 5 次。主排序数值是打乱后 Log Loss 增加量，正值越大表明该模型在该评估季越依赖该列。league_level 单独全局打乱。标准差反映 5 次打乱之间的变化，不是跨赛季置信区间。

所有 66 列的完整排名已保存为 CSV。这里列出前 15 名及特征组重要性。

### lr

| rank | feature | permutation_logloss_mean | permutation_logloss_std | permutation_accuracy_mean |
| --- | --- | --- | --- | --- |
| 1 | home_avg_sot_10 | 0.020728 | 0.004337 | 0.019352 |
| 2 | away_avg_sot_10 | 0.015984 | 0.002626 | 0.016994 |
| 3 | away_avg_shots_5 | 0.014878 | 0.000957 | 0.014637 |
| 4 | away_attack_strength_5 | 0.013947 | 0.002084 | -0.002259 |
| 5 | away_goal_differential_5 | 0.011707 | 0.001625 | 0.012083 |
| 6 | away_avg_sot_5 | 0.007996 | 0.002299 | 0.006778 |
| 7 | away_avg_shots_10 | 0.007045 | 0.000532 | 0.011788 |
| 8 | away_goal_differential_10 | 0.006755 | 0.002547 | 0.010707 |
| 9 | away_win_margin_goals_10 | 0.005745 | 0.001257 | 0.005305 |
| 10 | home_win_margin_goals_5 | 0.004917 | 0.001819 | 0.003242 |
| 11 | away_goals_for_5 | 0.004666 | 0.001581 | 0.004617 |
| 12 | away_defense_strength_5 | 0.004354 | 0.001872 | 0.005108 |
| 13 | away_attack_strength_10 | 0.004271 | 0.001658 | -0.001965 |
| 14 | home_defense_strength_5 | 0.003477 | 0.000590 | 0.004715 |
| 15 | away_team_state_points_5 | 0.003188 | 0.000980 | 0.001081 |

| family | n_features | permutation_logloss_mean | permutation_logloss_std | permutation_accuracy_mean |
| --- | --- | --- | --- | --- |
| shooting | 16 | 0.022788 | 0.003942 | 0.020236 |
| goals_strength | 20 | 0.009326 | 0.003922 | 0.01002 |
| margins | 8 | 0.008714 | 0.002338 | 0.008546 |
| h2h | 5 | 0.005008 | 0.002078 | 0.001965 |
| form | 16 | 0.00374 | 0.002104 | -0.001473 |
| league | 1 | 0.001022 | 0.000933 | 9.8e-05 |

### rf

| rank | feature | permutation_logloss_mean | permutation_logloss_std | permutation_accuracy_mean |
| --- | --- | --- | --- | --- |
| 1 | away_avg_shots_5 | 0.001775 | 0.000260 | 0.002554 |
| 2 | away_avg_shots_10 | 0.001507 | 0.000586 | 0.003929 |
| 3 | home_avg_sot_10 | 0.000923 | 0.000484 | -0.001375 |
| 4 | h2h_home_avg_goals | 0.000794 | 0.000252 | -0.000098 |
| 5 | home_avg_shots_10 | 0.000716 | 0.000708 | -0.000295 |
| 6 | away_conversion_rate_5 | 0.000589 | 0.000362 | -0.000786 |
| 7 | away_win_margin_goals_5 | 0.000537 | 0.000279 | 0.000393 |
| 8 | home_goal_differential_10 | 0.000447 | 0.000679 | -0.000393 |
| 9 | h2h_away_avg_goals | 0.000407 | 0.000227 | -0.002063 |
| 10 | home_win_margin_goals_10 | 0.000395 | 0.000208 | -0.001179 |
| 11 | home_avg_sot_5 | 0.000357 | 0.000237 | -0.000196 |
| 12 | away_draw_rate_10 | 0.000334 | 0.000090 | -0.001866 |
| 13 | home_win_margin_goals_5 | 0.000326 | 0.000176 | -0.000295 |
| 14 | away_win_rate_5 | 0.000266 | 0.000117 | -0.001768 |
| 15 | home_goals_against_10 | 0.000197 | 0.000325 | -0.000688 |

| family | n_features | permutation_logloss_mean | permutation_logloss_std | permutation_accuracy_mean |
| --- | --- | --- | --- | --- |
| goals_strength | 20 | 0.011394 | 0.001525 | 0.018566 |
| shooting | 16 | 0.009945 | 0.002652 | 0.012672 |
| h2h | 5 | 0.001241 | 0.000583 | 0.000786 |
| margins | 8 | 0.000267 | 0.001462 | 0.001473 |
| league | 1 | 3.9e-05 | 8.9e-05 | -0.001768 |
| form | 16 | -0.00046 | 0.000864 | 0.002259 |

## 三个历史赛季的排名稳定性

2022/23 是选择候选前的排序季。额外核查 2023/24、2024/25 的排名，每季仍只用此前赛季训练。这些结果是描述性稳定性分析，没有反过来改变已冻结特征组合，完全未使用最后评估季。

### lr 跨开发季

| feature | mean_logloss_increase | mean_rank | worst_rank | positive_seasons | top15_seasons |
| --- | --- | --- | --- | --- | --- |
| away_attack_strength_5 | 0.016438 | 2.0 | 4 | 3 | 3 |
| away_avg_sot_10 | 0.01416 | 2.0 | 2 | 3 | 3 |
| home_avg_sot_10 | 0.013799 | 4.666667 | 10 | 3 | 3 |
| away_avg_sot_5 | 0.010352 | 4.333333 | 6 | 3 | 3 |
| away_avg_shots_5 | 0.010167 | 6.333333 | 11 | 3 | 3 |
| away_goal_differential_5 | 0.010035 | 5.0 | 6 | 3 | 3 |
| away_avg_shots_10 | 0.006565 | 9.666667 | 15 | 3 | 3 |
| away_attack_strength_10 | 0.00518 | 9.333333 | 13 | 3 | 3 |
| away_goals_for_5 | 0.004999 | 9.333333 | 11 | 3 | 3 |
| away_goal_differential_10 | 0.004541 | 11.333333 | 16 | 3 | 2 |

### rf 跨开发季

| feature | mean_logloss_increase | mean_rank | worst_rank | positive_seasons | top15_seasons |
| --- | --- | --- | --- | --- | --- |
| away_avg_shots_10 | 0.001881 | 1.666667 | 2 | 3 | 3 |
| home_avg_shots_10 | 0.00142 | 2.666667 | 5 | 3 | 3 |
| away_avg_shots_5 | 0.001111 | 12.333333 | 33 | 3 | 2 |
| home_goal_differential_10 | 0.001048 | 5.0 | 8 | 3 | 3 |
| home_avg_sot_10 | 0.000895 | 5.333333 | 7 | 3 | 3 |
| away_avg_sot_10 | 0.000735 | 11.333333 | 24 | 3 | 2 |
| h2h_away_avg_goals | 0.00058 | 9.0 | 11 | 3 | 3 |
| home_avg_shots_5 | 0.000498 | 20.666667 | 45 | 2 | 2 |
| h2h_away_wins | 0.000409 | 14.333333 | 16 | 3 | 2 |
| home_team_state_points_10 | 0.00038 | 21.666667 | 48 | 2 | 2 |

## 逐组删除的控制实验

这些删减项目在查看最后评估结果之前已固定，用于解释信息组贡献。它们不是根据最后评估季挑选的模型，不能据此追加特征选择。

| model | config | n_features | n_matches | accuracy | macro_f1 | draw_f1 | log_loss | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr | all_66 | 66 | 2036 | 46.71% | 0.34019 | 0.00000 | 1.05192 | 0.21117 |
| lr | drop_form | 50 | 2036 | 47.15% | 0.34354 | 0.00377 | 1.05042 | 0.21086 |
| lr | drop_goals_strength | 46 | 2036 | 46.86% | 0.34076 | 0.00000 | 1.05122 | 0.21098 |
| lr | drop_h2h | 61 | 2036 | 46.37% | 0.33429 | 0.00000 | 1.05102 | 0.21112 |
| lr | drop_league | 65 | 2036 | 46.46% | 0.33751 | 0.00000 | 1.05111 | 0.21100 |
| lr | drop_margins | 58 | 2036 | 47.15% | 0.34413 | 0.00374 | 1.05024 | 0.21077 |
| lr | drop_shooting | 50 | 2036 | 46.71% | 0.33402 | 0.00000 | 1.05749 | 0.21238 |
| rf | all_66 | 66 | 2036 | 45.87% | 0.33072 | 0.00000 | 1.05366 | 0.21158 |
| rf | drop_form | 50 | 2036 | 46.27% | 0.33257 | 0.00000 | 1.05253 | 0.21135 |
| rf | drop_goals_strength | 46 | 2036 | 46.22% | 0.33287 | 0.00000 | 1.05429 | 0.21176 |
| rf | drop_h2h | 61 | 2036 | 45.92% | 0.33014 | 0.00000 | 1.05438 | 0.21183 |
| rf | drop_league | 65 | 2036 | 46.37% | 0.33399 | 0.00000 | 1.05318 | 0.21151 |
| rf | drop_margins | 58 | 2036 | 46.02% | 0.33178 | 0.00000 | 1.05413 | 0.21173 |
| rf | drop_shooting | 50 | 2036 | 46.22% | 0.33207 | 0.00000 | 1.05718 | 0.21239 |

## 实验协议

数据为 English4 的 13,988 场，来自 E0–E3 的 2019/20–2025/26。每场输入都在更新当场及当天结果之前构造，球队历史跨联赛延续。

1. 重要性季：2022/23；训练季为 2019/20–2021/22。
2. 组合选择：分别训练到 2022/23、2023/24，验证 2023/24、2024/25，再合并逐场预测计算指标。
3. 最后评估：用 2019/20–2024/25 训练，评估 2025/26。
4. 赔率只用于独立市场基准，不用于重要性排序、模型训练或候选选择。所有市场比较使用有效 Bet365 收盘赔率。
5. 未使用半场信息、当场原始统计、FootyStats 或当前赛果标签作为特征。历史进球、射门和射正只在比赛完成后进入历史队列。

## 搜索范围

六组分别为 form（平均积分与胜平负历史）、goals_strength（进失球、净胜球、攻防强度）、margins（胜负分差）、shooting（射门、射正及效率）、h2h（交锋）、league（联赛级别）。所有非空组组合共有 63 种；另外加入 Top-8/16/24/32/48/60、删除末位 5/10/20、正置换重要性、5 场或 10 场窗口和相关性阈值 0.95 去冗余。重复特征集去重。

实际完成 380 个模型与参数配置、760 次验证训练。原逻辑回归 C=1；随机森林保持 250 树、深度 10、叶节点最少 8；标准化逻辑回归额外搜索 C=0.1/1/10。标准化比较是独立预处理分支，其改进不能全部归因于删特征。

仅计算有限候选内的验证最优组合；没有穷举 66 列的全部组合，也不保证数学意义上的全局最优。

## 数据及实现审查

- 原构建代码将 league_level 同时保留在原表和特征表，选择时实际传入 67 列；本实验只保留一列。
- 按月份划分赛季会把七月的揭幕比赛归到上一季。本次按 CSV 所属赛季改正 35 场，并检查每折训练最大日期早于评估最小日期。
- 原攻防归一化的累计联赛均值会吸收此前遍历的同一天比赛。本实验按日期批量构造，用前一天以前的结果计算均值。
- 当前平均积分=3×胜率+平局率，净胜球=进球−失球，胜平负率之和=1，存在精确线性冗余。打乱单列会破坏这些关系；特征间的代偿也会影响数值，所以单列置换排名不能直接解释为因果作用或独立足球价值。
- 五项自动检查已通过：当前及同日结果不影响输入、七月赛季归属与单列联赛标记、时间重叠拒绝、类别与 Brier 指标口径、整组置换保留行内关系及联赛范围。

- 导出的六个冻结模型已与原逐场预测核对；重新加载模型后概率也一致。重读特征 CSV 必须使用 float_precision='round_trip'，保留浮点值的精确往返；否则在这些高度相关的变量下，微小解析差异可能改变优化停止点。

## 结论的边界

2025/26 曾用于该项目之前的实验。本次代码没有用它调参，但它不是项目从未观察过的新数据。只有一个最后评估赛季，仍需要新赛季或另一个独立联赛确认。低重要性并不证明信息无效，增加噪声列也未必让所有模型表现更差；删除效果必须看重训练后的实际验证结果。

原报告 46.01% / 45.62% 是多个赛季的整体数字；本报告表格是 2025/26 单季。修正后的数据和输入列数也不同，因此不会将二者直接相减并称为提升。

## 文献与复现

文献排序见 `../literature_review.md`。

```powershell
python -B -u experiments/feature_search_20261010/run_feature_search.py --project-root . --output-dir experiments/feature_search_20261010/results
```

`protocol.json` 记录预先规定的分割、随机种子、原始 CSV 哈希；`selection_before_test.json` 在最后评估之前写入选择结果；`frozen_selections.json` 列出保留列；`final_test_predictions.csv` 可逐场审计。
