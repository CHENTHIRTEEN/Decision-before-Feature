# 14i · Behavior 策略风险与特征/身份诊断（Task 13K5-K6/13Q/13R）

- 日期：2026-08-29。产物：`oof_policy_performance.parquet`、`feature_importance_diagnostic.parquet`、`problem_id_diagnostic.parquet`、行级 `results/analysis_v5/task13/{oof_action_loss_predictions, oof_problem_id_diagnostic_predictions}.parquet`。

## 1. 策略风险（K5/K6，RF 主 carrier）

| 量 | M0（current+FE） | M2（current+FE+B） | 说明 |
|---|---:|---:|---|
| harmful rate（realized > continue + δ_pair） | 0.065 / 0.044 | **0.136 / 0.137** | M2 切换更激进，Practical 有害比例约翻倍——**风险项，须如实声明** |
| switch rate（部署动作 ≠ current） | 0.256 / 0.269 | 0.642 / 0.591 | 基率仅 0.26；M2 在大量 continue-acceptable 状态也切换 |
| switch precision（P(Z=1 \| switched)） | 0.151 / 0.235 | **0.293 / 0.310** | 高于基率 0.258/0.265，但绝对水平低 |
| switch recall | 0.149 / 0.238 | **0.730 / 0.692** | 抓住多数真实 switch opportunity |
| unnecessary-switch rate | 0.849 / 0.766 | 0.707 / 0.690 | — |
| 在"已切换且 Z=1"中选中 12.1 最优 target | — | **69.4%**（n=353） | target 选择与 practical 集合高度一致 |

判读：M2 用"更高切换频率 + 中等精度"换取 fb loss 改善——平均收益为正（两 suite），但**单状态风险上升**（harmful 6.5%→13.6%/13.7%）。部署形态上 M2 是"进取型"策略；若部署侧对单步损失敏感，应搭配 min-dwell/confirm 机制（后续 post-handoff 轮的设计输入，本轮不实现）。

## 2. 特征贡献（13Q，permutation importance，全量拟合，诊断性）

- BBOB top：`cur_cso`（0.093，current 身份）、`bf_elite_centroid_shift_w05`（0.065）、`bf_diversity_mean_pairwise`（0.059）、`bf_population_centroid_shift_w05`、`bf_centroid_shift_rate_w05`、`bf_population_chamfer_distance_w05`——**种群运动/形状组主导**；
- MA top：`bf_fitness_diversity_rel`（0.122）、`bf_fitness_wasserstein_rate_w02`（0.071）、`bf_best_fitness_slope_rel_w05`、`bf_fitness_distribution_improvement_rate_w02`、`bf_improvement_rate_w02`——**fitness 分布改进组主导**；
- time-like 的 `bf_fe_ratio` 未进入任一 suite top-8（与 14h 剔除敏感性一致）。本诊断仅为 group 级关联证据，**不构成因果贡献声明**。

## 3. Problem-ID 诊断（13R，NON-DEPLOYABLE DIAGNOSTIC）

| carrier | suite | [problem, current, FE] | + Behavior | Δ |
|---|---|---:|---:|---:|
| RF | BBOB | −1.5698 | −1.6129 | **+0.0431** |
| RF | MA | −4.5262 | −4.5823 | **+0.0561** |
| Ridge | BBOB | −1.6012 | −1.5830 | −0.0182 |
| Ridge | MA | −4.5299 | −4.5174 | −0.0125 |

RF 下即便 problem identity 已知，Behavior 仍带来与 deployment 口径同量的增量（+0.043/+0.056）——**Behavior 不是隐式 problem identifier**（与 14g 的 within-problem 结果互证）。Ridge 反向（42 个 problem 哑变量 + 4-row/低信噪场景下线性模型退化），再次确认载体能力边界。**部署模型严格不含 problem_id/function_id/family/cv_group_id/suite**（诊断模型单独标记，不进入部署结论）。

## 4. 边界

- harmful/regret 的参照（continue、逐状态最佳动作）分别受"分寸 δ 语义"与"winner's curse（14a：fb +0.112/+0.034）"影响，仅作辅助；
- 特征贡献为置换关联，不构成因果归因；
- 诊断模型一律 NON-DEPLOYABLE，不参与任何 verdict。
