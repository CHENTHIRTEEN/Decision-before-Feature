# 04 · Task 9D：修正后的 family-OOF action uncertainty 诊断

- 日期：2026-08-29
- 修复内容（相对 analysis_v2 Task 6 的两处实现问题）：
  1. **风险标签错误**：旧实现 `realized_switch_gain = true_switch_gain.max(axis=1)` 度量的是"真实最佳切换"的增益，而非模型实际选中动作的真实增益。修正为 `realized_selected_switch_gain = targets[prefix] - targets[selected_switch_action]`。
  2. **不确定性被低估**：旧实现在训练状态上用 full-train 集成计算树级方差。修正为 5 折 family-OOF：每折在其余 BBOB family + 无组件泄漏的 MA 行上重训同一回归载体（与 Task 1 OOF 完全同折、同种子），只对 held-out family 的 25,469 个 state 做逐树预测。已校验：OOF 集成均值与 Task 1 的 `oof_predictions.parquet` 逐行一致（偏差 < 1e-9）。
- 标签：`harmful H0`（选中切换真实增益 < 0）、`H50`（< -0.0564）、`H95`（< -1.4639）、`missed helpful`（未切换但存在真实增益 > +1.4639 的机会）。阈值保持 v2 的 0.1451 不变。
- 产物：`analysis_v3/task4/{uncertainty_roc_auc,uncertainty_pr_auc,risk_by_decile}.csv`；25,469 state 的完整 OOF 不确定性表在 `results/analysis_v3/task4/`。

## 1. 决策与修正后的错误结构（OOF，25,469 states）

| 统计 | 数值 |
|---|---:|
| 决策切换率 | 0.6736（17,156 次切换） |
| harmful H0（切换后真实增益 < 0） | 4,247（占切换 24.8%，占 state 16.7%） |
| harmful H50 | 3,496 |
| harmful H95 | **829**（旧标签为 744） |
| missed helpful（> +delta_95） | 140 |

新旧标签对照：旧"最佳切换增益"标签漏掉了 **85 个**"选中的动作有害、但恰好存在另一个高增益动作"的 state（反向不成立：旧标签没有多报）。即旧 Task 6 的风险率被系统性低估约 10%（H95 口径 744 → 829）。

## 2. 不确定性特征的识别能力（OOF；"高→风险"方向的 AUC，括号内为反向）

| 特征 | H0 | H50 | H95 | missed helpful |
|---|---:|---:|---:|---:|
| predicted_selected_switch_gain（预测选中切换增益） | **0.818** | **0.805** | 0.625 | (0.764) |
| top1-top2 loss margin | 0.726 | 0.749 | 0.701 | (0.727) |
| tree argmin disagreement | (0.669) | (0.691) | (0.849) | 0.781 |
| selected-action tree std | (0.505) | (0.504) | (0.648) | (0.515) |
| tree argmin vote entropy | (0.672) | (0.694) | (0.850) | 0.763 |

PR-AUC（基线率：H0 0.167、H50 0.137、H95 0.033、missed 0.0055）：增益边际对 H0/H50 的 PR-AUC 0.38/0.31；树一致性特征对 H95 的 PR-AUC 0.11。

## 3. 风险十分位曲线（要点）

- 按 `predicted_selected_switch_gain` 十分位：H0 风险率从底部三个十分位的 **0.0%** 单调升至顶部十分位的 **47.0%**；H95 风险率在中间十分位最高（d5 10.8%），顶部反而最低（d9 1.2%）——深度有害切换不发生在"预测增益最大"的状态，而在预测边际中等偏上的状态。
- 按 `tree_argmin_disagreement` 十分位：H95 集中在最低分歧端（d0 9.65%，d7–d9 为 0）——"集成一致但一致地错"的结论在 OOF 与修正标签下**依然成立**；而 missed helpful 集中在高分歧端（d8 3.4% vs d0 0.19%）。

## 4. 结论与边界

1. **修正标签 + 真正 OOF 后，action uncertainty 仍携带可识别的风险信号**：H0/H50 由预测增益边际排序（AUC ≈ 0.81–0.82，低边际几乎无风险），H95 由"高树间一致性 + 低预测边际"的组合标记。方向与旧 Task 6 一致，但数值更诚实（旧值基于训练集树方差 + 错误标签）。
2. 需要注意的混杂：预测增益边际与 prefix 高度相关（弱起始才有大预测增益，也有大风险），其排序能力部分是 prefix 信息的重述；在 cmaes-start 场景下该特征几乎无变化范围。
3. 按工作单判据：OOF 下不确定性可以识别 harmful decisions，**允许在后续研究中讨论 uncertainty-aware abstention 或 optional ELA query**；但 H95 基线率仅 3.3%、PR-AUC 低，直接做门控的代价（放弃大量大增益切换）需要专门评估。当前阶段仍不实现 Query Gate。
