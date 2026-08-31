# Task16A Protocol Amendment

日期：2026-08-31。

## 修改原因

在预先指定协议提交后、生成任何动作结果前，状态边界诊断发现：SHADE 与 CSO 的 2000/4000/6000/8000 FE checkpoint 均位于完整原生 update 边界；L-SHADE 的 2000、4000、6000 FE checkpoint 可能位于部分 update 中，其 pending trial 与当前 population 尚未整合。

若只为 Perturb 清除 pending buffer，会给 Perturb 引入额外状态变化；若在动作后完成 pending update，替换位置又可能被随后整合覆盖。两种做法都不能保持五动作从同一可解释状态开始。

## 修订后的 source endpoint

对每个名义 checkpoint，使用不晚于该位置的最近完整原生 update 作为实际 source state：

- 保存 `source_FE_nominal`、`source_FE_actual`、`source_FE_alignment_gap`；
- 要求 `0 <= source_FE_nominal - source_FE_actual < local_native_update_FE`；
- 五个动作从同一个实际完整状态分支；
- 每个动作从 `source_FE_actual` 起严格消耗 1000 FE；
- maturity 与 Early/Mid/Late 分层仍使用 `source_FE_nominal / 10000`；
- P/H/S 与个体 Targeted 排序只使用 `source_FE_actual` 及此前完整 update 历史。

该处理与项目对行为窗口的完整原生 update 对齐原则一致，并避免算法间因 pending buffer 处理不同而产生混杂。

## 对结果解释的影响

本修订不读取动作 loss，不改变 q、sigma、replacement、动作集合、repetition 选择、noise threshold、GO/NO-GO 判据或统计方法。L-SHADE 的实际动作起点可比名义 checkpoint 略早，必须在 source-state 与 solver/checkpoint 分层报告中披露；因此 checkpoint 结论解释为名义预算阶段附近的完整 update 状态，而非声称全部算法恰在相同整数 FE 上分支。

