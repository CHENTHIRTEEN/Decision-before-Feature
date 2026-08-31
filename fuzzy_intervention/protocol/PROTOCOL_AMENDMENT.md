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

## 原任务定性条件的操作定义

原任务对 A1/P1/“强 P2”使用了“非平凡”“稳健”“不由单一分层驱动”等定性条件。为避免结果后解释，在生成动作结果前确定如下操作定义：

- Continue 保留实际价值：pooled 非支配率至少 0.10，且两个 suite 均至少 0.05；
- Switch 保留实际价值：pooled `Z_S` 至少 0.10，且两个 suite 均至少 0.05；
- 分层覆盖：Targeted Perturb 非支配率与 `Z_S` 分别在至少两个 current algorithm、至少三个名义 checkpoint 中达到 0.05；
- A1：同时满足上述 Continue、Switch、分层覆盖、原 structural floor、原双向 0.08 complementarity，且每个状态的实际非支配集合非空；
- A2：不满足 A1，但 pooled Targeted Perturb 非支配率至少 0.05 且 `D_P` 至少 0.03；
- A3：不满足 A1 或 A2；
- P1 的 R2 动作结构条件：R2 中 `Z_P` 与 `Z_S` pooled 比例均至少 0.05；
- P1 的 maturity 条件：`Delta_M` 在 pooled、BBOB、MA-BBOB 三个口径符号一致且均不为 0；
- P1 的多算法条件：`Delta_intervention` 在至少两个 current algorithm 大于 0，且 R2 中 `Z_P`、`Z_S` 各自在至少两个 current algorithm 出现；
- strong P2：R2 相对 R1 的 `Z_I` 差在两个 suite 均大于 0、至少一个 suite 的 95% 区间下界大于 0，同时满足 R2 动作结构条件和多算法条件，但 P1 的 maturity 条件不成立；
- ordinary P2：存在 R2 相对 R1 的富集或单一 suite 清晰差异，但不满足 strong P2；
- P3：两个 suite 的 `Delta_intervention` 均不大于 0，或 R2 的 `Z_P` 与 `Z_S` 至少一项低于 0.05 且 maturity 不呈一致差异。

F1 仍要求 A1 且 P1 或 strong P2；F2 要求 A1 且未达到该 probe 条件；其余为 F3。
