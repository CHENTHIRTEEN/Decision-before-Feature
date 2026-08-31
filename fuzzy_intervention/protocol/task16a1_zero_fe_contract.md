# Task16A.1 零 FE 事后诊断合同

Task16A.1 为零 FE 的事后诊断性复核，不替代 Task16A 预先指定的正式结论。

## 数据边界

本分析只读取 `fuzzy_intervention/results/task16a/` 中已生成的 source states、primary action outcomes、repetitions、probe ranks、noise scales 与 practical action sets。分析源码不得导入 benchmark、optimizer、trajectory runner 或 objective evaluation 接口。

资源账本必须满足 `new_objective_FE = 0`。若输入缺失或需要重新评价 objective，任务立即停止。

## 保持不变的定义

- P：`bf_fitness_distribution_improvement_rate_w02`；
- H：`1-bf_centroid_shift_coherence_w05`；
- S：`bf_stagnation_w10`；
- M：`source_FE/10000`；
- LOW：rank ≤ 1/3；MED：1/3 < rank < 2/3；HIGH：rank ≥ 2/3；
- R1：P=HIGH、S=LOW；
- R2：P=LOW、S=HIGH；
- R3：R2 且 H=LOW；
- R4：R2 且 H=HIGH。

必须从 Task16A rank 重建 regime，并与 Task16A 保存的 regime flags 逐行一致。

## 基础 cell 与支持度

基础 cell 固定为 `suite × current_algorithm × source_FE`。任何 regime contrast 只有在 cell 内两个被比较 regime 各至少 5 个状态时才进入标准化聚合；其他 cell 保留并标记 `LOW_SUPPORT`。

标准化主估计对有效基础 cell 等权；敏感性估计按两个 regime 的状态总数加权。每个 cell 内的二元比例先按 `cv_group_id` 求均值，再计算 regime 差。

## Grouped bootstrap

- 5000 draws；
- `cv_group_id/problem` 为重采样单位；
- 每次 draw 在每个 cell 重新计算 regime contrast，再按有效 cell 等权聚合；
- 报告点估计、95% percentile interval、有效 cell 数与原始分布统计；
- 本任务不进行结果后 cutoff 搜索，不删除 outlier，不选择性省略分层。

## 连续 advantage

- `G_P = L_C - L_PT`；
- `G_PR = L_C - L_PR`；
- `G_S = L_C - min(L_S1,L_S2)`；
- `G_SP = L_PT - min(L_S1,L_S2)`；
- `G_I = max(G_P,G_S)`，仅作逐状态最佳已观测干预诊断。

主位置统计量为 median；同时报告 function-balanced mean、q25、q75 与 `P(G>0)`。B1/B2/B3 比较 R2-R1；B4 比较 R3-R4。最终解释以基础 cell 等权标准化结果为主。

## Noise 异质性

沿用 Task16A 的 repetition difference：对每个重复 state-action，计算 repetition 1/2 相对 repetition 0 的 loss 绝对差，并取 95% 分位数。

报告 global、suite、current、FE、suite×current、current×FE。只有 `n_repeated_pairs ≥ 15` 的 cell 才给出 cell-specific delta95；其余保留支持数并标记 `LOW_SUPPORT`。

每种 conditioning 内报告 `max(delta)/min(delta)` 与 `sd(delta)/mean(delta)`。总体 noise verdict 的主视图预先指定为 `current × FE`：

- N1：至少两个 concrete actions 的支持充分 cell 满足 ratio ≥ 2；
- N2：未达到 N1，但至少两个 actions 满足 ratio ≥ 1.3，或一个 action 满足 ratio ≥ 2；
- N3：其余情况。

仅 N1 时构造事后 local-threshold sensitivity，优先 `action×current×FE`，不足时依次使用 `action×current`、`action global`。该 sensitivity 不改写 Task16A。

## Verdict 操作定义

### Part A

- A-REVERSAL：标准化 pooled 差 >0；至少一个 suite CI 下界 >0；另一 suite CI 上界 ≥0；
- A-CONFIRM：标准化 pooled 差 <0；两个 suite 点估计均 <0；至少一个 suite CI 上界 <0；
- A-NULL：其余情况。

### Part B

B-STRUCTURE 要求 B1 `G_I` 标准化 R2-R1 在两个 suite 均 >0、至少一个 suite CI 下界 >0、另一 suite CI 上界 ≥0、至少两个 solvers 点估计 >0，并且 B4 至少一个 action advantage 在两个 suite 呈相同非零符号且至少一个 suite CI 不跨 0。

B-PARTIAL：不满足 B-STRUCTURE，但 B1 在某个 suite 或 solver 的 CI 下界 >0，或 B4 满足上述 entropy interaction 条件。其余为 B-NONE。

### Part M

- M1：BBOB/MA 的 R2 Spearman 符号一致，至少两个 solvers 同符号，且 pooled 或至少一个 suite CI 不跨 0；
- M2：不满足 M1，但任一 suite×solver CI 不跨 0，或至少两个局部分层具有相同符号；
- M3：其余情况。

### Joint

J1 要求 A-REVERSAL 或 B-STRUCTURE，同时两个 suite 各至少 6 个有效基础 cell、不是单一 solver 驱动、无显著相反 suite；若结论依赖 local threshold，还必须为 N1。J1 也不允许直接运行模糊控制器。

若 `(A-CONFIRM 或 A-NULL) 且 B-NONE 且 M3`，并且 local sensitivity 不提供跨 suite/solver 稳定结构或按规则未构造，则为 J3。其他局部或不确定结构为 J2。

Task16A 的 F3 与当前 Perturb 的 A3 在所有情况下保持不变。

