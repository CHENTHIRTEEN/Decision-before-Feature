# Task16A Probe Contract

本映射来自 Task15A Stage-A 的实际 `screened_behavior_concepts.parquet`，不读取任何 Task16A 动作结果。

| probe | Task15A concept | aggregate representative | direction used here |
|---|---|---|---|
| Productivity P | Progress / Contribution | `bf_fitness_distribution_improvement_rate_w02` | 数值越大表示 population fitness distribution 的单位尺度进展越大 |
| Entropy / Coordination H | Direction / Coordination | `bf_centroid_shift_coherence_w05` | `H = 1 - coherence`，数值越大表示群体运动越不一致 |
| Stagnation S | Stagnation | `bf_stagnation_w10` | 数值越大表示 best-so-far 停滞更严重 |
| Maturity M | budget context | `source_FE / 10000` | 只作 context，不是行为特征 |

P/H/S 必须从自然轨迹完整原生 update 历史构造的正式 behavior row 读取。不得使用真实最优值、未来评价、动作结果、function ID 或算法内部参数。

Task15A 用于 Targeted subset 的个体 primitive 定义保持不变：

- individual stagnation：最近 500 FE 窗口内，距最后一次有意义 fitness 改进的 FE age，截断到 500 FE；
- individual progress：最近 500 FE 内各完整原生 update 的非负 fitness 改进之和，除以 transition 数与当前 population fitness scale；
- stable agent identity：自然轨迹搜索槽位/target lineage；L-SHADE 缩减时按稳定排序保留对应身份，CSO 保持 loser slot 身份。

Targeted 排序不得读取 aggregate P/H/S 的 outcome strata，也不得读取动作 loss。

