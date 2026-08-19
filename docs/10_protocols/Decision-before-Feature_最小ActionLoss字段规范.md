# 最小 Action Loss 字段规范 v1

> 唯一活动字段规范（2026-08-16）。本文定义 action loss shard 的最小必需字段集合。方案 A 主功效为 `G_FE`（等总 FE，runtime 不进入主标签）；时间保留为辅助端点。旧 `u_query_joint_lamT_*` / `need_query_joint_lamT_*` / `performance_gain_norm` / `time_cost_norm` 等仅作兼容诊断，不作为活动训练或评估的默认 target。

## 1. 核心原则

1. 每条 action 记录只保留一个 canonical loss（`action_loss`）。
2. canonical loss 必须能从科学端点和时间端点复算或校验。
3. 不能只保留"收敛时间"或"找到最优解时间"作为唯一 action loss。
4. 时间是辅助端点，不是主功效标签的组成部分。
5. 凡是不能从其他字段无损复算、会影响 action 排序、会影响 selector 训练、会影响失败/完成解释或会影响时间端点比较的字段，都必须保留。

## 2. 五条科学路径

每条 action loss shard 对应五条科学路径之一：

| 路径 | 说明 | 仍预算 |
|---|---|---|
| `skip` | 原生 SBS continuation | $B - e_t$ |
| `query_joint` | Query + full Selector continuation | $B - e_t - e_q$ |
| `behavior_only_full_budget` | 不执行 query，full-budget 四动作 | $B - e_t$ |
| `query_matched_state_only` | Query + state-only Selector continuation | $B - e_t - e_q$ |
| `sampling_only_continue_current` | Query sample + 原生 continuation | $B - e_t - e_q$ |

`query_matched_state_only` 与 `sampling_only_continue_current` 只作 matched-acquisition 辅助分解，不作为主实验标签。

## 3. 最小必需字段

### 3.1 行标识字段

每条记录的唯一键：

| 字段 | 用途 |
|---|---|
| `split` | train / validation |
| `problem_id` | 静态问题标识 |
| `function_id` | BBOB 函数 ID |
| `family` | landscape family |
| `cv_group_id` | 交叉验证分组键 |
| `dimension` | 问题维度 |
| `prefix_algorithm` | 前缀算法 |
| `seed` | optimizer seed |
| `FE` | decision state 的整数 FE |
| `FE_ratio` | FE / FE_total |
| `query_id` | landscape query 配置 ID |
| `query_protocol` | query 协议版本 |
| `sample_design_id` | 样本设计 ID |
| `FE_total` | 总预算 |
| `FE_prefix` | 前缀 FE |
| `FE_query` | query 采样 FE |
| `FE_no_query_optimization` | no-query continuation FE |
| `FE_action_optimization` | action continuation FE |
| `action` | 动作名 |
| `target_algorithm` | 目标算法 |
| `transition_mode` | 转换模式 |
| `action_status` | completed / timed_out / failed |
| `sampling_protocol` | 采样协议 |
| `sampling_phase` | 采样阶段 |
| `sampling_triggers` | 采样触发器 |

### 3.2 科学端点字段

定义"这个 action 到底有多好"的核心字段：

| 字段 | 含义 |
|---|---|
| `benchmark_reference_value` | 基准参考值 |
| `log10_gap_floor` | gap 下限 |
| `log10_gap_cap` | gap 上限 |
| `success_gap_target` | 成功阈值 |
| `observed_first_hit_FE` | 首次命中 FE |
| `target_hit_observed` | 是否命中 target |
| `target_hit_before_failure` | 失败前是否已命中 |
| `path_completed` | 路径是否完成 |
| `endpoint_success` | 命中且完成 |
| `planned_FE` | 计划 FE |
| `effective_FE` | 实际有效 FE |

五条路径分别加前缀 `skip_` / `query_path_` / `behavior_path_` / `query_matched_state_only_path_` / `sampling_only_path_`（在 utility label 层），或在 action loss shard 层直接使用无前缀版本。

### 3.3 时间端点字段（辅助参考）

时间保留但不进入 `G_FE` 主标签。每条路径保存：

| 字段 | 含义 | 角色 |
|---|---|---|
| `runtime_*_median` | 三次 censored time 中位数 | 主时间端点 |
| `runtime_*_raw_observed_median` | 三次 raw observed time 中位数 | 仅诊断 |
| `runtime_*_repetitions` | 三次 raw observed time 列表 | 原始数据 |
| `runtime_*_censored_repetitions` | 三次 censored time 列表 | 复算用 |

completed repetition 的 censored time = raw；timed_out/failed repetition 的 censored time = `max(raw, role_timeout)`。

另有完整 policy 时间端点：

| 字段 | 含义 |
|---|---|
| `runtime_full_run_wall_clock_median` | FE=0→terminal 完整 policy censored 中位数 |
| `runtime_full_run_wall_clock_raw_observed_median` | 对应 raw 诊断 |

`FE=0` policy wall-clock 独立于状态条件 $T_k$，不进入 Utility。

### 3.4 Canonical loss 字段

| 字段 | 含义 | 角色 |
|---|---|---|
| `action_loss` | 主损失 | canonical loss |
| `action_loss_raw` | 原始损失值 | 可复算 |
| `action_loss_norm` | 归一化损失 | 诊断 |
| `log10_action_loss` | log10 变换 | 诊断 |
| `selector_target_loss` | selector 训练目标 | 供 selector 学习 |
| `loss_gap_raw` | gap 分量 | 解释性拆分 |
| `loss_gap_norm` | 归一化 gap | 诊断 |
| `best_observed_loss` | 最佳已观测动作损失 | 参考 |
| `selector_regret_raw` | selector 遗憾 | 比较动作差异 |
| `potential_gain_raw` | 潜在增益 | 诊断 |
| `performance_gain_raw` | 性能增益原始值 | 诊断 |

### 3.5 方案 A 主功效字段

在 utility label 层附加，不在 action loss shard 层：

| 字段 | 含义 |
|---|---|
| `g_fe` | 主功效 $G_{FE} = \log\frac{E_{skip}+\epsilon_p}{E_{query}+\epsilon_p}$ |
| `g_fe_bounded` | 有界版本 |
| `g_fe_gt_zero` | 布尔标签 |
| `g_fe_gt_practical` | 实用意义布尔标签 |
| `epsilon_p` | 问题尺度协变稳定项 |
| `delta_practical` | 实用阈值 |

## 4. 诊断字段（保留但不属于最小必需）

这些字段有价值，但不属于"最小规范"：

- `failure_type`
- `failure_message`
- `failure_loss_cap`
- `timing_replay_status_*`
- `timing_replay_timeout_seconds`
- `timing_repetition_indices`
- `timing_order_protocol`
- `action_runtime_role`
- `action_outcome_execution_count`
- `peak_memory_*`
- `selected_matches_best_observed`
- `timing_replay_path_identity_consistent_*`
- `completed_timing_replay_outcomes_internally_consistent_*`
- `stage_a_to_completed_timing_replays_consistent_*`
- `timing_replay_status_instability_*`
- `stage_a_stage_b_completion_status_instability_*`

## 5. 退役字段

以下字段不再作为活动规范的一部分：

| 字段 | 退役原因 |
|---|---|
| `u_query_lamT_*` | 旧单一 Utility，已被 `g_fe` 替代 |
| `u_query_joint_lamT_*` | 旧 joint Utility，仅作兼容诊断 |
| `u_behavior_only_full_budget_lamT_*` | 旧 behavior-only Utility，仅作兼容 |
| `need_query_joint_lamT_*` | 旧布尔标签，被 `g_fe_gt_zero` 替代 |
| `need_behavior_only_full_budget_lamT_*` | 旧 behavior-only 布尔标签 |
| `query_operational_increment_lamT_*` | 旧增量，仅作兼容 |
| `performance_gain_norm` | 旧归一化增益，已失效 |
| `performance_gain_norm_gap` | 旧 gap 版本 |
| `time_cost_norm` | 旧时间归一化，已被 censored median 替代 |
| `analysis_compute_cost_norm` | 旧计算成本归一化 |
| `memory_cost_norm` | 旧内存归一化 |
| `runtime_*_failure_worst_case_median` | 旧别名，等同 censored median |

旧 `results/ela/`、`FE_analysis`、`p_ela`、`u_ela_*`、`need_ela_*` 和缺少当前协议字段的 artifact 均为撤回结果，活动读取器必须明确拒绝。

## 6. 一致性校验入口

- `query-consistency`：检查三档预算、feature whitelist、共享样本键、零额外函数评价、BBOB group failure、整列缺失和 action-loss 预算隔离。
- `utility-labels-validate`：逐行重算五条路径及加法分解，校验 `g_fe` 从 `p_skip_raw` / `p_query_raw` / `benchmark_reference_value` / `epsilon_p` 复算一致，校验 `g_fe_gt_zero == (g_fe > 0)`。

## 7. 数据隔离规则

- `query_id`、`sample_design_id` 和 `FE_query` 必须在每条 action loss 记录中显式标注。
- 不同 `query_id`、`sample_design_id` 或 `FE_query` 的 outcomes 不得合并。
- Query-adjusted matrix 与 full-budget matrix 是两个不同 estimand 的数据，不能互换。
- Skip 的 `continue_current` 与 full-budget Behavior-only 的 `continue_current` 语义相同时只生成一次。
- population transfer 的影响已包含在 observed action loss 中，不作为额外减项重复计入。
