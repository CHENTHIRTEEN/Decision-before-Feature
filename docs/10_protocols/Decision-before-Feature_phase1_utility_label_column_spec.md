# Decision-before-Feature phase1 query-specific Utility label 字段规范

> 唯一活动字段规范（2026-08-15）。Utility 使用配置截断后的 `log10_gap` 差与三次真实 decision-state-to-terminal future-path **censored median** wall-clock 的 `log10` ratio；raw observed median 只作诊断。共享 prefix 与分支前共同 Behavior/Decision 成本视为 sunk/common cost，FE=0 policy wall-clock 另存。旧 `performance_gain_norm`、`time_cost_norm`、单一 `u_query_lamT_*` 及由其生成的布尔标签全部失效。

## 1. 文件范围与状态键

每个 query 独立写入：

```text
results/utility_labels/{query_id}/
```

每张表只描述一个活动 query：

```text
descriptor_cheap_invariant
pflacco_standard_invariant
pflacco_broad_invariant
```

跨 trajectory、Behavior、action loss、Selection Reference、Utility 与 Decision dataset 的状态键为：

```text
(split, problem_id, family, dimension, prefix_algorithm, seed, FE)
```

`FE` 是实际整数评价数；`FE_ratio=FE/FE_total` 只作 metadata 与允许的时间特征，不作连接键。名义 milestone/event 字段不得覆盖实际 FE。

## 2. Query 与 sampling metadata

每行必须包含：

```text
query_id
query_protocol
query_preprocessing_id
query_feature_columns
sample_design_id
FE_query
```

`descriptor_cheap_invariant` 与 `pflacco_standard_invariant` 读取同一 `lhs_50d` sample 与相同 5% query-budget action outcomes，但分别拟合 Selector；`pflacco_broad_invariant` 使用独立 `lhs_100d` 与 10% outcomes。不同 sample design、query protocol、feature whitelist 或 action budget 不能混读。

## 3. Population 与动作关系

主行要求：

```text
prefix_algorithm == default_algorithm == fold-specific SBS
skip_switches_from_prefix == false
no_query_algorithm == default_algorithm
```

算法与 transition metadata 至少包括：

```text
default_algorithm
no_query_algorithm
selected_algorithm
selected_action
selected_equals_default
selected_equals_prefix
handoff_required
handoff_type
skip_switches_from_prefix
no_query_transition_mode
query_transition_mode
selector_prediction_source
selector_target_transform
```

逐行必须满足：

```text
handoff_required = not selected_equals_prefix
handoff_required = (handoff_type == population_transfer_initialization)
selector_target_transform = clipped_log10_gap_advantage_vs_continue_current
```

不得生成 `label_source`，也不得用 `same_algorithm/changed_algorithm` 替代三个显式关系字段。上述字段均为 metadata/分层字段，不进入 Decision X。

## 4. FE 账本

```text
FE_total
FE_prefix
FE_query
FE_skip_optimization = FE_total - FE_prefix
FE_query_optimization = FE_total - FE_prefix - FE_query
FE_behavior_only_optimization = FE_total - FE_prefix
```

query sample 目标评价计入 `FE_query`，但 sample 不并入 optimizer population。Query path 已通过减少 continuation FE 支付 query FE，Utility 不再额外按 FE 数量扣除。

## 5. 三条 outcome 与 Selector 诊断

每行至少保存：

```text
p_skip
p_query
p_query_continuation_only
p_behavior_only
selected_query_action_loss
selected_behavior_only_action_loss
query_sample_best_value
query_first_hit_offset
best_observed_query_action
best_observed_query_loss
best_observed_behavior_action
best_observed_behavior_loss
query_selector_regret_raw
behavior_selector_regret_raw
query_action_loss_range_raw
behavior_action_loss_range_raw
```

`p_query_continuation_only` 等于 Query Selector 所选动作的 continuation outcome；`p_query` 是主 operational Query terminal best，取 prefix、真实 query sample 与 selected continuation 中的最佳已观察值。`p_behavior_only` 等于 Behavior-only Selector 所选动作 outcome。query sample 不进入 optimizer population，但其 best/observed hit 进入主 Query path 的 terminal gap 与 ERT。`best observed action` 只作离线诊断，不称为 VBS 或 oracle，也不进入 Decision X。Selector 主训练 target 是每套矩阵内相对 `continue_current` 的 continuation-only `clipped_log10_gap_advantage_vs_continue_current`；旧 statewise min-max target 只作敏感性分析。Utility 读取真实 selected-path endpoint，不直接使用 target 尺度。

除 full Query Selector 与主 full-budget Behavior-only Selector 外，第三个 `query_adjusted_state_only_selector` 与 full Query Selector 读取同一 query-adjusted 四动作 outcomes，不新增 action losses。两者的 OOF selected continuation-only outcomes 还保存：

```text
query_feature_predictive_increment_log10_gap
```

该诊断排除 query sample best，不新增 action losses，只称 query features 的 OOF 边际预测贡献。

## 6. Gap 字段

benchmark reference 只用于离线标签和最终评价。BBOB train/validation 与 CEC2017 的冻结配置为：

```text
failure_loss_cap = 1e20
log10_gap_floor = 1e-12
log10_gap_cap = 1e20
success_gap_target = 1e-8
```

五条路径保存：

```text
benchmark_reference_value
gap_skip_raw
gap_query_raw
gap_query_continuation_only_raw
gap_behavior_only_raw
gap_query_matched_state_only_raw
gap_sampling_only_continue_current_raw
gap_skip_clipped
gap_query_clipped
gap_query_continuation_only_clipped
gap_behavior_only_clipped
gap_query_matched_state_only_clipped
gap_sampling_only_continue_current_clipped
log10_gap_skip
log10_gap_query
log10_gap_query_continuation_only
log10_gap_behavior_only
log10_gap_query_matched_state_only
log10_gap_sampling_only_continue_current
query_sample_best_contribution_log10_gap
{skip,query_path,query_matched_state_only_path,sampling_only_path,behavior_path}_observed_first_hit_FE
{skip,query_path,query_matched_state_only_path,sampling_only_path,behavior_path}_target_hit_observed
{skip,query_path,query_matched_state_only_path,sampling_only_path,behavior_path}_target_hit_before_failure
{skip,query_path,query_matched_state_only_path,sampling_only_path,behavior_path}_path_completed
{skip,query_path,query_matched_state_only_path,sampling_only_path,behavior_path}_endpoint_success
```

对 $k\in\{skip,query,query\_matched\_state\_only,sampling\_only,behavior\}$：

\[
g_k^{clip}=\min(\max(g_k^{raw},10^{-12}),10^{20}),
\qquad
\ell_k=\log_{10}g_k^{clip}.
\]

floor/cap 在取 `log10` 前作用于非负 raw gap，不作用于 objective value。`query_sample_best_contribution_log10_gap = log10_gap_query_continuation_only - log10_gap_query`。主 Query observed hit/ERT 合并 prefix、query sample 和 continuation 的真实评价序列；`query_first_hit_offset` 把 query 内命中位置映射到全路径 FE。continuation-only endpoint 另作诊断。每条 Stage-A/online 路径严格满足：

```text
target_hit_observed := observed_first_hit_FE != null
target_hit_before_failure := target_hit_observed and not path_completed
path_completed := status == completed
endpoint_success := target_hit_observed and path_completed
```

正式 ERT 使用 `target_hit_observed`，不用 `endpoint_success`。兼容字段 `first_hit_FE`/`success` 若保留，分别严格别名为 `observed_first_hit_FE`/`target_hit_observed`；不得把 `success` 改作 `endpoint_success`。用于有限模型 target 的 failure cap 不能掩盖 failure/completion 状态。

## 7. 三次 decision-state future-path 计时

计时固定为：

```text
timing_repetitions = 3
timing_order_protocol = cyclic_complete_path_v1
```

Stage A 的一次科学运行与 Selection Reference outcome 决定 terminal gap、observed hit、completion、endpoint success 及科学路径 FE；Stage B 的三次 replay 只决定 wall-clock，并提供一致性/失败诊断。每次从同一复制 complete optimizer state 和相同 RNG state 到 terminal，全部目标评价真实执行。

状态条件计时起点在 query/no-query 分支确定之后。共享 prefix、该机会的 Behavior extraction、Decision inference 与 threshold comparison 是分支前共同成本，不进入五条路径差；它们只进入 FE=0→terminal online policy wall-clock。三条 acquisition path 真实执行同一 query sample/evaluation；`query_joint` 还执行 feature 与 full Selector，`query_matched_state_only` 执行 state-only Selector，`sampling_only_continue_current` 原生继续当前算法；Behavior-only 执行其 fold-role Selector；Skip 执行 native continuation。replay plan 已写入 action不能替代真实 Selector inference。canonical path order 在三个 repetitions 间循环移位。原始 timing 表按 state、fold role、path、action、query 与 repetition 保存，且不接受缺少下列状态字段的旧表：

```text
path
repetition_index
order_position
runtime_future_path_raw_observed
timing_repetitions
timing_order_protocol
timing_source
timing_origin
timing_environment_id
thread_count
selected_algorithm
terminal_gap
observed_first_hit_FE
target_hit_observed
target_hit_before_failure
endpoint_success
planned_FE
effective_FE
timed_out
path_completed
timing_replay_status  # completed | timed_out | failed
timing_replay_timeout_seconds
runtime_query_sampling_raw_observed
runtime_query_evaluation_raw_observed
runtime_query_feature_computation_raw_observed
runtime_selection_raw_observed
runtime_handoff_raw_observed
runtime_optimization_raw_observed
```

`timing_replay_status` 是 completion/timeout 的权威字段，并逐 repetition 满足：

```text
target_hit_observed := observed_first_hit_FE != null
target_hit_before_failure := target_hit_observed and not path_completed
path_completed := timing_replay_status == completed
timed_out := timing_replay_status == timed_out
endpoint_success := target_hit_observed and path_completed
```

若暂时保留 `runtime_seconds`、`first_hit_FE` 或 `success`，它们只可分别作为 `runtime_future_path_raw_observed`、`observed_first_hit_FE`、`target_hit_observed` 的严格兼容别名。

Utility row 保存 Stage-A 来源、全部三次 Stage-B 状态、raw observed/censored 时间和分解的一致性字段：

```text
scientific_endpoint_source  # stage_a_selection_reference_outcome
timing_replay_status_protocol  # stage_b_completed_timed_out_failed_v1
timing_replay_status_repetitions_{skip,query_joint,query_matched_state_only,sampling_only_continue_current,behavior_only_full_budget}
timing_replay_effective_FE_repetitions_{skip,query_joint,query_matched_state_only,sampling_only_continue_current,behavior_only_full_budget}
timing_replay_timed_out_flags_{skip,query_joint,query_matched_state_only,sampling_only_continue_current,behavior_only_full_budget}
timing_replay_path_completed_flags_{skip,query_joint,query_matched_state_only,sampling_only_continue_current,behavior_only_full_budget}
timing_replay_observed_first_hit_FE_repetitions_{skip,query_joint,query_matched_state_only,sampling_only_continue_current,behavior_only_full_budget}
timing_replay_target_hit_observed_flags_{skip,query_joint,query_matched_state_only,sampling_only_continue_current,behavior_only_full_budget}
timing_replay_target_hit_before_failure_flags_{skip,query_joint,query_matched_state_only,sampling_only_continue_current,behavior_only_full_budget}
timing_replay_endpoint_success_flags_{skip,query_joint,query_matched_state_only,sampling_only_continue_current,behavior_only_full_budget}
timing_replay_{completed,timeout,failure}_repetitions_{skip,query_joint,query_matched_state_only,sampling_only_continue_current,behavior_only_full_budget}
timing_replay_status_instability_{skip,query_joint,query_matched_state_only,sampling_only_continue_current,behavior_only_full_budget}
stage_a_stage_b_completion_status_instability_{skip,query_joint,query_matched_state_only,sampling_only_continue_current,behavior_only_full_budget}
timing_replay_path_identity_consistent_{skip,query_joint,query_matched_state_only,sampling_only_continue_current,behavior_only_full_budget}
completed_timing_replay_outcomes_internally_consistent_{skip,query_joint,query_matched_state_only,sampling_only_continue_current,behavior_only_full_budget}
stage_a_to_completed_timing_replays_consistent_{skip,query_joint,query_matched_state_only,sampling_only_continue_current,behavior_only_full_budget}
runtime_{skip,query_joint,query_matched_state_only,sampling_only_continue_current,behavior_only_full_budget}_raw_observed_repetitions
runtime_{skip,query_joint,query_matched_state_only,sampling_only_continue_current,behavior_only_full_budget}_raw_observed_median
runtime_{skip,query_joint,query_matched_state_only,sampling_only_continue_current,behavior_only_full_budget}_censored_repetitions
runtime_{skip,query_joint,query_matched_state_only,sampling_only_continue_current,behavior_only_full_budget}_median
time_cost_log10_ratio
behavior_time_cost_log10_ratio
query_vs_behavior_time_cost_log10_ratio
```

可另存跨五条 path 的 `timing_replay_status_instability` 与 `stage_a_stage_b_completion_status_instability` 汇总标志，但它们只是各 path 标志的 OR，不替代逐 path 字段。旧 `runtime_*_failure_worst_case_repetitions/median` 只可分别作为 `runtime_*_censored_repetitions`/`runtime_*_median` 的严格兼容别名；旧 `timing_replay_instability_*` 只可作为 `timing_replay_status_instability_*` 的兼容别名。旧单一 `completed_timing_replay_outcomes_consistent` 退出活动契约，因为它混合了不同适用条件。

逐行关系为：

\[
r_{q-s}=\log_{10}T_q-\log_{10}T_{skip},\quad
r_{b-s}=\log_{10}T_b-\log_{10}T_{skip},\quad
r_{q-b}=r_{q-s}-r_{b-s}.
\]

每个 raw observed future-path 时间均须为有限正数并原样保留；非正或非有限值是运行失败，不能常数填充。对 path $k$、repetition $j$：

\[
T^{censored}_{k,j}=
\begin{cases}
T^{raw}_{k,j}, & status_{k,j}=completed,\\
\max(T^{raw}_{k,j},\ timeout_k), & status_{k,j}\in\{timed\_out,failed\}.
\end{cases}
\]

`runtime_*_raw_observed_median` 是三次 raw observed 时间的中位数，只作诊断；主 `runtime_*_median` 是三次 censored 时间的中位数，五条路径的 $T_k$ 与所有主 Utility 只读后者。不得让快速失败降低主时间成本。

三类一致性分别定义：

1. `timing_replay_path_identity_consistent_{path}` 检查三次及 Stage-A plan 的 selected action/algorithm、planned FE、query/action budget 与路径身份；
2. `completed_timing_replay_outcomes_internally_consistent_{path}` 只比较 Stage-B completed repetitions 的 terminal gap、`observed_first_hit_FE` 与 `target_hit_observed`；少于两个 completed 时为 null/not_evaluable；
3. `stage_a_to_completed_timing_replays_consistent_{path}` 仅在 Stage-A completed 且至少一个 Stage-B completed 时比较相同科学量，否则为 null/not_evaluable。

`timing_replay_status_instability_{path}` 只表示 Stage-B 三次 status 混合；`stage_a_stage_b_completion_status_instability_{path}` 独立表示 Stage-A completion 与 Stage-B completion pattern 不一致。timed-out/failed replay 的 observed hit、completion、endpoint success 与 effective FE 只作诊断，不覆盖 Stage-A 字段。不得挑选或补跑 repetition。CPU time、已有批量 prediction 秒数或复制的 problem-level query 时间都不是活动计时。

FE=0 online policy 使用独立字段 `runtime_full_run_wall_clock_raw_observed_repetitions`、`runtime_full_run_wall_clock_raw_observed_median`、`runtime_full_run_wall_clock_censored_repetitions` 与作为 censored median 的主字段 `runtime_full_run_wall_clock_median`；censoring 使用 `policy_timeout_seconds`。这些字段不得替代状态条件 $T_k$，状态条件字段也不得冒充完整政策时间。

## 8. 联合 Utility、Behavior-only 与五路径分解

下式中的 $T_k$ 固定为相应路径三次 censored repetitions 的 `runtime_*_median`；不得改用 raw observed median。

对 `lambda_time in {0,0.25,0.5,1,2}`：

\[
U_q^{joint}=(\ell_{skip}-\ell_q)-\lambda_T r_{q-s},
\]

\[
U_b=(\ell_{skip}-\ell_b)-\lambda_T r_{b-s},
\]

\[
I_q=(\ell_b-\ell_q)-\lambda_T r_{q-b}=U_q^{joint}-U_b.
\]

字段组固定为：

```text
u_query_joint_lamT_{0,025,05,1,2}
u_behavior_only_full_budget_lamT_{0,025,05,1,2}
query_operational_increment_lamT_{0,025,05,1,2}
u_query_matched_state_only_lamT_{0,025,05,1,2}
u_sampling_only_continue_current_lamT_{0,025,05,1,2}
query_descriptor_use_increment_lamT_{0,025,05,1,2}
query_state_only_vs_sampling_increment_lamT_{0,025,05,1,2}
query_sampling_direct_increment_lamT_{0,025,05,1,2}
need_query_joint_lamT_{0,025,05,1,2}
need_behavior_only_full_budget_lamT_{0,025,05,1,2}
```

布尔标签分别由对应 Utility 是否大于 0 得到。主 Decision target 为 `u_query_joint_lamT_1`；self-thresholded Behavior-only target 为 `u_behavior_only_full_budget_lamT_1`。`query_operational_increment_lamT_1` 必须在全 eligible-state summary 与 Proposed 同一 first-trigger state summary 中分别报告；它包含 acquisition、sample-best、预算与 Selector 差异，不用于把 query 描述为纯信息效应或因果 intervention。

对每个 lambda 还必须逐行满足：

```text
query_descriptor_use_increment = u_query_joint - u_query_matched_state_only
query_state_only_vs_sampling_increment =
    u_query_matched_state_only - u_sampling_only_continue_current
query_sampling_direct_increment = u_sampling_only_continue_current
u_query_joint = query_descriptor_use_increment
              + query_state_only_vs_sampling_increment
              + query_sampling_direct_increment
```

该分解条件于固定模型、query realization、预算和 transition rule，不作因果解释。

主 `lambda_time=1, lambda_memory=0` 表示 gap 与 runtime 的十进制数量级变化等权；memory 单独报告。其他 lambda 完整输出为 sensitivity，不得根据结果改选。

## 9. Fold 与 first-trigger 字段

端到端证据须标明：

```text
outer_fold
inner_fold
sbs_fit_scope
selector_fit_scope
selector_prediction_source
decision_fit_scope
threshold_fit_scope
threshold_mode
decision_opportunity_index
policy_first_trigger
```

outer holdout 只读 outer-fit SBS/Selector/Decision/threshold；inner holdout 只读 inner-fit 上游组件。完整 BBOB-train OOF threshold 也必须来自 fold-specific 上游链。state-level Utility 只作 score diagnostics；policy outcome 每 run 最多一个 first-trigger state。

## 10. 失败与拒绝读取条件

BBOB train/validation 与 CEC2017 的单 state-action path timeout 为 `3600 s`，`observed_first_hit_FE` 在每次 objective evaluation 更新。timeout 计失败并保留；`target_hit_before_failure=true` 的路径仍按 `target_hit_observed` 进入正式 ERT，未命中项计完整预算。CEC2022 必须先实现 benchmark factory/config，工程问题必须先实现 problem factory/constraint/config；二者还须冻结 failure cap、gap floor/cap、success target、timeout、observed-first-hit 和 constraint rule。

以下任一情况阻止活动读取或模型拟合：

- query/sampling/状态键字段缺失或不一致；
- 两套 action budget 混用或缺少任一动作；
- 三次 future-path 计时缺失、order 不符合循环协议、raw time 非正、censoring 不可重算，或与 FE=0 policy wall-clock 混列；
- observed hit/completion/endpoint success 关系不成立，三类一致性被压成一个布尔量，或两类 instability 未分开；
- fold-role Selector artifact 路由缺失，导致 replay 未真实执行相应 Selector inference；
- 三条 Utility 不能逐行重算或不满足 `I_q=U_q^{joint}-U_b`；
- `FE_ratio != FE/FE_total`；
- fold-specific SBS/Selector/Utility 来源不完整；
- 失败行被删除，或 failure/cap/timeout 不可区分；
- 读取旧 `u_query_lamT_*`、`performance_gain_norm` 或 `time_cost_norm` 作为活动 target。

Decision X 只允许冻结的 permutation-invariant Behavior 字段；本文件中的 gap、runtime、Utility、query、function、dimension、algorithm 与 transition 字段均不得进入 Decision X。
