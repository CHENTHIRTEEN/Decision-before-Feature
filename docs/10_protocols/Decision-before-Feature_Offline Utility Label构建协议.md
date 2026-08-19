# Decision-before-Feature Offline Utility Label 构建协议

> 唯一活动协议（2026-08-16，方案 A + 最小 Action Loss 规范 v1 对齐修订）。本文件直接维护联合策略 estimand、Behavior-only 对照和 first-trigger 数据契约；每条 action 记录必须同时保留行标识、科学端点、censored runtime 和一个 canonical loss（`action_loss`）。这里的 canonical loss 是严格等总 FE 预算下的 FE-indexed optimization loss，不得用 wall-clock time 定义科学标签。旧单一 `G_FE`、一次性完整-train Selector 标签和逐状态 policy 汇总不再使用。

## 1. 研究问题与标签边界

主问题是：在相同完整 optimizer state 和等总 FE 下，执行固定 query 并调用 full Selector 的联合路径，相对 native fold-specific SBS continuation 是否具有净效用？

该 joint estimand 不能识别 query descriptors 的独立信息价值。为此必须同时生成不执行 query、保留完整剩余预算的 Behavior-only 四动作路径，并在 Proposed 的同一 first-trigger state 上计算 Query 相对 Behavior-only 的操作性净增量。三种标签是预测性、模型条件的比较，不作因果主张。

## 2. 主分析 population

第一篇论文主行满足：

```text
prefix_algorithm == default_algorithm == fold-specific SBS
skip_switches_from_prefix == false
no_query_transition_mode == native_continuation
```

SBS 在每个 outer fold 只由 outer-fit functions 的 `FE=FE_total` outcomes 计算。完整 BBOB-train SBS 仅用于 final refit、validation 和 external deployment。全 prefix rows 单独用于 cross-prefix robustness，不混入主 Decision fitting、threshold 或 RQ1/RQ3 汇总。

所有路径从同一个 complete checkpoint state 开始，包括 population、fitness、best-so-far、generation/native update、optimizer-specific dynamics 和 RNG state。同算法动作原生 continuation；跨算法动作只转移 population、fitness 和 best-so-far，并执行一次 `population_transfer_initialization`。

## 3. 五条路径与两套动作矩阵

令总预算为 `B`、prefix 已用 FE 为 `e_t`、固定 query FE 为 `FE_q`。

### 3.1 Skip

Skip 原生继续 SBS state，优化预算为：

\[
B_t^{skip}=B-e_t.
\]

终端 loss 记为 `P_skip`。

### 3.2 Query/full-Selector

query sample 使用显式整数 SeedSequence stream，目标评价计入总 FE；sample 不并入 optimizer population。四个唯一动作是 `continue_current` 加其余三个 portfolio algorithms，每个动作使用：

\[
B_t^q=B-e_t-FE_q.
\]

Query Selector 输入为 31 维 B3 Behavior、当前 query descriptors 和 `B_t^q/B`，预测基于 continuation outcomes、相对 `continue_current` 的 `clipped_log10_gap_advantage_vs_continue_current`。所选动作的 continuation-only observed loss 记为 `p_q_cont`。query sample 不进入 optimizer population，但它是真实已观察的 objective evaluations；主 operational Query path terminal best `p_q` 必须取 prefix/query sample/selected continuation 中的最佳已观察值，并把 query 内 first hit 映射到全路径 FE。另保存 continuation-only gap 与 query-sample-best contribution。

cheap 与 standard 共用同一 `lhs_50d` sample 和 5% action outcome matrix，但使用不同 descriptor columns 与独立 Selector；broad 使用独立 `lhs_100d` sample、10% matrix 与独立 Selector。跨 query 可复用语义完全相同的 Skip；不得复用预算不同的 action loss。

### 3.3 Query-matched state-only

执行与 `query_joint` 完全相同的 query realization，使用同一 sample endpoint、$B_t^q$、Query-adjusted 四动作 outcome 和计时起点，但由 `query_adjusted_state_only_selector`（B3 Behavior + query-adjusted remaining-budget ratio，不含 descriptors）选动作。该路径识别固定 acquisition 下实际使用 descriptors 的操作性增量，不新增 action losses。

### 3.4 Sampling-only continue-current

执行同一 query acquisition 并保留同一 sample endpoint，随后从完整 state 原生继续当前算法，continuation 预算同为 $B_t^q$。它复用 Query-adjusted matrix 的 `continue_current` outcome，不新增 action losses。

### 3.5 Behavior-only full-budget

同一四动作集合使用：

\[
B_t^b=B-e_t.
\]

该路径不生成、读取或计费 query sample/features。Behavior-only Selector 只输入 B3 Behavior 与 `B_t^b/B`，选中动作的 observed loss 记为 `p_b`。full-budget `continue_current` 与主 population Skip 语义相同，只生成一次；其余三个 full-budget actions 是新增 action losses。

`matched_trigger_behavior_only` 在 Proposed 的同一首次触发 state 使用该 Selector；`self_thresholded_behavior_only` 另由预测 `U_b` 的同名 Decision model 和自身 first-trigger threshold 决定何时使用该 Selector。两者共用 action data，不能共用 threshold 或政策标签。

## 4. Action target 与诊断

每套矩阵使用 suite 预先固定的 $g_{\min},g_{\max}$，并以 `continue_current` 为参照。主 target 为：

\[
Y_a=\log_{10}(\operatorname{clip}(L_a,g_{\min},g_{\max}))
-\log_{10}(\operatorname{clip}(L_{continue},g_{\min},g_{\max})).
\]

`continue_current` 的 target 恒为 0，产物保存 `selector_target_transform=clipped_log10_gap_advantage_vs_continue_current`。旧 statewise min-max transform 只作预设 Selector target sensitivity，不生成主 selected action。逐状态最小 loss 只称 `best observed action`。必须同时保存 raw action range、near-tie indicator、selected observed loss、potential performance difference 和 Selector regret。

除 full Query Selector 与主 full-budget Behavior-only Selector 外，同一 Query-adjusted matrix 还拟合第三个 `query_adjusted_state_only_selector`（Behavior + query-adjusted budget，不含 descriptors）。它与 full Query Selector 读取完全相同的四动作 outcomes，不新增 action losses。两者的 OOF selected continuation-only `log10_gap` 差保存为 `query_feature_predictive_increment_log10_gap`。该诊断只称 query features 的 OOF 边际预测贡献；它排除 query sample best，不是主策略效用、纯信息效应或因果效应。

动作关系字段固定保存：

```text
selected_equals_default
selected_equals_prefix
handoff_required
handoff_type
skip_switches_from_prefix
no_query_algorithm
```

其中 `handoff_required = not selected_equals_prefix = (handoff_type == population_transfer_initialization)`，逐行一致；不生成 `label_source` 或 `same_algorithm/changed_algorithm` 模糊别名。

## 5. 性能与时间定义

预指定 Stage-A 对每条路径执行一次科学运行，并由该行唯一固定 terminal gap、`observed_first_hit_FE`、`target_hit_observed`、`path_completed`、`endpoint_success`、planned FE、effective FE、ERT contribution 与科学失败状态。冻结语义为：

```text
target_hit_observed := observed_first_hit_FE != null
target_hit_before_failure := target_hit_observed and not path_completed
path_completed := status == completed
endpoint_success := target_hit_observed and path_completed
```

`target_hit_before_failure` 只作诊断；正式 ERT 使用 `target_hit_observed`，不用 `endpoint_success`。兼容字段 `first_hit_FE`/`success` 若保留，必须分别严格别名为 `observed_first_hit_FE`/`target_hit_observed`。五条 Stage-A 路径先计算非负 benchmark-reference raw gap $(g_{skip},g_q,g_m,g_c,g_b)$。三条 acquisition 路径使用同一 query sample endpoint；其中 `g_q` 另保存只看 selected continuation outcome 的 `g_q_cont` 与 `query_sample_best_contribution_log10_gap = ell_q_cont - ell_q`。Selector regret 和等预算 query-feature predictive diagnostic 只使用 continuation-only outcomes，不能把随机 query sample 的直接改进归因于 query features。BBOB train/validation 与 CEC2017 的配置在取对数前固定应用：

```text
log10_gap_floor = 1e-12
log10_gap_cap = 1e20
```

即：

\[
\ell_k=\log_{10}\!\left(\min(\max(g_k,10^{-12}),10^{20})\right),
\quad k\in\{skip,q,m,c,b\}.
\]

floor/cap 作用于 raw gap，不作用于 objective value 或已经取对数的数。`target_hit_observed` 由真实逐 evaluation first-hit 记录与 `success_gap_target=1e-8` 判定，不由最终截断 gap 倒推。Query operational path 的 observed hit/ERT 同时考虑 query sample evaluations 与后续 continuation；`query_first_hit_offset` 映射 query 内命中位置。continuation-only endpoint 另作诊断。known reference/gap 只用于离线标签和最终评价，不进入 Behavior、Selector inputs 或 Decision X。

状态条件 Utility 的计时起点是同一 decision state 上 query/no-query 分支已经确定之后。共享 optimizer prefix、该机会的 Behavior extraction、Decision inference 与 threshold comparison 在分支前已经发生，是五条路径的共同成本，不进入路径间 future-runtime ratio；这些共同成本只进入 FE=0→terminal 完整政策 wall-clock。

Query path 的动作特异增量包括 sample generation/evaluation、feature extraction、fold-role 对应 Query Selector inference、必要 handoff 与 continuation；Behavior-only 包含其 fold-role 对应 Selector inference、必要 handoff 与 continuation；Skip 只含实际 native continuation。不得把共同的 Decision inference 只向 Query/Behavior-only 收费，也不得因为 replay plan 已保存 selected action 而省略真实 Selector inference。

Stage A 的一次科学运行与其 Selection Reference outcome 固定五条路径的 terminal gap、observed hit、completion、endpoint success、planned/effective FE；Utility row 明确保存 `scientific_endpoint_source=stage_a_selection_reference_outcome`。Stage B 只估计 wall-clock：每条已选路径从同一 complete state 和相同 RNG state 到 terminal 真实 replay `timing_repetitions=3` 次，固定机器、线程与预加载常驻进程，canonical path order 按 `timing_order_protocol=cyclic_complete_path_v1` 循环移位。原始 timing 输入逐 repetition 显式保存：

```text
timing_replay_status  # completed | timed_out | failed
observed_first_hit_FE
target_hit_observed
target_hit_before_failure
path_completed
endpoint_success
effective_FE
timing_replay_timeout_seconds
runtime_*_raw_observed
```

每次 raw observed wall-clock 必须为有限正数并原样保留。对每条 path/repetition 定义 censored runtime：completed 时等于 raw；timed_out/failed 时等于 `max(raw, role timeout)`。Utility row 同时保存三次 raw/censored repetitions、`runtime_*_raw_observed_median` 与主 `runtime_*_median`；后者是 censored median，五条路径各自的 $T_k$ 只读该值。旧 `runtime_*_failure_worst_case_*` 仅可作为相同 censored 值的过渡别名。这样 timed-out/failed 的快速返回不会降低主时间成本。

每条路径分别保存三类一致性：

```text
timing_replay_path_identity_consistent_{path}
completed_timing_replay_outcomes_internally_consistent_{path}
stage_a_to_completed_timing_replays_consistent_{path}
```

第一类检查 selected action/algorithm、planned FE、query/action budget 与路径身份；第二类只比较 Stage-B completed outcomes，少于两个 completed 时为 null/not_evaluable；第三类只在 Stage-A completed 且至少一个 Stage-B completed 时比较 terminal gap、`observed_first_hit_FE` 与 `target_hit_observed`，否则为 null。另独立保存 `timing_replay_status_instability_{path}`（Stage-B 三次 status 是否混合）与 `stage_a_stage_b_completion_status_instability_{path}`（Stage-A completion 与 Stage-B completion 状态是否不一致）。不得再用一个 `completed_timing_replay_outcomes_consistent` 布尔量混合这些含义。

Stage-B 的 observed hit、completion 与 endpoint success只用于一致性和失败诊断，不覆盖 Stage-A 科学字段。FE=0→terminal policy wall-clock 另存。全部目标评价必须真实执行；不得只选完成项、补跑某次，或用 CPU time、批量 prediction 秒数、复制时间和常数替代真实路径时间。

## 6. 三个 Utility

\[
U_{query}^{joint}=(\ell_{skip}-\ell_q)
-\lambda_T(\log_{10}T_q-\log_{10}T_{skip}),
\]

\[
U_b=(\ell_{skip}-\ell_b)
-\lambda_T(\log_{10}T_b-\log_{10}T_{skip}),
\]

\[
query_operational_increment=(\ell_b-\ell_q)
-\lambda_T(\log_{10}T_q-\log_{10}T_b)
=U_{query}^{joint}-U_b.
\]

主字段为：

```text
G_FE
u_behavior_only_full_budget_lamT_1
query_operational_increment
```

主情景固定 `lambda_T=1, lambda_M=0`；`lambda_T=1` 表示 gap 与 runtime 的十进制数量级变化等权。`lambda_T={0,0.25,0.5,1,2}` 必须完整报告，不能依据任何正式结果改选；memory 是独立 endpoint。query FE 已通过缩短 `B_t^q` 进入 performance，不能再次扣除，population transfer 也已进入 observed action loss。

`query_operational_increment` 包含 query FE/runtime、sample best、较短 continuation budget 与两个 Selector 的差异，不是纯信息效应或因果 estimand。它必须在全 eligible states 与 Proposed first-trigger states 两个范围分别汇总；若同一范围 `G_FE_joint>0` 但 `query_operational_increment<=0`，只能支持联合路径优于 SBS。正式五路径还用 `query_matched_state_only` 与 `sampling_only_continue_current` 逐行计算 descriptor-use、state-only-vs-sampling 和 sampling-direct 三项增量；三项之和必须等于 `G_FE_joint`。该分解条件于固定模型、query realization、预算和 transition rule，不作因果解释。

Utility 是预设 scalarization，不代表普适资源偏好。主结果必须同时报告 `log10_gap` 与 `log10` wall-clock ratio；两者方向冲突时写明 trade-off。所有基于 raw-gap max-scale、线性相对 runtime 或单次计时的Utility 数值全部失效，即使字段名相同也不得复用。

## 7. Outer-fold-specific 标签生成

每个 Decision outer fold 必须：

1. 仅用 outer-fit functions 计算 `SBS_outer`；
2. 在 outer-fit functions 内 cross-fit Query Selector 与 Behavior-only Selector；
3. 用 cross-fit selected actions对应的 Stage-A `log10_gap` 和 Stage-B 三次 future-path wall-clock 中位数生成 outer-fit Decision labels；
4. 每个 inner fold 只用 inner-fit functions 重新计算 `SBS_inner`、cross-fit/拟合两类 Selector并生成 inner-fit labels；inner holdout labels、score 与 Utility 只能由 inner-fit 上游组件产生；
5. 用端到端 inner OOF first-trigger outcomes 冻结 outer thresholds；
6. 用 outer-fit 全量 Selector 生成 outer holdout labels，所有上游组件冻结后才评价 outer holdout。

outer 或 inner holdout 不得影响其评价链中的 SBS、Selector、Utility/cost processing、Decision、threshold、Random calibration 或 score-neighborhood。完整 BBOB-train OOF 也必须执行 fold-specific 上游链；一张由 full-train Selector 预制的 labels 表不能替代 outer/inner 链。

## 8. First-trigger policy label

state-level Utility 保留用于 score diagnostics；正式 policy label 按 run 构造。按整数 `FE` 排序，threshold 第一次越过时选定一个 state；未越过贡献 0，越过只取该 state Utility，后续 states 不可达。

policy-level query operational increment 必须在 Proposed 的同一首次触发 state 比较 `p_q` 与 `p_b`；state-distribution summary 则在全部 eligible states 计算同一定义。self-thresholded Behavior-only policy 从自己的 `U_b` score/threshold 产生独立 first trigger；不得把两个触发时点直接相减并称为 query operational increment。

## 9. 失败与 coverage

所有计划 rows 进入 coverage denominator。score 非有限的机会按 Skip，仍可在下一可达 state 检查。query 已触发后 feature/Selector failure 保留 query FE 和时间，fallback 为 query-adjusted `continue_current`。

BBOB train/validation 与 CEC2017 固定 `failure_loss_cap=1e20`、raw-gap floor/cap `1e-12/1e20`、`success_gap_target=1e-8`、单 state-action path timeout `3600 s`，并在 Stage-A 每次 objective evaluation 更新 `observed_first_hit_FE`。Stage-A timeout 计科学失败且保留；已经命中但随后失败的路径保存 `target_hit_before_failure=true`，正式 ERT 仍按 `target_hit_observed` 计命中，未命中项计该路径完整 planned budget。Stage-B timeout/completion 只进入逐次 timing 状态、censoring、一致性、instability 与 failure sensitivity，不改写 Stage-A 科学 endpoint 或 FE。新 suite 必须先具备可执行 benchmark/problem factory 与约束处理，再固定这些字段及 constraint rule。报告必须同时包含 attempted denominator/coverage、科学路径 failure、timing replay failure/timeout/instability、ERT、complete-pair 与双向极端 sensitivity；未闭合 pair 按 gap floor/cap、`target_hit_observed` 1/0、未命中 ERT 项的完整 planned budget、runtime complete-pair 最小正值/timeout 赋值并重算 Utility，不能用 cap 掩盖失败。

缺失 shared-state key、缺一套 action matrix 或 paired outcome 不完整属于数据生成错误，不做统计插补，修复后重生成对应 shard。

## 10. 产物与失效范围

正式顺序：

```text
complete trajectory + final outcome
-> Behavior + query samples/features
-> Stage-A query-adjusted and full-budget action matrices once
   (freeze scientific gap/observed-hit/completion/endpoint-success/planned/effective FE/failure)
-> persist outer/inner/full-train Selectors and fold-role/family artifact routing
-> fold-role selected replay plan
-> Stage-B three-repeat decision-state-to-terminal timing only
   (save per-repeat raw/censored time, status, observed hit, effective FE,
    timeout/completion, three consistency classes and two instability classes)
-> Stage-A log-gap + Stage-B censored-median log-runtime three Utility labels
-> fold-specific upstream Decision inner first-trigger thresholds
-> outer holdout policy outcomes
```

当前 offline decision-state-to-terminal runner、`cv_group_id = function_id` cross-fitted Selector artifact 持久化与路由尚未实现；在三者闭合且 replay plan 物化核对前，Stage-B 与最终 Utility 均不可执行。

旧单一 `G_FE`、旧 max-scale/relative-time Utility、静态 problem/budget bucket Selection Reference、重建式 continuation、一次计时复制、非嵌套 labels 和逐状态 policy summary 无正式证据资格。基础 trajectory 仅在 trajectory/snapshot/seed/time-truncation 检查通过后可作为新链输入。


## 11. Field manifest

本节吸收 `phase1_utility_label_column_spec` 的核心字段约定，作为当前活动字段清单：

- 文件作用域：`results/utility_labels/{query_id}/`，每张表只对应一个活动 query。
- 状态键：`(split, problem_id, family, dimension, prefix_algorithm, seed, FE)`，其中 `FE` 必须是实际整数评价数。
- Query metadata：`query_id`、`query_protocol`、`query_preprocessing_id`、`query_feature_columns`、`sample_design_id`、`FE_query`。
- FE 账本：`FE_total`、`FE_prefix`、`FE_query`、`FE_skip_optimization`、`FE_query_optimization`、`FE_behavior_only_optimization`。
- Outcome 诊断：`p_skip`、`p_query`、`p_query_continuation_only`、`p_behavior_only`、selected / best observed action 与 regret / range 字段。
- Gap / timing：五条路径的 raw / clipped / log10 gap、三次 raw / censored runtime 及其 median。
- 一致性：`target_hit_observed`、`target_hit_before_failure`、`path_completed`、`endpoint_success`、三类 consistency 与两类 instability。
- 兼容字段：`first_hit_FE` / `success` 只可作为严格别名；旧 `runtime_ratio`、`performance_gain_norm`、`time_cost_norm` 等均为历史字段。
- 旧 `phase1_utility_label_column_spec` 仅保留为历史来源，不再作为独立活动协议。
