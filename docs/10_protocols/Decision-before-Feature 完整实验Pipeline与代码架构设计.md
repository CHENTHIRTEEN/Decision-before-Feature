# Decision-before-Feature 完整实验 Pipeline 与代码架构

> 唯一活动依赖顺序（2026-08-15）。正式执行直接调用项目 CLI/Python 模块；不得用 wrapper shell 调用多个脚本，也不得让脚本递归调用脚本。本文只定义输入输出依赖，不表示正式实验已经运行。

## 1. 科学流程

```text
benchmark configuration
-> complete native optimizer trajectories + complete-budget endpoints
-> permutation-invariant Behavior
-> fixed query samples/features
-> query-adjusted and full-budget state-action outcomes
-> fold-specific SBS + Selectors
-> persisted fold-role Selector artifacts + selected replay plan
-> three-repeat decision-state-to-terminal raw/censored timing
-> log-gap/log-runtime Utility labels
-> fold-specific Decision models + first-trigger thresholds
-> baselines and frozen external evaluation
-> hierarchical inference and manuscript tables
```

正式模型选择是端到端 nested function evaluation，不是从一张 full-train label 表开始的 Decision-only cross-validation。

## 2. 配置与 suite

活动配置只有：

```text
configs/phase1_bbob_train.yaml
configs/phase1_bbob_validation.yaml
configs/phase1_cec2017_test.yaml
```

BBOB train 18 functions，validation 6 functions；二者为 function split，维度 10/20/40D、3 instances、30 seeds。CEC2017 为 29 functions、10/30/50D、30 seeds。总预算均为 `1000D`，population size 40。

三个配置共同冻结：

```text
failure_loss_cap = 1e20
log10_gap_floor = 1e-12
log10_gap_cap = 1e20
success_gap_target = 1e-8
action_timeout_seconds = 3600
first_hit_recording = every_objective_evaluation
timing_repetitions = 3
timing_order_protocol = cyclic_complete_path_v1
```

CEC2022 当前还缺 benchmark factory 与正式配置；工程问题还缺 problem factory、约束处理与正式配置。二者必须先实现并冻结 functions/problems、dimension、budget、repeats、reference、gap floor/cap、success target、timeout、observed-first-hit 与 constraint rule，才可进入正式 pipeline。

## 3. Trajectory 与完整预算 endpoint

模块/CLI：

```text
experiments.phase1_plan_shards        / phase1-plan-shards
experiments.phase1_collect_batch      / phase1-collect-batch
experiments.phase1_check_trajectory_shards / phase1-check-trajectory-shards
experiments.optimizer_state_consistency    / optimizer-state-check
```

每个 `problem_id × algorithm × seed` 生成 native complete-state trajectory。decision states 只在 `phase1_dynamic_budget_event_v1` emitted complete updates 上保存，跨表键使用整数 FE。wall-clock truncation 只保留最后完整 native update，不能伪造 `FE_total` decision state。

完整预算 endpoint 单独写 `final_performance.parquet`，每 run 在 `FE=FE_total` 恰好一行，保存：

```text
benchmark_reference_value
final_gap
log10_gap
log10_gap_floor
log10_gap_cap
success_gap_target
observed_first_hit_FE
target_hit_observed
target_hit_before_failure
path_completed
endpoint_success
performance_protocol = complete_budget_native_optimizer_run_with_first_hit_endpoints
```

`target_hit_observed := observed_first_hit_FE != null`；`target_hit_before_failure := target_hit_observed and not path_completed`；`path_completed := status == completed`；`endpoint_success := target_hit_observed and path_completed`。兼容 `first_hit_FE/success` 只可分别别名为 `observed_first_hit_FE/target_hit_observed`，正式 ERT 使用 `target_hit_observed`。first hit 在每次 objective evaluation 更新。Trajectory/Behavior 本身不保存 benchmark reference、gap、success label 或其他可能进入 Decision 的 outcome 字段。

## 4. Behavior

模块/CLI：

```text
behavior.extraction / behavior-extract
behavior.batch_extraction / behavior-extract-batch
experiments.behavior_permutation_consistency / behavior-permutation-check
```

输入是 emitted trajectory 与逐完整 native-update window summaries；输出是 34 个唯一 Behavior 字段、9 个实际窗口 metadata。w02/w05/w10 anchor 取不晚于名义位置的最近完整 update，实际窗口不少于名义跨度且偏差小于一次 population update。Decision 输入只用 31 个正式 `bf_*` 字段。

## 5. Query samples 与 features

模块/CLI：

```text
landscape_queries.batch_sampling / query-sample-batch
landscape_queries.batch_features / query-extract-cheap
tools/pflacco_query/extract.py    (隔离 Python 3.11/pflacco 1.2.2 进程)
landscape_queries.consistency     / query-consistency
```

活动 query IDs：

```text
descriptor_cheap_invariant       lhs_50d, 14 columns, primary
pflacco_standard_invariant       lhs_50d, 37 columns, sensitivity
pflacco_broad_invariant          lhs_100d, 52 columns, sensitivity
```

cheap/standard 共用相同 LHS `(X,y)`；broad 使用独立 sample。sample/feature 表保存 query protocol、sample design、FE/runtime、失败字段和 `query_first_hit_offset`。该 offset 用于把 query 内首次达到 success target 的评价位置映射到全路径 FE；它是离线 outcome metadata，不进入 Behavior、Selector input 或 Decision X。

## 6. State-action outcomes 与三次计时

模块/CLI：

```text
selection_reference.action_losses / selection-reference-evaluate-actions
experiments.selection_reference_consistency / selection-reference-check
```

每个 shared state 在 Stage A 生成 Query-adjusted 与 Behavior-only full-budget 两套四动作矩阵，每个动作只运行一次。该预指定 outcome 保留 continuation-only raw loss、gap、`observed_first_hit_FE`、`target_hit_observed`、completion、endpoint success、planned/effective FE、failure/timeout、transition 与完整 optimizer outcome，并作为科学端点唯一来源。Query sample 不进入 optimizer population，但主 operational Query Stage-A terminal best、observed hit/ERT 合并 sample 与 selected continuation；另保存 continuation-only gap 与 sample-best contribution。

Selector 冻结并按 fold-role/family 持久化后，Stage B 才对 selected Skip、Query 与 Behavior-only 从同一复制 state/RNG 到 terminal 真实 replay 预定三次，且只决定 timing。状态条件计时从 query/no-query 分支确定后开始；共享 prefix、Behavior extraction、Decision inference 与 threshold comparison 是分支前共同成本，只进入 FE=0→terminal policy wall-clock。Query/Behavior-only 必须按 artifact 路由真实执行相应 Selector inference，不能只读取 replay plan 的 selected action。

canonical path order 按 `cyclic_complete_path_v1` 循环移位；逐次保存 repetition/order、raw observed 组件/完整 future-path wall-clock、status、observed hit、effective FE、timeout/completion。completed repetition 的 censored time=raw，timed_out/failed repetition 的 censored time=`max(raw, role timeout)`；主 `runtime_*_median` 是三次 censored time 的中位数，raw observed median只作诊断。

每条 path 分别保存 `timing_replay_path_identity_consistent_*`、`completed_timing_replay_outcomes_internally_consistent_*`、`stage_a_to_completed_timing_replays_consistent_*`；后两者不适用时为 null/not_evaluable。`timing_replay_status_instability_*` 与 `stage_a_stage_b_completion_status_instability_*` 独立保存。不得用单一布尔量混合三类一致性，不得选择性补跑。Stage-B endpoint 只作一致性/失败诊断，不覆盖 Stage-A 字段；FE=0→terminal online policy wall-clock 分列报告。不得把已有 prediction 秒数或一次计时复制到多行。

## 7. Fold-specific SBS 与 Selectors

模块/CLI：

```text
selection_reference.build / selection-reference-build
decision.nested_learning  (由模型训练/比较 CLI 直接调用)
```

每个 outer fold 只用 outer-fit functions 计算 `SBS_outer`、cross-fit/拟合 Query 和 Behavior-only Selectors。每个 inner fold 再只用 inner-fit functions 计算 `SBS_inner`、cross-fit/拟合 Selectors并生成 inner Decision labels。Selector 主 target 是相对 `continue_current` 的 `clipped_log10_gap_advantage_vs_continue_current`；旧 statewise min-max target 只作敏感性分析。Utility 使用 selected observed gap 与 state-to-terminal future-path 时间，不直接使用 Selector target。

当前实现尚未持久化 outer/inner `cv_group_id = function_id` cross-fitted Selector artifacts，也未在 replay plan 中提供历史 `fold role/cv_group_id -> Selector artifact` key 路由；offline decision-state-to-terminal runner 同样尚未实现。三项均是 Stage-B blocker，未关闭前不得生成最终 Utility。

Query-adjusted state-only 与 full Selector 在同一 action matrix 上的 OOF selected continuation-only `log10_gap` 差输出为 `query_feature_predictive_increment_log10_gap`，排除 query sample best，只作预测诊断。

## 8. Utility labels

模块/CLI：

```text
utility_labels.generation / utility-labels-generate
utility_labels.batch_generation / utility-labels-generate-batch
utility_labels.validation / utility-labels-validate
```

先在 raw gap 上应用 suite floor/cap，再取 `log10`；时间项只读三次 censored future-path time 的 `runtime_*_median`，raw observed median只作诊断。三类标签为：

```text
u_query_joint_lamT_*
u_behavior_only_full_budget_lamT_*
query_operational_increment_lamT_*
```

并逐行满足 `query_operational_increment = joint - behavior_only`。主 lambda=1、memory weight=0；其他 lambda 只作 sensitivity。旧 max-scale/relative-time Utility 不得读取。

## 9. Decision 与 threshold

模块/CLI：

```text
decision.train_full_decision_model / decision-train-full
decision.compare_feature_group_training / decision-compare-feature-groups
decision.check_model_protocol / decision-check-model-protocol
decision.threshold_sweep / decision-threshold-sweep
```

单一预物化 Utility 表无法表示 fold-specific SBS→Selector→Utility→Decision→threshold 链路，不是活动实验入口；退出提示模块不列入本活动 Pipeline。

活动候选仅 LDA、Logistic Regression、Ridge。B3 outer-function OOF run-level first-trigger mean joint Utility 选择模型名；完整 BBOB-train 端到端 OOF 冻结 `oof_utility_first_trigger` 与 Behavior-only threshold。BBOB-validation 已被历史开发查看，只可作为 post-development 内部评价；后续仍禁止它与 external suites 参与 preprocessing、选择或 threshold，但这不能恢复其 untouched/confirmatory 资格。

Decision 训练产物按 `feature_group_ablation/{feature_group}/{opportunity_scope}/` 显式隔离。B3 模型家族只由 `B3/all_accepted` 的 nested OOF 选择；RQ2 与正式特征组消融统一读取六个 `{feature_group}/milestone_only` 目录。`decision-compare-feature-groups` 必须同时核对 train OOF 与 BBOB-validation 的整数 state keys、FE ratio 和 sampling metadata，并输出 `rq2_milestone_b3_minus_t0_run_rows.parquet`。该表中的 BBOB-train OOF 行只作开发诊断，BBOB-validation 行只作已见内部评价，不承担确认性证据。

## 10. Baselines 与外部评价

模块/CLI：

```text
decision.compare_controller_baselines / decision-compare-controller-baselines
decision.online_controller_evaluate / decision-online-controller-evaluate
experiments.hierarchical_inference  # internal analysis library, not a CLI
```

baseline outcome 包括 Never/SBS、Always Query、matched-rate Random、FE=0 pre-run AAS、milestone-only T0、self-thresholded Behavior-only、Proposed。VBS 在每个 function × instance × dimension problem 内先跨 seeds 比较四算法完整预算 mean clipped `log10_gap`，再汇总所选算法的 paired seed outcomes；不逐 seed 选最小，且与逐状态 best observed action 是不同 hindsight references。所有顺序策略最多一次 first trigger。

`decision-compare-controller-baselines` 是整政策比较：动态 Proposed、Always、Random 与 Behavior-only 使用全部 accepted opportunities，`milestone_only_T0` 只使用 12 个 milestones。其 Proposed--T0 差异包含调度、特征、拟合分数与 threshold 差异，不解释为 RQ2 的 Behavior 增量；RQ2 只读上一节严格 milestone-matched 的 B3--T0 配对表。

Matched Random 只读 BBOB-train OOF Proposed 的 run call rate 与 trigger-FE 经验分布；每 run 预抽是否调用及目标 FE，在首个不早于目标的在线合格机会触发。

online policy 科学行统一保存 `scientific_endpoint_source=stage_a_online_policy_outcome`；不再另设含义重叠的 `scientific_endpoint_stage`。其 `observed_first_hit_FE/target_hit_observed/path_completed/endpoint_success` 与 Stage-B full-run timing repetitions 使用第 3、6 节相同语义。FE=0 full-run 同时保存 `runtime_full_run_wall_clock_raw_observed_repetitions`、`runtime_full_run_wall_clock_raw_observed_median`、`runtime_full_run_wall_clock_censored_repetitions` 与作为 censored median 的主字段 `runtime_full_run_wall_clock_median`。

当前 online evaluator 只支持 main cheap CEC 路径、固定 `instance=1`。因此已见 BBOB-validation 内部评价集的三 instances、standard/broad 各自完整 full-policy、CEC2022 benchmark factory/config 与工程问题 factory/constraint/config 均是 blocker；不得把 main-only 或部分 suite 输出扩写为三档/跨 suite 结论。

## 11. 运行前数据质量检查

正式矩阵前必须确认：

1. 72 个 BBOB train/validation trajectory 与 final endpoint shards 成对完整；
2. state/reservoir snapshot 只对应 emitted integer FE，timeout 不产生部分 update state；
3. Behavior 不含 reference/gap/outcome，窗口来自完整 native updates；
4. 三个 query ID、sample design、feature whitelist、query first-hit offset 与 FE charge 一致；
5. 两套 action budgets 语义唯一，允许复用的 Skip/5% outcome 只计算一次；
6. outer/inner SBS、Selectors、Utility、Decision 与 thresholds 均只读 fit functions；
7. Proposed、T0、Always、Random、Behavior-only 与汇总均为 first-trigger；
8. 三次计时、循环 order、机器、线程、raw/censored runtime、observed hit/completion/endpoint success、三类一致性与两类 instability 齐全；
9. gap floor/cap、`target_hit_observed`、`endpoint_success`、timeout、failure 与 ERT 可逐行重算，ERT 明确使用 `target_hit_observed`；
10. Stage-B runner、fold-role Selector artifact 路由、BBOB instance-aware online、standard/broad full-policy 均已闭合；BBOB-validation 必须直接按 validation config 核对 F5/F9/F13/F14/F19/F24、instances 1/2/3、10/20/40D、seeds 1--30 与完整 problem/state coverage，不能只从实际输入推断 families；
11. CEC2022 factory/config 与工程问题 factory/constraint/config 完整前不运行，活动路径不读取撤回产物。

任一项失败都先修复相应依赖并重生成，不通过删除行、复制 outcome 或更改统计范围处理。

## 12. 结果路径与隔离

```text
results/phase1_refined_sampling/{split}/
results/landscape_queries/{samples,features}/
results/selection_reference/{query_id}/
results/utility_labels/{query_id}/
results/decision/{query_id}/
```

不同 query/split/fold 的表必须由字段核对后连接。旧 `results/ela/`、重建式 trajectory、静态 bucket Selector、旧 Utility、逐状态 policy 与一次计时结果均不属于活动输入。
