# Decision-before-Feature 完整实验 Pipeline 与代码架构

> 唯一活动依赖顺序。本文是 `docs/10_protocols/` 的流程总纲，汇总数据生成、trajectory、query、behavior、state-action outcome、timing replay 与验证顺序；其余协议只负责更细的标签、模型、baseline 或字段细节。

## 1. 协议索引

- `Decision-before-Feature_最小ActionLoss字段规范.md`：`action_loss` 与 `G_FE` 的最小字段规范。
- `Decision-before-Feature_Offline Utility Label构建协议.md`：五路径 utility / endpoint / timing 的构建协议。
- `Decision-before-Feature Decision Model设计与训练协议.md`：Decision Model、threshold、nested OOF 与训练口径。
- `Decision-before-Feature Baseline与公平比较协议.md`：SBS / VBS / Random / Always Query / pre-run AAS / matched-acquisition baselines。
- `Decision-before-Feature Behavior Feature Taxonomy与指标选择协议.md`：Behavior 特征组、窗口与指标选择。

## 2. 总体科学流程

```text
benchmark configuration
-> complete native optimizer trajectories + complete-budget endpoints
-> permutation-invariant Behavior
-> fixed query samples/features
-> query-adjusted and full-budget state-action outcomes
-> fold-specific SBS + Selectors
-> persisted fold-role Selector artifacts + selected replay plan
-> three-repeat decision-state-to-terminal timing
-> G_FE + auxiliary runtime/resource outputs
-> fold-specific Decision models + first-trigger thresholds
-> baselines and predefined external evaluation
-> hierarchical inference and manuscript tables
```

正式模型选择是端到端 nested function evaluation，不是先生成一张 full-train label 表再做 Decision-only cross-validation。

## 3. 配置与 suite

活动配置只有：

```text
configs/phase1_bbob_train.yaml
configs/phase1_bbob_validation.yaml
configs/phase1_cec2017_test.yaml
```

共同é¢åæå®：

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

BBOB train / validation 为 function split，维度 10 / 20 / 40D、3 instances、5 optimizer seeds（2026-08-21 由 30 下调）；CEC2017 为 29 functions、10 / 30 / 50D、5 seeds。CEC2022 与工程问题只有在 benchmark factory、约束规则、reference、budget、repeats、endpoint 和 contrast é¢åæå®后才可纳入正式 pipeline。

## 4. Trajectory 与完整预算 endpoint

每个 `problem_id × algorithm × seed` 生成 native complete-state trajectory。decision states 只在 `phase1_dynamic_budget_event_v1` emitted complete updates 上保存，跨表键使用整数 `FE`。wall-clock truncation 只保留最后完整 native update，不能伪造 `FE_total` decision state。

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

`target_hit_observed := observed_first_hit_FE != null`；`target_hit_before_failure := target_hit_observed and not path_completed`；`path_completed := status == completed`；`endpoint_success := target_hit_observed and path_completed`。兼容 `first_hit_FE/success` 只能作为别名，不进入活动语义。

## 5. Behavior 与窗口

Behavior 只从完整 native-update history 计算，不从稀疏 checkpoint 选 anchor。w02/w05/w10 若名义位置不是 update 边界，就取不晚于目标的最近完整 update；实际 span 不小于名义 span，且偏差小于一次 population update。

窗口 metadata：

```text
effective_window_ratio_w02/w05/w10
effective_window_fe_w02/w05/w10
effective_native_updates_w02/w05/w10
```

这些只作数据质量检查，不进入 Decision X。Permutation-invariant Behavior 使用 set / distribution summaries；不得把 population 行号解释为跨代个体身份。正式 feature groups 为 `T0/B1/B2/B2+Motion/B2+Maturity/B3`，其中 B3 是活动组。

## 6. Query samples 与 features

活动 query IDs：

```text
descriptor_cheap_invariant
pflacco_standard_invariant
pflacco_broad_invariant
```

- `descriptor_cheap_invariant`：`lhs_50d`，14 columns，主配置。
- `pflacco_standard_invariant`：`lhs_50d`，37 columns，稳健性配置。
- `pflacco_broad_invariant`：`lhs_100d`，52 columns，稳健性配置。

cheap / standard 共用同一 `(X,y)`；broad 使用独立 sample。sample / feature 表保存 `query_protocol`、`query_preprocessing_id`、`sample_design_id`、`FE_query`、runtime、失败字段和 `query_first_hit_offset`。该 offset 只用于把 query 内首次达到 success target 的评价位置映射到全路径 FE，不进入 Behavior、Selector 或 Decision X。

## 7. Query reservoir 诊断

`trajectory_query_reservoir_v1` 只用于零额外 FE 的 trajectory-query 诊断，不替代主独立 query，也不并入主 estimand。

```text
query_id = descriptor_cheap_invariant
query_source_mode = trajectory_reservoir_zero_extra_fe
query_protocol = trajectory_query_reservoir_v1
query_preprocessing_id = unit_cube_x__median_iqr_y_v1
reservoir_size = 50 * dimension
```

Reservoir 只保存已评价点流的固定容量代表性子集；它能回答“既有评价流是否提供有用的 trajectory descriptors”，不能回答独立 landscape query 是否值得额外采样，也不能与主 Query Utility 直接合并比较。

## 8. State-action outcomes

每个 state 运行四个唯一动作：`continue_current` 加其余三个 algorithms。Query-adjusted matrix 使用 `B-FE_prefix-FE_query`；Behavior-only matrix 使用 `B-FE_prefix`。Selector target 使用 continuation-only raw action losses 的 statewise min-max 变换。

同一 query-adjusted matrix 上的 state-only 与 full Query Selectors，用 OOF selected continuation-only `log10_gap` 定义 `query_feature_predictive_increment_log10_gap`；该诊断排除 sample best，不新增 action losses，不作因果解释。

## 9. Stage-A 科学端点与 Stage-B 三次 future-path 计时

Stage-A 两套 action matrices 的预指定单次 outcome 唯一固定每条科学路径的 terminal gap、observed hit、path completion、endpoint success、planned/effective FE 与失败状态。Selector é¢åæå®后，Stage-B 将 selected Skip / Query / Behavior-only 从同一复制 state/RNG 到 terminal 真实 replay 预定三次，固定机器 / 线程 / 常驻进程，但只决定 wall-clock。canonical order 按 `cyclic_complete_path_v1` 循环移位；逐次保存 repetition、order、raw/censored 组件 / 完整路径时间、status、observed hit、path completion、endpoint success 与 effective FE。completed repetition 的 censored time 等于 raw，timed-out/failed repetition 为 `max(raw, role timeout)`，主时间使用三次 censored median，raw median 只作诊断。

路径身份、completed replays 内部 endpoint、Stage-A→completed replay endpoint 一致性分别保存；Stage-B status instability 与跨阶段 completion instability 也分别保存。任何 replay 不得覆盖科学字段或被选择性补跑。共享 prefix 是 sunk cost；FE=0→terminal policy wall-clock 独立保存且不进入 Utility。

## 10. Fold-specific 数据范围

每个 outer holdout 只读 outer-fit functions 的 SBS、Selectors、labels、Decision 和 threshold。每个 inner holdout 又只读 inner-fit functions，并重算 `SBS_inner`、cross-fit / 拟合 Selectors、生成三类 Utility。完整 BBOB-train threshold / Random calibration 也使用端到端 fold-specific OOF。

训练 label、outer evaluation 与 external deployment 必须保存 fit scope / fold metadata。不得让同一 function 的 in-sample Selector prediction 成为其 Decision OOF label。

## 11. 失败规则

BBOB train / validation 与 CEC2017 固定：failure cap `1e20`、取 log 前 gap floor / cap `1e-12/1e20`、success target `1e-8`、state-action timeout `3600 s`、Stage-A 逐 objective evaluation observed first hit。Stage-A timeout / failed path 的 final gap 按 cap 保留；若失败前已经命中，标准 ERT 保留 observed first hit，`endpoint_success=false` 继续表示路径未完成；未命中项计完整 planned budget。Stage-B timeout / failure 使用删失时间进入主 runtime，并另进入 timing failure / instability sensitivity。

所有计划运行先进入 coverage denominator。缺失状态键 / 矩阵是不完整数据生成，修复后重生成 shard；科学运行失败则保留有限 target 与 failure status。

## 12. 一致性检查

正式数据要求：

- trajectory / final endpoints 成对覆盖；
- emitted state / reservoir 均对齐 integer `FE`；
- Behavior permutation invariance 与窗口跨度成立；
- trajectory / Behavior 不含 reference / gap / outcome 输入字段；
- query sample、features、first-hit offset 与 FE charge 一致；
- 两套 action budgets 和 transition 字段一致；
- 三次计时与 cyclic order 完整；
- outer / inner fit scope 可重建；
- failure / timeout / target-hit / path-completion / endpoint-success / ERT 可逐行核对。

旧重建式 trajectory、identity-dependent Behavior、静态 bucket Selection Reference、Utility 和依赖它们的模型 / 结果全部历史。


## 13. Resource accounting

- Stage-A 只负责一次性生成 action matrices 与科学端点；Stage-B 只负责三次 timing replay 与 full-policy wall-clock。
- `FE=0` 完整在线 policy 时间与 decision-state future-path 时间分开保存，不能互相替代。
- 任何按 milestone / role / trajectory 估出来的 FE 数量都只是排期辅助，不是资源承诺。
- 旧的 cost-only 派生文档已被并入本总纲；若与最终 replay plan 冲突，以 replay plan 和具体 shard 为准。
