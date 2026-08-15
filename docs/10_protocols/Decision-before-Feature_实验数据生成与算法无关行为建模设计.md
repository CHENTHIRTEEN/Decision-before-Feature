# Decision-before-Feature 实验数据生成与算法无关行为建模设计

> 唯一活动数据协议（2026-08-14）。数据以 complete native optimizer state、整数 FE 状态键和逐完整 native-update Behavior 为基础。旧 population-only checkpoint 重建、跨代行号身份与稀疏 checkpoint 窗口全部退出。

## 1. 数据生成目标

离线数据必须支持：

1. 从同一可续跑 optimizer state 构造 Skip、Query 和 Behavior-only paths；
2. 只用 query 前 Behavior 训练 Decision；
3. 在 fit functions 内完整重拟合 SBS、Selectors、Utility 和 Decision；
4. 重建 run-level first-trigger policy 与终端 gap/runtime/target-hit/path-completion/ERT；
5. 分开保存 Stage-A 科学失败/端点、Stage-B 三次 decision-state future-path 计时及逐次状态/instability，以及 FE=0 policy wall-clock。

## 2. Complete optimizer state

每个 emitted state 至少保留 population、fitness、best-so-far、FE、generation/native update、算法内部动态量和 RNG state。DE/PSO/CMA-ES/SHADE 的同算法 continuation 必须原生恢复；跨算法只转移 population、fitness、best position，并初始化新算法内部状态一次。

trajectory snapshot 只对应 emitted complete native update 的实际整数 FE。timeout 若发生在 update 中间，只保留最后完整 update，不把部分 state 写成 decision opportunity。完整预算 endpoint 单独保存，不能为 `FE_total` 伪造 trajectory state。

## 3. 动态状态采样

`phase1_dynamic_budget_event_v1` 在 `0.20–0.60`、步长 0.01 的监测网格上工作。必选 milestones：

```text
0.20, 0.22, 0.24, 0.26, 0.28, 0.30,
0.34, 0.38, 0.42, 0.46, 0.50, 0.60
```

事件为 improvement resume、stagnation onset、effective-rank change、elite migration 和 diversity recovery。每个跨过监测网格的完整 native update 只判定一次；同一 update 含 milestone 时合并为该 milestone row，不消耗 event-only 配额。每阶段最多 2 个 event-only states，实际 ratio 间隔至少 0.02，每 run 共 12–18 states。

`FE_ratio=FE/FE_total`；`budget_milestone_ratio` 仅是名义节点。样本不由模型分数选择，不作事后重加权。

## 4. 完整预算 endpoints

`final_performance.parquet` 对每个 `problem_id × algorithm × seed` 在 `FE=FE_total` 恰好一行，协议为：

```text
complete_budget_native_optimizer_run_with_first_hit_endpoints
```

保存：

```text
benchmark_reference_value
final_gap
log10_gap
log10_gap_floor
log10_gap_cap
success_gap_target
success
first_hit_FE
```

first hit 在每次 objective evaluation 记录。该表用于 fold-specific SBS、静态 VBS 和最终评价。trajectory 与 Behavior 表本身不保存 reference、gap、`observed_first_hit_FE`、`target_hit_observed`、`path_completed`、`endpoint_success` 或 Utility，防止这些 outcome 进入 Decision 输入。

## 5. Behavior windows

w02/w05/w10 使用逐次完整 native-update history。若名义 anchor 不是 update 边界，取不晚于目标的最近完整 update；实际 FE span 不小于名义 span，且偏差小于一次 population update。所有 rate/slope 使用实际 `ΔFE/FE_total`。

窗口 metadata：

```text
effective_window_ratio_w02/w05/w10
effective_window_fe_w02/w05/w10
effective_native_updates_w02/w05/w10
```

只作数据质量检查，不进入 Decision X。

## 6. Permutation-invariant Behavior

跨窗口 population comparison 使用经验 Wasserstein、centroid/elite shifts、Chamfer-style set distance 与 covariance summaries；fitness 使用排序分位数、IQR 与分布 Wasserstein。不得把 population 行号解释为跨代个体身份。

正式 feature groups 为 T0/B1/B2/B2+Motion/B2+Maturity/B3，字段数 1/19/25/28/28/31。总输出 34 个唯一字段，31 个活动输入、3 个 diagnostic-only。Search Maturity 只是既有 Behavior 的三项确定性变换。

## 7. Query samples 与 operational endpoint

三档 query IDs 为 `descriptor_cheap_invariant`、`pflacco_standard_invariant`、`pflacco_broad_invariant`。cheap/standard 共用同一 `lhs_50d` `(X,y)`；broad 使用独立 `lhs_100d`。随机流只由显式整数与 `numpy.random.SeedSequence` 构造。

query sample 不插入 optimizer population，但它属于真实已观察 objective evaluations。sample 表保存 `query_first_hit_offset`。主 Query path terminal best、`observed_first_hit_FE` 与 ERT 合并 prefix、query sample 和 selected continuation；`target_hit_observed`、`path_completed` 与 `endpoint_success` 分列。同时保存 continuation-only gap 与 `query_sample_best_contribution_log10_gap`。

## 8. State-action outcomes

每个 state 运行四个唯一动作：`continue_current` 加其余三个 algorithms。Query-adjusted matrix 使用 `B-FE_prefix-FE_query`；Behavior-only matrix 使用 `B-FE_prefix`。Selector target 使用 continuation-only raw action losses 的 statewise min-max 变换。

同一 query-adjusted matrix 上的 state-only 与 full Query Selectors 用 OOF selected continuation-only `log10_gap` 定义 `query_feature_predictive_increment_log10_gap`；该诊断排除 sample best、不新增动作运行且不作因果解释。

## 9. Stage-A 科学端点与 Stage-B 三次 future-path 计时

Stage-A 两套 action matrices 的预指定单次 outcome 唯一固定每条科学路径的 terminal gap、observed hit、path completion、endpoint success、planned/effective FE 与失败状态。Selector 冻结后，Stage-B 将 selected Skip/Query/Behavior-only 从同一复制 state/RNG 到 terminal 真实 replay 预定三次，固定机器/线程/常驻进程，但只决定 wall-clock。canonical order 按 `cyclic_complete_path_v1` 循环移位；逐次保存 repetition、order、raw/censored 组件/完整路径时间、status、observed hit、path completion、endpoint success 与 effective FE。completed repetition 的 censored time 等于 raw，timed-out/failed repetition 为 `max(raw, role timeout)`，主时间使用三次 censored median，raw median 只作诊断。路径身份、completed replays 内部 endpoint、Stage-A→completed replay endpoint 一致性分别保存；Stage-B status instability 与跨阶段 completion instability 也分别保存。任何 replay 不得覆盖科学字段或被选择性补跑。共享 prefix 是 sunk cost；FE=0→terminal policy wall-clock 独立保存且不进入 Utility，并采用相同科学/计时分离。

## 10. Fold-specific 数据范围

每个 outer holdout 只读 outer-fit functions 的 SBS、Selectors、labels、Decision 和 threshold。每个 inner holdout 又只读 inner-fit functions，并重算 `SBS_inner`、cross-fit/拟合 Selectors、生成三类 Utility。完整 BBOB-train threshold/Random calibration 也使用端到端 fold-specific OOF。

训练 label、outer evaluation 与 external deployment 必须保存 fit scope/fold metadata。不得让同一 function 的 in-sample Selector prediction成为其 Decision OOF label。

## 11. 失败规则

BBOB train/validation 与 CEC2017 固定：failure cap `1e20`、取 log 前 gap floor/cap `1e-12/1e20`、success target `1e-8`、state-action timeout `3600 s`、Stage-A 逐 objective evaluation observed first hit。Stage-A timeout/failed path 的 final gap 按 cap 保留；若失败前已经命中，标准 ERT 保留 observed first hit，`endpoint_success=false` 继续表示路径未完成；未命中项计完整 planned budget。Stage-B timeout/failure 使用删失时间进入主 runtime，并另进入 timing failure/instability sensitivity。

所有计划运行先进入 coverage denominator。缺失状态键/矩阵是不完整数据生成，修复后重生成 shard；科学运行失败则保留有限 target 与 failure status。CEC2022/工程问题必须先冻结同类 endpoint 与 constraint rule。

## 12. 一致性检查

正式数据要求：

- trajectory/final endpoints 成对覆盖；
- emitted state/reservoir 均对齐 integer FE；
- Behavior permutation invariance 与窗口跨度成立；
- trajectory/Behavior 不含 reference/gap/outcome 输入字段；
- query sample、features、first-hit offset 与 FE charge 一致；
- 两套 action budgets 和 transition 字段一致；
- 三次计时与 cyclic order 完整；
- outer/inner fit scope 可重建；
- failure/timeout/target-hit/path-completion/endpoint-success/ERT 可逐行核对。

旧重建式 trajectory、identity-dependent Behavior、静态 bucket Selection Reference、旧 Utility 和依赖它们的模型/结果全部撤回。
