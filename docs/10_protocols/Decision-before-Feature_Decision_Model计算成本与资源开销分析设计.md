# Decision-before-Feature 计算成本与资源开销分析

> 唯一活动成本口径（2026-08-15）。12 个必选 milestones 的平均 prefix ratio 恰为 `0.35`，故 mandatory-only future-path remaining ratio 为 `0.65`，相对旧“全部 state 位于 0.60B”的 `0.40` 是 `1.625` 倍。本文只给出不含 event-only states/failure 的 mandatory-only 算术情景，并区分跨 Stage-A matrices 共享、进一步复用基础 trajectory 终值和当前未复用 producer。event-only 增量必须由物化 state/replay plan 另算；本文数字不是完整点估计、已完成运行量或资源承诺。

## 1. 两种时间端点

主 Utility 的科学 gap、`observed_first_hit_FE`、`target_hit_observed`、`endpoint_success`、planned FE、effective FE 与失败状态全部由预指定 Stage-A 单次运行固定。Stage-B 的 $T_s,T_q,T_b$ 从同一 decision state 的 query/no-query 分支确定之后计到 terminal，只决定 timing 项，不得改写 Stage-A 科学端点。共享 prefix、该机会的 Behavior 提取、Decision inference 与 threshold comparison 在三个分支确定前均已发生，是共同成本，不进入状态条件路径差；它们只进入 FE=0→terminal 完整政策 wall-clock。

Query future path 的动作特异增量包括真实 query sample generation/evaluation、feature extraction、fold-role 对应 Query Selector inference、必要 handoff 与 continuation；Behavior-only 包括其 fold-role 对应 Selector inference、必要 handoff 与 continuation；Skip 只含 native continuation。不得把共同的 Decision inference 只向 Query/Behavior-only 收费，也不得以 replay plan 中已经写好的 action 省略真实 Selector inference。每条 selected future path 在 Stage-B 真实执行预定 3 次，按 `cyclic_complete_path_v1` 交错。

最小 action loss 字段规范 v1 规定：每条 action 记录必须同时保留行标识、科学端点、censored runtime 和一个 canonical loss `action_loss`；其他旧 Utility 变体仅作兼容诊断。

每个 Stage-B repetition 保存 order、raw observed 组件/完整路径时间、`timing_replay_status in {completed,timed_out,failed}`、`observed_first_hit_FE`、`target_hit_observed`、effective FE、timeout、completion 和失败字段。`path_completed := status == completed`；`endpoint_success := target_hit_observed and path_completed`；`target_hit_before_failure := target_hit_observed and not path_completed` 只作诊断。正式 ERT 使用 `target_hit_observed`，不用 `endpoint_success`。

每条路径同时保存三次 raw observed runtime 及其中位数，并把 completed repetition 保持 raw、timed-out/failed repetition 置为 `max(raw observed runtime, role timeout)`，形成 censored repetitions 与主 `runtime_*_median`。主 Utility 的 $T_k$ 使用 censored median；raw observed median只作计时诊断。旧 `failure_worst_case` 字段仅可作为同一 censored 值的过渡兼容别名。

一致性不得压缩为一个布尔量。每条路径分别保存：`timing_replay_path_identity_consistent_{path}`；`completed_timing_replay_outcomes_internally_consistent_{path}`（少于两个 completed 时为 null/not_evaluable）；`stage_a_to_completed_timing_replays_consistent_{path}`（仅 Stage-A completed 且至少一个 Stage-B completed 时适用，否则为 null）；另存 `timing_replay_status_instability_{path}` 与 `stage_a_stage_b_completion_status_instability_{path}`。三次重复在运行前确定，不得在看到 runtime 或状态后选择性补跑、删除或替换。

在线政策另从 `FE=0` 计到 terminal，包含 prefix、逐机会 Behavior/Decision 与触发后的路径。该 full-policy wall-clock 是 RQ3 资源端点，不进入状态条件 Utility。它同样保存 `runtime_full_run_wall_clock_raw_observed_repetitions`、`runtime_full_run_wall_clock_raw_observed_median`、按 `policy_timeout_seconds` 处理的 `runtime_full_run_wall_clock_censored_repetitions`，以及作为 censored median 的主字段 `runtime_full_run_wall_clock_median`。两类时间必须分字段保存；完整政策运行所需的 prefix replay 不包含在下述阶段 B 数字中。

## 2. 基础 BBOB 量级

BBOB train + validation：

\[
24\times3\times3\times4\times30=25{,}920\text{ trajectory runs},
\]

\[
24\times3\times4\times30\times(10{,}000+20{,}000+40{,}000)
=0.6048\text{B FE}.
\]

12 个必选 milestones 为：

\[
0.20,0.22,0.24,0.26,0.28,0.30,0.34,0.38,0.42,0.46,0.50,0.60,
\]

其平均 prefix ratio 为：

\[
\bar r_{prefix}=\frac{4.20}{12}=0.35,
\qquad
\overline{1-r_{prefix}}=0.65.
\]

四个 prefix algorithms、每 run 12 个必选 states 的总预算 state basis 为：

\[
12\times0.6048=7.2576\text{B}.
\]

但 fold-specific 主 replay 每个 role 只使用其 `SBS_fold` prefix，不能把四个 prefix 全部计入。一个 prefix 的 state basis 为：

\[
7.2576/4=1.8144\text{B},
\]

其中 train 为 $1.3608$B，validation 为 $0.4536$B。

## 3. 阶段 A：一次 action outcomes

阶段 A 生成监督 Selector 所需的 action outcomes，各 action 只运行一次；同一预指定运行还唯一固定后续 Utility 使用的科学 terminal gap、observed hit、completion、endpoint success、planned/effective FE 与失败状态：

```text
trajectories/final performance
-> query sample artifacts/features
-> Query-adjusted four-action matrices
-> Behavior-only full-budget four-action matrix
-> main cheap FE=0 four-algorithm outcomes
-> fold-specific SBS/Selectors
-> selected replay plan
```

Query sample artifact 由 `problem × sample_design` 生成一次，不按 prefix、optimizer seed 或 state 重复。Action-loss continuation 只按 `FE_query` 扣减预算，不在阶段 A 内再次执行 query。因而：

\[
FE_{lhs50}=24\times3\times50\times(10+20+40)=0.000252\text{B},
\]

\[
FE_{lhs100}=0.000504\text{B}.
\]

全 prefix mandatory-only action matrices 必须按各 milestone 的实际剩余预算求和；由于这 12 个 milestone 的平均 ratio 恰为 0.35，可等价写成：

\[
FE_{query,5\%}=7.2576\times(4\times(1-0.35-0.05))
=17.41824\text{B},
\]

\[
FE_{query,10\%}=7.2576\times(4\times(1-0.35-0.10))
=15.96672\text{B},
\]

\[
FE_{behavior}=7.2576\times(4\times(1-0.35))
=18.86976\text{B}.
\]

主 cheap FE=0 四算法 outcomes 不重复 query artifact；四算法各运行 $0.95B$：

\[
FE_{pre\mbox{-}run,A}=0.1512\times(4\times0.95)=0.57456\text{B}.
\]

若 Query-adjusted 与 Behavior-only 两个 producer 共用一条 full-budget Skip，并让它与 Behavior-only matrix 的 `continue_current` action 共用同一 outcome，但尚不复用基础 trajectory 终值，则阶段 A 的跨矩阵共享情景为：

\[
FE_{A,main}^{mandatory}=0.6048+17.41824+18.86976+0.000252+0.57456
=37.467612\text{B},
\]

\[
FE_{A,three}^{mandatory}=0.6048+17.41824+15.96672+18.86976
+0.000252+0.000504+0.57456
=53.434836\text{B}.
\]

这里的 $37.467612$B 和 $53.434836$B 已经各包含 Behavior-only 四动作中的一条 full-budget `continue_current`，所以不能称为“与基础 trajectory 复用”。只有在逐 state 证明基础 trajectory 的 terminal gap、`observed_first_hit_FE`、`target_hit_observed`、planned/effective FE、status/completion 与该 continuation 完全同义后，才可再减：

\[
FE_{base\mbox{-}reuse}^{mandatory}=7.2576\times0.65=4.71744\text{B}.
\]

于是基础 trajectory 复用情景为：

\[
FE_{A,main,base\mbox{-}reuse}^{mandatory}=37.467612-4.71744
=32.750172\text{B},
\]

\[
FE_{A,three,base\mbox{-}reuse}^{mandatory}=53.434836-4.71744
=48.717396\text{B}.
\]

当前 `selection_reference/action_losses.py::_evaluate_state_action_outcomes_once`
对每个 action-budget CLI 使用 `path_labels=("skip", *portfolio)`，即在四个
actions 之外又执行一次 Skip。main cheap 的 Query-adjusted 与 Behavior-only
两个矩阵因此各多一次 mandatory-only 平均 $0.65B$ Skip：

\[
FE_{A,main,current}^{mandatory}=37.467612+2\times7.2576\times0.65
=46.902492\text{B}.
\]

三档当前实现量必须由最终 Stage-A 调用计划枚举，不在这里猜测矩阵调用次数。上述数值只覆盖 12 个必选 milestones；每条 event-only state 的 Query-adjusted continuation、Behavior-only actions 与重复 Skip 都按其实际 `FE_prefix` 另加，不能用平均 0.35 代替。

正式运行前必须三选一并按实际调用图定案：只实现跨矩阵共享；在逐行证明科学端点同义后进一步复用基础 trajectory；或保持未复用 producer 并按其调用量排期。此前按全部 state 位于 0.60B 得到的 Stage-A 数字，以及把 query sample 按 state/prefix 重复计入的估算，全部退出活动口径。

## 4. Fold-role replay multiplicity

活动链为 5 个 outer folds、每个 outer 内 4 个 inner folds，再加 final full-train→validation role。

- final role 覆盖全 train 一次和 validation 一次；
- 5 个 outer roles 各覆盖全 train 一次；
- 每个 train function 属于 4 个 outer-fit sets；每个 outer-fit set 有 4 个 inner roles，因此每个 train state另出现 $4\times4=16$ 次；
- 每个 train state 合计 $1+5+16=22$ 个 replay roles；validation state 只有 final role。

故 fold-role 加权 state basis 为：

\[
FE_{role\mbox{-}state}=22\times1.3608+0.4536=30.3912\text{B}.
\]

该式必须由实际 replay-plan 输出逐角色复核；若 folds、fit/holdout coverage 或 selector roles 改变，不能继续使用该数字。

这 22 项首先是逻辑 fold roles。Stage-A 科学 outcome 可由同一四动作矩阵按实际 selected action 复用，不必按 role 重跑。下述 Stage-B 算术情景仍保守地按每个 role 执行完整 selected path，因为活动 timing estimand 包含 artifact-specific Selector inference 与随后完整路径。若改为按 `state × matrix × actual action` 去重 physical continuation 并另计 Selector inference，必须先冻结逐 repetition 组件合成、cache/order、censoring 和 artifact 路由，并验证该组件化时间与完整路径 wall-clock 的目标一致；在此之前不得用去重量替换当前情景。

## 5. 阶段 B：selected future paths timing-only ×3

Stage-B 只测量 selected paths 的完整 future-path wall-clock 与重复状态；它执行真实目标评价以获得真实时间，但其 terminal/scientific outcomes 不进入 Utility 性能端点。对 12 个 mandatory milestones，Skip、Behavior-only 和任一 Query path 从 decision state 起平均各计划消耗 $0.65B$；Query 的 $0.65B$ 包含真实 sample FE 与 query-adjusted continuation。相对旧的 $0.40B$ 假设，mandatory-only remaining-path multiplier 为：

\[
\frac{0.65}{0.40}=1.625.
\]

主 cheap 的三条路径为：

\[
FE_{B,main}^{mandatory}=30.3912\times(3\times0.65)\times3
=177.78852\text{B}.
\]

三档配置可共享 Skip 与 Behavior-only，但三个 Query Selectors 可能选择不同动作，故为五条路径：

\[
FE_{B,three}^{mandatory}=30.3912\times(5\times0.65)\times3
=296.3142\text{B}.
\]

主 pre-run AAS 的 outer-holdout/train 与 final validation paired Stage-A outcomes 合计覆盖每个 problem-seed 一次；从 FE=0 的 selected policy path 另做三次 timing-only 运行：

\[
FE_{pre\mbox{-}run,B}=0.1512\times1.00\times3=0.4536\text{B}.
\]

由此得到只含 mandatory milestones、跨矩阵共享但不复用基础 trajectory 的算术情景：

\[
FE_{main}^{mandatory}=37.467612+177.78852+0.4536
=215.709732\text{B},
\]

\[
FE_{three}^{mandatory}=53.434836+296.3142+0.4536
=350.202636\text{B}.
\]

这里“三档”只表示三种 query 的 state-level Query paths，加上当前主 cheap pre-run AAS；它不包含 standard/broad 各自完整 FE=0 policy/baseline paths，因而不是三档完整评价总量。

若进一步满足基础 trajectory 复用条件，则总量为：

\[
FE_{main,base\mbox{-}reuse}^{mandatory}=32.750172+177.78852+0.4536
=210.992292\text{B},
\]

\[
FE_{three,base\mbox{-}reuse}^{mandatory}=48.717396+296.3142+0.4536
=345.485196\text{B}.
\]

若保持当前 main Stage-A producer 的两次额外 Skip，则相应 main 总量为：

\[
FE_{main,current}^{mandatory}=46.902492+177.78852+0.4536
=225.144612\text{B}.
\]

三档 current-producer 总量暂不填写，等待实际 Stage-A 调用图枚举。上述任何总量都不含 event-only states；其增量至少依赖 event 数、每个 event 的实际 `FE_prefix`、fold role、path 与 selected action，必须从最终 replay plan 求和。此前按全部 state 位于 0.60B 或未展开 fold-role multiplicity 得到的较小总量均退出活动排期口径。

## 6. FE=0 完整在线政策的额外量

当前 online evaluator 对每个 `function × dimension × instance × optimizer seed` base tuple 执行 7 条固定政策路径和 30 条 matched-rate Random 路径；每条路径又包含 1 次 Stage-A 科学运行和 3 次 Stage-B timing replay。因此每个 base tuple 需要：

\[
(7+30)\times(1+3)=148\text{ full runs}.
\]

按当前 CEC2017 配置的 29 functions、1 instance、10D/30D/50D、30 seeds 与 `FE_total=1000D`，仅这一 online policy evaluator 的 planned FE 为：

\[
29\times30\times148\times1000\times(10+30+50)
=11.5884\text{B FE}.
\]

它尚未包含 query artifacts、静态 VBS 所需的四算法终值、CEC2022 或工程问题。当前函数集 F1--F29 仍有 F2/F30 blocker，因此 $11.5884$B 只描述现配置，不表示 benchmark 口径已确认。

已见的 BBOB-validation 内部评价集若按 6 functions、3 instances、10D/20D/40D、30 seeds 执行同一 148-run 政策集合，则 planned FE 为：

\[
6\times3\times30\times148\times1000\times(10+20+40)
=5.5944\text{B FE}.
\]

但当前 `decision.online_controller_evaluate` 只接受 CEC suite、固定 `instance=1`，query feature/sample 键也没有 instance，因此这 $5.5944$B 是所需覆盖的成本说明，不是当前可执行成本。支持该已见内部评价集的 instance-aware full-policy endpoint 是正式运行 blocker；即使实现，也只能提供 post-development internal evaluation，不能恢复“未查看确认集”资格。

## 7. Mandatory-only 情景之外的未知增量

跨矩阵共享情景的 215.709732B 与 350.202636B、基础 trajectory 复用情景的 210.992292B 与 345.485196B，以及当前 main producer 情景的 225.144612B，只是 12 个必选 milestones 的算术口径，仍未包含：

- 12 个 milestones 之外的 event-only states 及其实际 remaining budget；
- BBOB-validation 与 CEC 的 FE=0→terminal 完整在线政策；上一节只给出当前配置下的独立 planned-FE 说明，尚未并入 replay 总表；
- standard/broad 各自的 FE=0 action outcomes、baseline 与 full-policy paths；
- selector role/selected action 在实际 replay plan 中产生的额外不可复用路径；
- 当前 Stage-A Skip 重复与目标复用方案之间尚未关闭的实现差异；
- timeout、失败、修复后重跑和 coverage 补齐；
- CEC2017、CEC2022、工程问题和额外 query replicate；
- I/O、feature/Selector/Decision 计算、存储和峰值内存。

event-only 的 Stage-A 与 Stage-B 增量分别必须按实际 state/action 调用图和 fold-role replay plan 逐行求和；没有物化数据时不得填单一比例或点估计。因此这些数字只能称为“mandatory-only 算术情景”，不能称为完整下界、完整预算或资源可行性证明。

## 8. 正式运行 blockers

正式运行前必须同时满足：

1. 阶段 A 的 query artifact 与 action-loss FE 语义按实际 producer 一致；
2. replay planner 仅有枚举能力；offline decision-state-to-terminal runner 尚未实现，fold-role-complete plan 尚未物化实测并核对，是正式运行 blocker；
3. outer/inner/full-train `cv_group_id = function_id` cross-fitted Selectors 必须持久化，replay plan 必须提供历史 `fold role/cv_group_id -> Selector artifact` key 路由；否则无法真实计入 Selector inference；
4. planner 已物化正式 outer/inner/full-train replay plan，并逐角色核对 mandatory/event states、paths、repetitions 与 FE；
5. Stage-A 科学端点与 Stage-B timing-only replay 使用不同字段；后者逐次保存 status、raw/censored runtime、observed hit、effective FE、timeout/completion、三类一致性与两类 instability，且不存在选择性重跑；
6. future-path timing 与 FE=0 full-policy wall-clock 使用不同字段且均为真实运行；
7. 同一硬件/线程/预加载条件下已有 evaluator throughput、组件 timing、存储和 peak-memory 测量；
8. event-only 与外部 suite 成本已加入排期；
9. Stage-A producer 已明确采用跨矩阵共享、经证明的基础 trajectory 复用或当前未复用调用图之一，并据此重算资源；
10. online evaluator 必须支持已见 BBOB-validation 内部评价集的全部 instances，且入口直接按 validation config 强制核对 functions/instances/dimensions/seeds/problem-state coverage；否则论文不得报告其 full-policy endpoint；
11. standard/broad 各自的 Selector、Decision/threshold、baseline、Stage-B future paths 与 FE=0 full-policy 路径已实现并计入资源；当前 main-only online evaluator 不满足该条件；
12. CEC2017 F2/F30 已闭合，CEC2022 benchmark factory 与完整配置已实现，工程问题 factory、约束处理和完整配置已实现；
13. CEC2022 与工程问题已在查看 outcome 前冻结 endpoint-specific precision target、只使用开发集合方差信息的 repeats 确定方法与最终重复数；BBOB/CEC2017 的既定 30 seeds 不得作为充分性依据直接外推。

当前上述条件未闭合，资源与排期是正式运行 blocker。允许按 main cheap BBOB → standard/broad robustness → CEC2017 → factory/配置/约束均闭合的 CEC2022/工程问题分阶段执行，但任何阶段不得冒充未完成的全协议结论；BBOB-validation 只作已见内部评价。
