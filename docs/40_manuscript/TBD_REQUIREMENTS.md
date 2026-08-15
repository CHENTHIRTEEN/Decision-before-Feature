# 论文待补证据清单

## 1. 当前证据状态

本文只有方法与结果占位，不含当前协议下的正式实证结论：

- 72 个 BBOB train/validation trajectory shards 尚未启动；
- 旧 trajectory 下游数值已撤回，不得进入摘要、主文表图或结论；
- BBOB-validation 已被历史开发结果查看，现只可作为已见的 post-development 内部评价集，不能称 untouched/confirmatory；该集合与 CEC2017、CEC2022、工程问题均未完成当前协议下的评价；
- CEC2017 F2/F30 口径尚待核对；CEC2022 与工程问题的 suite、预算、reference、失败端点和约束规则尚未冻结；
- replay planner 已有枚举能力，但正式 plan 尚未物化实测；decision-state-to-terminal runner、fold-role→Selector artifact 路由、BBOB-validation instance-aware online coverage、standard/broad full-policy、Stage-A Skip 复用、真实 evaluator timing 与资源排期尚未闭合；BBOB-validation 的 functions/instances/dimensions/seeds/problem-state coverage 也尚未由 validation 配置逐项强制核对；
- CEC2022 benchmark factory/配置与工程问题 factory/约束处理/配置尚未实现，不能仅以文档占位视为外部套件可执行。

任何 TBD 只能由当前活动协议的新产物补齐。禁止读取 archive、旧 query ID、旧 Utility 字段或非完整嵌套预测。

## 2. 共同证据链与正式产物

正式链必须依次完成：trajectory/final performance → permutation-invariant Behavior → query samples/features → Stage-A 两套四动作 matrices 与 FE=0 outcomes 各运行一次并冻结科学端点 → fold-specific SBS/Selectors → outer/inner/full-train selected replay plan → Stage-B selected Skip/Query/Behavior-only future paths 与 FE=0 policy paths 各真实计时三次 → endpoint consistency/instability 检查 → Utility/Decision → first-trigger policies → 统计表图。不得从 action matrices 直接生成带主计时的最终 Utility，也不得用 Stage-B outcome 改写 Stage-A 科学端点。

| 代码 | 正式产物 |
| --- | --- |
| P1 | BBOB train/validation 72 个 trajectory shards 与配对 `final_performance`；整数 FE 状态键、完整 optimizer state、12--18 状态/run |
| P2 | 基于完整 native updates 的 31 个正式 Behavior 输入、3 个诊断字段和窗口 metadata；逐状态 population permutation consistency |
| P3 | `descriptor_cheap_invariant` 14 列、`pflacco_standard_invariant` 37 列、`pflacco_broad_invariant` 52 列；query/sample/protocol/FE 逐行一致 |
| P4 | Stage-A Query-adjusted 与 Behavior-only full-budget 两套四动作 matrices，以及各需报告 query 的 FE=0 四算法 outcomes；逐行保存科学 gap、observed hit、completion、endpoint success、planned/effective FE/failure，失败动作保留 |
| P5 | outer/inner/full-train fold-specific SBS 和四类 Selector 的 cross-fitted/frozen predictions、持久化 artifacts、`fold role/family -> Selector artifact` 路由、selected actions、regret 与 replay plan |
| P6 | Stage-B selected Skip/Query/Behavior-only decision-state-to-terminal paths 与 FE=0 policy paths 各三次真实 timing replay；逐次原始顺序、raw/censored time、status、observed hit、effective FE、timeout/completion/failure，三类一致性与两类 instability；不选择性重跑 |
| P7 | query-specific Utility/Decision 数据；主字段 `u_query_joint_lamT_1`，并含 Behavior-only Utility、两类增量、sample endpoint 与动作关系 |
| P8 | LDA、Logistic Regression、Ridge 的完整 nested grouped-by-function OOF；B3 选模、完整 train `oof_utility_first_trigger` threshold 与 Random calibration |
| P9 | 九个预设角色、八个不重复 outcome 的等预算 policy 输出；matched-trigger Behavior-only 仅作诊断 |
| P10 | 六组 T0/B1/B2/B2+Motion/B2+Maturity/B3 消融、线性方向稳定性和 maturity 关联 |
| P11 | 冻结模型在已见 BBOB-validation 内部评价集、CEC2017、CEC2022 与工程问题上的 suite-specific 覆盖、失败、性能和资源输出；BBOB 结果不得称为未查看确认性证据 |
| P12 | 固定六个 BBOB-validation functions 的条件 bootstrap、逐 function effects、有限集均值、仅作函数组成敏感性的 function-resampling、假设敏感的 exact sign-flip/Holm、描述性 Bonferroni simultaneous interval、operational tolerance、专用 ERT ratio bootstrap 与 failure sensitivity 表图 |

## 3. 所有结果必须保存或派生的核心字段

### 3.1 Utility 与路径端点

```text
gap_skip_terminal
gap_query_terminal
gap_behavior_only_terminal
gap_query_continuation_only
query_sample_best_contribution_log10_gap
query_first_hit_offset
scientific_endpoint_source  # stage_a_selection_reference_outcome | stage_a_online_policy_outcome
{skip,query_path,behavior_path}_{planned,effective}_FE
{skip,query_path,behavior_path}_observed_first_hit_FE
{skip,query_path,behavior_path}_target_hit_observed
{skip,query_path,behavior_path}_target_hit_before_failure
{skip,query_path,behavior_path}_path_completed
{skip,query_path,behavior_path}_endpoint_success
runtime_{skip,query_joint,behavior_only_full_budget}_raw_observed_repetitions
runtime_{skip,query_joint,behavior_only_full_budget}_raw_observed_median
runtime_{skip,query_joint,behavior_only_full_budget}_censored_repetitions
runtime_{skip,query_joint,behavior_only_full_budget}_median  # censored median, main Utility
timing_replay_status_repetitions_*
timing_replay_effective_FE_repetitions_*
timing_replay_timed_out_flags_*
timing_replay_path_completed_flags_*
timing_replay_observed_first_hit_FE_repetitions_*
timing_replay_target_hit_observed_flags_*
timing_replay_target_hit_before_failure_flags_*
timing_replay_endpoint_success_flags_*
timing_replay_path_identity_consistent_*
completed_timing_replay_outcomes_internally_consistent_*
stage_a_to_completed_timing_replays_consistent_*
timing_replay_status_instability_*
stage_a_stage_b_completion_status_instability_*
runtime_full_run_wall_clock_raw_observed_repetitions  # online FE=0 policy
runtime_full_run_wall_clock_raw_observed_median
runtime_full_run_wall_clock_censored_repetitions
runtime_full_run_wall_clock_median  # censored median
u_query_joint_lamT_{0,025,05,1,2}
u_behavior_only_full_budget_lamT_{0,025,05,1,2}
query_operational_increment_lamT_{0,025,05,1,2}
query_feature_predictive_increment_log10_gap
```

主 Utility 的科学性能与 FE 字段只取预指定 Stage-A；时间项使用同一 decision state 分支后的 Stage-B censored median。共享 prefix、Behavior extraction、Decision inference 与 threshold comparison 是分支前共同成本，只进入 FE=0→terminal policy wall-clock。每次 raw observed time 原样保留；completed 的 censored time=raw，timed_out/failed 的 censored time=`max(raw, role timeout)`，避免快速失败降低主时间成本。三类一致性字段按各自适用条件保存 null/not_evaluable，不得压成一个布尔量；两类 instability 分开保存。不得按观察到的时间或状态选择性重跑。

`query_operational_increment_lamT_1` 比较不同预算的 Query 与 Behavior-only operational paths，包含 query FE/runtime、sample best、预算差与 Selector 差异；必须在全 eligible states 与 Proposed-triggered states 同时报，且不是纯信息效应或因果 estimand。`query_feature_predictive_increment_log10_gap` 比较同一 Query-adjusted outcomes 上 state-only 与 full Query Selector 的 OOF selected continuation-only gap。正式五路径还必须提供 descriptor-use、state-only-vs-sampling 与 sampling-direct 操作性增量及逐行加法一致性；这些量回答不同问题，均不作因果解释。

### 3.2 样本、动作与失败

```text
query_id
query_protocol
sample_design_id
FE_query
query_sample_best
query_first_hit_offset
best_observed_action
potential_gain_raw
selector_regret_raw
selected_equals_default
selected_equals_prefix
handoff_required
handoff_type
controller_status
query_status
selector_status
action_status
optimizer_status
attempted_coverage
```

逐行检查 `handoff_required = not selected_equals_prefix = (handoff_type == population_transfer_initialization)`，并满足 `target_hit_observed := observed_first_hit_FE != null`、`target_hit_before_failure := target_hit_observed and not path_completed`、`path_completed := status == completed`、`endpoint_success := target_hit_observed and path_completed`。兼容 `first_hit_FE/success` 只能分别别名为 `observed_first_hit_FE/target_hit_observed`；正式 ERT 使用 `target_hit_observed`。Query sample 不进入 population，但其 best/observed hit 属于 operational Query endpoint。timeout/失败行保留并使用 suite 冻结 cap；所有计划 run 先进入 coverage denominator。

FE=0 AAS 另显式固定：`prefix_algorithm=selected_algorithm`（只作关系记账）、`selected_equals_prefix=true`、`handoff_required=false`、`handoff_type=fresh_optimizer_initialization`、`default_algorithm=no_query_algorithm=SBS_fold`。

## 4. Abstract 与 Results 映射

| ID | 所需产物 | 必需分析与完成判据 |
| --- | --- | --- |
| `TBD-ABS-RQ1` / `TBD-RQ1-*` | P1--P7、P12 | 限于主 14 维 query、SBS prefix、eligible states 与固定 problem-keyed LHS realizations；报告 $U_q^{joint}$、$U_b$、全 eligible/Proposed-triggered $I_q$、`Pr(U<=0)`、continuation/sample 分解、效应量和区间；state → run → static problem → fixed dimension stratum → function |
| `TBD-ABS-RQ2` / `TBD-RQ2-*` | P7--P8、P11--P12 | B3 train grouped-by-function outer-OOF first-trigger mean只用于三候选选模与开发诊断；AUROC/AP/Spearman 辅助，RMSE 仅 Ridge；selected procedure 与 milestone-only B3--T0 在已见 BBOB-validation 上只作 post-development 内部评价，不具 untouched/confirmatory 资格；各真正外部 suite 分开报告 |
| `TBD-ABS-RQ3` / `TBD-RQ3-*` | P7--P9、P12 | 九角色由八个不重复 outcome 覆盖：Never/SBS 共用一行，其余 Always、Random、pre-run AAS、VBS、milestone T0、self-thresholded Behavior-only、Proposed 各一行；报告 Utility、final gap、target-hit rate、endpoint success、ERT、FE、future-path time、FE=0 policy wall-clock、call/trigger/handoff rate 和 failure |
| `TBD-ABS-RQ4` / `TBD-RQ4-*` | P8、P11--P12 | BBOB-validation 与每个外部 suite 分开报告完整计划覆盖、失败率、效应量和区间；所有 train-derived components 冻结；未关闭 suite blocker 或覆盖不足时拒绝泛化主张 |
| `TBD-ABS-RQ5` / `TBD-RQ5-*` | P8、P10--P12 | 六组 `1/19/25/28/28/31` 在同一 rows、folds、模型名下比较；六个预设 contrasts、方向稳定性与 maturity 关联只作预测解释，不重选模型或输入组 |

### 4.1 RQ1 必需表图

- 表：按 split、dimension、actual FE ratio 分层报告 $U_q^{joint}$、$U_b$、$I_q$、性能项、future-path runtime ratio、sample-best contribution、N、coverage 与 95% CI。
- 图：主 Utility 分布与零线；family-aware uncertainty；Query terminal gap 与 continuation-only gap 的配对差。
- 统计：BBOB-validation 的内部评价 estimand 为固定 F5/F9/F13/F14/F19/F24、dimensions 与 instances 1/2/3 上的等权有限集均值；10,000 次条件 CI 不重抽 function 或 static problem，只在每个固定 static problem 内配对重抽 optimizer seeds，并对每个抽中 seed/run 保留完整有序 states。function-resampling 只作函数组成敏感性。由于该集合已被历史开发查看，这些分析均不得称为确认性，也不作 function 或 transformed-instance 超总体推断。

### 4.2 RQ2 必需表图

- 面板 A：B3 上 LDA/Logistic/Ridge 的 train grouped-by-function outer-OOF first-trigger Utility；列出 AUROC、AP、Spearman，Ridge 另列 RMSE。
- 面板 B：冻结所选模型名后，milestone-only B3--T0 的 train OOF 与 BBOB-validation 配对效应。
- 三候选的 3 个两两 train-OOF contrasts 只作选模诊断，不作为 selected procedure 的无偏估计；milestone-only B3--T0 是 BBOB-validation 的预指定内部主要比较，但不是未查看确认性检验。
- BBOB-validation `n=6` 的双侧 exact sign-flip raw p 最小为 0.03125；RQ3 与 RQ5 各自六 contrast Holm family 的最小 adjusted p 均为 0.1875，故在 0.05 下不可能拒绝。RQ4 按 suite 与 endpoint 分开，不建立跨四个 suites 的 Holm family。鉴于集合已见，表中 p 值只作内部描述，主输出为逐 function/problem effects、固定有限集均值和条件 CI。

### 4.3 RQ3 必需表图

- 八个 outcome 行明确九个角色语义；Never Query 的 query-decision Utility 为 0，SBS/VBS 对该指标为 N/A，slash 行不是两个估计的平均。VBS 必须在每个 function × instance × dimension problem 内先按 seeds 聚合各算法完整预算 clipped `log10_gap`、选择均值最低算法，再汇总该算法的 paired seed outcomes；不得逐 seed 选择最小算法。
- `self_thresholded_behavior_only` 使用自己的 $U_b$ threshold；`matched_trigger_behavior_only` 只用于 Proposed 首次触发状态的 $I_q$，不作为第十个 baseline。
- Random 的 30 streams 先在同一科学 run 内平均。Never/SBS、Always、Random、pre-run AAS、milestone T0、self-thresholded Behavior-only 构成 Proposed 的 6 个 endpoint-specific Holm contrasts；VBS 和 matched-trigger诊断不进入。
- 图中同时显示 final `log10_gap`、future-path runtime 与 FE=0 policy wall-clock 的区间；不得仅凭 call rate 或单点连线宣称性能优势/Pareto。

### 4.4 RQ4 必需表图

- 每个 suite 列出预注册 functions/problems、dimensions、seeds、budget、reference、success target、timeout、constraint rule、attempted coverage 和各类 failure。
- Proposed 对 Never、Always、milestone T0、Behavior-only 的 paired effects按 suite与 endpoint 分开，不把 benchmark 池化为单一显著性结论。
- 未闭合 pair 按 gap floor/cap、`target_hit_observed` 1/0、未命中 ERT 项的 full planned budget、runtime complete-pair 最小正值/timeout 双向赋值，并重算 Utility。attempted coverage <95%，或 sensitivity 改变方向、区间相对 operational tolerance 的位置任一项时，对应 suite × endpoint 结论标为“未建立”。

### 4.5 RQ5 必需表图

```text
B1 - T0
B2 - B1
B2+Motion - B2
B2+Maturity - B2
B3 - (B2+Motion)
B3 - (B2+Maturity)
```

六个 contrasts 组成一个 Holm family。系数或 LDA 判别方向必须来自标准化的 fold-specific fit，并报告跨 function/fold 的符号和幅度稳定性。Maturity 曲线必须带区间；不能仅凭视觉宣称非单调，也不能把确定性基函数解释为因果阶段。

## 5. Discussion、Reproducibility 与 Conclusion 映射

| ID | 完成要求 |
| --- | --- |
| `TBD-DISC-01` | 只解释主 14 维 query 的 $U_q^{joint}$、$U_b$、$I_q$ 与 sample endpoint，写明比较方向、效应量和区间 |
| `TBD-DISC-02` | 区分主 first-trigger Utility 与 AUROC/AP/Spearman/Ridge RMSE；解释 B3--milestone T0，不隐藏指标冲突 |
| `TBD-DISC-03` | 联合解释 equal-FE final gap、target-hit/endpoint-success/ERT、future-path runtime、FE=0 policy wall-clock、call/handoff 和无效调用成本 |
| `TBD-DISC-04` | BBOB-validation、CEC2017、CEC2022、工程问题逐 suite 讨论覆盖、失败与区间；不从部分运行外推 |
| `TBD-DISC-05` | 六组消融与 stability 只作预测关联解释；不得声称 Search Maturity 单独致效 |
| `TBD-DISC-06` | 披露状态覆盖、query/selector/action/optimizer failures、family dependence、handoff、lambda/query sensitivity 和双向极端 failure analysis |
| `TBD-REPRO-RESOURCE-01` | 记录 processor、OS、Python environments、thread、batch/cache、memory procedure、replay multiplicity、存储、同系统 evaluator/behavior/query/selector/controller timings；区分 future-path 与 FE=0 policy time |
| `TBD-CONC-RQ1`--`RQ5` | 逐字复用 Results 已支持的范围、效应与区间；未完成项保持“尚未验证”，不得新增因果、普适或跨 suite 结论 |

## 6. 敏感性与诊断的完整边界

`TBD-SENS-DIAG-01` 需要：

- 三档 query 各自完整 Selector、Utility、Decision/threshold、baseline、Stage-B future paths 与 FE=0 full-policy 评价；当前 online evaluator 只支持 main cheap，因此 standard/broad 是正式 blocker；
- `lambda_T={0,0.25,0.5,1,2}` 全部报告，主值仍为 1；
- `selected_equals_default`、`selected_equals_prefix`、`handoff_required` 分别分层；
- Query-adjusted state-only、query-only、full Query 与 Behavior-only Selector 的 observed selected loss/regret；
- `query_feature_predictive_increment_log10_gap` 与 `query_operational_increment_lamT_1` 分栏，不能互相替代；
- `schedule_conditioned_T0`、reservoir、全 prefix、不同 query 和 lambda 只作敏感性/诊断，不进入主模型选择。
- 主 unweighted state-row fit 与 function/dimension/static-problem/run-balanced fit 的预设 sensitivity；后者尚未实现/运行，是 blocker。
- 若运行前增加多个独立 LHS replicates，只作为 sampling-robustness sensitivity 并纳入配对/层级统计；否则明确结论条件于单个固定 realization，standard--broad 不拆分归因。

## 7. 补齐顺序

1. 先关闭 decision-state-to-terminal runner、fold-role→Selector artifact 路由、已见 BBOB-validation 的 instance-aware online coverage、standard/broad full-policy、资源、CEC2017、CEC2022 benchmark factory/config 与工程问题 factory/constraint/config blockers；
2. 完成阶段 A，以预指定单次运行冻结科学端点、fold-specific SBS/Selectors 并枚举 replay plan；
3. 完成阶段 B 三次预定 future-path/FE=0 timing replays，保存逐次状态并完成 endpoint consistency/instability 检查，再生成 P7--P10；
4. BBOB 已见内部评价链闭合后，再按各自完整且未使用其 outcome 调整过模型的 suite 协议执行 P11；
5. 最后生成 P12，并按 Results → Discussion → Conclusion → Abstract 逆向补齐；
6. 在任一正式结果完成前，所有 TBD 保持占位，不用撤回数值临时填表。
