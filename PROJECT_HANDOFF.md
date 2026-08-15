# Decision-before-Feature 项目交接记录

本文用于跨对话同步唯一活动研究协议。开始新任务时依次阅读：

1. `AGENTS.md`
2. `README.md`
3. 本文件
4. `DEVELOPMENT_DECISIONS.md`
5. 与任务直接相关的 `docs/`

所有项目判断只使用当前仓库。不得从其他目录寻找旧代码、数据或结果。

## 1. 当前状态（2026-08-14）

当前只有研究协议与方法定义，没有可用于论文结论的正式结果。旧 trajectory 下游产物来自重建式 continuation、依赖 population 行号的 behavior、旧 query 构念、非完整嵌套标签或逐状态 threshold，已全部撤回证据资格。72 个正式 BBOB trajectory shards 尚未启动；BBOB-validation、CEC2017、CEC2022 和工程问题均无当前冻结协议下的正式评价结果。

本轮允许修改此前“冻结”的方案，并已修正以下会改变主结论的设计问题：

- 主 query 从含两个预处理恒量的 16 列缩减为 14 个有效描述符；
- Utility 从旧 raw-gap/线性相对时间口径改为截断 `log10_gap` 差与 `log10` runtime ratio；
- 把 Query 联合路径效用、不同预算路径的操作性增量、同预算 query-feature 预测增量分开；
- SBS 改为与主端点一致的 clipped `log10_gap`、run → static problem → fixed dimension stratum → function 等权聚合，不再使用平均 rank；
- 模型选择、threshold 和 Random calibration 改为完整 outer/inner grouped-by-function 链，所有上游组件按 fold 重算；历史 `family=bbob_fNNN` 仅是 function-ID group key；
- 决策评价由逐状态改为每 run 最多一次 first trigger；
- Random Analysis 改为 train-OOF 匹配 run call rate 和 first-trigger 时机；
- Time-only 主比较限制在相同 12 个 milestones，避免事件调度本身泄漏 behavior 信息；
- Always Query 与 FE=0 Traditional AAS 分开；self-thresholded Behavior-only 与 matched-trigger 诊断分开；
- query sample 的真实 FE、直接找到更优点和 observed first hit 纳入 Query operational endpoint；
- Utility 数据生成改为 Selector 冻结前后的两阶段执行，不能从一次 action matrix 直接得到带真实三次计时的最终标签。

协议文档已统一到上述口径，但源码、配置、数据生产器和资源排期仍需逐项核对。任何正式运行都必须等第 10 节 blocker 关闭后开始。

## 2. 研究对象与结论边界

研究问题是：在连续黑盒优化的预定义在线状态分布上，执行一个固定 landscape-descriptor query 并调用其下游 Selector，相比原生继续 fold-specific SBS，是否改善成本调整后的终点性能；仅使用 query 前算法无关行为，能否改善这个调用决策。

第一篇论文不设计新优化器，也不把 Selection Reference 作为算法选择贡献。`Search Maturity` 是 Behavior 的确定性基函数，不是已验证的潜在阶段或因果中介。共享状态上的多动作续跑是配对性能评价，不作因果主张。逐状态候选中的最小 loss 称为 `best observed action`；VBS 是静态 problem-level hindsight reference，二者不得混用。VBS 在每个 `function × instance × dimension` problem 内先对各算法的完整预算 clipped `log10_gap` 跨 seeds 取均值，再选均值最低算法并用该算法的逐 seed paired outcomes 汇总；不得逐 seed 事后挑最小算法。

当前贡献只覆盖三个预定义 query 配置。即使三档结果一致，也只能支持配置稳健性，不能推出“所有 ELA 都值得或不值得”。

## 3. 主数据协议

BBOB train functions：

```text
1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 15, 16, 17, 18, 20, 21, 22, 23
```

BBOB-validation functions：

```text
5, 9, 13, 14, 19, 24
```

共同设置：

```text
dimensions: 10, 20, 40
instances: 1, 2, 3
optimizer seeds: 1 ... 30
algorithms: de, pso, cmaes, shade
population_size: 40
FE_total: 1000 * dimension
sampling_protocol: phase1_dynamic_budget_event_v1
monitor_grid: 0.20--0.60, step 0.01
budget_milestones: 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.60
states_per_run: 12--18
```

每个 shard 同时生成 decision trajectory 和独立的 `FE=FE_total` 终值表。状态连接使用整数 `FE`，不用浮点 ratio 作键。正式主 population 只包含 `prefix_algorithm == default_algorithm == SBS_fold` 的状态；全 prefix 数据只作预定义稳健性分析。

Behavior 的 w02/w05/w10 从逐次完整原生 optimizer update 历史取 anchor。名义 FE 不在 update 边界时，选择不晚于目标的最近完整 update；实际窗口不小于名义窗口且偏差小于一次 population update。实际 FE、ratio 和 native-update 数只作 metadata，不进入 Decision 输入。

## 4. Query 配置与样本端点

| query | 角色 | sample | FE | 有效列数 |
| --- | --- | --- | ---: | ---: |
| `descriptor_cheap_invariant` | 主分析 | `lhs_50d` | 5% 总 FE | 14 |
| `pflacco_standard_invariant` | 配置稳健性 | 与主 query 共享 `lhs_50d` | 5% 总 FE | 37 |
| `pflacco_broad_invariant` | 配置稳健性 | 独立 `lhs_100d` | 10% 总 FE | 52 |

统一预处理为 `unit_cube_x__median_iqr_y_v1`。`descriptor_y_median` 和 `descriptor_y_iqr` 在该预处理下恒为 0/1，已从活动 whitelist 删除；query ID、样本、FE 和 action outcomes 不变。

Query sample 是真实 objective evaluations，但不并入 optimizer population。Query operational terminal best、`target_hit_observed` 和 ERT 必须同时考虑 sample best 与 continuation；observed first hit 可发生在 sample 内，并保存 `query_first_hit_offset`。`endpoint_success` 另与 path completion 联合定义。另存 continuation-only gap 和 `query_sample_best_contribution_log10_gap`。Selector regret 与同预算 query-feature 预测诊断只使用 continuation-only outcomes。

Reservoir 仅是零额外 FE 的 trajectory-query 诊断，唯一合同为：

```text
query_id=descriptor_cheap_invariant
query_source_mode=trajectory_reservoir_zero_extra_fe
query_protocol=trajectory_query_reservoir_v1
```

它不得并入独立 LHS 主 estimand。

## 5. Action matrices、Selector 与动作关系

每个共享状态的唯一动作集合为 `continue_current` 加其余三个 portfolio algorithms。同算法恢复完整 optimizer state 原生继续；跨算法只执行一次 `population_transfer_initialization`。必须分别生成：

- Query-adjusted 四动作矩阵：remaining FE 为 `FE_total-FE_prefix-FE_query`；
- Behavior-only full-budget 四动作矩阵：remaining FE 为 `FE_total-FE_prefix`；
- 主 cheap 的 FE=0 pre-run AAS 四算法 outcomes。

Selection Reference 固定为多输出 `RandomForestRegressor`，目标变换为：

```text
(L_a - min_b L_b) / max(max_b L_b - min_b L_b, 1e-12)
```

Query Selector 输入 query features、算法无关 Behavior 和连续 remaining-budget ratio；Behavior-only Selector 不读 query features；`query_adjusted_state_only_selector` 在 Query-adjusted 同一动作矩阵上只读 Behavior 与预算；FE=0 Selector 只读 query features 与静态预算信息。

Selection Reference、Utility、Decision dataset 与在线输出均保存：

```text
selected_equals_default
selected_equals_prefix
handoff_required
handoff_type
```

并逐行满足：

```text
handoff_required = not selected_equals_prefix
handoff_required = (handoff_type == population_transfer_initialization)
```

不得生成 `label_source`，也不得用 `same_algorithm/changed_algorithm` 替代这些显式关系。

## 6. Utility 与三个不同问题

对 `k in {skip, query, behavior_only}`：

```text
ell_k = log10(min(max(g_k, 1e-12), 1e20))
runtime_k_censored_rep_j = runtime_k_raw_rep_j                     if completed
                           max(runtime_k_raw_rep_j, role_timeout)  otherwise
T_k = median(runtime_k_censored_rep_1,
             runtime_k_censored_rep_2,
             runtime_k_censored_rep_3)
```

`runtime_k_raw_observed_median` 另存但只作诊断；主 Utility 不读取 raw median。旧 `failure_worst_case` 只可作为同一 censored 值的兼容别名。

主标签：

```text
U_query_joint = (ell_skip - ell_query)
              - lambda_T * (log10(T_query) - log10(T_skip))

U_behavior_only_full_budget = (ell_skip - ell_behavior_only)
                            - lambda_T * (log10(T_behavior_only) - log10(T_skip))

query_operational_increment = (ell_behavior_only - ell_query)
                            - lambda_T * (log10(T_query) - log10(T_behavior_only))
                            = U_query_joint - U_behavior_only_full_budget
```

主字段为 `u_query_joint_lamT_1`，主设置为 `lambda_time=1, lambda_memory=0`。`lambda_time={0,0.25,0.5,1,2}` 全部作为预定义敏感性分析报告。

三个量不能混称：

- `u_query_joint_lamT_1`：Query/full-Selector operational path 相对 native SBS 的联合路径净差；
- `query_operational_increment_lamT_1`：Query 与 full-budget Behavior-only 两条不同预算 operational paths 的净差，包含 query FE、runtime、sample best、较短 continuation budget 与 Selector 差异；它不是纯信息效应或因果 estimand；
- `query_feature_predictive_increment_log10_gap`：同一 Query-adjusted outcomes 上，state-only 与 full Query Selector 的 OOF selected continuation-only `log10_gap` 差；不新增 action loss、不含 sample best、不扣 query cost，只是预测诊断。

`I_q` 必须同时在全 eligible states 与 Proposed first-trigger states 报告。若 `U_query_joint>0` 而同一范围的 `I_q<=0`，只能支持联合路径优于 SBS，不能支持 query acquisition 优于 full-budget Behavior-only。本轮不新增 matched-acquisition 第四路径，故 descriptors、sampling、sample best、FE、runtime 与预算差的纯操作性归因仍不可识别。

每个 problem/sample design 当前只有一个 problem-keyed LHS realization；正式 estimand 因而条件于该固定 realization。若未在看结果前加入多个 sampling replicates，不得泛化到 LHS sampling randomness。standard--broad 同时改变 representation、sample size、sample realization 与 action budget，只能比较整体 query configuration。

主 Utility 的 runtime 是从同一 decision state 上 query/no-query 分支确定之后到 terminal 的条件未来路径时间。共享 prefix、该机会的 Behavior extraction、Decision inference 与 threshold comparison 是分支前共同成本，不进入 $T_k$ 差；Query/Behavior-only 各自只计真实 Selector inference 与后续动作特异增量，Skip 计 native continuation。在线政策另报告从 FE=0 到 terminal 的完整 run wall-clock并包含共同成本；它分别保存 `runtime_full_run_wall_clock_raw_observed_repetitions`、`runtime_full_run_wall_clock_raw_observed_median`、`runtime_full_run_wall_clock_censored_repetitions` 与作为 censored median 的 `runtime_full_run_wall_clock_median`。二者回答不同问题，不得都命名为同一个 `complete_path_runtime`，也不得用 FE=0 全程时间替代状态条件 Utility 的分母或反之。

Stage-A/online 路径统一保存 `observed_first_hit_FE`，并满足：`target_hit_observed := observed_first_hit_FE != null`；`target_hit_before_failure := target_hit_observed and not path_completed`；`path_completed := status == completed`；`endpoint_success := target_hit_observed and path_completed`。兼容 `first_hit_FE/success` 只可分别别名为 `observed_first_hit_FE/target_hit_observed`；正式 ERT 使用 `target_hit_observed`。

每条 Stage-B path 分开保存 `timing_replay_path_identity_consistent_*`、`completed_timing_replay_outcomes_internally_consistent_*` 与 `stage_a_to_completed_timing_replays_consistent_*`；后二者不适用时为 null/not_evaluable。`timing_replay_status_instability_*` 与 `stage_a_stage_b_completion_status_instability_*` 也必须分开，不能再由一个布尔字段兼任。

## 7. Decision Model、输入组与完整嵌套

活动候选固定为：

```text
LDA
Logistic Regression
Ridge
```

主模型按 B3 上 BBOB-train grouped-by-function outer OOF 的 run-level first-trigger mean `u_query_joint_lamT_1` 选择；并列顺序为 LDA → Logistic Regression → Ridge。train outer OOF 仅用于候选选择和开发诊断，不是选择后 procedure 的无偏估计。BBOB-validation 已被历史开发结果查看，现只可承担 post-development 内部评价，不能称 untouched/confirmatory；selected procedure 与 RQ2 milestone-only B3--T0 在该集合上的结果必须据此降级，各真正 external suite 单独评价。三候选两两 OOF 对比只作选模诊断。AUROC、Average Precision、Spearman 是辅助指标，连续 Utility RMSE 只对 Ridge 定义。后续链仍禁止 BBOB-validation 参与 preprocessing、选模、特征组选择、threshold 或 Random calibration，但该限制不能恢复其未查看资格。

正式输入组：

```text
T0 / B1 / B2 / B2+Motion / B2+Maturity / B3
1  / 19 / 25 / 28        / 28          / 31
```

`all_candidates -> B3`；`primary_with_maturity -> B2+Maturity`。B2+Motion 与 B2+Maturity 是兄弟组。模型名先由 B3 冻结，再用于全部六组；各组不重新选更有利的模型。

主 Time-only 比较为 `milestone_only_T0`：T0 与 B3 都只使用相同 12 个 milestones。完整 dynamic schedule 上的 `schedule_conditioned_T0` 只能作为敏感性分析，因为事件机会本身由 Behavior 条件产生。

完整嵌套要求：

1. 每个 outer fold 只用 outer-fit functions 重算 `SBS_outer`、Query/Behavior-only/FE=0 Selectors、Utility、Decision preprocessing/model、first-trigger threshold；
2. 每个 inner fold只用 inner-fit functions 重算 `SBS_inner`、全部 Selectors、Utility 与 Decision；
3. outer holdout 只接受 outer-fit components 的一次评价；
4. 完整 BBOB-train 的 `oof_utility_first_trigger` threshold 与 Random calibration 也必须来自 fold-specific 上游 OOF；
5. 选定模型后才在完整 train 重拟合最终 SBS、Selectors 和 Decision，再评价已见 BBOB-validation 内部集合与各 external suites。

不得先用完整 train 生成固定 Utility 表，再只对 Decision 做 nested CV。

主 fit 已在活动协议中改为 `cluster_balanced_fit`：每个 fit fold 内依次使 function、固定 dimension stratum、static problem 与 optimizer run 等权，再把一个 run 的权重均分给其合格 states，并将 fold 内 row weights 归一化到均值 1。权重只由 fit-fold grouping 决定，不读取 holdout 或 outcome 方向。旧 `sample_weight=1` 降为 `row_weighted_fit` sensitivity。当前 Decision 与 Selector estimator 尚未接入该权重，因此这是正式 blocker；在 wiring 闭合前不能冻结模型，也不能把协议口径误写成当前实现已经完成。

## 8. Baseline 与统计单位

预设九个角色、八个不重复 outcome：

1. Never Query；
2. SBS（主 population 中与 Never Query 共用一个 native outcome）；
3. Always Query（第一个在线合格机会）；
4. matched-rate Random Analysis；
5. `pre_run_aas_fe0`：FE=0、query-only、sample-isolated pre-run AAS；
6. VBS；
7. `milestone_only_T0`；
8. `self_thresholded_behavior_only`；
9. Proposed Decision-before-Feature。

`matched_trigger_behavior_only` 只在 Proposed 首次触发的同一状态计算 `query_operational_increment`，不是 self-thresholded policy，也不进入 baseline 或 Holm family。

Random Analysis 只从完整 train 上游 OOF Proposed 预测冻结 run-level call rate 与已调用 run 的 first-trigger `FE_ratio` 经验分布。每 run 开始预抽是否调用和目标 ratio，在首个不早于目标的合格机会触发；不用 constant hazard，也不从事后完整机会列表均匀抽 state。30 个随机流先在同一科学 run 内平均，再进入 run → static problem → fixed dimension stratum → function 聚合。

`pre_run_aas_fe0` 的 query sample 不用于 optimizer population 初始化；它及所有 portfolio 结论仅条件于仓内冻结实现、参数与 `population_size=40`。

FE=0 AAS 关系字段：`prefix_algorithm=selected_algorithm`（只作显式关系记账）、`selected_equals_prefix=true`、`handoff_required=false`、`handoff_type=fresh_optimizer_initialization`；`default_algorithm=no_query_algorithm=SBS_fold`。

RQ1 状态 estimand 的聚合为 state → run → static problem → fixed dimension stratum → function；政策端点从 run 层开始同序聚合。BBOB-validation 的内部评价 estimand 是 F5/F9/F13/F14/F19/F24、固定 dimensions、instances 1/2/3、static problems 与 query realizations 上的等权有限集均值。主条件 bootstrap 固定保留 6 functions、全部 dimensions 与全部 static problems，只在每个固定 static problem 内配对重抽 optimizer seeds；RQ1 对每个抽中 seed/run 保留完整有序 states。不得在主 bootstrap 中重抽 static problem；function resampling 只作单独命名的函数组成敏感性。该集合已见，所有这些输出均不得称为 untouched、确认性或推断到 function/transformed-instance 超总体。当前 3 instances × 30 seeds 没有仓内精度依据；不得据此声称功效充分，CEC2022/工程问题的重复数须在查看其 outcome 前完成精度设计并冻结。

ERT 另用专用 paired hierarchical ratio bootstrap，仍固定 functions、dimensions 与全部 static problems，只在每个固定 problem 内联合配对重抽 optimizer runs。每个 stratum/replicate 保留 finite、$+\infty$（仅 treatment 零命中）、$-\infty$（仅 reference 零命中）与 undefined（双方零命中，或聚合时同时含两种无穷）质量，不得静默删除。undefined mass 按 `conservative_two_tail_allocation_on_extended_real_line_v1` 保守分配到两端；达到单侧 $\alpha/2$ 时区间为 $[-\infty,+\infty]$，全部 undefined 时区间界为 undefined。`interval_established` 依据观测 contrast 和按该质量规则得到的分位点是否有定义；无界区间仍可 established，不能因任意一次 replicate 出现零命中就自动失败。报告必须给出 `interval_status`、`interval_unbounded`、finite/$+\infty$/$-\infty$/undefined mass、各类零命中计数与 defined replicate 数。

Utility/gap/runtime/rate 边界只称项目内预设 operational tolerance。主条件 95% CI 仅逐项描述；预列 family 的 Bonferroni 描述性区间按每项双侧 level `1-0.05/m` 提供 family-wise 95% coverage，也不形成确认性等价声明。该集合 `n=6` 的双侧 exact sign-flip raw p 最小为 0.03125；RQ3 与 RQ5 各自六 contrast Holm family 的最小 adjusted p 均为 0.1875，因此在 0.05 下不能拒绝。RQ4 按 suite/endpoint 分开，不把四个 suites 组成一个 Holm family。p 值只作内部透明描述。失败敏感性须同时报告 attempted denominator 与 complete pairs，并按 gap floor/cap、`target_hit_observed` 1/0、ERT adverse 未命中的 full planned budget、ERT favorable 命中的最早可行 first-hit FE、runtime timeout/该 suite complete pairs 最小正值双向赋值，再重算 Utility；first-hit FE 或最小正 runtime 不可重建时 sensitivity 为 undefined。方向或区间相对 tolerance 的位置改变、coverage<95% 或 sensitivity undefined，均使结论未建立。细则见 `DEVELOPMENT_DECISIONS.md` 第 17.1--17.2 节；BBOB-validation 在两处均已明确降级为已见内部有限集评价。

## 9. 两阶段正式数据生成与资源边界

Utility 不能在一次 action-matrix 生成后直接完成，正式顺序是：

```text
阶段 A
trajectory/final performance
-> behavior + query samples/features
-> Query-adjusted、Behavior-only、FE=0 action outcomes 各运行一次
-> fold-specific SBS 与 Selectors
-> 持久化 outer/inner/full-train Selectors，写入 fold-role/family artifact 路由
-> 枚举 outer/inner/full-train selected replay plan

阶段 B
按 replay plan 从对应 decision state 恢复完整状态
-> selected Skip / Query / Behavior-only future paths 各真实执行 3 次
-> 每个需完整报告的 query 配置从 FE=0 执行其独立 policy paths
-> 保存 raw/censored 时序、censored future-path 中位时间、三类一致性、
   两类 instability 与 FE=0 policy wall-clock
-> 构造 Utility、Decision labels、threshold、baselines 和评价
```

12 个必选 milestones 的平均 prefix ratio 是 0.35，mandatory-only remaining ratio 是 0.65；相对旧的全在 0.60B 假设，remaining-path multiplier 为 `0.65/0.40=1.625`。当前 mandatory-only 算术情景为：

| Mandatory-only 情景口径 | 三档 query | 仅 main cheap |
| --- | ---: | ---: |
| 阶段 A 跨 matrices 共享：共享 query artifacts，Skip 与 Behavior `continue_current` 只运行一次；不复用基础 trajectory | 53.434836B FE | 37.467612B FE |
| 阶段 A 进一步复用基础 trajectory（须逐行证明端点同义） | 48.717396B FE | 32.750172B FE |
| 阶段 B：22-role train/1-role validation selected future paths ×3，加主 pre-run AAS | 296.7678B FE | 178.24212B FE |
| 阶段 A+B：仅跨 matrices 共享 | 350.202636B FE | 215.709732B FE |
| 阶段 A+B：进一步复用基础 trajectory | 345.485196B FE | 210.992292B FE |

表中“三档”只含三种 query 的 state-level Query paths与当前主 cheap pre-run AAS，不含 standard/broad 各自完整 FE=0 policy/baseline paths。所有数字也只对应 12 个 mandatory states 且无 event/failure，不是完整点估计或资源可行性证明。当前 Stage-A producer 在 Query-adjusted 和 Behavior-only CLI 中均执行 `skip + 4 actions`，main cheap 因而比“仅跨 matrices 共享”多 9.43488B，当前 mandatory-only 实现情景为 Stage A 46.902492B、阶段 A+B 225.144612B。三档当前实现量等待实际调用图枚举，不自行猜测。event-only 增量必须按其实际 `FE_prefix`、fold role、path 与 selected action 从物化 plan 求和。进一步复用基础 trajectory 还需逐行证明 terminal gap、observed hit、completion、planned/effective FE 与 status 同义，不能只凭算法名相同扣减成本。

完整在线政策另计：每个 base tuple 有 7 条固定政策和 30 条 matched-rate Random，每条执行 1 次 Stage-A 科学运行与 3 次 Stage-B timing replay，共 148 个 full runs。当前 CEC2017 配置对应约 11.5884B planned FE；若已见 BBOB-validation 内部评价集全部 6 functions × 3 instances × 3 dimensions × 30 seeds 使用相同集合，则约 5.5944B FE。后者当前不可执行，因为 online evaluator 只接受 CEC、固定 `instance=1`，且 query feature/sample 键不含 instance。standard/broad 各自完整 online paths 也未实现。上述 online 量都尚未并入表内 mandatory-only state replay 情景。

## 10. 正式运行 blockers

以下任一项未关闭，都不得启动对应正式运行：

1. replay planner 已有枚举能力，但尚无已核对的 offline decision-state-to-terminal runner；
2. 尚未物化并实测 outer/inner/full-train fold-role selected replay plan，mandatory/event multiplicity 与 FE 未闭合；
3. grouped-by-function cross-fitted Selector 子模型尚未持久化，replay plan 也没有历史 `fold-role/family` key 到对应 Selector artifact 的路由，Stage-B 因而不能真实计入相应 Selector inference；
4. online evaluator 尚不支持已见 BBOB-validation 内部评价集的 3 个 instances，因而不能生成其 full-policy endpoint；训练/评价入口也尚未直接按 validation config 强制核对 F5/F9/F13/F14/F19/F24、instances 1/2/3、10/20/40D、seeds 1--30 与完整 problem/state coverage。两项实现后仍不得称该集合为未查看确认集；
5. 尚无同一正式硬件上的真实 evaluator timing、线程/缓存/内存测量和可承受资源排期；
6. 72 个 BBOB trajectory/final-performance pair 尚未按当前协议生成并通过整数 FE、完整状态、native-update window 和 permutation-invariance 检查；
7. 当前 `configs/phase1_cec2017_test.yaml` 为 F1--F29，即包含 F2、排除 F30；仓库内没有依据证明它符合实际 CEC2017 实现与正式 benchmark 口径，必须在查看 policy outcomes 前核对并冻结；
8. standard/broad 尚无各自完整 Selector、Decision/threshold、baseline、Stage-B future paths 与 FE=0 full-policy 实现，当前 main-only online evaluator 不能支持“三档完整稳健性”口径；
9. CEC2022 benchmark factory 与正式配置尚未实现；工程问题 factory、约束处理与正式配置尚未实现；二者的 functions/problems、dimensions、budgets、repeats、reference values、gap floor/cap、success target、三种 timeout、observed-first-hit 和 constraint rule 均未闭合；
10. Stage-A producer 尚未在“跨 matrices 共享”“进一步复用基础 trajectory”或“保持当前未复用调用图”之间闭合实现与成本；
11. raw/censored timing、observed hit/completion/endpoint success、三类一致性与两类 instability 的 producer/consumer 仍须按当前字段规范实现；
12. cluster-balanced 主 fit 与 row-weighted sensitivity 的 estimator wiring 尚未实现；所有活动源码、配置、字段名和输出路径仍须逐项证明与本交接记录一致，且不读取撤回 artifact；
13. BBOB/CEC2017 的 3 instances × 30 seeds 或 30 seeds 只是一项固定开发设计，仓内没有 precision/power 依据；不得宣称样本量充分。CEC2022 与工程问题必须在查看其 outcome 前冻结 endpoint-specific precision target、重复数确定方法与最终 repeats。
14. 基础 trajectory producer 遇单 run 异常时仍可能终止整个 shard，`final_performance` 也尚未完整物化 status、`path_completed`、planned/effective FE、`observed_first_hit_FE`、`target_hit_observed`、`target_hit_before_failure`、`endpoint_success` 与 failure context；修复前不能形成 attempted denominator 或失败敏感性输入；
15. query sample producer/feature loader 尚不能把 failed/timed-out sample 作为带 status、实际 FE、observed hit 与 failure context 的正式行保留下来，因而可能把 query failure 静默变成缺行；
16. Decision inference/score 异常尚未统一落为该机会 No-query；query 已触发后的 feature/Selector 异常也尚未统一保留已消耗 query FE 与时间，并按 query-adjusted remaining budget 执行原生 `continue_current`；两条 fallback 均须输出显式 status/failure context；
17. ERT 专用扩展实数 bootstrap 尚未接入 suite-level attempted denominator/coverage、双向 failure sensitivity 与报告 consumer。consumer 必须保留 finite/$+\infty$/$-\infty$/undefined mass、零命中计数、defined replicate 数、`interval_status` 与 `interval_unbounded`，不得只读取 complete-pair finite ratio 或把任一次零命中等同于 `interval_established=false`。
18. `dimension_stratified_T0` 目前只有协议定义，训练、threshold、first-trigger 汇总与输出 consumer 尚未实现；闭合前不得声称 Behavior 的预测信息不能由已评估 dimension strata 解释；
19. CEC2022 与工程集合尚未冻结 suite-specific 顶层有限集单位、固定 strata、权重和相应区间 consumer。CEC suite 以 function 为顶层单位；工程集合以预先命名的 engineering problem 为顶层单位，不能复用 BBOB 字段名掩盖不同层级。

BBOB train/validation 与 CEC2017 已冻结：

```text
failure_loss_cap = 1e20
log10_gap_floor = 1e-12
log10_gap_cap = 1e20
success_gap_target = 1e-8
action_timeout_seconds = 3600
timing_replay_timeout_seconds = 3600
policy_timeout_seconds = 3600
first_hit_recording = every_objective_evaluation
timing_repetitions = 3
timing_order_protocol = cyclic_complete_path_v1
```

正式运行可按完整科学子矩阵分阶段：main cheap BBOB → standard/broad robustness → CEC2017 → factory 与配置均闭合后的 CEC2022/工程问题。阶段结果必须明确限定范围，不能冒充全协议结论；已见 BBOB-validation 只能称内部评价。

## 11. 结果与活动目录

所有旧数值仅用于说明撤回范围，不得写入摘要、主表、主图或结论。活动目录契约：

```text
results/phase1_refined_sampling/
results/landscape_queries/samples/{sample_design_id}/{split}/
results/landscape_queries/features/{query_id}/{split}/
results/selection_reference/{query_id}/
results/utility_labels/{query_id}/
results/decision/{query_id}/
```

`results/ela/`、旧 query ID、旧 Utility 字段、缺少 query 协议字段或来自 archive 的 artifact 都不得进入活动链。当前没有可直接复用的正式模型或结论数值。

## 12. 下一步

先关闭方法实现和资源 blocker，不直接启动 72 shards。建议顺序：

1. 对照唯一活动协议只读检查源码、配置和输出字段，列出尚未实现项；
2. 持久化并路由 fold-specific Selector artifacts，使用已有 replay planner 物化并核对正式 plan，实现 decision-state-to-terminal runner；
3. 实现 raw/censored timing、observed hit/completion/endpoint success、三类一致性与两类 instability 字段，再枚举 mandatory/event replay multiplicity 与 Stage-A 实际调用图；
4. 修复 trajectory/final-performance 与 query sample 的逐 run failure materialization，并实现 Decision 异常 → No-query、query 后 feature/Selector 异常 → 保留 query 成本并 `continue_current` 的 fallback；
5. 接入固定-static-problem 的主条件 bootstrap 与 ERT 扩展实数报告 consumer，逐 suite 输出 attempted coverage、双向 failure sensitivity 和 finite/$+\infty$/$-\infty$/undefined mass；
6. 实现已见 BBOB-validation 的 instance-aware online endpoint及 standard/broad 完整政策路径；这些产物仍按各自证据等级解释；
7. 在不查看新 policy outcomes 的前提下闭合 CEC2017 F2/F30；为 CEC2022/工程问题先冻结 endpoint-specific precision target、repeats 确定方法与最终重复数，再实现/冻结 benchmark factory/config 与工程问题 factory/constraint/config；
8. blocker 全部关闭后，才按阶段 A → Selector/replay plan → 阶段 B → Utility 的顺序运行 main cheap BBOB。

可直接复制的下一步 prompt：

```text
请只使用当前 Decision-before-Feature 项目，先阅读 AGENTS.md、README.md、PROJECT_HANDOFF.md、DEVELOPMENT_DECISIONS.md 和相关 docs。不要启动正式实验。请持久化并路由 outer/inner/full-train 的 fold-specific SBS、Query Selector、Behavior-only Selector 与 FE=0 Selector artifacts，物化并核对 replay plan，实现 decision-state-to-terminal runner，并让 online evaluator 支持已见 BBOB-validation 内部评价集的全部 instances及 standard/broad 完整政策。逐项检查：14 维 descriptor_cheap_invariant；两套 action budgets；oof_utility_first_trigger threshold 与 matched-rate Random calibration；Stage-A observed hit/completion/endpoint success；Stage-B raw/censored timing、三类一致性与两类 instability；future-path timing 与 FE=0 policy wall-clock 分离；六组特征；九角色八 outcome；cluster-balanced 主 fit 与 row-weighted sensitivity；CEC2017 F2/F30；CEC2022 benchmark factory/config；工程问题 factory/constraint/config。修复 trajectory/final-performance 和 query sample 的逐 run failure materialization；实现 Decision exception → No-query，以及 query 后 feature/Selector exception → 保留已消耗 query FE/时间并按 query-adjusted budget `continue_current`。主 bootstrap 固定所有 static problems、只在每个 problem 内配对重抽 optimizer runs，function resampling 仅作函数组成敏感性；接入 ERT suite-level report consumer，保留 finite/+inf/-inf/undefined mass、零命中计数、defined replicate 数、interval status、attempted coverage 与双向 failure sensitivity，不能因任意一次零命中自动判定区间未建立。对 CEC2022/工程问题必须在查看 outcome 前冻结 endpoint-specific precision target、repeats 确定方法与最终重复数。另按 12 个 mandatory milestones 的平均 prefix=0.35 与实际 event states 枚举 Stage-A/Stage-B 调用图，在“只跨 matrices 共享 Skip/Behavior continue_current”“经逐行证明后进一步复用基础 trajectory”或“保持未复用 producer”三种实现中明确选择并重算资源。Runner、实测 plan、artifact 路由、failure/fallback、推断 consumer、字段契约、precision/repeats 与资源未闭合前，不得生成最终 Utility 或启动 72 个正式 shards。
```
