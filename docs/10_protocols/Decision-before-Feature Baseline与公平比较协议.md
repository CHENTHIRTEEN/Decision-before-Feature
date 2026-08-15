# Decision-before-Feature Baseline 与公平比较协议

> 活动协议（2026-08-14）：本文件直接替代旧的逐机会 `p=0.5` Random、`Always Query = Traditional AAS`、逐状态政策汇总和完整动态 schedule 上的主 T0 口径。本文只定义待实现与待运行的方案，不声称 baseline 已完成或已有结果。

## 1. 比较单位与 first-trigger 规则

所有顺序策略以一条 optimization trajectory 为决策单位。对每个 run，合格机会按整数 `FE`、再按 `decision_opportunity_index` 排序。策略最多触发一次；首次触发后的状态在该策略下不可达，不进入该 run 的 threshold、call rate、precision、utility capture、policy Utility 或任何模型比较。未触发 run 的 joint decision utility 为 0。

所有 paired comparison 必须共享相同 problem、dimension、instance、optimizer seed、total FE、portfolio、bounds、停止规则和可适用的机会集合。主 Query policy 的 `FE_query` 通过减少 continuation budget 体现，不重复扣除。各策略只承担其实际使用的 behavior、Decision、query、Selector、handoff 与 optimization wall-clock。

Never Query 与 SBS 在主 population 中是同一个 native outcome，只计一次；VBS 只作静态 hindsight reference。其余 baseline 都产生独立 outcome，不得用别名或复制其他策略的运行结果代替。

## 2. Required baselines

### 2.1 Never Query / SBS

SBS 只由相应 fit functions、`FE=FE_total` 的 `final_performance.parquet` 冻结。对每个算法按 run → static problem（function × dimension × instance）→ fixed dimension stratum → function 等权聚合配置截断后的 `log10_gap`，选择均值最低者；并列按 `de,pso,cmaes,shade`。这与主性能端点和 function 顶层权重一致，不用平均 rank 丢弃效应量。outer/inner/full-train SBS 分别只由对应 fit functions 重算。

主 population 的 prefix 与 default 都是对应 SBS。Never Query 从完整 checkpoint 原生继续到预算结束，不执行 Decision、query、Selector 或 handoff。它的 query decision utility 定义为 0；SBS 作为算法选择角色时该指标为 N/A。共享 outcome 只进入统计一次。

### 2.2 Always Query

Always Query 在其机会集合的第一个在线合格 state 触发一次固定 query，调用与 Proposed 相同的 Query Selector，并按选中动作原生继续或执行一次 population-transfer initialization。它不是 FE=0 的 pre-run AAS。

### 2.3 Matched-call-rate Random Analysis

`matched_rate_random` 只使用完整 BBOB-train 端到端 function-OOF Proposed first-trigger predictions 冻结 run-level 调用率 `r_call` 与已调用 runs 的 first-trigger `FE_ratio` 经验分布。validation 和外部 suite 不参与匹配。

每个 run 开始时先以 `r_call` 做一次 Bernoulli draw；若决定调用，再从冻结的 train-OOF trigger-FE 经验分布抽一个目标 `FE_ratio`，并在当前 run 第一个不早于该目标的合格在线机会触发。若不存在该机会则该 run 不触发。30 个重复使用显式整数 stream code 和 `numpy.random.SeedSequence`；同一科学 run 的 30 个 Random outcomes 必须先平均成一个 run-level outcome，再进入 problem/function 聚合和配对检验，不能视为 30 个独立统计复制。该规则不需要预知当前 run 未来共有多少事件机会。不得逐机会使用 `p=0.5`、从完整机会集合事后均匀抽 state、重复 Bernoulli 直到成功，或根据 evaluation Utility/call rate/timing 重新匹配。

训练分布与各 evaluation suite 分别报告实际 run-level call rate 和 trigger-FE 分布；“matched”只表示 calibration 来自训练 OOF 的调用率和触发时机分布，不保证外部 suite 的 realized rate 完全相等。

### 2.4 Traditional pre-run AAS

`pre_run_aas_fe0` 在任何 portfolio optimizer update 之前、`FE=0` 执行当前固定 query。它是 query-only、sample-isolated 的 pre-run AAS：输入只含 query descriptors 与静态 remaining-budget ratio，不含 trajectory Behavior。选中算法从其原生 initialization 开始，以 `B-FE_query` 运行；query sample 不用于初始化或扩充 optimizer population。其结论只条件于仓内固定 portfolio 实现、冻结参数与 `population_size=40`，不能代表所有 Traditional AAS。

该 FE=0 outcome 的关系字段固定为 `prefix_algorithm=selected_algorithm`（仅作显式关系记账）、`selected_equals_prefix=true`、`handoff_required=false`、`handoff_type=fresh_optimizer_initialization`；`default_algorithm=no_query_algorithm=SBS_fold`。不得将 fresh initialization 记为 population transfer；其组件时间单列 `runtime_fresh_initialization`，`runtime_handoff` 只表示已有 population 的 transfer initialization。

该 baseline 需要 FE=0 query sample、四个初始算法的等预算 outcome 与独立 train-only Selector。它不复用 Always Query 的 statewise outcome，也不使用 population transfer。所有 SBS、Selector 与模型拟合仍在相应 outer-fit functions 内完成，validation/external 只加载 full-train frozen component。

### 2.5 VBS

VBS 是不可部署的静态 problem-level hindsight reference。对每个 `problem = function × instance × dimension`，先将每个算法的完整预算 raw gap 按 suite 端点截断并转为 `log10_gap`，再跨 optimizer seeds 取算术均值；选择均值最低算法，并列按 `de,pso,cmaes,shade`。随后只用这个已选算法在该 problem 上的逐 seed paired outcomes 进入 run → static problem → fixed dimension stratum → function 汇总。VBS 不得逐 seed 选择最小算法；共享 state 上四个 continuation actions 的最小 loss 称为 `best observed action`，只用于 selector regret 与 action-range 诊断，也不能替代 VBS。由于 VBS 是静态 problem-level 参照，它不保证对允许状态条件切换的政策构成逐 run 数值上界。

### 2.6 Milestone-only Time Controller

`milestone_only_T0` 输入严格为 `X={FE_ratio}`，只在 12 个固定预算 milestones 拟合、threshold 和评价。其模型名由 B3 的 RQ2 选择冻结，preprocessing 与 first-trigger threshold 只用对应 outer-fit/完整-train OOF 信息。与 T0 直接比较的 B3 也必须限制在同一 milestone rows。

事件机会由 Behavior 条件产生；完整 dynamic schedule 上的 T0 只能记为 `schedule_conditioned_T0` sensitivity，不能用于“Behavior 超过纯时间”的主结论。

另设 `dimension_stratified_T0`：在 10D/20D/40D 内分别执行同一 train-only 拟合与 first-trigger threshold 过程，每层输入仍严格为 `FE_ratio`。它控制调用前已知的 dimension/绝对预算上下文，不把 dimension 加入主 Decision X。主 cheap 的 planned query-cost ratio 若在所有维度恒为 0.05，则没有可估计变化，只作为协议常数报告。RQ2 的强表述要求 B3 同时相对 milestone-only T0 与该分层对照有稳定效应。

### 2.7 Self-thresholded Behavior-only Selection

`self_thresholded_behavior_only` 不执行 query。它使用与主 B3 相同的冻结模型家族，但单独拟合 `U_behavior_only_full_budget`，并从自己的 train-only inner-OOF scores 冻结 first-trigger threshold。在其首次触发 state，Behavior-only Selector 从 full-budget 四动作矩阵中选择动作；所有动作使用 `B-FE_prefix`，所以 query call rate 和 query FE 均为 0，trigger rate 与 handoff rate仍报告。

`matched_trigger_behavior_only` 是另一个用途：它在 Proposed 首次 query trigger 的同一 state 执行 Behavior-only action，用来计算 `query_operational_increment`。它不是 self-thresholded policy，不进入主 baseline 名单或 Holm family。两者可复用同一 full-budget action matrix 和 Behavior-only Selector，但不能复用 threshold、trigger 分布或政策名称。

### 2.8 Decision-before-Feature

Proposed 使用 B3 pre-query Behavior 与 `oof_utility_first_trigger` threshold。score 首次超过 threshold 时执行一次固定 query，再调用 Query Selector；未触发则 native SBS continuation。它的主 Decision target 是 `u_query_joint_lamT_1`，不把 joint path advantage 称为 query descriptors 的独立信息价值。

RQ2 与 RQ3 使用两个不同但预先指定的比较。RQ2 的唯一主要 contrast 是 `milestone-only B3 - milestone-only T0`：两者只读完全相同的 12 个 milestone states，且由 `decision-compare-feature-groups` 逐 state key 与 sampling metadata 核对后输出配对 run rows。RQ3 可保留动态 Proposed 与 `milestone_only_T0` 的整政策比较，但前者可在全部 accepted dynamic opportunities 触发、后者只能在 milestones 触发；其差异同时包含机会调度、输入特征、拟合分数和 first-trigger threshold，不能归因为 Behavior 超出 `FE_ratio` 的增量作用。

## 3. Outer-fold-specific 公平性

对每个 Decision outer holdout function group，必须按以下顺序构造所有 train-side 和 holdout-side baseline：

1. 只用 outer-fit functions 的 complete-budget outcomes 计算 `SBS_outer`；
2. 在 outer-fit 内 cross-fit Query Selector、Behavior-only Selector 和 pre-run query-only Selector，生成 Decision train labels；
3. 生成 joint、Behavior-only 与 operational-increment Utility；
4. 在每个 outer-fit inner function fold 内，仅用 inner-fit functions 重算 `SBS_inner`、cross-fit/拟合两类 Selector、生成三类 Utility并拟合 Decision；用端到端 inner-holdout scores 分别冻结 Query、T0 和 Behavior-only first-trigger thresholds；
5. 用 outer-fit 全量数据拟合各 frozen component；
6. 最后且仅一次在 outer holdout 上评价，随后用拼接的 Proposed OOF calls 与 first-trigger FE 分布供 Random 冻结。

outer/inner holdout 不得参与其评价链中的 SBS、imputation、scaling、Selector、Utility label、Decision model、threshold、Random calibration 或 score-neighborhood。完整 BBOB-train 组件只用于最终重拟合和 validation/external deployment；部署 threshold 与 Random calibration 需要完整 train 的端到端 fold-specific OOF 链。

## 4. 指标与公平比较

主 policy estimand 按 run first-trigger 得到一个 outcome，再按 run → static problem → fixed dimension stratum → function 聚合。必须报告：

- joint decision Utility；
- matched-trigger `query_operational_increment`；
- final `log10_gap`、target-hit rate、endpoint-success rate、ERT；
- end-to-end wall-clock ratio 与拆分时间；
- query FE、total FE、query-call rate、trigger rate、handoff rate；
- first-call precision、non-beneficial first calls 及其累计 Utility；
- first-trigger utility capture；
- coverage、query/selector/action/optimizer failure 和 failure sensitivity。

utility capture 对所有策略共享同一 run-level hindsight opportunity reference：在 native SBS/default trajectory 的全部预定义合格机会中取 `H_r=max_t max(U_t,0)`。策略分子只取其首次触发 state 的 `max(U,0)`，未触发为 0；聚合时报告加权分子、分母与二者比值，并单报 `H_r=0` 的 runs。该参照使用离线 hindsight，不是可部署 policy，也不得随策略首次触发位置改变。若研究 state-score ranking，可另报逐状态 auxiliary capture，但不得称为政策 capture。

主 `lambda_T=1, lambda_M=0` 不按结果修改；`lambda_T={0,0.25,0.5,1,2}` 完整报告为 sensitivity。Utility 使用截断后 `log10_gap` 差减去 `lambda_T` 乘 `log10` runtime ratio；逐行严格满足 `I_q=U_q^{joint}-U_b`，并满足五路径 matched-acquisition 三项增量之和等于 `U_q^{joint}`。Utility、final `log10_gap` 和 wall-clock ratio 同时报告，任何方向冲突都写成 trade-off。

## 5. 统计比较 family

function 是最高聚合层。BBOB-validation 已被旧模型比较、调参与消融查看，只能报告六个固定 functions、固定 dimensions 与 instances 1/2/3 上的等权有限集开发期 estimand；10,000 次条件 bootstrap 固定保留六函数、全部 dimensions 与全部 static problems，只在每个固定 static problem 内配对重抽 optimizer seeds。重抽 function 只作函数组成敏感性，不进入主 CI，也不支持 function 或 transformed-instance 超总体推断。

Utility ±0.01、`log10_gap` ±0.05、runtime ratio `[0.95,1.05]`、call/target-hit-rate 差 ±0.05 是项目内预设 operational tolerance，不是领域通用边界。条件 CI 与 Bonferroni simultaneous interval 只描述区间相对 tolerance 的位置；第一篇论文不作确认性等价声明。endpoint 判断不能借助 Utility 中 gap/runtime 抵消。

每个预设 contrast 产生六个固定且已见的 validation function effects。双侧 exact sign-flip 额外假设其 signs 可交换并穷举 64 个符号向量，raw p 最小为 `2/64=0.03125`。RQ3 的六 contrast Holm family 最小 adjusted p 为 `6×2/64=0.1875`，故在 0.05 下数学上不能拒绝。RQ3 改为估计性 family：逐 function effects、固定六函数有限集效应及条件 CI 为主，p 值只作假设敏感辅助。

RQ3 对每个主 endpoint 建立一个辅助 Holm family：Proposed 分别与 Never/SBS、Always Query、matched-rate Random、pre-run AAS、milestone-only T0、self-thresholded Behavior-only 比较，共 6 个整政策 contrasts。Proposed--milestone-only T0 在这里不作为 RQ2 的 Behavior 信息对比。VBS、matched-trigger operational increment 和 diagnostics 不进入该估计性 family。不同 endpoint 与 suite 不池化。

## 6. 失败规则

所有计划 run 先进入 coverage denominator。score 缺失的机会按 No-query 继续，后续仍可达机会可再次检查；query 一旦触发，已消耗 FE 与 wall-clock 保留。query/Selector 失败固定 fallback 为 query-adjusted `continue_current`，不得改记为 Never Query。

BBOB train/validation 与 CEC2017 固定 `failure_loss_cap=1e20`、取对数前 raw-gap floor/cap `1e-12/1e20`、`success_gap_target=1e-8`、单 state-action path timeout `3600 s`，并在每次 objective evaluation 记录 `observed_first_hit_FE`。`target_hit_observed` 驱动标准 ERT；`path_completed` 与 `endpoint_success=target_hit_observed and path_completed` 分列。timeout/failed path 的 final gap 按失败 cap 保留，失败前已命中的 first hit 不抹除；未命中项的 ERT contribution 计完整 planned budget。Stage-B 主 runtime 对非完成 repetition 使用 `max(raw, role timeout)`。新 suite 必须在运行前给出同类字段及 constraint rule。主报告始终同时给 attempted denominator、failure rate、coverage、target-hit rate、endpoint-success rate、ERT、complete-pair estimate 与双向极端 sensitivity。未闭合 pair 的 gap 用 floor/cap、target hit 用 1/0、ERT 未命中计 full budget、runtime 用该 suite complete pairs 最小正值/timeout，并从这些 endpoint 重算 Utility。已按 cap/timeout 保留的科学 path failure 不当作 missing pair；方向或 operational-tolerance 状态改变、或 coverage<95% 时结论未建立。

## 7. 产物影响

旧逐机会 `p=0.5` Random、`Always Query/Traditional AAS` 共享 outcome、逐状态 policy mean/capture、完整 event schedule T0 和非嵌套 baseline 产物全部失效。基础完整 trajectory 可在通过当前 state/snapshot/seed 检查后重用；新增科学动作数据包括 full-budget 四动作矩阵和 FE=0 pre-run AAS 四算法 outcome。所有 baseline、threshold 和统计汇总必须按本文件重新生成。
