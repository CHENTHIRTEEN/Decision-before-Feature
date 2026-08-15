# Decision-before-Feature phase1 refined sampling 指标口径

> 唯一活动口径（2026-08-14）。本文件冻结 state、policy、性能、时间、Utility、capture、precision、success/ERT 与统计聚合定义。旧 raw-performance max-scale、线性相对时间、逐状态 policy 汇总和一次计时指标全部失效。

## 1. 状态与目标分布

主 RQ1 population 固定为 `descriptor_cheap_invariant`、fold-specific SBS prefix 和 `phase1_dynamic_budget_event_v1` 的合格状态。跨表使用整数 `FE` 状态键；`FE_ratio=FE/FE_total`，名义 milestone 只作 metadata。

RQ1 聚合顺序为 state → run → static problem → fixed dimension stratum → function。政策端点先为每个 run 产生唯一 first-trigger outcome，再从 run 层按同序聚合。function 是顶层统计单位；state、seed、instance 或 function × dimension 不是独立顶层复制。

## 2. 五条路径

主行满足 `prefix_algorithm == default_algorithm == fold-specific SBS`。五条路径都从同一 complete optimizer state 与同一 RNG state 开始：

```text
Skip: native SBS continuation, B - FE_prefix
Query joint: fixed query + Query Selector, B - FE_prefix - FE_query
Query-matched state-only: same acquisition + state-only Selector, B - FE_prefix - FE_query
Sampling-only continue-current: same acquisition + native continuation, B - FE_prefix - FE_query
Behavior-only: no query + Behavior-only Selector, B - FE_prefix
```

四动作集合为 `continue_current` 加其余三个 portfolio algorithms。同算法保持完整内部状态原生推进；跨算法只转移 population、fitness 与 best-so-far，并执行一次 `population_transfer_initialization`。query sample 不并入 optimizer population。

## 3. Gap、target hit、path completion 与 ERT

令五条路径的非负 benchmark-reference raw gap 为 $(g_{skip},g_q,g_m,g_c,g_b)$。三条 acquisition paths 使用同一 query realization 与 sample endpoint。BBOB train/validation 与 CEC2017 固定：

```text
failure_loss_cap = 1e20
log10_gap_floor = 1e-12
log10_gap_cap = 1e20
success_gap_target = 1e-8
action_timeout_seconds = 3600
first_hit_recording = every_objective_evaluation
```

先对 raw gap 应用 floor/cap，再取 `log10`：

\[
\ell_k=\log_{10}\!\left(\min(\max(g_k,10^{-12}),10^{20})\right).
\]

每次 objective evaluation 更新 `observed_first_hit_FE`；`target_hit_observed := observed_first_hit_FE != null`，`path_completed := status == completed`，`target_hit_before_failure := target_hit_observed and not path_completed`，`endpoint_success := target_hit_observed and path_completed`。标准 ERT 的命中计数使用 `target_hit_observed`：若失败前已经命中，保留 observed first hit；未命中项计完整 planned budget。timeout/failed path 的 final-gap endpoint 仍按 failure cap 保留，不能用 endpoint success 抹除已观察 target hit。CEC2022/工程问题未冻结同类字段与 constraint rule 前不得正式运行。

## 4. Decision-state future-path 与政策时间

Stage A 的预指定科学运行与 Selection Reference outcome 固定 gap、observed hit、path completion、endpoint success 和科学路径 FE。Stage B 只为计时：同一 state 的五条 future paths 各真实 replay 三次，每次从相同复制 state/RNG 到 terminal，所有 objective evaluations 都重新执行；固定机器、线程与预加载常驻进程。canonical path order 按 `cyclic_complete_path_v1` 在三个 repetition 间循环移位。共享 prefix 是 sunk cost；在线政策另从 FE=0 计到 terminal。

每次显式保存 `timing_replay_status in {completed,timed_out,failed}`、observed hit、path completion、endpoint success、effective FE、repetition index、order position及 raw/censored 组件与 future-path wall-clock。completed repetition 的 censored time 等于 raw，timed-out/failed repetition 为 `max(raw, timing_replay_timeout_seconds)`；每条路径的 $T_k$ 固定为三次 censored time 的中位数，raw observed median只作诊断。路径身份、completed replays 内部 endpoint、Stage-A→completed replay endpoint 一致性分别保存；Stage-B status instability 与跨阶段 completion instability 也分别保存。任何 repetition 不得选择性补跑或覆盖 Stage-A 科学端点。主 Utility 时间项为 `log10` censored future-path ratios；FE=0 policy wall-clock 是独立端点：

\[
r_{q-s}=\log_{10}(T_q/T_{skip}),\quad
r_{b-s}=\log_{10}(T_b/T_{skip}),\quad
r_{q-b}=r_{q-s}-r_{b-s}.
\]

非正或非有限时间是运行失败，不能用 epsilon 代替。CPU time、批量 prediction 秒数、复制时间或事后选择的 repetition 不是主计时。

## 5. 联合 Utility 与五路径分解

\[
U_q^{joint}=(\ell_{skip}-\ell_q)-\lambda_T r_{q-s},
\]

\[
U_b=(\ell_{skip}-\ell_b)-\lambda_T r_{b-s},
\]

\[
I_q=(\ell_b-\ell_q)-\lambda_T r_{q-b}=U_q^{joint}-U_b.
\]

再令 $U_m$、$U_c$ 分别为 `query_matched_state_only` 与 `sampling_only_continue_current` 相对 Skip 的同公式 Utility：

\[
I_{descriptor}=U_q^{joint}-U_m,\qquad
I_{state\mid acquisition}=U_m-U_c,\qquad
I_{sampling}=U_c,
\]

并逐行满足 $U_q^{joint}=I_{descriptor}+I_{state\mid acquisition}+I_{sampling}$。该等式是固定模型、query realization、预算和 transition rule 下的操作性分解，不作因果解释。

主字段为：

```text
u_query_joint_lamT_1
u_behavior_only_full_budget_lamT_1
query_operational_increment_lamT_1
query_descriptor_use_increment_lamT_1
query_state_only_vs_sampling_increment_lamT_1
query_sampling_direct_increment_lamT_1
```

主 `lambda_time=1, lambda_memory=0` 表示 gap 与 runtime 的十进制数量级变化等权；`lambda_time={0,0.25,0.5,1,2}` 全部报告为 sensitivity。query FE 已通过减少 Query continuation 预算体现，不重复扣除。主报告必须同时给五条 `log10_gap`、wall-clock ratio、Utility 与 memory 端点；scalarized Utility 不替代二维端点。

## 6. First-trigger policy

每个 run 的合格机会按整数 `FE` 和 `decision_opportunity_index` 排序。给定 threshold，策略只在 score 首次超过 threshold 的 state 触发；未超过则不触发。首次触发后，该策略的后续状态不可达，不再进入 threshold、模型选择、call rate、precision、capture 或政策效用。

Never/SBS 的 query decision Utility 为 0。Always Query 在首个在线合格机会触发。`pre_run_aas_fe0` 是独立 FE=0 query-only Selector outcome。`self_thresholded_behavior_only` 使用自己的 `U_b` score/threshold；`matched_trigger_behavior_only` 只在 Proposed 的相同 first-trigger state 计算 (I_q)。

## 7. Precision、non-beneficial calls 与 capture

first-call precision 为被调用 runs 中 first-trigger `U_q^{joint}>0` 的比例。必须同时报告 first-trigger `U_q^{joint}\le0` 的调用数、比例和 Utility 总和；不得以逐状态分类精度替代。

所有策略共享同一 run-level hindsight opportunity reference：

\[
H_r=\max_{t\in E_r}\max(U_{q,t}^{joint},0),
\]

其中 (E_r) 是 native SBS/default trajectory 上全部预定义合格机会。策略分子为其 first-trigger state 的 `max(U,0)`，未触发为 0。聚合时报告加权分子、加权分母、二者比值与 `H_r=0` 的 run 比例。该分母不随策略触发时点改变，也不是可部署策略。

逐状态 positive-row capture 仅可作为 auxiliary score diagnostic，必须与 first-trigger policy capture 分开命名。

## 8. Fold-specific 生成

每个 outer holdout function 的完整链只读 outer-fit functions。outer-fit 内每个 inner holdout 也必须只读 inner-fit functions，并重算 `SBS_inner`、cross-fit/拟合两类 Selector、生成三类 Utility、拟合 preprocessing/Decision 与 inner-holdout outcome。端到端 inner OOF first-trigger outcomes 用于 outer threshold。

outer threshold 冻结后，再用 `SBS_outer` 和 outer-fit 全量组件评价 outer holdout一次。完整 BBOB-train 部署 threshold 与 Random calibration也执行 fold-specific 上游 OOF；不得从 full-train 预制 labels 进行 Decision-only OOF。

## 9. 不确定性与检验

聚合层级统一为 state → run → static problem → fixed dimension stratum → function。BBOB-validation 已被旧流程查看，只能报告六个固定 functions、fixed dimensions 与 instances 1/2/3 上的等权有限集开发期 estimand；10,000 次条件 bootstrap 始终保留全部六函数、fixed dimensions 与 static problems，只在每个固定 static problem 内配对重抽 optimizer seeds。RQ1 对每个抽中 seed/run 保留完整有序 state 簇。function-resampling 只作函数组成敏感性，不进入主 CI，也不表示 function 或 transformed-instance 超总体。

Utility ±0.01、`log10_gap` ±0.05、runtime ratio `[0.95,1.05]`、call/target-hit-rate 差 ±0.05 只称项目内预设 operational tolerance。条件 CI 仅逐项描述相对边界的位置；未来未查看评价集若作等价判断，须在 outcome 前冻结有领域含义的边界与 simultaneous interval。非显著不能替代等价。

每个预设 BBOB-validation contrast 生成六个固定且已见的 function effects；双侧 sign-flip 额外假设其 signs 可交换并穷举 64 个符号向量，exact raw p 最小为 0.03125。RQ3 与 RQ5 各自六 contrast Holm family 的最小 adjusted p 均为 0.1875，故 `alpha=0.05` 下数学上不能拒绝。RQ4 按 suite 与 endpoint 分开，不建立跨四个 suites 的 Holm family。相关 RQ 改为估计性分析；逐 function/problem effects、有限集效应与条件 CI 为主，adjusted p 只作假设敏感辅助。

ERT 的区间必须在每个 bootstrap replicate 内逐 `function × dimension` 重算两政策的 FE numerator、target-hit count 与 log-ratio，再依次对固定 dimensions/functions 等权。单方零命中保留为有符号无穷，双方零命中及聚合时同时含两种无穷记为 undefined；不得删除。undefined mass 保守分配到两侧尾部，`interval_established` 仅由观测 contrast 与扩展实数分位点是否有定义决定，无界区间仍可建立。

## 10. 失败与结论边界

所有计划运行先进入 coverage denominator。Decision score 缺失时该机会按 Skip且可继续观察；query 已触发后的 query/Selector failure 保留 query FE 与时间，并按 query-adjusted `continue_current`。action timeout 或失败使用预先配置的有限 cap 保留 target，但必须同时报告 failure、coverage、ERT、complete-pair estimate 与双向极端 sensitivity。

每个 suite × endpoint 同时报 attempted denominator 与 complete pairs。未闭合 pair 的双向极端赋值为 gap floor/cap、target hit 1/0、ERT 未命中计 full budget、runtime 取该 suite complete pairs 最小正值/timeout，并由这些 endpoint 重算 Utility；已按 cap 保留的科学 path failure 不是 missing pair。方向或 operational-tolerance 状态任一改变、或 coverage<95% 时，对应结论未建立。旧 Utility、旧逐状态 threshold/capture、旧一次计时和非嵌套产物不得进入当前模型或论文结果。
