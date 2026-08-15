# Decision-before-Feature 数学定义与方法章节

> 唯一活动版本（2026-08-14）。本文定义第一篇论文的数学对象与信息边界。主 query 为 `descriptor_cheap_invariant`；`pflacco_standard_invariant` 与 `pflacco_broad_invariant` 只作预定义配置稳健性。旧 max-scale performance、线性 relative-time Utility、逐状态 threshold、重建式 continuation 和四组消融全部失效。

## 1. 问题与信息时间

给定连续黑盒最小化问题 $p\in\mathcal P$：

$$
\min_{x\in\mathcal X_p} f_p(x),
$$

总 objective-evaluation 预算为 $B=FE_{total}$。Portfolio 固定为

$$
\mathcal A=\{\mathrm{DE},\mathrm{PSO},\mathrm{CMA\mbox{-}ES},\mathrm{SHADE}\}.
$$

在整数 FE 为 $FE_t$ 的完整 checkpoint 上，状态 $S_t$ 包含 population、fitness、best-so-far、generation/native-update 计数、算法内部动态量和 RNG state。论文的 Decision 输入不是 $S_t$ 全部内容，而是 query 前可得的 permutation-invariant Behavior 向量 $X_t$。

严格禁止进入 Decision $X_t$：

- query descriptors 或 query samples；
- function/suite/instance/dimension/algorithm identity；
- optimizer-specific parameters、internal dynamics 或 RNG；
- benchmark reference、known optimum、所有 gap 字段；
- action losses、Selector predictions/regret、Utility 或未来 trajectory。

算法 identity、dimension 和上述运行信息只作 metadata、split 或分层报告。

## 2. Analysis-selection decision

在仍可达的在线机会 $t$ 定义：

$$
d_t\in\{0,1\},\qquad
d_t=1\ \text{表示执行固定 query},\quad d_t=0\ \text{表示继续观察。}
$$

策略按整数 $FE_t$ 和 `decision_opportunity_index` 排序，每 run 最多一次 first trigger：

$$
t_r^*=\min\{t:z_\theta(X_{r,t})>\tau\}.
$$

若集合为空，该 run 不调用 query；一旦触发，后续状态在该策略下不可达。未触发 run 的 joint decision Utility 为 0，触发 run 只使用 $t_r^*$ 的 Utility 和 terminal outcome。

这一定义是预测性 policy，不是在线训练 controller，也不表示潜在结果意义上的反事实干预。

## 3. Default、共享状态与动作

SBS 只在相应 fit functions 的完整预算 outcomes 上计算。每个 run 的 raw gap 先按 suite 配置截断并取 `log10_gap`，再按 run → static problem（function × dimension × instance）→ fixed dimension stratum → function 等权聚合；选择 function 等权均值最低算法，并列顺序固定为 `de,pso,cmaes,shade`。

主 population 满足

$$
a_{prefix}=a_{default}=SBS_{fold}.
$$

因此 Skip 从完整 SBS state 原生继续；不重启、不修改参数。共享状态上的唯一动作集合是

$$
\mathcal A(S_t)=\{\texttt{continue\_current}\}\cup
\{a\in\mathcal A:a\ne a_{prefix}\}.
$$

同算法动作恢复完整 state；跨算法动作只继承算法无关的 population、fitness 和 best-so-far，并执行一次 `population_transfer_initialization`。必须逐行保存：

```text
selected_equals_default
selected_equals_prefix
handoff_required
handoff_type
```

并满足

$$
\texttt{handoff\_required}
=\neg\texttt{selected\_equals\_prefix}
=\mathbb I[\texttt{handoff\_type}=\texttt{population\_transfer\_initialization}].
$$

不得生成 `label_source`，也不得用 `same_algorithm/changed_algorithm` 替代这些显式关系。

## 4. 两套动作预算与三个 Selectors

Query 路径执行固定 sample 后，四动作 continuation 预算为

$$
B_q=B-FE_t-FE_q.
$$

Behavior-only full-budget 路径不执行 query，四动作预算为

$$
B_b=B-FE_t.
$$

两套真实 action outcomes 不得互相冒充。对状态 $S_t$ 与动作 $a$，保存 observed continuation terminal loss $L_a(S_t)$，并在各矩阵内部变换：

$$
\widetilde L_a=
\frac{L_a-\min_b L_b}
{\max(\max_b L_b-\min_b L_b,10^{-12})}.
$$

`best observed action` 是本次已运行动作中 loss 最小者，不称为 oracle。

活动 Selectors 均为多输出 `RandomForestRegressor`：

1. full Query Selector：query descriptors、Behavior、连续 query-adjusted remaining-budget ratio；
2. `query_adjusted_state_only_selector`：Behavior、同一 query-adjusted remaining-budget ratio；
3. Behavior-only full-budget Selector：Behavior、full-budget remaining-budget ratio。

Traditional `pre_run_aas_fe0` 另有 FE=0 query-only Selector，只用 query descriptors 和静态 remaining-budget ratio，从原生 initialization 选择一个算法；它不等于在线 Always Query。

## 5. Terminal endpoint 与 query sample

Query sample 不进入 optimizer population，但每个 sample point 是实际 objective evaluation。令：

$$
g_s=\text{Skip terminal best gap},
$$

$$
g_q=\min\{\text{query-sample best gap},\text{selected-continuation best gap}\},
$$

$$
g_b=\text{Behavior-only selected-continuation best gap}.
$$

预指定 Stage-A 对每条路径只运行一次，并由该行固定 terminal gap、`observed_first_hit_FE`、`target_hit_observed`、`target_hit_before_failure`、`path_completed`、`endpoint_success`、planned FE、effective FE、ERT contribution 与科学失败状态。其中 `target_hit_observed := observed_first_hit_FE != null`，`target_hit_before_failure := target_hit_observed and not path_completed`，`endpoint_success := target_hit_observed and path_completed`；标准 ERT 使用 `target_hit_observed`。Query 路径的性能端点使用 $g_q$；sample 内首次达到 success target 时保存 `query_first_hit_offset`。同时保存：

- `gap_query_continuation_only`；
- `query_sample_best_contribution_log10_gap`；
- sample 与 continuation 各自的 first-hit/terminal metadata。

这样可区分 sample 直接找到更优点与 query descriptors 改善动作选择的来源。

## 6. Utility 与 query 操作性增量

对 $k\in\{s,q,b\}$，使用 suite 预先冻结的 raw-gap floor/cap：

$$
\ell_k=\log_{10}\left(\min\{\max(g_k,10^{-12}),10^{20}\}\right).
$$

Stage-B 将 `skip`、`query_joint`、`query_matched_state_only`、`sampling_only_continue_current` 与 `behavior_only_full_budget` 五条 selected 路径从同一 decision state/RNG 到 terminal 真实执行预定三次，但只用于计时；按 `cyclic_complete_path_v1` 交错顺序。每次保留 raw observed wall-clock；completed repetition 的 censored time 等于 raw，timed-out/failed repetition 的 censored time 为 `max(raw, role timeout)`，$T_k$ 固定为三次 censored time 的中位数。raw observed median 只作诊断，旧 failure-worst-case 字段只作同一 censored 值的兼容别名。每次保存 status、observed hit、path completion、endpoint success、effective FE 与失败字段。路径身份、completed replays 内部 endpoint 和 Stage-A 到 completed replay 的 endpoint 一致性分别保存；Stage-B 内部 status instability 与 Stage-A/Stage-B completion instability 也分别保存。任何 replay 都不得改写 Stage-A 科学端点或被选择性补跑。共享 prefix 视为 sunk cost；FE=0→terminal online policy wall-clock 另作政策端点，不进入 Utility。

主配置 $\lambda_T=1,\lambda_M=0$：

$$
U_q^{joint}
=(\ell_s-\ell_q)
-\lambda_T(\log_{10}T_q-\log_{10}T_s),
$$

$$
U_b
=(\ell_s-\ell_b)
-\lambda_T(\log_{10}T_b-\log_{10}T_s),
$$

$$
I_q
=(\ell_b-\ell_q)
-\lambda_T(\log_{10}T_q-\log_{10}T_b)
=U_q^{joint}-U_b.
$$

`u_query_joint_lamT_1` 是主 Decision target；其二元标签为

$$
y_q=\mathbb I[U_q^{joint}>0].
$$

`query_operational_increment_lamT_1` 比较两条可操作路径，包含 query sample FE、sample best、预算差、Selector 和 runtime，不得称为纯信息效应或 causal effect。

另定义 query-feature 预测诊断：

$$
\Delta_{pred}
=\ell_{\text{state-only selected, continuation-only}}
-\ell_{\text{full-query selected, continuation-only}},
$$

对应 `query_feature_predictive_increment_log10_gap`。两种 Selector 使用同一 query-budget outcomes，均取 OOF prediction；该量不新增 action losses、不纳入 sample best，也不扣 query acquisition cost。

`lambda_time in {0,0.25,0.5,1,2}` 是完整敏感性集合。$\lambda_T=1$ 表示 gap 与 runtime 的十进制数量级变化等权，不代表通用资源偏好。所有主表必须同时报告 Utility、terminal `log10_gap` 和 runtime ratio。

## 7. Selector decomposition

在同一动作矩阵中，令

$$
L^*=\min_{a\in\mathcal A(S_t)}L_a,
$$

$$
\text{potential gain}=L_{skip}-L^*,
$$

$$
\text{selector regret}=L_{selected}-L^*,
$$

$$
L_{skip}-L_{selected}
=\text{potential gain}-\text{selector regret}.
$$

Population transfer 已包含在 observed $L_a$ 和 path runtime 中；query FE 已通过 $B_q$ 以及 terminal outcome 进入比较，二者都不能在 Utility 里再次扣除。

## 8. Behavior representation

Decision state 由 query 前 Behavior 构成。活动输出共 34 个唯一 `bf_*` 字段：31 个正式输入、3 个诊断字段。正式输入按以下六组冻结：

| 组 | 内容 | 字段数 |
|---|---|---:|
| T0 | 仅实际 `bf_fe_ratio=FE_t/B` | 1 |
| B1 | core permutation-invariant Behavior | 19 |
| B2 | B1 + longitudinal set dynamics | 25 |
| B2+Motion | B2 + set-motion | 28 |
| B2+Maturity | B2 + Maturity basis | 28 |
| B3 | B2 + Motion + Maturity | 31 |

### 8.1 Native-update windows

名义窗口 $W\in\{0.02,0.05,0.10\}$。若目标 $FE_t-\operatorname{round}(WB)$ 不在完整 update 边界，选不晚于目标的最近完整 native update 作为 anchor $a$，使：

$$
\operatorname{round}(WB)
\le FE_t-FE_a
<\operatorname{round}(WB)+N,
$$

其中 $N$ 为 population size。所有 rate/slope 使用实际 $(FE_t-FE_a)/B$；实际 FE、ratio 与 native-update 数只作 metadata，不进入 Decision $X$。

### 8.2 Permutation invariance

跨 checkpoint 不把 population 行号当个体身份。空间变化用等权经验 Wasserstein、centroid shift/coherence 与 covariance summaries；fitness 变化用排序经验分位数。任何依赖 row-wise displacement、individual ID、ancestry 或 CMA-ES 样本身份的字段均不进入活动协议。

## 9. Search Maturity basis

由既有 Behavior 确定性得到 $ES_t,XS_t,E_t,X_t$，活动基函数为：

$$
M_t=ES_t(1-XS_t),
$$

$$
M_t^{linear}=\frac{ES_t+(1-XS_t)}{2},
$$

$$
R_t^{EE}=\frac{E_t}{X_t+10^{-12}}.
$$

字段为：

```text
bf_search_maturity
bf_search_maturity_linear
bf_explore_exploit_ratio
```

它们不增加原始信息，只是预设非线性基函数。Search Maturity 不是独立观测、latent state、收敛判据、因果中介或 Utility 标签；不定义早/中/晚类别，也不预设单调、U 形或倒 U 形关系。

## 10. Decision learning 与完整嵌套

活动候选固定为 LDA、Logistic Regression 与 Ridge。每个 fit fold 内独立拟合 median imputer、scaler 和模型；训练 fold 整列缺失时停止，不从 holdout/validation 补值。

每个 `cv_group_id = function_id` outer holdout 的链为：

```text
outer-fit functions
-> SBS_outer
-> cross-fitted/fitted Query, Behavior-only and FE=0 Selectors
-> outer-fit Utility labels
-> inner-function OOF Decision scores and first-trigger threshold
-> fitted Decision model
-> one outer-holdout evaluation
```

每个 inner fold 又必须只用 inner-fit functions 重算 `SBS_inner`、Selectors、Utility 和 Decision。不得先用完整 BBOB-train 生成标签，再对 Decision 单独 OOF。

主模型选择指标为拼接 outer holdout 后，按 first-trigger policy 重建的 run-level mean `u_query_joint_lamT_1`。AUROC、Average Precision 和 Spearman 只作辅助；连续 Utility RMSE 只对 Ridge 定义。BBOB-validation 已在旧流程中用于模型比较、调参与消融，不能再作为未查看评价集或 selected procedure 的无偏性能估计；当前流程仍禁止用其继续拟合或重选。

主拟合权重依次使 function、固定 dimension stratum、static problem 与 optimizer run 等权，再把每个 run 的权重均分给其 eligible states，并缩放到平均 row weight 为 1。旧 `sample_weight=1` 只作敏感性。现有 estimator wiring 尚未闭合该权重，闭合前不得启动正式拟合。

完整 BBOB-train 的 threshold 和 matched-rate Random calibration也来自 fold-specific 上游 OOF：

- Proposed：`oof_utility_first_trigger` threshold；
- Time-only 与六个 feature groups：各自 train-only first-trigger threshold；
- Behavior-only：对 $U_b$ 的自身 train-only threshold；
- Random：Proposed OOF run call rate 与 first-trigger FE-ratio 经验分布。

## 11. Time-only、ablation 与部署

主 Behavior-versus-time 比较只使用 12 个 mandatory milestones：`milestone_only_T0` 与 milestone-restricted B3 的行完全相同。事件 opportunity 的产生受 Behavior 条件影响，因此完整 dynamic schedule 的 T0 只能作为 `schedule_conditioned_T0` sensitivity。另做预先按 dimension 分层、每层仍只输入 FE ratio 的 `dimension_stratified_T0`，用于控制 dimension/绝对预算这一调用前静态上下文；它不把 dimension 加入主 Decision X。

RQ5 的六个 groups 使用 RQ2 在 B3 上选定的同一模型家族，预设 contrasts 为：

```text
B1 - T0
B2 - B1
B2+Motion - B2
B2+Maturity - B2
B3 - (B2+Motion)
B3 - (B2+Maturity)
```

部署时原生 SBS 持续推进；在每个合格机会计算 Behavior score。首次越阈值则执行一次固定 query 和 full Query Selector，否则继续原生搜索。部署中不在线更新模型、不重复选择 query type。

## 12. 失败、统计与解释边界

BBOB train/validation 与 CEC2017 冻结：raw-gap floor/cap `1e-12/1e20`、`failure_loss_cap=1e20`、`success_gap_target=1e-8`、action timeout `3600 s`、Stage-A 每次 objective evaluation 记录 first hit、Stage-B 三次 decision-state future-path timing-only replay 和独立 FE=0 policy wall-clock。Stage-A timeout/failed path 的 final-gap endpoint按失败 cap 保留；若失败前已经命中 target，standard ERT 仍保留 observed first hit，而 `endpoint_success=false` 明确路径未完成。未命中项的 ERT contribution 计完整 planned budget。Stage-B timeout/failure 用删失 runtime 进入主时间成本，并另进入 timing failure/instability sensitivity，但不改写 Stage-A gap 或 path completion。

function 是最高聚合层。BBOB-validation estimand 是六个已见固定 functions、固定 dimensions 与 instances 1/2/3 上的等权有限集均值；10,000 次条件 bootstrap 固定全部六函数、dimensions 与 static problems，只在每个固定 static problem 内配对重抽 optimizer seeds；RQ1 对每个抽中 seed/run 保留完整有序 state 序列。function-resampling 只作函数组成敏感性，不作 function 或 transformed-instance 超总体推断。六函数 sign-flip 还要求 signs 可交换；RQ3 与 RQ5 的六 contrast Holm families 数学上不能在 0.05 下拒绝，只作假设敏感辅助。RQ4 按 suite/endpoint 分开，不构造跨 suite 的四 contrast Holm family。有限集效应量、逐 function/problem 结果和条件区间是主证据。Utility $\pm0.01$、`log10_gap` $\pm0.05$、runtime ratio $[0.95,1.05]$、call/target-hit rate $\pm0.05$ 只称项目内预设 operational tolerance。

ERT 不使用通用算术均值 bootstrap。每个 replicate 固定每个 `function × dimension` stratum 内的 static problems，只在每个 problem 内联合配对重抽 optimizer runs，分别重算两政策的 FE numerator 与 target-hit count，再形成 `log10(ERT_treatment/ERT_reference)`；固定 dimensions 和 functions 依次等权。单方零命中保留为有符号无穷，双方零命中及同一聚合层同时出现两种无穷时记为 undefined，均不得删除。undefined mass 按预设规则保守分配到两侧尾部；`interval_established` 只由观测 contrast 与扩展实数分位点是否有定义决定，无界区间仍可建立。

本方法只能建立：在固定 query、Selector、portfolio、预算、transition rule 和目标状态分布下的预测与操作性性能差。它不能单独建立 query descriptors 的因果价值、Search Maturity 的真实存在或任意 benchmark/representation 的普遍规律。
