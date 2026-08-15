# Decision-before-Feature Master Research Specification

> 实现同步（2026-08-15）：完整 optimizer-state continuation 已替换 checkpoint 重建机制。正式 estimand 分为 Query/full-Selector 联合策略效用、Behavior-only full-budget 效用与二者差；性能只使用预指定 Stage-A 单次科学运行经配置截断后的 `log10_gap`，时间只使用 Stage-B 三次真实 decision-state-to-terminal future-path wall-clock 中位数的 `log10` ratio，共享 prefix 视为 sunk cost。Stage-B replay 不改写科学 gap、`observed_first_hit_FE`、`target_hit_observed`、`path_completed`、`endpoint_success` 或 planned/effective FE；FE=0→terminal policy wall-clock 是独立政策端点。SBS、两类 Selector、Utility、Decision 与 first-trigger threshold 必须在 outer/inner function 链中逐层拟合。活动 query ID 统一为 `descriptor_cheap_invariant`、`pflacco_standard_invariant` 与 `pflacco_broad_invariant`。BBOB-validation 与 CEC2017 分别降级为已见内部评价集和已见外部开发集；确认性外部证据等待完整冻结后首次运行的 CEC2022 与工程集合。旧 Utility、labels、逐行 threshold、模型和评价结果已撤回，必须从相应依赖位置重生成。

## 0. Document Purpose

本文档是 Decision-before-Feature 项目的最高层研究规范。

用途：

1. 指导论文研究设计；
2. 指导后续 Vibe/Codex 实验开发；
3. 固化方法、数据、实验和工程约束；
4. 避免开发过程中偏离论文核心问题。

核心原则：

> 本研究不是设计新的优化算法，而是研究在黑盒优化中，所评估的固定
> landscape-analysis query 是否值得执行。

构念边界：当前主 query `descriptor_cheap_invariant` 是 14 维 permutation-invariant 自定义低成本描述符；统一 median/IQR preprocessing 后恒为 0/1 的 `descriptor_y_median` 与 `descriptor_y_iqr` 已从活动 whitelist 删除，query ID、`lhs_50d`、5% FE 与 action losses 不变。它不代表完整 ELA、完整 pflacco 或一般意义上的所有 Landscape Analysis。第一篇论文不得把三档结果外推到未评价的 pflacco feature groups、NeurELA、Deep-ELA 或其他 landscape representation。

Sampling 边界：query sampling 是显式整数 `sample_seed` 定义的确定性算子，base seed、stream code、function number、instance、dimension 与 sample-design code 一起进入 `numpy.random.SeedSequence`。每个静态 problem × sample design 只有一个 problem-keyed LHS realization；optimizer seed、decision state 与 action共用该 realization。主 estimand 因而严格条件于这组固定 query realizations，不能外推为对重复 LHS sampling randomness 的平均效应或稳健性。本轮不增加 LHS replicates，也不新增相应 action losses。cheap 与 standard 共享 `lhs_50d`；broad 同时改变 representation、sample size、sample realization 与 action budget，因此 standard--broad 只能作为整套 query configuration 比较，不能拆分归因。

---

# 1. Research Positioning

## 1.1 Research Area

所属领域：

- Automated Algorithm Selection (AAS)
- Landscape analysis and Exploratory Landscape Analysis (ELA)
- Metaheuristic Behavior Analysis
- Resource-aware Optimization

---

## 1.2 Core Research Question

传统流程：

    Problem

    ↓

    Fixed Landscape-query Feature Extraction

    ↓

    Algorithm Selection

    ↓

    Optimizer

隐含假设：

    The evaluated feature query is always beneficial

本文提出：

    A fixed analysis query should be selected conditionally.

即：

> 在执行预先定义的 landscape-analysis query 之前，判断是否值得付出该配置的采样与计算成本。

---

# 2. Problem Definition

## 2.1 Analysis Selection Problem

给定黑盒问题：

$$
p
$$

决策：

$$
d\in\{0,1\}
$$

其中：

0:

No-query

1:

Run the evaluated fixed query

目标：

最大化：

$$
Utility(d,p)
$$

---

# 3. Core Framework

整体框架：

    Unknown Black-box Problem

    |

    v

    Cheap Optimization Probe

    |

    v

    Algorithm-agnostic Behavior Extraction

    |

    v

    Search Maturity Estimation

    |

    v

    Query Utility Prediction

    |

    v

    Decision-before-Feature

    ------------------

    |                |

    No-query        Run Query

    |                |

    Default Solver   Query + Algorithm Selection

---

# 4. Offline Learning Strategy

## 4.1 Decision

采用：

Offline trajectory collection + supervised learning。

不采用：

Online controller training。

---

## 4.2 Reason

原因：

1. Query Utility需要离线计算；
2. 避免credit assignment问题；
3. 保证Analysis Selection问题独立。

---

# 5. Optimization Experience Dataset

## 5.1 Optimizer Pool

必须包含多种搜索机制：

- Differential Evolution
- Particle Swarm Optimization
- CMA-ES
- SHADE/L-SHADE

目的：

学习通用搜索行为。

---

# 6. Algorithm-agnostic Behavior Representation

## 6.1 禁止输入

不允许：

Algorithm-specific parameters。

例如：

PSO:

- inertia
- c1
- c2

DE:

- F
- CR

CMA-ES:

- covariance
- sigma

原因：

避免模型学习算法身份。

---

## 6.2 Behavior Feature Taxonomy

### Progress

- FE ratio
- improvement rate
- improvement frequency

### Diversity

- population diversity
- diversity change

### Exploration

- population Wasserstein change rate
- centroid shift and shift coherence
- covariance spectral concentration

### Fitness Distribution

- quantile improvement fraction
- mean distribution improvement rate
- one-dimensional fitness Wasserstein rate

### Exploitation

- distance decay
- stagnation
- convergence rate

---

## 6.3 Relation to Prior Behavioral Metrics

本项目的behavior representation受两类metaheuristic behavior analysis文献启发：

- exploitation behavior diagnostics；
- metaheuristic behavioral similarity analysis。

但当前指标不是对已有论文指标的逐式复现。

关键口径：

1. *How do metaheuristics exploit?* 中的distance-to-reference decay、directional entropy和stagnation indicators主要用于分析和调节late-stage exploitation behavior，并使用iteration窗口和exploitation phase划分；本文不采用需要跨代个体对应关系的directional entropy，也不实现其direction bins，而以permutation-invariant集合分布变化指标替代。
2. *Determining Metaheuristic Similarity Using Behavioral Analysis* 中的behavioral characteristics主要用于whole-run algorithm similarity，包括diversity/accuracy/convergence/locality/communication/evaluation-effort等整段搜索特征。
3. 本项目只继承其中可解释、低成本、算法无关的行为语义，并改写为checkpoint-level、FE-ratio-normalized、permutation-invariant behavior state。

因此论文主线必须表述为：

> inspired by prior behavioral analysis metrics

或：

> adapted into budget-normalized algorithm-agnostic behavior states

不得表述为：

> directly using the metrics from prior work

当前主模型不使用：

- true global optimum；
- function identity；
- query feature；
- whole-run knee-point extraction；
- STN/IN similarity metrics；
- DBSCAN frequency map。

这些内容可作为相关工作或扩展实验背景，不进入Decision Model主输入。

---

# 7. Search Maturity

## Definition

Search Maturity表示一个研究内预定义的 behavior 派生坐标：探索结构稳定化与开发饱和之间的组合位置。它不由 Utility 反向定义，也不预设 query 有价值。

`M_t=ES_t(1-XS_t)` 是已有 Behavior 的确定性变换，不增加新的原始信息，不是独立 latent state、因果中介或已建立的 convergence measure。它只能通过冻结的预测比较检验其对当前线性 Decision 候选是否提供有用的非线性基函数。

---

## Difference

  概念                       含义

---

  Convergence                是否接近最优
  Exploration/Exploitation   搜索行为
  Search Maturity            预定义 behavior 派生坐标

---

## Conceptual Model

由两个因素组成：

### Exploration Stabilization

探索是否形成稳定结构。

### Exploitation Saturation

开发是否过度饱和。

成熟度不要求随 FE 单调增加。是否与 Query Utility 存在单调或非单调关系属于待估计结果，不能由定义预设“最佳分析窗口”。

---

# 8. Offline Utility Label

No-query 与 Run Query 的 paired continuation 使用同一完整 optimizer checkpoint state。

主实验固定为 Population Transfer：

- 第一篇论文主协议令 prefix/default 都等于训练集 SBS，No-query 原生继续该 SBS 的 population、fitness、内部动态量、generation、best-so-far 与 RNG state；
- 若 Run Query 仍选择 prefix algorithm，则从同一完整状态原生继续，但后续优化预算扣除 `FE_query`；
- 若 selector 切换算法，新算法重新初始化自身内部状态；
- 跨算法初始化只转移 checkpoint population、fitness 与 best-so-far position，并明确记为 population transfer；
- 不使用 Best-so-far Warm Start；
- query 采样点不进入 optimizer population，但其真实 best/first hit 进入 operational Query terminal gap、`target_hit_observed` 与 ERT；`endpoint_success` 仍要求 continuation path 完成。

同一 shared state 还必须运行 `behavior_only_full_budget` 四动作矩阵：不执行 query，所有动作使用 `B-e_t`；Selector 只读取 query 前可得 Behavior 与 full-budget remaining ratio。Query-adjusted 与 full-budget action loss 是两个不同 estimand 的数据，不能互换。

全 prefix trajectory 只用于 cross-probe robustness、leave-one-probe-out 与 algorithm-agnostic 泛化。正式标签必须保存 `selected_equals_default`、`selected_equals_prefix`、`handoff_required`、`skip_switches_from_prefix`、`no_query_algorithm` 和 `handoff_type`；其中 `no_query_algorithm=default_algorithm`，`handoff_type=query_transition_mode`，`handoff_required = not selected_equals_prefix`。活动输出不再使用 selected-vs-default 字符串别名。

## 8.1 No-query

得到：

$$
P_{skip}
$$

---

## 8.2 Run Query

流程：

    Problem

    ↓

    Query Feature Extraction

    ↓

    Algorithm Selection

    ↓

    Optimizer

得到：

$$
p_{query}
$$

正式 Selection Reference 不是 problem 级静态分类器。对每个共享 checkpoint state，离线分别运行 Query-adjusted 和 full-budget 的 `continue_current` 与其余 portfolio actions，记录 continuation-only action loss。Query Selector 使用 query features、算法无关 behavior 与 query-adjusted `remaining_budget_ratio`；`behavior_only_full_budget` Selector 只用 behavior 与 full-budget ratio。训练行使用按 function 的交叉拟合预测，validation/test 不参与拟合。

除 full Query Selector 与主 full-budget Behavior-only Selector 外，同一 query-adjusted 四动作矩阵还拟合第三个 `query_adjusted_state_only_selector`。它与 full Query Selector 读取完全相同的 action outcomes，不新增 action losses；两者 OOF selected continuation-only `log10_gap` 差保存为 `query_feature_predictive_increment_log10_gap`。该诊断排除 query sample best，只表示 query features 的 OOF 边际预测贡献，不是主策略指标或因果效应。

---

## 8.3 Utility

预指定 Stage-A 对每条路径执行一次科学运行；该运行唯一固定 terminal gap、`observed_first_hit_FE`、`target_hit_observed`、`target_hit_before_failure`、`path_completed`、`endpoint_success`、planned FE、effective FE、ERT contribution 与科学失败状态。`target_hit_observed := observed_first_hit_FE != null`，`target_hit_before_failure := target_hit_observed and not path_completed`，`path_completed := status == completed`，`endpoint_success := target_hit_observed and path_completed`；标准 ERT 使用 `target_hit_observed`。三条 Stage-A 终端值转成非负 benchmark-reference raw gap (g_{skip},g_q,g_b)。BBOB train/validation 与 CEC2017 在取对数前固定应用 `log10_gap_floor=1e-12` 与 `log10_gap_cap=1e20`：

$$
\ell_k=\log_{10}\!\left(\min(\max(g_k,10^{-12}),10^{20})\right),
\quad k\in\{skip,q,b\}.
$$

known reference/gap 只用于离线标签与最终评价，不进入 trajectory/Behavior、Selector 输入或 Decision X。主 Query gap (g_q) 包含 query sample best；另报告 continuation-only gap 与 `query_sample_best_contribution_log10_gap`。

Stage A 的一次科学运行与 Selection Reference outcome 决定五条路径的 terminal gap、observed hit、completion、endpoint success 与科学路径 FE，保存 `scientific_endpoint_source=stage_a_selection_reference_outcome`。Stage B 在同一固定机器、线程与预加载进程中，从同一 complete state/RNG 对每条 selected path 真实 replay 3 次，但只决定 wall-clock。canonical path order 按 `cyclic_complete_path_v1` 循环移位；逐次保存 `timing_replay_status in {completed,timed_out,failed}`、raw observed runtime、censored runtime、`observed_first_hit_FE`、`target_hit_observed`、`target_hit_before_failure`、`path_completed`、`endpoint_success`、effective FE、timeout、order position 与组件时间。completed repetition 的 censored runtime 等于 raw observed runtime；timed-out/failed repetition 的 censored runtime 为 `max(raw observed runtime, role timeout)`。三次 raw time 原样保留但不按 status 筛选，五条路径各自的主 $T_k$ 固定取三次 censored runtime 的中位数；raw observed median 只作诊断，不得补跑。路径身份、completed replays 内部 endpoint、Stage-A 到 completed replay endpoint 分别使用独立 consistency 字段；Stage-B status instability 与 Stage-A/Stage-B completion instability 也分别保存。Stage-A path 未完成时，completed replays 不能覆盖 Stage-A 失败端点。planned FE 必须保持路径身份一致；replay effective FE 不与 Stage A 或其他 repetitions 强制相同。旧 `failure_worst_case` 只可作为相同 censored 值的兼容别名。共享 prefix 为 sunk cost。FE=0→terminal policy wall-clock 另存且不进入 Utility。不得用 CPU time、批量 prediction 秒数、复制时间或事后选择的重复替代。

三个 Utility 定义为：

$$
U_{query}^{joint}=(\ell_{skip}-\ell_q)
-\lambda_T(\log_{10}T_q-\log_{10}T_{skip}),
$$

$$
U_b=(\ell_{skip}-\ell_b)
-\lambda_T(\log_{10}T_b-\log_{10}T_{skip}),
$$

$$
I_q=(\ell_b-\ell_q)
-\lambda_T(\log_{10}T_q-\log_{10}T_b)
=U_{query}^{joint}-U_b.
$$

Query sampling FE 已通过较少 continuation budget进入 outcome，不能再次按 FE 扣除。$U_{query}^{joint}$ 回答 Query/full-Selector operational path 是否优于 native SBS；$I_q$ 回答相对 full-budget Behavior-only path 的操作性净增量，包含 query FE、sample best、时间、budget 与 transition，不能称纯信息效应。`query_feature_predictive_increment_log10_gap` 才是同预算/同动作矩阵上的 OOF 预测诊断，但仍不作因果解释。

第一篇论文主操作性情景固定为 $\lambda_T=1,\lambda_M=0$；$\lambda_T=1$ 表示 gap 与 runtime 的十进制数量级变化等权。$\lambda_T\in\{0,0.25,0.5,1,2\}$ 只作完整预设敏感性分析。所有 raw-gap max-scale、线性相对时间和一次计时旧 Utility 全部失效。

逐状态最佳已观测动作只用于诊断分解：

$$
\ell_{skip}-\ell_{q,continuation}
=
(\ell_{skip}-\ell_{best\ observed})
-(\ell_{q,continuation}-\ell_{best\ observed}).
$$

跨算法 Population Transfer 的影响已包含在 observed action loss 中，不作为额外减项重复计入。

---

# 9. Decision Model

## Input

Algorithm-agnostic behavior state。

不包含：

- Function ID
- Dimension
- Algorithm ID
- Query Feature

---

## Output

预测：

$$
\widehat U_{query}^{joint}
$$

同名冻结模型家族还单独拟合 $U_b$，得到 self-thresholded Behavior-only policy。它使用自身的 BBOB-train inner-OOF first-trigger threshold，不复用 Query policy threshold，也不重新选择模型家族。$I_q$ 的政策级报告则在 Proposed 首次触发的同一 state 上匹配 Query 与 Behavior-only 路径；self-thresholded policy 与 matched-trigger query 操作性增量是两个不同对象。

$I_q$ 的活动字段名为 `query_operational_increment_lamT_*`。它必须在全部 eligible states 与 Proposed first-trigger states 两个范围分别报告，并包含 query FE/runtime、sample best、较短 continuation budget 和 Selector 差异；它不是纯信息效应或因果 estimand。若 $U_q^{joint}>0$ 而同一范围 $I_q\le0$，只能支持联合 Query/full-Selector 路径优于 SBS，不能支持 query acquisition 优于 full-budget Behavior-only。正式五路径加入 matched-acquisition 的 `query_matched_state_only` 与 `sampling_only_continue_current`，分别构造 descriptor-use、state-only-vs-sampling 和 sampling-direct 增量，并要求三项逐行加和为 $U_q^{joint}$。这些量是固定模型、query realization、预算和 transition rule 下的操作性分解，不作因果解释。

---

## Recommended Models

活动候选固定为：

- LDA classification：预测 `u_query_joint_lamT_1 > 0` 的分类分数；
- Logistic Regression classification：预测 `u_query_joint_lamT_1 > 0` 的分类分数；
- Ridge regression：预测连续 `u_query_joint_lamT_1`。

不继续把 Random Forest、XGBoost、LightGBM、MLP 或其变体加入 Decision Model 活动模型搜索。Selection Reference 中固定的 action-loss Random Forest regression 属于不同组件，不受本条影响。

主模型按 BBOB-train outer-function OOF 的 run-level first-trigger mean `u_query_joint_lamT_1` 选择。每个 outer fold 只用 outer-fit functions 计算 `SBS_outer`，cross-fit Query 与 Behavior-only Selectors生成 outer-fit Decision labels。每个 inner fold 还必须只用 inner-fit functions 重算 `SBS_inner`、cross-fit/拟合两类 Selectors、生成三类 Utility并拟合 preprocessing/Decision；inner-holdout Utility 与 score 只能由 inner-fit 上游组件产生。拼接端到端 inner OOF outcomes 后分别拟合两个 first-trigger thresholds。outer holdout 只在全部组件冻结后评价一次，不得影响 SBS、Selector、Utility、preprocessing、Decision、threshold 或 Random calibration。不得先在完整 BBOB-train 生成一张 Selector/Utility 表，再把它当作端到端 OOF 证据。随后用完整 BBOB-train 的 grouped-by-function 上游 OOF 分数按 first-trigger 规则冻结 `oof_utility_first_trigger` 与 `oof_behavior_utility_first_trigger`，并冻结 Proposed OOF call rate/trigger-FE 分布；最后在完整 train 重拟合 SBS、Selectors 与 Decision fits。train outer OOF 只承担候选选择和开发期诊断，不是选中 procedure 的无偏估计。BBOB-validation 已被历史开发读取，只给已见内部有限集估计；CEC2017 只给已见外部开发估计。selected procedure 与 milestone-only B3--T0 的确认性证据等待闭合后首次运行的 CEC2022 与工程集合。

主拟合使用 `cluster_balanced_fit`：在每个 fit fold 内依次使 function、fixed dimension stratum、static problem 与 optimizer run 等权，再把 run 权重等分到其合格 states，并把 fold 内 row weights 归一化到均值 1。`sample_weight=1` 的 state-row 等权拟合降为 `row_weighted_fit` sensitivity。nested first-trigger evaluation 不会自动修正训练权重；imputation/scaling 与三个候选 estimator 尚未完成统一的 cluster-weight wiring，是模型冻结 blocker。

对 run $r$ 的有序机会 $t_{r1},\ldots,t_{rK}$，令

$$
J_r(\tau)=\min\{j:z(t_{rj})>\tau\}.
$$

若集合为空，该 run 的 decision utility 为 0；否则只使用 $U_{query}^{joint}(t_{rJ_r})$。首次触发后的状态不可达，不得进入 threshold、模型选择、call rate、precision 或 utility capture。逐状态 AUROC、Average Precision、Spearman 与 Ridge RMSE 只作 score-level 辅助评价。

---

# 10. Dataset Protocol

## Training

BBOB：

Dimensions:

- 10D
- 20D
- 40D

Algorithms:

- DE
- PSO
- CMA-ES
- SHADE

---

## Validation

BBOB：

- function families: 5, 9, 13, 14, 19, 24
- dimensions: 10D / 20D / 40D

注意：

- 当前 COCO `bbob` suite 不支持 50D。
- BBOB 50D 不进入主协议。
- 如需 50D / 100D 泛化，必须另设扩展实验并选择 COCO 支持的 suite。

---

## Evaluation Roles

- BBOB-validation：已见内部评价集；历史开发读取过 outcome，只报告固定六函数有限集估计。
- CEC2017：已见外部开发集；历史 online/targeted 诊断读取过 outcome，只报告外部开发有限集估计。
- CEC2022：预定 untouched external confirmation；当前缺少冻结配置，在 functions、dimensions、budgets、repetitions、references、失败端点、factory/runner 与 contrasts 闭合前不得运行。
- Engineering problems：预定 untouched external confirmation；当前还缺少问题清单、配置、factory、预算、重复、reference/gap、constraint rule 与 endpoints，全部是确认性运行 blocker。

“确认性”只表示 suite 与分析在首次 outcome 前 prospective 冻结，不表示预列 benchmark functions 是函数超总体的概率样本。

---

# 11. Function Split Rules

禁止：

random instance split。

原因：

同一 base function 的 transformed instances 共享 function identity；random instance split 会让同一 function 同时出现在 fit 与 holdout。

采用：

`cv_group_id = function_id` / function-level split。历史字段 `family=bbob_fNNN` 仅是景观 taxonomy 字段，不是 CV 分组键；论文不得据此声称跨 landscape-family 泛化。

---

# 12. Dynamic State Sampling Protocol

禁止：

固定100 FE。

采用冻结协议 `phase1_dynamic_budget_event_v1`。候选监测网格为 `0.20–0.60`、步长 `0.01`，必选预算里程碑为：

    20%, 22%, 24%, 26%, 28%, 30%, 34%, 38%, 42%, 46%, 50%, 60%

事件状态由 improvement resume、stagnation onset、covariance effective-rank change、elite migration 和 diversity recovery 的冻结阈值触发。每个跨过至少一个 `0.01` 监测网格的完整原生 update 只判定一次事件；若同一 update 跨过的监测点含预算里程碑，则以该里程碑为合并行名义节点，附着事件但不消耗 event-only 配额、最小间隔锚点或 `event_index_in_phase`；若不含里程碑，则以最新跨过的监测点为名义节点。冻结的 `population_size=40` 与 `FE_total=1000D` 保证一次 update 的 ratio 跨度不超过 `0.01`，不会同时跨过两个预算里程碑。每阶段最多 2 个 event-only 状态，event-only 实际 FE-ratio 间隔至少 `0.02`；被 gap/quota 抑制落盘的 crossing 仍推进再武装状态。每个 run 输出 12–18 个状态。

`FE_ratio` 必须等于实际 `FE/FE_total`，名义里程碑保存为 `budget_milestone_ratio`。跨表状态键使用整数 `FE`，不使用浮点 ratio。首轮离线样本不由模型分数选择，不按 outcome 事后改权；主拟合使用 fit-fold 内计算的 `cluster_balanced_fit`，state-row 等权只作 sensitivity。阈值 Q10 邻近带只能在模型冻结后作 online 附加复查，且所有比较策略必须共享相同 decision opportunities。

RQ2 的主 Time-only 比较只使用 12 个预算里程碑：`milestone_only_T0` 与 milestone-only B3 必须在相同行上配对。事件机会的出现由 Behavior 条件决定，因此完整 dynamic schedule 上的 T0 只能作为 schedule-conditioned sensitivity。

另设估计性 `dimension_stratified_T0`：在 BBOB 10D/20D/40D 内分别完成 grouped-by-function OOF 与 threshold，每层模型仍只输入 `bf_fe_ratio`。它不进入主 Decision X，也不替换必需 T0。当前 `FE_total=1000d` 使 `log(d)` 与 `log(FE_total)` 完全共线，主 cheap planned query-cost ratio 恒为 0.05，不能制造为无信息或重复的模型列。该诊断只用于限制“Behavior 超过时间”的解释，不外推到未训练维度。

Selection Reference 将扣除 query sampling FE 后的 `remaining_budget_ratio` 作为连续输入，不使用 nearest bucket。

---

# 13. Baseline Protocol

必须包含：

## Never Query

最低分析成本。

## Always Query

在第一个在线合格机会调用一次固定 query；它不是 pre-run AAS。

## Random Analysis

`matched_rate_random`：run-level 调用率与已调用 run 的 first-trigger `FE_ratio` 经验分布只由 BBOB-train 端到端 OOF Proposed policy 冻结。每个 run 开始时预抽是否调用和目标 FE ratio，在第一个不早于目标的在线合格机会触发；不存在该机会则不触发。30 个 Random streams 先在同一 run 内平均，再进入 problem/function 聚合，不能作为独立统计复制。不得事后从完整机会集合均匀选 state，外部结果不参与 calibration。

## Traditional AAS

`pre_run_aas_fe0`：在 optimizer prefix 前、`FE=0` 执行 query，只用 query features 与剩余预算选择初始算法，并以 `B-FE_query` 运行；它是 query-only、sample-isolated 的 pre-run AAS，query sample 不初始化或扩充 optimizer population。它与 Always Query 分开运行、分开报告。该基线及 portfolio 结论只条件于仓内固定 DE/PSO/CMA-ES/SHADE 实现、冻结参数与 `population_size=40`，不代表所有 Traditional AAS 或所有 portfolio instantiations。

FE=0 AAS 的显式关系记账固定为：`prefix_algorithm=selected_algorithm`（仅表示 fresh run 的选中算法，不表示存在 optimizer prefix）、`selected_equals_prefix=true`、`handoff_required=false`、`handoff_type=fresh_optimizer_initialization`；`default_algorithm` 与 `no_query_algorithm` 均为对应 fold 的 SBS。该关系不能误写为 population transfer。

## Behavior-only Full-budget Selection

不执行 query，用 query 前 Behavior 选择动作，全部动作保留 `B-e_t`。它有两个预先区分的用途：`matched_trigger_behavior_only` 在 Proposed 的同一首次触发 state 执行动作，只用于计算 `query_operational_increment`；`self_thresholded_behavior_only` 用同名模型家族预测 $U_b$ 并从自己的 train-only first-trigger threshold 决定是否执行 Behavior Selector，是独立可部署 baseline。二者共用 full-budget 四动作 outcome 与 Behavior Selector，但不得共用 threshold 或混报。

## SBS

Single Best Solver。

SBS 在相应 fit functions 的 complete-budget endpoints 上，按 run → static problem（function × dimension × instance）→ fixed dimension stratum → function 等权聚合 clipped `log10_gap`，选择均值最低算法；并列按 `de,pso,cmaes,shade`。这与主性能端点及 function 顶层权重一致，不使用平均 rank 丢弃效应量。outer/inner/full-train 分别重算，holdout 不参与。

## VBS

Virtual Best Solver。

VBS 保留为标准静态 problem-level hindsight reference。对每个 `problem = function × instance × dimension`，先对每个算法的完整预算 clipped `log10_gap` 跨 optimizer seeds 取算术均值，选择均值最低算法（并列按 `de,pso,cmaes,shade`），再用该算法在同一 problem 上的逐 seed paired outcomes 进入 run → static problem → fixed dimension stratum → function 汇总；不得逐 seed 选择最小算法。共享状态上已运行候选动作的最小 loss 另称为 `best observed action`，不得与 VBS 或现实可部署方法混称。VBS 不保证对含动态切换的政策构成逐 run 数值上界。

## Time-only Controller

`milestone_only_T0` 输入严格为 $X=\{FE\_ratio\}$，并与 milestone-only B3 使用同名模型、同一 outer-function 链、first-trigger threshold 和 held-out function。完整 event schedule 上的 T0 另称 `schedule_conditioned_T0`。`dimension_stratified_T0` 仍只输入 `FE_ratio`，但在每个已评估 BBOB dimension 内独立拟合/比较；它是静态上下文估计性诊断，不是主控制器或外部部署 baseline。

---

# 14. Evaluation Protocol

## Optimization Metrics

- Final error
- ERT
- `target_hit_observed` rate 与 `endpoint_success` rate（分列）

## Decision Metrics

- 主选择：outer-function OOF run-level first-trigger mean `u_query_joint_lamT_1`
- 辅助：AUROC、Average Precision、Spearman
- 连续 Utility 回归：Ridge 的 RMSE
- 策略：run-level query call rate、first-trigger utility capture、precision under first calls、`query_operational_increment`、最终优化性能
- 预测诊断：同一 query-adjusted 矩阵上的 `query_feature_predictive_increment_log10_gap`（continuation-only，不含 sample best）

分类概率或判别分数不与连续 Utility 直接计算 RMSE。

所有策略共享同一 run-level hindsight capture reference：在 native SBS/default trajectory 的全部预定义合格机会取 `H_r=max_t max(U_t,0)`；策略分子只取 first-trigger state 的 `max(U,0)`，未触发为 0。聚合报告加权分子、分母、比值与 `H_r=0` runs。该离线参照不是可部署 policy，也不随策略触发时点改变。

## Resource Metrics

- FE cost
- decision-state future-path ratio 与 FE=0→terminal policy wall-clock；另报告 behavior、Decision inference、query、Selector、handoff 和 optimizer continuation 分量
- peak memory（主 $\lambda_M=0$，仅作独立端点）

---

# 14.1 Statistical Estimands and Inference

RQ1 的目标分布固定为 `descriptor_cheap_invariant`、SBS prefix 和 `phase1_dynamic_budget_event_v1` eligible states。统一聚合层级为 state → run → static problem → fixed dimension stratum → function：run 内 eligible states 等权；`function × dimension × instance` static problem 内 paired optimizer runs 等权；每个 `function × dimension` 内 static problems 等权；最后在 function 内使 10D/20D/40D 固定 strata 等权。policy endpoint 从 run 层开始同序聚合。function 是顶层统计单位；state、seed、instance 或 function × dimension 不是独立顶层复制。

BBOB-validation 的 estimand 是 F5、F9、F13、F14、F19、F24 六个固定 functions 的等权有限集均值，条件于这 6 个 functions、冻结 dimensions/instances/static problems 与固定 query realizations。历史模型比较、调参、消融与采样设计已经读取该集合，因此它是已见内部评价集；撤回旧结果不能恢复未见性。CEC2017 已有历史 online/targeted 诊断，是已见外部开发集。二者均不提供确认性证据，也不支持 function 或 transformed-instance 超总体推断。

当前 BBOB 的 3 instances × 30 optimizer seeds 与 CEC2017 的 30 seeds 没有仓内 precision/power 依据，只是固定开发期采样设计；报告必须给出实际区间宽度，不能把重复次数本身当作充分性证明。CEC2022 与工程问题必须在首次 outcome 前冻结 endpoint-specific precision target、只使用开发集合方差信息的 repeats 确定方法与最终重复数；未闭合时不得启动对应前瞻评价。

主 95% CI 固定为 10,000 次条件配对层级 bootstrap：所有 replicate 均保留全部 6 个 validation functions、全部固定 dimensions 与 instances 1/2/3 对应的全部 static problems；只在每个固定 static problem 内配对重抽 optimizer seeds。RQ1 对每个抽中 seed/run 保留完整有序 state 序列。不得在主 CI 中重抽 function 或 static problem。有放回重抽 6 个 functions 的结果只作函数组成敏感性，必须单独命名，不得替代主 CI、恢复确认性或扩张为 function/transformed-instance 超总体区间。

ERT 使用专用 paired hierarchical ratio bootstrap。主区间条件于固定 functions、fixed dimensions、instances/static problems 与 query realizations：每个 replicate 保留全部 static problems，只在每个固定 problem 内联合配对重抽 optimizer runs，分别重算 treatment/reference 的 `N_FE` 与 `N_hit`，计算 $ERT=N_{FE}/N_{hit}$ 及 `log10(ERT_treatment/ERT_reference)`；之后对 fixed dimensions 等权形成 function effect，再对 fixed functions 等权。function resampling 只允许作为单独命名的函数组成敏感性，不得替代主区间或产生 function/transformed-instance 超总体推断。

零命中按扩展实数语义处理：仅 treatment 零命中为 $+\infty$，仅 reference 零命中为 $-\infty$，双方零命中为 undefined；同一聚合层同时含 $+\infty$ 与 $-\infty$ 也为 undefined。任何 stratum 或 replicate 都不得静默删除。bootstrap 输出必须分别保存 finite、$+\infty$、$-\infty$ 与 undefined 的 stratum/replicate 质量。undefined 在扩展实数线上不可排序，区间计算按 `conservative_two_tail_allocation_on_extended_real_line_v1` 将其质量保守分配到两端：若 undefined 质量达到单侧 $\alpha/2$，区间为 $[-\infty,+\infty]$；若全部 replicate 均 undefined，区间界为 undefined，观测 contrast 则独立按完整固定样本计算。`interval_established` 只由观测 contrast 是否有定义以及按上述质量规则得到的两个分位点是否有定义决定；有限或无界区间都可 established，不能因为任意一次 replicate 出现单方或双方零命中就自动置为 false。必须同时报告 `interval_status`、`interval_unbounded`、undefined mass、各类零命中计数与 defined replicate 数。绝对 ERT 逐 function × dimension 报告；不同 dimensions 的 raw FE 不得先池化为总体 ERT。若另报总体绝对量，只允许预先定义的 budget-normalized ERT 有限集汇总，且不替代主 ratio。

代码中的专用 ERT ratio/strata 函数目前尚未接入 suite-level attempted denominator/coverage、双向 failure sensitivity 与报告表 consumer；这项 wiring 是正式推断 blocker。仅生成 complete-pair ERT ratio 不能视为该 endpoint 已闭合。

Utility ±0.01、`log10_gap` ±0.05、runtime ratio `[0.95,1.05]` 及 call/target-hit-rate 差 ±0.05 只称“项目内预设 operational tolerance”；项目内尚无独立领域依据将其称为 confirmatory equivalence。BBOB-validation 与 CEC2017 使用条件区间逐项描述相对 tolerance 的位置，非显著差异不建立等价，Utility 内的 gap/runtime 抵消也不建立单独 endpoint 等价。未来 untouched external suite 若预列 simultaneous intervals，其 family 与 interval level 必须在首次 outcome 前冻结；第一篇论文仍不据项目内 tolerance 作确认性等价声明。

RQ2 唯一主要科学 contrast 是冻结模型家族的 milestone-only B3--T0。BBOB-validation 与 CEC2017 分别给已见内部/外部开发估计；确认性证据只来自配置、factory/runner、端点与 contrasts 全部闭合后首次运行的 CEC2022 与工程集合。LDA、Logistic Regression 与 Ridge 的 train outer-OOF 两两 contrasts 只作选模诊断。

RQ3--RQ5 在六函数 BBOB-validation 上全部是有限集估计性分析。paired sign-flip 依赖固定 function effects 的 signs 在零假设下可交换这一额外假设；六函数双侧 exact raw p 最小为 0.03125，RQ3 与 RQ5 各自六 contrast Holm family 的最小 adjusted p 均为 0.1875，在 0.05 下不可能拒绝。RQ4 按 suite 与 endpoint 分开，不把四个 suites 组成同一 Holm family；未来某 suite 内若有多个 contrasts，须在首次 outcome 前单独冻结 family。逐 function/problem effects、固定有限集均值、条件 95% CI、coverage 与失败敏感性是主证据；sign-flip/Holm 若保留，只进入 assumption-sensitive 辅助表，不承担 RQ 成败判据。未拒绝不表示等价或无效，任何 p 值都不产生函数超总体推断。

---

# 15. Core Research Questions

## RQ1

主 `descriptor_cheap_invariant` query/full-Selector 联合路径在所评估机会分布中的净收益如何？

验证：

`U_{query}^{joint}\leq0` 的状态比例、效应量与区间；在全 eligible states 与 Proposed-triggered states 分别报告 `query_operational_increment`，不得以联合效用替代 Query 相对 full-budget Behavior-only 的操作性净增量。`query_feature_predictive_increment_log10_gap` 只作同 query-adjusted 预算下的 OOF Selector 输入诊断。

---

## RQ2

在 milestone-only 机会中，Behavior 是否能在 FE ratio 之外预测 run-level first-trigger 联合效用？

---

## RQ3

Decision-before-Feature 是否减少无效 query 调用？

---

## RQ4

是否具有跨benchmark泛化能力？

---

## RQ5

哪些预定义 Behavior 组与效用存在稳定预测关联？Search Maturity 只作为确定性派生基函数评价，不预设有效，也不作因果解释。

---

# 16. Required Ablations

正式 feature-group 消融固定为：

| 组 | 输入范围 | 字段数 |
|---|---|---:|
| T0 | 仅 `bf_fe_ratio`，即 $X=\{FE\_ratio\}$ | 1 |
| B1 | core permutation-invariant behavior | 19 |
| B2 | B1 + DynamoRep-lite longitudinal set dynamics | 25 |
| B2+Motion | B2 + 3 set-motion fields | 28 |
| B2+Maturity | B2 + 3 deterministic maturity fields | 28 |
| B3 | B2 + movement/maturity behavior | 31 |

六组使用相同的 Decision dataset、outer-function 链、同名模型、train-only preprocessing 与 run-level first-trigger threshold 过程。`B2+Motion` 与 `B2+Maturity` 是兄弟组，用于区分 motion 与派生 Maturity；不得按 observed result 选择其中一组作为新的主输入。`all_candidates` 仅是 B3 的兼容别名，不含 3 个 `diagnostic_only` 字段。算法特定参数仍禁止进入 Decision 输入。`dimension_stratified_T0` 是额外估计性静态上下文诊断，不是第七个 Decision 输入组。

---

# 16.1 Failure Rules

- 所有计划 run 先进入 coverage denominator；失败记录不得按结果方向删除。
- Decision score 缺失时该机会按 No-query；query 触发后发生 feature/selector failure 时，已消耗的 query FE 与时间保留，并以 query-adjusted budget 原生 `continue_current`。
- BBOB train/validation 与 CEC2017 固定 `failure_loss_cap=1e20`、取 `log10` 前 raw-gap floor/cap `1e-12/1e20`、`success_gap_target=1e-8`、单 state-action path timeout `3600 s`，并逐 objective evaluation 记录 `observed_first_hit_FE`。timeout/failed path 保留为路径失败；若失败前已经命中，`target_hit_observed=true` 且标准 ERT 保留 observed first hit，同时 `endpoint_success=false`。只有未命中项的 ERT contribution 计该路径完整 planned budget。failure cap 只用于保留有限 target 与双向极端 sensitivity，不得掩盖 failure/coverage/ERT。CEC2022/工程问题必须在首次前瞻评价运行前冻结同类字段；工程集合还必须冻结 constraint rule，缺少配置/factory 时不得启动。
- 缺失状态键或配对 outcome 是数据生成错误，必须重生成对应 shard，不作统计插补。
- 每个 suite × endpoint 报告 attempted denominator/coverage、失败率、complete-pair estimate 与双向极端 failure sensitivity。未闭合 pair 的 favorable/adverse 值运行前固定：gap 用 floor/cap；`target_hit_observed` 用 1/0；ERT adverse 未命中 contribution 用 full planned budget，ERT favorable 命中 contribution 用该行在已知 prefix、已计 query FE 与 endpoint 时间原点下最早可行的 objective-evaluation index（prefix 已命中则保留其实际 first hit，无 prefix 信息的 FE=0 全路径下界为 1）；runtime 用该 suite complete pairs 的最小正值/timeout。若最早可行 first-hit FE 或最小正 runtime 不可重建，相应 sensitivity 为 undefined。Utility 从同一组极端 endpoints 重算。科学 path failure 已按 cap/timeout 保留时不当作 missing pair。只要效应方向、区间相对 operational tolerance 的位置任一改变、coverage<95%，或 sensitivity undefined，对应结论即未建立。

---

# 16.2 Computational Scenarios and Pre-run Checklist

BBOB train + validation 包含 25,920 个基础 optimizer runs、至少 311,040 个 mandatory states 和 0.6048B 基础 FE。Query sample artifact 按 `problem × sample_design` 生成一次，不能按 state/prefix 重复计入 action-loss producer。12 个 mandatory milestones 的平均 prefix ratio 为 0.35、平均 future-path ratio 为 0.65；旧 `FE_prefix=0.60B` 单点假设把 future-path FE 低估为真实 mandatory 平均的 $0.40/0.65=1/1.625$。在无 event/failure 的算术情景中，跨 Stage-A matrices 共享一次 Skip/Behavior `continue_current` 但不复用基础 trajectory时，Stage A 为 main cheap 37.467612B FE、三档 53.434836B FE；若逐行证明基础 trajectory 终值同义并复用，分别为 32.750172B/48.717396B。当前 main producer 每个 action-budget CLI 另执行一次 Skip，Stage A 为 46.902492B；三档 current-producer 量等待实际调用图枚举。

主 replay 每个 role 只走一个 fold-specific SBS prefix；5 outer × 4 inner 加 full-train→validation 使 train state 有 22 个逻辑 roles、validation state 有 1 个 role，预算加权 role-state basis 为 30.3912B。Stage-B 三次 selected future-path timing 加主 pre-run AAS timing 为 main cheap 178.24212B FE、三档 296.7678B FE。与 Stage A 合并后，仅跨 matrices 共享时为 215.709732B/350.202636B；进一步复用基础 trajectory 时为 210.992292B/345.485196B；保持当前 main producer 时主 query 为 225.144612B，三档 current-producer 量仍待枚举。完整 online evaluator 每个 base tuple 有 7 条固定政策和 30 条 Random，每条 1 次 Stage-A 加 3 次 Stage-B；现 CEC2017 配置另需 11.5884B planned FE，BBOB-validation 全 instances 另需 5.5944B FE，但当前 evaluator 不支持后者。上述仍不含 event-only/更早 states、失败、CEC query/VBS、CEC2022、工程问题或额外 query replicates，只能称 mandatory-only 算术情景，不能称严格下界或资源可行性证明。

22 个 fold roles 是逻辑评价依赖，不自动等于 22 条不可共享的科学 outcome；Stage-A 四动作矩阵已经允许不同 roles 读取同一实际 action outcome。当前 Stage-B 资源情景仍按每个 role 的完整 selected path 计费，因为主 timing estimand 要求真实执行对应 Selector inference 与完整 future path。若要按 `state × matrix × actual action` 去重物理 continuation，必须先冻结逐 repetition 的组件合成、cache/order、censoring 与 artifact-specific inference 计时规则，并证明它与完整路径 wall-clock 的目标一致；在此裁决前不得把较小去重量写成正式预算。Planner 已有枚举能力，但 grouped-by-function Selector artifact 路由、runner、物化实测 plan、Stage-A 共享/复用、BBOB instance-aware online endpoint、timing、资源与排期仍是 blockers。

正式运行前必须确认：

1. trajectory 与 reservoir snapshot 连接到同一 emitted integer FE，终点保存不再要求伪造 decision state；
2. reservoir replacement randomness 只由显式整数与 `numpy.random.SeedSequence` 构造，时间截断保留完整 native update；
3. 固定 query seed/operator、三个活动 query ID、sample design、`query_first_hit_offset` 和 FE charge 一致，主 Query terminal gap/`target_hit_observed`/ERT 计入真实 sample evaluations，`endpoint_success` 另与 continuation completion 联合定义，而 Selector 诊断使用 continuation-only outcomes；本轮不增加 LHS replicates 或 action losses；
4. Query-adjusted 与 full-budget 两套 action outcome 语义唯一，允许复用的 Skip/5% outcome 只计算一次；
5. 每个 outer/inner fold 的 SBS、两类 Selector、三类 Utility、Decision 与 thresholds 均只读对应 fit functions；主 fit 完成 cluster-balanced wiring，row-weighted fit 只作 sensitivity；
6. Proposed、T0、`dimension_stratified_T0`、self-thresholded behavior-only、Always、Random 和在线汇总均为 run-level first-trigger；
7. `pre_run_aas_fe0` 在 FE=0 独立生成，online `FE_query` 只按实际触发路径收费；
8. replay planner 目前仅有枚举能力；offline decision-state-to-terminal runner 尚未实现，fold-role-complete plan 尚未物化实测并核对，均为正式运行 blocker。runner 实现后还必须把 Stage-A 科学端点与 Stage-B timing-only 字段分开，保存三次 future-path wall-clock 与独立 FE=0 policy wall-clock、`cyclic_complete_path_v1` order、线程、机器、组件、原始时间、逐次 status/effective FE/timeout/completion、完成端点一致性与 instability；三次预定重复不得选择性重跑；
9. 72 个 trajectory/final-performance pair、reference/final gap/log10 gap、`observed_first_hit_FE`、`target_hit_observed`、`path_completed`、`endpoint_success`、失败字段、cap/floor/timeout 与 coverage denominator 完整，trajectory/Behavior 不含这些 outcome 输入字段；ERT 使用逐 function × dimension 重算的专用 ratio bootstrap，零命中 stratum 不删除；
10. CEC2022 的配置及工程集合的配置/factory/constraint endpoints 在首次确认性 outcome 前补齐并冻结；活动输出不读取撤回产物，BBOB-validation/CEC2017 不标为确认性。

任一项未确认时不得启动正式矩阵。上述数字是计划下界，不是已经完成的运行量。

---

# 17. Repository Specification

    decision_before_feature/

    ├── configs/

    ├── benchmarks/

    ├── optimizers/

    ├── trajectory/

    ├── behavior/

    ├── landscape_queries/

    ├── utility_labels/

    ├── selection_reference/

    ├── decision/

    ├── experiments/

    ├── evaluation/

    └── results/

---

# 18. Vibe/Codex Development Rules

禁止：

1. 修改benchmark split；
2. 使用test数据训练；
3. 输入query feature到Decision Model；
4. 输入algorithm-specific parameter；
5. 未记录配置新增实验；
6. 删除失败实验结果。

---

# 19. Development Order

Phase 1:

Trajectory Collector

Phase 2:

Behavior Extractor

Phase 3:

Offline Utility Label

Phase 4:

Decision Model

Phase 5:

OOD Evaluation

Phase 6:

Paper Experiment Reproduction

---

# 20. Final Research Statement

本文拟检验以下受限证据链，而不是预设其成立：

    Optimization Experience

    ↓

    Algorithm-agnostic Search Behavior

    ↓

    Prespecified Behavior / Maturity Basis

    ↓

    Query Utility

    ↓

    Analysis Selection

    ↓

    Resource-efficient Algorithm Selection

若正式结果支持，核心方法结论可写为：

> For the evaluated query, portfolio, state distribution, budget and benchmark suites, query acquisition can be treated as a state-conditional resource-aware decision.

方法层贡献是把该决策显式化；性能与资源方向仍须由 run-level 联合策略效用、各 terminal endpoints 与 `query_operational_increment` 共同建立。只出现联合路径优势而 query 操作性增量不高于零时，只能说明该固定 Query/Selector 路径优于 SBS continuation，不能说明 landscape descriptors 本身值得获取。

该结论只覆盖 `descriptor_cheap_invariant` 主配置；若 `pflacco_standard_invariant`/`pflacco_broad_invariant` 方向一致，只能进一步表述为“在三个预定义 query 配置上具有稳健性”。若方向不一致，必须报告 representation dependence，不能重新定义 query 或选择性隐藏结果。
