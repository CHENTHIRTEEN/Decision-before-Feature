# Decision-before-Feature 开发前裁决

本文档记录已经确认的项目级裁决，用于解决 `AGENTS.md` 与 `docs/` 研究设计文档之间的约束冲突。

优先级规则：

1. `AGENTS.md` 是最高优先级。
2. 本文档用于解释如何在开发中落实 `AGENTS.md` 和研究文档。
3. 若 `docs/` 中早期方案与本文档冲突，开发时按本文档执行。

当前状态（2026-08-15）：优化器 continuation 已改为完整状态原生推进。正式方法进一步分开联合策略效用与 query 操作性增量，增加 `behavior_only_full_budget`，并把 SBS、Selector、Utility、Decision 与 inner threshold 纳入同一 outer-function 链；部署和模型选择统一使用 run-level first-trigger。此前由重建式 continuation 或旧逐行 threshold 口径生成的 trajectory 下游产物、utility labels、Decision dataset、模型和评价结果均无正式证据资格，必须从 trajectory/action-loss 依赖位置重新生成。preliminary/MVE 口径仍退出当前运行面。

## 0.1 本轮八项修改的科学影响

下表先于源码与其他协议修改冻结。`新增 action losses` 仅指是否需要新增真实
`state × action` continuation；新增 Selector、selected-path replay 或在线政策运行不等同于新增四动作矩阵。

| 项目 | 改变的研究问题或证据角色 | 新增 action losses | 数据泄漏影响 | 对已有产物的影响 |
|---|---|---|---|---|
| 1. 主 estimand 与 matched acquisition | RQ1 继续以 Query/full-Selector 相对 native SBS 的联合效用为主；新增 `query_matched_state_only` 与 `sampling_only_continue_current`，分别识别 descriptors 对动作选择的操作性增量和 query sampling 的直接终点贡献 | 否；两者复用 Query-adjusted 四动作矩阵，新增 selected-path timing 与在线政策 outcome | 不增加可用信息；两个对照必须使用同一 fold-fit Selector、同一 query realization、同一 FE/runtime/sample-best 记账 | 旧 Utility、Decision labels、replay plan、baseline 与在线汇总全部失效；Stage-A 四动作 outcome 可在新 target 校验后复用其原始科学端点 |
| 2. trajectory first trigger | RQ2、RQ3 与全部政策比较统一回答“每条 trajectory 是否以及何时首次调用一次 query”，不再把调用后的不可达状态当独立决策 | 否 | 消除 post-trigger states 进入 threshold、模型选择和策略指标的偏差 | 旧逐状态 threshold、call rate、precision、utility capture 与 baseline 汇总失效 |
| 3. 端到端 function-level 嵌套 | RQ2 的模型选择及 RQ3 的政策估计改为 outer-fold-specific SBS → Selectors → Utility → Decision → inner first-trigger threshold 的完整链 | 否；不同 fold role 只读取 fit functions 上拟合的组件，可共享相同原始 action outcomes | 降低；outer/inner holdout function 不得进入 SBS、Selector、label、preprocessing、模型、threshold 或 Random 校准 | 任何从完整 train 预生成固定 Utility 后只嵌套 Decision 的产物失效 |
| 4. lambda、性能与时间端点 | RQ1/RQ3 的主 operating condition 固定 `lambda_time=1`；terminal `log10_gap`、decision-state future-path runtime ratio 与 FE=0 policy wall-clock 分列。其他 lambda 仅对同一冻结主政策重计分，并报告 break-even lambda | Stage-A 否；Stage-B 需要三次预定真实计时 | 不按 validation/external outcome 选择 lambda；计时重复、顺序、timeout 与删失规则在 outcome 前固定 | 旧一次计时、raw-gap scale、线性相对时间和按 lambda 重选有利政策的产物失效 |
| 5. RQ1 与统计推断 | 冻结 state → run → static problem → fixed dimension stratum → function 的有限集 estimand、条件层级区间、项目内 operational tolerance、同时区间 family 与辅助多重比较 | 否 | 不改变训练信息；统计 consumer 只能读取冻结政策的 held-out/外部行 | 旧 row-level 独立区间、随机 instance split、function 超总体表述和未定义 family 的 p 值失效 |
| 6. baseline、消融与失败 | RQ2 增加相同 milestone rows 的 T0；RQ3 使用 matched-rate Random、FE=0 pre-run AAS、Behavior-only 与 matched-acquisition；RQ5 分离 Motion/Maturity；RQ4 使用 suite-specific failure endpoints | FE=0 AAS 使用其独立四算法 outcome；其余新增对照复用现有矩阵 | Random 校准只读 train OOF；pre-run AAS、T0、消融和失败规则均不得读取 validation/external outcome 调整 | 旧 per-opportunity Random、把 Always Query 当 Traditional AAS、合并 Motion/Maturity 与 complete-case-only 外部结果失效 |
| 7. trajectory/query/online/seed 实现 | 不改变 RQ；使实际 producer 与前六项 estimand 一致。native RNG 增加 suite/function/instance/dimension 整数，query/online failure 物化为计划行 | 否，但 RNG 改变所有新轨迹与 continuation | 同一 static problem 内政策继续共享 state/RNG；不同 static problems 不再复用同一 native stream，匹配当前 problem 内 bootstrap | RNG 改变使全部历史 trajectory、checkpoint、action outcome 及其下游产物失效；不得混用旧行 |
| 8. 计算量与运行前检查 | 不改变 RQ；区分 fold role 的逻辑依赖与物理执行。物理 timing 单元按 `state × budget mode × query configuration × actual action × repetition` 共享 | 否；减少重复执行同一真实 path，不减少任何唯一科学 action | fold-specific prediction 与 artifact 路由仍独立；只共享输入、动作、预算和 RNG 完全相同的物理 continuation | 旧 22-role 全路径重复预算与未物化 replay plan 失效；Stage-A 科学端点不因共享被改写 |

跨项裁决：Selection Reference 主 target 改为
`clipped_log10_gap_advantage_vs_continue_current`。对状态 (s) 和动作 (a)：

$$
Y_{s,a}=\ell_{s,a}-\ell_{s,\mathrm{continue}},\qquad
\ell_{s,a}=\log_{10}\!\left(\min(\max(L_{s,a},g_{\min}),g_{\max})\right).
$$

它与主 terminal `log10_gap` 风险同单位，`continue_current` 的 target 恒为 0；原
`statewise_minmax_observed_action_loss` 只保留为 Selector target sensitivity。该改动不新增 continuation，
但所有 Selector、selected action、Utility、Decision 与 replay plan 必须重生成。

---

## 1. 验证方式

裁决：

- 不建立 `tests/` 目录。
- 不引入 `pytest`、测试依赖、JSON Schema 或 schema registry。
- 使用真实小规模实验运行、数据质量检查和一致性检查验证模块。

允许：

- 轻量字段定义。
- 运行前后的数据质量检查。
- 真实 benchmark 上的最小可验证实验。

禁止：

- dry、smoke、synthetic validation。
- 与真实科学运行无关的替代工作流。
- 用测试框架替代真实实验验证。

---

## 2. 数据字段约定

裁决：

- 可以使用轻量字段定义来描述 trajectory、behavior、utility label 的列。
- 不实现 JSON Schema、schema registry 或任何强制注册系统。
- 文件名和模块名避免使用会暗示治理机制的命名。

开发口径：

- `schema` 一词只表示普通数据字段约定。
- 不用于文件身份、完整性、授权、运行许可或执行解锁。

---

## 3. Oracle 术语

裁决：

- 代码目录不使用 `oracle/` 作为主模块名。
- `Query Utility Oracle` 在开发中改称 `utility_labels` 或 `offline utility label generation`。
- `Selection Oracle` 在开发中改称 `selection_reference`。
- VBS 保留为静态 hindsight reference，不作为现实可部署方法，也不保证对含动态切换的政策构成逐 run 数值上界。

推荐目录：

```text
utility_labels/
selection_reference/
portfolio_reference/
```

避免目录：

```text
oracle/
selection_oracle/
oracle_generator/
```

---

## 4. 共享前缀配对续跑

裁决：

- 不使用“反事实”或 `counterfactual` 描述共享状态上的多分支完整运行。
- 使用“共享前缀配对续跑”或 `shared-prefix paired continuation`。
- 不提出因果主张。

开发口径：

```text
same complete optimizer checkpoint state
-> No-query: use the train-derived SBS/default; continue natively when default == prefix, otherwise use one population-transfer initialization
-> Run Query, same algorithm: native continuation with the query-adjusted budget
-> Run Query, changed algorithm: one population-transfer initialization
-> paired comparison
```

完整 checkpoint state 必须包括 population、fitness、generation、best-so-far、优化器内部动态量和 RNG state。不得把仅含 population/fitness 的算法重建称为同算法 continuation。

---

## 4.1 逐状态 Selection Reference

裁决：

- 不再用 problem 级静态最佳算法标签和 nearest performance bucket 生成正式 Selection Reference。
- 对每个共享状态运行唯一动作集合：`continue_current` 加其余三个 portfolio algorithm；同算法使用完整状态原生 continuation，跨算法使用一次 population transfer。
- 保存每个 `state × action` 的 observed final loss；逐状态最小值称为 `best_observed_action`，不称为 oracle。
- Selector 使用固定多输出 Random Forest 预测相对 `continue_current` 的截断 `log10_gap` 差，并把 `remaining_budget_ratio` 作为连续输入；主产物字段固定为 `selector_target_transform=clipped_log10_gap_advantage_vs_continue_current`，`continue_current` 的 target 恒为 0。`statewise_minmax_observed_action_loss` 只作预设 target sensitivity，不生成主 selected action 或下游标签。
- Selector 可使用 query features、permutation-invariant algorithm-agnostic behavior 和连续剩余预算；这些 selector 输入不改变 Decision Model 的禁止输入边界。
- 正式链路必须生成两套四动作 outcome：Query-adjusted budget 使用 `FE_total-FE_prefix-FE_query`；`behavior_only_full_budget` 不执行 query，使用 `FE_total-FE_prefix`。两套 outcome 不得互相冒充或复用 loss。
- Query Selector 输入为 query features、permutation-invariant behavior 与 query-adjusted `remaining_budget_ratio`；`behavior_only_full_budget` Selector 只输入 behavior 与 full-budget `remaining_budget_ratio`。
- 正式 train Selection Reference 输出必须来自 grouped-by-function cross-fitting；validation 和外部 benchmark 只允许使用 BBOB train 拟合的最终模型。Decision 外层评价时，Selector 还必须在对应 outer-fit functions 内重新拟合，outer holdout function 不得影响 SBS、Selector 或 Utility。历史数据字段 `family=bbob_fNNN` 只保存 function-ID grouping key，不表示经典 landscape family。
- 原静态 bucket classifier 只可作为被替代方法诊断，不得继续生成正式 Utility 标签。

Utility 分解口径：

```text
potential_gain_raw = loss_noquery - loss_best_observed
selector_regret_raw = loss_selector - loss_best_observed
performance_gain_raw = potential_gain_raw - selector_regret_raw
```

Population transfer 的影响已经包含在各 action 的 observed loss 中，不能在主 Utility 再扣一次 `handoff cost`。Query sampling FE 已通过减少 Query 分支后续优化预算体现，也不能重复扣除；主 Utility 只额外扣除尚未进入 performance loss 的 time/memory 等成本。

## 4.2 联合策略 estimand 与 query 操作性增量

裁决：

- `u_query_joint_lamT_1` 是第一篇论文的主 Decision target，定义“执行固定 query 并调用 full Selector”相对原生继续 fold-specific SBS 的联合路径净差。
- 该量同时包含 query acquisition、Selector 误差、动作切换和 continuation，不得称为 query features 的独立边际价值。
- `behavior_only_full_budget` 是 query 前即可部署的行为选择路径；它不执行 query，四个动作都保留完整剩余 FE。
- `query_operational_increment_lamT_1` 比较 Query/full-Selector 路径与 `behavior_only_full_budget` 路径，回答“在已有 Behavior 决策路径上增加固定 query 后的操作性净增量”。它仍是固定模型、预算和 transition rule 下的预测性 estimand，不作因果解释。
- `I_q` 包含 query FE、采样与描述符 runtime、sample best、较短 continuation budget 以及 Selector 差异；因此它不是纯信息效应，也不是因果 estimand。活动字段统一命名为 `query_operational_increment_lamT_*` 与 `query_operational_increment`，任何读取器或表格不得继续输出旧字段名。
- `I_q` 必须同时在全部 eligible states 与 Proposed first-trigger states 上报告。前者描述冻结状态分布上的操作性增量，后者描述实际调用策略所选择状态上的增量；不得只报告其中更有利的一种。
- 若 `u_query_joint_lamT_1 > 0` 但同一范围的 `query_operational_increment_lamT_1 <= 0`，证据至多支持 Query/full-Selector 联合路径优于 native SBS，不能支持 query acquisition 优于已有的 full-budget Behavior-only 路径。
- `query_adjusted_state_only_selector` 与 full Query Selector 在同一 query-adjusted 四动作 outcomes 上的 OOF 比较另记为 `query_feature_predictive_increment_log10_gap`。它只比较两者所选动作的 continuation-only `log10_gap`，不增加 action-loss 运行、不计 query sample best，也不扣 acquisition cost；因此只回答 query features 对动作预测的边际贡献，不能替代 full-budget 的操作性 `query_operational_increment_lamT_1`。
- query-only/full Selector 的其余比较继续用于信息来源诊断，但不能替代上述两个定义不同的增量。
- 正式加入 `query_matched_state_only`：它执行与 `query_joint` 相同的 query realization，使用相同 sample endpoint、query-adjusted 四动作 outcomes、剩余 FE 和计时起点，但由不含 query descriptors 的 state-only Selector 选动作。`query_descriptor_use_increment_lamT_* = U_query_joint-U_query_matched_state_only`，用于评价在固定 acquisition 下使用 query descriptors 的操作性增量。
- 正式加入 `sampling_only_continue_current`：它执行相同 query acquisition 和 sample endpoint，但原生继续当前算法。`query_state_only_vs_sampling_increment_lamT_* = U_query_matched_state_only-U_sampling_only_continue_current`，`query_sampling_direct_increment_lamT_* = U_sampling_only_continue_current`。逐行必须满足 `U_query_joint = query_descriptor_use_increment + query_state_only_vs_sampling_increment + query_sampling_direct_increment`。
- 上述 matched-acquisition 分解比三路径设计减少了表示、采样和动作选择的混合，但仍只是在固定模型、query realization、预算和 transition rule 下的操作性分解；不得写成 descriptors、sample realization 或动作选择的因果效应。

## 4.3 固定 query seed 与条件 estimand

- query sampling 是显式整数 `sample_seed` 定义的确定性算子：base seed、stream code、function number、instance、dimension 与 sample-design code 一起进入 `numpy.random.SeedSequence`。每个静态 problem × sample design 只生成一个 problem-keyed LHS realization；optimizer seed、decision state 与 action 不改变该 realization。
- 当前主结果严格条件于这组固定 query realizations，不估计对重复 LHS sampling randomness 的期望，也不得声称对 sampling randomness 稳健。本轮冻结为不增加 LHS replicates、不新增相应 action losses；未来若研究 sampling randomness，必须作为新的完整敏感性实验另行预设，而不能复用本轮条件 estimand 的结论。
- cheap 与 standard 共享同一 `lhs_50d`，可比较 representation；broad 同时改变 representation、sample size、sample realization 与 action budget。standard--broad 差异只能解释为整套 query configuration 的差异，不能拆分归因。

对应协议见：

`docs/10_protocols/Decision-before-Feature_逐状态动作损失Selection Reference修订.md`

---

## 5. Train / Seen Evaluation / Untouched External Confirmation

裁决：

- Train: BBOB, 10D / 20D / 40D。
- Seen internal evaluation: BBOB-validation，按第 5.1 节 function-level trajectory 协议执行；历史开发已读取其 outcome，因此不再具有确认性资格。
- Seen external development: CEC2017；历史 online/targeted 诊断已经读取其 outcome，只能报告外部开发集有限集估计。
- Untouched external confirmation: CEC2022 与工程集合。二者只有在本次协议冻结后补齐并冻结 suite/problem、dimension、budget、seed/repetition、reference/constraint rule、失败端点、runner/factory 和 contrasts，且此前未生成或查看任何 outcome，才具有确认性资格。

禁止：

- 用 CEC2022 或 engineering confirmatory outcomes 调参；CEC2017/BBOB-validation 若触发方法改动，必须先记录改动并重新冻结，再只由 untouched external suites 评价。
- 用测试函数训练 Decision Model、selection reference 或 threshold。
- 随机 function instance split。

---

## 5.1 正式 BBOB trajectory 数据采集协议

裁决：

- BBOB trajectory 采集只使用 COCO `bbob` suite。
- 不使用手写 benchmark 函数作为 BBOB 替代。
- BBOB function-ID group 以函数编号为单位，记为 `bbob_f001` 至 `bbob_f024`。
- `problem_id` 保存具体问题粒度，例如 `bbob_f001_i01_d10`。
- 历史字段 `family` 保存 function-ID grouping key，例如 `bbob_f001`；它不是经典 landscape family 标签。
- 禁止按 instance、seed 或 dimension 随机拆分 train / validation。

正式 grouped-by-function / function-level split：

```text
train:
1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 15, 16, 17, 18, 20, 21, 22, 23

validation:
5, 9, 13, 14, 19, 24
```

正式维度：

```text
train dimensions:
10, 20, 40

validation dimensions:
10, 20, 40
```

50D 裁决：

- 当前 COCO `bbob` suite 不支持 50D。
- 早期文档中的 BBOB 50D validation 视为历史方案，不进入主协议。
- 如需 50D / 100D 泛化，必须另设扩展实验并选择 COCO 支持的 suite，不得混入主 BBOB validation。

正式采集重复设置：

```text
instances:
1, 2, 3

optimizer seeds:
1 ... 30
```

预算口径：

```text
FE_total = 1000 * D

10D: 10000
20D: 20000
40D: 40000
```

人口规模：

```text
population_size = 40
```

理由：

- 上述预算均可被 `population_size = 40` 整除，便于保存完整 population checkpoint。
- 状态采样使用 FE ratio 监测网格，不使用固定 FE 间隔。

正式采样协议：

```text
sampling_protocol = phase1_dynamic_budget_event_v1
monitor_grid = 0.20 ... 0.60, step 0.01
budget_milestones = 0.20, 0.22, 0.24, 0.26, 0.28,
                    0.30, 0.34, 0.38, 0.42, 0.46,
                    0.50, 0.60
```

裁决理由：

- preliminary 覆盖分析显示 `selected_equals_default=false` 且 `U_query>0` 的主要机会集中在 `FE_ratio=0.30-0.55`；该观察只用于冻结采样范围，不作为正式结果。
- early `[0.20,0.30)` 使用 `0.20/0.22/0.24/0.26/0.28`，mid `[0.30,0.50)` 使用 `0.30/0.34/0.38/0.42/0.46`，late `[0.50,0.60]` 使用 `0.50/0.60`；这些里程碑不与任何 performance bucket 绑定。
- `0.60` 保留为机会区之后的衰减参照。
- very early checkpoints 例如 `0.005-0.15` 和 late endpoints 例如 `0.75/1.00` 不进入正式 phase1 主采样频率；如需研究 early/late 行为，应另设扩展实验。
- 事件触发包括 `improvement_resume`、`stagnation_onset`、`rank_change`、`elite_migration` 和 `diversity_recovery`；每个跨过至少一个 `0.01` 监测网格的完整原生 update 只判定一次事件。同一 update 跨过多个监测点时，若包含预算里程碑，则以该里程碑作为合并行的名义节点，里程碑与多事件合并为一行且不消耗 event-only 配额、最小间隔锚点或 `event_index_in_phase`；若不含里程碑，则以最新跨过的监测点作为名义节点。冻结的 `population_size=40` 与 `FE_total=1000D` 保证一次 update 的 ratio 跨度不超过 `0.01`，因而不会同时跨过间距至少 `0.02` 的两个预算里程碑。每阶段最多 2 个 event-only 状态，event-only 实际 ratio 间隔至少 `0.02`，每个 run 输出 12–18 个状态。被 gap/quota 抑制落盘的事件 crossing 仍推进再武装状态。

在线测评行为采样口径：

- 在线测评中的行为采样频率定义为 `decision-check frequency`。
- 每个采样点同时是 behavior observation 点，也是 controller、Random Analysis 和 Always Query 可以触发固定 query 的决策点；一个 run 首次调用 query 后不再产生可达的 query 决策。
- RQ2 的主 Time-only 比较固定为 `milestone_only_T0`：T0 与配对的 B3 只使用 12 个预定义预算里程碑。事件触发机会本身含有 Behavior 条件，因此事件机会上的 T0 只能作为 `schedule_conditioned_T0` 敏感性分析，不能用于“超出时间本身”的主张。
- 增加估计性 `dimension_stratified_T0`：在 BBOB 10D/20D/40D 内分别执行 grouped-by-function OOF 与 threshold 链，每层仍只输入 `bf_fe_ratio`。它不进入主 Decision X，也不替换必需 T0。当前 `FE_total=1000d` 使 `log(d)` 与 `log(FE_total)` 完全共线，主 cheap planned query-cost ratio 恒为 0.05，三者不能作为同一 pooled baseline 的可识别独立列。只有 B3 相对 milestone-only T0 与该逐维度诊断方向一致，才能把观察限定为不能仅由已评估维度层解释。
- 主在线策略比较中的每一对策略必须使用相同的合格机会集合；milestone-only 与完整 dynamic schedule 不得混在一个配对效应中。
- 正式首轮离线训练样本不由模型分数决定，也不按结果事后改权。主拟合使用第 16.2 节冻结的 `cluster_balanced_fit`；每行 `sample_weight=1` 只保留为 `row_weighted_fit` sensitivity。
- 只有在模型与 `oof_utility_first_trigger` threshold 冻结后，online 附加复查才可使用阈值邻近带；带宽为完整 BBOB-train grouped-by-function OOF 上 `abs(score-threshold)` 的第 10 百分位数，BBOB-validation 与外部套件不得拟合该带宽。该附加机会必须同时提供给所有被比较策略。
- 增加状态观测不得重新初始化算法，也不得改变同 seed 的原生搜索轨迹。

Run-level first-trigger 口径：

- 对每个 run，按整数 `FE` 与 `decision_opportunity_index` 排序；策略只在最早满足 `score > threshold` 的状态执行一次 query。
- 未越阈值的 run 贡献 0 joint decision utility；已触发 run 只贡献首次触发状态的 observed joint utility。首次触发后的状态在该策略下不可达，不进入 call rate、precision、utility capture、threshold 或模型选择。
- threshold utility 相同时先选择调用 run 更少的 threshold；仍相同时选择数值更大的 threshold。
- 所有顺序策略的 capture 分母使用同一条原生 SBS/default trajectory 上该 run 全部可达合格机会的 hindsight maximum；分子只使用策略 first-trigger 状态，不能让不同策略各自改变分母机会集。Random 的 30 个流先在同一科学 run 内平均，再进入 problem/function 聚合和统计推断；30 个流不是 30 个独立科学重复。

正式配置文件：

```text
configs/phase1_bbob_train.yaml
configs/phase1_bbob_validation.yaml
```

输出路径：

```text
results/phase1_refined_sampling/bbob_train_trajectories.parquet
results/phase1_refined_sampling/bbob_validation_trajectories.parquet
```

---

## 5.2 正式 BBOB trajectory 分片运行策略

裁决：

- 正式 BBOB train / validation 采集不得写入单个 Parquet 文件作为主运行方式。
- 正式采集采用按 `split / function-ID group / dimension` 分片输出。
- 单文件输出不是正式采集入口；正式运行只使用分片输出。

推荐分片路径：

```text
results/phase1_refined_sampling/bbob_train/bbob_f001/dimension_10/trajectories.parquet
results/phase1_refined_sampling/bbob_validation/bbob_f005/dimension_10/trajectories.parquet
```

分片粒度：

```text
train:
18 functions * 3 dimensions = 54 shards

validation:
6 functions * 3 dimensions = 18 shards
```

每个正式 shard 包含：

```text
3 instances * 30 optimizer seeds * 4 algorithms * (12--18 states)
= 4320--6480 trajectory rows
```

具体行数由冻结事件规则决定，下游不使用某个固定精确行数作为覆盖代理；必须按整数 `FE` 状态键对实际状态集做双向覆盖检查。

续跑口径：

- 若目标 shard 文件已存在，默认跳过。
- trajectory 与 `final_performance.parquet` 作为同一输出 pair 发布；覆盖采集期间不得并发启动 behavior、SBS 或其他下游读取。
- 显式传入 `--overwrite` 时允许重新生成目标 shard。
- 单个 shard 失败时，只重跑该 shard。
- 不实现哈希、checksum、manifest、receipt、append-only 或执行解锁机制。
- 文件存在性只作为人工续跑便利判断，不作为数据身份或完整性证明。

正式采集前必须先运行 shard plan：

```text
uv run phase1-plan-shards --config configs/phase1_bbob_train.yaml
uv run phase1-plan-shards --config configs/phase1_bbob_validation.yaml
```

---

## 5.3 Behavior 的跨 checkpoint 对应关系

裁决：

- Decision 主输入必须对 checkpoint 内 population 行排列不变。
- 不在 trajectory 中增加 individual ID 或 ancestry；DE、PSO、SHADE 的稳定行序也不作为统一输入协议的前提。
- population 跨窗口变化使用等权经验 Wasserstein-1、centroid shift、centroid/Wasserstein coherence 和当前协方差谱集中度。
- fitness 跨窗口变化使用排序后的经验分位数，计算改善分位数比例、平均分布改善率和一维 Wasserstein 变化率。
- 所有变化量按实际 `(FE_t-FE_anchor)/FE_total` 归一化；空间距离同时除以 `sqrt(dimension)`。
- CMA-ES 每代样本不存在稳定个体身份，因此不得使用 row-wise displacement、row-wise fitness improvement 或由其派生的方向统计。
- w02、w05、w10 必须从逐次完整原生 update 的运行历史中选择 anchor，不得再从稀疏正式 checkpoint 中回退选择。目标分别为当前 FE 之前 2%、5%、10% 总预算的位置；若目标不落在完整 update 边界，则选择不晚于目标位置的最近完整 update，使实际跨度不小于名义跨度且偏差小于一次 population update。
- trajectory 必须保存 `FE_total`、已完成的 `native_updates`、三个窗口的轻量集合/fitness 统计和最近10%预算内的逐 update 标量历史。behavior 对每个窗口分别保存 `effective_window_ratio_w02/w05/w10`、`effective_window_fe_w02/w05/w10` 与 `effective_native_updates_w02/w05/w10`；rate与slope使用实际跨度，窗口 metadata 不进入 Decision 输入。
- 上述9个窗口字段是计算来源 metadata，不属于 `BEHAVIOR_FEATURE_COLUMNS`，不得进入 Decision Model 或 Selector 输入。
- directional entropy 及其 direction bins 已由 permutation-invariant 集合分布变化指标替代；活动实现不得重新引入依赖 individual ID、ancestry 或跨 checkpoint 行对应关系的方向统计。
- 旧 behavior、utility labels、Decision dataset、模型和评价结果不得与新集合特征混用。

---

## 6. Dimension 使用

裁决：

- `dimension` 必须保存为 metadata。
- `dimension` 不进入 Decision Model 输入列。
- `dimension` 仅用于 split、分组报告和 OOD 分析。

同类规则：

- `function_id`、`algorithm_id`、算法内部参数、query feature 都不进入 Decision Model。
- `algorithm` 可保存为 metadata，但不作为模型输入。

---

## 7. Utility 成本账本

裁决：

- 主协议采用等总 FE 预算。
- 固定 query 消耗的 FE 通过减少后续优化预算体现。
- 若 query FE 已经从优化预算中扣除，Utility 中不能再扣同一笔 FE 成本。
- Utility 中额外扣除 wall-time、feature computation、memory 等非 FE 成本。

必须保存：

```text
FE_total
FE_prefix
FE_query
FE_no_query_optimization
FE_query_optimization
runtime_query
runtime_selection
runtime_handoff
runtime_no_query_handoff
runtime_no_query_optimization
runtime_query_optimization
runtime_query_total
runtime_no_query_total
runtime_net
FE_behavior_only_optimization
runtime_behavior_only_selection
runtime_behavior_only_handoff
runtime_behavior_only_optimization
runtime_behavior_only_total
gap_skip_terminal
gap_query_terminal
gap_behavior_only_terminal
gap_query_continuation_only
query_first_hit_offset
query_sample_best_contribution_log10_gap
runtime_skip_path_rep_1 ... runtime_skip_path_rep_3
runtime_query_path_rep_1 ... runtime_query_path_rep_3
runtime_behavior_only_path_rep_1 ... runtime_behavior_only_path_rep_3
```

对 `skip`、`query_joint`、`query_matched_state_only`、`sampling_only_continue_current` 和 `behavior_only_full_budget` 五条 operational path，先按 suite 配置冻结的下限/上限截断 terminal gap，再取十进制对数：

```text
ell_k = log10(min(max(g_k, 1e-12), 1e20))
T_k = median(runtime_k_censored_rep_1,
             runtime_k_censored_rep_2,
             runtime_k_censored_rep_3)
```

其中 `k in {skip, query_joint, query_matched_state_only, sampling_only_continue_current, behavior_only_full_budget}`。`runtime_query_total` 覆盖 query 采样、样本目标评价、特征计算、选择、handoff 与后续优化；`runtime_no_query_total` 覆盖 No-query handoff 与后续优化。纯 feature/selection/handoff 的计算开销只作诊断，不进入主 Utility。

主标签口径：

```text
U_query_joint = (ell_skip - ell_query)
              - lambda_time * (log10(T_query) - log10(T_skip))

U_behavior_only_full_budget = (ell_skip - ell_behavior_only)
                            - lambda_time * (log10(T_behavior_only) - log10(T_skip))

query_operational_increment = (ell_behavior_only - ell_query)
                            - lambda_time * (log10(T_query) - log10(T_behavior_only))

query_operational_increment = U_query_joint - U_behavior_only_full_budget

query_descriptor_use_increment = U_query_joint - U_query_matched_state_only

query_state_only_vs_sampling_increment =
    U_query_matched_state_only - U_sampling_only_continue_current

query_sampling_direct_increment = U_sampling_only_continue_current

U_query_joint = query_descriptor_use_increment
              + query_state_only_vs_sampling_increment
              + query_sampling_direct_increment
```

第一篇论文的主操作性情景固定为 `lambda_time=1`、`lambda_memory=0`，活动主字段为 `u_query_joint_lamT_1` 与 `query_operational_increment_lamT_1`。`lambda_time in {0,0.25,0.5,1,2}` 是预先定义的敏感性集合；不得依据 BBOB-train、BBOB-validation、CEC 或工程结果改选主 lambda，也不得只报告最有利的 lambda。`lambda_time=1` 表示 terminal gap 与 decision-state-to-terminal future-path runtime 的十进制数量级变化等权，而不是普适资源偏好。旧 max-scale performance、线性 relative-time、`performance_gain_norm` 和 `time_cost_norm` 全部退出活动口径。

Query sample 不进入 optimizer population，但其每次目标函数调用属于真实 FE。Query 路径的 terminal gap、`target_hit_observed` 与 ERT 取 query sample best 和 continuation best 的共同 best-so-far；first hit 可发生在 sample 内并保存 `query_first_hit_offset`。`path_completed` 与 `endpoint_success=target_hit_observed and path_completed` 分列，失败前已经发生的 observed hit 不得抹除。同时单独报告不含 sample best 的 continuation-only gap 以及 `query_sample_best_contribution_log10_gap`，防止把 sample 直接找到的更优点误归因于 selector 或 continuation。

科学端点与计时裁决：预指定 Stage-A 科学运行及其 Selection Reference outcome 唯一固定 terminal gap、`observed_first_hit_FE`、`target_hit_observed`、`target_hit_before_failure`、`path_completed`、`endpoint_success`、planned/effective FE 与 ERT contribution；主 Utility 的性能项只读这些 Stage-A 字段，并保存 `scientific_endpoint_source=stage_a_selection_reference_outcome`。Stage-B 只为同一 selected path 从共同 decision state 到 terminal 测量 future-path wall-clock，不使用 CPU time 或 validation 批量预测时间替代，也不得把 replay outcome 回写为科学端点。共同 prefix 已经执行，属于 sunk cost，不进入五条状态条件时间。计时进程预加载依赖和固定模型，线程数固定；同一 complete state、action、query 和显式随机状态真实执行预定 3 次，输入必须显式给出 `timing_replay_status in {completed,timed_out,failed}` 与正的 `timing_replay_timeout_seconds`。五条路径使用预定义循环顺序交错；每次原样保存 raw observed runtime，同时把 completed repetition 的 censored time 设为 raw，把 timed-out/failed repetition 的 censored time 设为 `max(raw observed runtime, role timeout)`。主 $T_k$ 使用三次 censored time 的中位数，raw observed median 只作诊断；不得按 status 筛选或补跑。路径身份、completed replays 内部 endpoint 与 Stage-A→completed-replay endpoint 分别使用三个 consistency 字段；Stage-B status instability 与 Stage-A/Stage-B completion instability 也分别保存。Stage-A path 未完成时，completed replays 只检查彼此一致并不得覆盖 Stage-A 失败端点。planned FE 必须保持路径身份一致；replay effective FE 不与 Stage A 或其他 repetitions 强制相同。旧 `failure_worst_case` 只可作为相同 censored 值的兼容别名。在线 policy 另从 `FE=0` 计到 terminal，包含 prefix、逐机会 Behavior/Decision、query 和 continuation；其科学 gap/observed-hit/completion/planned/effective FE 同样只取一条预指定 Stage-A 科学运行，`runtime_full_run_wall_clock_median` 与逐次状态只取三条 Stage-B timing replays并遵循相同 censoring 规则。该 full-run wall-clock 是政策端点，不进入状态条件 Utility。科学端点、future-path timing 与 full-policy timing 三类字段必须分开，不得互相替代。主结果始终把 Stage-A `log10_gap`、Stage-B future-path ratio 与 FE=0 policy wall-clock 作为分列端点同 Utility 一起报告。

若某实验允许额外 query FE，必须另设清晰公式并单独记录：

```text
U_query = performance_gain
        - lambda_FE * extra_FE_cost
        - lambda_time * time_cost
        - lambda_memory * memory_cost
```

---

## 8. Performance 方向

裁决：

- Utility 标签主性能量使用配置截断后的 terminal `log10_gap`，越小越好；Query terminal endpoint 包括 query sample best，continuation-only endpoint只作诊断。
- 对最小化问题：

```text
performance_gain_query_vs_skip = ell_skip - ell_query
performance_gain_behavior_vs_skip = ell_skip - ell_behavior_only
performance_gain_query_vs_behavior = ell_behavior_only - ell_query
```

- `target_hit_observed` rate、`endpoint_success` rate 等越大越好的指标只用于分列报告，不直接放入该差值，且两者不得合并成未定义的 `success rate`。

若使用综合指标，必须先统一方向并冻结公式。

---

## 9. Default Optimizer

裁决：

- 在线部署主设置使用 SBS 作为初始 default optimizer。
- SBS 只由 BBOB-train 的完整预算终值确定。数据源为与 decision trajectory 分离的 `final_performance.parquet`；每个 `problem_id × algorithm × seed` 在 `FE=FE_total` 恰好一行。
- SBS 的冻结计算顺序为：对每个完整预算 run 的 gap 按 suite 配置截断并取 `log10_gap`，再按 run → static problem（function × dimension × instance）→ fixed dimension stratum → function 等权聚合；选择 function 等权均值最低的算法，并列按 `de, pso, cmaes, shade`。不得改用平均 rank 丢弃效应量，也不得读取 `0.20–0.60` trajectory 的最后一行代替完整预算终值。
- nested outer-function 模型选择中的 default/prefix 使用 `SBS_outer`，且只由该 outer-fit functions 的完整预算终值计算；完整 BBOB-train SBS 只用于最终重拟合、BBOB-validation 与外部部署。不得让 outer holdout function 参与其自身评价所用的 SBS。
- 固定 CMA-ES 或 DE 可作为敏感性分析。
- 第一篇论文的主 probe/default 都固定为训练集 SBS；主 Decision 数据只保留 `prefix_algorithm == default_algorithm` 的行。
- 主 No-query 路径原生继续 SBS prefix 的完整状态，不重启、不改参数、不改变总预算口径。
- DE、PSO、CMA-ES、SHADE 的全 prefix 标签继续生成，但只进入 cross-probe robustness、leave-one-probe-out 和 algorithm-agnostic 泛化分析，不得混入主结果。
- 在非主 prefix 行中，Skip 使用训练集 SBS/default；若 default 与 prefix 不同，必须显式记录一次 population transfer，不能称为原生 continuation。

历史小规模链路验证已退出当前运行面，不再保留对应配置或结果入口。

---

## 10. SHADE / L-SHADE

裁决：

- 软件接口统一使用 `shade` 或 `shade_family`。
- 当前完整状态实现使用 SHADE；不得在本轮重生成中切换为 L-SHADE。

### 10.1 完整状态实现的算法定义

为保证初始化、完整运行、checkpoint 恢复和分支续跑使用同一逻辑，四种算法均由 `optimizers/state.py` 的状态推进器执行，不再混用外部完整运行器与本地 continuation adapter。

- DE：`rand/1/bin`，固定 `F=0.5`、`CR=0.9`。
- PSO：固定 `w=0.72`、`c1=c2=1.49`，速度上限为每维搜索区间的 `0.2`。
- CMA-ES：population size 等于正式配置，`mu=lambda/2`，包含 rank-one/rank-mu covariance update、step-size path、covariance path 和完整 eigensystem state；初始 mean 为边界中点，初始 sigma 为平均边界跨度的 `0.3`。
- SHADE：memory size 为 `5`，保留 `M_F`、`M_CR`、archive 与 memory index，使用 current-to-pbest/1 和成功历史更新。

上述定义属于新的数据生成机制。旧 pymoo/cma 与 population-only adapter 混合生成的结果不得与新结果合并。

---

## 11. Random Analysis

裁决：

- 统一术语为 `Random Analysis`。
- 旧“每个机会独立以 0.5 概率触发”的定义在 12--18 个机会下几乎必然调用 query，退出正式主 baseline。
- 正式主随机 baseline 为 `matched_rate_random`。先用完整 BBOB-train grouped-by-function 上游 OOF Proposed first-trigger predictions 冻结 run-level 调用率 `r_call` 与 first-trigger `FE_ratio` 的经验分布；seen evaluation 与 external confirmation suites 只加载这两个 train-OOF 校准量。
- 每个 run 开始时先按 `r_call` 决定是否计划调用；若计划调用，再预抽一个目标 trigger ratio，并在该 run 第一个不早于目标 ratio 的在线合格机会触发。每个 run 最多一次，不从完整机会列表事后均匀抽行，也不使用常数 per-opportunity hazard。使用 30 个由显式整数和 `SeedSequence` 产生的随机流。
- 30 个 Random 流先在同一个 benchmark function/problem/seed 科学 run 内平均，之后才进入 run → static problem → fixed dimension stratum → function 聚合；不得把 30 个流当作独立复制扩大样本量。
- Random baseline 不读取 validation/external Utility，不按结果匹配调用率，也不在一个 run 内重复 Bernoulli 直到成功。

代码配置名：

```text
matched_rate_random
```

---

## 12. 运行记录

裁决：

- 不自动采集 `git_commit`。
- 不实现哈希、checksum、digest、manifest、receipt、source closure、quarantine 或执行解锁机制。
- 运行记录保存普通实验元数据。

允许保存：

```text
config
seed
benchmark version
optimizer settings
model settings
timestamp
```

---

## 13. 正式实验与 preliminary 运行

裁决：

- 正式论文结果只来自冻结配置、完整 grouped-by-function split 和预定重复次数；该 function-level 设计不得写成经典 landscape-family 泛化。
- preliminary 运行只用于定位实现或资源问题，必须与正式输出目录隔离。
- preliminary 运行不得进入论文主表、模型选择或 threshold 拟合。
- 完整论文实验执行 30 independent runs、完整 baselines、外部评价和统计分析。

---

## 14. 三档 Landscape Query 与表示范围

裁决：

- 第一篇论文只以 `descriptor_cheap_invariant` 为主 query；它是固定 14 维 permutation-invariant 自定义低成本描述符，不得称为 Full ELA 或完整 pflacco。
- `pflacco_standard_invariant` 与 `pflacco_broad_invariant` 是预先定义的配置稳健性实验，不得根据 validation 结果替换主 query。
- 三档分别学习 Selector、Utility target 与 Decision Model，不把 `query_id` 作为模型输入，也不训练动态 query-type selector。
- 所有 query-sensitive 的 Selection Reference、Utility、Decision、baseline 与外部评价入口必须显式接收 `query_id`，并核对 `query_protocol`、`sample_design_id` 与实际列；默认输出只能写入对应的 query 目录。
- `Always Query` 表示在第一个在线合格机会调用；`pre_run_aas_fe0` 是 FE=0、query-only、sample-isolated 的 pre-run AAS：在 optimizer prefix 之前执行 query，只用 query features 与剩余预算选择初始算法，并用 `B-FE_query` 运行；query sample 不初始化或扩充 optimizer population。二者不共享运行结果、不得合并报告。该 baseline 以及所有 portfolio 比较只条件于仓内固定 DE/PSO/CMA-ES/SHADE 实现、冻结参数和 `population_size=40`，不能代表所有 Traditional AAS 或所有 portfolio instantiations。VBS 必须由静态 per-problem 完整候选结果计算：对每个 `problem = function × instance × dimension`，先对每个算法的完整预算 clipped `log10_gap` 跨 optimizer seeds 取算术均值，选择均值最低算法（并列按 `de,pso,cmaes,shade`），再用该算法在同一 problem 上的逐 seed paired outcomes 进入 run → static problem → fixed dimension stratum → function 汇总。不得逐 seed 选择最小算法，也不能以逐状态 `best observed action` 替代。
- FE=0 AAS 的关系字段固定为 `prefix_algorithm=selected_algorithm`（只作显式关系记账）、`selected_equals_prefix=true`、`handoff_required=false`、`handoff_type=fresh_optimizer_initialization`；`default_algorithm=no_query_algorithm=SBS_fold`。不得将 fresh initialization 误记为 population transfer；其组件时间单列 `runtime_fresh_initialization`，`runtime_handoff` 只表示已有 population 向另一算法的 transfer initialization。
- NeurELA、Deep-ELA、Progressive ELA 和其他学习式或动态 landscape representation 只用于说明表示异质性，本轮不实现。

---

## 15. Search Maturity

裁决：

- 正式模型同时比较 Direct Behavior -> Utility 与 Maturity-aware 表征。
- Search Maturity 可作为 Phase 3 的派生字段。
- Search Maturity 是 Behavior 的确定性派生变换，不增加新的原始信息；其消融只能支持“该预设非线性基函数对固定线性候选的预测作用”，不得称为独立 latent state 或因果中介。
- Phase 5 固定比较：

```text
T0 / B1 / B2
B2+Motion
B2+Maturity
B3 = B2+Motion+Maturity
```

- 不预设 Search Maturity 一定有效。
- `M_t = ES_t(1 - XS_t)` 是本文启发式定义，必须通过消融和 OOD 结果验证。
- Decision feature-group 正式消融固定为 `T0/B1/B2/B2+Motion/B2+Maturity/B3=1/19/25/28/28/31`。`B2+Motion` 加 3 个 set-motion 字段；`B2+Maturity` 加 3 个 maturity 派生字段；B3 同时加入两组。六组使用相同模型名、outer functions、train-only preprocessing、run-level first-trigger threshold 和评价指标，不重新选模或选择输入组。
- BBOB-validation 只能给 milestone-only 六函数有限集估计，CEC2017 只能给已见外部开发集估计；二者不建立确认性主张。若 B3 相对必需 `milestone_only_T0` 与估计性 `dimension_stratified_T0` 的方向不一致，不得声称 Behavior 的预测价值不能由阶段或已评估维度层解释。跨 benchmark 的确认性支持等待按第 17 节闭合后首次运行的 CEC2022 与工程集合；事件 schedule 上的优势只能解释为超出该行为条件调度的增量。

---

## 16. 开发主线

冻结开发主线：

```text
configs
-> benchmarks / optimizers
-> trajectory
-> behavior
-> landscape_queries
-> selection_reference
-> utility_labels
-> decision
-> evaluation
```

避免主目录名：

```text
oracle
tests
schema_registry
verification
```

## 16.1 算法切换后的初始化口径

裁决：

- 主实验的跨算法切换采用 Population Transfer；同算法路径采用原生完整状态 continuation。
- 主协议中 `prefix_algorithm == default_algorithm ==` 训练集 SBS，因此 No-query 必须原生继续该完整 checkpoint state。
- 全 prefix 稳健性数据中，Skip 使用 default algorithm；只有 `default_algorithm != prefix_algorithm` 时才执行一次 population transfer，并将 `skip_switches_from_prefix` 记为真。
- Run Query 若选择 prefix algorithm，必须继续同一完整 checkpoint state，只减少 query 消耗的后续优化预算。
- selector、Always Query、Random Analysis 和 Traditional AAS 只有在共享 checkpoint 后确实切换算法时，才使用该 checkpoint 的 `population`、`fitness` 和 `best-so-far position` 初始化新算法。
- Best-so-far Warm Start 不作为主实验默认口径，只能作为后续稳健性分析候选。
- query 采样点不并入后续优化 population；固定 query 只提供 selector 所需特征，并通过减少 `FE_query_optimization` 体现 FE 成本。

实现含义：

- 新算法继承的是算法无关搜索状态：位置、fitness 和当前 best。
- 新算法不继承前缀算法内部状态。
- DE、PSO、CMA-ES 和 SHADE 只在跨算法切换时按新算法初始化内部状态；该操作必须标记为 `population_transfer_initialization`，不得称为原生 continuation。
- 同算法路径保存并恢复完整状态：DE 保留 generation 与 RNG；PSO 额外保留 velocity、personal/global best；CMA-ES 额外保留 mean、covariance、sigma、evolution paths 和 strategy state；SHADE 额外保留 `M_F`、`M_CR`、archive 与 memory index。
- `p_query` 表示付出固定 query 成本后，selection reference 选择的算法从同一 checkpoint population 继续优化得到的 final performance；不是围绕 best-so-far 重启得到的 performance。

---

## 16.2 Decision Model 三候选与嵌套 OOF 口径

裁决：

- 活动候选固定为 LDA、Logistic Regression 和 Ridge，不继续搜索 Random Forest、XGBoost、LightGBM、MLP 或其超参数变体。
- 三个候选的 preprocessing 与估计器参数均预先固定；imputer 和 scaler 必须在每个 OOF fit fold 内独立拟合。
- 模型主选择指标固定为 BBOB-train outer-function OOF 的 run-level first-trigger mean `u_query_joint_lamT_1`。逐状态 mean utility 不得作为候选排序指标。
- train outer OOF 只用于三个预先固定候选的选择与开发期诊断；选出最大 OOF 候选后，不得把同一 OOF 数值称为 selected procedure 的无偏性能估计。
- 每个 outer fold 必须只用 outer-fit functions 依次完成：计算 `SBS_outer`；拟合/cross-fit Query Selector 与 `behavior_only_full_budget` Selector；生成 outer-fit Utility；在 outer-fit 内生成 Decision inner-function OOF score 并拟合 first-trigger threshold；拟合 Decision；最后在 outer holdout 上用 `SBS_outer`、outer-fit Selector、outer threshold 生成一次完整评价。outer holdout 不得影响上述任一组件。
- 每个 inner fold 也必须只用 inner-fit functions 重新计算 `SBS_inner`、两类 Selector、Utility labels 和 Decision preprocessing/model，再在 inner holdout 上产生 first-trigger score 与 Utility。不得在 outer-fit 上先制成一张固定标签表后只对 Decision 做 inner OOF。
- outer-fit 内用于训练 Decision 的标签由 outer-fit functions 内部 selector cross-fitting 生成；outer holdout Utility 只由 outer-fit 全量 Selector 产生。不得先用完整 BBOB-train 生成一张 Utility 表再把它当作整个两层管线的 outer-OOF 证据。
- 完整 BBOB-train 的 grouped-by-function OOF 分数必须来自 fold-specific SBS、Selectors 和 Utility，按 run-level first-trigger 冻结 `oof_utility_first_trigger` threshold及 matched-rate Random 校准，随后才在完整 BBOB-train 上重拟合最终 SBS、两类 Selector 与 Decision Model。BBOB-validation 已被历史开发读取，重建后的评价仍只能称为已见内部有限集估计；撤回旧产物不能恢复未见性。
- selected procedure 及 RQ2 milestone-only B3--T0 的 BBOB-validation 与 CEC2017 数值分别只是已见内部评价和已见外部开发估计。确认性证据只来自本次冻结后闭合并首次运行的 CEC2022 与工程集合；train outer OOF B3--T0 仅作开发期诊断。
- AUROC、Average Precision 和 Spearman 是辅助指标；连续 Utility RMSE 只对 Ridge 定义。LDA 与 Logistic Regression 的分类分数不得直接与连续 Utility 计算 RMSE。
- B3（兼容代码名 `all_candidates`）决定主 Controller 的模型名；`primary_with_maturity` 只解析为 B2+Maturity。T0/B1/B2/B2+Motion/B2+Maturity 与 B3 比较必须读取同名候选的预测，不能各自改选更有利的模型。
- 分阶段阈值只能使用 BBOB-train OOF 信息预先拟合并作为稳健性分析；BBOB-validation 不得用于阈值网格选择。
- 旧 LDA/Ridge/复杂模型数值继续保留在撤回结果说明中，但不得据此预设重生成后的赢家或把 Utility 解释为已证实的分类边界构念。
- 主训练经验风险改为 `cluster_balanced_fit`：每个 fit fold 内使 functions 等权；每个 function 内固定 dimension strata 等权；每个 function × dimension 内 static problems 等权；每个 static problem 内 optimizer runs 等权；最后把每个 run 的权重等分到其合格 states，并在 fold 内把 row weights 归一化到均值 1。权重不得使用 holdout rows 或 outcome 方向计算。
- `sample_weight=1` 的 state-row 等权拟合降为 `row_weighted_fit` sensitivity。nested first-trigger evaluation 不会自动纠正拟合权重；imputation/scaling 与三个候选 estimator 必须使用同一 fit-fold population，并对 cluster weights 提供兼容实现。该 wiring 尚未完成，是正式模型冻结前 blocker。

---

## 17. 当前仍需单独冻结的实验细节

证据角色冻结为：BBOB-validation 是已见内部评价集，CEC2017 是已见外部开发集；二者不再提供确认性证据。CEC2022 与工程集合是预定的 untouched external confirmation，但目前尚未闭合：

- CEC2022：benchmark factory 已有接口，但缺少冻结配置；必须先冻结函数范围、维度、预算、seeds/repetitions、reference、success target、timeout/failure rule 与 contrasts。
- 工程集合：当前缺少冻结问题清单、配置、factory、预算、重复、reference/gap 定义、constraint-handling rule 与 endpoint；这些内容在任何 outcome 生成或查看前必须一次冻结。

CEC2017 的维度、重复与预算当前写为 10D / 30D / 50D、30 seeds 和 `FE_total=1000D`，但函数集仍是可信有限集估计的 blocker：`configs/phase1_cec2017_test.yaml` 当前列出 F1--F29，即包含 F2、排除 F30；项目内尚无依据说明这是否与所用 CEC2017 实现及官方口径一致。必须核对实现/技术报告并冻结 F2/F30 处理；它即使闭合也仍是已见开发集，不恢复确认性资格。

## 17.1 正式失败规则

对 BBOB train/validation 与 CEC2017，当前冻结端点为：

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

CEC2022 与工程问题必须在首次确认性运行前分别冻结同类字段；工程集合还必须冻结正式 constraint-handling rule。缺少配置或 factory 时不得启动，任何先看 outcome 再补规则的集合自动降级为开发性评价。

- 所有计划的 problem、dimension、seed 和 run 必须先进入 coverage denominator；不得在看到失败与效应方向后删除 run、query 或 feature group。
- Decision score 在某机会缺失或非有限时，该机会固定按 No-query 处理并记录 `controller_status`；同一 run 可在后续仍可达机会再次检查。
- query 一旦触发即消耗其采样 FE 与实测时间。若 query feature 或 Selector 输出失败，固定 fallback 为从该 checkpoint 原生 `continue_current`，使用 query-adjusted remaining budget；该 run 不得回退为未付 query 成本的 Never Query。
- 单个候选 action continuation 失败或超过 `action_timeout_seconds` 时，必须使用 suite 配置中在运行前定义的有限 `failure_loss_cap`；若 suite 未定义该值，不得启动其正式 action-loss 生成。失败动作仍保留在四动作矩阵并记录状态；失败前已经发生的 `target_hit_observed` 保留，`path_completed=false` 且 `endpoint_success=false`，只有未命中项的 ERT contribution 计完整 planned budget。
- first hit 必须在每次 objective evaluation 后记录；Query sample 内的首次达到 `success_gap_target` 也属于 Query 路径 first hit，不能等到 continuation checkpoint 才记录。
- terminal gap、observed first hit、target hit、path completion、endpoint success、planned/effective FE、ERT contribution 和科学 failure 只取预指定 Stage-A 行。Stage-B timing replay 的逐次 endpoint 只用于其对应适用条件下的一致性检查，任何 replay 均不得替换 Stage-A 行；timing timeout/failure/两类 instability 另作计时失败率与 sensitivity。
- 缺失 shared-state outcome、缺失 full-budget 对照或状态键不完整属于数据生成失败，不做统计插补；修复后只重生成对应 shard。正式分析开始前要求计划 shard 全部有终态，允许的科学运行失败按上一条保留。
- 每个 suite 与 endpoint 同时报告 attempted denominator/coverage、query/selector/controller/action failure rate、complete-pair estimate 与预设双向极端 failure sensitivity。未闭合 pair 的两种赋值在运行前固定：gap 分别取 suite floor/cap；`target_hit_observed` 分别取 1/0；ERT 的 adverse 未命中项计完整 planned budget，favorable 命中项的 contribution 是在该行已知 prefix history、已计 query FE 与 endpoint 时间原点下最早可行的 objective-evaluation index（若已在 prefix 命中则保留实际 first hit；若没有可用 prefix，则 FE=0 全路径的下界为 1）；runtime 的 adverse 值取 timeout，favorable 值取该 suite complete pairs 的最小正 runtime。若最早可行 hit FE 或最小正 runtime 无法由正式行重建，对应 sensitivity 为 undefined且该 endpoint 结论未建立。Utility 必须从同一组极端 gap/runtime endpoint 重新计算，不能直接给 Utility 任意常数。科学路径本身已以 cap/timeout 得到的失败 outcome 保留为观测，不当作 missing pair。
- 若双向 failure sensitivity 改变效应方向、改变区间相对项目内 operational tolerance 的位置，或 attempted coverage 低于 95%，该 suite × endpoint 的结论记为“未建立”。attempted denominator 与 complete-pair denominator 必须并列报告，不能只给成功子集。

## 17.2 统计目标与推断口径

- RQ1 的目标分布限于 `descriptor_cheap_invariant`、SBS prefix、冻结 `phase1_dynamic_budget_event_v1` eligible states。统一聚合层级为 state → run → static problem → fixed dimension stratum → function：先在 run 内对 eligible states 等权，再在 `function × dimension × instance` static problem 内对 paired optimizer runs 等权，在每个固定 dimension stratum 内对 static problems 等权，最后在 function 内对 10D/20D/40D strata 等权。policy endpoint 从 run 层开始同序聚合。function 是顶层统计单位。
- BBOB-validation 的 estimand 是 6 个固定 functions（F5、F9、F13、F14、F19、F24）的等权有限集均值，条件于这 6 个函数及其冻结 dimensions/instances。历史模型比较、调参、消融与采样设计已经读取该集合，因此它是已见内部评价集；撤回旧数值不能恢复确认性。CEC2017 因已有 online/targeted 诊断而是已见外部开发集。二者均不支持函数超总体推断。
- 外部 suite 不机械复用 BBOB 层级。CEC2017/CEC2022 的 suite-specific estimand 以预列 function 为顶层有限集单位，在 function 内对固定 dimensions/instances（若有）与 paired optimizer runs 按运行前冻结的顺序等权。工程集合以每个预先命名的 engineering problem 为顶层有限集单位；其 dimension、load case、constraint variant 或其他变体只能作为该 problem 内预先冻结的 fixed strata。各 suite 分开报告，不重抽顶层有限集单位作主区间，也不推断到未列函数或工程问题总体。该层级、问题清单和权重未写入配置与 consumer 前，对应前瞻评价不得运行。
- BBOB-validation 的 95% CI 使用 10,000 次条件配对层级 bootstrap：始终保留全部 6 个固定 functions、全部 fixed dimensions 与 instances 1/2/3 对应的全部 static problems；只在每个固定 static problem 内配对重抽 optimizer seeds。RQ1 每个抽中 seed/run 的完整有序 state 序列作为不可拆分簇保留。不得在主 CI 中重抽 function 或 static problem；该区间不推断到 function 或 transformed-instance 超总体。
- 另可有放回重抽 6 个 validation functions，保留抽中 function 的全部 fixed dimensions/static problems，并在 problem 内配对重抽 optimizer seeds；该结果只能标为“函数组成敏感性”，用于显示有限集均值对所含函数的依赖，不能作为主 CI、确认性证据或 function/transformed-instance 超总体区间。
- CEC2022 与工程集合只有在第 17 节全部配置、factory/runner、端点与 contrasts 于首次 outcome 前冻结后，才构成确认性外部评价；该“确认性”只表示 prospective protocol，不把预列有限 suite 当作函数超总体概率样本。
- ERT 使用专用 paired hierarchical ratio bootstrap，不能把 run-level `ERT_FE` 当作普通算术均值型 effect。每个 replicate 固定各 `function × dimension` 内的 static problems，只在每个 problem 内联合配对重抽 optimizer runs，分别重算 treatment/reference 的 `N_FE` 与 `N_hit`，计算 `ERT=N_FE/N_hit` 及 `log10(ERT_treatment/ERT_reference)`，再对 fixed dimensions 等权形成 function effect，最后对 fixed functions 等权。单方零命中保留为有符号无穷，双方零命中记为显式 undefined mass；任何此类 stratum 或 replicate 均不得静默删除。区间使用扩展实数分位数，将 undefined mass 保守分配到两侧尾部，并分开报告 finite/unbounded/undefined observed contrast 状态、undefined mass 与各类零命中计数。`interval_established` 只表示 observed contrast 与扩展实数边界是否有定义，不再因为任一 bootstrap replicate 出现零命中而改变。绝对 ERT 逐 function × dimension 报告；不同维度 raw FE 不先池化。
- `paired_ert_strata` 与 `paired_hierarchical_ert_log10_ratio_interval` 当前只实现专用计算核，尚未接到 suite-level attempted denominator/coverage、双向 failure sensitivity 与正式报告 consumer。该 wiring 完成前，complete-pair ERT ratio 不能单独支持 suite 结论。
- Utility `±0.01`、mean `log10_gap ±0.05`、geometric-mean runtime ratio `[0.95,1.05]` 及 call/target-hit-rate 差 `±0.05` 统一称为“项目内预设 operational tolerance”。项目内目前没有独立领域依据把它们称为 confirmatory equivalence bounds；主条件 CI 固定为 95%，只能逐项描述相对 tolerance 的位置，不能形成确认性等价声明。若同一预设 family 同时描述多个 contrasts，Bonferroni 区间使用每项双侧 level `1-0.05/m`，提供 family-wise 95% coverage；它仍不是等价检验。Utility 中 gap/runtime 抵消也不能替代各 endpoint 的判断。
- 当前项目内 tolerance 没有独立领域依据，第一篇论文不作确认性等价声明。BBOB-validation 与 CEC2017 只用条件区间描述相对 tolerance 的位置；差异不显著不等于等价。未来 untouched external suite 若预列 simultaneous intervals，contrast family 与 interval level 必须在首次 outcome 前冻结，且仍只能称为相对项目内 tolerance 的 prospective comparison。
- 当前 3 个 BBOB instances × 30 optimizer seeds 与 CEC2017 的 30 seeds 没有仓内 precision/power 依据，只能视为固定开发期采样设计；结果须报告实际区间宽度，不得事后声称样本量充分。CEC2022/工程问题在首次 outcome 前必须冻结 endpoint-specific precision target、利用开发集合方差信息但不读取目标 suite outcome 的重复数确定方法，以及最终 repeats；未闭合即不得启动对应前瞻评价。
- BBOB-validation 的 paired sign-flip 以 6 个固定 function effects 为单位，依赖零假设下 signs 可交换（关于零对称）的额外假设。穷举 `2^6=64` 后双侧 exact raw p 最小为 `2/64=0.03125`；它只能作为假设敏感辅助，不产生函数超总体推断。
- RQ2 唯一主要科学 contrast 仍是冻结模型家族的 `milestone-only B3 - milestone-only T0`。BBOB-validation 与 CEC2017 分别给已见内部/外部开发估计；确认性证据等待 untouched CEC2022 与工程集合。LDA、Logistic Regression、Ridge 的 train outer-OOF 两两差异只作候选选择诊断。
- RQ3 与 RQ5 在六函数 BBOB-validation 上各有 6 个预设辅助 contrasts；其最小 Holm-adjusted p 均为 `0.1875`，在 `alpha=0.05` 下不可能拒绝。RQ4 是按 suite 与 endpoint 分开的迁移评价，四个 suites 不是同一零假设下的四个可交换 contrasts，不建立跨 suite 的四 contrast Holm family；未来某 suite 内若有多个 contrasts，须在该 suite 首次 outcome 前单独冻结 family。RQ3--RQ5 均以逐 function/problem effects、固定有限集效应、条件 95% CI、coverage 与失败敏感性为主；raw/adjusted p 若报告，只能放在明确标为 assumption-sensitive 的辅助表中。未拒绝不表示无效或等价，suite、endpoint 与 RQ 不池化。

## 17.3 计算量与正式运行前检查

- 基础 BBOB train + validation 为 25,920 runs、约 604.8 million FE；12 states/run 给出 311,040 states 的最低行数。
- 在 `FE_prefix=0.60B`、12 states/run、无 event/failure 的简化情景下，query sample artifact 按 `problem × sample_design` 只生成一次；action-loss continuation 只扣除 sample FE，不再次执行 query。跨 Query-adjusted/Behavior-only matrices 共享一次 full-budget Skip/Behavior `continue_current`、但不复用基础 trajectory 时，阶段 A 为 main cheap 22.952412B FE、三档 query 31.662036B FE。若逐行证明基础 trajectory 终值同义，还可各减 2.90304B，得到 20.049372B/28.758996B。当前 producer 每个 action-budget CLI 执行 `skip + 4 actions`，main cheap 两 matrices 额外增加 5.80608B，Stage A 为 28.758492B；三档实际调用量须由调用图枚举。
- 主 replay 每个 fold role 只使用 `SBS_fold` prefix。5 outer × 4 inner 加 full-train→validation 的当前覆盖使每个 train state 出现 22 个 roles、validation state 出现 1 个 role；预算加权 role-state basis 为 30.3912B。Stage-B 对 selected decision-state-to-terminal paths 各真实重跑预定 3 次，只形成计时中位数与重复状态诊断；加入主 pre-run AAS 的对应计时 repeats 后，仅跨 matrices 共享时为 main cheap 132.814332B FE、三档 query 214.462836B FE；进一步复用基础 trajectory 时为 129.911292B/211.559796B；保持当前 main producer 则为 138.620412B FE。
- 完整在线 evaluator 每个 base tuple 当前包含 7 条固定政策和 30 条 matched-rate Random，每条 1 次 Stage-A 加 3 次 Stage-B，共 148 个 full runs。现 CEC2017 配置对应 11.5884B planned FE；BBOB-validation 全 instances 对应 5.5944B FE，但当前 evaluator 只接受 CEC、固定 `instance=1`，无法执行后者。
- 上述数字仍未包含 event-only 与更早 states、失败、额外 LHS、CEC query artifacts/VBS、CEC2022、工程问题和实际 replay plan 可能揭示的额外不可复用路径，因此只能称为简化情景，不是严格下界或排期承诺。Replay planner 已有枚举能力，但 grouped-by-function Selector artifact 路由、offline runner、物化实测 plan、Stage-A 共享/复用裁决、BBOB instance-aware online endpoint 与资源排期尚未闭合。
- 资源、机器吞吐和排期尚未在项目内建立，是正式运行 blocker。完整矩阵按 main cheap BBOB → standard/broad 配置稳健性 → 已见 CEC2017 开发估计 → 配置/factory/端点闭合后的 untouched CEC2022 与工程确认性评价分阶段执行；任何阶段不能用缩减、代理或单次计时结果冒充正式证据。
- 正式运行前必须逐项确认：trajectory/reservoir state 与整数 FE 对齐；固定 query seed/operator；活动 query ID；两套 action budget；outer/inner fold-specific SBS/Selector/Utility/Decision/threshold；cluster-balanced main fit 与 row-weighted sensitivity；`dimension_stratified_T0`；Stage-A 科学端点与 Stage-B timing-only replay 字段分离；offline selected decision-state-to-terminal producer 与 replay plan；state-conditional future-path timing 和 FE=0 policy wall-clock 分离；每次 replay 的 status/effective FE/timeout/completion、完成端点一致性和 instability 标记；run-level first-trigger baseline；三次预定计时且无选择性重跑；SeedSequence 整数流；失败 cap/timeout；ERT 专用 ratio bootstrap 与零命中状态；72 个 trajectory/final-performance pair；CEC2017 F2/F30 口径；CEC2022/工程配置、factory、constraint rule 与 contrasts；按最终 replay plan 可承受的资源与排期；结果路径不含撤回产物。任一未确认即暂停正式运行。

已经冻结并进入正式 phase1 的细节：

- 主 query 为 `descriptor_cheap_invariant`（14 维、`lhs_50d`、5% FE）；统一 median/IQR preprocessing 后恒为 0/1 的 `descriptor_y_median` 与 `descriptor_y_iqr` 已从活动 whitelist 删除，不改变 query ID、采样或 action losses。`pflacco_standard_invariant`（37 维、`lhs_50d`、5% FE）与 `pflacco_broad_invariant`（52 维、`lhs_100d`、10% FE）只作预定义配置稳健性。
- cheap 与 standard 共享完全相同的 (X,y)；broad 使用独立样本与 action-loss 预算。
- 主环境不依赖 pflacco；标准特征只由 `tools/pflacco_query/` 的 Python 3.11、pflacco 1.2.2 环境从 Parquet 样本提取。
- BBOB train / validation grouped-by-function split，见第 5.1 节；历史字段名 `family` 不改变其 function-ID 语义。
- BBOB train / validation dimensions：10D / 20D / 40D。
- Phase1 主采样协议：`phase1_dynamic_budget_event_v1`，12 个预算里程碑加冻结事件状态，每个 run 产生 12–18 行。
- `dimension`、`function_id`、`algorithm_id`、算法内部参数、query features 不进入 Decision Model 输入；这些字段只作为 metadata 和分层诊断使用。
- 必须保存 `selected_equals_default`、`selected_equals_prefix`、`handoff_required` 和 `skip_switches_from_prefix`；四者分别回答 selector 是否选择 SBS、query 路径是否继续当前算法、Query 路径是否需要 Population Transfer、No-query 是否离开当前算法。
- `handoff_required = not selected_equals_prefix = (handoff_type == population_transfer_initialization)`；活动数据和报告不得再生成 selected-vs-default 字符串别名。
- 主协议因 `prefix_algorithm == default_algorithm`，此时 `selected_equals_default == selected_equals_prefix`，但仍分别保存两个字段。
- Selection Reference / query-conditioned selection pipeline 是固定下游组件，不作为本文方法贡献点。
- 算法切换后的主初始化口径为 Population Transfer；query 采样点不复用到后续优化。
- `lambda_time=1`、`lambda_memory=0` 的主操作性情景与 `lambda_time={0,0.25,0.5,1,2}` 敏感性集合已经冻结；不得结果后改选。
- Decision Model 候选、outer-fold-specific 全链选择、`oof_utility_first_trigger` threshold 冻结和外部评价自动加载所选模型的协议已经冻结。
