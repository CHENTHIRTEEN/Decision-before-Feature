# Decision-before-Feature 完整论文结构与 Contribution 设计

> 唯一活动版本。本文只定义研究叙事、贡献边界与 RQ—证据关系，不声称正式实验已经完成。旧的单一 Query Utility、Search Maturity 潜在状态、Always Query 与 Traditional AAS 合并、逐状态 policy 汇总和四组消融都已退出。

## 1. 论文定位与可检验主张

研究领域包括 Automated Algorithm Selection、Exploratory Landscape Analysis、metaheuristic behavior analysis 与 resource-aware optimization。第一篇论文只研究一个受限而可检验的问题：

> 在连续黑盒优化的预定义在线状态分布上，执行固定 `descriptor_cheap_invariant` landscape query 并调用其下游 Selector，相比原生继续 fold-specific SBS，是否产生更高的成本调整终点性能；仅使用 query 前算法无关行为，能否改善这一调用决策？

`descriptor_cheap_invariant` 是 14 维自定义、permutation-invariant、低成本描述符配置；统一 median/IQR preprocessing 后恒为 0/1 的两个响应量统计已删除。它不代表 Full ELA 或完整 pflacco。`pflacco_standard_invariant` 与 `pflacco_broad_invariant` 只检验三个预定义配置之间的 representation dependence，不能在看到结果后替换主 query，也不能支持对任意 landscape representation 的外推。

论文目标不是设计新优化器，也不是证明“所有 Landscape Analysis 都应成为优化对象”。可支持的结论必须限定到：

- 固定 portfolio：DE、PSO、CMA-ES、SHADE；
- 固定 query、采样和成本口径；
- BBOB-train é¢åæå®组件及指定的 held-out/external suites；
- 一次 first-trigger 决策；
- 观察性、预测性和操作性比较，不作因果主张。

## 2. 研究缺口与核心故事

典型 landscape-based AAS 在获取 descriptors 后选择算法：

```text
problem -> landscape query -> selector -> optimizer
```

本项目增加一个 query 前决策，但不把下游 Selector 写成新贡献：

```text
native SBS/default prefix
-> permutation-invariant pre-query behavior
-> first-trigger Decision-before-Feature
   -> skip: native SBS continuation
   -> query: fixed sample + descriptors -> statewise Selector -> continuation/handoff
```

缺口不是“ELA 一定昂贵”，而是固定 query 的净结果取决于采样 FE、sample best、descriptor 计算、Selector 误差、动作切换、continuation budget 和真实 wall-clock；这些组成不能从 feature accuracy 或算法选择 accuracy 单独推出。

## 3. 三个必须分开的 estimands

五条 operational paths 在同一完整 optimizer state 上执行：`skip`、Query/full Selector、matched-acquisition state-only、sampling-only continue-current、Behavior-only full-budget Selector。预指定 Stage-A 单次科学运行固定 terminal gap、observed first hit、target-hit、path-completion、planned/effective FE 与失败状态。设五条 Stage-A 路径的 terminal raw gap 为 $g_s,g_q,g_m,g_c,g_b$，按 suite 配置截断后：

$$
\ell_k=\log_{10}\!\left(\min\{\max(g_k,10^{-12}),10^{20}\}\right),
\qquad k\in\{s,q,b\}.
$$

Stage-B 将每条 selected 路径从同一 decision state/RNG 到 terminal 真实运行预定三次，但只决定 timing。每次保存 raw observed wall-clock；completed repetition 的 censored time 等于 raw，timed-out/failed repetition 的 censored time 为 `max(raw, role timeout)`，$T_k$ 固定为三次 censored time 的中位数，raw median 只作诊断。逐次保存 status、`observed_first_hit_FE`、`target_hit_observed`、`target_hit_before_failure`、`path_completed`、`endpoint_success`、effective FE 与失败字段。路径身份、completed replays 内部 endpoint、Stage-A 到 completed replay 的 endpoint 一致性分别保存；Stage-B 内部 status instability 与 Stage-A/Stage-B completion instability 也分别保存。Stage-B 不得覆盖 Stage-A 科学端点或选择性补跑。共享 prefix 视为 sunk cost，FE=0 policy wall-clock 另报并遵循相同科学/计时分离。主设置 $\lambda_T=1,\lambda_M=0$：

$$
G_FE=(\ell_s-\ell_q)-\lambda_T(\log_{10}T_q-\log_{10}T_s),
$$

$$
U_b=(\ell_s-\ell_b)-\lambda_T(\log_{10}T_b-\log_{10}T_s),
$$

$$
query_operational_increment=(\ell_b-\ell_q)-\lambda_T(\log_{10}T_q-\log_{10}T_b)
=G_FE-U_b.
$$

对应解释严格区分：

- `g_fe_selected_path`：固定 query、full Selector、动作选择与 continuation 的联合路径相对 Skip 的等总 FE 净差，是主 Decision target；`g_fe` 仅是最佳已观测动作诊断；
- `u_behavior_only_full_budget_lamT_1`：不执行 query 的行为选择路径相对 Skip 的净差；
- `query_operational_increment`：Query 路径相对 Behavior-only full-budget 路径的操作性增量，包含 query FE、时间、sample best 与预算差，不是纯信息效应；
- `query_feature_predictive_increment_log10_gap`：`query_adjusted_state_only_selector` 与 full Query Selector 在同一 query-budget 四动作 outcomes 上的 OOF selected continuation-only `log10_gap` 差；不新增 action losses、不计 sample best、不扣 acquisition cost，只是 query descriptors 的边际预测贡献诊断。

Query sample 不进入 optimizer population，但属于真实 objective evaluations。Query terminal gap 使用 sample best 与 continuation best 的共同 best-so-far。`observed_first_hit_FE` 保留失败前已经发生的 target hit，`target_hit_observed` 驱动标准 ERT，`endpoint_success=target_hit_observed and path_completed` 另表示完整路径且命中；不得把这两个成功概念混用。另报 `query_first_hit_offset`、continuation-only gap 与 `query_sample_best_contribution_log10_gap`。

## 4. 方法贡献边界

### Contribution 1：Analysis-selection formulation

把“是否执行一个预先定义的 landscape query”形式化为状态条件、资源感知的二元预测问题，并给出与 terminal `log10_gap` 和真实 decision-state future-path runtime 一致的可计算 estimands，同时分开报告 FE=0 policy wall-clock。

不声称：该问题已覆盖所有 ELA、所有 feature acquisition 或所有资源偏好。

### Contribution 2：Pre-query behavior representation

从完整原生 optimizer updates 提取 query 前、算法无关且对 population 行排列不变的行为表征。w02/w05/w10 anchor 来自逐 update 历史；实际 FE、ratio 和 native-update 数只作 metadata，不进入 Decision 输入。

不使用 query features、function/dimension/algorithm identity、已知最优值 gap 或 optimizer 内部参数。输入组固定为：

```text
T0 = FE ratio                                      (1)
B1 = base behavior                                (19)
B2 = B1 + longitudinal behavior                  (25)
B2+Motion                                        (28)
B2+Maturity                                      (28)
B3 = B2 + Motion + Maturity                      (31)
```

Search Maturity 只是由既有 Behavior 变量确定性计算的三维非线性基函数组；它不是独立观测、latent state、收敛判据、因果中介或由 Utility 反向定义的标签。不预设倒 U 关系。

### Contribution 3：End-to-end leakage-controlled evaluation

SBS、Query/Behavior-only/FE=0 Selectors、Utility、Decision preprocessing/model、first-trigger threshold 与 Random calibration 全部进入 `cv_group_id = function_id` nested split。每个 outer fold 重算 `SBS_outer` 和全部上游组件；每个 inner fold又只用 inner-fit functions 重算 `SBS_inner`、Selectors、Utility 与 Decision。这里的 group 是 function ID（即 `cv_group_id`），不是经典 landscape-family taxonomy；禁止先在完整 train 制成 labels 再执行 Decision-only OOF。

Decision Model 活动候选固定为 LDA、Logistic Regression 与 Ridge。模型主选择使用 BBOB-train outer-holdout run-level first-trigger mean `G_FE`；AUROC、Average Precision、Spearman 是辅助指标，连续 Utility RMSE 只对 Ridge 定义。BBOB-validation 已被历史模型比较、调参与消融查看，只能作已见内部评价；CEC2017 也只能作已见外部开发评价。

### Contribution 4：Resource-aware policy benchmarking

在等总 FE、真实 wall-clock 和显式失败规则下比较 first-trigger policies，并把联合 Utility、terminal `log10_gap`、target-hit/endpoint-success/ERT、runtime、query FE、call/trigger/handoff、coverage 与 failure 同时报告。Utility 内部的 gap/runtime 抵消不能替代各 endpoint 对区间相对 operational tolerance 位置的独立判断。

## 5. Research Questions 与最小证据

### RQ1：主 query 的状态条件联合效用是什么？

目标分布限于 `descriptor_cheap_invariant`、SBS prefix、`phase1_dynamic_budget_event_v1` 合格状态。报告：

- $G_FE$、$U_b$、$query_operational_increment$ 及组成；
- terminal 与 continuation-only `log10_gap`；
- query sample best contribution、first hit、runtime；
- state → run → static problem → fixed dimension stratum → function 的分布、效应与 95% CI。

RQ1 估计状态分布，不把每个 state 当独立统计单位，也不预设“多数状态”方向。

### RQ2：Behavior 是否提供超出预算阶段的信息？

先只用 B3 的 nested BBOB-train first-trigger Utility 选择 LDA、Logistic Regression 或 Ridge。随后é¢åæå®同一模型家族，在相同 12 个 mandatory milestone rows 上比较 B3 与 `milestone_only_T0`；完整 dynamic schedule 上的 T0 只能称为 `schedule_conditioned_T0` sensitivity。由于维度和绝对预算也是调用前已知上下文，另做按 dimension 预先分层、每层仍只输入 FE ratio 的 `dimension_stratified_T0`；主 Decision X 不加入 dimension。若主 cheap 的 planned query-cost ratio 恒定，则明确其不可识别，不把常数列包装成对照。

只有 B3 相对 milestone-only T0 与 dimension-stratified T0 的 Utility、terminal endpoints 和未查看外部评价方向稳定时，才能写“Behavior 提供超出阶段及已知维度上下文的预测价值”。

### RQ3：Decision-before-Feature 的端到端权衡如何？

预设九个角色、八个不重复 outcome：

1. Never Query / SBS（同一 native outcome，只统计一次）；
2. Always Query；
3. matched-rate Random Analysis；
4. Traditional `pre_run_aas_fe0`；
5. VBS（静态 problem-level hindsight reference：每个 function × instance × dimension problem 内先按 seeds 聚合四算法完整预算 clipped `log10_gap`，选均值最低算法，再汇总该算法的 paired seed outcomes；不逐 seed 选最小）；
6. milestone-only Time Controller；
7. self-thresholded Behavior-only Selection；
8. Decision-before-Feature。

所有顺序策略每 run 最多一次 first trigger。Random é¢åæå® train-OOF run call rate 与 trigger-FE 经验分布；30 个流先在同一科学 run 内平均。Policy capture 对所有策略共享 native SBS/default trajectory 的 run-level hindsight maximum，分子只取 first-trigger state。

### RQ4：é¢åæå®过程能否迁移？

分别报告已见 BBOB-validation、已见 CEC2017、前瞻 CEC2022 与前瞻工程问题，不能池化成单一“OOD”结论。前两者只作开发期有限集评价；后两者只有在本次协议之后先é¢åæå®完整配置和分析规则、再首次生成 outcome 时，才可承担独立确认性评价。任何 external suite 不参与 preprocessing、选模、feature-group 选择、threshold 或 Random calibration。

CEC2017 的 F2/F30 口径、CEC2022/工程问题的函数、维度、预算、failure endpoint 与 constraint rule 必须在运行前é¢åæå®；未é¢åæå®部分不是可执行实验。

### RQ5：哪些预设 Behavior 基函数与预测表现相关？

使用 RQ2 在 B3 上选定的同一模型家族，比较 T0/B1/B2/B2+Motion/B2+Maturity/B3 六组。每组在相同 folds 和适用机会集合上独立完成 train-only threshold 过程，但不重新选模型家族或输入组。

系数、判别方向、Spearman 和 maturity 关系只作预测关联解释；不得写成因果作用、独立中介或普遍非单调规律。

## 6. 论文结构

### Section 1 Introduction

依次建立 Algorithm Selection 背景、固定 query acquisition 的协议依赖成本、pre-query trajectory information、Analysis Selection 问题和证据边界。贡献段只写方法定义与评价设计，正式结果完成前不写性能方向。

### Section 2 Related Work

覆盖 Algorithm Selection、ELA-based selection、sampling sensitivity、trajectory/probing features、behavior analysis 与 adaptive control。明确现有 trajectory-based selection 不等于预测独立 landscape query 的净效用。新颖性结论只限项目已检索文献，投稿前仍需更新检索。

### Section 3 Problem Formulation

定义完整 optimizer state、五条 operational paths、两个动作预算、terminal endpoint、联合/Behavior-only Utility、四类操作性增量、query sample endpoint、first-trigger policy 与三个显式动作关系字段：

```text
selected_equals_default
selected_equals_prefix
handoff_required = not selected_equals_prefix
```

并核对 `handoff_required == (handoff_type == population_transfer_initialization)`。

### Section 4 Method

描述 native trajectory、permutation-invariant Behavior、三档 query、两套四动作 matrices、三类 Selector、完整 nested chain、六组 Behavior 表征与一次 first-trigger 部署。Selection Reference 是固定下游组件，不作为贡献。

### Section 5 Experimental Setup

é¢åæå® `cv_group_id = function_id` split、SBS/VBS 聚合、baseline、Stage-A 科学失败端点、Stage-B 三次 decision-state future-path timing-only replay、raw/censored runtime、observed hit/path completion/endpoint success、三类一致性、两类 instability、FE=0 policy wall-clock、禁止选择性重跑、固定六函数条件 bootstrap、函数组成敏感性和项目内 operational tolerance。BBOB-validation 的 estimand 只是已见固定六函数有限集均值；RQ3--RQ5 在该集合上采用估计性分析，sign-flip/Holm 仅作假设敏感的辅助描述。

### Section 6 Results

严格按 RQ1--RQ5 填入真实结果。每张主表同时保留 Utility、terminal `log10_gap`、runtime 和失败/coverage；不允许只展示 scalarized Utility。不得填入占位符。

### Section 7 Discussion

讨论 representation dependence、sample-best 与 continuation 的区别、Selector regret、Behavior schedule 泄漏风险、资源约束及外推限制。区分：query 操作性增量、query-feature 预测增量与因果信息价值。

### Section 8 Reproducibility

记录配置、显式整数 seed/stream code、`SeedSequence`、完整状态与 RNG、query/sample/action budget、Stage-A 科学端点来源、Stage-B 三次计时顺序/status/effective FE/timeout/completion/instability、完成端点一致性、function splits、失败规则和结果字段。不引入 checksum、manifest、receipt、执行解锁或替代性运行流程。

### Section 9 Conclusion

只总结正式证据支持的方向和范围。最强可允许结论是“在所评估 query、portfolio、状态分布和 suites 下，固定 query acquisition 可被建模为状态条件决策”；不能改写为所有 Landscape Analysis 一般都应被跳过或成为新的优化变量。

## 7. 统计与资源边界

function 是最高聚合层。BBOB-validation 的条件 CI 固定 F5/F9/F13/F14/F19/F24、全部 dimensions 与 instances 1/2/3 对应的 static problems，只在每个固定 static problem 内配对重抽 optimizer seeds；RQ1 对每个抽中 seed/run 保留完整 state sequence。function-resampling 只作函数组成敏感性，不推断到 transformed-instance 超总体。项目内 operational tolerance 没有领域普适含义。六函数双侧 exact sign-flip 还要求 signs 可交换，raw p 最小 0.03125；RQ3 与 RQ5 各自六 contrast 的最小 Holm-adjusted p 均为 0.1875，故在 0.05 下不可能拒绝。RQ4 不把四个 suites 组成一个 Holm family，而是按 suite 与 endpoint 分别给有限集估计。ERT 在每个 bootstrap replicate 内固定 static problems、只在 problem 内配对重抽 runs，并逐 `function × dimension` 重新计算 numerator/target-hit count 与 policy log-ratio；零命中保留为有符号无穷或 undefined mass，扩展实数分位数决定有限/无界/undefined 状态，不能因任一 replicate 零命中就自动判定区间未建立。

基础 BBOB trajectory 为 0.1008B FE（5 seeds）。12 个 mandatory milestones 的平均 prefix ratio 为 0.35，future path 平均为 0.65B，旧 0.60B 单点假设低估 1.625 倍。只含 mandatory states 时，Stage-A 跨 matrices 共享但不复用基础 trajectory 为 main cheap 6.244602B、三档 8.905806B；进一步复用基础 trajectory 为 5.458362B/8.119566B；当前 main producer 为 7.817082B。计入 fold-role selected paths 的三次 timing 与 pre-run AAS 后，总量分别为 35.951622B/58.367106B、35.165382B/57.580866B，当前 main producer 为 37.524102B。event-only states、外部 suites、失败和额外 query replicates 均未计入，因此这些只称 mandatory-only 算术情景。当前 CEC2017 online evaluator 另需 1.9314B planned FE（5 seeds）；已见 BBOB-validation 全 instances 需 0.9324B，但当前 evaluator 不支持。

## 8. 允许与禁止的结论

正式证据齐备后可按结果写：

- 主 query 的联合 Utility、Behavior-only Utility 与query 操作性增量在目标状态分布中的方向、效应量和区间；
- B3 相对 milestone-only T0 的 first-trigger policy 差异；
- Proposed 相对预设 baselines 的 terminal performance、runtime、calls 和 failures；
- 三档预定义 query 的一致性或 representation dependence；
- 六组 Behavior 基函数与预测的稳定关联。

始终禁止：

- 把多动作共享状态运行称为反事实并作因果主张；
- 把 `best observed action` 称为 oracle；
- 把 Search Maturity 写成独立 latent state 或既有公认概念；
- 由 AUROC、系数、SHAP 或单一 Utility 推出 feature 必要性；
- 由 BBOB-validation 六个已见 functions 的 p 值重选模型，或把其估计性分析写成独立确认；
- 把未执行、失败删除、代理运行或旧历史数值写成正式证据。
