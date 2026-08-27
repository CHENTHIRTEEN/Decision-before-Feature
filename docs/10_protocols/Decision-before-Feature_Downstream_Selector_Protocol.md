# Decision-before-Feature Downstream Selector Protocol

> 本文件定义固定下游 Selection Reference / Selector 的输入、动作、标签、模型、评价和诊断字段。它不改变 Decision Model 的成熟度形式消融；Decision Model 使用主协议指定的 Random Forest Regressor。

## 1. 目的与边界

本项目研究的问题是：在黑盒优化运行到某个状态时，是否值得先执行一个固定 landscape descriptor query，再把得到的信息交给下游 Selector 选择后续动作。Selector 是该问题中的固定下游组件，用来把已经可得的信息映射到 portfolio action；它不是本文的主要方法贡献，也不是 Decision Model 的候选模型。

Mersmann et al. (2011) 支撑 ELA 作为用底层 landscape descriptors 表征未知优化问题并服务算法选择的传统来源。Guo et al. (2025) 支撑 contemporary pre-run AAS 中“采样、计算 ELA、监督 selector、计入特征 FE 成本、与 SBS 比较”的基本流程，并说明 LightGBM 是该类传统 AAS 的相关强基线。根据当前项目核读边界，这两篇文献不能支撑本项目必须采用 LightGBM，也不能支撑 descriptor cost 在 statewise 场景中自动值得支付。

因此，本协议只把 ELA / AS-LGBM 作为下游算法选择背景和敏感性参照；主实验中的 query 执行时机、Decision label、run-level first-trigger threshold 和模型选择均由 Decision Model 协议控制。

## 2. 两类 Selector 必须区分

### 2.1 Traditional pre-run AAS

Traditional pre-run AAS 在任何 portfolio optimizer update 之前工作：

- `FE = FE_prefix = 0`；
- 先执行固定 query，计算 query descriptors；
- query sample 不初始化、不扩充 optimizer population；
- Selector 只读取 query descriptors 与静态 `remaining_budget_ratio`；
- 选定一个 portfolio algorithm 后，从 fresh native initialization 开始运行，预算为 `FE_total - FE_query`；
- 关系记账为 `prefix_algorithm = selected_algorithm`、`selected_equals_prefix = true`、`handoff_required = false`、`handoff_type = fresh_optimizer_initialization`；
- `default_algorithm = no_query_algorithm = fold-specific SBS`。

该基线对应传统 AAS 的“先表征问题、再选算法”流程。它不是 statewise Always Query，因为 Always Query 在已有 optimizer trajectory 的第一个在线机会触发，并可能执行 population transfer。

### 2.2 Statewise Dynamic Selector

Statewise dynamic Selector 在已有完整 optimizer state 上工作。状态包含 prefix population、fitness、best-so-far、native update 位置和 RNG state。给定状态只在 Decision Model first-trigger 后使用一次；未触发 run 不调用 query Selector。

主 statewise Selector 有三种操作性路径：

- `query_joint`：读取 28 个无成熟度 B2+Motion Behavior、当前 query descriptors 和 query-adjusted `remaining_budget_ratio`；
- `query_matched_state_only`：读取 28 个无成熟度 B2+Motion Behavior 和同一 query-adjusted `remaining_budget_ratio`，但移除 descriptors；
- `behavior_only_full_budget`：不执行 query，读取 28 个无成熟度 B2+Motion Behavior 和 full-budget `remaining_budget_ratio`。

这些 Selector 共享 portfolio，但 action budget、输入信息和输出解释不同，不能互相替代。

### 2.3 MA-BBOB formal augmentation

用户指定的 `configs/phase1_mabbob_formal.yaml` 中 24 个 MA-BBOB candidate 可作为 Selector 训练增强集加入下游 Selection Reference fitting。该增强只扩大 Selector 的 action-loss regression 训练覆盖，不改变 Decision Model 活动候选、Decision X、threshold 拟合口径或 first-trigger policy 定义。

MA-BBOB augmentation 的样本单位仍必须是已有 optimizer trajectory 上的 eligible decision state，而不是 benchmark-level、function-level 或 fixed-stage AAS 标签。每个 state 必须拥有同一共享状态下四个 action outcome，并使用 `continue_current` 作为 target transform 的参照；不得因为引入 MA-BBOB 而改为“每个 benchmark 训练一个 selector”或“每个 problem instance 运行前选一个算法”的协议。

增强训练必须满足：

- `bbob_train` 仍是必需训练 split；
- MA-BBOB split 固定为 `mabbob_formal`；
- candidate IDs 必须等于 `selection_manifest_path` 中的 `selected_candidate_ids`；
- `selection_manifest` 中 `validation_component_guard = true`，且所有 selected entries 的 `is_val_component = false`；
- train-default SBS 仍优先由 `bbob_train` 的 complete-budget outcomes 估计，MA-BBOB rows 只扩充 Selector regression 的 state/action 覆盖。

因此，使用 MA-BBOB 增强后的 Selector 产物不能再把 MA-BBOB formal 当作独立确认性评价集；若需要外部确认，必须另用未参与 Selector fitting、且在首次 outcome 前é¢åæå®的 CEC2022 或工程集合。

### 2.4 MA-BBOB validation-side evaluation set（mabbob_validation）

`configs/phase1_mabbob_validation.yaml` 定义验证侧 MA-BBOB 集合：18 个定义（6 anchor + 6 pairwise + 3 triple + 3 dense），全部由 BBOB validation 函数 F5/F9/F13/F14/F19/F24 的 ManyAffine 组合生成，candidate ID 从 201 起，dense 权重支持集严格限制在六个 validation 函数内。正式采集为 10/20/40D、instance 1、seeds 1–5、reflect 边界，与主协议 endpoint 常数一致。

该集合是**evaluation-only**：

- 不得加入 Selector 的任何 fitting split（`FORMAL_SELECTOR_TRAINING_SPLITS` 只含 `bbob_train` 与 `mabbob_formal`）；
- 只能经 `--predict-action-losses` 等 held-out 评价入口消费，预测来源必须为 `train_fit`；
- guard 方向与训练侧相反：`selection_manifest` 记录 `component_scope=validation` 与 `train_component_guard=true`，选中 entries 必须全部 `is_val_component=true` 且 components ⊆ F_val；任一 train 成分进入即违反协议。

扩充后 BBOB-validation estimand 采用两层 50/50 预指定组成（见主规范 §14.1 与 README）：0.5 ×（六原函数等权子均值）+ 0.5 ×（18 个 mabbob 定义等权子均值）。该组成固定，不得事后调整；六原函数子均值与旧口径同构，乘 2 可还原。该集合仍是已见内部评价集，不承担确认性证据；两层信息同源于六个 validation 函数族，50/50 只表示权重而非独立信息量。

## 3. é¢åæå®输入模式

当前 `selection_reference.model` 中的输入模式固定为：

| `selector_input_mode` | 输入字段 | 角色 |
|---|---|---|
| `query_full` | 28 个无成熟度 B2+Motion `bf_*` + 当前 query descriptor columns + `remaining_budget_ratio` | 主 Query/full Selector |
| `state_only` | 28 个无成熟度 B2+Motion `bf_*` + `remaining_budget_ratio` | matched-acquisition descriptor-use 诊断 |
| `query_only` | 当前 query descriptor columns + `remaining_budget_ratio` | query-only 信息诊断 |
| `behavior_only_full_budget` | 28 个无成熟度 B2+Motion `bf_*` + full-budget `remaining_budget_ratio` | Behavior-only full-budget Selector |
| `pre_run_query_only` | FE=0 query descriptor columns + `remaining_budget_ratio` | Traditional pre-run AAS baseline |

主 query `descriptor_cheap_invariant` 固定为 14 个 `descriptor_*` 字段，协议为 `landscape_query_v3:descriptor_cheap_invariant_14_lhs_50d`，预处理为 `unit_cube_x__median_iqr_y_v1`。Selector behavior 固定为 `behavior/features.py::SELECTOR_BEHAVIOR_FEATURE_COLUMNS` 的 28 列无成熟度行为，不包含三个成熟度字段或三个 diagnostic-only behavior 字段。

预算字段分两层处理：`FE_prefix`、`FE_total` 和 `remaining_budget_ratio` 必须同时从 action-loss / Selection Reference 输入中加载，用于校验 shared-state FE 账本、query-adjusted action budget 和 budget mode；实际进入 Selector regression 的预算 covariate 只允许是连续 `remaining_budget_ratio = (FE_total - FE_prefix - FE_query) / FE_total`（或 full-budget 路径中的 `(FE_total - FE_prefix) / FE_total`）。`FE_prefix` 和 `FE_total` 不直接作为模型特征，以避免模型利用绝对预算、维度或 suite 规模形成捷径；它们保留为 metadata 与一致性检查字段。

直接从 `selection_reference.parquet` 构建 Selector 输入时，不能假设 `bf_*` 已自动存在；必须按 AGENTS.md 规定从 `behavior.parquet` 以 `(problem_id, function_id, family, cv_group_id, dimension, algorithm=prefix_algorithm, seed, FE)` join，并断言存在 28 个无成熟度行为字段和主 query 的 14 个 descriptor 字段。`remaining_budget_ratio` 属于 Selector 输入；Decision Model X 只使用当前成熟度消融指定的行为字段。

禁止进入任何 Selector 输入：

- function ID、algorithm ID 作为预测特征；
- optimizer-specific parameters 或内部控制参数；
- benchmark reference value、known optimum、gap、action loss、best observed action；
- runtime 或 wall-clock time。

上述字段可以作为离线标签、评价、分层和诊断 metadata 保存。

## 4. 动作集与预算

Portfolio algorithm 顺序固定为：

```text
de, pso, cmaes, shade
```

Statewise 动作集不是五个算法，而是四个互不重复动作：

```text
continue_current
其余三个 portfolio algorithms
```

若 `target_algorithm == prefix_algorithm`，动作写作 `continue_current`，transition 为 `native_optimizer_state`，handoff runtime 必须为 0。若 `target_algorithm != prefix_algorithm`，动作写作目标 algorithm 名称，transition 为 `population_transfer_initialization`。

预算模式固定为：

| `action_budget_mode` | FE 预算 | 用途 |
|---|---|---|
| `query_adjusted_budget` | `FE_total - FE_prefix - FE_query` | `query_full`、`state_only`、`query_only` |
| `behavior_only_full_budget` | `FE_total - FE_prefix` | `behavior_only_full_budget` |
| `pre_run_query_adjusted_budget` | `FE_total - FE_query` 且 `FE=0` | `pre_run_query_only` |

所有科学标签使用 FE-indexed optimization loss。Wall-clock/runtime 只用于端到端部署评价、资源分析和 Utility 的时间项，不得进入 action-loss 科学标签。

## 5. 标签与 target transform

### 5.1 Statewise 主 target

Statewise Query 和 Behavior-only Selector 的主 target 固定为：

\[
Y_{s,a} =
\log_{10}(\operatorname{clip}(L_{s,a}, g_{\min}, g_{\max}))
-
\log_{10}(\operatorname{clip}(L_{s,\mathrm{continue}}, g_{\min}, g_{\max})).
\]

其中 `continue_current` 的 target 恒为 0，序列化字段为：

```text
selector_target_transform = clipped_log10_gap_advantage_vs_continue_current
```

Selector 预测每个 portfolio algorithm 的 target，选择预测 target 最小的 algorithm。逐状态最小 observed action loss 只能称为 `best observed action`，用于诊断和 regret，不得写成现实中可部署的理想决策规则，也不进入 Decision X。

旧 transform：

```text
statewise_minmax_observed_action_loss
```

只允许作为预设 Selector target sensitivity。它不得生成主 selected action、Utility、Decision label 或政策评价。

### 5.2 Pre-run AAS target

Traditional pre-run AAS 没有 `continue_current` 参照，target 固定为绝对 clipped log loss：

```text
selector_target_transform = clipped_log10_observed_action_loss
selection_reference_protocol = pre_run_query_only_observed_algorithm_loss_regression
```

它选择的是 fresh initialization algorithm，不表示已有 population handoff。

## 6. 主 Selector 与敏感性位置

下一轮 Decision Model 训练使用的正式下游 Selector 固定为 `dimension_aware_hybrid_selector`。该 Selector 保持四动作 statewise Selection Reference 契约不变，但按维度路由到两个已é¢åæå®组件：

```text
selector_status = dimension_aware_hybrid_selector
selector_10d_20d = formal_multioutput_rf
selector_40d = pairwise_aggregation_rf_classifier
selector_fit_weight_mode = cluster_balanced_fit / pairwise_margin_weighted_fit
random_state = 1701
```

`formal_multioutput_rf` 是原正式多输出 `RandomForestRegressor`，在 10D 和 20D 保持为主路径，同时作为旧 RF 基线继续报告。`pairwise_aggregation_rf_classifier` 是六个 one-vs-one `RandomForestClassifier` 的四算法排序聚合器，在 40D 作为主路径，并作为全维度 selector sensitivity 报告。两者都是下游 Selection Reference 组件，不是 Decision Model 候选；它们不允许把 RandomForestRegressor、LightGBM、XGBoost、MLP 或其他模型加入 Decision Model 活动候选。

LightGBM 只允许放在以下位置：

- 作为 Guo et al. (2025) AS-LGBM 背景下的传统 AAS 相关模型；
- 作为预先声明的 Selector model sensitivity；
- 使用同一 action-loss matrices、同一输入模式、同一 splits、同一 first-trigger policy evaluator 和同一字段契约；
- 报告 `selector_regret_*`、acceptable action、selected relation fields、policy endpoint 差异。

LightGBM sensitivity 不得：

- 替代主 Selection Reference 的 dimension-aware selected action；
- 参与 Decision Model 候选选择、threshold 拟合或 B3/T0 主比较；
- 因 BBOB-validation、CEC2017 或 external outcome 表现更好而改写主模型；
- 生成新的主 Utility label、主 Decision label 或新的 action set。

若执行 LightGBM sensitivity，结论只能写成“在相同下游 action-loss regression 任务下，LightGBM sensitivity 的 selector regret / policy endpoint 与é¢åæå® RF 的差异”，不能写成 AS-LGBM 已证明本项目 query 值得执行。

### 6.1 Pairwise aggregation 与 dimension-aware hybrid

`pairwise_aggregation_rf_classifier` 是正式 selector 方案的一部分，定义为六个算法对的一对一分类器：

```text
de vs pso
de vs cmaes
de vs shade
pso vs cmaes
pso vs shade
cmaes vs shade
```

每个 pairwise classifier 预测左侧算法相对右侧算法是否有更低 clipped log10 action loss；训练样本权重使用 formal train split 内的 cluster-balanced 权重，并乘以该算法对 observed log10-loss advantage 的截断绝对值 multiplier。预测时每个 pair 投一票给概率较高的算法，最终选择得票最多的算法；若出现票数并列，用该算法在所有 pair 中的概率和破局。

`dimension_aware_hybrid_selector` 是下一轮 Decision Model 训练的主 Selection Reference，定义为：

```text
10D / 20D: use formal_multioutput_rf
40D:       use pairwise_aggregation_rf_classifier
```

该规则的动机是 40D validation high-regret 诊断中出现的 CMA-ES/SHADE 错序集中失误。é¢åæå®该规则时必须先使用 formal train-only OOF：训练 split 固定为 `bbob_train` 与 `mabbob_formal`，held-out 单位为 `cv_group_id`，候选比较至少包括 `formal_multioutput_rf`、`pairwise_aggregation_rf_classifier` 与 `dimension_aware_hybrid_selector`。只有在 train-only OOF 已记录并固定上述维度路由后，才允许对 `bbob_validation` 与 `mabbob_validation` 做只读评价。

该正式方案不改变基础输入契约。`dimension` 只作为预定义 strata 路由变量，不加入 RF regressor 或 pairwise classifier 的 feature columns；基础 Selector 特征为 28 列无成熟度 B2+Motion behavior、query descriptors 与 `remaining_budget_ratio`。该方案也不改变 portfolio、action-loss target transform、Decision Model 的成熟度形式消融、threshold 拟合或 run-level first-trigger 规则。

正式训练、预测与评价统一由以下入口完成：

```text
uv run selection-reference-build
```

该入口一次生成以下活动产物：

```text
results/selection_reference/{query_id}/selection_reference.parquet
results/selection_reference/{query_id}/pairwise_aggregation_sensitivity.parquet
results/selection_reference/{query_id}/formal_multioutput_rf_baseline.parquet
results/selection_reference/{query_id}/selector_evaluation_summary.parquet
results/selection_reference/{query_id}/statewise_selector.joblib
```

`selection_reference.parquet` 是 dimension-aware hybrid 主表；其余三个 parquet 分别保存全维度 pairwise 敏感性、旧多输出 RF 基线和统一评价汇总。模型文件同时包含多输出 RF 与六个 pairwise RF。旧版 `statewise_selector.joblib` 仍可由兼容读取逻辑加载，并按旧多输出 RF 行为预测；旧诊断目录与旧主表仅作历史比较，不属于新的正式结果。进入下一轮 Decision Model 训练前，必须用统一入口重新生成 Selection Reference、Utility labels、Decision dataset 与 online evaluation 输入产物，并报告与 `formal_multioutput_rf` 基线的差异；不得只替换模型文件而复用旧 RF 派生的 Utility 或 Decision labels。

## 7. Cross-fitting 与 run-level first-trigger

任何用于训练证据、模型选择、threshold、validation 或 baseline 的 Selector prediction 都必须来自相应 fold 的上游链：

- outer Decision fold 中，Selector 只在 outer-fit functions 上拟合；
- inner Decision fold 中，Selector 只在 inner-fit functions 上拟合；
- full-train Selector 只用于 BBOB-validation 和外部部署；
- 完整 BBOB-train fit 不能为自身生成 OOF Utility labels。

Statewise Selector 行只定义某个 eligible state 的 selected action 和 endpoint。政策评价必须提升到完整 run：

1. 每条 trajectory 的机会按整数 `FE` 升序排序；
2. 同一 `FE` 有多行时按 `decision_opportunity_index` 排序；
3. 给定 threshold，只取最早满足 `score > threshold` 的机会；
4. 未触发 run 的政策 Utility 和 Query path endpoint 贡献为 0 或 No-query 对应值，按具体 policy endpoint 定义；
5. 首次触发后的后续状态在该 policy 下不可达。

Always Query、matched-rate Random、Decision-before-Feature、self-thresholded Behavior-only、milestone-only T0 和所有 threshold 选择都必须使用这个 run-level first-trigger 规则。逐状态 AUROC、Average Precision、Spearman、Ridge RMSE 或逐状态 selector accuracy 只能作辅助诊断。

Traditional pre-run AAS 是 FE=0 单次 policy，不参与 statewise threshold 扫描，也不能与 Always Query 混称。

## 8. 必备输出字段

### 8.1 关系字段

Selection Reference、Utility、Decision dataset 与 online output 必须逐行保存：

```text
selected_algorithm
selected_action
selected_equals_default
selected_equals_prefix
selected_transition_mode
handoff_required
handoff_type
default_algorithm
no_query_algorithm
prefix_algorithm
```

并满足：

```text
handoff_required = not selected_equals_prefix
handoff_required = (handoff_type == population_transfer_initialization)
selected_action = continue_current iff selected_equals_prefix
```

不得生成 `label_source`，也不得用 `same_algorithm` / `changed_algorithm` 替代这些显式关系。Pre-run AAS 的 `handoff_type = fresh_optimizer_initialization` 是单独语义，不等同于 population handoff。

### 8.2 Observed loss 与 prediction 字段

每个 Selector row 至少保存：

```text
observed_loss_{algorithm}
predicted_selector_target_{algorithm}
best_observed_algorithm
best_observed_loss
selected_predicted_selector_target
action_loss
selector_prediction_source
selector_input_mode
selector_target_transform
selection_reference_protocol
performance_value_mode
performance_loss_mode
```

其中 `action_loss` 是 selected action 的 canonical known-optimum gap，`best_observed_loss` 是同一 state 四动作 observed loss 的最小值。

### 8.3 Selector regret 字段

Selector regret 只比较同一 action matrix 上 selected action 与 best observed action：

\[
\mathrm{selector\_regret\_raw}
= L_{selected} - L_{best}.
\]

必须保存：

```text
selector_regret_raw
selector_regret_norm
selected_matches_best_observed
potential_gain_raw
```

`selector_regret_raw` 必须非负，容差只允许用于浮点误差检查。`selector_regret_norm` 使用同一 state 的 observed loss range 归一化：

\[
\frac{L_{selected}-L_{best}}
{\max(L_{worst}-L_{best}, 10^{-12})}.
\]

这些字段用于误差分析和失败分析，不进入 Decision Model X。

### 8.4 Acceptable action 字段

Acceptable action 是近似并列诊断，不改变 selected action。主 selected action 仍由预测 target 最小决定。

在每个 state 内定义 clipped log10 observed loss：

\[
\ell_a = \log_{10}(\operatorname{clip}(L_a, g_{\min}, g_{\max})),
\quad
\ell_* = \min_a \ell_a.
\]

项目内预设 `log10_gap` operational tolerance 为 `0.05`。据此必须保存：

```text
acceptable_action_tolerance_log10_gap = 0.05
acceptable_action_set
selected_is_acceptable_action
acceptable_action_count
selector_regret_log10_gap
```

其中：

```text
acceptable_action_set = {a: ell_a - ell_* <= 0.05}
selected_is_acceptable_action = selected_algorithm in acceptable_action_set
acceptable_action_count = len(acceptable_action_set)
selector_regret_log10_gap = ell_selected - ell_*
```

该 tolerance 只称项目内 operational tolerance，不是确认性等价边界。若某 suite 在首次 outcome 前另行é¢åæå® endpoint-specific tolerance，可以新增独立字段，但不得覆盖上述字段含义。

## 9. 与 Utility 五路径的连接

Downstream Selector 必须服务五条逐状态配对路径：

```text
skip
query_joint
query_matched_state_only
sampling_only_continue_current
behavior_only_full_budget
```

`query_joint` 与 `query_matched_state_only` 使用同一 query realization、sample endpoint、query-adjusted action matrix 和剩余预算，只差 descriptors 是否进入 Selector。`sampling_only_continue_current` 执行同一 query acquisition 后原生继续当前算法。`behavior_only_full_budget` 不执行 query，使用完整剩余预算和单独 full-budget action matrix。

五路径分解只表示固定模型、预算和 transition rule 下的操作性分解，不作因果解释。Selector regret 和 acceptable action 均基于 continuation-only action matrix；不得把 query sample best 的直接改进归因于 query descriptors。

## 10. 产物资格与禁止改写

可进入主结果的 Selector 产物必须满足：

- action set 为 `continue_current` 加其余三个 portfolio algorithms；
- statewise 主 target 为 `clipped_log10_gap_advantage_vs_continue_current`；
- pre-run target 为 `clipped_log10_observed_action_loss`；
- Selector 主模型为é¢åæå® RF action-loss regression；
- fold-specific prediction 不含同 function in-sample 泄漏；
- relation、regret、acceptable action 字段齐全；
- runtime / wall-clock 对比若涉及 SBS、VBS、Selector 或 policy，必须来自 online replay 实测。

以下产物不能进入主模型选择或论文结果：

- 用完整 train Selector 给 train 本身生成的 in-sample Utility label；
- 用 `statewise_minmax_observed_action_loss` 生成的主 selected action；
- 把 LightGBM sensitivity 当成主 Selector 或 Decision Model 候选的结果；
- 用 component runtime 拼接的 complete-path timing；
- 把 pre-run AAS、Always Query 和 statewise first-trigger policy 混成同一 policy 的输出；
- 缺少 `selected_equals_default`、`selected_equals_prefix`、`handoff_required`、`handoff_type`、`selector_regret_*` 或 acceptable action 字段的 Selection Reference。

## 11. 与 Decision Model 的接口

Decision Model 只接收 query 前算法无关 Behavior 特征组。它不得读取：

```text
descriptor_*
remaining_budget_ratio
selected_algorithm
selected_action
predicted_selector_target_*
observed_loss_*
best_observed_*
selector_regret_*
selected_is_acceptable_action
```

上述字段只能作为 offline label 构建、Utility 分解、baseline、Selector 误差分析和论文诊断使用。Decision Model 活动候选、tie-break、threshold 和 run-level first-trigger 评价完全由 `Decision-before-Feature Decision Model设计与训练协议.md` é¢åæå®，本文件不得扩展或替代。
