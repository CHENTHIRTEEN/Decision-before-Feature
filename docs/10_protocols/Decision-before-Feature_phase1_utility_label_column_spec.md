# Decision-before-Feature phase1 query-specific utility label column spec

> 实现同步（2026-08-11）：旧 utility labels 未保留完整 optimizer state，并缺少三档 query 协议字段，已撤回正式证据资格。活动接口不提供旧 ELA 字段兼容层。

## 1. 文档目的

本文冻结 query-specific utility label 的活动字段。每档 query 独立生成标签并写入：

```text
results/utility_labels/{query_id}/
```

标签表只描述一个 `query_id`。缺少 `query_id`、`query_protocol`、`query_feature_columns` 或 `sample_design_id` 的旧表必须明确拒绝。

---

## 2. Query 协议字段

以下字段必需，且必须与 `LandscapeQuerySpec` 完全一致：

| 字段 | 类型 | 含义 |
|---|---|---|
| `query_id` | string | `descriptor_cheap`、`pflacco_standard` 或 `pflacco_broad` |
| `query_protocol` | string | 版本化固定 query 协议 |
| `query_feature_columns` | string | 实际固定特征列的 JSON 列表 |
| `sample_design_id` | string | `lhs_50d` 或 `lhs_100d` |
| `FE_query` | int | 该 query 的函数评价数 |

`descriptor_cheap` 与 `pflacco_standard` 必须使用相同的 `lhs_50d`、5% FE action-loss 表；`pflacco_broad` 必须使用独立 `lhs_100d`、10% FE action-loss 表。不同 sample design 或 FE 预算不得混用。

---

## 3. State 与算法关系字段

以下 metadata 不进入 Decision Model 输入：

```text
split
problem_id
family
dimension
prefix_algorithm
seed
FE
FE_ratio
default_algorithm
no_query_algorithm
selection_reference_default_algorithm
selection_reference_protocol
selector_prediction_source
selected_algorithm
selected_action
selected_equals_default
selected_equals_prefix
skip_switches_from_prefix
no_query_transition_mode
query_transition_mode
handoff_type
```

主表要求 `prefix_algorithm == default_algorithm == train-derived SBS`。No-query 原生继续该完整状态；Query 选择 prefix 时同样原生继续，选择其他算法时使用一次 population-transfer initialization。全 prefix 数据只能进入独立 cross-probe 稳健性分析。

兼容字段关系固定为：

```text
no_query_algorithm = default_algorithm
handoff_type = query_transition_mode
```

`handoff_type` 只描述 Query-selected action 从 prefix state 进入 continuation 时使用 `native_optimizer_state` 还是 `population_transfer_initialization`；No-query 分支继续由 `no_query_transition_mode` 单独描述。

---

## 4. FE 与运行时间账本

必需字段：

```text
FE_total
FE_prefix
FE_query
FE_no_query_optimization
FE_query_optimization
runtime_query
runtime_selection
runtime_no_query_optimization
runtime_query_optimization
time_cost_norm
memory_cost_norm
```

逐行关系：

```text
FE_no_query_optimization = FE_total - FE_prefix
FE_query_optimization = FE_total - FE_prefix - FE_query
time_cost_norm =
    (runtime_query + runtime_selection)
    / max(runtime_no_query_optimization, 1e-12)
memory_cost_norm = 0.0
```

`runtime_query` 已等于 query 样本评价时间与 feature computation 时间之和。`FE_query` 已通过减少 Query continuation budget 进入 `p_query`，主 Utility 不得重复扣除。

---

## 5. 性能与 Selector 分解字段

必需字段：

```text
p_skip
p_query
selected_action_loss
best_observed_algorithm
best_observed_loss
selected_matches_best_observed
potential_gain_raw
selector_regret_raw
performance_norm_scale
potential_gain_norm
selector_regret_decomposition_norm
performance_gain_raw
performance_gain_norm
```

逐行必须满足：

```text
p_query = selected_action_loss
potential_gain_raw = p_skip - best_observed_loss
selector_regret_raw = p_query - best_observed_loss
performance_gain_raw = p_skip - p_query
performance_gain_raw = potential_gain_raw - selector_regret_raw
performance_norm_scale = max(abs(p_skip), abs(p_query), 1e-12)
performance_gain_norm = performance_gain_raw / performance_norm_scale
```

`best observed action` 只用于离线诊断，不称为 oracle，也不进入 Decision Model 输入。

---

## 6. Utility 与布尔标签

活动 utility 字段只有：

```text
u_query_lamT_0
u_query_lamT_025
u_query_lamT_05
u_query_lamT_1
u_query_lamT_2
```

对应布尔字段只有：

```text
need_query_lamT_0
need_query_lamT_025
need_query_lamT_05
need_query_lamT_1
need_query_lamT_2
```

对 `lambda_time in {0, 0.25, 0.5, 1, 2}`：

```text
u_query_lamT_* = performance_gain_norm - lambda_time * time_cost_norm
need_query_lamT_* = (u_query_lamT_* > 0)
```

主 Decision target 固定为 `u_query_lamT_1`。其他 lambda 只用于敏感性分析。

---

## 7. Decision 输入边界

Utility label、query feature、function、dimension、algorithm relation、best-observed-action 与成本字段都不得进入 Decision X。Decision X 仍只包含冻结的 permutation-invariant、算法无关 behavior fields。

Selector 可以使用 behavior、当前 query 的固定 feature columns 与连续 `remaining_budget_ratio`；这不改变 Decision Model 的输入边界。三个 query 独立训练 Selector、Utility target 与 Decision Model，不把 `query_id` 作为模型输入。

---

## 8. 读取失败条件

以下任一情况必须阻止活动读取或模型拟合：

- 缺少 query 协议字段；
- query feature columns 与版本化白名单不一致；
- `FE_query` 与 sample design 不一致；
- 5% 与 10% action losses 混用；
- `p_query != selected_action_loss`；
- 性能分解、时间成本或布尔标签无法逐行重算；
- BBOB train/validation 存在 group-level extraction failure；
- BBOB-train 某个 query feature 整列缺失。

旧 `results/ela/`、旧 ELA 标签名和缺少新协议字段的模型均为撤回结果，不提供别名或兼容读取。
