# Decision-before-Feature phase1 utility label column spec

## 1. 文档目的

本文冻结 phase1 refined sampling 后续生成 utility label 数据集时的字段分层。

字段分为四类：

- metadata：只用于定位 state、split、分组报告和续跑管理；
- main label：Decision Model 可使用的主监督目标及其直接组成字段；
- cost ledger：成本账本字段，参与主 utility 或解释主 utility 的预算来源；
- diagnostic-only：只用于诊断、上界参照或敏感性分析，不进入主 Decision Model 训练 target。

本文只定义后续数据集字段口径，不修改当前 `utility_labels`、`selection_reference`、phase1 配置或正式 feature extractor。

---

## 2. 命名原则

phase1 refined sampling 后续输出可保留当前代码字段，同时在文档和汇总表中使用更清晰的语义别名。

当前已有字段与推荐语义别名：

| 推荐字段 | 当前近似字段 | 含义 |
|---|---|---|
| `loss_skip` | `p_skip` | skip-ELA 路径最终 performance，越小越好 |
| `loss_selector` | `p_ela` | fixed `selection_reference` 路径最终 performance，越小越好 |
| `gain_real` | `performance_gain_norm` | observed selector 路径相对 skip 的归一化性能收益 |
| `u_ela_real_primary` | `u_ela_lamT_1` | phase1 主 utility target |
| `decision_label` | `need_ela_lamT_1` | phase1 主二分类标签 |

若后续新增字段，优先使用本文推荐字段名；若为了兼容当前代码继续使用旧字段名，报告中必须明确别名关系。

---

## 3. Metadata 字段

Metadata 字段不进入 Decision Model 输入，不作为训练 target。

| 字段 | 类型 | 是否必需 | 口径 |
|---|---|---:|---|
| `state_id` | string | 是 | state 的可读组合键，不使用 hash、checksum 或 digest |
| `split` | string | 是 | `train`、`validation` 或后续明确命名的外部 test split |
| `problem_id` | string | 是 | 具体问题粒度，例如 `bbob_f005_i01_d10` |
| `family` | string | 是 | function family 粒度，例如 `bbob_f005` |
| `dimension` | int | 是 | 只作 metadata、split 和分层报告 |
| `prefix_algorithm` | string | 是 | 产生共享前缀 state 的优化算法，只作分层诊断 |
| `seed` | int | 是 | 显式 optimizer seed |
| `FE` | int | 是 | checkpoint 已消耗 FE |
| `FE_ratio` | float | 是 | checkpoint ratio |
| `default_algorithm` | string | 是 | skip-ELA 路径使用的默认算法 |
| `selected_algorithm` | string | 是 | fixed `selection_reference` 在 ELA 路径中选择的算法 |
| `label_source` | string | 是 | `same_algorithm` 或 `changed_algorithm` |

`state_id` 推荐由显式字段拼接生成：

```text
state_id =
    split
    + "|" + problem_id
    + "|" + prefix_algorithm
    + "|" + seed
    + "|" + FE
```

不得用文件哈希、checksum、digest 或 Python 内置 `hash()` 生成 `state_id`。

---

## 4. Cost ledger 字段

Cost ledger 字段用于解释预算和非 FE 成本。它们可以参与主 utility 的组成，但不能被重复扣除。

| 字段 | 类型 | 是否必需 | 口径 |
|---|---|---:|---|
| `FE_total` | int | 是 | 当前 problem/dimension 的总 FE 预算 |
| `FE_prefix` | int | 是 | checkpoint 前缀已消耗 FE |
| `FE_analysis` | int | 是 | ELA analysis 使用的 FE |
| `FE_skip_optimization` | int | 是 | `FE_total - FE_prefix` |
| `FE_ela_optimization` | int | 是 | `FE_total - FE_prefix - FE_analysis` |
| `ela_sampling_fe` | int | 是 | 推荐别名，等价于 `FE_analysis` |
| `ela_runtime` | float | 是 | 推荐别名，等价于 `runtime_analysis` |
| `selector_runtime` | float | 是 | 推荐别名，等价于 `runtime_selection` |
| `runtime_skip_optimization` | float | 是 | skip-ELA continuation runtime |
| `runtime_selector_optimization` | float | 是 | 推荐别名，等价于当前 `runtime_ela_optimization` |
| `ela_peak_rss` | float 或 null | 建议 | ELA 路径额外 peak RSS；未正式记录时为空 |
| `time_cost_norm` | float | 是 | `(ela_runtime + selector_runtime) / max(runtime_skip_optimization, 1e-12)` |
| `memory_cost_norm` | float | 是 | 当前主口径为 `0.0` |

`ela_sampling_fe` 必须保留作预算账本，但主 `U_ELA` 不再额外扣除它，因为它已经通过减少 `FE_ela_optimization` 体现在 `loss_selector` 中。

主标签采用 Population Transfer：

- `loss_skip` 和 `loss_selector` 从同一 checkpoint `population`、`fitness`、`best_fitness` 派生；
- `loss_selector` 不包含 best-so-far warm start；
- ELA 采样点只用于计算 ELA features，不进入 selector 路径的 continuation population。

---

## 5. Main label 字段

Main label 字段是 phase1 Decision Model 的主监督口径。

| 字段 | 类型 | 是否必需 | 口径 |
|---|---|---:|---|
| `loss_skip` | float | 是 | skip-ELA 路径最终 loss / error / regret，越小越好 |
| `loss_selector` | float | 是 | fixed `selection_reference` 路径最终 loss / error / regret，越小越好 |
| `performance_gain_raw` | float | 是 | `loss_skip - loss_selector` |
| `gain_real` | float | 是 | `(loss_skip - loss_selector) / max(abs(loss_skip), abs(loss_selector), 1e-12)` |
| `u_ela_real_primary` | float | 是 | `gain_real - time_cost_norm`，当前 `memory_cost_norm=0.0` |
| `decision_label` | bool | 是 | `u_ela_real_primary > 0` |

主训练 target 固定为：

```text
u_ela_real_primary
```

若输出仍沿用当前字段名，则等价于：

```text
target_column = u_ela_lamT_1
decision_label = need_ela_lamT_1
```

主 Decision Model 训练只使用算法无关 behavior features 作为输入，不使用本节以外的 metadata、selector、VBS 或 diagnostic-only 字段作为输入。

---

## 6. Lambda sensitivity 字段

以下字段可以随主 label 一起保留，但只用于敏感性分析：

| 字段 | 类型 | 口径 |
|---|---|---|
| `u_ela_lamT_0` | float | `gain_real` |
| `u_ela_lamT_025` | float | `gain_real - 0.25 * time_cost_norm` |
| `u_ela_lamT_05` | float | `gain_real - 0.5 * time_cost_norm` |
| `u_ela_lamT_1` | float | `gain_real - 1.0 * time_cost_norm` |
| `u_ela_lamT_2` | float | `gain_real - 2.0 * time_cost_norm` |
| `need_ela_lamT_*` | bool | 对应 `u_ela_lamT_* > 0` |

`u_ela_lamT_1` 是主列；其他 lambda 不作为主训练目标、主 threshold calibration 目标或主结论口径。

---

## 7. Diagnostic-only 字段

Diagnostic-only 字段不得进入主 Decision Model 训练 target，也不得作为可部署策略的输入。

| 字段 | 类型 | 是否建议保留 | 口径与边界 |
|---|---|---:|---|
| `loss_vbs` | float | 是 | 当前 state / problem-stage 下逐状态最佳动作或 VBS 参照的 final performance |
| `gain_vbs` | float | 是 | `(loss_skip - loss_vbs) / max(abs(loss_skip), abs(loss_vbs), 1e-12)` |
| `u_ela_vbs_diagnostic` | float | 是 | `gain_vbs - time_cost_norm`，仅作上界或诊断参照 |
| `selected_matches_vbs` | bool | 是 | `selected_algorithm == vbs_algorithm` |
| `vbs_algorithm` | string | 是 | VBS 参照算法，只用于诊断 |
| `utility_threshold` | float | 否 | 不建议写入主 label 文件；应由 calibration / evaluation 输出 |
| `bucket_proxy_p_ela` | float | 可选 | 只用于 selector proxy 诊断 |
| `bucket_proxy_u` | float | 可选 | 只用于 selector proxy 诊断 |
| `bucket_proxy_precision` | float | 可选 | 只用于 proxy 报告，不等于 observed precision |

`loss_vbs` 与 `u_ela_vbs_diagnostic` 是诊断上界，不是现实可部署 Decision Model 的训练目标。

`utility_threshold` 不属于 label 生成结果。它依赖模型、训练 split、threshold mode 和 calibration 规则，应写入 `decision/` 或 `evaluation/` 输出，而不是写死在 utility label 数据集中。

---

## 8. 输出表建议

后续 phase1 可以按用途拆分输出：

### 8.1 主 utility label 表

主表保留：

- metadata 字段；
- cost ledger 字段；
- main label 字段；
- lambda sensitivity 字段；
- behavior feature columns。

主表不保留或不推荐保留：

- `utility_threshold`；
- alternate selector proxy 字段；
- threshold calibration 结果。

### 8.2 诊断表

诊断表可保留：

- `loss_vbs`；
- `gain_vbs`；
- `u_ela_vbs_diagnostic`；
- `selected_matches_vbs`；
- `vbs_algorithm`；
- bucket proxy 字段；
- selector sensitivity 输出。

诊断表必须明确标注其字段不进入主 Decision Model 训练 target。

### 8.3 Evaluation / calibration 表

Evaluation 或 calibration 输出保留：

- `utility_threshold`；
- model score；
- decision call；
- `positive_row_capture_rate`；
- `utility_capture_rate`；
- observed precision；
- bucket proxy precision；
- runtime / Pareto 指标。

这些字段不回写到主 utility label 表。

---

## 9. 最小必需字段清单

若只生成一个 phase1 utility label 主表，最小必需字段为：

```text
state_id
split
problem_id
family
dimension
prefix_algorithm
seed
FE
FE_ratio
default_algorithm
selected_algorithm
label_source

FE_total
FE_prefix
FE_analysis
FE_skip_optimization
FE_ela_optimization
ela_sampling_fe
ela_runtime
selector_runtime
runtime_skip_optimization
runtime_selector_optimization
ela_peak_rss
time_cost_norm
memory_cost_norm

loss_skip
loss_selector
performance_gain_raw
gain_real
u_ela_real_primary
decision_label

u_ela_lamT_0
u_ela_lamT_025
u_ela_lamT_05
u_ela_lamT_1
u_ela_lamT_2
need_ela_lamT_0
need_ela_lamT_025
need_ela_lamT_05
need_ela_lamT_1
need_ela_lamT_2

behavior feature columns
```

若生成诊断表，额外保留：

```text
loss_vbs
gain_vbs
u_ela_vbs_diagnostic
vbs_algorithm
selected_matches_vbs
bucket_proxy_p_ela
bucket_proxy_u
bucket_proxy_precision
```

`utility_threshold` 只在 calibration / evaluation 表中保留。
