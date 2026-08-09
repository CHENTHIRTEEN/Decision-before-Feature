# Decision-before-Feature phase1 refined sampling 指标口径冻结

## 1. 文档目的

本文用于在正式 phase1 refined sampling 启动前统一 `P_skip`、`P_ELA`、`performance_gain_raw`、`performance_gain_norm`、`time_cost_norm`、`memory_cost_norm`、`U_ELA`、`need_ela`、observed utility、proxy utility、`selected=VBS`、`U_ELA>0 capture` 和 `precision` 的定义。

本文只冻结指标口径和报告边界，不修改：

- 原始 `utility_labels`；
- 原始 `selection_reference`；
- phase1 配置；
- 正式 behavior feature extractor；
- 已有 min_support 结果文件。

若本文与早期 `Offline Utility Label` 协议中的泛化成本描述冲突，phase1 refined sampling 按本文执行。

---

## 2. 主性能方向与字段公式

`P_skip` 与 `P_ELA` 均表示越小越好的 final performance 值，例如 final loss、error 或 regret。

二者都从同一共享 checkpoint population state 续跑得到。

`P_ELA` 的主口径是：

- 扣除 `FE_analysis` 后；
- 使用 selection reference 选择的算法；
- 从 checkpoint `population`、`fitness` 和 `best_fitness` 继续优化；
- 不使用 Best-so-far Warm Start；
- 不复用 ELA 采样点。

对最小化问题：

```text
performance_gain_raw = P_skip - P_ELA
```

因此：

- `performance_gain_raw > 0` 表示 ELA 路径得到更低的最终 performance 值；
- `performance_gain_raw = 0` 表示两条路径最终 performance 相同；
- `performance_gain_raw < 0` 表示 skip-ELA 路径更低。

归一化性能收益统一为：

```text
performance_gain_norm =
    (P_skip - P_ELA) / max(abs(P_skip), abs(P_ELA), 1e-12)
```

该分母同时适用于正式 `utility_labels` 和后续 selector proxy 诊断。不得在 proxy 诊断中改用仅含 `abs(P_skip)` 的分母。

时间成本归一化为：

```text
time_cost_norm =
    (runtime_analysis + runtime_selection) / max(runtime_skip_optimization, 1e-12)
```

当前 phase1 主口径中：

```text
memory_cost_norm = 0.0
```

只有后续正式记录同量纲额外内存开销时，`memory_cost_norm` 才进入主效用公式。

---

## 3. ELA Utility 与主 target column

主 utility 定义为：

```text
U_ELA =
    performance_gain_norm
    - lambda_time * time_cost_norm
    - lambda_memory * memory_cost_norm
```

当前 `u_ela_lamT_*` 字段表示：

```text
lambda_memory = 0
lambda_time in {0, 0.25, 0.5, 1, 2}
```

phase1 refined sampling 的 Decision Model 主 target column 固定为：

```text
u_ela_lamT_1
```

对应主二分类派生标签为：

```text
need_ela_lamT_1 = u_ela_lamT_1 > 0
```

其他 lambda 字段只用于敏感性分析，不作为主训练目标、主阈值选择目标或主结论口径。

---

## 4. 成本账本与 FE 扣除边界

phase1 主协议采用等总 FE 预算。

ELA 分析消耗的 FE 已通过减少后续优化预算体现：

```text
FE_ela_optimization = FE_total - FE_prefix - FE_analysis
FE_skip_optimization = FE_total - FE_prefix
```

因此，主 `U_ELA` 中不得再次扣除同一笔 ELA FE 成本。主 `U_ELA` 只额外扣除已记录的非 FE 成本，例如 ELA feature computation runtime、selection runtime，以及后续可能正式记录的同量纲内存开销。

只有另设“额外 ELA FE”扩展实验，并且 ELA FE 不从优化预算中扣除时，才允许使用单独公式：

```text
U_ELA =
    performance_gain_norm
    - lambda_FE * extra_FE_cost
    - lambda_time * time_cost_norm
    - lambda_memory * memory_cost_norm
```

该扩展不得混入 phase1 refined sampling 主结果。

---

## 5. same_algorithm 与 changed_algorithm

标签行按下游 selector 是否改变算法分为两类：

```text
same_algorithm:
    selected_algorithm == default_algorithm

changed_algorithm:
    selected_algorithm != default_algorithm
```

`same_algorithm` 行中的 `U_ELA` 只能解释为共享前缀配对续跑随机差异参照及成本影响。即使 `same_algorithm` 行出现 `U_ELA > 0`，也不得写成“ELA selector 切换算法带来收益”。

`changed_algorithm` 行更接近“执行 ELA 后 selector 选择了不同优化算法”的效用来源，是解释 ELA 改变算法选择后带来效用的主分层。

报告时必须至少区分：

- all rows；
- `same_algorithm` rows；
- `changed_algorithm` rows。

Decision Model 输入仍不得使用 `selected_algorithm`、`default_algorithm`、`prefix_algorithm`、algorithm id、function id 或 ELA features；这些字段只用于 metadata、split 和分层报告。

---

## 6. Observed Utility 与 Proxy Utility

Observed `P_ELA`、observed `performance_gain_norm` 和 observed `U_ELA` 只来自已经生成的 utility label 行。

对替代 selector 或诊断 selector：

- 若诊断 selector 选择的算法与原 utility label 行中的 `selected_algorithm` 相同，则该行可报告 observed `P_ELA/U_ELA`；
- 若诊断 selector 选择了不同算法，则不能把原 utility label 的 `P_ELA/U_ELA` 当作该诊断 selector 的 observed value；
- 若使用已有 trajectory bucket 估算替代算法表现，必须命名为 bucket proxy `P_ELA/U_ELA`。

bucket proxy `P_ELA/U_ELA` 只用于 selector 诊断和敏感性说明，不是新生成的 utility label，不进入 Decision Model 训练 target，也不能替代 phase1 主效用标签。

bucket proxy 的归一化公式必须与主标签一致：

```text
bucket_proxy_performance_gain_norm =
    (bucket_proxy_p_skip - bucket_proxy_p_ela)
    / max(abs(bucket_proxy_p_skip), abs(bucket_proxy_p_ela), 1e-12)

bucket_proxy_u =
    bucket_proxy_performance_gain_norm
    - lambda_time * time_cost_norm
```

若 proxy 使用 neighbor bucket、interpolated bucket 或 stage-wise bucket，它必须在输出表和报告文字中标明为 proxy，不得与 observed utility 混报。

---

## 7. selected=VBS、capture 与 precision

`selected=VBS` 只表示 `selection_reference` problem-stage 粒度下：

```text
selected_algorithm == vbs_algorithm
```

它是下游 ELA-based selector 的一致率指标，不代表 `P_ELA` 一定更低，也不代表 `U_ELA` 一定改善。任何 selector 诊断都必须同时报告 `selected=VBS`、`P_ELA` 或 proxy `P_ELA`、`U_ELA` 或 proxy `U_ELA`、capture 和 precision。

`U_ELA>0 capture` 的主口径为捕获到的正效用总量占全部正效用总量：

```text
utility_capture_rate =
    sum(U_ELA for rows where policy_calls_ela and U_ELA > 0)
    / sum(U_ELA for rows where U_ELA > 0)
```

`positive_row_capture_rate` 是行数捕获率：

```text
positive_row_capture_rate =
    count(policy_calls_ela and U_ELA > 0)
    / count(U_ELA > 0)
```

二者必须分开命名，不得混用。

`precision` 定义为被策略调用 ELA 的行中有正效用的比例：

```text
precision =
    count(policy_calls_ela and U_ELA > 0)
    / count(policy_calls_ela)
```

若使用 bucket proxy 判断调用是否有正效用，必须写成 `bucket_proxy_precision` 或在报告中明确说明 `precision` 来自 `bucket_proxy_u > 0`，不得作为 observed precision 报告。

---

## 8. Decision Model、Threshold Calibration 与 Pareto 评估

Decision Model 训练：

- 主回归 target 使用 `u_ela_lamT_1`；
- 主二分类派生指标使用 `u_ela_lamT_1 > 0`；
- 其他 lambda 只作为敏感性分析；
- 输入列只允许算法无关 behavior features。

Threshold calibration：

- 可使用 train-derived threshold；
- 可使用 score quantile；
- 可使用 FE stage；
- 可使用 behavior bucket 或 search-maturity bucket；
- 不得使用 held-out family-stage 作为可部署 threshold key。

Pareto 评估必须同时报告：

- utility；
- final performance；
- runtime；
- ELA call rate；
- positive-row capture；
- utility capture；
- unhelpful call cost。

其中 final performance 仍按越小越好解释；utility 按 `U_ELA` 越大越好解释。

---

## 9. 后续诊断脚本同步项

本节只记录后续需要同步的脚本口径，当前文档任务不执行脚本修改。

- `decision/min_support_selection_reference_h3_label_source_diagnostic.py`
  - 将 bucket proxy denominator 同步为 `max(abs(bucket_proxy_p_skip), abs(bucket_proxy_p_ela), 1e-12)`。
- `decision/min_support_selection_reference_h5_model_capacity.py`
  - 将 bucket proxy denominator 同步为 `max(abs(bucket_proxy_p_skip), abs(bucket_proxy_p_ela), 1e-12)`。
  - 将 proxy capture / precision 明确命名为 bucket-proxy 指标。
- `decision/min_support_bucket_smoothing_diagnostics.py`
  - 保持 neighbor / interpolated bucket 只作为 proxy 报告。
  - 不与 observed utility 混报。
- `decision/min_support_performance_bucket_sensitivity.py`
  - 保持 neighbor bucket sensitivity 只作为 proxy 报告。
  - 不生成或暗示 alternate observed utility labels。
- `decision/min_support_evaluate.py`
  - 核心计算无需修改。
  - 后续报告文字引用本文中 `u_ela_lamT_1`、capture 和 precision 定义。
- `decision/min_support_model_sensitivity.py`
  - 核心计算无需修改。
  - 后续报告文字引用本文中 observed utility 与主 target 定义。
- `decision/min_support_stage_threshold_diagnostics.py`
  - 核心计算无需修改。
  - 后续报告需明确 train-derived threshold 与不可部署 held-out family-stage threshold 的边界。
- `decision/min_support_gated_pareto_diagnostic.py`
  - 核心计算无需修改。
  - 后续 Pareto 报告需同时列出 utility、final performance、runtime、call rate、capture 和 unhelpful call cost。
- `decision/min_support_selection_reference_generalization_data_quality.py`
  - 后续输出说明需同步 `same_algorithm` / `changed_algorithm` 解释边界。
- `decision/min_support_label_source_check.py`
  - 后续输出说明需同步 `same_algorithm` 作为共享前缀配对续跑随机差异参照的边界。
- `decision/min_support_changed_algorithm_diagnostics.py`
  - 后续输出说明需同步主 target `u_ela_lamT_1` 与 `changed_algorithm` 主分层口径。

---

## 10. 当前冻结结论

phase1 refined sampling 主实验沿用等总 FE 预算，不回写、不重算当前 min_support label 文件。

正式训练与主报告使用：

```text
target_column = u_ela_lamT_1
need_ela = u_ela_lamT_1 > 0
performance_gain_norm denominator = max(abs(P_skip), abs(P_ELA), 1e-12)
```

selector proxy 诊断保留在诊断层，不进入正式 Decision Model 训练数据。
