# Decision-before-Feature phase1 refined sampling 指标口径冻结

## 1. 文档目的

本文统一 phase1 refined sampling 使用的 `P_skip`、`p_query`、`performance_gain_raw`、`performance_gain_norm`、`time_cost_norm`、`memory_cost_norm`、`U_query`、`need_query_lamT_*`、observed utility、`best observed action`、`U_query>0 capture` 和 `precision` 定义。旧正式数据不满足完整 optimizer-state continuation，修正后必须重新生成。

本文只冻结指标口径和报告边界，不修改：

- 研究问题与 offline supervised-learning 设计；
- phase1 配置；
- Decision Model 的算法无关输入边界；
- 已冻结的 baseline 集合。

若本文与早期 `Offline Utility Label` 协议中的泛化成本描述冲突，phase1 refined sampling 按本文执行。

---

## 2. 主性能方向与字段公式

`P_skip` 与 `p_query` 均表示越小越好的 final performance 值，例如 final loss、error 或 regret。

二者都从同一共享完整 optimizer checkpoint state 派生。第一篇论文主行满足 `prefix_algorithm == default_algorithm ==` 训练集 SBS，因此 No-query 路径原生继续当前 SBS；Query 路径仅在 `selected_algorithm != prefix_algorithm` 时执行 population transfer 初始化。其他 prefix 行只用于独立的 cross-probe 稳健性分析。

`p_query` 的主口径是：

- 扣除 `FE_query` 后；
- 使用 selection reference 选择的算法；
- 从 checkpoint `population`、`fitness` 和 `best_fitness` 继续优化；
- 不使用 Best-so-far Warm Start；
- 不复用 query 采样点。

对最小化问题：

```text
performance_gain_raw = P_skip - p_query
```

因此：

- `performance_gain_raw > 0` 表示 Query 路径得到更低的最终 performance 值；
- `performance_gain_raw = 0` 表示两条路径最终 performance 相同；
- `performance_gain_raw < 0` 表示 No-query 路径更低。

归一化性能收益统一为：

```text
performance_gain_norm =
    (P_skip - p_query) / max(abs(P_skip), abs(p_query), 1e-12)
```

该分母同时适用于正式 `utility_labels` 和后续 selector proxy 诊断。不得在 proxy 诊断中改用仅含 `abs(P_skip)` 的分母。

时间成本归一化为：

```text
time_cost_norm =
    (runtime_query + runtime_selection) / max(runtime_no_query_optimization, 1e-12)
```

当前 phase1 主口径中：

```text
memory_cost_norm = 0.0
```

只有后续正式记录同量纲额外内存开销时，`memory_cost_norm` 才进入主效用公式。

---

## 3. Query Utility 与主 target column

主 utility 定义为：

```text
U_query =
    performance_gain_norm
    - lambda_time * time_cost_norm
    - lambda_memory * memory_cost_norm
```

当前 `u_query_lamT_*` 字段表示：

```text
lambda_memory = 0
lambda_time in {0, 0.25, 0.5, 1, 2}
```

phase1 refined sampling 的 Decision Model 主 target column 固定为：

```text
u_query_lamT_1
```

对应主二分类派生标签为：

```text
need_query_lamT_1 = u_query_lamT_1 > 0
```

其他 lambda 字段只用于敏感性分析，不作为主训练目标、主阈值选择目标或主结论口径。

---

## 4. 成本账本与 FE 扣除边界

phase1 主协议采用等总 FE 预算。

固定 query 消耗的 FE 已通过减少后续优化预算体现：

```text
FE_query_optimization = FE_total - FE_prefix - FE_query
FE_no_query_optimization = FE_total - FE_prefix
```

因此，主 `U_query` 中不得再次扣除同一笔 query FE 成本。主 `U_query` 只额外扣除已记录的非 FE 成本，例如 query feature computation runtime、selection runtime，以及后续可能正式记录的同量纲内存开销。

只有另设“额外 query FE”扩展实验，并且 query FE 不从优化预算中扣除时，才允许使用单独公式：

```text
U_query =
    performance_gain_norm
    - lambda_FE * extra_FE_cost
    - lambda_time * time_cost_norm
    - lambda_memory * memory_cost_norm
```

该扩展不得混入 phase1 refined sampling 主结果。

---

## 5. 算法关系字段与报告分层

`prefix_algorithm`、`default_algorithm` 和 `selected_algorithm` 是三个不同概念，必须逐行保存以下字段：

```text
selected_equals_default = (selected_algorithm == default_algorithm)
selected_equals_prefix = (selected_algorithm == prefix_algorithm)
handoff_required = not selected_equals_prefix
skip_switches_from_prefix = (default_algorithm != prefix_algorithm)
no_query_algorithm = default_algorithm
handoff_type = query_transition_mode
```

`handoff_type` 描述 Query-selected action 的 transition；No-query 分支仍由 `no_query_transition_mode` 独立描述。`handoff_required` 必须与 `handoff_type == population_transfer_initialization` 逐行一致。活动数据、分层统计和报告分别使用三个显式布尔关系，不再生成 selected-vs-default 字符串别名。

第一篇论文主数据只保留：

```text
prefix_algorithm == default_algorithm == train-derived SBS
skip_switches_from_prefix == false
```

因此主数据中 `selected_equals_default == selected_equals_prefix`，但两个字段仍分别保存。`handoff_required=false` 的行不调整参数、不重启、不改变预算或风险策略；Query 路径与 No-query 路径从同一完整状态和 RNG state 原生推进，只少 `FE_query` 对应的优化预算。对保存 best-so-far 的实现，`performance_gain_raw` 应不大于 0，`U_query` 还需扣除非 FE 成本，因此不得把偶然大于 0 的值解释为“确认信息价值”。若观察到这类值，应先检查状态一致性、预算账本和数值记录。

其他 prefix 行写入独立 cross-probe 数据，用于 cross-probe robustness、leave-one-probe-out 与 algorithm-agnostic 泛化，不进入主训练、主 threshold 或主结果汇总。

Decision Model 输入仍不得使用上述算法关系字段、algorithm id、function id 或 query features；它们只用于数据范围限定、metadata 与分层报告。

---

## 6. Observed Utility 与替代 Selector 诊断

Observed `p_query`、observed `performance_gain_norm` 和 observed `U_query` 只来自逐共享状态 action-loss table 中现实 selector 选择的动作。正式 Selection Reference 不再使用 trajectory bucket proxy。

对替代 selector 或诊断 selector，可按相同 state key 从完整 action-loss table 读取其所选动作的 observed loss；不得把另一个动作的 loss 当作该 selector 的结果。若某 state 未运行完整动作集合，则该 state 不能进入正式 selector regret 或替代 selector 比较。

必须保存：

```text
best_observed_loss
best_observed_algorithm
selected_action_loss
potential_gain_raw
selector_regret_raw
selected_matches_best_observed
```

这些字段只作离线标签与诊断，不进入 Decision Model 输入。

---

## 7. best-observed-action、capture 与 precision

`selected_matches_best_observed` 只表示现实 selector 是否选择了逐状态已运行候选动作中的最小 loss 动作。它不代表 `U_query` 必然大于 0，因为 No-query 具有更多 continuation FE，且 Query 还需计算非 FE 成本。任何 selector 诊断都必须同时报告 `selector_regret_raw`、observed `p_query`、observed `U_query`、capture 和 precision。

`U_query>0 capture` 的主口径为捕获到的正效用总量占全部正效用总量：

```text
utility_capture_rate =
    sum(U_query for rows where policy_calls_query and U_query > 0)
    / sum(U_query for rows where U_query > 0)
```

`positive_row_capture_rate` 是行数捕获率：

```text
positive_row_capture_rate =
    count(policy_calls_query and U_query > 0)
    / count(U_query > 0)
```

二者必须分开命名，不得混用。

`precision` 定义为被策略调用 query 的行中有正效用的比例：

```text
precision =
    count(policy_calls_query and U_query > 0)
    / count(policy_calls_query)
```

Precision 只能由 observed utility label 计算；不再接受 nearest-bucket proxy precision 作为正式结果。

---

## 8. Decision Model、Threshold Calibration 与 Pareto 评估

Decision Model 训练：

- 活动候选只包括 LDA、Logistic Regression 与 Ridge；
- Ridge target 使用 `u_query_lamT_1`；
- LDA 与 Logistic Regression target 使用 `u_query_lamT_1 > 0`；
- 其他 lambda 只作为敏感性分析；
- 输入列只允许算法无关 behavior features。

Threshold calibration：

- 主 threshold 只使用完整 BBOB-train family-OOF score 与 Utility，模式名为 `oof_utility`；
- score quantile、FE stage、behavior bucket 或 search-maturity bucket 只能从 BBOB-train OOF 信息拟合并作为预定义稳健性分析；
- 不得使用 held-out family-stage 作为可部署 threshold key。
- BBOB-validation 不得参与模型或 threshold 选择。

Pareto 评估必须同时报告：

- utility；
- final performance；
- runtime；
- query call rate；
- positive-row capture；
- utility capture；
- unhelpful call cost。

其中 final performance 仍按越小越好解释；utility 按 `U_query` 越大越好解释。

---

## 9. 当前实现同步

代码已按本文口径完成字段与数据范围隔离，但正式数据尚需从 trajectory 开始重生成：

- 按当前固定配置，主 SBS-prefix Decision dataset 预期为 64,800 rows，train / validation 为 48,600 / 16,200；
- all-prefix cross-probe dataset 预期为 259,200 rows，train / validation 为 194,400 / 64,800；
- 主 target：`u_query_lamT_1`；
- preprocessing、模型和 deployable threshold 只在 BBOB train 上拟合；
- 主 Decision 数据只含训练集 SBS prefix；全 prefix 数据单独输出作稳健性分析；
- `selected_equals_default`、`selected_equals_prefix` 与 `handoff_required` 分别进入分层统计，活动输出不生成字符串别名；
- 旧 bucket selector 及其 proxy 产物只作为已替代方法的历史记录，不进入数据质量解释、主 Decision target 或正式结果。

当前模型选择、baseline 同步状态和外部评价边界见 `docs/30_results/phase1_current_results.md`。

---

## 10. 当前冻结结论

phase1 refined sampling 主实验沿用等总 FE 预算。当前目录中的旧 labels 不含新关系字段且依赖已撤回的 trajectory/behavior 口径，不能作为正式证据；必须重生成到 `results/utility_labels/phase1_refined_sampling/` 后再物化主数据和 cross-probe 数据。模型比较和阈值分析不得回写标签。

正式训练与主报告使用：

```text
target_column = u_query_lamT_1
need_query_lamT_1 = u_query_lamT_1 > 0
performance_gain_norm denominator = max(abs(P_skip), abs(p_query), 1e-12)
```

替代 selector 诊断必须从同一逐状态 action-loss table 读取其实际所选动作的 observed loss，并且不进入正式 Decision Model 训练数据。
