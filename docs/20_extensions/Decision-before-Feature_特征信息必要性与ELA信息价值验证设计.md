# Decision-before-Feature Query 操作性增量与表示依赖性验证设计

> 唯一活动扩展协议（2026-08-14，方案 A 对齐修订）。本文把“query 操作性增量”和“query features 的边际预测贡献”明确分开；主功效采用方案 A 的 `G_FE`，`query_operational_increment` 仅作辅助诊断。旧 Full/Compact ELA 二分、把两类增量混写的方案全部退出。

## 1. 三档固定 query

三档以主协议为准：

| query_id | descriptors | sample design | FE | 角色 |
|---|---:|---|---:|---|
| `descriptor_cheap_invariant` | 14 个自定义 permutation-invariant descriptors | `lhs_50d` | $0.05B$ | 第一篇论文主 query |
| `pflacco_standard_invariant` | 37 个预定义 pflacco 1.2.2 descriptors | `lhs_50d` | $0.05B$ | 配置稳健性 |
| `pflacco_broad_invariant` | 52 个预定义 pflacco 1.2.2 descriptors | `lhs_100d` | $0.10B$ | 配置稳健性 |

cheap 与 standard 共享同一 `lhs_50d` 样本和 5% query-budget action outcomes；broad 使用独立 `lhs_100d` 样本与 10% outcomes。三档分别拟合 Selector、Utility、Decision 与 threshold，`query_id` 只作数据隔离和协议核对，不进入模型输入。

Cheap 原设计中的 `descriptor_y_median` 与 `descriptor_y_iqr` 在统一 median/IQR preprocessing 后恒为 0 和 1，已从活动 whitelist 删除；该构念修正不改变 query ID、样本、FE 或 action-loss 表。每条 action 记录必须同时保留行标识、科学端点、censored runtime 和一个 canonical loss `action_loss`；Utility 变体作为诊断。

NeurELA、Deep-ELA、Progressive ELA 与动态 query-type selection 不在第一篇论文范围。三档不是 Full ELA 的覆盖性分级，也不能按 validation 结果替换主 query。

## 2. 主问题：固定 query 的联合路径是否值得执行

主问题是：

> 在é¢åæå®状态分布、portfolio、Selector、预算和 first-trigger policy 下，query 前算法无关 Behavior 能否预测执行 `descriptor_cheap_invariant` 与 full Selector 相对原生 SBS continuation 的联合净效用？

方案 A 下，主功效使用等总 FE 的 `G_FE = log((E_skip + epsilon_p) / (E_query + epsilon_p))`，runtime 不进入主标签。对 Skip 与 Query path：

$$
G_FE=(\ell_s-\ell_q)-\lambda_T(\log_{10}T_q-\log_{10}T_s).
$$

$\ell_q$ 使用 Query terminal best，包括 query sample best 和 selected continuation best。Query sample 不并入 optimizer population，但 sample FE、first hit、terminal performance 与 runtime 均计入真实 Query path。另报 continuation-only gap 和 sample-best contribution。

这一 estimand 同时包含 acquisition、Selector error、handoff 和 continuation，不能称为 descriptors 的独立边际价值。

## 3. query 操作性增量

Behavior-only full-budget path 不执行 query，使用 $B-FE_t$ 的四动作 outcomes 与 Behavior-only Selector。定义：

$$
U_b=(\ell_s-\ell_b)-\lambda_T(\log_{10}T_b-\log_{10}T_s),
$$

$$
query_operational_increment=(\ell_b-\ell_q)-\lambda_T(\log_{10}T_q-\log_{10}T_b)
=G_FE-U_b.
$$

字段为 `query_operational_increment`。方案 A 主标签为 `G_FE`。它回答：在已有 Behavior-only 可部署路径上，增加固定 query 后的操作性净差。由于 Query 与 Behavior-only 的剩余 FE、sample best 和 acquisition time 不同，`query_operational_increment` 不是纯信息效应，也不是因果 estimand。

`matched_trigger_behavior_only` 在 Proposed first-trigger 的同一 state 计算 $query_operational_increment$；它是 matched-trigger diagnostic。主 baseline `self_thresholded_behavior_only` 则用 $U_b$ 的自身 train-only OOF threshold 决定 trigger，不能共用 Proposed threshold 或政策名称。

## 4. Query-feature predictive increment

为了诊断 query descriptors 对动作选择的边际预测贡献，在同一 query-adjusted budget、同一四动作 outcomes、同一 function OOF split 上比较：

- `query_adjusted_state_only_selector`：Behavior + 连续 remaining-budget ratio；
- full Query Selector：Behavior + query descriptors + 同一 remaining-budget ratio。

定义：

$$
\Delta_{pred}
=\ell_{\text{state-only selected, continuation-only}}
-\ell_{\text{full-query selected, continuation-only}}.
$$

活动字段为 `query_feature_predictive_increment_log10_gap`。该诊断：

- 不新增 action-loss runs；
- 不使用 sample best；
- 不扣 query FE、runtime 或 memory；
- 只使用 OOF selected continuation-only endpoint；
- 不能替代 `query_operational_increment` 或 `G_FE`。

若 $\Delta_{pred}$ 接近零，只能说明当前 descriptors 对当前 query-budget action choice 的增量未建立；不能推出 landscape information 普遍无用。若 $\Delta_{pred}>0$ 而 $query_operational_increment\le0$，可解释为预测改善未覆盖 acquisition/预算代价。若方向相反，应完整报告 Selector error、sample contribution 和 runtime，不得选择性展示。

## 5. 完整嵌套与泄漏控制

每个 outer fold 只用 outer-fit functions 重算 `SBS_outer`、Query/State-only/Behavior-only Selectors、Utility 和 Decision；每个 inner fold 又只用 inner-fit functions 重算 `SBS_inner`、Selectors、Utility 与 Decision。完整 BBOB-train threshold、matched-rate Random calibration 与本扩展的 OOF diagnostics 也必须来自 fold-specific 上游 OOF。

禁止：

- 用完整 train 的 action labels/Selectors 生成后再做 Decision-only OOF；
- 用 BBOB-validation 或外部 suite 选择 query、features、模型或 threshold；
- 把 query descriptors 放入 Decision X；
- 混读不同 `query_id`、`sample_design_id` 或 `FE_query` 的 outcomes。

Trajectory reservoir 的 `query_source_mode=trajectory_reservoir_zero_extra_fe` 只作额外诊断；其 `query_protocol=trajectory_query_reservoir_v1` 与独立 LHS 主 estimand 的 sample design 不同，不能合并或替代主结果。

## 6. 统计与报告

Function 是最高聚合层。先在每个科学 run 内得到 first-trigger outcome；Random 的 30 个 streams 先在 run 内平均。BBOB-validation 主区间固定全部六个 functions、dimensions 与 instances 1/2/3 对应的 static problems，只在每个固定 static problem 内配对重抽 optimizer seeds。function-resampling 只作函数组成敏感性；有限集均值不外推到 function 或 transformed-instance 超总体。每档至少报告：

- $G_{\mathrm{FE}}$（方案 A 主标签）、$U_b$、`query_operational_increment`；
- action loss canonical loss `action_loss`、`action_loss_raw`、`action_loss_norm`、`selector_regret_raw` 与 `best_observed_loss`；
- $\Delta_{pred}$ 与两 Selector 的 selected continuation-only `log10_gap`；
- query/sample FE、sample-best contribution、完整路径 runtime；
- terminal `log10_gap`、target-hit rate、endpoint-success rate、ERT、query/selector/action failure；
- first-trigger call/trigger/handoff 与 coverage；
- function-level effects 和 95% CI。

Utility、`log10_gap`、runtime ratio、call/target-hit rate 只按各自项目内 operational tolerance 作描述；`endpoint_success` rate 若分析须另行预设边界。不能通过 scalarized Utility 的内部抵消代替任一 endpoint 判断，也不据这些边界作确认性等价声明。方案 A 下主功效以 `G_FE` 为主。

## 7. 允许的结论

若三个配置方向一致，只能写“结论在三个预定义 landscape-query 配置上稳定”。若不一致，报告 representation dependence，并检查 descriptors、样本量、sample best、失败率、Selector regret、FE 与 runtime 的组成。

不得写成：

- 对完整 ELA、全部 pflacco 或任意 learned representation 成立；
- $query_operational_increment$ 是 descriptors 的纯信息价值或因果效应；
- $\Delta_{pred}$ 证明 features 必要/不必要；
- query samples 是免费的或可视为 optimizer population；
- 根据结果改选主 query、lambda、features 或 sample design。
